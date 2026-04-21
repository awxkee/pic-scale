/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use crate::avx2::utils::shuffle;
use std::arch::x86_64::*;

#[inline(always)]
fn _mm256_scale_by_alpha(px: __m256i, low_low_a: __m256, low_high_a: __m256) -> __m256i {
    unsafe {
        let zeros = _mm256_setzero_si256();
        let ls = _mm256_unpacklo_epi16(px, zeros);
        let hs = _mm256_unpackhi_epi16(px, zeros);

        let low_px = _mm256_cvtepi32_ps(ls);
        let high_px = _mm256_cvtepi32_ps(hs);

        let lvs = _mm256_mul_ps(low_px, low_low_a);
        let hvs = _mm256_mul_ps(high_px, low_high_a);

        let new_ll = _mm256_cvtps_epi32(lvs);
        let new_lh = _mm256_cvtps_epi32(hvs);

        _mm256_packus_epi32(new_ll, new_lh)
    }
}

/// Exact division by 1023 with rounding to nearest
#[inline(always)]
pub(crate) fn _mm256_div_by_1023_epi32(v: __m256i) -> __m256i {
    unsafe {
        const DIVIDING_BY: i32 = 10;
        let addition = _mm256_set1_epi32(1 << (DIVIDING_BY - 1));
        let v = _mm256_add_epi32(v, addition);
        _mm256_srli_epi32::<DIVIDING_BY>(_mm256_add_epi32(v, _mm256_srli_epi32::<DIVIDING_BY>(v)))
    }
}

/// Exact division by 4095 with rounding to nearest
#[inline(always)]
pub(crate) fn _mm256_div_by_4095_epi32(v: __m256i) -> __m256i {
    unsafe {
        const DIVIDING_BY: i32 = 12;
        let addition = _mm256_set1_epi32(1 << (DIVIDING_BY - 1));
        let v = _mm256_add_epi32(v, addition);
        _mm256_srli_epi32::<DIVIDING_BY>(_mm256_add_epi32(v, _mm256_srli_epi32::<DIVIDING_BY>(v)))
    }
}

/// Exact division by 65535 with rounding to nearest
#[inline(always)]
pub(crate) fn _mm256_div_by_65535_epi32(v: __m256i) -> __m256i {
    unsafe {
        const DIVIDING_BY: i32 = 16;
        let addition = _mm256_set1_epi32(1 << (DIVIDING_BY - 1));
        let v = _mm256_add_epi32(v, addition);
        _mm256_srli_epi32::<DIVIDING_BY>(_mm256_add_epi32(v, _mm256_srli_epi32::<DIVIDING_BY>(v)))
    }
}

#[inline(always)]
fn _mm256_div_by_epi32<const BIT_DEPTH: usize>(v: __m256i) -> __m256i {
    if BIT_DEPTH == 10 {
        _mm256_div_by_1023_epi32(v)
    } else if BIT_DEPTH == 12 {
        _mm256_div_by_4095_epi32(v)
    } else {
        _mm256_div_by_65535_epi32(v)
    }
}

pub(crate) fn avx_premultiply_alpha_rgba_u16(dst: &mut [u16], src: &[u16], bit_depth: usize) {
    unsafe {
        avx_premultiply_alpha_rgba_u16_row(dst, src, bit_depth);
    }
}

trait Avx2PremultiplyExecutor {
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize);
}

#[derive(Default)]
struct Avx2PremultiplyExecutorDefault<const BIT_DEPTH: usize> {}

impl<const BIT_DEPTH: usize> Avx2PremultiplyExecutorDefault<BIT_DEPTH> {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn premultiply_chunk(&self, dst: &mut [u16; 16], src: &[u16; 16]) {
        let shuffle_alpha_mask = _mm256_setr_epi8(
            6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15, 6, 7, 6, 7, 6, 7, 6, 7, 14, 15,
            14, 15, 14, 15, 14, 15,
        );
        let rgba = unsafe { _mm256_loadu_si256(src.as_ptr().cast()) };
        let copy_alpha_mask = _mm256_set1_epi64x(i64::from_ne_bytes([0, 0, 0, 0, 0, 0, 255, 255]));

        let alpha = _mm256_shuffle_epi8(rgba, shuffle_alpha_mask);

        let p0 = _mm256_unpacklo_epi16(rgba, _mm256_setzero_si256());
        let p1 = _mm256_unpackhi_epi16(rgba, _mm256_setzero_si256());

        let a0 = _mm256_unpacklo_epi16(alpha, _mm256_setzero_si256());
        let a1 = _mm256_unpackhi_epi16(alpha, _mm256_setzero_si256());

        let mut s0 = _mm256_madd_epi16(p0, a0);
        let mut s1 = _mm256_madd_epi16(p1, a1);

        s0 = _mm256_div_by_epi32::<BIT_DEPTH>(s0);
        s1 = _mm256_div_by_epi32::<BIT_DEPTH>(s1);

        let u0 = _mm256_packus_epi32(s0, s1);
        let q0 = _mm256_blendv_epi8(u0, rgba, copy_alpha_mask);

        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), q0);
        }
    }
}
impl<const BIT_DEPTH: usize> Avx2PremultiplyExecutor for Avx2PremultiplyExecutorDefault<BIT_DEPTH> {
    #[target_feature(enable = "avx2")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], _: usize) {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(src_rem.as_chunks::<16>().0.iter())
        {
            self.premultiply_chunk(dst, src);
        }

        rem = rem.as_chunks_mut::<16>().1;
        src_rem = src_rem.as_chunks::<16>().1;

        if !rem.is_empty() {
            let mut buffer: [u16; 16] = [0u16; 16];
            let mut dst_buffer: [u16; 16] = [0u16; 16];
            buffer[..src_rem.len()].copy_from_slice(src_rem);

            self.premultiply_chunk(&mut dst_buffer, &buffer);

            rem.copy_from_slice(&dst_buffer[..rem.len()]);
        }
    }
}

#[derive(Default)]
struct Avx2PremultiplyExecutorAnyBit {}

impl Avx2PremultiplyExecutorAnyBit {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn premultiply_chunk(&self, dst: &mut [u16; 16], src: &[u16; 16], scale: __m256) {
        let shuffle_alpha_mask = _mm256_setr_epi8(
            6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15, 6, 7, 6, 7, 6, 7, 6, 7, 14, 15,
            14, 15, 14, 15, 14, 15,
        );
        let rgba = unsafe { _mm256_loadu_si256(src.as_ptr().cast()) };
        let copy_alpha_mask = _mm256_set1_epi64x(i64::from_ne_bytes([0, 0, 0, 0, 0, 0, 255, 255]));

        let alpha = _mm256_shuffle_epi8(rgba, shuffle_alpha_mask);

        let p0 = _mm256_unpacklo_epi16(rgba, _mm256_setzero_si256());
        let p1 = _mm256_unpackhi_epi16(rgba, _mm256_setzero_si256());

        let a0 = _mm256_unpacklo_epi16(alpha, _mm256_setzero_si256());
        let a1 = _mm256_unpackhi_epi16(alpha, _mm256_setzero_si256());

        let mut s0 = _mm256_mul_ps(_mm256_cvtepi32_ps(p0), _mm256_cvtepi32_ps(a0));
        let mut s1 = _mm256_mul_ps(_mm256_cvtepi32_ps(p1), _mm256_cvtepi32_ps(a1));

        s0 = _mm256_mul_ps(s0, scale);
        s1 = _mm256_mul_ps(s1, scale);

        let e0 = _mm256_cvtps_epi32(s0);
        let e1 = _mm256_cvtps_epi32(s1);

        let u0 = _mm256_packus_epi32(e0, e1);
        let q0 = _mm256_blendv_epi8(u0, rgba, copy_alpha_mask);

        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), q0);
        }
    }
}

impl Avx2PremultiplyExecutor for Avx2PremultiplyExecutorAnyBit {
    #[target_feature(enable = "avx2")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        let max_colors = (1 << bit_depth) - 1;

        let mut rem = dst;
        let mut src_rem = src;

        let v_scale_colors = _mm256_set1_ps((1. / max_colors as f64) as f32);
        for (dst, src) in rem
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(src_rem.as_chunks::<16>().0.iter())
        {
            self.premultiply_chunk(dst, src, v_scale_colors);
        }

        rem = rem.as_chunks_mut::<16>().1;
        src_rem = src_rem.as_chunks::<16>().1;

        if !rem.is_empty() {
            let mut buffer: [u16; 16] = [0u16; 16];
            let mut dst_buffer: [u16; 16] = [0u16; 16];

            buffer[..src_rem.len()].copy_from_slice(src_rem);

            self.premultiply_chunk(&mut dst_buffer, &buffer, v_scale_colors);

            rem.copy_from_slice(&dst_buffer[..rem.len()]);
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
fn avx_premultiply_alpha_rgba_u16_row(dst: &mut [u16], src: &[u16], bit_depth: usize) {
    if bit_depth == 10 {
        avx_pa_dispatch(
            dst,
            src,
            bit_depth,
            Avx2PremultiplyExecutorDefault::<10>::default(),
        );
    } else if bit_depth == 12 {
        avx_pa_dispatch(
            dst,
            src,
            bit_depth,
            Avx2PremultiplyExecutorDefault::<12>::default(),
        );
    } else {
        avx_pa_dispatch(
            dst,
            src,
            bit_depth,
            Avx2PremultiplyExecutorAnyBit::default(),
        );
    };
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
#[inline]
fn avx_pa_dispatch(
    dst: &mut [u16],
    src: &[u16],
    bit_depth: usize,
    dispatch: impl Avx2PremultiplyExecutor,
) {
    unsafe {
        dispatch.premultiply(dst, src, bit_depth);
    }
}

pub(crate) fn avx_unpremultiply_alpha_rgba_u16(in_place: &mut [u16], bit_depth: usize) {
    unsafe {
        avx_unpremultiply_alpha_rgba_u16_row_avx(in_place, bit_depth);
    }
}

#[target_feature(enable = "avx2")]
fn avx_unpremultiply_alpha_rgba_u16_row_avx(in_place: &mut [u16], bit_depth: usize) {
    let max_colors = (1u32 << bit_depth) - 1;

    let v_scale_colors = _mm256_set1_ps(max_colors as f32);
    let v_max_test = _mm256_set1_epi16(max_colors as i16);

    let shuffle_alphas_mask = _mm256_setr_epi8(
        6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15, 6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14,
        15, 14, 15, 14, 15,
    );
    let prepare_alphas_mask = _mm256_setr_epi8(
        6, 7, 14, 15, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15, 6, 7, 14, 15, 6, 7, 6, 7, 14, 15,
        14, 15, 14, 15, 14, 15,
    );
    let copy_alpha_mask = _mm256_set1_epi64x(i64::from_ne_bytes([0, 0, 0, 0, 0, 0, 255, 255]));

    let mut rem = in_place;

    for dst in rem.as_chunks_mut::<32>().0.iter_mut() {
        let rgba0 = unsafe { _mm256_loadu_si256(dst.as_ptr().cast()) };
        let rgba1 = unsafe { _mm256_loadu_si256(dst[16..].as_ptr().cast()) };

        let is_zero_alpha_mask0 = _mm256_cmpeq_epi16(
            _mm256_shuffle_epi8(rgba0, shuffle_alphas_mask),
            _mm256_setzero_si256(),
        );

        let is_zero_alpha_mask1 = _mm256_cmpeq_epi16(
            _mm256_shuffle_epi8(rgba1, shuffle_alphas_mask),
            _mm256_setzero_si256(),
        );

        let alphas32_0 = _mm256_shuffle_epi8(rgba0, prepare_alphas_mask);
        let alphas32_1 = _mm256_shuffle_epi8(rgba1, prepare_alphas_mask);

        let ua = _mm256_permute4x64_epi64::<{ shuffle(3, 1, 2, 0) }>(_mm256_unpacklo_epi32(
            alphas32_0, alphas32_1,
        ));
        let ua32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(ua));

        let prepared_alpha = _mm256_cvtepi32_ps(ua32);
        let alphas_f32 = _mm256_div_ps(v_scale_colors, prepared_alpha);

        let mut lo0 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rgba0, _mm256_setzero_si256()));
        let mut hi0 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rgba0, _mm256_setzero_si256()));

        let mut lo1 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rgba1, _mm256_setzero_si256()));
        let mut hi1 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rgba1, _mm256_setzero_si256()));

        lo0 = _mm256_mul_ps(
            lo0,
            _mm256_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(alphas_f32, alphas_f32),
        );
        hi0 = _mm256_mul_ps(
            hi0,
            _mm256_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(alphas_f32, alphas_f32),
        );

        lo1 = _mm256_mul_ps(
            lo1,
            _mm256_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(alphas_f32, alphas_f32),
        );
        hi1 = _mm256_mul_ps(
            hi1,
            _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(alphas_f32, alphas_f32),
        );

        let lo_u0 = _mm256_cvtps_epi32(lo0);
        let hi_u0 = _mm256_cvtps_epi32(hi0);

        let lo_u1 = _mm256_cvtps_epi32(lo1);
        let hi_u1 = _mm256_cvtps_epi32(hi1);

        let mut packed0 = _mm256_packus_epi32(lo_u0, hi_u0);
        let mut packed1 = _mm256_packus_epi32(lo_u1, hi_u1);

        packed0 = _mm256_min_epu16(packed0, v_max_test);
        packed0 = _mm256_blendv_epi8(packed0, _mm256_setzero_si256(), is_zero_alpha_mask0);
        packed0 = _mm256_blendv_epi8(packed0, rgba0, copy_alpha_mask);

        packed1 = _mm256_min_epu16(packed1, v_max_test);
        packed1 = _mm256_blendv_epi8(packed1, _mm256_setzero_si256(), is_zero_alpha_mask1);
        packed1 = _mm256_blendv_epi8(packed1, rgba1, copy_alpha_mask);

        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), packed0);
            _mm256_storeu_si256(dst[16..].as_mut_ptr().cast(), packed1);
        }
    }

    rem = rem.as_chunks_mut::<32>().1;

    for dst in rem.as_chunks_mut::<16>().0.iter_mut() {
        let rgba = unsafe { _mm256_loadu_si256(dst.as_ptr().cast()) };

        let alphas = _mm256_shuffle_epi8(rgba, shuffle_alphas_mask);
        let alphas32 = _mm256_unpacklo_epi16(
            _mm256_shuffle_epi8(rgba, prepare_alphas_mask),
            _mm256_setzero_si256(),
        );
        let is_zero_alpha_mask = _mm256_cmpeq_epi16(alphas, _mm256_setzero_si256());

        let alphas_f32 = _mm256_div_ps(v_scale_colors, _mm256_cvtepi32_ps(alphas32));

        let mut lo = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rgba, _mm256_setzero_si256()));
        let mut hi = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rgba, _mm256_setzero_si256()));

        lo = _mm256_mul_ps(
            lo,
            _mm256_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(alphas_f32, alphas_f32),
        );
        hi = _mm256_mul_ps(
            hi,
            _mm256_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(alphas_f32, alphas_f32),
        );

        let lo_u = _mm256_cvtps_epi32(lo);
        let hi_u = _mm256_cvtps_epi32(hi);

        let mut packed = _mm256_packus_epi32(lo_u, hi_u);

        packed = _mm256_min_epu16(packed, v_max_test);
        packed = _mm256_blendv_epi8(packed, _mm256_setzero_si256(), is_zero_alpha_mask);
        packed = _mm256_blendv_epi8(packed, rgba, copy_alpha_mask);

        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), packed);
        }
    }

    rem = rem.as_chunks_mut::<16>().1;

    if !rem.is_empty() {
        let mut dst_buffer: [u16; 16] = [0u16; 16];
        dst_buffer[..rem.len()].copy_from_slice(rem);

        let rgba = unsafe { _mm256_loadu_si256(dst_buffer.as_ptr().cast()) };

        let alphas = _mm256_shuffle_epi8(rgba, shuffle_alphas_mask);
        let alphas32 = _mm256_unpacklo_epi16(
            _mm256_shuffle_epi8(rgba, prepare_alphas_mask),
            _mm256_setzero_si256(),
        );
        let is_zero_alpha_mask = _mm256_cmpeq_epi16(alphas, _mm256_setzero_si256());

        let alphas_f32 = _mm256_div_ps(v_scale_colors, _mm256_cvtepi32_ps(alphas32));

        let mut lo = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rgba, _mm256_setzero_si256()));
        let mut hi = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rgba, _mm256_setzero_si256()));

        lo = _mm256_mul_ps(
            lo,
            _mm256_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(alphas_f32, alphas_f32),
        );
        hi = _mm256_mul_ps(
            hi,
            _mm256_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(alphas_f32, alphas_f32),
        );

        let lo_u = _mm256_cvtps_epi32(lo);
        let hi_u = _mm256_cvtps_epi32(hi);

        let mut packed = _mm256_packus_epi32(lo_u, hi_u);

        packed = _mm256_min_epu16(packed, v_max_test);

        packed = _mm256_blendv_epi8(packed, _mm256_setzero_si256(), is_zero_alpha_mask);
        packed = _mm256_blendv_epi8(packed, rgba, copy_alpha_mask);

        unsafe {
            _mm256_storeu_si256(dst_buffer.as_mut_ptr().cast(), packed);
        }

        rem.copy_from_slice(&dst_buffer[..rem.len()]);
    }
}
