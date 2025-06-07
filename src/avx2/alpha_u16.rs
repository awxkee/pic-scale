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

use crate::avx2::utils::{
    _mm256_select_si256, avx_deinterleave_rgba_epi16, avx_interleave_rgba_epi16,
};
use novtb::{ParallelZonedIterator, TbSliceMut};
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn _mm256_scale_by_alpha(px: __m256i, low_low_a: __m256, low_high_a: __m256) -> __m256i {
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
pub(crate) unsafe fn _mm256_div_by_1023_epi32(v: __m256i) -> __m256i {
    unsafe {
        const DIVIDING_BY: i32 = 10;
        let addition = _mm256_set1_epi32(1 << (DIVIDING_BY - 1));
        let v = _mm256_add_epi32(v, addition);
        _mm256_srli_epi32::<DIVIDING_BY>(_mm256_add_epi32(v, _mm256_srli_epi32::<DIVIDING_BY>(v)))
    }
}

/// Exact division by 4095 with rounding to nearest
#[inline(always)]
pub(crate) unsafe fn _mm256_div_by_4095_epi32(v: __m256i) -> __m256i {
    unsafe {
        const DIVIDING_BY: i32 = 12;
        let addition = _mm256_set1_epi32(1 << (DIVIDING_BY - 1));
        let v = _mm256_add_epi32(v, addition);
        _mm256_srli_epi32::<DIVIDING_BY>(_mm256_add_epi32(v, _mm256_srli_epi32::<DIVIDING_BY>(v)))
    }
}

/// Exact division by 65535 with rounding to nearest
#[inline(always)]
pub(crate) unsafe fn _mm256_div_by_65535_epi32(v: __m256i) -> __m256i {
    unsafe {
        const DIVIDING_BY: i32 = 16;
        let addition = _mm256_set1_epi32(1 << (DIVIDING_BY - 1));
        let v = _mm256_add_epi32(v, addition);
        _mm256_srli_epi32::<DIVIDING_BY>(_mm256_add_epi32(v, _mm256_srli_epi32::<DIVIDING_BY>(v)))
    }
}

#[inline(always)]
unsafe fn _mm256_div_by_epi32<const BIT_DEPTH: usize>(v: __m256i) -> __m256i {
    unsafe {
        if BIT_DEPTH == 10 {
            _mm256_div_by_1023_epi32(v)
        } else if BIT_DEPTH == 12 {
            _mm256_div_by_4095_epi32(v)
        } else {
            _mm256_div_by_65535_epi32(v)
        }
    }
}

pub(crate) fn avx_premultiply_alpha_rgba_u16(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    width: usize,
    height: usize,
    src_stride: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    unsafe {
        avx_premultiply_alpha_rgba_u16_impl(
            dst, dst_stride, src, width, height, src_stride, bit_depth, pool,
        );
    }
}

trait Avx2PremultiplyExecutor {
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize);
}

#[derive(Default)]
struct Avx2PremultiplyExecutorDefault<const BIT_DEPTH: usize> {}

impl<const BIT_DEPTH: usize> Avx2PremultiplyExecutorDefault<BIT_DEPTH> {
    #[inline(always)]
    unsafe fn premultiply_chunk(&self, dst: &mut [u16], src: &[u16]) {
        unsafe {
            let src_ptr = src.as_ptr();
            let lane0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let lane1 = _mm256_loadu_si256(src_ptr.add(16) as *const __m256i);
            let lane2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let lane3 = _mm256_loadu_si256(src_ptr.add(48) as *const __m256i);

            let pixel = avx_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let zeros = _mm256_setzero_si256();
            let low_alpha = _mm256_unpacklo_epi16(pixel.3, zeros);
            let high_alpha = _mm256_unpackhi_epi16(pixel.3, zeros);

            let rl32 = _mm256_unpacklo_epi16(pixel.0, zeros);
            let rh32 = _mm256_unpackhi_epi16(pixel.0, zeros);
            let gl32 = _mm256_unpacklo_epi16(pixel.1, zeros);
            let gh32 = _mm256_unpackhi_epi16(pixel.1, zeros);
            let bl32 = _mm256_unpacklo_epi16(pixel.2, zeros);
            let bh32 = _mm256_unpackhi_epi16(pixel.2, zeros);

            let rl32 = _mm256_madd_epi16(rl32, low_alpha);
            let rh32 = _mm256_madd_epi16(rh32, high_alpha);
            let gl32 = _mm256_madd_epi16(gl32, low_alpha);
            let gh32 = _mm256_madd_epi16(gh32, high_alpha);
            let bl32 = _mm256_madd_epi16(bl32, low_alpha);
            let bh32 = _mm256_madd_epi16(bh32, high_alpha);

            let lr32 = _mm256_div_by_epi32::<BIT_DEPTH>(rl32);
            let hr32 = _mm256_div_by_epi32::<BIT_DEPTH>(rh32);
            let lg32 = _mm256_div_by_epi32::<BIT_DEPTH>(gl32);
            let hg32 = _mm256_div_by_epi32::<BIT_DEPTH>(gh32);
            let lb32 = _mm256_div_by_epi32::<BIT_DEPTH>(bl32);
            let hb32 = _mm256_div_by_epi32::<BIT_DEPTH>(bh32);

            let new_rrr = _mm256_packus_epi32(lr32, hr32);

            let new_ggg = _mm256_packus_epi32(lg32, hg32);
            let new_bbb = _mm256_packus_epi32(lb32, hb32);

            let dst_ptr = dst.as_mut_ptr();

            let (d_lane0, d_lane1, d_lane2, d_lane3) =
                avx_interleave_rgba_epi16(new_rrr, new_ggg, new_bbb, pixel.3);

            _mm256_storeu_si256(dst_ptr as *mut __m256i, d_lane0);
            _mm256_storeu_si256(dst_ptr.add(16) as *mut __m256i, d_lane1);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, d_lane2);
            _mm256_storeu_si256(dst_ptr.add(48) as *mut __m256i, d_lane3);
        }
    }
}
impl<const BIT_DEPTH: usize> Avx2PremultiplyExecutor for Avx2PremultiplyExecutorDefault<BIT_DEPTH> {
    #[target_feature(enable = "avx2")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], _: usize) {
        unsafe {
            let mut rem = dst;
            let mut src_rem = src;

            for (dst, src) in rem
                .chunks_exact_mut(16 * 4)
                .zip(src_rem.chunks_exact(16 * 4))
            {
                self.premultiply_chunk(dst, src);
            }

            rem = rem.chunks_exact_mut(16 * 4).into_remainder();
            src_rem = src_rem.chunks_exact(16 * 4).remainder();

            if !rem.is_empty() {
                assert!(src_rem.len() < 16 * 4);
                assert!(rem.len() < 16 * 4);
                assert_eq!(src_rem.len(), rem.len());

                let mut buffer: [u16; 16 * 4] = [0u16; 16 * 4];
                let mut dst_buffer: [u16; 16 * 4] = [0u16; 16 * 4];
                std::ptr::copy_nonoverlapping(src_rem.as_ptr(), buffer.as_mut_ptr(), src_rem.len());

                self.premultiply_chunk(&mut dst_buffer, &buffer);

                std::ptr::copy_nonoverlapping(dst_buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }
}

#[derive(Default)]
struct Avx2PremultiplyExecutorAnyBit {}

impl Avx2PremultiplyExecutorAnyBit {
    #[inline(always)]
    unsafe fn premultiply_chunk(&self, dst: &mut [u16], src: &[u16], scale: __m256) {
        unsafe {
            let src_ptr = src.as_ptr();
            let lane0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let lane1 = _mm256_loadu_si256(src_ptr.add(16) as *const __m256i);
            let lane2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let lane3 = _mm256_loadu_si256(src_ptr.add(48) as *const __m256i);

            let pixel = avx_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let zeros = _mm256_setzero_si256();

            let la = _mm256_unpacklo_epi16(pixel.3, zeros);
            let ha = _mm256_unpackhi_epi16(pixel.3, zeros);

            let lla = _mm256_cvtepi32_ps(la);
            let hla = _mm256_cvtepi32_ps(ha);

            let low_alpha = _mm256_mul_ps(lla, scale);
            let high_alpha = _mm256_mul_ps(hla, scale);

            let new_rrr = _mm256_scale_by_alpha(pixel.0, low_alpha, high_alpha);
            let new_ggg = _mm256_scale_by_alpha(pixel.1, low_alpha, high_alpha);
            let new_bbb = _mm256_scale_by_alpha(pixel.2, low_alpha, high_alpha);

            let dst_ptr = dst.as_mut_ptr();

            let (d_lane0, d_lane1, d_lane2, d_lane3) =
                avx_interleave_rgba_epi16(new_rrr, new_ggg, new_bbb, pixel.3);

            _mm256_storeu_si256(dst_ptr as *mut __m256i, d_lane0);
            _mm256_storeu_si256(dst_ptr.add(16) as *mut __m256i, d_lane1);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, d_lane2);
            _mm256_storeu_si256(dst_ptr.add(48) as *mut __m256i, d_lane3);
        }
    }

    #[inline(always)]
    unsafe fn premultiply_work(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        unsafe {
            let max_colors = (1 << bit_depth) - 1;

            let mut rem = dst;
            let mut src_rem = src;

            let v_scale_colors = _mm256_set1_ps((1. / max_colors as f64) as f32);
            for (dst, src) in rem
                .chunks_exact_mut(16 * 4)
                .zip(src_rem.chunks_exact(16 * 4))
            {
                self.premultiply_chunk(dst, src, v_scale_colors);
            }

            rem = rem.chunks_exact_mut(16 * 4).into_remainder();
            src_rem = src_rem.chunks_exact(16 * 4).remainder();

            if !rem.is_empty() {
                assert!(src_rem.len() < 16 * 4);
                assert!(rem.len() < 16 * 4);
                assert_eq!(src_rem.len(), rem.len());

                let mut buffer: [u16; 16 * 4] = [0u16; 16 * 4];
                let mut dst_buffer: [u16; 16 * 4] = [0u16; 16 * 4];
                std::ptr::copy_nonoverlapping(src_rem.as_ptr(), buffer.as_mut_ptr(), src_rem.len());

                self.premultiply_chunk(&mut dst_buffer, &buffer, v_scale_colors);

                std::ptr::copy_nonoverlapping(dst_buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn premultiply_avx(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        unsafe {
            self.premultiply_work(dst, src, bit_depth);
        }
    }
}

impl Avx2PremultiplyExecutor for Avx2PremultiplyExecutorAnyBit {
    #[target_feature(enable = "avx2")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        unsafe {
            self.premultiply_avx(dst, src, bit_depth);
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn avx_premultiply_alpha_rgba_u16_row(dst: &mut [u16], src: &[u16], bit_depth: usize) {
    unsafe {
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
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
#[inline]
unsafe fn avx_pa_dispatch(
    dst: &mut [u16],
    src: &[u16],
    bit_depth: usize,
    dispatch: impl Avx2PremultiplyExecutor,
) {
    unsafe {
        dispatch.premultiply(dst, src, bit_depth);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_premultiply_alpha_rgba_u16_impl(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    width: usize,
    _: usize,
    src_stride: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    dst.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| unsafe {
            let src = &src[y * src_stride..(y + 1) * src_stride];
            avx_premultiply_alpha_rgba_u16_row(&mut dst[..width * 4], &src[..width * 4], bit_depth);
        });
}

pub(crate) fn avx_unpremultiply_alpha_rgba_u16(
    in_place: &mut [u16],
    stride: usize,
    width: usize,
    height: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    unsafe {
        avx_unpremultiply_alpha_rgba_u16_impl(in_place, stride, width, height, bit_depth, pool);
    }
}

/// This inlining is required to activate all features for runtime dispatch
#[inline(always)]
unsafe fn avx_unpremultiply_alpha_rgba_u16_row_impl(in_place: &mut [u16], bit_depth: usize) {
    unsafe {
        let max_colors = (1u32 << bit_depth) - 1;

        let v_scale_colors = _mm256_set1_ps(max_colors as f32);
        let v_max_test = _mm256_set1_epi16(max_colors as i16);

        let mut rem = in_place;

        for dst in rem.chunks_exact_mut(16 * 4) {
            let src_ptr = dst.as_ptr();
            let lane0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let lane1 = _mm256_loadu_si256(src_ptr.add(16) as *const __m256i);
            let lane2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let lane3 = _mm256_loadu_si256(src_ptr.add(48) as *const __m256i);

            let pixel = avx_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let zeros = _mm256_setzero_si256();

            let is_zero_alpha_mask = _mm256_cmpeq_epi16(pixel.3, zeros);

            let mut low_alpha =
                _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_unpacklo_epi16(pixel.3, zeros)));

            low_alpha = _mm256_mul_ps(low_alpha, v_scale_colors);

            let mut high_alpha =
                _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_unpackhi_epi16(pixel.3, zeros)));

            high_alpha = _mm256_mul_ps(high_alpha, v_scale_colors);

            let mut new_rrr = _mm256_scale_by_alpha(pixel.0, low_alpha, high_alpha);
            new_rrr = _mm256_select_si256(is_zero_alpha_mask, pixel.0, new_rrr);
            let mut new_ggg = _mm256_scale_by_alpha(pixel.1, low_alpha, high_alpha);
            new_ggg = _mm256_select_si256(is_zero_alpha_mask, pixel.1, new_ggg);
            let mut new_bbb = _mm256_scale_by_alpha(pixel.2, low_alpha, high_alpha);
            new_bbb = _mm256_select_si256(is_zero_alpha_mask, pixel.2, new_bbb);

            new_rrr = _mm256_min_epu16(new_rrr, v_max_test);
            new_ggg = _mm256_min_epu16(new_ggg, v_max_test);
            new_bbb = _mm256_min_epu16(new_bbb, v_max_test);

            let dst_ptr = dst.as_mut_ptr();
            let (d_lane0, d_lane1, d_lane2, d_lane3) =
                avx_interleave_rgba_epi16(new_rrr, new_ggg, new_bbb, pixel.3);

            _mm256_storeu_si256(dst_ptr as *mut __m256i, d_lane0);
            _mm256_storeu_si256(dst_ptr.add(16) as *mut __m256i, d_lane1);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, d_lane2);
            _mm256_storeu_si256(dst_ptr.add(48) as *mut __m256i, d_lane3);
        }

        rem = rem.chunks_exact_mut(16 * 4).into_remainder();

        if !rem.is_empty() {
            assert!(rem.len() < 16 * 4);

            let mut dst_buffer: [u16; 16 * 4] = [0u16; 16 * 4];
            std::ptr::copy_nonoverlapping(rem.as_ptr(), dst_buffer.as_mut_ptr(), rem.len());

            let lane0 = _mm256_loadu_si256(dst_buffer.as_ptr() as *const __m256i);
            let lane1 = _mm256_loadu_si256(dst_buffer.as_ptr().add(16) as *const __m256i);
            let lane2 = _mm256_loadu_si256(dst_buffer.as_ptr().add(32) as *const __m256i);
            let lane3 = _mm256_loadu_si256(dst_buffer.as_ptr().add(48) as *const __m256i);

            let pixel = avx_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let zeros = _mm256_setzero_si256();

            let is_zero_alpha_mask = _mm256_cmpeq_epi16(pixel.3, zeros);

            let mut low_alpha =
                _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_unpacklo_epi16(pixel.3, zeros)));

            low_alpha = _mm256_mul_ps(low_alpha, v_scale_colors);

            let mut high_alpha =
                _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_unpackhi_epi16(pixel.3, zeros)));

            high_alpha = _mm256_mul_ps(high_alpha, v_scale_colors);

            let mut new_rrr = _mm256_scale_by_alpha(pixel.0, low_alpha, high_alpha);
            new_rrr = _mm256_select_si256(is_zero_alpha_mask, pixel.0, new_rrr);
            let mut new_ggg = _mm256_scale_by_alpha(pixel.1, low_alpha, high_alpha);
            new_ggg = _mm256_select_si256(is_zero_alpha_mask, pixel.1, new_ggg);
            let mut new_bbb = _mm256_scale_by_alpha(pixel.2, low_alpha, high_alpha);
            new_bbb = _mm256_select_si256(is_zero_alpha_mask, pixel.2, new_bbb);

            new_rrr = _mm256_min_epu16(new_rrr, v_max_test);
            new_ggg = _mm256_min_epu16(new_ggg, v_max_test);
            new_bbb = _mm256_min_epu16(new_bbb, v_max_test);

            let (d_lane0, d_lane1, d_lane2, d_lane3) =
                avx_interleave_rgba_epi16(new_rrr, new_ggg, new_bbb, pixel.3);

            _mm256_storeu_si256(dst_buffer.as_mut_ptr() as *mut __m256i, d_lane0);
            _mm256_storeu_si256(dst_buffer.as_mut_ptr().add(16) as *mut __m256i, d_lane1);
            _mm256_storeu_si256(dst_buffer.as_mut_ptr().add(32) as *mut __m256i, d_lane2);
            _mm256_storeu_si256(dst_buffer.as_mut_ptr().add(48) as *mut __m256i, d_lane3);

            std::ptr::copy_nonoverlapping(dst_buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn avx_unpremultiply_alpha_rgba_u16_row_avx(in_place: &mut [u16], bit_depth: usize) {
    unsafe {
        avx_unpremultiply_alpha_rgba_u16_row_impl(in_place, bit_depth);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_u16_impl(
    in_place: &mut [u16],
    stride: usize,
    width: usize,
    _: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    let dispatch = avx_unpremultiply_alpha_rgba_u16_row_avx;

    in_place
        .tb_par_chunks_exact_mut(stride)
        .for_each(pool, |row| unsafe {
            dispatch(&mut row[..width * 4], bit_depth);
        });
}
