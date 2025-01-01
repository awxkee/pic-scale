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

use crate::alpha_handle_u16::{premultiply_alpha_rgba_row, unpremultiply_alpha_rgba_row};
use crate::avx2::utils::{
    _mm256_select_si256, avx_deinterleave_rgba_epi16, avx_interleave_rgba_epi16,
};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn _mm256_scale_by_alpha(px: __m256i, low_low_a: __m256, low_high_a: __m256) -> __m256i {
    let zeros = _mm256_setzero_si256();
    let low_px = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(px, zeros));
    let high_px = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(px, zeros));

    let new_ll = _mm256_cvtps_epi32(_mm256_round_ps::<0x02>(_mm256_mul_ps(low_px, low_low_a)));
    let new_lh = _mm256_cvtps_epi32(_mm256_round_ps::<0x02>(_mm256_mul_ps(high_px, low_high_a)));

    _mm256_packus_epi32(new_ll, new_lh)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_div_by_1023_epi32(v: __m256i) -> __m256i {
    const DIVIDING_BY: i32 = 10;
    let addition = _mm256_set1_epi32(1 << (DIVIDING_BY - 1));
    let v = _mm256_add_epi32(v, addition);
    _mm256_srli_epi32::<DIVIDING_BY>(_mm256_add_epi32(v, _mm256_srli_epi32::<DIVIDING_BY>(v)))
}

#[inline(always)]
pub(crate) unsafe fn _mm256_div_by_4095_epi32(v: __m256i) -> __m256i {
    const DIVIDING_BY: i32 = 12;
    let addition = _mm256_set1_epi32(1 << (DIVIDING_BY - 1));
    let v = _mm256_add_epi32(v, addition);
    _mm256_srli_epi32::<DIVIDING_BY>(_mm256_add_epi32(v, _mm256_srli_epi32::<DIVIDING_BY>(v)))
}

#[inline(always)]
pub(crate) unsafe fn _mm256_div_by_65535_epi32(v: __m256i) -> __m256i {
    const DIVIDING_BY: i32 = 16;
    let addition = _mm256_set1_epi32(1 << (DIVIDING_BY - 1));
    let v = _mm256_add_epi32(v, addition);
    _mm256_srli_epi32::<DIVIDING_BY>(_mm256_add_epi32(v, _mm256_srli_epi32::<DIVIDING_BY>(v)))
}

#[inline(always)]
unsafe fn _mm256_div_by_epi32<const BIT_DEPTH: usize>(v: __m256i) -> __m256i {
    if BIT_DEPTH == 10 {
        _mm256_div_by_1023_epi32(v)
    } else if BIT_DEPTH == 12 {
        _mm256_div_by_4095_epi32(v)
    } else {
        _mm256_div_by_65535_epi32(v)
    }
}

pub(crate) fn avx_premultiply_alpha_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        avx_premultiply_alpha_rgba_u16_impl(dst, src, width, height, bit_depth, pool);
    }
}

trait Avx2PremultiplyExecutor {
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize);
}

#[derive(Default)]
struct Avx2PremultiplyExecutorDefault<const BIT_DEPTH: usize> {}

impl<const BIT_DEPTH: usize> Avx2PremultiplyExecutor for Avx2PremultiplyExecutorDefault<BIT_DEPTH> {
    #[target_feature(enable = "avx2")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        let max_colors = (1 << bit_depth) - 1;

        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .chunks_exact_mut(16 * 4)
            .zip(src_rem.chunks_exact(16 * 4))
        {
            let src_ptr = src.as_ptr();
            let lane0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let lane1 = _mm256_loadu_si256(src_ptr.add(16) as *const __m256i);
            let lane2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let lane3 = _mm256_loadu_si256(src_ptr.add(48) as *const __m256i);

            let pixel = avx_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let zeros = _mm256_setzero_si256();
            let low_alpha = _mm256_unpacklo_epi16(pixel.3, zeros);
            let high_alpha = _mm256_unpackhi_epi16(pixel.3, zeros);

            let new_rrr = _mm256_packus_epi32(
                _mm256_div_by_epi32::<BIT_DEPTH>(_mm256_madd_epi16(
                    _mm256_unpacklo_epi16(pixel.0, zeros),
                    low_alpha,
                )),
                _mm256_div_by_epi32::<BIT_DEPTH>(_mm256_madd_epi16(
                    _mm256_unpackhi_epi16(pixel.0, zeros),
                    high_alpha,
                )),
            );
            let new_ggg = _mm256_packus_epi32(
                _mm256_div_by_epi32::<BIT_DEPTH>(_mm256_madd_epi16(
                    _mm256_unpacklo_epi16(pixel.1, zeros),
                    low_alpha,
                )),
                _mm256_div_by_epi32::<BIT_DEPTH>(_mm256_madd_epi16(
                    _mm256_unpackhi_epi16(pixel.1, zeros),
                    high_alpha,
                )),
            );
            let new_bbb = _mm256_packus_epi32(
                _mm256_div_by_epi32::<BIT_DEPTH>(_mm256_madd_epi16(
                    _mm256_unpacklo_epi16(pixel.2, zeros),
                    low_alpha,
                )),
                _mm256_div_by_epi32::<BIT_DEPTH>(_mm256_madd_epi16(
                    _mm256_unpackhi_epi16(pixel.2, zeros),
                    high_alpha,
                )),
            );

            let dst_ptr = dst.as_mut_ptr();

            let (d_lane0, d_lane1, d_lane2, d_lane3) =
                avx_interleave_rgba_epi16(new_rrr, new_ggg, new_bbb, pixel.3);

            _mm256_storeu_si256(dst_ptr as *mut __m256i, d_lane0);
            _mm256_storeu_si256(dst_ptr.add(16) as *mut __m256i, d_lane1);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, d_lane2);
            _mm256_storeu_si256(dst_ptr.add(48) as *mut __m256i, d_lane3);
        }

        rem = rem.chunks_exact_mut(16 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(16 * 4).remainder();

        premultiply_alpha_rgba_row(rem, src_rem, max_colors);
    }
}

#[derive(Default)]
struct Avx2PremultiplyExecutorAnyBit {}

impl Avx2PremultiplyExecutor for Avx2PremultiplyExecutorAnyBit {
    #[target_feature(enable = "avx2")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        let max_colors = (1 << bit_depth) - 1;

        let mut rem = dst;
        let mut src_rem = src;

        let v_scale_colors = _mm256_set1_ps((1. / max_colors as f64) as f32);
        for (dst, src) in rem
            .chunks_exact_mut(16 * 4)
            .zip(src_rem.chunks_exact(16 * 4))
        {
            let src_ptr = src.as_ptr();
            let lane0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let lane1 = _mm256_loadu_si256(src_ptr.add(16) as *const __m256i);
            let lane2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let lane3 = _mm256_loadu_si256(src_ptr.add(48) as *const __m256i);

            let pixel = avx_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let zeros = _mm256_setzero_si256();

            let low_alpha = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(pixel.3, zeros)),
                v_scale_colors,
            );
            let high_alpha = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(pixel.3, zeros)),
                v_scale_colors,
            );

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

        rem = rem.chunks_exact_mut(16 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(16 * 4).remainder();

        premultiply_alpha_rgba_row(rem, src_rem, max_colors);
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn avx_premultiply_alpha_rgba_u16_row(dst: &mut [u16], src: &[u16], bit_depth: usize) {
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
    } else if bit_depth == 16 {
        avx_pa_dispatch(
            dst,
            src,
            bit_depth,
            Avx2PremultiplyExecutorDefault::<16>::default(),
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
unsafe fn avx_pa_dispatch(
    dst: &mut [u16],
    src: &[u16],
    bit_depth: usize,
    dispatch: impl Avx2PremultiplyExecutor,
) {
    dispatch.premultiply(dst, src, bit_depth);
}

#[target_feature(enable = "avx2")]
unsafe fn avx_premultiply_alpha_rgba_u16_impl(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    _: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(width * 4)
                .zip(src.par_chunks_exact(width * 4))
                .for_each(|(dst, src)| unsafe {
                    avx_premultiply_alpha_rgba_u16_row(dst, src, bit_depth);
                });
        });
    } else {
        dst.chunks_exact_mut(width * 4)
            .zip(src.chunks_exact(width * 4))
            .for_each(|(dst, src)| unsafe {
                avx_premultiply_alpha_rgba_u16_row(dst, src, bit_depth);
            });
    }
}

pub(crate) fn avx_unpremultiply_alpha_rgba_u16(
    in_place: &mut [u16],
    width: usize,
    height: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        avx_unpremultiply_alpha_rgba_u16_impl(in_place, width, height, bit_depth, pool);
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn avx_unpremultiply_alpha_rgba_u16_row(in_place: &mut [u16], bit_depth: usize) {
    let max_colors = (1 << bit_depth) - 1;

    let v_scale_colors = _mm256_set1_ps(max_colors as f32);

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

        let dst_ptr = dst.as_mut_ptr();
        let (d_lane0, d_lane1, d_lane2, d_lane3) =
            avx_interleave_rgba_epi16(new_rrr, new_ggg, new_bbb, pixel.3);

        _mm256_storeu_si256(dst_ptr as *mut __m256i, d_lane0);
        _mm256_storeu_si256(dst_ptr.add(16) as *mut __m256i, d_lane1);
        _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, d_lane2);
        _mm256_storeu_si256(dst_ptr.add(48) as *mut __m256i, d_lane3);
    }

    rem = rem.chunks_exact_mut(16 * 4).into_remainder();

    unpremultiply_alpha_rgba_row(rem, max_colors);
}

#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_u16_impl(
    in_place: &mut [u16],
    width: usize,
    _: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            in_place
                .par_chunks_exact_mut(width * 4)
                .for_each(|row| unsafe {
                    avx_unpremultiply_alpha_rgba_u16_row(row, bit_depth);
                });
        });
    } else {
        in_place.chunks_exact_mut(width * 4).for_each(|row| unsafe {
            avx_unpremultiply_alpha_rgba_u16_row(row, bit_depth);
        });
    }
}
