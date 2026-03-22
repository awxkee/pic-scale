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

use crate::filter_weights::FilterBounds;
use crate::mlaf::mlaf;
use crate::sse::_mm_prefer_fma_ps;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn convolve_column_sse_u16(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[f32],
    bit_depth: u32,
) {
    unsafe {
        convolve_column_lb_u16_def(bounds, src, dst, src_stride, weight, bit_depth);
    }
}

#[target_feature(enable = "sse4.1")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_column_lb_u16_def(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[f32],
    bit_depth: u32,
) {
    convolve_column_lb_u16_impl::<false>(bounds, src, dst, src_stride, weight, bit_depth);
}

#[inline(always)]
fn convolve_column_lb_u16_impl<const FMA: bool>(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[f32],
    bit_depth: u32,
) {
    unsafe {
        let max_colors = (1 << bit_depth) - 1;
        let mut cx = 0usize;

        let bounds_size = bounds.size;

        let zeros_ps = _mm_setzero_ps();
        let zeros = _mm_setzero_si128();

        let v_max_colors = _mm_set1_epi32(max_colors);

        let v_px = cx;

        let iter16 = dst.chunks_exact_mut(16);

        for (x, dst) in iter16.enumerate() {
            let mut store0 = zeros_ps;
            let mut store1 = zeros_ps;
            let mut store2 = zeros_ps;
            let mut store3 = zeros_ps;

            let v_dx = v_px + x * 16;

            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm_set1_ps(k_weight);

                let item_row0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
                let item_row1 = _mm_loadu_si128(src_ptr.as_ptr().add(8) as *const __m128i);

                store0 = _mm_prefer_fma_ps::<FMA>(
                    store0,
                    _mm_cvtepi32_ps(_mm_unpacklo_epi16(item_row0, zeros)),
                    v_weight,
                );
                store1 = _mm_prefer_fma_ps::<FMA>(
                    store1,
                    _mm_cvtepi32_ps(_mm_unpackhi_epi16(item_row0, zeros)),
                    v_weight,
                );
                store2 = _mm_prefer_fma_ps::<FMA>(
                    store2,
                    _mm_cvtepi32_ps(_mm_unpacklo_epi16(item_row1, zeros)),
                    v_weight,
                );
                store3 = _mm_prefer_fma_ps::<FMA>(
                    store3,
                    _mm_cvtepi32_ps(_mm_unpackhi_epi16(item_row1, zeros)),
                    v_weight,
                );
            }

            let v_st0 = _mm_min_epi32(_mm_cvtps_epi32(store0), v_max_colors);
            let v_st1 = _mm_min_epi32(_mm_cvtps_epi32(store1), v_max_colors);
            let v_st2 = _mm_min_epi32(_mm_cvtps_epi32(store2), v_max_colors);
            let v_st3 = _mm_min_epi32(_mm_cvtps_epi32(store3), v_max_colors);

            let item0 = _mm_packus_epi32(v_st0, v_st1);
            let item1 = _mm_packus_epi32(v_st2, v_st3);

            _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, item0);
            _mm_storeu_si128(dst.as_mut_ptr().add(8) as *mut __m128i, item1);

            cx += 16;
        }

        let tail16 = dst.chunks_exact_mut(16).into_remainder();
        let iter8 = tail16.chunks_exact_mut(8);

        let v_px = cx;

        for (x, dst) in iter8.enumerate() {
            let mut store0 = zeros_ps;
            let mut store1 = zeros_ps;

            let v_dx = v_px + x * 8;

            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm_set1_ps(k_weight);

                let item_row = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);

                store0 = _mm_prefer_fma_ps::<FMA>(
                    store0,
                    _mm_cvtepi32_ps(_mm_unpacklo_epi16(item_row, zeros)),
                    v_weight,
                );
                store1 = _mm_prefer_fma_ps::<FMA>(
                    store1,
                    _mm_cvtepi32_ps(_mm_unpackhi_epi16(item_row, zeros)),
                    v_weight,
                );
            }

            let v_st0 = _mm_min_epi32(_mm_cvtps_epi32(store0), v_max_colors);
            let v_st1 = _mm_min_epi32(_mm_cvtps_epi32(store1), v_max_colors);

            let item = _mm_packus_epi32(v_st0, v_st1);
            _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, item);

            cx += 8;
        }

        let tail8 = tail16.chunks_exact_mut(8).into_remainder();
        let iter4 = tail8.chunks_exact_mut(4);

        let v_cx = cx;

        for (x, dst) in iter4.enumerate() {
            let mut store0 = zeros_ps;

            let v_dx = v_cx + x * 4;

            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm_set1_ps(k_weight);

                let item_row = _mm_loadu_si64(src_ptr.as_ptr() as *const u8);

                store0 = _mm_prefer_fma_ps::<FMA>(
                    store0,
                    _mm_cvtepi32_ps(_mm_unpacklo_epi16(item_row, zeros)),
                    v_weight,
                );
            }

            let v_st = _mm_min_epi32(_mm_cvtps_epi32(store0), v_max_colors);

            let u_store0 = _mm_packus_epi32(v_st, v_st);
            _mm_storeu_si64(dst.as_mut_ptr() as *mut u8, u_store0);

            cx += 4;
        }

        let tail4 = tail8.chunks_exact_mut(4).into_remainder();

        let a_px = cx;

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = 0.;

            let v_px = a_px + x;

            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_ptr = src.get_unchecked(offset..(offset + 1));

                store0 = mlaf(store0, src_ptr[0] as f32, k_weight);
            }

            *dst = store0.round().max(0.).min(max_colors as f32) as u16;
        }
    }
}
