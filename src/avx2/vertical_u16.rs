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

use crate::avx2::utils::{_mm_prefer_fma_ps, _mm256_prefer_fma_ps, shuffle};
use crate::filter_weights::FilterBounds;
use crate::mlaf::mlaf;
use std::arch::x86_64::*;

pub(crate) fn convolve_column_avx_u16(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[f32],
    bit_depth: u32,
) {
    unsafe {
        if std::arch::is_x86_feature_detected!("fma") {
            convolve_column_lb_u16_fma(bounds, src, dst, src_stride, weight, bit_depth);
        } else {
            convolve_column_lb_u16_def(bounds, src, dst, src_stride, weight, bit_depth);
        }
    }
}

#[target_feature(enable = "avx2")]
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

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_column_lb_u16_fma(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[f32],
    bit_depth: u32,
) {
    convolve_column_lb_u16_impl::<true>(bounds, src, dst, src_stride, weight, bit_depth);
}

#[inline(always)]
fn convolve_32_items<const FMA: bool>(
    chunks: &mut [[u16; 32]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weights: &[f32],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = (1i32 << bit_depth) - 1;
    let mut cx = cx;

    unsafe {
        let bounds_size = bounds.size;

        let v_cap_colors = _mm256_set1_epi16((max_colors as u16) as i16);

        let v_px = cx;

        for (x, dst) in chunks.iter_mut().enumerate() {
            let mut store0 = _mm256_setzero_ps();
            let mut store1 = _mm256_setzero_ps();
            let mut store2 = _mm256_setzero_ps();
            let mut store3 = _mm256_setzero_ps();

            let v_dx = v_px + x * 32;

            let mut j = 0usize;

            while j + 4 <= bounds.size {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let weights = _mm_loadu_ps(weights.get_unchecked(j..).as_ptr());

                let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);
                let xw2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(weights, weights);
                let xw3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(weights, weights);

                let w0 = _mm256_setr_m128(xw0, xw0);
                let w1 = _mm256_setr_m128(xw1, xw1);
                let w2 = _mm256_setr_m128(xw2, xw2);
                let w3 = _mm256_setr_m128(xw3, xw3);

                let item_row0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);
                let item_row1 =
                    _mm256_loadu_si256(src_ptr.get_unchecked(16..).as_ptr() as *const __m256i);

                store0 = _mm256_prefer_fma_ps::<FMA>(
                    store0,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row0, _mm256_setzero_si256())),
                    w0,
                );
                store1 = _mm256_prefer_fma_ps::<FMA>(
                    store1,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row0, _mm256_setzero_si256())),
                    w0,
                );
                store2 = _mm256_prefer_fma_ps::<FMA>(
                    store2,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row1, _mm256_setzero_si256())),
                    w0,
                );
                store3 = _mm256_prefer_fma_ps::<FMA>(
                    store3,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row1, _mm256_setzero_si256())),
                    w0,
                );

                let item_row0 = _mm256_loadu_si256(
                    src_ptr.get_unchecked(src_stride..).as_ptr() as *const __m256i
                );
                let item_row1 = _mm256_loadu_si256(
                    src_ptr.get_unchecked(src_stride + 16..).as_ptr() as *const __m256i,
                );

                store0 = _mm256_prefer_fma_ps::<FMA>(
                    store0,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row0, _mm256_setzero_si256())),
                    w1,
                );
                store1 = _mm256_prefer_fma_ps::<FMA>(
                    store1,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row0, _mm256_setzero_si256())),
                    w1,
                );
                store2 = _mm256_prefer_fma_ps::<FMA>(
                    store2,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row1, _mm256_setzero_si256())),
                    w1,
                );
                store3 = _mm256_prefer_fma_ps::<FMA>(
                    store3,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row1, _mm256_setzero_si256())),
                    w1,
                );

                let item_row0 = _mm256_loadu_si256(
                    src_ptr.get_unchecked(src_stride * 2..).as_ptr() as *const __m256i,
                );
                let item_row1 = _mm256_loadu_si256(
                    src_ptr.get_unchecked(src_stride * 2 + 16..).as_ptr() as *const __m256i,
                );

                store0 = _mm256_prefer_fma_ps::<FMA>(
                    store0,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row0, _mm256_setzero_si256())),
                    w2,
                );
                store1 = _mm256_prefer_fma_ps::<FMA>(
                    store1,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row0, _mm256_setzero_si256())),
                    w2,
                );
                store2 = _mm256_prefer_fma_ps::<FMA>(
                    store2,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row1, _mm256_setzero_si256())),
                    w2,
                );
                store3 = _mm256_prefer_fma_ps::<FMA>(
                    store3,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row1, _mm256_setzero_si256())),
                    w2,
                );

                let item_row0 = _mm256_loadu_si256(
                    src_ptr.get_unchecked(src_stride * 3..).as_ptr() as *const __m256i,
                );
                let item_row1 = _mm256_loadu_si256(
                    src_ptr.get_unchecked(src_stride * 3 + 16..).as_ptr() as *const __m256i,
                );

                store0 = _mm256_prefer_fma_ps::<FMA>(
                    store0,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row0, _mm256_setzero_si256())),
                    w3,
                );
                store1 = _mm256_prefer_fma_ps::<FMA>(
                    store1,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row0, _mm256_setzero_si256())),
                    w3,
                );
                store2 = _mm256_prefer_fma_ps::<FMA>(
                    store2,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row1, _mm256_setzero_si256())),
                    w3,
                );
                store3 = _mm256_prefer_fma_ps::<FMA>(
                    store3,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row1, _mm256_setzero_si256())),
                    w3,
                );

                j += 4;
            }

            while j + 2 <= bounds.size {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let weights =
                    _mm_castsi128_ps(_mm_loadu_epi64(weights.get_unchecked(j..).as_ptr().cast()));

                let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);

                let w0 = _mm256_setr_m128(xw0, xw0);
                let w1 = _mm256_setr_m128(xw1, xw1);

                let item_row0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);
                let item_row1 =
                    _mm256_loadu_si256(src_ptr.get_unchecked(16..).as_ptr() as *const __m256i);

                store0 = _mm256_prefer_fma_ps::<FMA>(
                    store0,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row0, _mm256_setzero_si256())),
                    w0,
                );
                store1 = _mm256_prefer_fma_ps::<FMA>(
                    store1,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row0, _mm256_setzero_si256())),
                    w0,
                );
                store2 = _mm256_prefer_fma_ps::<FMA>(
                    store2,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row1, _mm256_setzero_si256())),
                    w0,
                );
                store3 = _mm256_prefer_fma_ps::<FMA>(
                    store3,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row1, _mm256_setzero_si256())),
                    w0,
                );

                let item_row0 = _mm256_loadu_si256(
                    src_ptr.get_unchecked(src_stride..).as_ptr() as *const __m256i
                );
                let item_row1 = _mm256_loadu_si256(
                    src_ptr.get_unchecked(src_stride + 16..).as_ptr() as *const __m256i,
                );

                store0 = _mm256_prefer_fma_ps::<FMA>(
                    store0,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row0, _mm256_setzero_si256())),
                    w1,
                );
                store1 = _mm256_prefer_fma_ps::<FMA>(
                    store1,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row0, _mm256_setzero_si256())),
                    w1,
                );
                store2 = _mm256_prefer_fma_ps::<FMA>(
                    store2,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row1, _mm256_setzero_si256())),
                    w1,
                );
                store3 = _mm256_prefer_fma_ps::<FMA>(
                    store3,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row1, _mm256_setzero_si256())),
                    w1,
                );

                j += 2;
            }

            let weights = &weights[j..bounds_size];

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm256_set1_ps(k_weight);

                let item_row0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);
                let item_row1 =
                    _mm256_loadu_si256(src_ptr.get_unchecked(16..).as_ptr() as *const __m256i);

                store0 = _mm256_prefer_fma_ps::<FMA>(
                    store0,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row0, _mm256_setzero_si256())),
                    v_weight,
                );
                store1 = _mm256_prefer_fma_ps::<FMA>(
                    store1,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row0, _mm256_setzero_si256())),
                    v_weight,
                );
                store2 = _mm256_prefer_fma_ps::<FMA>(
                    store2,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row1, _mm256_setzero_si256())),
                    v_weight,
                );
                store3 = _mm256_prefer_fma_ps::<FMA>(
                    store3,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row1, _mm256_setzero_si256())),
                    v_weight,
                );
            }

            let v_st0 = _mm256_cvtps_epi32(store0);
            let v_st1 = _mm256_cvtps_epi32(store1);
            let v_st2 = _mm256_cvtps_epi32(store2);
            let v_st3 = _mm256_cvtps_epi32(store3);

            let item0 = _mm256_min_epu16(_mm256_packus_epi32(v_st0, v_st1), v_cap_colors);
            let item1 = _mm256_min_epu16(_mm256_packus_epi32(v_st2, v_st3), v_cap_colors);

            _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, item0);
            _mm256_storeu_si256(
                dst.get_unchecked_mut(16..).as_mut_ptr() as *mut __m256i,
                item1,
            );

            cx = v_dx;
        }
        cx
    }
}

#[inline(always)]
fn convolve_16_items<const FMA: bool>(
    chunks: &mut [[u16; 16]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weights: &[f32],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = (1i32 << bit_depth) - 1;
    let mut cx = cx;

    unsafe {
        let bounds_size = bounds.size;

        let v_cap_colors = _mm256_set1_epi16((max_colors as u16) as i16);

        let v_px = cx;

        for (x, dst) in chunks.iter_mut().enumerate() {
            let mut store0 = _mm256_setzero_ps();
            let mut store1 = _mm256_setzero_ps();

            let v_dx = v_px + x * 16;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm256_set1_ps(k_weight);

                let item_row0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);

                store0 = _mm256_prefer_fma_ps::<FMA>(
                    store0,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row0, _mm256_setzero_si256())),
                    v_weight,
                );
                store1 = _mm256_prefer_fma_ps::<FMA>(
                    store1,
                    _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(item_row0, _mm256_setzero_si256())),
                    v_weight,
                );
            }

            let v_st0 = _mm256_cvtps_epi32(store0);
            let v_st1 = _mm256_cvtps_epi32(store1);

            let item0 = _mm256_min_epu16(_mm256_packus_epi32(v_st0, v_st1), v_cap_colors);

            _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, item0);

            cx = v_dx;
        }
        cx
    }
}

#[inline(always)]
fn convolve_8_items<const FMA: bool>(
    chunks: &mut [[u16; 8]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weights: &[f32],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = (1i32 << bit_depth) - 1;
    let mut cx = cx;

    unsafe {
        let bounds_size = bounds.size;

        let v_cap_colors = _mm256_set1_epi16((max_colors as u16) as i16);

        let v_px = cx;

        for (x, dst) in chunks.iter_mut().enumerate() {
            let mut store0 = _mm256_setzero_ps();

            let v_dx = v_px + x * 8;

            const S: i32 = shuffle(3, 1, 2, 0);

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm256_set1_ps(k_weight);

                let item_row = _mm256_permute4x64_epi64::<S>(_mm256_castsi128_si256(
                    _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i),
                ));

                store0 = _mm256_prefer_fma_ps::<FMA>(
                    store0,
                    _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(item_row, _mm256_setzero_si256())),
                    v_weight,
                );
            }

            let v_st0 = _mm256_cvtps_epi32(store0);

            let item = _mm256_min_epu16(
                _mm256_permute4x64_epi64::<S>(_mm256_packus_epi32(v_st0, _mm256_setzero_si256())),
                v_cap_colors,
            );
            _mm_storeu_si128(
                dst.as_mut_ptr() as *mut __m128i,
                _mm256_castsi256_si128(item),
            );

            cx = v_dx;
        }
        cx
    }
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
        let max_colors = (1i32 << bit_depth) - 1;
        let mut cx = 0usize;

        let bounds_size = bounds.size;

        let zeros_ps = _mm_setzero_ps();
        let zeros = _mm_setzero_si128();

        let v_cap_colors = _mm256_set1_epi16((max_colors as u16) as i16);

        cx = convolve_32_items::<FMA>(
            dst.as_chunks_mut::<32>().0,
            bounds,
            src,
            src_stride,
            weight,
            bit_depth,
            cx,
        );

        let mut rem = dst.as_chunks_mut::<32>().1;

        cx = convolve_16_items::<FMA>(
            rem.as_chunks_mut::<16>().0,
            bounds,
            src,
            src_stride,
            weight,
            bit_depth,
            cx,
        );

        rem = rem.as_chunks_mut::<16>().1;

        cx = convolve_8_items::<FMA>(
            rem.as_chunks_mut::<8>().0,
            bounds,
            src,
            src_stride,
            weight,
            bit_depth,
            cx,
        );
        let tail8 = rem.as_chunks_mut::<8>().1;

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

            let v_st = _mm_cvtps_epi32(store0);

            let u_store0 = _mm_min_epu16(
                _mm_packus_epi32(v_st, v_st),
                _mm256_castsi256_si128(v_cap_colors),
            );
            _mm_storeu_si64(dst.as_mut_ptr() as *mut u8, u_store0);

            cx = v_dx;
        }

        let tail4 = tail8.chunks_exact_mut(4).into_remainder();

        let a_px = cx;

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = 0.;

            let v_px = a_px + x;

            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_ptr = src.get_unchecked(offset);

                store0 = mlaf(store0, *src_ptr as f32, k_weight);
            }

            *dst = store0.round().max(0.).min(max_colors as f32) as u16;
        }
    }
}
