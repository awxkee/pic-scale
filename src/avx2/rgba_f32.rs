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

use crate::avx2::utils::{_mm256_fma_ps, avx_combine_ps, shuffle};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

#[inline(always)]
fn convolve_horizontal_parts_one_rgba_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.get_unchecked(start_x * COMPONENTS..);
        let rgb_pixel = _mm_loadu_ps(src_ptr.as_ptr());
        _mm256_fma_ps::<FMA>(
            store_0,
            avx_combine_ps(rgb_pixel, _mm_setzero_ps()),
            weight0,
        )
    }
}

#[inline(always)]
fn convolve_horizontal_parts_4_rgba_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m256,
    weight1: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.get_unchecked(start_x * COMPONENTS..).as_ptr();

        let rgb_pixel_0 = _mm256_loadu_ps(src_ptr);
        let rgb_pixel_1 = _mm256_loadu_ps(src_ptr.add(8));

        let mut acc = _mm256_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
        acc = _mm256_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
        acc
    }
}

#[inline(always)]
fn convolve_horizontal_parts_8_rgba_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m256,
    weight1: __m256,
    weight2: __m256,
    weight3: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.get_unchecked(start_x * COMPONENTS..).as_ptr();

        let rgb_pixel_0 = _mm256_loadu_ps(src_ptr);
        let rgb_pixel_1 = _mm256_loadu_ps(src_ptr.add(8));
        let rgb_pixel_2 = _mm256_loadu_ps(src_ptr.add(16));
        let rgb_pixel_3 = _mm256_loadu_ps(src_ptr.add(24));

        let mut acc = _mm256_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
        acc = _mm256_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
        acc = _mm256_fma_ps::<FMA>(acc, rgb_pixel_2, weight2);
        acc = _mm256_fma_ps::<FMA>(acc, rgb_pixel_3, weight3);
        acc
    }
}

#[inline(always)]
fn convolve_horizontal_parts_2_rgba_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.get_unchecked(start_x * COMPONENTS..);

        let rgb_pixel = _mm256_loadu_ps(src_ptr.as_ptr());

        _mm256_fma_ps::<FMA>(store_0, rgb_pixel, weight0)
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_rows_4_f32<const FMA: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgba_avx_rows_4_f32_fma(
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        } else {
            convolve_horizontal_rgba_avx_rows_4_f32_regular(
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_horizontal_rgba_avx_rows_4_f32_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let unit = Row4ExecutionUnit::<false>::default();
    unit.pass(filter_weights, src, src_stride, dst, dst_stride);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_horizontal_rgba_avx_rows_4_f32_fma(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let unit = Row4ExecutionUnit::<true>::default();
    unit.pass(filter_weights, src, src_stride, dst, dst_stride);
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> Row4ExecutionUnit<FMA> {
    #[inline(always)]
    fn pass(
        &self,
        filter_weights: &FilterWeights<f32>,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        dst_stride: usize,
    ) {
        unsafe {
            const CHANNELS: usize = 4;
            let mut filter_offset = 0usize;
            let zeros = _mm256_setzero_ps();
            let weights_ptr = &filter_weights.weights;

            let dst_width = filter_weights.bounds.len();

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store_0 = zeros;
                let mut store_1 = zeros;
                let mut store_2 = zeros;
                let mut store_3 = zeros;

                while jx + 8 <= bounds.size {
                    let w_ptr = weights_ptr.get_unchecked(jx + filter_offset..);

                    let weights = _mm256_loadu_ps(w_ptr.as_ptr());
                    let w_lo = _mm256_castps256_ps128(weights);
                    let w_hi = _mm256_extractf128_ps::<1>(weights);

                    let w0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w_lo, w_lo);
                    let w1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w_lo, w_lo);
                    let w2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w_lo, w_lo);
                    let w3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w_lo, w_lo);
                    let w4 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w_hi, w_hi);
                    let w5 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w_hi, w_hi);
                    let w6 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w_hi, w_hi);
                    let w7 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w_hi, w_hi);

                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);
                    let w45 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w4), w5);
                    let w67 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w6), w7);

                    let filter_start = jx + bounds.start;

                    store_0 = convolve_horizontal_parts_8_rgba_f32::<FMA>(
                        filter_start,
                        src,
                        w01,
                        w23,
                        w45,
                        w67,
                        store_0,
                    );
                    store_1 = convolve_horizontal_parts_8_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride..),
                        w01,
                        w23,
                        w45,
                        w67,
                        store_1,
                    );
                    store_2 = convolve_horizontal_parts_8_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 2..),
                        w01,
                        w23,
                        w45,
                        w67,
                        store_2,
                    );
                    store_3 = convolve_horizontal_parts_8_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 3..),
                        w01,
                        w23,
                        w45,
                        w67,
                        store_3,
                    );
                    jx += 8;
                }

                while jx + 4 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weights = _mm_loadu_ps(ptr.as_ptr());
                    let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                    let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);
                    let xw2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(weights, weights);
                    let xw3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(weights, weights);

                    let weight0 = avx_combine_ps(xw0, xw1);
                    let weight1 = avx_combine_ps(xw2, xw3);
                    let filter_start = jx + bounds.start;

                    store_0 = convolve_horizontal_parts_4_rgba_f32::<FMA>(
                        filter_start,
                        src,
                        weight0,
                        weight1,
                        store_0,
                    );
                    store_1 = convolve_horizontal_parts_4_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride..),
                        weight0,
                        weight1,
                        store_1,
                    );
                    store_2 = convolve_horizontal_parts_4_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 2..),
                        weight0,
                        weight1,
                        store_2,
                    );
                    store_3 = convolve_horizontal_parts_4_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 3..),
                        weight0,
                        weight1,
                        store_3,
                    );
                    jx += 4;
                }

                while jx + 2 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weights = _mm_castsi128_ps(_mm_loadu_si64(ptr.as_ptr().cast()));
                    let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                    let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);
                    let weight = avx_combine_ps(xw0, xw1);
                    let filter_start = jx + bounds.start;
                    store_0 = convolve_horizontal_parts_2_rgba_f32::<FMA>(
                        filter_start,
                        src,
                        weight,
                        store_0,
                    );
                    store_1 = convolve_horizontal_parts_2_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride..),
                        weight,
                        store_1,
                    );
                    store_2 = convolve_horizontal_parts_2_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 2..),
                        weight,
                        store_2,
                    );
                    store_3 = convolve_horizontal_parts_2_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 3..),
                        weight,
                        store_3,
                    );
                    jx += 2
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset);
                    let filter_start = jx + bounds.start;
                    let weight0 = _mm256_set1_ps(*ptr);
                    store_0 = convolve_horizontal_parts_one_rgba_f32::<FMA>(
                        filter_start,
                        src,
                        weight0,
                        store_0,
                    );
                    store_1 = convolve_horizontal_parts_one_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride..),
                        weight0,
                        store_1,
                    );
                    store_2 = convolve_horizontal_parts_one_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 2..),
                        weight0,
                        store_2,
                    );
                    store_3 = convolve_horizontal_parts_one_rgba_f32::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 3..),
                        weight0,
                        store_3,
                    );
                    jx += 1;
                }

                let px = x * CHANNELS;
                let dest_ptr = dst.get_unchecked_mut(px..);
                _mm_storeu_ps(
                    dest_ptr.as_mut_ptr(),
                    _mm_add_ps(
                        _mm256_castps256_ps128(store_0),
                        _mm256_extractf128_ps::<1>(store_0),
                    ),
                );

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride..);
                _mm_storeu_ps(
                    dest_ptr.as_mut_ptr(),
                    _mm_add_ps(
                        _mm256_castps256_ps128(store_1),
                        _mm256_extractf128_ps::<1>(store_1),
                    ),
                );

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2..);
                _mm_storeu_ps(
                    dest_ptr.as_mut_ptr(),
                    _mm_add_ps(
                        _mm256_castps256_ps128(store_2),
                        _mm256_extractf128_ps::<1>(store_2),
                    ),
                );

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3..);
                _mm_storeu_ps(
                    dest_ptr.as_mut_ptr(),
                    _mm_add_ps(
                        _mm256_castps256_ps128(store_3),
                        _mm256_extractf128_ps::<1>(store_3),
                    ),
                );

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_row_one_f32<const FMA: bool>(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgba_avx_row_one_f32_fma(filter_weights, src, dst);
        } else {
            convolve_horizontal_rgba_avx_row_one_f32_regular(filter_weights, src, dst);
        }
    }
}

#[target_feature(enable = "avx2")]
fn convolve_horizontal_rgba_avx_row_one_f32_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = OneRowExecutionUnit::<false>::default();
    unit.pass(filter_weights, src, dst);
}

#[target_feature(enable = "avx2", enable = "fma")]
fn convolve_horizontal_rgba_avx_row_one_f32_fma(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = OneRowExecutionUnit::<true>::default();
    unit.pass(filter_weights, src, dst);
}

#[derive(Copy, Clone, Default)]
struct OneRowExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> OneRowExecutionUnit<FMA> {
    #[inline(always)]
    fn pass(&self, filter_weights: &FilterWeights<f32>, src: &[f32], dst: &mut [f32]) {
        unsafe {
            const CHANNELS: usize = 4;
            let mut filter_offset = 0usize;
            let weights_ptr = &filter_weights.weights;

            let dst_width = filter_weights.bounds.len();

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store = _mm256_setzero_ps();

                while jx + 8 <= bounds.size {
                    let w_ptr = weights_ptr.get_unchecked(jx + filter_offset..);

                    let weights = _mm256_loadu_ps(w_ptr.as_ptr());
                    let w_lo = _mm256_castps256_ps128(weights);
                    let w_hi = _mm256_extractf128_ps::<1>(weights);

                    let w0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w_lo, w_lo);
                    let w1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w_lo, w_lo);
                    let w2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w_lo, w_lo);
                    let w3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w_lo, w_lo);
                    let w4 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w_hi, w_hi);
                    let w5 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w_hi, w_hi);
                    let w6 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w_hi, w_hi);
                    let w7 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w_hi, w_hi);

                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);
                    let w45 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w4), w5);
                    let w67 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w6), w7);

                    let filter_start = jx + bounds.start;

                    store = convolve_horizontal_parts_8_rgba_f32::<FMA>(
                        filter_start,
                        src,
                        w01,
                        w23,
                        w45,
                        w67,
                        store,
                    );
                    jx += 8;
                }

                while jx + 4 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);

                    let weights = _mm_loadu_ps(ptr.as_ptr());
                    let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                    let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);
                    let xw2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(weights, weights);
                    let xw3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(weights, weights);

                    let weight0 = avx_combine_ps(xw0, xw1);
                    let weight1 = avx_combine_ps(xw2, xw3);
                    let filter_start = jx + bounds.start;
                    store = convolve_horizontal_parts_4_rgba_f32::<FMA>(
                        filter_start,
                        src,
                        weight0,
                        weight1,
                        store,
                    );
                    jx += 4;
                }

                while jx + 2 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weights = _mm_castsi128_ps(_mm_loadu_si64(ptr.as_ptr().cast()));
                    let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                    let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);
                    let weight = avx_combine_ps(xw0, xw1);
                    let filter_start = jx + bounds.start;
                    store = convolve_horizontal_parts_2_rgba_f32::<FMA>(
                        filter_start,
                        src,
                        weight,
                        store,
                    );
                    jx += 2
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset);
                    let weight0 = _mm256_set1_ps(*ptr);
                    let filter_start = jx + bounds.start;
                    store = convolve_horizontal_parts_one_rgba_f32::<FMA>(
                        filter_start,
                        src,
                        weight0,
                        store,
                    );
                    jx += 1;
                }

                let px = x * CHANNELS;
                let dest_ptr = dst.get_unchecked_mut(px..);
                _mm_storeu_ps(
                    dest_ptr.as_mut_ptr(),
                    _mm_add_ps(
                        _mm256_castps256_ps128(store),
                        _mm256_extractf128_ps::<1>(store),
                    ),
                );

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}
