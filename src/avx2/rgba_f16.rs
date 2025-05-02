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

use crate::avx2::routines::*;
use crate::avx2::utils::{_mm256_fma_ps, avx_combine_ps};
use crate::filter_weights::FilterWeights;
use core::f16;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn convolve_horizontal_parts_one_rgba_f16<const FMA: bool>(
    start_x: usize,
    src: *const f16,
    weight0: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.add(start_x * COMPONENTS);
        let rgb_pixel = _mm_loadu_si64(src_ptr as *const u8);
        let pixels = avx_combine_ps(_mm_cvtph_ps(rgb_pixel), _mm_setzero_ps());
        _mm256_fma_ps::<FMA>(store_0, pixels, weight0)
    }
}

#[inline(always)]
unsafe fn convolve_horizontal_parts_4_rgba_f16<const FMA: bool>(
    start_x: usize,
    src: *const f16,
    weight0: __m256,
    weight1: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.add(start_x * COMPONENTS);

        let rgb_pixels_row_0 = _mm256_loadu_si256(src_ptr as *const __m256i);

        let rgb_pixel_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(rgb_pixels_row_0));
        let rgb_pixel_1 = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(rgb_pixels_row_0));

        let acc = _mm256_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
        _mm256_fma_ps::<FMA>(acc, rgb_pixel_1, weight1)
    }
}

#[inline(always)]
unsafe fn convolve_horizontal_parts_8_rgba_f16<const FMA: bool>(
    start_x: usize,
    src: *const f16,
    weight0: __m256,
    weight1: __m256,
    weight2: __m256,
    weight3: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.add(start_x * COMPONENTS);

        let rgb_pixels_row_0 = _mm256_loadu_si256(src_ptr as *const __m256i);
        let rgb_pixels_row_1 = _mm256_loadu_si256(src_ptr.add(16) as *const __m256i);

        let rgb_pixel_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(rgb_pixels_row_0));
        let rgb_pixel_1 = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(rgb_pixels_row_0));
        let rgb_pixel_2 = _mm256_cvtph_ps(_mm256_castsi256_si128(rgb_pixels_row_1));
        let rgb_pixel_3 = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(rgb_pixels_row_1));

        let mut acc = _mm256_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
        acc = _mm256_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
        acc = _mm256_fma_ps::<FMA>(acc, rgb_pixel_2, weight2);
        acc = _mm256_fma_ps::<FMA>(acc, rgb_pixel_3, weight3);
        acc
    }
}

#[inline(always)]
unsafe fn convolve_horizontal_parts_2_rgba_f16<const FMA: bool>(
    start_x: usize,
    src: *const f16,
    weight0: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.add(start_x * COMPONENTS);
        let rgb_pixels = _mm_loadu_si128(src_ptr as *const __m128i);
        _mm256_fma_ps::<FMA>(store_0, _mm256_cvtph_ps(rgb_pixels), weight0)
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_row_one_f16<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgba_avx_row_one_f16_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        } else {
            convolve_horizontal_rgba_avx_row_one_f16_regular(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        }
    }
}

#[target_feature(enable = "avx2", enable = "f16c", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgba_avx_row_one_f16_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    unsafe {
        convolve_horizontal_rgba_avx_row_one_f16_impl::<true>(
            dst_width,
            src_width,
            filter_weights,
            src,
            dst,
        );
    }
}

#[target_feature(enable = "avx2", enable = "f16c")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgba_avx_row_one_f16_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    unsafe {
        convolve_horizontal_rgba_avx_row_one_f16_impl::<false>(
            dst_width,
            src_width,
            filter_weights,
            src,
            dst,
        );
    }
}

#[inline]
unsafe fn convolve_horizontal_rgba_avx_row_one_f16_impl<const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    unsafe {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = _mm256_setzero_ps();

            while jx + 8 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let (weight0, weight1, weight2, weight3) = load_8_weights_group_4_avx!(ptr);
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_8_rgba_f16::<FMA>(
                    filter_start,
                    src.as_ptr(),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store,
                );
                jx += 8;
            }

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let (weight0, weight1) = load_4_weights_group_2_avx!(ptr);
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_4_rgba_f16::<FMA>(
                    filter_start,
                    src.as_ptr(),
                    weight0,
                    weight1,
                    store,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_load1_ps(ptr);
                let weight1 = _mm_load1_ps(ptr.add(1));
                let weight = avx_combine_ps(weight0, weight1);
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_2_rgba_f16::<FMA>(
                    filter_start,
                    src.as_ptr(),
                    weight,
                    store,
                );
                jx += 2
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm256_set1_ps(ptr.read_unaligned());
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_one_rgba_f16::<FMA>(
                    filter_start,
                    src.as_ptr(),
                    weight0,
                    store,
                );
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            let converted_f16 = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_add_ps(
                _mm256_castps256_ps128(store),
                _mm256_extractf128_ps::<1>(store),
            ));
            _mm_storeu_si64(dest_ptr as *mut u8, converted_f16);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_rows_4_f16<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgba_avx_rows_4_f16_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        } else {
            convolve_horizontal_rgba_avx_rows_4_f16_regular(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        }
    }
}

#[target_feature(enable = "avx2", enable = "f16c")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgba_avx_rows_4_f16_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    unsafe {
        convolve_horizontal_rgba_avx_rows_4_f16_impl::<false>(
            dst_width,
            src_width,
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "avx2", enable = "f16c", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgba_avx_rows_4_f16_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    unsafe {
        convolve_horizontal_rgba_avx_rows_4_f16_impl::<true>(
            dst_width,
            src_width,
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[inline(always)]
unsafe fn convolve_horizontal_rgba_avx_rows_4_f16_impl<const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let zeros = _mm256_setzero_ps();
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 8 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let (weight0, weight1, weight2, weight3) = load_8_weights_group_4_avx!(ptr);
                let filter_start = jx + bounds.start;

                store_0 = convolve_horizontal_parts_8_rgba_f16::<FMA>(
                    filter_start,
                    src.as_ptr(),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_8_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride..).as_ptr(),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_8_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..).as_ptr(),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_8_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 3..).as_ptr(),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_3,
                );
                jx += 8;
            }

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let (weight0, weight1) = load_4_weights_group_2_avx!(ptr);
                let filter_start = jx + bounds.start;

                store_0 = convolve_horizontal_parts_4_rgba_f16::<FMA>(
                    filter_start,
                    src.as_ptr(),
                    weight0,
                    weight1,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_4_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride..).as_ptr(),
                    weight0,
                    weight1,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_4_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..).as_ptr(),
                    weight0,
                    weight1,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_4_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 3..).as_ptr(),
                    weight0,
                    weight1,
                    store_3,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_load1_ps(ptr);
                let weight1 = _mm_load1_ps(ptr.add(1));
                let weight = avx_combine_ps(weight0, weight1);
                let filter_start = jx + bounds.start;
                store_0 = convolve_horizontal_parts_2_rgba_f16::<FMA>(
                    filter_start,
                    src.as_ptr(),
                    weight,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_2_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride..).as_ptr(),
                    weight,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_2_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..).as_ptr(),
                    weight,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_2_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 3..).as_ptr(),
                    weight,
                    store_3,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let filter_start = jx + bounds.start;
                let weight0 = _mm256_set1_ps(ptr.read_unaligned());
                store_0 = convolve_horizontal_parts_one_rgba_f16::<FMA>(
                    filter_start,
                    src.as_ptr(),
                    weight0,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_one_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride..).as_ptr(),
                    weight0,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_one_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..).as_ptr(),
                    weight0,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_one_rgba_f16::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 3..).as_ptr(),
                    weight0,
                    store_3,
                );
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            let converted_f16_0 = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_add_ps(
                _mm256_castps256_ps128(store_0),
                _mm256_extractf128_ps::<1>(store_0),
            ));
            _mm_storeu_si64(dest_ptr as *mut u8, converted_f16_0);

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
            let converted_f16_1 = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_add_ps(
                _mm256_castps256_ps128(store_1),
                _mm256_extractf128_ps::<1>(store_1),
            ));
            _mm_storeu_si64(dest_ptr as *mut u8, converted_f16_1);

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
            let converted_f16_2 = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_add_ps(
                _mm256_castps256_ps128(store_2),
                _mm256_extractf128_ps::<1>(store_2),
            ));
            _mm_storeu_si64(dest_ptr as *mut u8, converted_f16_2);

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
            let converted_f16_3 = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_add_ps(
                _mm256_castps256_ps128(store_3),
                _mm256_extractf128_ps::<1>(store_3),
            ));
            _mm_storeu_si64(dest_ptr as *mut u8, converted_f16_3);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
