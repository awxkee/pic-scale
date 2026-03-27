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

use crate::avx2::utils::{_mm_hsum_ps, _mm_prefer_fma_ps, _mm256_prefer_fma_ps};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

#[inline(always)]
fn conv_horiz_plane_4_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    set1: __m128,
    store: __m128,
) -> __m128 {
    unsafe {
        let src_ptr = src.get_unchecked(start_x..).as_ptr();
        let rgb_pixel = _mm_loadu_ps(src_ptr);
        _mm_prefer_fma_ps::<FMA>(store, rgb_pixel, set1)
    }
}

#[inline(always)]
fn conv_horiz_plane_8_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    set1: __m256,
    store: __m256,
) -> __m256 {
    unsafe {
        let src_ptr = src.get_unchecked(start_x..).as_ptr();
        let rgb_pixel = _mm256_loadu_ps(src_ptr);
        _mm256_prefer_fma_ps::<FMA>(store, rgb_pixel, set1)
    }
}

#[inline(always)]
fn conv_horiz_plane_4_f32_avx<const FMA: bool>(
    start_x: usize,
    src0: &[f32],
    src1: &[f32],
    set1: __m256,
    store: __m256,
) -> __m256 {
    unsafe {
        let rgb_pixel0 = _mm_loadu_ps(src0.get_unchecked(start_x..).as_ptr());
        let rgb_pixel1 = _mm_loadu_ps(src1.get_unchecked(start_x..).as_ptr());
        let rgb_pixel = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel0), rgb_pixel1);
        _mm256_prefer_fma_ps::<FMA>(store, rgb_pixel, set1)
    }
}

#[inline(always)]
fn conv_horiz_plane_2_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    set: __m128,
    store: __m128,
) -> __m128 {
    unsafe {
        let src_ptr = src.get_unchecked(start_x..).as_ptr();
        let rgb_pixel = _mm_castsi128_ps(_mm_loadu_si64(src_ptr.cast()));
        _mm_prefer_fma_ps::<FMA>(store, rgb_pixel, set)
    }
}

#[inline(always)]
fn conv_horiz_plane_2_f32_avx<const FMA: bool>(
    start_x: usize,
    src0: &[f32],
    src1: &[f32],
    set: __m256,
    store: __m256,
) -> __m256 {
    unsafe {
        let rgb_pixel0 = _mm_castsi128_ps(_mm_loadu_si64(
            src0.get_unchecked(start_x..).as_ptr().cast(),
        ));
        let rgb_pixel1 = _mm_castsi128_ps(_mm_loadu_si64(
            src1.get_unchecked(start_x..).as_ptr().cast(),
        ));
        let rgb_pixel = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel0), rgb_pixel1);
        _mm256_prefer_fma_ps::<FMA>(store, rgb_pixel, set)
    }
}

#[inline(always)]
fn conv_horiz_plane_1_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    set: __m128,
    store: __m128,
) -> __m128 {
    unsafe {
        let src_ptr = src.get_unchecked(start_x);
        let rgb_pixel = _mm_load_ss(src_ptr);
        _mm_prefer_fma_ps::<FMA>(store, rgb_pixel, set)
    }
}

#[inline(always)]
fn conv_horiz_plane_1_f32_avx<const FMA: bool>(
    start_x: usize,
    src0: &[f32],
    src1: &[f32],
    set: __m256,
    store: __m256,
) -> __m256 {
    unsafe {
        let rgb_pixel0 = _mm_load_ss(src0.get_unchecked(start_x));
        let rgb_pixel1 = _mm_load_ss(src1.get_unchecked(start_x));
        let rgb_pixel = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel0), rgb_pixel1);
        _mm256_prefer_fma_ps::<FMA>(store, rgb_pixel, set)
    }
}

pub(crate) fn convolve_horizontal_plane_avx_row_one_f32_default(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_plane_avx_row_one_regular(filter_weights, src, dst);
    }
}

pub(crate) fn convolve_horizontal_plane_avx_row_one_f32_fma(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_plane_avx_row_one_fma_impl(filter_weights, src, dst);
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_horizontal_plane_avx_row_one_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    convolve_horizontal_plane_avx_row_one_impl::<false>(filter_weights, src, dst);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_horizontal_plane_avx_row_one_fma_impl(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    convolve_horizontal_plane_avx_row_one_impl::<true>(filter_weights, src, dst);
}

#[inline(always)]
fn convolve_horizontal_plane_avx_row_one_impl<const FMA: bool>(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        let mut filter_offset = 0usize;

        let dst_width = filter_weights.bounds.len();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store256 = _mm256_setzero_ps();

            let local_filters = filter_weights.weights.get_unchecked(filter_offset..);

            while jx + 8 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = local_filters.get_unchecked(jx..);
                let read_weights = _mm256_loadu_ps(ptr.as_ptr());
                store256 = conv_horiz_plane_8_f32::<FMA>(bounds_start, src, read_weights, store256);
                jx += 8;
            }

            let mut store = _mm_add_ps(
                _mm256_castps256_ps128(store256),
                _mm256_extractf128_ps::<1>(store256),
            );

            while jx + 4 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = local_filters.get_unchecked(jx..);
                let read_weights = _mm_loadu_ps(ptr.as_ptr());
                store = conv_horiz_plane_4_f32::<FMA>(bounds_start, src, read_weights, store);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let w = local_filters.get_unchecked(jx..);
                let weights = _mm_castsi128_ps(_mm_loadu_si64(w.as_ptr().cast()));
                store = conv_horiz_plane_2_f32::<FMA>(bounds_start, src, weights, store);
                jx += 2;
            }

            while jx < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = local_filters.get_unchecked(jx..);
                let weight0 = _mm_load_ss(ptr.as_ptr());
                store = conv_horiz_plane_1_f32::<FMA>(bounds_start, src, weight0, store);
                jx += 1;
            }

            let dest_ptr = dst.get_unchecked_mut(x);
            _mm_store_ss(dest_ptr, _mm_hsum_ps(store));

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_plane_avx_rows_4_f32_default(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_plane_avx_rows_4_regular(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

pub(crate) fn convolve_horizontal_plane_avx_rows_4_f32_fma(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_plane_avx_rows_4_fma_impl(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_horizontal_plane_avx_rows_4_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    convolve_horizontal_plane_avx_rows_4_impl::<false>(
        filter_weights,
        src,
        src_stride,
        dst,
        dst_stride,
    );
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_horizontal_plane_avx_rows_4_fma_impl(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    convolve_horizontal_plane_avx_rows_4_impl::<true>(
        filter_weights,
        src,
        src_stride,
        dst,
        dst_stride,
    );
}

#[inline(always)]
fn convolve_horizontal_plane_avx_rows_4_impl<const FMA: bool>(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let mut filter_offset = 0usize;

        let dst_width = filter_weights.bounds.len();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;

            let local_filters = filter_weights.weights.get_unchecked(filter_offset..);

            let mut store256_0 = _mm256_setzero_ps();
            let mut store256_1 = _mm256_setzero_ps();
            let mut store256_2 = _mm256_setzero_ps();
            let mut store256_3 = _mm256_setzero_ps();

            while jx + 8 <= bounds.size {
                let ptr = local_filters.get_unchecked(jx..);
                let w0 = _mm256_loadu_ps(ptr.as_ptr());

                let bounds_start = bounds.start + jx;
                let s_ptr1 = src.get_unchecked(src_stride..);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store256_0 = conv_horiz_plane_8_f32::<FMA>(bounds_start, src, w0, store256_0);
                store256_1 = conv_horiz_plane_8_f32::<FMA>(bounds_start, s_ptr1, w0, store256_1);
                store256_2 = conv_horiz_plane_8_f32::<FMA>(bounds_start, s_ptr2, w0, store256_2);
                store256_3 = conv_horiz_plane_8_f32::<FMA>(bounds_start, s_ptr3, w0, store256_3);
                jx += 8;
            }

            const HI_HI: i32 = 0b0011_0001;
            const LO_LO: i32 = 0b0010_0000;

            let mut store_0 = _mm256_add_ps(
                _mm256_permute2f128_ps::<LO_LO>(store256_0, store256_1),
                _mm256_permute2f128_ps::<HI_HI>(store256_0, store256_1),
            );
            let mut store_1 = _mm256_add_ps(
                _mm256_permute2f128_ps::<LO_LO>(store256_2, store256_3),
                _mm256_permute2f128_ps::<HI_HI>(store256_2, store256_3),
            );

            while jx + 4 <= bounds.size {
                let ptr = local_filters.get_unchecked(jx..);
                let w0 = _mm_loadu_ps(ptr.as_ptr());

                let weights = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w0);

                let bounds_start = bounds.start + jx;
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_0 =
                    conv_horiz_plane_4_f32_avx::<FMA>(bounds_start, src, s_ptr_1, weights, store_0);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_1 = conv_horiz_plane_4_f32_avx::<FMA>(
                    bounds_start,
                    s_ptr2,
                    s_ptr3,
                    weights,
                    store_1,
                );
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w = local_filters.get_unchecked(jx..);
                let w0 = _mm_castsi128_ps(_mm_loadu_si64(w.as_ptr().cast()));
                let weights = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w0);
                let bounds_start = bounds.start + jx;
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_0 =
                    conv_horiz_plane_2_f32_avx::<FMA>(bounds_start, src, s_ptr_1, weights, store_0);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_1 = conv_horiz_plane_2_f32_avx::<FMA>(
                    bounds_start,
                    s_ptr2,
                    s_ptr3,
                    weights,
                    store_1,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = local_filters.get_unchecked(jx..);
                let w0 = _mm_load_ss(ptr.as_ptr());
                let weights = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w0);

                let bounds_start = bounds.start + jx;
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_0 =
                    conv_horiz_plane_1_f32_avx::<FMA>(bounds_start, src, s_ptr_1, weights, store_0);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_1 = conv_horiz_plane_1_f32_avx::<FMA>(
                    bounds_start,
                    s_ptr2,
                    s_ptr3,
                    weights,
                    store_1,
                );
                jx += 1;
            }

            let dest_ptr = dst.get_unchecked_mut(x);
            _mm_store_ss(dest_ptr, _mm_hsum_ps(_mm256_castps256_ps128(store_0)));

            let dest_ptr = dst.get_unchecked_mut(x + dst_stride);
            _mm_store_ss(dest_ptr, _mm_hsum_ps(_mm256_extractf128_ps::<1>(store_0)));

            let dest_ptr = dst.get_unchecked_mut(x + dst_stride * 2);
            _mm_store_ss(dest_ptr, _mm_hsum_ps(_mm256_castps256_ps128(store_1)));

            let dest_ptr = dst.get_unchecked_mut(x + dst_stride * 3);
            _mm_store_ss(dest_ptr, _mm_hsum_ps(_mm256_extractf128_ps::<1>(store_1)));

            filter_offset += filter_weights.aligned_size;
        }
    }
}
