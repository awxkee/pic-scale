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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use core::f16;

use crate::filter_weights::FilterWeights;
use crate::sse::f16_utils::{_mm_cvtph_psx, _mm_cvtps_phx};
use crate::sse::{_mm_prefer_fma_ps, load_4_weights, shuffle};

#[inline(always)]
fn convolve_horizontal_parts_one_rgba_f16<const F16C: bool, const FMA: bool>(
    start_x: usize,
    src: &[f16],
    weight0: __m128,
    store_0: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked(start_x * CN..);
        let rgb_pixel = _mm_loadu_si64(src_ptr.as_ptr().cast());
        let pixels = _mm_cvtph_psx::<F16C>(rgb_pixel);
        _mm_prefer_fma_ps::<FMA>(store_0, pixels, weight0)
    }
}

#[inline(always)]
fn convolve_horizontal_parts_4_rgba_f16<const F16C: bool, const FMA: bool>(
    start_x: usize,
    src: &[f16],
    weight0: __m128,
    weight1: __m128,
    weight2: __m128,
    weight3: __m128,
    store_0: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked(start_x * CN..);

        let rgb_pixels_row_0 = _mm_loadu_si128(src_ptr.as_ptr().cast());
        let rgb_pixels_row_1 = _mm_loadu_si128(src_ptr.get_unchecked(8..).as_ptr().cast());

        let rgb_pixel_0 = _mm_cvtph_psx::<F16C>(rgb_pixels_row_0);
        let rgb_pixel_1 = _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(rgb_pixels_row_0));
        let rgb_pixel_2 = _mm_cvtph_psx::<F16C>(rgb_pixels_row_1);
        let rgb_pixel_3 = _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(rgb_pixels_row_1));

        let acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
        let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
        let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_2, weight2);
        _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_3, weight3)
    }
}

#[inline(always)]
fn convolve_horizontal_parts_2_rgba_f16<const F16C: bool, const FMA: bool>(
    start_x: usize,
    src: &[f16],
    weight0: __m128,
    weight1: __m128,
    store_0: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked(start_x * CN..);

        let rgb_pixels = _mm_loadu_si128(src_ptr.as_ptr().cast());

        let acc = _mm_prefer_fma_ps::<FMA>(store_0, _mm_cvtph_psx::<F16C>(rgb_pixels), weight0);
        _mm_prefer_fma_ps::<FMA>(
            acc,
            _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(rgb_pixels)),
            weight1,
        )
    }
}

pub(crate) fn convolve_horizontal_rgba_sse_row_one_f16(
    src: &[f16],
    dst: &mut [f16],
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgba_sse_row_one_f16_regular(filter_weights, src, dst);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn convolve_horizontal_rgba_sse_row_one_f16_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    convolve_horizontal_rgba_sse_row_one_f16_impl::<false, false>(filter_weights, src, dst);
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn convolve_horizontal_rgba_sse_row_one_f16_impl<const F16C: bool, const FMA: bool>(
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    const CN: usize = 4;
    for ((dst, bounds), weights) in dst
        .as_chunks_mut::<CN>()
        .0
        .iter_mut()
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;
        let mut store = _mm_setzero_ps();

        while jx + 4 <= bounds.size {
            let w_s = unsafe { weights.get_unchecked(jx..) };
            let (weight0, weight1, weight2, weight3) = unsafe { load_4_weights!(w_s.as_ptr()) };
            let filter_start = jx + bounds.start;
            store = convolve_horizontal_parts_4_rgba_f16::<F16C, FMA>(
                filter_start,
                src,
                weight0,
                weight1,
                weight2,
                weight3,
                store,
            );
            jx += 4;
        }

        while jx + 2 <= bounds.size {
            let w_s = unsafe { weights.get_unchecked(jx..) };
            let weights = unsafe { _mm_castsi128_ps(_mm_loadu_si64(w_s.as_ptr().cast())) };
            const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
            let weight0 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_0>(_mm_castps_si128(weights)));
            const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
            let weight1 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_1>(_mm_castps_si128(weights)));
            let filter_start = jx + bounds.start;
            store = convolve_horizontal_parts_2_rgba_f16::<F16C, FMA>(
                filter_start,
                src,
                weight0,
                weight1,
                store,
            );
            jx += 2
        }

        while jx < bounds.size {
            let w_s = unsafe { weights.get_unchecked(jx..) };
            let weight0 = unsafe { _mm_load1_ps(w_s.as_ptr()) };
            let filter_start = jx + bounds.start;
            store = convolve_horizontal_parts_one_rgba_f16::<F16C, FMA>(
                filter_start,
                src,
                weight0,
                store,
            );
            jx += 1;
        }

        let converted_f16 = _mm_cvtps_phx::<F16C>(store);
        unsafe {
            _mm_storeu_si64(dst.as_mut_ptr().cast(), converted_f16);
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_sse_rows_4_f16(
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgba_sse_rows_4_f16_regular(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "sse4.1")]
fn convolve_horizontal_rgba_sse_rows_4_f16_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    const F16C: bool = false;
    const FMA: bool = false;
    const CN: usize = 4;
    let zeros = _mm_setzero_ps();
    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
    let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
    let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
    let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

    for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
        .iter_mut()
        .zip(iter_row1.iter_mut())
        .zip(iter_row2.iter_mut())
        .zip(iter_row3.iter_mut())
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;
        let mut store_0 = zeros;
        let mut store_1 = zeros;
        let mut store_2 = zeros;
        let mut store_3 = zeros;
        while jx + 4 <= bounds.size {
            let w_s = unsafe { weights.get_unchecked(jx..) };
            let (weight0, weight1, weight2, weight3) = unsafe { load_4_weights!(w_s.as_ptr()) };
            let filter_start = jx + bounds.start;

            store_0 = convolve_horizontal_parts_4_rgba_f16::<F16C, FMA>(
                filter_start,
                src,
                weight0,
                weight1,
                weight2,
                weight3,
                store_0,
            );
            store_1 = convolve_horizontal_parts_4_rgba_f16::<F16C, FMA>(
                filter_start,
                unsafe { src.get_unchecked(src_stride..) },
                weight0,
                weight1,
                weight2,
                weight3,
                store_1,
            );
            store_2 = convolve_horizontal_parts_4_rgba_f16::<F16C, FMA>(
                filter_start,
                unsafe { src.get_unchecked(src_stride * 2..) },
                weight0,
                weight1,
                weight2,
                weight3,
                store_2,
            );
            store_3 = convolve_horizontal_parts_4_rgba_f16::<F16C, FMA>(
                filter_start,
                unsafe { src.get_unchecked(src_stride * 3..) },
                weight0,
                weight1,
                weight2,
                weight3,
                store_3,
            );
            jx += 4;
        }

        while jx + 2 <= bounds.size {
            let w_s = unsafe { weights.get_unchecked(jx..) };
            let weights = unsafe { _mm_castsi128_ps(_mm_loadu_si64(w_s.as_ptr().cast())) };
            const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
            let weight0 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_0>(_mm_castps_si128(weights)));
            const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
            let weight1 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_1>(_mm_castps_si128(weights)));
            let filter_start = jx + bounds.start;
            store_0 = convolve_horizontal_parts_2_rgba_f16::<F16C, FMA>(
                filter_start,
                src,
                weight0,
                weight1,
                store_0,
            );
            store_1 = convolve_horizontal_parts_2_rgba_f16::<F16C, FMA>(
                filter_start,
                unsafe { src.get_unchecked(src_stride..) },
                weight0,
                weight1,
                store_1,
            );
            store_2 = convolve_horizontal_parts_2_rgba_f16::<F16C, FMA>(
                filter_start,
                unsafe { src.get_unchecked(src_stride * 2..) },
                weight0,
                weight1,
                store_2,
            );
            store_3 = convolve_horizontal_parts_2_rgba_f16::<F16C, FMA>(
                filter_start,
                unsafe { src.get_unchecked(src_stride * 3..) },
                weight0,
                weight1,
                store_3,
            );
            jx += 2;
        }

        while jx < bounds.size {
            let w_s = unsafe { weights.get_unchecked(jx..) };
            let filter_start = jx + bounds.start;
            let weight0 = unsafe { _mm_load1_ps(w_s.as_ptr()) };
            store_0 = convolve_horizontal_parts_one_rgba_f16::<F16C, FMA>(
                filter_start,
                src,
                weight0,
                store_0,
            );
            store_1 = convolve_horizontal_parts_one_rgba_f16::<F16C, FMA>(
                filter_start,
                unsafe { src.get_unchecked(src_stride..) },
                weight0,
                store_1,
            );
            store_2 = convolve_horizontal_parts_one_rgba_f16::<F16C, FMA>(
                filter_start,
                unsafe { src.get_unchecked(src_stride * 2..) },
                weight0,
                store_2,
            );
            store_3 = convolve_horizontal_parts_one_rgba_f16::<F16C, FMA>(
                filter_start,
                unsafe { src.get_unchecked(src_stride * 3..) },
                weight0,
                store_3,
            );
            jx += 1;
        }

        let converted_f16_0 = _mm_cvtps_phx::<F16C>(store_0);
        let converted_f16_1 = _mm_cvtps_phx::<F16C>(store_1);
        let converted_f16_2 = _mm_cvtps_phx::<F16C>(store_2);
        let converted_f16_3 = _mm_cvtps_phx::<F16C>(store_3);

        unsafe {
            _mm_storeu_si64(chunk0.as_mut_ptr().cast(), converted_f16_0);
            _mm_storeu_si64(chunk1.as_mut_ptr().cast(), converted_f16_1);
            _mm_storeu_si64(chunk2.as_mut_ptr().cast(), converted_f16_2);
            _mm_storeu_si64(chunk3.as_mut_ptr().cast(), converted_f16_3);
        }
    }
}
