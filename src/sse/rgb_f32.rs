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

use crate::filter_weights::FilterWeights;
use crate::sse::{_mm_prefer_fma_ps, load_4_weights, shuffle};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn convolve_horizontal_parts_4_rgb_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m128,
    weight1: __m128,
    weight2: __m128,
    weight3: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..).as_ptr();

    let rgb_pixel_0 = _mm_loadu_ps(src_ptr);
    let rgb_pixel_1 = _mm_loadu_ps(src_ptr.add(3));
    let rgb_pixel_2 = _mm_loadu_ps(src_ptr.add(6));
    let rgb_pixel_3 = _mm_setr_ps(
        src_ptr.add(9).read_unaligned(),
        src_ptr.add(10).read_unaligned(),
        src_ptr.add(11).read_unaligned(),
        0f32,
    );

    let acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
    let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
    let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_2, weight2);
    _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_3, weight3)
}

#[inline(always)]
unsafe fn convolve_horizontal_parts_2_rgb_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m128,
    weight1: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..).as_ptr();

    let orig1 = _mm_loadu_ps(src_ptr);
    let rgb_pixel_0 = orig1;
    let rgb_pixel_1 = _mm_setr_ps(
        src_ptr.add(3).read_unaligned(),
        src_ptr.add(4).read_unaligned(),
        src_ptr.add(5).read_unaligned(),
        0f32,
    );

    let mut acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
    acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
    acc
}

#[inline(always)]
unsafe fn convolve_horizontal_parts_one_rgb_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..).as_ptr();
    let rgb_pixel = _mm_setr_ps(
        src_ptr.add(0).read_unaligned(),
        src_ptr.add(1).read_unaligned(),
        src_ptr.add(2).read_unaligned(),
        0f32,
    );
    _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel, weight0)
}

pub(crate) fn convolve_horizontal_rgb_sse_row_one_f32<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgb_sse_row_one_f32_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        } else {
            convolve_horizontal_rgb_sse_row_one_f32_regular(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgb_sse_row_one_f32_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    convolve_horizontal_rgb_sse_row_one_f32_impl::<false>(
        dst_width,
        src_width,
        filter_weights,
        src,
        dst,
    );
}

#[target_feature(enable = "sse4.1", enable = "fma")]
unsafe fn convolve_horizontal_rgb_sse_row_one_f32_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    convolve_horizontal_rgb_sse_row_one_f32_impl::<true>(
        dst_width,
        src_width,
        filter_weights,
        src,
        dst,
    );
}

#[inline]
unsafe fn convolve_horizontal_rgb_sse_row_one_f32_impl<const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    const CHANNELS: usize = 3;
    let mut filter_offset = 0usize;
    let weights_ptr = filter_weights.weights.as_ptr();

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store = _mm_setzero_ps();

        while jx + 4 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let (weight0, weight1, weight2, weight3) = load_4_weights!(ptr);
            let filter_start = jx + bounds.start;
            store = convolve_horizontal_parts_4_rgb_f32::<FMA>(
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

        while jx + 2 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weights = _mm_castsi128_ps(_mm_loadu_si64(ptr as *const u8));
            const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
            let weight0 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_0>(_mm_castps_si128(weights)));
            const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
            let weight1 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_1>(_mm_castps_si128(weights)));
            let filter_start = jx + bounds.start;
            store = convolve_horizontal_parts_2_rgb_f32::<FMA>(
                filter_start,
                src,
                weight0,
                weight1,
                store,
            );
            jx += 2;
        }

        while jx < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight0 = _mm_load1_ps(ptr);
            let filter_start = jx + bounds.start;
            store = convolve_horizontal_parts_one_rgb_f32::<FMA>(filter_start, src, weight0, store);
            jx += 1;
        }

        let px = x * CHANNELS;
        let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(store));
        (dest_ptr as *mut i32)
            .add(2)
            .write_unaligned(_mm_extract_ps::<2>(store));

        filter_offset += filter_weights.aligned_size;
    }
}

pub(crate) fn convolve_horizontal_rgb_sse_rows_4_f32<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgb_sse_rows_4_f32_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        } else {
            convolve_horizontal_rgb_sse_rows_4_f32_regular(
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

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgb_sse_rows_4_f32_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    convolve_horizontal_rgb_sse_rows_4_f32_impl::<false>(
        dst_width,
        src_width,
        filter_weights,
        src,
        src_stride,
        dst,
        dst_stride,
    );
}

#[target_feature(enable = "sse4.1", enable = "fma")]
unsafe fn convolve_horizontal_rgb_sse_rows_4_f32_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    convolve_horizontal_rgb_sse_rows_4_f32_impl::<true>(
        dst_width,
        src_width,
        filter_weights,
        src,
        src_stride,
        dst,
        dst_stride,
    );
}

#[inline(always)]
unsafe fn convolve_horizontal_rgb_sse_rows_4_f32_impl<const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    const CHANNELS: usize = 3;
    let mut filter_offset = 0usize;
    let zeros = _mm_setzero_ps();
    let weights_ptr = filter_weights.weights.as_ptr();

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store_0 = zeros;
        let mut store_1 = zeros;
        let mut store_2 = zeros;
        let mut store_3 = zeros;

        while jx + 4 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let (weight0, weight1, weight2, weight3) = load_4_weights!(ptr);
            let filter_start = jx + bounds.start;
            store_0 = convolve_horizontal_parts_4_rgb_f32::<FMA>(
                filter_start,
                src,
                weight0,
                weight1,
                weight2,
                weight3,
                store_0,
            );
            store_1 = convolve_horizontal_parts_4_rgb_f32::<FMA>(
                filter_start,
                src.get_unchecked(src_stride..),
                weight0,
                weight1,
                weight2,
                weight3,
                store_1,
            );
            store_2 = convolve_horizontal_parts_4_rgb_f32::<FMA>(
                filter_start,
                src.get_unchecked(src_stride * 2..),
                weight0,
                weight1,
                weight2,
                weight3,
                store_2,
            );
            store_3 = convolve_horizontal_parts_4_rgb_f32::<FMA>(
                filter_start,
                src.get_unchecked(src_stride * 3..),
                weight0,
                weight1,
                weight2,
                weight3,
                store_3,
            );
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weights = _mm_castsi128_ps(_mm_loadu_si64(ptr as *const u8));
            const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
            let weight0 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_0>(_mm_castps_si128(weights)));
            const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
            let weight1 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_1>(_mm_castps_si128(weights)));
            let filter_start = jx + bounds.start;
            store_0 = convolve_horizontal_parts_2_rgb_f32::<FMA>(
                filter_start,
                src,
                weight0,
                weight1,
                store_0,
            );
            store_1 = convolve_horizontal_parts_2_rgb_f32::<FMA>(
                filter_start,
                src.get_unchecked(src_stride..),
                weight0,
                weight1,
                store_1,
            );
            store_2 = convolve_horizontal_parts_2_rgb_f32::<FMA>(
                filter_start,
                src.get_unchecked(src_stride * 2..),
                weight0,
                weight1,
                store_2,
            );
            store_3 = convolve_horizontal_parts_2_rgb_f32::<FMA>(
                filter_start,
                src.get_unchecked(src_stride * 3..),
                weight0,
                weight1,
                store_3,
            );
            jx += 2;
        }

        while jx < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight0 = _mm_load1_ps(ptr);
            let filter_start = jx + bounds.start;
            store_0 =
                convolve_horizontal_parts_one_rgb_f32::<FMA>(filter_start, src, weight0, store_0);
            store_1 = convolve_horizontal_parts_one_rgb_f32::<FMA>(
                filter_start,
                src.get_unchecked(src_stride..),
                weight0,
                store_1,
            );
            store_2 = convolve_horizontal_parts_one_rgb_f32::<FMA>(
                filter_start,
                src.get_unchecked(src_stride * 2..),
                weight0,
                store_2,
            );
            store_3 = convolve_horizontal_parts_one_rgb_f32::<FMA>(
                filter_start,
                src.get_unchecked(src_stride * 3..),
                weight0,
                store_3,
            );
            jx += 1;
        }

        let px = x * CHANNELS;
        let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(store_0));
        (dest_ptr as *mut i32)
            .add(2)
            .write_unaligned(_mm_extract_ps::<2>(store_0));

        let dest_ptr = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
        _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(store_1));
        (dest_ptr as *mut i32)
            .add(2)
            .write_unaligned(_mm_extract_ps::<2>(store_1));

        let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
        _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(store_2));
        (dest_ptr as *mut i32)
            .add(2)
            .write_unaligned(_mm_extract_ps::<2>(store_2));

        let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
        _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(store_3));
        (dest_ptr as *mut i32)
            .add(2)
            .write_unaligned(_mm_extract_ps::<2>(store_3));

        filter_offset += filter_weights.aligned_size;
    }
}
