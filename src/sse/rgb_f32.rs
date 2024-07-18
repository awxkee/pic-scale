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

use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::sse::_mm_prefer_fma_ps;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_4_rgb_f32(
    start_x: usize,
    src: *const f32,
    weight0: __m128,
    weight1: __m128,
    weight2: __m128,
    weight3: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);
    let zeros = _mm_setzero_ps();
    let mask = _mm_setr_ps(1f32, 1f32, 1f32, 0f32);

    let rgb_pixel_0 = _mm_blendv_ps(zeros, _mm_loadu_ps(src_ptr), mask);
    let rgb_pixel_1 = _mm_blendv_ps(zeros, _mm_loadu_ps(src_ptr.add(3)), mask);
    let rgb_pixel_2 = _mm_blendv_ps(zeros, _mm_loadu_ps(src_ptr.add(6)), mask);
    let rgb_pixel_3 = _mm_setr_ps(
        src_ptr.add(9).read_unaligned(),
        src_ptr.add(10).read_unaligned(),
        src_ptr.add(11).read_unaligned(),
        0f32,
    );

    let acc = _mm_prefer_fma_ps(store_0, rgb_pixel_0, weight0);
    let acc = _mm_prefer_fma_ps(acc, rgb_pixel_1, weight1);
    let acc = _mm_prefer_fma_ps(acc, rgb_pixel_2, weight2);
    let acc = _mm_prefer_fma_ps(acc, rgb_pixel_3, weight3);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_2_rgb_f32(
    start_x: usize,
    src: *const f32,
    weight0: __m128,
    weight1: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);
    let zeros = _mm_setzero_ps();
    let mask = _mm_setr_ps(1f32, 1f32, 1f32, 0f32);

    let rgb_pixel_0 = _mm_blendv_ps(zeros, _mm_loadu_ps(src_ptr), mask);
    let rgb_pixel_1 = _mm_setr_ps(
        src_ptr.add(3).read_unaligned(),
        src_ptr.add(4).read_unaligned(),
        src_ptr.add(5).read_unaligned(),
        0f32,
    );

    let acc = _mm_prefer_fma_ps(store_0, rgb_pixel_0, weight0);
    let acc = _mm_prefer_fma_ps(acc, rgb_pixel_1, weight1);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_one_rgb_f32(
    start_x: usize,
    src: *const f32,
    weight0: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);
    let rgb_pixel = _mm_setr_ps(
        src_ptr.add(0).read_unaligned(),
        src_ptr.add(1).read_unaligned(),
        src_ptr.add(2).read_unaligned(),
        0f32,
    );
    let acc = _mm_prefer_fma_ps(store_0, rgb_pixel, weight0);
    acc
}

pub fn convolve_horizontal_rgb_sse_row_one_f32(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    unsafe {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = _mm_setzero_ps();

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_set1_ps(ptr.read_unaligned());
                let weight1 = _mm_set1_ps(ptr.add(1).read_unaligned());
                let weight2 = _mm_set1_ps(ptr.add(2).read_unaligned());
                let weight3 = _mm_set1_ps(ptr.add(3).read_unaligned());
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_4_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0,
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
                let weight0 = _mm_set1_ps(ptr.read_unaligned());
                let weight1 = _mm_set1_ps(ptr.add(1).read_unaligned());
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_2_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    store,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_set1_ps(ptr.read_unaligned());
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_one_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0,
                    weight0,
                    store,
                );
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            dest_ptr.write_unaligned(f32::from_bits(_mm_extract_ps::<0>(store) as u32));
            dest_ptr
                .add(1)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<1>(store) as u32));
            dest_ptr
                .add(2)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<2>(store) as u32));

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_sse_rows_4_f32(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f32,
    dst_stride: usize,
) {
    unsafe {
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
                let weight0 = _mm_set1_ps(ptr.read_unaligned());
                let weight1 = _mm_set1_ps(ptr.add(1).read_unaligned());
                let weight2 = _mm_set1_ps(ptr.add(2).read_unaligned());
                let weight3 = _mm_set1_ps(ptr.add(3).read_unaligned());
                let filter_start = jx + bounds.start;
                store_0 = convolve_horizontal_parts_4_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_4_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_4_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_4_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0.add(src_stride * 3),
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
                let weight0 = _mm_set1_ps(ptr.read_unaligned());
                let weight1 = _mm_set1_ps(ptr.add(1).read_unaligned());
                let filter_start = jx + bounds.start;
                store_0 = convolve_horizontal_parts_2_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_2_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    weight1,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_2_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    weight1,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_2_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0.add(src_stride * 3),
                    weight0,
                    weight1,
                    store_3,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_set1_ps(ptr.read_unaligned());
                let filter_start = jx + bounds.start;
                store_0 = convolve_horizontal_parts_one_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0,
                    weight0,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_one_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_one_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_one_rgb_f32(
                    filter_start,
                    unsafe_source_ptr_0.add(src_stride * 3),
                    weight0,
                    store_3,
                );
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            dest_ptr.write_unaligned(f32::from_bits(_mm_extract_ps::<0>(store_0) as u32));
            dest_ptr
                .add(1)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<1>(store_0) as u32));
            dest_ptr
                .add(2)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<2>(store_0) as u32));

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            dest_ptr.write_unaligned(f32::from_bits(_mm_extract_ps::<0>(store_1) as u32));
            dest_ptr
                .add(1)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<1>(store_1) as u32));
            dest_ptr
                .add(2)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<2>(store_1) as u32));

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            dest_ptr.write_unaligned(f32::from_bits(_mm_extract_ps::<0>(store_2) as u32));
            dest_ptr
                .add(1)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<1>(store_2) as u32));
            dest_ptr
                .add(2)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<2>(store_2) as u32));

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            dest_ptr.write_unaligned(f32::from_bits(_mm_extract_ps::<0>(store_3) as u32));
            dest_ptr
                .add(1)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<1>(store_3) as u32));
            dest_ptr
                .add(2)
                .write_unaligned(f32::from_bits(_mm_extract_ps::<2>(store_3) as u32));

            filter_offset += filter_weights.aligned_size;
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_16_f32(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: *const f32,
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm_setzero_ps();
    let mut store_1 = _mm_setzero_ps();
    let mut store_2 = _mm_setzero_ps();
    let mut store_3 = _mm_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let v_weight = _mm_set1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm_loadu_ps(s_ptr);
        let item_row_1 = _mm_loadu_ps(s_ptr.add(4));
        let item_row_2 = _mm_loadu_ps(s_ptr.add(8));
        let item_row_3 = _mm_loadu_ps(s_ptr.add(12));

        store_0 = _mm_prefer_fma_ps(store_0, item_row_0, v_weight);
        store_1 = _mm_prefer_fma_ps(store_1, item_row_1, v_weight);
        store_2 = _mm_prefer_fma_ps(store_2, item_row_2, v_weight);
        store_3 = _mm_prefer_fma_ps(store_3, item_row_3, v_weight);
    }

    let dst_ptr = dst.add(px);
    _mm_storeu_ps(dst_ptr, store_0);
    _mm_storeu_ps(dst_ptr.add(4), store_1);
    _mm_storeu_ps(dst_ptr.add(8), store_2);
    _mm_storeu_ps(dst_ptr.add(12), store_3);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_8_f32(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: *const f32,
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm_setzero_ps();
    let mut store_1 = _mm_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let v_weight = _mm_set1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm_loadu_ps(s_ptr);
        let item_row_1 = _mm_loadu_ps(s_ptr.add(4));

        store_0 = _mm_prefer_fma_ps(store_0, item_row_0, v_weight);
        store_1 = _mm_prefer_fma_ps(store_1, item_row_1, v_weight);
    }

    let dst_ptr = dst.add(px);
    _mm_storeu_ps(dst_ptr, store_0);
    _mm_storeu_ps(dst_ptr.add(4), store_1);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_4_f32(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: *const f32,
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let v_weight = _mm_set1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm_loadu_ps(s_ptr);

        store_0 = _mm_prefer_fma_ps(store_0, item_row_0, v_weight);
    }

    let dst_ptr = dst.add(px);
    _mm_storeu_ps(dst_ptr, store_0);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_f32(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: *const f32,
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let v_weight = _mm_set1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm_set1_ps(s_ptr.read_unaligned());

        store_0 = _mm_prefer_fma_ps(store_0, item_row_0, v_weight);
    }

    let dst_ptr = dst.add(px);
    dst_ptr.write_unaligned(f32::from_bits(_mm_extract_ps::<0>(store_0) as u32));
}

#[inline(always)]
pub(crate) fn convolve_vertical_rgb_sse_row_f32<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    src_stride: usize,
    weight_ptr: *const f32,
) {
    let mut cx = 0usize;
    let dst_width = CHANNELS * width;

    while cx + 16 < dst_width {
        unsafe {
            convolve_vertical_part_sse_16_f32(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 16;
    }

    while cx + 8 < dst_width {
        unsafe {
            convolve_vertical_part_sse_8_f32(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 8;
    }

    while cx + 4 < dst_width {
        unsafe {
            convolve_vertical_part_sse_4_f32(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 4;
    }

    while cx < dst_width {
        unsafe {
            convolve_vertical_part_sse_f32(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }
        cx += 1;
    }
}
