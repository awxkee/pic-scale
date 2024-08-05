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

use crate::filter_weights::FilterBounds;
use crate::sse::_mm_prefer_fma_ps;

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_f16(
    start_y: usize,
    start_x: usize,
    src: *const half::f16,
    src_stride: usize,
    dst: *mut half::f16,
    filter: *const f32,
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j);
        let v_weight = _mm_load1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm_set1_epi16(s_ptr.read_unaligned().to_bits() as i16);

        store_0 = _mm_prefer_fma_ps(store_0, _mm_cvtph_ps(item_row_0), v_weight);
    }

    let dst_ptr = dst.add(px);
    let converted = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(store_0);
    let first_item = _mm_extract_epi16::<0>(converted) as u16;
    (dst_ptr as *mut u16).write_unaligned(first_item);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_4_f16(
    start_y: usize,
    start_x: usize,
    src: *const half::f16,
    src_stride: usize,
    dst: *mut half::f16,
    filter: *const f32,
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j);
        let v_weight = _mm_load1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm_loadu_si64(s_ptr as *const u8);

        store_0 = _mm_prefer_fma_ps(store_0, _mm_cvtph_ps(item_row_0), v_weight);
    }

    let dst_ptr = dst.add(px);
    let acc = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(store_0);
    std::ptr::copy_nonoverlapping(&acc as *const _ as *const u8, dst_ptr as *mut u8, 8);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_16_16(
    start_y: usize,
    start_x: usize,
    src: *const half::f16,
    src_stride: usize,
    dst: *mut half::f16,
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
        let weight = filter.add(j);
        let v_weight = _mm_load1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm_loadu_si128(s_ptr as *const __m128i);
        let item_row_1 = _mm_loadu_si128(s_ptr.add(8) as *const __m128i);

        let items0 = _mm_cvtph_ps(item_row_0);
        let items1 = _mm_cvtph_ps(_mm_srli_si128::<8>(item_row_0));
        let items2 = _mm_cvtph_ps(item_row_1);
        let items3 = _mm_cvtph_ps(_mm_srli_si128::<8>(item_row_1));

        store_0 = _mm_prefer_fma_ps(store_0, items0, v_weight);
        store_1 = _mm_prefer_fma_ps(store_1, items1, v_weight);
        store_2 = _mm_prefer_fma_ps(store_2, items2, v_weight);
        store_3 = _mm_prefer_fma_ps(store_3, items3, v_weight);
    }

    let dst_ptr = dst.add(px);

    let acc0 = _mm_unpacklo_epi64(
        _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(store_0),
        _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(store_1),
    );
    let acc1 = _mm_unpacklo_epi64(
        _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(store_2),
        _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(store_3),
    );

    _mm_storeu_si128(dst_ptr as *mut __m128i, acc0);
    _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, acc1);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_8_f16(
    start_y: usize,
    start_x: usize,
    src: *const half::f16,
    src_stride: usize,
    dst: *mut half::f16,
    filter: *const f32,
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm_setzero_ps();
    let mut store_1 = _mm_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j);
        let v_weight = _mm_load1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = _mm_loadu_si128(s_ptr as *const __m128i);
        let items0 = _mm_cvtph_ps(item_row);
        let items1 = _mm_cvtph_ps(_mm_srli_si128::<8>(item_row));

        store_0 = _mm_prefer_fma_ps(store_0, items0, v_weight);
        store_1 = _mm_prefer_fma_ps(store_1, items1, v_weight);
    }

    let dst_ptr = dst.add(px);
    let acc0 = _mm_unpacklo_epi64(
        _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(store_0),
        _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(store_1),
    );
    _mm_storeu_si128(dst_ptr as *mut __m128i, acc0);
}

#[inline(always)]
pub fn convolve_vertical_rgb_sse_row_f16<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const half::f16,
    unsafe_destination_ptr_0: *mut half::f16,
    src_stride: usize,
    weight_ptr: *const f32,
) {
    let mut cx = 0usize;
    let dst_width = CHANNELS * width;

    while cx + 16 < dst_width {
        unsafe {
            convolve_vertical_part_sse_16_16(
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
            convolve_vertical_part_sse_8_f16(
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
            convolve_vertical_part_sse_4_f16(
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
            convolve_vertical_part_sse_f16(
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
