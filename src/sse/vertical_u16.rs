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
use crate::avx2::utils::shuffle;
use crate::filter_weights::FilterBounds;
use crate::sse::{_mm_packus_epi64, _mm_srai_epi64x};
use crate::support::{PRECISION, ROUNDING_APPROX};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn consume_u16_16(
    start_y: usize,
    start_x: usize,
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    filter: *const i16,
    bounds: &FilterBounds,
    max_colors: i32,
) {
    let vld = _mm_set1_epi64x(ROUNDING_APPROX as i64);
    let mut store_0 = vld;
    let mut store_1 = vld;
    let mut store_2 = vld;
    let mut store_3 = vld;
    let mut store_4 = vld;
    let mut store_5 = vld;
    let mut store_6 = vld;
    let mut store_7 = vld;

    let px = start_x;

    let zeros = _mm_set1_epi32(0);

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j).read_unaligned();
        let v_weight = _mm_set1_epi32(weight as i32);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm_loadu_si128(s_ptr as *const __m128i);
        let item_row_1 = _mm_loadu_si128(s_ptr.add(8) as *const __m128i);

        let item_low_0 = _mm_unpacklo_epi16(item_row_0, zeros);
        let item_high_0 = _mm_unpackhi_epi16(item_row_0, zeros);

        let item_low_1 = _mm_unpacklo_epi16(item_row_1, zeros);
        let item_high_1 = _mm_unpackhi_epi16(item_row_1, zeros);

        const SHUFFLE_MASK_LO: i32 = shuffle(2, 1, 3, 0);
        const SHUFFLE_MASK_HI: i32 = shuffle(1, 3, 0, 2);

        store_0 = _mm_add_epi64(
            store_0,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(item_low_0), v_weight),
        );
        store_1 = _mm_add_epi64(
            store_1,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(item_low_0), v_weight),
        );

        store_2 = _mm_add_epi64(
            store_2,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(item_high_0), v_weight),
        );
        store_3 = _mm_add_epi64(
            store_3,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(item_high_0), v_weight),
        );

        store_4 = _mm_add_epi64(
            store_4,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(item_low_1), v_weight),
        );
        store_5 = _mm_add_epi64(
            store_5,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(item_low_1), v_weight),
        );

        store_6 = _mm_add_epi64(
            store_6,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(item_high_1), v_weight),
        );
        store_7 = _mm_add_epi64(
            store_7,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(item_high_1), v_weight),
        );
    }

    let v_max_colors = _mm_set1_epi32(max_colors);
    let n_store_0 = _mm_srai_epi64x::<PRECISION>(store_0);
    let n_store_1 = _mm_srai_epi64x::<PRECISION>(store_1);
    let n_store_2 = _mm_srai_epi64x::<PRECISION>(store_2);
    let n_store_3 = _mm_srai_epi64x::<PRECISION>(store_3);

    let n_store_4 = _mm_srai_epi64x::<PRECISION>(store_4);
    let n_store_5 = _mm_srai_epi64x::<PRECISION>(store_5);
    let n_store_6 = _mm_srai_epi64x::<PRECISION>(store_6);
    let n_store_7 = _mm_srai_epi64x::<PRECISION>(store_7);

    let mut new_store_0 = _mm_packus_epi64(n_store_0, n_store_1);
    new_store_0 = _mm_min_epi32(_mm_max_epi32(new_store_0, zeros), v_max_colors);

    let mut new_store_1 = _mm_packus_epi64(n_store_2, n_store_3);
    new_store_1 = _mm_min_epi32(_mm_max_epi32(new_store_1, zeros), v_max_colors);

    let mut new_store_2 = _mm_packus_epi64(n_store_4, n_store_5);
    new_store_2 = _mm_min_epi32(_mm_max_epi32(new_store_2, zeros), v_max_colors);

    let mut new_store_3 = _mm_packus_epi64(n_store_6, n_store_7);
    new_store_3 = _mm_min_epi32(_mm_max_epi32(new_store_3, zeros), v_max_colors);

    let store_u16_0 = _mm_packus_epi32(new_store_0, new_store_1);
    let store_u16_1 = _mm_packus_epi32(new_store_2, new_store_3);

    let dst_ptr = dst.add(px);
    _mm_storeu_si128(dst_ptr as *mut __m128i, store_u16_0);
    _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, store_u16_1);
}

#[inline(always)]
pub unsafe fn consume_u16_8(
    start_y: usize,
    start_x: usize,
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    filter: *const i16,
    bounds: &FilterBounds,
    max_colors: i32,
) {
    let vld = _mm_set1_epi64x(ROUNDING_APPROX as i64);
    let mut store_0 = vld;
    let mut store_1 = vld;
    let mut store_2 = vld;
    let mut store_3 = vld;

    let px = start_x;

    let zeros = _mm_set1_epi32(0);

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j).read_unaligned();
        let v_weight = _mm_set1_epi32(weight as i32);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = _mm_loadu_si128(s_ptr as *const __m128i);

        let item_low = _mm_unpacklo_epi16(item_row, zeros);
        let item_high = _mm_unpackhi_epi16(item_row, zeros);

        const SHUFFLE_MASK_LO: i32 = shuffle(2, 1, 3, 0);
        const SHUFFLE_MASK_HI: i32 = shuffle(1, 3, 0, 2);

        store_0 = _mm_add_epi64(
            store_0,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(item_low), v_weight),
        );
        store_1 = _mm_add_epi64(
            store_1,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(item_low), v_weight),
        );

        store_2 = _mm_add_epi64(
            store_2,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(item_high), v_weight),
        );
        store_3 = _mm_add_epi64(
            store_3,
            _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(item_high), v_weight),
        );
    }

    let v_max_colors = _mm_set1_epi32(max_colors);
    let n_store_0 = _mm_srai_epi64x::<PRECISION>(store_0);
    let n_store_1 = _mm_srai_epi64x::<PRECISION>(store_1);
    let n_store_2 = _mm_srai_epi64x::<PRECISION>(store_2);
    let n_store_3 = _mm_srai_epi64x::<PRECISION>(store_3);

    let mut new_store_0 = _mm_packus_epi64(n_store_0, n_store_1);
    new_store_0 = _mm_min_epi32(_mm_max_epi32(new_store_0, zeros), v_max_colors);

    let mut new_store_1 = _mm_packus_epi64(n_store_2, n_store_3);
    new_store_1 = _mm_min_epi32(_mm_max_epi32(new_store_1, zeros), v_max_colors);

    let store_u16 = _mm_packus_epi32(new_store_0, new_store_1);

    let dst_ptr = dst.add(px);
    _mm_storeu_si128(dst_ptr as *mut __m128i, store_u16);
}

#[inline(always)]
pub unsafe fn consume_u16_4(
    start_y: usize,
    start_x: usize,
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    filter: *const i16,
    bounds: &FilterBounds,
    max_colors: i32,
) {
    let vld = _mm_set1_epi64x(ROUNDING_APPROX as i64);
    let mut store_0 = vld;
    let mut store_1 = vld;

    let px = start_x;

    let zeros = _mm_setzero_si128();

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j).read_unaligned();
        let v_weight = _mm_set1_epi32(weight as i32);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = _mm_loadu_si64(s_ptr as *const u8);

        let item_row_rescaled = _mm_unpacklo_epi16(item_row, zeros);

        const SHUFFLE_MASK_LO: i32 = shuffle(2, 1, 3, 0);
        let lo = _mm_shuffle_epi32::<SHUFFLE_MASK_LO>(item_row_rescaled);

        store_0 = _mm_add_epi64(store_0, _mm_mul_epi32(lo, v_weight));

        const SHUFFLE_MASK_HI: i32 = shuffle(1, 3, 0, 2);
        let hi = _mm_shuffle_epi32::<SHUFFLE_MASK_HI>(item_row_rescaled);

        store_1 = _mm_add_epi64(store_1, _mm_mul_epi32(hi, v_weight));
    }

    let v_max_colors = _mm_set1_epi32(max_colors);
    let n_store_0 = _mm_srai_epi64x::<PRECISION>(store_0);
    let n_store_1 = _mm_srai_epi64x::<PRECISION>(store_1);

    let mut new_store = _mm_packus_epi64(n_store_0, n_store_1);
    new_store = _mm_min_epi32(_mm_max_epi32(new_store, zeros), v_max_colors);

    let store_u16 = _mm_packus_epi32(new_store, new_store);

    let dst_ptr = dst.add(px);
    std::ptr::copy_nonoverlapping(&store_u16 as *const _ as *const u8, dst_ptr as *mut u8, 8);
}

#[inline(never)]
pub unsafe fn consume_u16_1(
    start_y: usize,
    start_x: usize,
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    filter: *const i16,
    bounds: &FilterBounds,
    max_colors: i32,
) {
    let vld = _mm_set1_epi64x(ROUNDING_APPROX as i64);
    let mut store = vld;

    let px = start_x;

    let zeros = _mm_setzero_si128();

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j).read_unaligned();
        let v_weight = _mm_set1_epi32(weight as i32);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = _mm_set1_epi16(s_ptr.read_unaligned() as i16);

        let low = _mm_unpacklo_epi16(item_row, zeros);

        store = _mm_add_epi64(store, _mm_mul_epi32(low, v_weight));
    }

    let v_max_colors = _mm_set1_epi32(max_colors);

    let shrinked_64 = _mm_srai_epi64x::<PRECISION>(store);
    let shrinked = _mm_packus_epi64(shrinked_64, shrinked_64);
    let shrinked_store = _mm_min_epi32(_mm_max_epi32(shrinked, zeros), v_max_colors);
    let dst_ptr = dst.add(px);
    let value = _mm_extract_epi32::<0>(shrinked_store);
    dst_ptr.write_unaligned(value as u16);
}

pub fn convolve_vertical_rgb_sse_row_u16<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const u16,
    unsafe_destination_ptr_0: *mut u16,
    src_stride: usize,
    weight_ptr: *const i16,
    bit_depth: usize,
) {
    let max_colors = 2i32.pow(bit_depth as u32) - 1i32;
    let mut cx = 0usize;
    let dst_width = width * CHANNELS;

    while cx + 16 < dst_width {
        unsafe {
            consume_u16_16(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
                max_colors,
            );
        }

        cx += 16;
    }

    while cx + 8 < dst_width {
        unsafe {
            consume_u16_8(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
                max_colors,
            );
        }

        cx += 8;
    }

    while cx + 4 < dst_width {
        unsafe {
            consume_u16_4(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
                max_colors,
            );
        }

        cx += 4;
    }

    while cx < dst_width {
        unsafe {
            consume_u16_1(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
                max_colors,
            );
        }
        cx += 1;
    }
}
