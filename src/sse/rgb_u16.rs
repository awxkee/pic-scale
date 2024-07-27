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
use crate::load_4_weights_epi32;
use crate::sse::{_mm_packus_epi64, _mm_srai_epi64x, _mm_store3_u16, shuffle};
use crate::support::PRECISION;
use crate::support::ROUNDING_APPROX;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn consume_u16_4(
    start_x: usize,
    src: *const u16,
    weight0: __m128i,
    weight1: __m128i,
    weight2: __m128i,
    weight3: __m128i,
    store_0: __m128i,
    store_1: __m128i,
    shuffle_table: __m128i,
) -> (__m128i, __m128i) {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);
    let mut pixel_0 = _mm_loadu_si128(src_ptr as *const __m128i);
    let mut pixel_1 = _mm_loadu_si128(src_ptr.add(6) as *const __m128i);

    pixel_0 = _mm_shuffle_epi8(pixel_0, shuffle_table);
    pixel_1 = _mm_shuffle_epi8(pixel_1, shuffle_table);

    let pixel_lo = _mm_unpacklo_epi16(pixel_0, _mm_setzero_si128());
    let pixel_hi = _mm_unpackhi_epi16(pixel_0, _mm_setzero_si128());

    let pixel_lo_1 = _mm_unpacklo_epi16(pixel_1, _mm_setzero_si128());
    let pixel_hi_1 = _mm_unpackhi_epi16(pixel_1, _mm_setzero_si128());

    const SHUFFLE_MASK_LO: i32 = shuffle(2, 1, 3, 0);
    const SHUFFLE_MASK_HI: i32 = shuffle(1, 3, 0, 2);

    let mut acc0 = _mm_add_epi64(
        store_0,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(pixel_lo), weight0),
    );
    let mut acc1 = _mm_add_epi64(
        store_1,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(pixel_lo), weight0),
    );
    acc0 = _mm_add_epi64(
        acc0,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(pixel_hi), weight1),
    );
    acc1 = _mm_add_epi64(
        acc1,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(pixel_hi), weight1),
    );

    acc0 = _mm_add_epi64(
        acc0,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(pixel_lo_1), weight2),
    );
    acc1 = _mm_add_epi64(
        acc1,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(pixel_lo_1), weight2),
    );

    acc0 = _mm_add_epi64(
        acc0,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(pixel_hi_1), weight3),
    );
    acc1 = _mm_add_epi64(
        acc1,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(pixel_hi_1), weight3),
    );

    (acc0, acc1)
}

#[inline(always)]
unsafe fn consume_u16_2(
    start_x: usize,
    src: *const u16,
    weight0: __m128i,
    weight1: __m128i,
    store_0: __m128i,
    store_1: __m128i,
    shuffle_table: __m128i,
) -> (__m128i, __m128i) {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);
    let mut pixel = _mm_loadu_si128(src_ptr as *const __m128i);

    pixel = _mm_shuffle_epi8(pixel, shuffle_table);

    let pixel_lo = _mm_unpacklo_epi16(pixel, _mm_setzero_si128());
    let pixel_hi = _mm_unpackhi_epi16(pixel, _mm_setzero_si128());

    const SHUFFLE_MASK_LO: i32 = shuffle(2, 1, 3, 0);
    const SHUFFLE_MASK_HI: i32 = shuffle(1, 3, 0, 2);

    let mut acc0 = _mm_add_epi64(
        store_0,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(pixel_lo), weight0),
    );
    let mut acc1 = _mm_add_epi64(
        store_1,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(pixel_lo), weight0),
    );
    acc0 = _mm_add_epi64(
        acc0,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(pixel_hi), weight1),
    );
    acc1 = _mm_add_epi64(
        acc1,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(pixel_hi), weight1),
    );
    (acc0, acc1)
}

#[inline(always)]
unsafe fn consume_u16_1(
    start_x: usize,
    src: *const u16,
    weight: __m128i,
    store_0: __m128i,
    store_1: __m128i,
) -> (__m128i, __m128i) {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);
    let item_row = _mm_setr_epi16(
        src_ptr.read_unaligned() as i16,
        src_ptr.add(1).read_unaligned() as i16,
        src_ptr.add(2).read_unaligned() as i16,
        0,
        0,
        0,
        0,
        0,
    );
    let pixel = _mm_unpacklo_epi16(item_row, _mm_setzero_si128());

    const SHUFFLE_MASK_LO: i32 = shuffle(2, 1, 3, 0);
    const SHUFFLE_MASK_HI: i32 = shuffle(1, 3, 0, 2);

    let acc0 = _mm_add_epi64(
        store_0,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_LO>(pixel), weight),
    );
    let acc1 = _mm_add_epi64(
        store_1,
        _mm_mul_epi32(_mm_shuffle_epi32::<SHUFFLE_MASK_HI>(pixel), weight),
    );
    (acc0, acc1)
}

pub fn convolve_horizontal_rgb_sse_rows_4_u16(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u16,
    dst_stride: usize,
    bit_depth: usize,
) {
    let max_colors = 2i32.pow(bit_depth as u32) - 1i32;
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = approx_weights.weights.as_ptr();
        const CHANNELS: usize = 3;
        let zeros = _mm_setzero_si128();
        let v_max_colors = _mm_set1_epi32(max_colors);

        #[rustfmt::skip]
        let v_shuffle_table = _mm_setr_epi8(0, 1, 2, 3, 4, 5, -1, -1, 6, 7, 8, 9, 10, 11, -1, -1);

        let init = _mm_set1_epi64x(ROUNDING_APPROX as i64);
        for x in 0..dst_width {
            let bounds = approx_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = init;
            let mut store_1 = init;
            let mut store_2 = init;
            let mut store_3 = init;
            let mut store_4 = init;
            let mut store_5 = init;
            let mut store_6 = init;
            let mut store_7 = init;

            while jx + 4 < bounds.size && bounds.start + jx + 5 < src_width {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let (weight0, weight1, weight2, weight3) = load_4_weights_epi32!(ptr);
                let ptr_0 = unsafe_source_ptr_0;
                (store_0, store_1) = consume_u16_4(
                    bounds_start,
                    ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_0,
                    store_1,
                    v_shuffle_table,
                );
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                (store_2, store_3) = consume_u16_4(
                    bounds_start,
                    ptr_1,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_2,
                    store_3,
                    v_shuffle_table,
                );
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                (store_4, store_5) = consume_u16_4(
                    bounds_start,
                    ptr_2,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_4,
                    store_5,
                    v_shuffle_table,
                );
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                (store_6, store_7) = consume_u16_4(
                    bounds_start,
                    ptr_3,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_6,
                    store_7,
                    v_shuffle_table,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                let weight1 = _mm_set1_epi32(ptr.add(1).read_unaligned() as i32);
                let ptr_0 = unsafe_source_ptr_0;
                (store_0, store_1) = consume_u16_2(
                    bounds_start,
                    ptr_0,
                    weight0,
                    weight1,
                    store_0,
                    store_1,
                    v_shuffle_table,
                );
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                (store_2, store_3) = consume_u16_2(
                    bounds_start,
                    ptr_1,
                    weight0,
                    weight1,
                    store_2,
                    store_3,
                    v_shuffle_table,
                );
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                (store_4, store_5) = consume_u16_2(
                    bounds_start,
                    ptr_2,
                    weight0,
                    weight1,
                    store_4,
                    store_5,
                    v_shuffle_table,
                );
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                (store_6, store_7) = consume_u16_2(
                    bounds_start,
                    ptr_3,
                    weight0,
                    weight1,
                    store_6,
                    store_7,
                    v_shuffle_table,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                let ptr_0 = unsafe_source_ptr_0;
                (store_0, store_1) = consume_u16_1(bounds_start, ptr_0, weight0, store_0, store_1);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                (store_2, store_3) = consume_u16_1(bounds_start, ptr_1, weight0, store_2, store_3);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                (store_4, store_5) = consume_u16_1(bounds_start, ptr_2, weight0, store_4, store_5);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                (store_6, store_7) = consume_u16_1(bounds_start, ptr_3, weight0, store_6, store_7);
                jx += 1;
            }

            let px = x * CHANNELS;

            let new_store_0 = _mm_srai_epi64x::<PRECISION>(store_0);
            let new_store_1 = _mm_srai_epi64x::<PRECISION>(store_1);
            let new_store_2 = _mm_srai_epi64x::<PRECISION>(store_2);
            let new_store_3 = _mm_srai_epi64x::<PRECISION>(store_3);
            let new_store_4 = _mm_srai_epi64x::<PRECISION>(store_4);
            let new_store_5 = _mm_srai_epi64x::<PRECISION>(store_5);
            let new_store_6 = _mm_srai_epi64x::<PRECISION>(store_6);
            let new_store_7 = _mm_srai_epi64x::<PRECISION>(store_7);

            let store_u32 = _mm_min_epi32(
                _mm_max_epi32(_mm_packus_epi64(new_store_0, new_store_1), zeros),
                v_max_colors,
            );
            let store_16 = _mm_packus_epi32(store_u32, store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px);
            _mm_store3_u16(dest_ptr, store_16);

            let store_u32 = _mm_min_epi32(
                _mm_max_epi32(_mm_packus_epi64(new_store_2, new_store_3), zeros),
                v_max_colors,
            );
            let store_16 = _mm_packus_epi32(store_u32, store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            _mm_store3_u16(dest_ptr, store_16);

            let store_u32 = _mm_min_epi32(
                _mm_max_epi32(_mm_packus_epi64(new_store_4, new_store_5), zeros),
                v_max_colors,
            );
            let store_16 = _mm_packus_epi32(store_u32, store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            _mm_store3_u16(dest_ptr, store_16);

            let store_u32 = _mm_min_epi32(
                _mm_max_epi32(_mm_packus_epi64(new_store_6, new_store_7), zeros),
                v_max_colors,
            );
            let store_16 = _mm_packus_epi32(store_u32, store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            _mm_store3_u16(dest_ptr, store_16);

            filter_offset += approx_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgb_sse_row_u16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u16,
    unsafe_destination_ptr_0: *mut u16,
    bit_depth: usize,
) {
    let max_colors = 2i32.pow(bit_depth as u32) - 1i32;
    unsafe {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;

        let weights_ptr = filter_weights.weights.as_ptr();

        let v_max_colors = _mm_set1_epi32(max_colors);
        let zeros = _mm_setzero_si128();

        #[rustfmt::skip]
        let v_shuffle_table = _mm_setr_epi8(0, 1, 2, 3, 4, 5, -1, -1, 6, 7, 8, 9, 10, 11, -1, -1);

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store0 = _mm_set1_epi64x(ROUNDING_APPROX as i64);
            let mut store1 = _mm_set1_epi64x(ROUNDING_APPROX as i64);

            while jx + 4 < bounds.size && bounds.start + jx + 5 < src_width {
                let ptr = weights_ptr.add(jx + filter_offset);

                let (weight0, weight1, weight2, weight3) = load_4_weights_epi32!(ptr);

                let bounds_start = bounds.start + jx;
                (store0, store1) = consume_u16_4(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store0,
                    store1,
                    v_shuffle_table,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                let weight1 = _mm_set1_epi32(ptr.add(1).read_unaligned() as i32);
                let bounds_start = bounds.start + jx;
                (store0, store1) = consume_u16_2(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    store0,
                    store1,
                    v_shuffle_table,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                let bounds_start = bounds.start + jx;
                (store0, store1) =
                    consume_u16_1(bounds_start, unsafe_source_ptr_0, weight0, store0, store1);
                jx += 1;
            }

            let px = x * CHANNELS;

            let new_store_0 = _mm_srai_epi64x::<PRECISION>(store0);
            let new_store_1 = _mm_srai_epi64x::<PRECISION>(store1);

            let store_u32 = _mm_min_epi32(
                _mm_max_epi32(_mm_packus_epi64(new_store_0, new_store_1), zeros),
                v_max_colors,
            );
            let store_16 = _mm_packus_epi32(store_u32, store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px);
            _mm_store3_u16(dest_ptr, store_16);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
