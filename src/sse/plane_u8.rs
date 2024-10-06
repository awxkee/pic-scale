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

use crate::filter_weights::FilterWeights;
use crate::sse::{_mm_hsum_epi32, _mm_muladd_epi32};
use crate::support::{PRECISION, ROUNDING_CONST};

macro_rules! s_accumulate_8_horiz {
    ($store: expr, $ptr: expr, $weights: expr) => {{
        let pixel_colors = _mm_loadu_si64($ptr);
        let px_16 = _mm_cvtepu8_epi16(pixel_colors);
        let px_lo = _mm_unpacklo_epi16(px_16, _mm_setzero_si128());
        let px_hi = _mm_unpackhi_epi16(px_16, _mm_setzero_si128());

        $store = _mm_muladd_epi32($store, px_lo, $weights.0);
        $store = _mm_muladd_epi32($store, px_hi, $weights.1);
    }};
}

macro_rules! s_accumulate_4_horiz {
    ($store: expr, $ptr: expr, $weights: expr) => {{
        let pixel_colors = _mm_setr_epi32(
            $ptr.read_unaligned() as i32,
            $ptr.add(1).read_unaligned() as i32,
            $ptr.add(2).read_unaligned() as i32,
            $ptr.add(3).read_unaligned() as i32,
        );
        $store = _mm_muladd_epi32($store, pixel_colors, $weights);
    }};
}

macro_rules! s_accumulate_1_horiz {
    ($store: expr, $ptr: expr, $weight: expr) => {{
        let pixel_colors = _mm_setr_epi32($ptr.read_unaligned() as i32, 0, 0, 0);
        $store = _mm_muladd_epi32($store, pixel_colors, $weight);
    }};
}

pub fn convolve_horizontal_plane_sse_rows_4_u8(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    unsafe {
        convolve_horizontal_plane_sse_rows_4_u8_impl(
            dst_width,
            src_width,
            filter_weights,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            dst_stride,
        );
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_plane_sse_rows_4_u8_impl(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    let mut filter_offset = 0usize;

    let weights_ptr = filter_weights.weights.as_ptr();

    let zeros = _mm_setzero_si128();

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store0 = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);
        let mut store1 = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);
        let mut store2 = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);
        let mut store3 = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);

        let row1 = unsafe_source_ptr_0.add(src_stride);
        let row2 = unsafe_source_ptr_0.add(src_stride * 2);
        let row3 = unsafe_source_ptr_0.add(src_stride * 3);

        while jx + 8 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weights_i16 = _mm_loadu_si128(ptr as *const __m128i);
            let weights = (
                _mm_unpacklo_epi16(weights_i16, zeros),
                _mm_unpackhi_epi16(weights_i16, zeros),
            );
            let bounds_start = bounds.start + jx;

            let src_ptr = unsafe_source_ptr_0.add(bounds_start);
            s_accumulate_8_horiz!(store0, src_ptr, weights);

            let src_ptr1 = row1.add(bounds_start);
            s_accumulate_8_horiz!(store1, src_ptr1, weights);

            let src_ptr2 = row2.add(bounds_start);
            s_accumulate_8_horiz!(store2, src_ptr2, weights);

            let src_ptr3 = row3.add(bounds_start);
            s_accumulate_8_horiz!(store3, src_ptr3, weights);

            jx += 8;
        }

        while jx + 4 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weights = _mm_cvtepi16_epi32(_mm_loadu_si64(ptr as *const u8));
            let bounds_start = bounds.start + jx;

            let src_ptr = unsafe_source_ptr_0.add(bounds_start);
            s_accumulate_4_horiz!(store0, src_ptr, weights);

            let src_ptr1 = row1.add(bounds_start);
            s_accumulate_4_horiz!(store1, src_ptr1, weights);

            let src_ptr2 = row2.add(bounds_start);
            s_accumulate_4_horiz!(store2, src_ptr2, weights);

            let src_ptr3 = row3.add(bounds_start);
            s_accumulate_4_horiz!(store3, src_ptr3, weights);

            jx += 4;
        }

        while jx < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight = _mm_setr_epi32(ptr.read_unaligned() as i32, 0, 0, 0);
            let bounds_start = bounds.start + jx;

            let src_ptr = unsafe_source_ptr_0.add(bounds_start);
            s_accumulate_1_horiz!(store0, src_ptr, weight);

            let src_ptr1 = row1.add(bounds_start);
            s_accumulate_1_horiz!(store1, src_ptr1, weight);

            let src_ptr2 = row2.add(bounds_start);
            s_accumulate_1_horiz!(store2, src_ptr2, weight);

            let src_ptr3 = row3.add(bounds_start);
            s_accumulate_1_horiz!(store3, src_ptr3, weight);

            jx += 1;
        }

        let sums = _mm_hsum_epi32(store0).max(0);
        let shifted = sums >> PRECISION;
        let value = shifted.min(255) as u8;

        let px = x;
        let dest_ptr = unsafe_destination_ptr_0.add(px);
        dest_ptr.write_unaligned(value);

        let sums = _mm_hsum_epi32(store1).max(0);
        let shifted = sums >> PRECISION;
        let value = shifted.min(255) as u8;

        let px = x;
        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
        dest_ptr.write_unaligned(value);

        let sums = _mm_hsum_epi32(store2).max(0);
        let shifted = sums >> PRECISION;
        let value = shifted.min(255) as u8;

        let px = x;
        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
        dest_ptr.write_unaligned(value);

        let sums = _mm_hsum_epi32(store3).max(0);
        let shifted = sums >> PRECISION;
        let value = shifted.min(255) as u8;

        let px = x;
        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
        dest_ptr.write_unaligned(value);

        filter_offset += filter_weights.aligned_size;
    }
}

pub fn convolve_horizontal_plane_sse_row(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    unsafe {
        convolve_horizontal_plane_sse_row_impl(
            dst_width,
            src_width,
            filter_weights,
            unsafe_source_ptr_0,
            unsafe_destination_ptr_0,
        );
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_plane_sse_row_impl(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    unsafe {
        let mut filter_offset = 0usize;

        let weights_ptr = filter_weights.weights.as_ptr();
        let zeros = _mm_setzero_si128();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);

            while jx + 8 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights_i16 = _mm_loadu_si128(ptr as *const __m128i);
                let weights = (
                    _mm_unpacklo_epi16(weights_i16, zeros),
                    _mm_unpackhi_epi16(weights_i16, zeros),
                );
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                s_accumulate_8_horiz!(store, src_ptr, weights);

                jx += 8;
            }

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = _mm_cvtepi16_epi32(_mm_loadu_si64(ptr as *const u8));
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                s_accumulate_4_horiz!(store, src_ptr, weights);

                jx += 4;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight = _mm_setr_epi32(ptr.read_unaligned() as i32, 0, 0, 0);
                let bounds_start = bounds.start + jx;
                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                s_accumulate_1_horiz!(store, src_ptr, weight);
                jx += 1;
            }

            let sums = _mm_hsum_epi32(store).max(0);
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;

            let px = x;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            dest_ptr.write_unaligned(value);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
