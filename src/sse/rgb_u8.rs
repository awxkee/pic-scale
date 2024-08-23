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
use crate::sse::{compress_i32, convolve_horizontal_parts_one_sse_rgb, shuffle};
use crate::support::ROUNDING_APPROX;

pub fn convolve_horizontal_rgb_sse_rows_4(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    unsafe {
        convolve_horizontal_rgb_sse_rows_4_impl(
            dst_width,
            src_width,
            approx_weights,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            dst_stride,
        );
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgb_sse_rows_4_impl(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    unsafe {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;
        let weights_ptr = approx_weights.weights.as_ptr();

        #[rustfmt::skip]
        let shuffle_lo = _mm_setr_epi8(0, -1,
                                               3, -1,
                                               1, -1,
                                               4, -1,
                                               2, -1 ,
                                               5,-1,
                                               -1, -1,
                                               -1, -1);

        #[rustfmt::skip]
        let shuffle_hi = _mm_setr_epi8(6, -1,
                                               9, -1,
                                               7, -1,
                                               10, -1 ,
                                               8,-1,
                                               11, -1,
                                               -1, -1,
                                               -1, -1);

        let vld = _mm_set1_epi32(ROUNDING_APPROX);

        for x in 0..dst_width {
            let bounds = approx_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = vld;
            let mut store_1 = vld;
            let mut store_2 = vld;
            let mut store_3 = vld;

            // Will make step in 4 items however since it is RGB it is necessary to make a safe offset
            while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = _mm_loadu_si64(ptr as *const u8);
                const SHUFFLE_01: i32 = shuffle(0, 0, 0, 0);
                let weight01 = _mm_shuffle_epi32::<SHUFFLE_01>(weights);
                const SHUFFLE_23: i32 = shuffle(1, 1, 1, 1);
                let weight23 = _mm_shuffle_epi32::<SHUFFLE_23>(weights);
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start * CHANNELS);

                let rgb_pixel_0 = _mm_loadu_si128(src_ptr as *const __m128i);
                let rgb_pixel_1 = _mm_loadu_si128(src_ptr.add(src_stride) as *const __m128i);
                let rgb_pixel_2 = _mm_loadu_si128(src_ptr.add(src_stride * 2) as *const __m128i);
                let rgb_pixel_4 = _mm_loadu_si128(src_ptr.add(src_stride * 3) as *const __m128i);

                let hi_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_hi);
                let lo_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_lo);
                let hi_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_hi);
                let lo_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_lo);
                let hi_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_hi);
                let lo_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_lo);
                let hi_3 = _mm_shuffle_epi8(rgb_pixel_4, shuffle_hi);
                let lo_3 = _mm_shuffle_epi8(rgb_pixel_4, shuffle_lo);

                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo_0, weight01));
                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(hi_0, weight23));

                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo_1, weight01));
                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(hi_1, weight23));

                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo_2, weight01));
                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(hi_2, weight23));

                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo_3, weight01));
                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(hi_3, weight23));
                jx += 4;
            }

            while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight01 = _mm_set1_epi32((ptr as *const i32).read_unaligned());

                let src_ptr = unsafe_source_ptr_0.add(bounds_start * CHANNELS);

                let rgb_pixel_0 = _mm_loadu_si128(src_ptr as *const __m128i);
                let rgb_pixel_1 = _mm_loadu_si128(src_ptr.add(src_stride) as *const __m128i);
                let rgb_pixel_2 = _mm_loadu_si128(src_ptr.add(src_stride * 2) as *const __m128i);
                let rgb_pixel_4 = _mm_loadu_si128(src_ptr.add(src_stride * 3) as *const __m128i);

                let lo_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_lo);
                let lo_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_lo);
                let lo_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_lo);
                let lo_3 = _mm_shuffle_epi8(rgb_pixel_4, shuffle_lo);

                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo_0, weight01));
                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo_1, weight01));
                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo_2, weight01));
                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo_3, weight01));

                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;

                let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);

                store_0 = convolve_horizontal_parts_one_sse_rgb(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_one_sse_rgb(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_one_sse_rgb(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_one_sse_rgb(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 3),
                    weight0,
                    store_3,
                );
                jx += 1;
            }

            let store_0_8 = compress_i32(store_0);
            let store_1_8 = compress_i32(store_1);
            let store_2_8 = compress_i32(store_2);
            let store_3_8 = compress_i32(store_3);

            let element_0 = _mm_extract_epi32::<0>(store_0_8);
            let element_1 = _mm_extract_epi32::<0>(store_1_8);
            let element_2 = _mm_extract_epi32::<0>(store_2_8);
            let element_3 = _mm_extract_epi32::<0>(store_3_8);

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);

            let bytes = element_0.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (dest_ptr as *mut u16).write_unaligned(first_byte);
            dest_ptr.add(2).write_unaligned(bytes[2]);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);

            let bytes = element_1.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (dest_ptr as *mut u16).write_unaligned(first_byte);
            dest_ptr.add(2).write_unaligned(bytes[2]);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);

            let bytes = element_2.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (dest_ptr as *mut u16).write_unaligned(first_byte);
            dest_ptr.add(2).write_unaligned(bytes[2]);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);

            let bytes = element_3.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (dest_ptr as *mut u16).write_unaligned(first_byte);
            dest_ptr.add(2).write_unaligned(bytes[2]);

            filter_offset += approx_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgb_sse_row_one(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    unsafe {
        convolve_horizontal_rgb_sse_row_one_impl(
            dst_width,
            src_width,
            approx_weights,
            unsafe_source_ptr_0,
            unsafe_destination_ptr_0,
        );
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgb_sse_row_one_impl(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    const CHANNELS: usize = 3;
    let mut filter_offset = 0usize;
    let weights_ptr = approx_weights.weights.as_ptr();

    #[rustfmt::skip]
    let shuffle_lo = unsafe { _mm_setr_epi8(0, -1,
                                                    3, -1,
                                                    1, -1,
                                                    4, -1,
                                                    2, -1 ,
                                                    5,-1,
                                                    -1, -1,
                                                    -1, -1) };

    #[rustfmt::skip]
    let shuffle_hi = unsafe { _mm_setr_epi8(6, -1,
                                                    9, -1,
                                                    7, -1,
                                                    10, -1 ,
                                                    8,-1,
                                                    11, -1,
                                                    -1, -1,
                                                    -1, -1) };

    for x in 0..dst_width {
        let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
        let mut jx = 0usize;
        let mut store = unsafe { _mm_setzero_si128() };

        while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
            let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
            unsafe {
                let weights = _mm_loadu_si64(ptr as *const u8);
                const SHUFFLE_01: i32 = shuffle(0, 0, 0, 0);
                let weight01 = _mm_shuffle_epi32::<SHUFFLE_01>(weights);
                const SHUFFLE_23: i32 = shuffle(1, 1, 1, 1);
                let weight23 = _mm_shuffle_epi32::<SHUFFLE_23>(weights);
                let bounds_start = bounds.start + jx;
                let src_ptr_0 = unsafe_source_ptr_0.add(bounds_start * CHANNELS);

                let rgb_pixel = _mm_loadu_si128(src_ptr_0 as *const __m128i);
                let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight01));
                store = _mm_add_epi32(store, _mm_madd_epi16(hi, weight23));
            }
            jx += 4;
        }

        while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
            let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
            unsafe {
                let weight0 = _mm_set1_epi32((ptr as *const i32).read_unaligned());
                let src_ptr = unsafe_source_ptr_0.add((bounds.start + jx) * 3);
                let rgb_pixel = _mm_loadu_si64(src_ptr);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
                store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight0));
            }
            jx += 2;
        }

        while jx < bounds.size {
            let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
            unsafe {
                let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                store = convolve_horizontal_parts_one_sse_rgb(
                    bounds.start + jx,
                    unsafe_source_ptr_0,
                    weight0,
                    store,
                );
            }
            jx += 1;
        }

        let store_16_8 = compress_i32(store);

        let px = x * CHANNELS;
        let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

        let element = unsafe { _mm_extract_epi32::<0>(store_16_8) };
        let bytes = element.to_le_bytes();
        unsafe {
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (dest_ptr as *mut u16).write_unaligned(first_byte);
            dest_ptr.add(2).write_unaligned(bytes[2]);
        }

        filter_offset += approx_weights.aligned_size;
    }
}
