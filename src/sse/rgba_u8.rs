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
use crate::sse::{compress_i32};
use crate::support::ROUNDING_APPROX;

unsafe fn convolve_horizontal_parts_one_rgba_sse(
    start_x: usize,
    src: *const u8,
    weight0: __m128i,
    store_0: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);

    let src_ptr_32 = src_ptr as *const i32;
    let rgba_pixel = _mm_cvtsi32_si128(src_ptr_32.read_unaligned());
    let lo = _mm_cvtepu8_epi16(rgba_pixel);

    let acc = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(lo), weight0));
    acc
}

pub fn convolve_horizontal_rgba_sse_rows_4(
    dst_width: usize,
    _: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let weights_ptr = approx_weights.weights.as_ptr();

        #[rustfmt::skip]
        let shuffle_lo = _mm_setr_epi8(0, -1,
                                               4, -1,
                                               1, -1,
                                               5, -1,
                                               2, -1 ,
                                               6,-1,
                                               3, -1,
                                               7, -1);

        #[rustfmt::skip]
        let shuffle_hi = _mm_setr_epi8(8, -1,
                                               12, -1,
                                               9, -1,
                                               13, -1 ,
                                               10,-1,
                                               14, -1,
                                               11, -1,
                                               15, -1);

        let vld = _mm_set1_epi32(ROUNDING_APPROX);

        for x in 0..dst_width {
            let bounds = approx_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = vld;
            let mut store_1 = vld;
            let mut store_2 = vld;
            let mut store_3 = vld;

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight01 = _mm_set1_epi32((ptr as *const i32).read_unaligned());
                let weight23 = _mm_set1_epi32((ptr.add(2) as *const i32).read_unaligned());
                let start_bounds = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(start_bounds * CHANNELS);
                let rgb_pixel = _mm_loadu_si128(src_ptr as *const __m128i);

                let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo, weight01));
                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(hi, weight23));

                let rgb_pixel = _mm_loadu_si128(src_ptr.add(src_stride) as *const __m128i);

                let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo, weight01));
                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(hi, weight23));

                let rgb_pixel = _mm_loadu_si128(src_ptr.add(src_stride * 2) as *const __m128i);

                let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo, weight01));
                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(hi, weight23));

                let rgb_pixel = _mm_loadu_si128(src_ptr.add(src_stride * 3) as *const __m128i);

                let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo, weight01));
                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(hi, weight23));
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;

                let weight01 = _mm_set1_epi32((ptr as *const i32).read_unaligned());
                let src_ptr = unsafe_source_ptr_0.add(bounds_start * CHANNELS);
                let rgb_pixel = _mm_loadu_si64(src_ptr);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo, weight01));

                let rgb_pixel = _mm_loadu_si64(src_ptr.add(src_stride));
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo, weight01));

                let rgb_pixel = _mm_loadu_si64(src_ptr.add(src_stride * 2));
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo, weight01));

                let rgb_pixel = _mm_loadu_si64(src_ptr.add(src_stride * 3));
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo, weight01));
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                let start_bounds = bounds.start + jx;
                store_0 = convolve_horizontal_parts_one_rgba_sse(
                    start_bounds,
                    unsafe_source_ptr_0,
                    weight0,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_one_rgba_sse(
                    start_bounds,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_one_rgba_sse(
                    start_bounds,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_one_rgba_sse(
                    start_bounds,
                    unsafe_source_ptr_0.add(src_stride * 3),
                    weight0,
                    store_3,
                );
                jx += 1;
            }

            let px = x * CHANNELS;

            let store_16_8_0 = compress_i32(store_0);
            let store_16_8_1 = compress_i32(store_1);
            let store_16_8_2 = compress_i32(store_2);
            let store_16_8_3 = compress_i32(store_3);

            let pixel_0 = _mm_extract_epi32::<0>(store_16_8_0);
            let pixel_1 = _mm_extract_epi32::<0>(store_16_8_1);
            let pixel_2 = _mm_extract_epi32::<0>(store_16_8_2);
            let pixel_3 = _mm_extract_epi32::<0>(store_16_8_3);

            let dest_ptr = unsafe_destination_ptr_0.add(px);
            let dest_ptr_32 = dest_ptr as *mut i32;
            dest_ptr_32.write_unaligned(pixel_0);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            let dest_ptr_32 = dest_ptr as *mut i32;
            dest_ptr_32.write_unaligned(pixel_1);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            let dest_ptr_32 = dest_ptr as *mut i32;
            dest_ptr_32.write_unaligned(pixel_2);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            let dest_ptr_32 = dest_ptr as *mut i32;
            dest_ptr_32.write_unaligned(pixel_3);

            filter_offset += approx_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgba_sse_rows_one(
    dst_width: usize,
    _: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let weights_ptr = approx_weights.weights.as_ptr();

        #[rustfmt::skip]
        let shuffle_lo = _mm_setr_epi8(0, -1,
                                               4, -1,
                                               1, -1,
                                               5, -1,
                                               2, -1 ,
                                               6,-1,
                                               3, -1,
                                               7, -1);

        #[rustfmt::skip]
        let shuffle_hi = _mm_setr_epi8(8, -1,
                                               12, -1,
                                               9, -1,
                                               13, -1 ,
                                               10,-1,
                                               14, -1,
                                               11, -1,
                                               15, -1);

        let vld = _mm_set1_epi32(ROUNDING_APPROX);

        for x in 0..dst_width {
            let bounds = approx_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vld;

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight01 = _mm_set1_epi32((ptr as *const i32).read_unaligned());
                let weight23 = _mm_set1_epi32((ptr.add(2) as *const i32).read_unaligned());

                let src_ptr = unsafe_source_ptr_0.add(bounds_start * CHANNELS);
                let rgb_pixel = _mm_loadu_si128(src_ptr as *const __m128i);

                let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight01));
                store = _mm_add_epi32(store, _mm_madd_epi16(hi, weight23));
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight01 = _mm_set1_epi32((ptr as *const i32).read_unaligned());
                let src_ptr = unsafe_source_ptr_0.add(bounds_start * CHANNELS);
                let rgb_pixel = _mm_loadu_si64(src_ptr);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
                store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight01));
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                store = convolve_horizontal_parts_one_rgba_sse(
                    bounds.start + jx,
                    unsafe_source_ptr_0,
                    weight0,
                    store,
                );
                jx += 1;
            }

            let store_16_8 = compress_i32(store);
            let pixel = _mm_extract_epi32::<0>(store_16_8);

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            let dest_ptr_32 = dest_ptr as *mut i32;
            dest_ptr_32.write_unaligned(pixel);

            filter_offset += approx_weights.aligned_size;
        }
    }
}