/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod sse_rgb {
    use crate::filter_weights::{FilterBounds, FilterWeights};
    use crate::sse::sse_convolve_u8;
    use crate::support::ROUNDING_APPROX;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    pub(crate) fn convolve_horizontal_rgba_sse_rows_4(
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
           let shuffle_lo =_mm_setr_epi8(0, -1,
                                         4, -1,
                                         1, -1,
                                         5, -1,
                                         2, -1 ,
                                         6,-1,
                                         3, -1,
                                         7, -1);

            #[rustfmt::skip]
           let shuffle_hi =_mm_setr_epi8(8, -1,
                                         12, -1,
                                         9, -1,
                                         13, -1 ,
                                         10,-1,
                                         14, -1,
                                         11, -1,
                                         15, -1);

            let vld = _mm_set1_epi32(ROUNDING_APPROX);

            for x in 0..dst_width {
                let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
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
                    let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                    let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                    let start_bounds = bounds.start + jx;
                    store_0 = sse_convolve_u8::convolve_horizontal_parts_one_rgba_sse(
                        start_bounds,
                        unsafe_source_ptr_0,
                        weight0,
                        store_0,
                    );
                    store_1 = sse_convolve_u8::convolve_horizontal_parts_one_rgba_sse(
                        start_bounds,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        store_1,
                    );
                    store_2 = sse_convolve_u8::convolve_horizontal_parts_one_rgba_sse(
                        start_bounds,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        store_2,
                    );
                    store_3 = sse_convolve_u8::convolve_horizontal_parts_one_rgba_sse(
                        start_bounds,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        store_3,
                    );
                    jx += 1;
                }
                let store_16_8 = sse_convolve_u8::compress_i32(store_0);
                let pixel = unsafe { _mm_extract_epi32::<0>(store_16_8) };

                let px = x * CHANNELS;
                let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
                let dest_ptr_32 = dest_ptr as *mut i32;
                dest_ptr_32.write_unaligned(pixel);

                let store_16_8 = sse_convolve_u8::compress_i32(store_1);
                let pixel = unsafe { _mm_extract_epi32::<0>(store_16_8) };

                let px = x * CHANNELS;
                let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride) };
                let dest_ptr_32 = dest_ptr as *mut i32;
                dest_ptr_32.write_unaligned(pixel);

                let store_16_8 = sse_convolve_u8::compress_i32(store_2);
                let pixel = unsafe { _mm_extract_epi32::<0>(store_16_8) };

                let px = x * CHANNELS;
                let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 2) };
                let dest_ptr_32 = dest_ptr as *mut i32;
                dest_ptr_32.write_unaligned(pixel);

                let store_16_8 = sse_convolve_u8::compress_i32(store_3);
                let pixel = unsafe { _mm_extract_epi32::<0>(store_16_8) };

                let px = x * CHANNELS;
                let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 3) };
                let dest_ptr_32 = dest_ptr as *mut i32;
                dest_ptr_32.write_unaligned(pixel);

                filter_offset += approx_weights.aligned_size;
            }
        }
    }

    pub(crate) fn convolve_horizontal_rgba_sse_rows_one(
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
           let shuffle_lo =_mm_setr_epi8(0, -1,
                                         4, -1,
                                         1, -1,
                                         5, -1,
                                         2, -1 ,
                                         6,-1,
                                         3, -1,
                                         7, -1);

            #[rustfmt::skip]
           let shuffle_hi =_mm_setr_epi8(8, -1,
                                         12, -1,
                                         9, -1,
                                         13, -1 ,
                                         10,-1,
                                         14, -1,
                                         11, -1,
                                         15, -1);

            let vld = _mm_set1_epi32(ROUNDING_APPROX);

            for x in 0..dst_width {
                let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
                let mut jx = 0usize;
                let mut store = vld;

                while jx + 4 < bounds.size {
                    let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
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
                    let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                    let bounds_start = bounds.start + jx;
                    let weight01 = _mm_set1_epi32((ptr as *const i32).read_unaligned());
                    let src_ptr = unsafe_source_ptr_0.add(bounds_start * CHANNELS);
                    let rgb_pixel = _mm_loadu_si64(src_ptr);
                    let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
                    store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight01));
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                    let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                    store = sse_convolve_u8::convolve_horizontal_parts_one_rgba_sse(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                    );
                    jx += 1;
                }

                let store_16_8 = sse_convolve_u8::compress_i32(store);
                let pixel = unsafe { _mm_extract_epi32::<0>(store_16_8) };

                let px = x * CHANNELS;
                let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
                let dest_ptr_32 = dest_ptr as *mut i32;
                dest_ptr_32.write_unaligned(pixel);

                filter_offset += approx_weights.aligned_size;
            }
        }
    }

    pub fn convolve_horizontal_rgb_sse_rows_4(
        src_width: usize,
        dst_width: usize,
        approx_weights: &FilterWeights<i16>,
        unsafe_source_ptr_0: *const u8,
        src_stride: usize,
        unsafe_destination_ptr_0: *mut u8,
        dst_stride: usize,
    ) {
        unsafe {
            const CHANNES: usize = 3;
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

            let vld = unsafe { _mm_set1_epi32(ROUNDING_APPROX) };

            for x in 0..dst_width {
                let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
                let mut jx = 0usize;
                let mut store_0 = vld;
                let mut store_1 = vld;
                let mut store_2 = vld;
                let mut store_3 = vld;

                // Will make step in 4 items however since it is RGB it is necessary to make a safe offset
                while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                    let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                    unsafe {
                        let weight01 = _mm_set1_epi32((ptr as *const i32).read_unaligned());
                        let weight23 = _mm_set1_epi32((ptr.add(2) as *const i32).read_unaligned());
                        let bounds_start = bounds.start + jx;

                        let src_ptr_0 = unsafe_source_ptr_0.add(bounds_start * CHANNES);

                        let rgb_pixel = _mm_loadu_si128(src_ptr_0 as *const __m128i);
                        let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                        let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo, weight01));
                        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(hi, weight23));

                        let src_ptr = src_ptr_0.add(src_stride);
                        let rgb_pixel = _mm_loadu_si128(src_ptr as *const __m128i);
                        let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                        let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                        store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo, weight01));
                        store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(hi, weight23));

                        let src_ptr = src_ptr_0.add(src_stride * 2);
                        let rgb_pixel = _mm_loadu_si128(src_ptr as *const __m128i);
                        let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                        let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                        store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo, weight01));
                        store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(hi, weight23));

                        let src_ptr = src_ptr_0.add(src_stride * 3);
                        let rgb_pixel = _mm_loadu_si128(src_ptr as *const __m128i);
                        let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                        let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                        store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo, weight01));
                        store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(hi, weight23));
                    }
                    jx += 4;
                }

                while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
                    let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                    unsafe {
                        let bounds_start = bounds.start + jx;
                        let weight01 = _mm_set1_epi32((ptr as *const i32).read_unaligned());
                        store_0 = sse_convolve_u8::convolve_horizontal_parts_two_sse_rgb(
                            bounds_start,
                            unsafe_source_ptr_0,
                            weight01,
                            store_0,
                            shuffle_lo,
                        );
                        store_1 = sse_convolve_u8::convolve_horizontal_parts_two_sse_rgb(
                            bounds_start,
                            unsafe_source_ptr_0.add(src_stride),
                            weight01,
                            store_1,
                            shuffle_lo,
                        );
                        store_2 = sse_convolve_u8::convolve_horizontal_parts_two_sse_rgb(
                            bounds_start,
                            unsafe_source_ptr_0.add(src_stride * 2),
                            weight01,
                            store_2,
                            shuffle_lo,
                        );
                        store_3 = sse_convolve_u8::convolve_horizontal_parts_two_sse_rgb(
                            bounds_start,
                            unsafe_source_ptr_0.add(src_stride * 3),
                            weight01,
                            store_3,
                            shuffle_lo,
                        );
                    }
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                    let bounds_start = bounds.start + jx;
                    unsafe {
                        let weight0 = _mm_set1_epi32(ptr.read_unaligned() as i32);
                        store_0 = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                            bounds_start,
                            unsafe_source_ptr_0,
                            weight0,
                            store_0,
                        );
                        store_1 = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                            bounds_start,
                            unsafe_source_ptr_0.add(src_stride),
                            weight0,
                            store_1,
                        );
                        store_2 = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                            bounds_start,
                            unsafe_source_ptr_0.add(src_stride * 2),
                            weight0,
                            store_2,
                        );
                        store_3 = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                            bounds_start,
                            unsafe_source_ptr_0.add(src_stride * 3),
                            weight0,
                            store_3,
                        );
                    }
                    jx += 1;
                }
                let store_0_8 = sse_convolve_u8::compress_i32(store_0);

                let px = x * CHANNES;
                let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

                let element = unsafe { _mm_extract_epi32::<0>(store_0_8) };
                let bytes = element.to_le_bytes();
                unsafe {
                    dest_ptr.write_unaligned(bytes[0]);
                    dest_ptr.add(1).write_unaligned(bytes[1]);
                    dest_ptr.add(2).write_unaligned(bytes[2]);
                }

                let store_1_8 = sse_convolve_u8::compress_i32(store_1);

                let px = x * CHANNES;
                let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride) };

                let element = unsafe { _mm_extract_epi32::<0>(store_1_8) };
                let bytes = element.to_le_bytes();
                unsafe {
                    dest_ptr.write_unaligned(bytes[0]);
                    dest_ptr.add(1).write_unaligned(bytes[1]);
                    dest_ptr.add(2).write_unaligned(bytes[2]);
                }

                let store_2_8 = sse_convolve_u8::compress_i32(store_2);

                let px = x * CHANNES;
                let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 2) };

                let element = unsafe { _mm_extract_epi32::<0>(store_2_8) };
                let bytes = element.to_le_bytes();
                unsafe {
                    dest_ptr.write_unaligned(bytes[0]);
                    dest_ptr.add(1).write_unaligned(bytes[1]);
                    dest_ptr.add(2).write_unaligned(bytes[2]);
                }

                let store_3_8 = sse_convolve_u8::compress_i32(store_3);

                let px = x * CHANNES;
                let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 3) };

                let element = unsafe { _mm_extract_epi32::<0>(store_3_8) };
                let bytes = element.to_le_bytes();
                unsafe {
                    dest_ptr.write_unaligned(bytes[0]);
                    dest_ptr.add(1).write_unaligned(bytes[1]);
                    dest_ptr.add(2).write_unaligned(bytes[2]);
                }

                filter_offset += approx_weights.aligned_size;
            }
        }
    }

    pub fn convolve_horizontal_rgb_sse_row_one(
        src_width: usize,
        dst_width: usize,
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

            while jx + 4 < bounds.size && x + 6 < src_width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                unsafe {
                    let weight01 = _mm_set1_epi32((ptr as *const i32).read_unaligned());
                    let weight23 = _mm_set1_epi32((ptr.add(2) as *const i32).read_unaligned());
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

            while jx + 2 < bounds.size && x + 3 < src_width {
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
                    store = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                    );
                }
                jx += 1;
            }

            let store_16_8 = sse_convolve_u8::compress_i32(store);

            let px = x * CHANNELS;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

            let element = unsafe { _mm_extract_epi32::<0>(store_16_8) };
            let bytes = element.to_le_bytes();
            unsafe {
                dest_ptr.write_unaligned(bytes[0]);
                dest_ptr.add(1).write_unaligned(bytes[1]);
                dest_ptr.add(2).write_unaligned(bytes[2]);
            }
            unsafe {
                dest_ptr.write_unaligned(bytes[0]);
                dest_ptr.add(1).write_unaligned(bytes[1]);
                dest_ptr.add(2).write_unaligned(bytes[2]);
            }

            filter_offset += approx_weights.aligned_size;
        }
    }

    #[inline]
    pub(crate) fn convolve_vertical_rgb_sse_row(
        total_width: usize,
        bounds: &FilterBounds,
        unsafe_source_ptr_0: *const u8,
        unsafe_destination_ptr_0: *mut u8,
        src_stride: usize,
        weight_ptr: *const i16,
    ) {
        let mut cx = 0usize;

        while cx + 32 < total_width {
            unsafe {
                sse_convolve_u8::convolve_vertical_part_sse_32(
                    bounds.start,
                    cx,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    weight_ptr,
                    bounds,
                );
            }

            cx += 32;
        }

        while cx + 16 < total_width {
            unsafe {
                sse_convolve_u8::convolve_vertical_part_sse_16(
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

        while cx + 8 < total_width {
            unsafe {
                sse_convolve_u8::convolve_vertical_part_sse_8::<false>(
                    bounds.start,
                    cx,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    weight_ptr,
                    bounds,
                    8,
                );
            }

            cx += 8;
        }

        let left = total_width - cx;
        if left > 0 {
            unsafe {
                sse_convolve_u8::convolve_vertical_part_sse_8::<true>(
                    bounds.start,
                    cx,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    weight_ptr,
                    bounds,
                    left,
                );
            }
        }
    }
}
