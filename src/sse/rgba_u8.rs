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
use crate::sse::{compress_i32, shuffle};
use crate::support::ROUNDING_CONST;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
unsafe fn convolve_horizontal_parts_one_rgba_sse(
    start_x: usize,
    src: &[u8],
    weight0: __m128i,
    store_0: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let src_ptr_32 = src_ptr.as_ptr() as *const i32;
    let rgba_pixel = _mm_cvtsi32_si128(src_ptr_32.read_unaligned());
    let lo = _mm_unpacklo_epi8(rgba_pixel, _mm_setzero_si128());

    _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(lo), weight0))
}

pub(crate) fn convolve_horizontal_rgba_sse_rows_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_sse_rows_4_impl(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgba_sse_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CHANNELS: usize = 4;

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

        let vld = _mm_set1_epi32(ROUNDING_CONST);

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
        let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
        let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
        let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

        for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
            .zip(iter_row1)
            .zip(iter_row2)
            .zip(iter_row3)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut jx = 0usize;
            let mut store_0 = vld;
            let mut store_1 = vld;
            let mut store_2 = vld;
            let mut store_3 = vld;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = _mm_loadu_si64(w_ptr.as_ptr() as *const u8);
                const SHUFFLE_01: i32 = shuffle(0, 0, 0, 0);
                let weight01 = _mm_shuffle_epi32::<SHUFFLE_01>(weights);
                const SHUFFLE_23: i32 = shuffle(1, 1, 1, 1);
                let weight23 = _mm_shuffle_epi32::<SHUFFLE_23>(weights);
                let start_bounds = bounds.start + jx;

                let rgb_pixel_0 = _mm_loadu_si128(
                    src0.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
                );
                let rgb_pixel_1 = _mm_loadu_si128(
                    src1.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
                );
                let rgb_pixel_2 = _mm_loadu_si128(
                    src2.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
                );
                let rgb_pixel_3 = _mm_loadu_si128(
                    src3.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
                );

                let hi_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_hi);
                let lo_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_lo);
                let hi_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_hi);
                let lo_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_lo);
                let hi_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_hi);
                let lo_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_lo);
                let hi_3 = _mm_shuffle_epi8(rgb_pixel_3, shuffle_hi);
                let lo_3 = _mm_shuffle_epi8(rgb_pixel_3, shuffle_lo);

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

            while jx + 2 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bounds_start = bounds.start + jx;

                let weight01 = _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());

                let rgb_pixel_0 =
                    _mm_loadu_si64(src0.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
                let rgb_pixel_1 =
                    _mm_loadu_si64(src1.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
                let rgb_pixel_2 =
                    _mm_loadu_si64(src2.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
                let rgb_pixel_3 =
                    _mm_loadu_si64(src3.get_unchecked((bounds_start * CHANNELS)..).as_ptr());

                let lo_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_lo);
                let lo_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_lo);
                let lo_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_lo);
                let lo_3 = _mm_shuffle_epi8(rgb_pixel_3, shuffle_lo);

                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo_0, weight01));
                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo_1, weight01));
                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo_2, weight01));
                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo_3, weight01));

                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));

                let weight0 = _mm_set1_epi32(w_ptr[0] as i32);

                let start_bounds = bounds.start + jx;

                store_0 =
                    convolve_horizontal_parts_one_rgba_sse(start_bounds, src0, weight0, store_0);
                store_1 =
                    convolve_horizontal_parts_one_rgba_sse(start_bounds, src1, weight0, store_1);
                store_2 =
                    convolve_horizontal_parts_one_rgba_sse(start_bounds, src2, weight0, store_2);
                store_3 =
                    convolve_horizontal_parts_one_rgba_sse(start_bounds, src3, weight0, store_3);
                jx += 1;
            }

            let store_16_8_0 = compress_i32(store_0);
            let store_16_8_1 = compress_i32(store_1);
            let store_16_8_2 = compress_i32(store_2);
            let store_16_8_3 = compress_i32(store_3);

            _mm_storeu_si32(
                chunk0.as_mut_ptr() as *mut _,
                _mm_packus_epi16(store_16_8_0, store_16_8_0),
            );
            _mm_storeu_si32(
                chunk1.as_mut_ptr() as *mut _,
                _mm_packus_epi16(store_16_8_1, store_16_8_1),
            );
            _mm_storeu_si32(
                chunk2.as_mut_ptr() as *mut _,
                _mm_packus_epi16(store_16_8_2, store_16_8_2),
            );
            _mm_storeu_si32(
                chunk3.as_mut_ptr() as *mut _,
                _mm_packus_epi16(store_16_8_3, store_16_8_3),
            );
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_sse_rows_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_sse_rows_one_impl(src, dst, filter_weights);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgba_sse_rows_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 4;

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

    let vld = _mm_set1_epi32(ROUNDING_CONST);

    for ((dst, bounds), weights) in dst
        .chunks_exact_mut(CHANNELS)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;
        let mut store = vld;

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let bounds_start = bounds.start + jx;

            let weights = _mm_loadu_si64(w_ptr.as_ptr() as *const u8);
            const SHUFFLE_01: i32 = shuffle(0, 0, 0, 0);
            let weight01 = _mm_shuffle_epi32::<SHUFFLE_01>(weights);
            const SHUFFLE_23: i32 = shuffle(1, 1, 1, 1);
            let weight23 = _mm_shuffle_epi32::<SHUFFLE_23>(weights);

            let src_ptr = src.get_unchecked((bounds_start * CHANNELS)..);

            let rgb_pixel = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);

            let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
            let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

            store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight01));
            store = _mm_add_epi32(store, _mm_madd_epi16(hi, weight23));
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;

            let weight01 = _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());

            let src_ptr = src.get_unchecked((bounds_start * CHANNELS)..);

            let rgb_pixel = _mm_loadu_si64(src_ptr.as_ptr());
            let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
            store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight01));

            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weight0 = _mm_set1_epi32(w_ptr.as_ptr().read_unaligned() as i32);
            store = convolve_horizontal_parts_one_rgba_sse(bounds.start + jx, src, weight0, store);
            jx += 1;
        }

        let store_16_8 = compress_i32(store);
        _mm_storeu_si32(
            dst.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8, store_16_8),
        );
    }
}
