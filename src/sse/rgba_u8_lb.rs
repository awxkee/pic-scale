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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
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
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(rgba_pixel, rgba_pixel));

    _mm_add_epi16(store_0, _mm_mulhrs_epi16(lo, weight0))
}

pub(crate) fn convolve_horizontal_rgba_sse_rows_4_lb(
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

#[inline(always)]
unsafe fn hdot4(
    store: __m128i,
    v0: __m128i,
    v1: __m128i,
    w01: __m128i,
    w23: __m128i,
    w45: __m128i,
    w67: __m128i,
) -> __m128i {
    let lo0 = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(v0, v0));
    let hi0 = _mm_srli_epi16::<2>(_mm_unpackhi_epi8(v0, v0));
    let lo1 = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(v1, v1));
    let hi1 = _mm_srli_epi16::<2>(_mm_unpackhi_epi8(v1, v1));
    let mut p = _mm_mulhrs_epi16(lo0, w01);
    p = _mm_add_epi16(p, _mm_mulhrs_epi16(hi0, w23));
    p = _mm_add_epi16(p, _mm_mulhrs_epi16(lo1, w45));
    p = _mm_add_epi16(p, _mm_mulhrs_epi16(hi1, w67));
    _mm_add_epi16(store, p)
}

#[inline(always)]
unsafe fn hdot2(store: __m128i, v: __m128i, w01: __m128i, w23: __m128i) -> __m128i {
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(v, v));
    let hi = _mm_srli_epi16::<2>(_mm_unpackhi_epi8(v, v));
    let mut p = _mm_mulhrs_epi16(lo, w01);
    p = _mm_add_epi16(p, _mm_mulhrs_epi16(hi, w23));
    _mm_add_epi16(store, p)
}

#[inline(always)]
unsafe fn hdot(store: __m128i, v: __m128i, w01: __m128i) -> __m128i {
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(v, v));
    _mm_add_epi16(store, _mm_mulhrs_epi16(lo, w01))
}

#[inline(always)]
unsafe fn _mm_add_hi_lo_epi16(v: __m128i) -> __m128i {
    let p = _mm_unpackhi_epi64(v, v);
    _mm_add_epi16(v, p)
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgba_sse_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 4;

    const SCALE: i32 = 6;
    const V_SHR: i32 = SCALE;
    const ROUNDING: i16 = 1 << (V_SHR - 1);

    let vld = _mm_setr_epi16(ROUNDING, ROUNDING, ROUNDING, ROUNDING, 0, 0, 0, 0);

    let shuffle_weights = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3);

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

        while jx + 8 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 8));

            let weight01 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );
            let weight23 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.get_unchecked(2..).as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let weight45 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.get_unchecked(4..).as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let weight67 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.get_unchecked(6..).as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let start_bounds = bounds.start + jx;

            let rgb_pixel_0 = _mm_loadu_si128(
                src0.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
            );
            let rgb_pixel_0_1 = _mm_loadu_si128(
                src0.get_unchecked((start_bounds * CHANNELS + 16)..)
                    .as_ptr() as *const __m128i,
            );
            let rgb_pixel_1 = _mm_loadu_si128(
                src1.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
            );
            let rgb_pixel_1_0 = _mm_loadu_si128(
                src1.get_unchecked((start_bounds * CHANNELS + 16)..)
                    .as_ptr() as *const __m128i,
            );
            let rgb_pixel_2 = _mm_loadu_si128(
                src2.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
            );
            let rgb_pixel_2_1 = _mm_loadu_si128(
                src2.get_unchecked((start_bounds * CHANNELS + 16)..)
                    .as_ptr() as *const __m128i,
            );
            let rgb_pixel_3 = _mm_loadu_si128(
                src3.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
            );
            let rgb_pixel_3_1 = _mm_loadu_si128(
                src3.get_unchecked((start_bounds * CHANNELS + 16)..)
                    .as_ptr() as *const __m128i,
            );

            store_0 = hdot4(
                store_0,
                rgb_pixel_0,
                rgb_pixel_0_1,
                weight01,
                weight23,
                weight45,
                weight67,
            );
            store_1 = hdot4(
                store_1,
                rgb_pixel_1,
                rgb_pixel_1_0,
                weight01,
                weight23,
                weight45,
                weight67,
            );
            store_2 = hdot4(
                store_2,
                rgb_pixel_2,
                rgb_pixel_2_1,
                weight01,
                weight23,
                weight45,
                weight67,
            );
            store_3 = hdot4(
                store_3,
                rgb_pixel_3,
                rgb_pixel_3_1,
                weight01,
                weight23,
                weight45,
                weight67,
            );

            jx += 8;
        }

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));

            let weight01 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );
            let weight23 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.get_unchecked(2..).as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );
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

            store_0 = hdot2(store_0, rgb_pixel_0, weight01, weight23);
            store_1 = hdot2(store_1, rgb_pixel_1, weight01, weight23);
            store_2 = hdot2(store_2, rgb_pixel_2, weight01, weight23);
            store_3 = hdot2(store_3, rgb_pixel_3, weight01, weight23);

            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;

            let weight01 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let rgb_pixel_0 =
                _mm_loadu_si64(src0.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
            let rgb_pixel_1 =
                _mm_loadu_si64(src1.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
            let rgb_pixel_2 =
                _mm_loadu_si64(src2.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
            let rgb_pixel_3 =
                _mm_loadu_si64(src3.get_unchecked((bounds_start * CHANNELS)..).as_ptr());

            store_0 = hdot(store_0, rgb_pixel_0, weight01);
            store_1 = hdot(store_1, rgb_pixel_1, weight01);
            store_2 = hdot(store_2, rgb_pixel_2, weight01);
            store_3 = hdot(store_3, rgb_pixel_3, weight01);

            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));

            let weight0 = _mm_set1_epi16(w_ptr[0]);

            let start_bounds = bounds.start + jx;

            store_0 = convolve_horizontal_parts_one_rgba_sse(start_bounds, src0, weight0, store_0);
            store_1 = convolve_horizontal_parts_one_rgba_sse(start_bounds, src1, weight0, store_1);
            store_2 = convolve_horizontal_parts_one_rgba_sse(start_bounds, src2, weight0, store_2);
            store_3 = convolve_horizontal_parts_one_rgba_sse(start_bounds, src3, weight0, store_3);
            jx += 1;
        }

        store_0 = _mm_add_hi_lo_epi16(store_0);
        store_1 = _mm_add_hi_lo_epi16(store_1);
        store_2 = _mm_add_hi_lo_epi16(store_2);
        store_3 = _mm_add_hi_lo_epi16(store_3);

        let store_16_8_0 = _mm_srai_epi16::<V_SHR>(store_0);
        let store_16_8_1 = _mm_srai_epi16::<V_SHR>(store_1);
        let store_16_8_2 = _mm_srai_epi16::<V_SHR>(store_2);
        let store_16_8_3 = _mm_srai_epi16::<V_SHR>(store_3);

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

pub(crate) fn convolve_horizontal_rgba_sse_rows_one_lb(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_sse_rows_one_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgba_sse_rows_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 4;

    let shuffle_weights = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3);

    const SCALE: i32 = 6;
    const V_SHR: i32 = SCALE;
    const ROUNDING: i16 = 1 << (V_SHR - 1);

    let vld = _mm_setr_epi16(ROUNDING, ROUNDING, ROUNDING, ROUNDING, 0, 0, 0, 0);

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

        while jx + 8 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 8));

            let weight01 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );
            let weight23 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.get_unchecked(2..).as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let weight45 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.get_unchecked(4..).as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let weight67 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.get_unchecked(6..).as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let start_bounds = bounds.start + jx;

            let rgb_pixel_0 = _mm_loadu_si128(
                src.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
            );
            let rgb_pixel_0_1 = _mm_loadu_si128(
                src.get_unchecked((start_bounds * CHANNELS + 16)..).as_ptr() as *const __m128i,
            );

            store = hdot4(
                store,
                rgb_pixel_0,
                rgb_pixel_0_1,
                weight01,
                weight23,
                weight45,
                weight67,
            );

            jx += 8;
        }

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let bounds_start = bounds.start + jx;

            let weight01 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );
            let weight23 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.get_unchecked(2..).as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let src_ptr = src.get_unchecked((bounds_start * CHANNELS)..);

            let rgb_pixel = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);

            store = hdot2(store, rgb_pixel, weight01, weight23);

            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;

            let weight01 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let src_ptr = src.get_unchecked((bounds_start * CHANNELS)..);

            let rgb_pixel = _mm_loadu_si64(src_ptr.as_ptr());

            store = hdot(store, rgb_pixel, weight01);

            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weight0 = _mm_set1_epi16(w_ptr[0]);

            let start_bounds = bounds.start + jx;

            store = convolve_horizontal_parts_one_rgba_sse(start_bounds, src, weight0, store);
            jx += 1;
        }

        store = _mm_add_hi_lo_epi16(store);

        let store_16_8 = _mm_srai_epi16::<V_SHR>(store);
        _mm_storeu_si32(
            dst.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8, store_16_8),
        );
    }
}
