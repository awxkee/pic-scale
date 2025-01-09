/*
 * Copyright (c) Radzivon Bartoshyk 01/2025. All rights reserved.
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
use crate::sse::shuffle;

#[must_use]
#[inline(always)]
pub(crate) unsafe fn _mm_hsum_epi16_and_compress<const PRECISION: i32>(x: __m128i) -> u8 {
    // [v0 + v4] [v1 + v5] [v2 + v6] [v3 + v7]
    let v0 = _mm_hadd_epi16(x, x);
    // Shuffle to [v2 + v6] [v3 + v7] [v0 + v4] [v1 + v5]
    const MASK: i32 = shuffle(0, 0, 0, 1);
    let v1 = _mm_shuffle_epi32::<MASK>(v0);
    // [v2 + v6 + v0 + v4] [v3 + v7 + v1 + v5]
    let v2 = _mm_add_epi16(v0, v1);
    let v3 = _mm_srai_epi16::<PRECISION>(_mm_hadd_epi16(v2, v2));
    let v4 = _mm_packus_epi16(v3, v3);
    _mm_extract_epi8::<0>(v4) as u8
}

#[must_use]
#[inline(always)]
unsafe fn s_accumulate_1_horiz(store: __m128i, ptr: *const u8, weight: __m128i) -> __m128i {
    let value = ptr.read_unaligned() as i8;
    let pixel_colors = _mm_srli_epi16::<2>(_mm_setr_epi8(
        value, value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ));
    _mm_adds_epi16(store, _mm_mulhrs_epi16(pixel_colors, weight))
}

#[must_use]
#[inline(always)]
unsafe fn s_accumulate_4_horiz(store: __m128i, ptr: *const u8, weight: __m128i) -> __m128i {
    let px = _mm_loadu_si32(ptr as *const _);
    let pixel_colors = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(px, px));
    _mm_adds_epi16(store, _mm_mulhrs_epi16(pixel_colors, weight))
}

#[must_use]
#[inline(always)]
unsafe fn s_accumulate_8_horiz(store: __m128i, ptr: *const u8, weight: __m128i) -> __m128i {
    let px = _mm_loadu_si64(ptr as *const _);
    let pixel_colors = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(px, px));
    _mm_adds_epi16(store, _mm_mulhrs_epi16(pixel_colors, weight))
}

#[must_use]
#[inline(always)]
unsafe fn s_accumulate_16_horiz(
    store: __m128i,
    ptr: *const u8,
    weight: (__m128i, __m128i),
) -> __m128i {
    let px = _mm_loadu_si128(ptr as *const _);
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(px, px));
    let hi = _mm_srli_epi16::<2>(_mm_unpackhi_epi8(px, px));

    let v0 = _mm_adds_epi16(store, _mm_mulhrs_epi16(lo, weight.0));
    _mm_adds_epi16(v0, _mm_mulhrs_epi16(hi, weight.1))
}

pub(crate) fn convolve_horizontal_plane_sse_rows_hrs_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_plane_sse_rows_hrs_4_u8_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_plane_sse_rows_hrs_4_u8_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i16 = 1 << (PRECISION - 1);

    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.iter_mut();
    let iter_row1 = row1_ref.iter_mut();
    let iter_row2 = row2_ref.iter_mut();
    let iter_row3 = row3_ref.iter_mut();

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
        let mut store0 = _mm_setr_epi16(ROUNDING_CONST, 0, 0, 0, 0, 0, 0, 0);
        let mut store1 = _mm_setr_epi16(ROUNDING_CONST, 0, 0, 0, 0, 0, 0, 0);
        let mut store2 = _mm_setr_epi16(ROUNDING_CONST, 0, 0, 0, 0, 0, 0, 0);
        let mut store3 = _mm_setr_epi16(ROUNDING_CONST, 0, 0, 0, 0, 0, 0, 0);

        let src0 = src;
        let src1 = src0.get_unchecked(src_stride..);
        let src2 = src1.get_unchecked(src_stride..);
        let src3 = src2.get_unchecked(src_stride..);

        while jx + 16 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 16));
            let w0 = _mm_loadu_si128(w_ptr.as_ptr() as *const __m128i);
            let w1 = _mm_loadu_si128(w_ptr.get_unchecked(8..).as_ptr() as *const __m128i);
            let bounds_start = bounds.start + jx;

            let src_ptr = src0.get_unchecked(bounds_start..);
            store0 = s_accumulate_16_horiz(store0, src_ptr.as_ptr(), (w0, w1));

            let src_ptr1 = src1.get_unchecked(bounds_start..);
            store1 = s_accumulate_16_horiz(store1, src_ptr1.as_ptr(), (w0, w1));

            let src_ptr2 = src2.get_unchecked(bounds_start..);
            store2 = s_accumulate_16_horiz(store2, src_ptr2.as_ptr(), (w0, w1));

            let src_ptr3 = src3.get_unchecked(bounds_start..);
            store3 = s_accumulate_16_horiz(store3, src_ptr3.as_ptr(), (w0, w1));

            jx += 16;
        }

        while jx + 8 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 8));
            let weights = _mm_loadu_si128(w_ptr.as_ptr() as *const __m128i);
            let bounds_start = bounds.start + jx;

            let src_ptr = src0.get_unchecked(bounds_start..);
            store0 = s_accumulate_8_horiz(store0, src_ptr.as_ptr(), weights);

            let src_ptr1 = src1.get_unchecked(bounds_start..);
            store1 = s_accumulate_8_horiz(store1, src_ptr1.as_ptr(), weights);

            let src_ptr2 = src2.get_unchecked(bounds_start..);
            store2 = s_accumulate_8_horiz(store2, src_ptr2.as_ptr(), weights);

            let src_ptr3 = src3.get_unchecked(bounds_start..);
            store3 = s_accumulate_8_horiz(store3, src_ptr3.as_ptr(), weights);

            jx += 8;
        }

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let weights = _mm_loadu_si64(w_ptr.as_ptr() as *const u8);
            let bounds_start = bounds.start + jx;

            let src_ptr = src0.get_unchecked(bounds_start..);
            store0 = s_accumulate_4_horiz(store0, src_ptr.as_ptr(), weights);

            let src_ptr1 = src1.get_unchecked(bounds_start..);
            store1 = s_accumulate_4_horiz(store1, src_ptr1.as_ptr(), weights);

            let src_ptr2 = src2.get_unchecked(bounds_start..);
            store2 = s_accumulate_4_horiz(store2, src_ptr2.as_ptr(), weights);

            let src_ptr3 = src3.get_unchecked(bounds_start..);
            store3 = s_accumulate_4_horiz(store3, src_ptr3.as_ptr(), weights);

            jx += 4;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weight = _mm_loadu_si16(w_ptr.as_ptr() as *const u8);
            let bounds_start = bounds.start + jx;

            let src_ptr = src0.get_unchecked(bounds_start..);
            store0 = s_accumulate_1_horiz(store0, src_ptr.as_ptr(), weight);

            let src_ptr1 = src1.get_unchecked(bounds_start..);
            store1 = s_accumulate_1_horiz(store1, src_ptr1.as_ptr(), weight);

            let src_ptr2 = src2.get_unchecked(bounds_start..);
            store2 = s_accumulate_1_horiz(store2, src_ptr2.as_ptr(), weight);

            let src_ptr3 = src3.get_unchecked(bounds_start..);
            store3 = s_accumulate_1_horiz(store3, src_ptr3.as_ptr(), weight);

            jx += 1;
        }

        let value0 = _mm_hsum_epi16_and_compress::<PRECISION>(store0);
        *chunk0 = value0;

        let value1 = _mm_hsum_epi16_and_compress::<PRECISION>(store1);
        *chunk1 = value1;

        let value2 = _mm_hsum_epi16_and_compress::<PRECISION>(store2);
        *chunk2 = value2;

        let value3 = _mm_hsum_epi16_and_compress::<PRECISION>(store3);
        *chunk3 = value3;
    }
}

pub(crate) fn convolve_horizontal_plane_sse_row_hrs(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_plane_sse_row_hrs_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_plane_sse_row_hrs_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i16 = 1 << (PRECISION - 1);

    for ((dst, bounds), weights) in dst.iter_mut().zip(filter_weights.bounds.iter()).zip(
        filter_weights
            .weights
            .chunks_exact(filter_weights.aligned_size),
    ) {
        let mut jx = 0usize;
        let mut store = _mm_setr_epi16(ROUNDING_CONST, 0, 0, 0, 0, 0, 0, 0);

        while jx + 16 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 16));
            let w0 = _mm_loadu_si128(w_ptr.as_ptr() as *const __m128i);
            let w1 = _mm_loadu_si128(w_ptr.get_unchecked(8..).as_ptr() as *const __m128i);

            let bounds_start = bounds.start + jx;

            let src_ptr = src.get_unchecked(bounds_start..);
            store = s_accumulate_16_horiz(store, src_ptr.as_ptr(), (w0, w1));

            jx += 16;
        }

        while jx + 8 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 8));
            let weights = _mm_loadu_si128(w_ptr.as_ptr() as *const __m128i);
            let bounds_start = bounds.start + jx;

            let src_ptr = src.get_unchecked(bounds_start..);
            store = s_accumulate_8_horiz(store, src_ptr.as_ptr(), weights);

            jx += 8;
        }

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let weights = _mm_loadu_si64(w_ptr.as_ptr() as *const u8);
            let bounds_start = bounds.start + jx;

            let src_ptr = src.get_unchecked(bounds_start..);
            store = s_accumulate_4_horiz(store, src_ptr.as_ptr(), weights);

            jx += 4;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weight = _mm_loadu_si16(w_ptr.as_ptr() as *const u8);
            let bounds_start = bounds.start + jx;
            let src_ptr = src.get_unchecked(bounds_start..);
            store = s_accumulate_1_horiz(store, src_ptr.as_ptr(), weight);
            jx += 1;
        }

        let value = _mm_hsum_epi16_and_compress::<PRECISION>(store);
        *dst = value;
    }
}
