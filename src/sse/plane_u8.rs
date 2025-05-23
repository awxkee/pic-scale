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
use crate::sse::{_mm_muladd_wide_epi16, shuffle};
use crate::support::{PRECISION, ROUNDING_CONST};

macro_rules! s_accumulate_8_horiz {
    ($store: expr, $ptr: expr, $weights: expr) => {{
        let pixel_colors = _mm_loadu_si64($ptr);
        let px_16 = _mm_unpacklo_epi8(pixel_colors, _mm_setzero_si128());
        let px_lo = _mm_unpacklo_epi16(px_16, _mm_setzero_si128());
        let px_hi = _mm_unpackhi_epi16(px_16, _mm_setzero_si128());

        $store = _mm_muladd_wide_epi16($store, px_lo, $weights.0);
        $store = _mm_muladd_wide_epi16($store, px_hi, $weights.1);
    }};
}

macro_rules! s_accumulate_4_horiz {
    ($store: expr, $ptr: expr, $weights: expr) => {{
        let zeros = _mm_setzero_si128();
        let pixel_colors = _mm_unpacklo_epi16(
            _mm_unpacklo_epi8(_mm_loadu_si32($ptr as *const _), zeros),
            zeros,
        );
        $store = _mm_muladd_wide_epi16($store, pixel_colors, $weights);
    }};
}

macro_rules! s_accumulate_1_horiz {
    ($store: expr, $ptr: expr, $weight: expr) => {{
        let pixel_colors = _mm_setr_epi32($ptr.read_unaligned() as i32, 0, 0, 0);
        $store = _mm_muladd_wide_epi16($store, pixel_colors, $weight);
    }};
}

pub(crate) fn convolve_horizontal_plane_sse_rows_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_plane_sse_rows_4_u8_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[inline(always)]
unsafe fn _mm_hsum_reduce_epi32<const PRECISION: i32>(x: __m128i) -> u8 {
    unsafe {
        const FIRST_MASK: i32 = shuffle(1, 0, 3, 2);
        let hi64 = _mm_shuffle_epi32::<FIRST_MASK>(x);
        let sum64 = _mm_add_epi32(hi64, x);
        const SM: i32 = shuffle(1, 0, 3, 2);
        let hi32 = _mm_shufflelo_epi16::<SM>(sum64);
        let sum32 = _mm_srai_epi32::<PRECISION>(_mm_add_epi32(sum64, hi32));
        let v0 = _mm_packus_epi32(sum32, sum32);
        let v1 = _mm_packus_epi16(v0, v0);
        _mm_extract_epi8::<0>(v1) as u8
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_plane_sse_rows_4_u8_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        let zeros = _mm_setzero_si128();

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
            let mut store0 = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);
            let mut store1 = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);
            let mut store2 = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);
            let mut store3 = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 8 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 8));
                let weights_i16 = _mm_loadu_si128(w_ptr.as_ptr() as *const __m128i);
                let weights = (
                    _mm_unpacklo_epi16(weights_i16, zeros),
                    _mm_unpackhi_epi16(weights_i16, zeros),
                );
                let bounds_start = bounds.start + jx;

                let src_ptr = src0.get_unchecked(bounds_start..);
                s_accumulate_8_horiz!(store0, src_ptr.as_ptr(), weights);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                s_accumulate_8_horiz!(store1, src_ptr1.as_ptr(), weights);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                s_accumulate_8_horiz!(store2, src_ptr2.as_ptr(), weights);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                s_accumulate_8_horiz!(store3, src_ptr3.as_ptr(), weights);

                jx += 8;
            }

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = _mm_cvtepi16_epi32(_mm_loadu_si64(w_ptr.as_ptr() as *const u8));
                let bounds_start = bounds.start + jx;

                let src_ptr = src0.get_unchecked(bounds_start..);
                s_accumulate_4_horiz!(store0, src_ptr.as_ptr(), weights);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                s_accumulate_4_horiz!(store1, src_ptr1.as_ptr(), weights);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                s_accumulate_4_horiz!(store2, src_ptr2.as_ptr(), weights);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                s_accumulate_4_horiz!(store3, src_ptr3.as_ptr(), weights);

                jx += 4;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let weight = _mm_setr_epi32(w_ptr.as_ptr().read_unaligned() as i32, 0, 0, 0);
                let bounds_start = bounds.start + jx;

                let src_ptr = src0.get_unchecked(bounds_start..);
                s_accumulate_1_horiz!(store0, src_ptr.as_ptr(), weight);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                s_accumulate_1_horiz!(store1, src_ptr1.as_ptr(), weight);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                s_accumulate_1_horiz!(store2, src_ptr2.as_ptr(), weight);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                s_accumulate_1_horiz!(store3, src_ptr3.as_ptr(), weight);

                jx += 1;
            }

            let value0 = _mm_hsum_reduce_epi32::<PRECISION>(store0);
            *chunk0 = value0;

            let value1 = _mm_hsum_reduce_epi32::<PRECISION>(store1);
            *chunk1 = value1;

            let value2 = _mm_hsum_reduce_epi32::<PRECISION>(store2);
            *chunk2 = value2;

            let value3 = _mm_hsum_reduce_epi32::<PRECISION>(store3);
            *chunk3 = value3;
        }
    }
}

pub(crate) fn convolve_horizontal_plane_sse_row(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_plane_sse_row_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_plane_sse_row_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        let zeros = _mm_setzero_si128();

        for ((dst, bounds), weights) in dst.iter_mut().zip(filter_weights.bounds.iter()).zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        ) {
            let mut jx = 0usize;
            let mut store = _mm_setr_epi32(ROUNDING_CONST, 0i32, 0i32, 0i32);

            while jx + 8 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 8));
                let weights_i16 = _mm_loadu_si128(w_ptr.as_ptr() as *const __m128i);
                let weights = (
                    _mm_unpacklo_epi16(weights_i16, zeros),
                    _mm_unpackhi_epi16(weights_i16, zeros),
                );
                let bounds_start = bounds.start + jx;

                let src_ptr = src.get_unchecked(bounds_start..);
                s_accumulate_8_horiz!(store, src_ptr.as_ptr(), weights);

                jx += 8;
            }

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = _mm_cvtepi16_epi32(_mm_loadu_si64(w_ptr.as_ptr() as *const u8));
                let bounds_start = bounds.start + jx;

                let src_ptr = src.get_unchecked(bounds_start..);
                s_accumulate_4_horiz!(store, src_ptr.as_ptr(), weights);

                jx += 4;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let weight = _mm_setr_epi32(w_ptr.as_ptr().read_unaligned() as i32, 0, 0, 0);
                let bounds_start = bounds.start + jx;
                let src_ptr = src.get_unchecked(bounds_start..);
                s_accumulate_1_horiz!(store, src_ptr.as_ptr(), weight);
                jx += 1;
            }

            let value = _mm_hsum_reduce_epi32::<PRECISION>(store);
            *dst = value;
        }
    }
}
