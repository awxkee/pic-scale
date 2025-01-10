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
use crate::sse::shuffle;

pub(crate) fn convolve_horizontal_cbcr_sse_hrs_rows_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_cbcr_sse_hrs_rows_4_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[inline(always)]
unsafe fn conv_cb_cr_1(start_x: usize, src: &[u8], weight0: __m128i, store_0: __m128i) -> __m128i {
    const COMPONENTS: usize = 2;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..).as_ptr();
    let vl = _mm_loadu_si16(src_ptr);
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(vl, vl));
    _mm_add_epi16(store_0, _mm_mulhrs_epi16(lo, weight0))
}

#[inline(always)]
unsafe fn conv_cb_cr_2(start_x: usize, src: &[u8], weight0: __m128i, store_0: __m128i) -> __m128i {
    const COMPONENTS: usize = 2;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..).as_ptr();
    let vl = _mm_loadu_si32(src_ptr as *const _);
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(vl, vl));
    _mm_add_epi16(store_0, _mm_mulhrs_epi16(lo, weight0))
}

#[inline(always)]
unsafe fn conv_cb_cr_4(start_x: usize, src: &[u8], weight0: __m128i, store_0: __m128i) -> __m128i {
    const COMPONENTS: usize = 2;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..).as_ptr();
    let vl = _mm_loadu_si64(src_ptr as *const _);
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(vl, vl));
    _mm_add_epi16(store_0, _mm_mulhrs_epi16(lo, weight0))
}

#[inline(always)]
unsafe fn conv_cb_cr_8(
    start_x: usize,
    src: &[u8],
    w0: __m128i,
    w1: __m128i,
    store_0: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 2;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..).as_ptr();
    let values = _mm_loadu_si128(src_ptr as *const _);
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(values, values));
    let hi = _mm_srli_epi16::<2>(_mm_unpackhi_epi8(values, values));
    let p = _mm_add_epi16(store_0, _mm_mulhrs_epi16(lo, w0));
    _mm_add_epi16(p, _mm_mulhrs_epi16(hi, w1))
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_cbcr_sse_hrs_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 2;
    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i16 = 1 << (PRECISION - 1);

    let weights_lo = _mm_setr_epi8(0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7);
    let weights_hi = _mm_setr_epi8(8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 14, 15);

    let vld = _mm_setr_epi16(ROUNDING_CONST, ROUNDING_CONST, 0, 0, 0, 0, 0, 0);

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
            let bounds_start = bounds.start + jx;

            let weights = _mm_loadu_si128(w_ptr.as_ptr() as *const _);
            let w0 = _mm_shuffle_epi8(weights, weights_lo);
            let w1 = _mm_shuffle_epi8(weights, weights_hi);

            store_0 = conv_cb_cr_8(bounds_start, src0, w0, w1, store_0);
            store_1 = conv_cb_cr_8(bounds_start, src1, w0, w1, store_1);
            store_2 = conv_cb_cr_8(bounds_start, src2, w0, w1, store_2);
            store_3 = conv_cb_cr_8(bounds_start, src3, w0, w1, store_3);

            jx += 8;
        }

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let bounds_start = bounds.start + jx;

            let weight0 = _mm_shuffle_epi8(_mm_loadu_si64(w_ptr.as_ptr() as *const _), weights_lo);

            store_0 = conv_cb_cr_4(bounds_start, src0, weight0, store_0);
            store_1 = conv_cb_cr_4(bounds_start, src1, weight0, store_1);
            store_2 = conv_cb_cr_4(bounds_start, src2, weight0, store_2);
            store_3 = conv_cb_cr_4(bounds_start, src3, weight0, store_3);

            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;

            let weight0 = _mm_shuffle_epi8(_mm_loadu_si32(w_ptr.as_ptr() as *const _), weights_lo);

            store_0 = conv_cb_cr_2(bounds_start, src0, weight0, store_0);
            store_1 = conv_cb_cr_2(bounds_start, src1, weight0, store_1);
            store_2 = conv_cb_cr_2(bounds_start, src2, weight0, store_2);
            store_3 = conv_cb_cr_2(bounds_start, src3, weight0, store_3);

            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let bounds_start = bounds.start + jx;

            let weight0 = _mm_shuffle_epi8(_mm_loadu_si16(w_ptr.as_ptr() as *const _), weights_lo);

            store_0 = conv_cb_cr_1(bounds_start, src0, weight0, store_0);
            store_1 = conv_cb_cr_1(bounds_start, src1, weight0, store_1);
            store_2 = conv_cb_cr_1(bounds_start, src2, weight0, store_2);
            store_3 = conv_cb_cr_1(bounds_start, src3, weight0, store_3);
            jx += 1;
        }

        let element_0 = _mm_reduce_epi16_x2::<PRECISION>(store_0);
        let element_1 = _mm_reduce_epi16_x2::<PRECISION>(store_1);
        let element_2 = _mm_reduce_epi16_x2::<PRECISION>(store_2);
        let element_3 = _mm_reduce_epi16_x2::<PRECISION>(store_3);

        _mm_storeu_si16(chunk0.as_mut_ptr() as *mut _, element_0);
        _mm_storeu_si16(chunk1.as_mut_ptr() as *mut _, element_1);
        _mm_storeu_si16(chunk2.as_mut_ptr() as *mut _, element_2);
        _mm_storeu_si16(chunk3.as_mut_ptr() as *mut _, element_3);
    }
}

pub(crate) fn convolve_horizontal_cbcr_sse_hrs_row_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_cbcr_sse_hrs_row_one_impl(src, dst, filter_weights);
    }
}

#[inline(always)]
unsafe fn _mm_reduce_epi16_x2<const PRECISION: i32>(x: __m128i) -> __m128i {
    // [J0 J2 J1 J3 J4 J6 J5 J7]
    let shuffle0 = _mm_setr_epi8(0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15);
    let j = _mm_shuffle_epi8(x, shuffle0);
    // [J0+J2 J1+J3 J4+J6 J5+J7]x2
    let v0 = _mm_hadd_epi16(j, j);
    const SHUF_1: i32 = shuffle(3, 2, 0, 1);
    let v1 = _mm_shuffle_epi32::<SHUF_1>(v0);
    let v2 = _mm_srai_epi16::<PRECISION>(_mm_add_epi16(v0, v1));
    _mm_packus_epi16(v2, v2)
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_cbcr_sse_hrs_row_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 2;
    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i16 = 1 << (PRECISION - 1);

    let weights_lo = _mm_setr_epi8(0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7);
    let weights_hi = _mm_setr_epi8(8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 14, 15);

    for ((dst, bounds), weights) in dst
        .chunks_exact_mut(CHANNELS)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let bounds_size = bounds.size;
        let mut jx = 0usize;
        let mut store = _mm_setr_epi16(ROUNDING_CONST, ROUNDING_CONST, 0, 0, 0, 0, 0, 0);

        while jx + 8 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 8));
            let weights = _mm_loadu_si128(w_ptr.as_ptr() as *const _);
            let w0 = _mm_shuffle_epi8(weights, weights_lo);
            let w1 = _mm_shuffle_epi8(weights, weights_hi);
            store = conv_cb_cr_8(bounds.start + jx, src, w0, w1, store);
            jx += 8;
        }

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let weight0 = _mm_shuffle_epi8(_mm_loadu_si64(w_ptr.as_ptr() as *const _), weights_lo);
            store = conv_cb_cr_4(bounds.start + jx, src, weight0, store);
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let weight0 = _mm_shuffle_epi8(_mm_loadu_si32(w_ptr.as_ptr() as *const _), weights_lo);
            store = conv_cb_cr_2(bounds.start + jx, src, weight0, store);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weight0 = _mm_shuffle_epi8(_mm_loadu_si16(w_ptr.as_ptr() as *const _), weights_lo);
            store = conv_cb_cr_1(bounds.start + jx, src, weight0, store);
            jx += 1;
        }

        let values = _mm_reduce_epi16_x2::<PRECISION>(store);
        _mm_storeu_si16(dst.as_mut_ptr() as *mut _, values);
    }
}
