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

#[inline]
#[target_feature(enable = "sse4.1")]
fn conv_cb_cr_1(start_x: usize, src: &[u8], weight0: __m128i, store_0: __m128i) -> __m128i {
    unsafe {
        const CN: usize = 2;
        let src_ptr = src.get_unchecked((start_x * CN)..).as_ptr();
        let vl = _mm_loadu_si16(src_ptr.cast());
        let lo = _mm_unpacklo_epi8(vl, _mm_setzero_si128());
        let wlo = _mm_unpacklo_epi16(lo, _mm_setzero_si128());
        _mm_add_epi32(store_0, _mm_madd_epi16(wlo, weight0))
    }
}

#[inline(always)]
fn conv_cb_cr_2(
    start_x: usize,
    src: &[u8],
    weight0: __m128i,
    store_0: __m128i,
    sh: __m128i,
) -> __m128i {
    unsafe {
        const CN: usize = 2;
        let src_ptr = src.get_unchecked((start_x * CN)..).as_ptr();
        let vl = _mm_loadu_si32(src_ptr.cast());
        let lo = _mm_shuffle_epi8(vl, sh);
        _mm_add_epi32(store_0, _mm_madd_epi16(lo, weight0))
    }
}

#[inline(always)]
fn conv_cb_cr_4(
    start_x: usize,
    src: &[u8],
    weight0: __m128i,
    store_0: __m128i,
    sh: __m128i,
) -> __m128i {
    unsafe {
        const CN: usize = 2;
        let src_ptr = src.get_unchecked((start_x * CN)..).as_ptr();
        let vl = _mm_loadu_si64(src_ptr.cast());
        let lo = _mm_shuffle_epi8(vl, sh);
        _mm_add_epi32(store_0, _mm_madd_epi16(lo, weight0))
    }
}

#[inline(always)]
fn conv_cb_cr_8(
    store_0: __m128i,
    start_x: usize,
    src: &[u8],
    w0: __m128i,
    w1: __m128i,
    sh_lo: __m128i,
    sh_hi: __m128i,
) -> __m128i {
    unsafe {
        const CN: usize = 2;
        let src_ptr = src.get_unchecked((start_x * CN)..).as_ptr();
        let values = _mm_loadu_si128(src_ptr.cast());
        let lo = _mm_shuffle_epi8(values, sh_lo);
        let hi = _mm_shuffle_epi8(values, sh_hi);
        let p = _mm_add_epi32(store_0, _mm_madd_epi16(lo, w0));
        _mm_add_epi32(p, _mm_madd_epi16(hi, w1))
    }
}

#[inline(always)]
fn reduce_store_cbcr<const PRECISION: i32>(store: __m128i) -> __m128i {
    unsafe {
        let shuf = _mm_shuffle_epi32::<{ shuffle(1, 0, 3, 2) }>(store);
        let summed = _mm_add_epi32(store, shuf);
        let shifted = _mm_srai_epi32::<PRECISION>(summed);
        let narrow16 = _mm_packs_epi32(shifted, shifted);
        _mm_packus_epi16(narrow16, narrow16)
    }
}

pub(crate) fn convolve_horizontal_cbcr_sse_rows_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_cbcr_sse_rows_4_impl(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[target_feature(enable = "sse4.1")]
fn convolve_horizontal_cbcr_sse_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CN: usize = 2;
        const PRECISION: i32 = 15;
        const ROUNDING: i32 = 1 << (PRECISION - 1);

        let idx_shuffle_lo = _mm_setr_epi8(0, -1, 2, -1, 1, -1, 3, -1, 4, -1, 6, -1, 5, -1, 7, -1);
        let idx_shuffle_hi =
            _mm_setr_epi8(8, -1, 10, -1, 9, -1, 11, -1, 12, -1, 14, -1, 13, -1, 15, -1);

        let weights_lo = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7);
        let weights_hi = _mm_setr_epi8(8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15);

        let vld = _mm_setr_epi32(ROUNDING, ROUNDING, 0, 0);

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
        let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
        let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
        let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

        for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
            .iter_mut()
            .zip(iter_row1.iter_mut())
            .zip(iter_row2.iter_mut())
            .zip(iter_row3.iter_mut())
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

            while jx + 8 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..).as_ptr();
                let raw_w = _mm_loadu_si128(w_ptr.cast());
                let w0 = _mm_shuffle_epi8(raw_w, weights_lo);
                let w1 = _mm_shuffle_epi8(raw_w, weights_hi);
                let bounds_start = bounds.start + jx;

                store_0 = conv_cb_cr_8(
                    store_0,
                    bounds_start,
                    src0,
                    w0,
                    w1,
                    idx_shuffle_lo,
                    idx_shuffle_hi,
                );
                store_1 = conv_cb_cr_8(
                    store_1,
                    bounds_start,
                    src1,
                    w0,
                    w1,
                    idx_shuffle_lo,
                    idx_shuffle_hi,
                );
                store_2 = conv_cb_cr_8(
                    store_2,
                    bounds_start,
                    src2,
                    w0,
                    w1,
                    idx_shuffle_lo,
                    idx_shuffle_hi,
                );
                store_3 = conv_cb_cr_8(
                    store_3,
                    bounds_start,
                    src3,
                    w0,
                    w1,
                    idx_shuffle_lo,
                    idx_shuffle_hi,
                );
                jx += 8;
            }

            while jx + 4 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..).as_ptr();
                let w0 = _mm_shuffle_epi8(_mm_loadu_si64(w_ptr.cast()), weights_lo);
                let bounds_start = bounds.start + jx;

                store_0 = conv_cb_cr_4(bounds_start, src0, w0, store_0, idx_shuffle_lo);
                store_1 = conv_cb_cr_4(bounds_start, src1, w0, store_1, idx_shuffle_lo);
                store_2 = conv_cb_cr_4(bounds_start, src2, w0, store_2, idx_shuffle_lo);
                store_3 = conv_cb_cr_4(bounds_start, src3, w0, store_3, idx_shuffle_lo);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..).as_ptr();
                let w0 = _mm_shuffle_epi8(_mm_loadu_si32(w_ptr.cast()), weights_lo);
                let bounds_start = bounds.start + jx;

                store_0 = conv_cb_cr_2(bounds_start, src0, w0, store_0, idx_shuffle_lo);
                store_1 = conv_cb_cr_2(bounds_start, src1, w0, store_1, idx_shuffle_lo);
                store_2 = conv_cb_cr_2(bounds_start, src2, w0, store_2, idx_shuffle_lo);
                store_3 = conv_cb_cr_2(bounds_start, src3, w0, store_3, idx_shuffle_lo);
                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = *weights.get_unchecked(jx);
                let w0 = _mm_set1_epi16(w_ptr);
                let bounds_start = bounds.start + jx;

                store_0 = conv_cb_cr_1(bounds_start, src0, w0, store_0);
                store_1 = conv_cb_cr_1(bounds_start, src1, w0, store_1);
                store_2 = conv_cb_cr_1(bounds_start, src2, w0, store_2);
                store_3 = conv_cb_cr_1(bounds_start, src3, w0, store_3);
                jx += 1;
            }

            _mm_storeu_si16(
                chunk0.as_mut_ptr().cast(),
                reduce_store_cbcr::<PRECISION>(store_0),
            );
            _mm_storeu_si16(
                chunk1.as_mut_ptr().cast(),
                reduce_store_cbcr::<PRECISION>(store_1),
            );
            _mm_storeu_si16(
                chunk2.as_mut_ptr().cast(),
                reduce_store_cbcr::<PRECISION>(store_2),
            );
            _mm_storeu_si16(
                chunk3.as_mut_ptr().cast(),
                reduce_store_cbcr::<PRECISION>(store_3),
            );
        }
    }
}

pub(crate) fn convolve_horizontal_cbcr_sse_row_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_cbcr_sse_row_one_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "sse4.1")]
fn convolve_horizontal_cbcr_sse_row_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CN: usize = 2;
        const PRECISION: i32 = 15;
        const ROUNDING: i32 = 1 << (PRECISION - 1);

        let idx_shuffle_lo = _mm_setr_epi8(0, -1, 2, -1, 1, -1, 3, -1, 4, -1, 6, -1, 5, -1, 7, -1);
        let idx_shuffle_hi =
            _mm_setr_epi8(8, -1, 10, -1, 9, -1, 11, -1, 12, -1, 14, -1, 13, -1, 15, -1);

        let weights_lo = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7);
        let weights_hi = _mm_setr_epi8(8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15);

        for ((dst, &bounds), weights) in dst
            .as_chunks_mut::<CN>()
            .0
            .iter_mut()
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut jx = 0usize;
            let mut store = _mm_setr_epi32(ROUNDING, ROUNDING, 0, 0);

            while jx + 8 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..).as_ptr();
                let raw_w = _mm_loadu_si128(w_ptr.cast());
                let w0 = _mm_shuffle_epi8(raw_w, weights_lo);
                let w1 = _mm_shuffle_epi8(raw_w, weights_hi);
                store = conv_cb_cr_8(
                    store,
                    bounds.start + jx,
                    src,
                    w0,
                    w1,
                    idx_shuffle_lo,
                    idx_shuffle_hi,
                );
                jx += 8;
            }

            while jx + 4 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..).as_ptr();
                let w0 = _mm_shuffle_epi8(_mm_loadu_si64(w_ptr.cast()), weights_lo);
                store = conv_cb_cr_4(bounds.start + jx, src, w0, store, idx_shuffle_lo);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..).as_ptr();
                let w0 = _mm_shuffle_epi8(_mm_loadu_si32(w_ptr.cast()), weights_lo);
                store = conv_cb_cr_2(bounds.start + jx, src, w0, store, idx_shuffle_lo);
                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = *weights.get_unchecked(jx);
                let w0 = _mm_set1_epi16(w_ptr);
                store = conv_cb_cr_1(bounds.start + jx, src, w0, store);
                jx += 1;
            }

            _mm_storeu_si16(
                dst.as_mut_ptr().cast(),
                reduce_store_cbcr::<PRECISION>(store),
            );
        }
    }
}
