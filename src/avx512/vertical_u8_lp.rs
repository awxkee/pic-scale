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
use crate::filter_weights::FilterBounds;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn convolve_vertical_avx512_row_lp(
    dst_width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i16],
) {
    unsafe {
        convolve_vertical_avx512_row_masked_impl(dst_width, bounds, src, dst, src_stride, weights);
    }
}

#[inline(always)]
unsafe fn m512dot(
    store0: __m512i,
    store1: __m512i,
    row: __m512i,
    weight: __m512i,
) -> (__m512i, __m512i) {
    let lo = _mm512_unpacklo_epi8(row, row);
    let hi = _mm512_unpackhi_epi8(row, row);

    let lwo = _mm512_srli_epi16::<2>(lo);
    let hwo = _mm512_srli_epi16::<2>(hi);

    let lli = _mm512_mulhrs_epi16(lwo, weight);
    let lhi = _mm512_mulhrs_epi16(hwo, weight);

    let store0 = _mm512_add_epi16(store0, lli);
    let store1 = _mm512_add_epi16(store1, lhi);
    (store0, store1)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// This inlining is required to activate all features for runtime dispatch
/// Protected loads with masking if available
unsafe fn convolve_vertical_avx512_row_masked_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    let bounds_size = bounds.size;
    const SCALE: u32 = 6;
    const R_SHR_SCALE: i32 = SCALE as i32;
    const AR_SHR_SCALE: u32 = SCALE;
    const ROUNDING: i16 = 1 << (R_SHR_SCALE - 1);

    let mut cx = 0usize;

    let iter_64 = dst.chunks_mut(64);

    for dst in iter_64 {
        let mut store0 = _mm512_set1_epi16(ROUNDING);
        let mut store1 = _mm512_set1_epi16(ROUNDING);

        let working_mask: __mmask64 = if dst.len() == 64 {
            0xffff_ffff_ffff_ffff
        } else {
            0xffff_ffff_ffff_ffff >> (64 - dst.len())
        };

        let px = cx;

        if bounds_size == 2 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..2);
            let v_weight0 = _mm512_set1_epi16(weights[0]);
            let v_weight1 = _mm512_set1_epi16(weights[1]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);

            let item_row0 = _mm512_maskz_loadu_epi8(working_mask, src_ptr0.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row0, v_weight0);

            let item_row1 = _mm512_maskz_loadu_epi8(working_mask, src_ptr1.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row1, v_weight1);
        } else if bounds_size == 3 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..3);
            let v_weight0 = _mm512_set1_epi16(weights[0]);
            let v_weight1 = _mm512_set1_epi16(weights[1]);
            let v_weight2 = _mm512_set1_epi16(weights[2]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..);

            let item_row0 = _mm512_maskz_loadu_epi8(working_mask, src_ptr0.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row0, v_weight0);

            let item_row1 = _mm512_maskz_loadu_epi8(working_mask, src_ptr1.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row1, v_weight1);

            let item_row2 = _mm512_maskz_loadu_epi8(working_mask, src_ptr2.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row2, v_weight2);
        } else if bounds_size == 4 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..4);
            let v_weight0 = _mm512_set1_epi16(weights[0]);
            let v_weight1 = _mm512_set1_epi16(weights[1]);
            let v_weight2 = _mm512_set1_epi16(weights[2]);
            let v_weight3 = _mm512_set1_epi16(weights[3]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..);
            let v_offset3 = src_stride * (py + 3) + px;
            let src_ptr3 = src.get_unchecked(v_offset3..);

            let item_row0 = _mm512_maskz_loadu_epi8(working_mask, src_ptr0.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row0, v_weight0);

            let item_row1 = _mm512_maskz_loadu_epi8(working_mask, src_ptr1.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row1, v_weight1);

            let item_row2 = _mm512_maskz_loadu_epi8(working_mask, src_ptr2.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row2, v_weight2);

            let item_row3 = _mm512_maskz_loadu_epi8(working_mask, src_ptr3.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row3, v_weight3);
        } else if bounds_size == 5 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..5);
            let v_weight0 = _mm512_set1_epi16(weights[0]);
            let v_weight1 = _mm512_set1_epi16(weights[1]);
            let v_weight2 = _mm512_set1_epi16(weights[2]);
            let v_weight3 = _mm512_set1_epi16(weights[3]);
            let v_weight4 = _mm512_set1_epi16(weights[4]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..);
            let v_offset3 = src_stride * (py + 3) + px;
            let src_ptr3 = src.get_unchecked(v_offset3..);
            let v_offset4 = src_stride * (py + 4) + px;
            let src_ptr4 = src.get_unchecked(v_offset4..);

            let item_row0 = _mm512_maskz_loadu_epi8(working_mask, src_ptr0.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row0, v_weight0);

            let item_row1 = _mm512_maskz_loadu_epi8(working_mask, src_ptr1.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row1, v_weight1);

            let item_row2 = _mm512_maskz_loadu_epi8(working_mask, src_ptr2.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row2, v_weight2);

            let item_row3 = _mm512_maskz_loadu_epi8(working_mask, src_ptr3.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row3, v_weight3);

            let item_row4 = _mm512_maskz_loadu_epi8(working_mask, src_ptr4.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row4, v_weight4);
        } else if bounds_size == 6 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..6);
            let v_weight0 = _mm512_set1_epi16(weights[0]);
            let v_weight1 = _mm512_set1_epi16(weights[1]);
            let v_weight2 = _mm512_set1_epi16(weights[2]);
            let v_weight3 = _mm512_set1_epi16(weights[3]);
            let v_weight4 = _mm512_set1_epi16(weights[4]);
            let v_weight5 = _mm512_set1_epi16(weights[5]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..);
            let v_offset3 = src_stride * (py + 3) + px;
            let src_ptr3 = src.get_unchecked(v_offset3..);
            let v_offset4 = src_stride * (py + 4) + px;
            let src_ptr4 = src.get_unchecked(v_offset4..);
            let v_offset5 = src_stride * (py + 5) + px;
            let src_ptr5 = src.get_unchecked(v_offset5..);

            let item_row0 = _mm512_maskz_loadu_epi8(working_mask, src_ptr0.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row0, v_weight0);

            let item_row1 = _mm512_maskz_loadu_epi8(working_mask, src_ptr1.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row1, v_weight1);

            let item_row2 = _mm512_maskz_loadu_epi8(working_mask, src_ptr2.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row2, v_weight2);

            let item_row3 = _mm512_maskz_loadu_epi8(working_mask, src_ptr3.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row3, v_weight3);

            let item_row4 = _mm512_maskz_loadu_epi8(working_mask, src_ptr4.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row4, v_weight4);

            let item_row5 = _mm512_maskz_loadu_epi8(working_mask, src_ptr5.as_ptr() as *const _);
            (store0, store1) = m512dot(store0, store1, item_row5, v_weight5);
        } else {
            for j in 0..bounds_size {
                let py = bounds.start + j;
                let weight = weight.get_unchecked(j);
                let v_weight = _mm512_set1_epi16(*weight);
                let v_offset = src_stride * py + px;
                let src_ptr = src.get_unchecked(v_offset..);
                let item_row0 = _mm512_maskz_loadu_epi8(working_mask, src_ptr.as_ptr() as *const _);

                (store0, store1) = m512dot(store0, store1, item_row0, v_weight);
            }
        }

        let rebased0 = _mm512_srai_epi16::<AR_SHR_SCALE>(store0);
        let rebased1 = _mm512_srai_epi16::<AR_SHR_SCALE>(store1);

        let shrank0 = _mm512_packus_epi16(rebased0, rebased1);

        _mm512_mask_storeu_epi8(dst.as_mut_ptr() as *mut _, working_mask, shrank0);

        cx += 64;
    }
}
