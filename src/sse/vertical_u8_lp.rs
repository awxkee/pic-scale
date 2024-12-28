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

pub(crate) fn convolve_vertical_sse_row_lp(
    dst_width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i16],
) {
    unsafe {
        convolve_vertical_sse_row_impl(dst_width, bounds, src, dst, src_stride, weights);
    }
}

#[inline(always)]
unsafe fn mdot<const SCALE: i32>(
    store0: __m128i,
    store1: __m128i,
    row: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let lo = _mm_unpacklo_epi8(row, zeros);
    let hi = _mm_unpackhi_epi8(row, zeros);

    let store0 = _mm_add_epi16(
        store0,
        _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(lo), weight),
    );
    let store1 = _mm_add_epi16(
        store1,
        _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(hi), weight),
    );
    (store0, store1)
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_vertical_sse_row_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    let zeros = _mm_setzero_si128();

    let bounds_size = bounds.size;
    const SCALE: i32 = 6;
    const R_SHR_SCALE: i32 = SCALE;
    const ROUNDING: i16 = 1 << (R_SHR_SCALE - 1);

    let mut cx = 0usize;

    let mut rem = dst;
    let iter_64 = rem.chunks_exact_mut(64);

    for dst in iter_64 {
        let mut store0 = _mm_set1_epi16(ROUNDING);
        let mut store1 = _mm_set1_epi16(ROUNDING);
        let mut store2 = _mm_set1_epi16(ROUNDING);
        let mut store3 = _mm_set1_epi16(ROUNDING);
        let mut store4 = _mm_set1_epi16(ROUNDING);
        let mut store5 = _mm_set1_epi16(ROUNDING);
        let mut store6 = _mm_set1_epi16(ROUNDING);
        let mut store7 = _mm_set1_epi16(ROUNDING);

        let px = cx;

        for j in 0..bounds_size {
            let py = bounds.start + j;
            let weight = weight.get_unchecked(j..(j + 1));
            let v_weight = _mm_set1_epi16(weight[0]);
            let v_offset = src_stride * py + px;
            let src_ptr = src.get_unchecked(v_offset..);
            let item_row0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
            let item_row1 = _mm_loadu_si128(src_ptr.get_unchecked(16..).as_ptr() as *const __m128i);
            let item_row2 = _mm_loadu_si128(src_ptr.get_unchecked(32..).as_ptr() as *const __m128i);
            let item_row3 = _mm_loadu_si128(src_ptr.get_unchecked(48..).as_ptr() as *const __m128i);

            (store0, store1) = mdot::<SCALE>(store0, store1, item_row0, v_weight);
            (store2, store3) = mdot::<SCALE>(store2, store3, item_row1, v_weight);
            (store4, store5) = mdot::<SCALE>(store4, store5, item_row2, v_weight);
            (store6, store7) = mdot::<SCALE>(store6, store7, item_row3, v_weight);
        }

        let rebased0 = _mm_srai_epi16::<R_SHR_SCALE>(store0);
        let rebased1 = _mm_srai_epi16::<R_SHR_SCALE>(store1);
        let rebased2 = _mm_srai_epi16::<R_SHR_SCALE>(store2);
        let rebased3 = _mm_srai_epi16::<R_SHR_SCALE>(store3);
        let rebased4 = _mm_srai_epi16::<R_SHR_SCALE>(store4);
        let rebased5 = _mm_srai_epi16::<R_SHR_SCALE>(store5);
        let rebased6 = _mm_srai_epi16::<R_SHR_SCALE>(store6);
        let rebased7 = _mm_srai_epi16::<R_SHR_SCALE>(store7);
        let shrank0 = _mm_packus_epi16(rebased0, rebased1);
        let shrank1 = _mm_packus_epi16(rebased2, rebased3);
        let shrank2 = _mm_packus_epi16(rebased4, rebased5);
        let shrank3 = _mm_packus_epi16(rebased6, rebased7);
        _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, shrank0);
        _mm_storeu_si128(
            dst.get_unchecked_mut(16..).as_mut_ptr() as *mut __m128i,
            shrank1,
        );
        _mm_storeu_si128(
            dst.get_unchecked_mut(32..).as_mut_ptr() as *mut __m128i,
            shrank2,
        );
        _mm_storeu_si128(
            dst.get_unchecked_mut(48..).as_mut_ptr() as *mut __m128i,
            shrank3,
        );

        cx += 64;
    }

    rem = rem.chunks_exact_mut(64).into_remainder();
    let iter_32 = rem.chunks_exact_mut(32);

    for dst in iter_32 {
        let mut store0 = _mm_set1_epi16(ROUNDING);
        let mut store1 = _mm_set1_epi16(ROUNDING);
        let mut store2 = _mm_set1_epi16(ROUNDING);
        let mut store3 = _mm_set1_epi16(ROUNDING);

        let px = cx;

        for j in 0..bounds_size {
            let py = bounds.start + j;
            let weight = weight.get_unchecked(j..(j + 1));
            let v_weight = _mm_set1_epi16(weight[0]);
            let v_offset = src_stride * py + px;
            let src_ptr = src.get_unchecked(v_offset..);
            let item_row0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
            let item_row1 = _mm_loadu_si128(src_ptr.get_unchecked(16..).as_ptr() as *const __m128i);

            (store0, store1) = mdot::<SCALE>(store0, store1, item_row0, v_weight);
            (store2, store3) = mdot::<SCALE>(store2, store3, item_row1, v_weight);
        }

        let rebased0 = _mm_srai_epi16::<R_SHR_SCALE>(store0);
        let rebased1 = _mm_srai_epi16::<R_SHR_SCALE>(store1);
        let rebased2 = _mm_srai_epi16::<R_SHR_SCALE>(store2);
        let rebased3 = _mm_srai_epi16::<R_SHR_SCALE>(store3);
        let shrank0 = _mm_packus_epi16(rebased0, rebased1);
        let shrank1 = _mm_packus_epi16(rebased2, rebased3);
        _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, shrank0);
        _mm_storeu_si128(
            dst.get_unchecked_mut(16..).as_mut_ptr() as *mut __m128i,
            shrank1,
        );

        cx += 32;
    }

    rem = rem.chunks_exact_mut(32).into_remainder();
    let iter_16 = rem.chunks_exact_mut(16);

    for dst in iter_16 {
        let mut store0 = _mm_set1_epi16(ROUNDING);
        let mut store1 = _mm_set1_epi16(ROUNDING);

        let px = cx;

        if bounds_size == 2 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..2);
            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);

            let item_row0 = _mm_loadu_si128(src_ptr0.as_ptr() as *const __m128i);
            (store0, store1) = mdot::<SCALE>(store0, store1, item_row0, v_weight0);
            let item_row1 = _mm_loadu_si128(src_ptr1.as_ptr() as *const __m128i);
            (store0, store1) = mdot::<SCALE>(store0, store1, item_row1, v_weight1);
        } else if bounds_size == 3 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..3);
            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_weight2 = _mm_set1_epi16(weights[2]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..);

            let item_row0 = _mm_loadu_si128(src_ptr0.as_ptr() as *const __m128i);
            (store0, store1) = mdot::<SCALE>(store0, store1, item_row0, v_weight0);
            let item_row1 = _mm_loadu_si128(src_ptr1.as_ptr() as *const __m128i);
            (store0, store1) = mdot::<SCALE>(store0, store1, item_row1, v_weight1);
            let item_row2 = _mm_loadu_si128(src_ptr2.as_ptr() as *const __m128i);
            (store0, store1) = mdot::<SCALE>(store0, store1, item_row2, v_weight2);
        } else if bounds_size == 4 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..4);
            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_weight2 = _mm_set1_epi16(weights[2]);
            let v_weight3 = _mm_set1_epi16(weights[3]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..);
            let v_offset3 = src_stride * (py + 3) + px;
            let src_ptr3 = src.get_unchecked(v_offset3..);

            let item_row0 = _mm_loadu_si128(src_ptr0.as_ptr() as *const __m128i);
            (store0, store1) = mdot::<SCALE>(store0, store1, item_row0, v_weight0);
            let item_row1 = _mm_loadu_si128(src_ptr1.as_ptr() as *const __m128i);
            (store0, store1) = mdot::<SCALE>(store0, store1, item_row1, v_weight1);
            let item_row2 = _mm_loadu_si128(src_ptr2.as_ptr() as *const __m128i);
            (store0, store1) = mdot::<SCALE>(store0, store1, item_row2, v_weight2);
            let item_row3 = _mm_loadu_si128(src_ptr3.as_ptr() as *const __m128i);
            (store0, store1) = mdot::<SCALE>(store0, store1, item_row3, v_weight3);
        } else {
            for j in 0..bounds_size {
                let py = bounds.start + j;
                let weight = weight.get_unchecked(j..(j + 1));
                let v_weight = _mm_set1_epi16(weight[0]);
                let v_offset = src_stride * py + px;
                let src_ptr = src.get_unchecked(v_offset..);
                let item_row = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);

                (store0, store1) = mdot::<SCALE>(store0, store1, item_row, v_weight);
            }
        }

        let rebased0 = _mm_srai_epi16::<R_SHR_SCALE>(store0);
        let rebased1 = _mm_srai_epi16::<R_SHR_SCALE>(store1);
        let shrank = _mm_packus_epi16(rebased0, rebased1);
        _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, shrank);

        cx += 16;
    }

    rem = rem.chunks_exact_mut(16).into_remainder();
    let iter_8 = rem.chunks_exact_mut(8);

    for dst in iter_8 {
        let mut store = _mm_set1_epi16(ROUNDING);

        let px = cx;

        if bounds_size == 2 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..2);
            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);

            let item_row0 = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr0.as_ptr()), zeros);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row0), v_weight0),
            );

            let item_row1 = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr1.as_ptr()), zeros);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row1), v_weight1),
            );
        } else if bounds_size == 3 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..3);
            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_weight2 = _mm_set1_epi16(weights[2]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..);

            let item_row0 = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr0.as_ptr()), zeros);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row0), v_weight0),
            );

            let item_row1 = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr1.as_ptr()), zeros);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row1), v_weight1),
            );

            let item_row2 = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr2.as_ptr()), zeros);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row2), v_weight2),
            );
        } else if bounds_size == 4 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..4);
            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_weight2 = _mm_set1_epi16(weights[2]);
            let v_weight3 = _mm_set1_epi16(weights[3]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..);
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..);
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..);
            let v_offset3 = src_stride * (py + 3) + px;
            let src_ptr3 = src.get_unchecked(v_offset3..);

            let item_row0 = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr0.as_ptr()), zeros);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row0), v_weight0),
            );

            let item_row1 = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr1.as_ptr()), zeros);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row1), v_weight1),
            );

            let item_row2 = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr2.as_ptr()), zeros);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row2), v_weight2),
            );

            let item_row3 = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr3.as_ptr()), zeros);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row3), v_weight3),
            );
        } else {
            for j in 0..bounds_size {
                let py = bounds.start + j;
                let weight = weight.get_unchecked(j..(j + 1));
                let v_weight = _mm_set1_epi16(weight[0]);
                let v_offset = src_stride * py + px;
                let src_ptr = src.get_unchecked(v_offset..);
                let item_row = _mm_unpacklo_epi8(_mm_loadu_si64(src_ptr.as_ptr()), zeros);

                let low = _mm_slli_epi16::<SCALE>(item_row);
                store = _mm_add_epi16(store, _mm_mulhrs_epi16(low, v_weight));
            }
        }

        let rebased = _mm_srai_epi16::<R_SHR_SCALE>(store);
        let shrank = _mm_packus_epi16(rebased, rebased);
        _mm_storeu_si64(dst.as_mut_ptr(), shrank);

        cx += 8;
    }

    rem = rem.chunks_exact_mut(8).into_remainder();
    let iter_1 = rem.iter_mut();

    for dst in iter_1 {
        let mut store = _mm_set1_epi16(ROUNDING);

        let px = cx;

        if bounds_size == 2 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..2);
            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..(v_offset0 + 1));
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..(v_offset1 + 1));

            let item_row0 = _mm_set1_epi16(src_ptr0[0] as i16);

            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row0), v_weight0),
            );

            let item_row1 = _mm_set1_epi16(src_ptr1[0] as i16);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row1), v_weight1),
            );
        } else if bounds_size == 3 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..3);
            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_weight2 = _mm_set1_epi16(weights[2]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..(v_offset0 + 1));
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..(v_offset1 + 1));
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..(v_offset2 + 1));

            let item_row0 = _mm_set1_epi16(src_ptr0[0] as i16);

            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row0), v_weight0),
            );

            let item_row1 = _mm_set1_epi16(src_ptr1[0] as i16);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row1), v_weight1),
            );

            let item_row2 = _mm_set1_epi16(src_ptr2[0] as i16);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row2), v_weight2),
            );
        } else if bounds_size == 4 {
            let py = bounds.start;
            let weights = weight.get_unchecked(0..4);
            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_weight2 = _mm_set1_epi16(weights[2]);
            let v_weight3 = _mm_set1_epi16(weights[3]);
            let v_offset0 = src_stride * py + px;
            let src_ptr0 = src.get_unchecked(v_offset0..(v_offset0 + 1));
            let v_offset1 = src_stride * (py + 1) + px;
            let src_ptr1 = src.get_unchecked(v_offset1..(v_offset1 + 1));
            let v_offset2 = src_stride * (py + 2) + px;
            let src_ptr2 = src.get_unchecked(v_offset2..(v_offset2 + 1));
            let v_offset3 = src_stride * (py + 3) + px;
            let src_ptr3 = src.get_unchecked(v_offset3..(v_offset3 + 1));

            let item_row0 = _mm_set1_epi16(src_ptr0[0] as i16);

            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row0), v_weight0),
            );

            let item_row1 = _mm_set1_epi16(src_ptr1[0] as i16);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row1), v_weight1),
            );

            let item_row2 = _mm_set1_epi16(src_ptr2[0] as i16);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row2), v_weight2),
            );

            let item_row3 = _mm_set1_epi16(src_ptr3[0] as i16);
            store = _mm_add_epi16(
                store,
                _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row3), v_weight3),
            );
        } else {
            for j in 0..bounds_size {
                let py = bounds.start + j;
                let weight = weight.get_unchecked(j..(j + 1));
                let v_weight = _mm_set1_epi16(weight[0]);
                let v_offset = src_stride * py + px;
                let src_ptr = src.get_unchecked(v_offset..(v_offset + 1));
                let item_row = _mm_set1_epi16(src_ptr[0] as i16);

                store = _mm_add_epi16(
                    store,
                    _mm_mulhrs_epi16(_mm_slli_epi16::<SCALE>(item_row), v_weight),
                );
            }
        }

        let rebased = _mm_srai_epi16::<R_SHR_SCALE>(store);
        let value = _mm_extract_epi8::<0>(_mm_packus_epi16(rebased, rebased));
        *dst = value as u8;

        cx += 1;
    }
}
