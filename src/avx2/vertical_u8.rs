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
use crate::avx2::utils::shuffle;
use crate::filter_weights::FilterBounds;
use crate::support::{PRECISION, ROUNDING_CONST};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn dot_prod(
    store_0: __m256i,
    store_1: __m256i,
    store_2: __m256i,
    store_3: __m256i,
    v: __m256i,
    w: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let zeros = _mm256_setzero_si256();
    let interleaved = _mm256_unpacklo_epi8(v, zeros);
    let pix = _mm256_unpacklo_epi8(interleaved, zeros);
    let store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(pix, w));
    let pix = _mm256_unpackhi_epi8(interleaved, zeros);
    let store_1 = _mm256_add_epi32(store_1, _mm256_madd_epi16(pix, w));

    let interleaved = _mm256_unpackhi_epi8(v, zeros);
    let pix = _mm256_unpacklo_epi8(interleaved, zeros);
    let store_2 = _mm256_add_epi32(store_2, _mm256_madd_epi16(pix, w));
    let pix = _mm256_unpackhi_epi8(interleaved, zeros);
    let store_3 = _mm256_add_epi32(store_3, _mm256_madd_epi16(pix, w));
    (store_0, store_1, store_2, store_3)
}

#[inline(always)]
unsafe fn convolve_vertical_part_avx_64(
    start_y: usize,
    start_x: usize,
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    filter: &[i16],
    bounds: &FilterBounds,
) {
    let zeros = _mm256_setzero_si256();
    let vld = _mm256_set1_epi32(ROUNDING_CONST);
    let mut store_0 = vld;
    let mut store_1 = vld;
    let mut store_2 = vld;
    let mut store_3 = vld;
    let mut store_4 = vld;
    let mut store_5 = vld;
    let mut store_6 = vld;
    let mut store_7 = vld;

    let px = start_x;

    let bounds_size = bounds.size;

    if bounds_size == 2 {
        let py = start_y;
        let weight = filter.get_unchecked(0..2);
        let v_weight0 = _mm256_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm256_set1_epi32(weight[1] as i32);

        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);

        let item_row_0 = _mm256_loadu_si256(src_ptr0.as_ptr() as *const __m256i);
        let item_row_1 =
            _mm256_loadu_si256(src_ptr0.get_unchecked(32..).as_ptr() as *const __m256i);
        let item_row_10 = _mm256_loadu_si256(src_ptr1.as_ptr() as *const __m256i);
        let item_row_11 =
            _mm256_loadu_si256(src_ptr1.get_unchecked(32..).as_ptr() as *const __m256i);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_0, v_weight0);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_1, v_weight0);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_10, v_weight1);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_11, v_weight1);
    } else if bounds_size == 3 {
        let py = start_y;
        let weight = filter.get_unchecked(0..3);
        let v_weight0 = _mm256_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm256_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm256_set1_epi32(weight[2] as i32);

        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);

        let item_row_0 = _mm256_loadu_si256(src_ptr0.as_ptr() as *const __m256i);
        let item_row_1 =
            _mm256_loadu_si256(src_ptr0.get_unchecked(32..).as_ptr() as *const __m256i);

        let item_row_10 = _mm256_loadu_si256(src_ptr1.as_ptr() as *const __m256i);
        let item_row_11 =
            _mm256_loadu_si256(src_ptr1.get_unchecked(32..).as_ptr() as *const __m256i);

        let item_row_20 = _mm256_loadu_si256(src_ptr2.as_ptr() as *const __m256i);
        let item_row_21 =
            _mm256_loadu_si256(src_ptr2.get_unchecked(32..).as_ptr() as *const __m256i);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_0, v_weight0);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_1, v_weight0);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_10, v_weight1);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_11, v_weight1);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_20, v_weight2);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_21, v_weight2);
    } else if bounds_size == 4 {
        let py = start_y;
        let weight = filter.get_unchecked(0..4);
        let v_weight0 = _mm256_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm256_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm256_set1_epi32(weight[2] as i32);
        let v_weight3 = _mm256_set1_epi32(weight[3] as i32);

        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
        let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);

        let item_row_0 = _mm256_loadu_si256(src_ptr0.as_ptr() as *const __m256i);
        let item_row_1 =
            _mm256_loadu_si256(src_ptr0.get_unchecked(32..).as_ptr() as *const __m256i);

        let item_row_10 = _mm256_loadu_si256(src_ptr1.as_ptr() as *const __m256i);
        let item_row_11 =
            _mm256_loadu_si256(src_ptr1.get_unchecked(32..).as_ptr() as *const __m256i);

        let item_row_20 = _mm256_loadu_si256(src_ptr2.as_ptr() as *const __m256i);
        let item_row_21 =
            _mm256_loadu_si256(src_ptr2.get_unchecked(32..).as_ptr() as *const __m256i);

        let item_row_30 = _mm256_loadu_si256(src_ptr3.as_ptr() as *const __m256i);
        let item_row_31 =
            _mm256_loadu_si256(src_ptr3.get_unchecked(32..).as_ptr() as *const __m256i);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_0, v_weight0);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_1, v_weight0);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_10, v_weight1);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_11, v_weight1);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_20, v_weight2);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_21, v_weight2);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_30, v_weight3);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_31, v_weight3);
    } else {
        let mut jj = 0usize;

        while jj < bounds_size.saturating_sub(2) {
            let py = start_y + jj;
            let f_ptr = filter.get_unchecked(jj..).as_ptr() as *const i32;
            let v_weight_2 = _mm256_set1_epi32(f_ptr.read_unaligned());
            let src_ptr = src.get_unchecked((src_stride * py + px)..);
            let s_ptr_next = src_ptr.get_unchecked(src_stride..);

            let item_row_0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);
            let item_row_1 = _mm256_loadu_si256(s_ptr_next.as_ptr() as *const __m256i);

            let interleaved = _mm256_unpacklo_epi8(item_row_0, item_row_1);
            let pix = _mm256_unpacklo_epi8(interleaved, zeros);
            store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(pix, v_weight_2));
            let pix = _mm256_unpackhi_epi8(interleaved, zeros);
            store_1 = _mm256_add_epi32(store_1, _mm256_madd_epi16(pix, v_weight_2));

            let interleaved = _mm256_unpackhi_epi8(item_row_0, item_row_1);
            let pix = _mm256_unpacklo_epi8(interleaved, zeros);
            store_2 = _mm256_add_epi32(store_2, _mm256_madd_epi16(pix, v_weight_2));
            let pix = _mm256_unpackhi_epi8(interleaved, zeros);
            store_3 = _mm256_add_epi32(store_3, _mm256_madd_epi16(pix, v_weight_2));

            let item_row_0 =
                _mm256_loadu_si256(src_ptr.get_unchecked(32..).as_ptr() as *const __m256i);
            let item_row_1 =
                _mm256_loadu_si256(s_ptr_next.get_unchecked(32..).as_ptr() as *const __m256i);

            let interleaved = _mm256_unpacklo_epi8(item_row_0, item_row_1);
            let pix = _mm256_unpacklo_epi8(interleaved, zeros);
            store_4 = _mm256_add_epi32(store_4, _mm256_madd_epi16(pix, v_weight_2));
            let pix = _mm256_unpackhi_epi8(interleaved, zeros);
            store_5 = _mm256_add_epi32(store_5, _mm256_madd_epi16(pix, v_weight_2));

            let interleaved = _mm256_unpackhi_epi8(item_row_0, item_row_1);
            let pix = _mm256_unpacklo_epi8(interleaved, zeros);
            store_6 = _mm256_add_epi32(store_6, _mm256_madd_epi16(pix, v_weight_2));
            let pix = _mm256_unpackhi_epi8(interleaved, zeros);
            store_7 = _mm256_add_epi32(store_7, _mm256_madd_epi16(pix, v_weight_2));

            jj += 2;
        }

        for j in jj..bounds_size {
            let py = start_y + j;
            let weight = *filter.get_unchecked(j);
            let v_weight = _mm256_set1_epi32(weight as i32);
            let src_ptr = src.get_unchecked((src_stride * py + px)..);

            let item_row_0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);
            let item_row_1 =
                _mm256_loadu_si256(src_ptr.get_unchecked(32..).as_ptr() as *const __m256i);

            (store_0, store_1, store_2, store_3) =
                dot_prod(store_0, store_1, store_2, store_3, item_row_0, v_weight);
            (store_4, store_5, store_6, store_7) =
                dot_prod(store_4, store_5, store_6, store_7, item_row_1, v_weight);
        }
    }

    store_0 = _mm256_srai_epi32::<PRECISION>(store_0);
    store_1 = _mm256_srai_epi32::<PRECISION>(store_1);
    store_2 = _mm256_srai_epi32::<PRECISION>(store_2);
    store_3 = _mm256_srai_epi32::<PRECISION>(store_3);
    store_4 = _mm256_srai_epi32::<PRECISION>(store_4);
    store_5 = _mm256_srai_epi32::<PRECISION>(store_5);
    store_6 = _mm256_srai_epi32::<PRECISION>(store_6);
    store_7 = _mm256_srai_epi32::<PRECISION>(store_7);

    let rgb0 = _mm256_packs_epi32(store_0, store_1);
    let rgb2 = _mm256_packs_epi32(store_2, store_3);
    let rgb = _mm256_packus_epi16(rgb0, rgb2);

    let dst_ptr = dst.get_unchecked_mut(px..);
    _mm256_storeu_si256(dst_ptr.as_mut_ptr() as *mut __m256i, rgb);

    let rgb0 = _mm256_packs_epi32(store_4, store_5);
    let rgb2 = _mm256_packs_epi32(store_6, store_7);
    let rgb = _mm256_packus_epi16(rgb0, rgb2);

    let dst_ptr = dst.get_unchecked_mut((px + 32)..);
    _mm256_storeu_si256(dst_ptr.as_mut_ptr() as *mut __m256i, rgb);
}

#[inline(always)]
unsafe fn convolve_vertical_part_avx_32(
    start_y: usize,
    start_x: usize,
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    filter: &[i16],
    bounds: &FilterBounds,
) {
    let vld = _mm256_set1_epi32(ROUNDING_CONST);
    let mut store_0 = vld;
    let mut store_1 = vld;
    let mut store_2 = vld;
    let mut store_3 = vld;

    let px = start_x;

    let bounds_size = bounds.size;

    if bounds_size == 2 {
        let py = start_y;
        let weight = filter.get_unchecked(0..2);
        let v_weight0 = _mm256_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm256_set1_epi32(weight[1] as i32);
        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);

        let item_row0 = _mm256_loadu_si256(src_ptr0.as_ptr() as *const __m256i);
        let item_row1 = _mm256_loadu_si256(src_ptr1.as_ptr() as *const __m256i);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row0, v_weight0);
        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row1, v_weight1);
    } else if bounds_size == 3 {
        let py = start_y;
        let weight = filter.get_unchecked(0..3);
        let v_weight0 = _mm256_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm256_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm256_set1_epi32(weight[2] as i32);
        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);

        let item_row0 = _mm256_loadu_si256(src_ptr0.as_ptr() as *const __m256i);
        let item_row1 = _mm256_loadu_si256(src_ptr1.as_ptr() as *const __m256i);
        let item_row2 = _mm256_loadu_si256(src_ptr2.as_ptr() as *const __m256i);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row0, v_weight0);
        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row1, v_weight1);
        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row2, v_weight2);
    } else if bounds_size == 4 {
        let py = start_y;
        let weight = filter.get_unchecked(0..4);
        let v_weight0 = _mm256_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm256_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm256_set1_epi32(weight[2] as i32);
        let v_weight3 = _mm256_set1_epi32(weight[3] as i32);
        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
        let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);

        let item_row0 = _mm256_loadu_si256(src_ptr0.as_ptr() as *const __m256i);
        let item_row1 = _mm256_loadu_si256(src_ptr1.as_ptr() as *const __m256i);
        let item_row2 = _mm256_loadu_si256(src_ptr2.as_ptr() as *const __m256i);
        let item_row3 = _mm256_loadu_si256(src_ptr3.as_ptr() as *const __m256i);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row0, v_weight0);
        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row1, v_weight1);
        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row2, v_weight2);
        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row3, v_weight3);
    } else {
        for j in 0..bounds_size {
            let py = start_y + j;
            let weight = *filter.get_unchecked(j);
            let v_weight = _mm256_set1_epi32(weight as i32);
            let src_ptr = src.get_unchecked((src_stride * py + px)..);

            let item_row = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);

            (store_0, store_1, store_2, store_3) =
                dot_prod(store_0, store_1, store_2, store_3, item_row, v_weight);
        }
    }

    store_0 = _mm256_srai_epi32::<PRECISION>(store_0);
    store_1 = _mm256_srai_epi32::<PRECISION>(store_1);
    store_2 = _mm256_srai_epi32::<PRECISION>(store_2);
    store_3 = _mm256_srai_epi32::<PRECISION>(store_3);

    let rgb0 = _mm256_packs_epi32(store_0, store_1);
    let rgb2 = _mm256_packs_epi32(store_2, store_3);
    let rgb = _mm256_packus_epi16(rgb0, rgb2);

    let dst_ptr = dst.get_unchecked_mut(px..);
    _mm256_storeu_si256(dst_ptr.as_mut_ptr() as *mut __m256i, rgb);
}

#[inline(always)]
unsafe fn convolve_vertical_part_8_avx(
    start_y: usize,
    start_x: usize,
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    filter: &[i16],
    bounds: &FilterBounds,
) {
    let vld = _mm256_set1_epi32(ROUNDING_CONST);
    let mut store_0 = vld;

    let zeros = _mm256_setzero_si256();

    let px = start_x;

    let bounds_size = bounds.size;

    for j in 0..bounds_size {
        let py = start_y + j;
        let weight = *filter.get_unchecked(j);
        let v_weight = _mm256_set1_epi32(weight as i32);
        let src_ptr = src.get_unchecked((src_stride * py + px)..);
        let item_row = _mm256_cvtepu16_epi32(_mm_cvtepu8_epi16(_mm_loadu_si64(src_ptr.as_ptr())));

        store_0 = _mm256_add_epi32(store_0, _mm256_mullo_epi32(item_row, v_weight));
    }

    store_0 = _mm256_max_epi32(store_0, zeros);

    const MASK: i32 = shuffle(3, 1, 2, 0);

    let low_16 = _mm256_permute4x64_epi64::<MASK>(_mm256_packus_epi32(
        _mm256_srai_epi32::<PRECISION>(store_0),
        zeros,
    ));

    let item = _mm256_permute4x64_epi64::<MASK>(_mm256_packus_epi16(low_16, low_16));
    let item_sse = _mm256_castsi256_si128(item);

    let dst_ptr = dst.get_unchecked_mut(px..);
    std::ptr::copy_nonoverlapping(&item_sse as *const _ as *const u8, dst_ptr.as_mut_ptr(), 8);
}

#[inline(always)]
unsafe fn convolve_vertical_part_avx(
    start_y: usize,
    start_x: usize,
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    filter: &[i16],
    bounds: &FilterBounds,
) {
    let vld = _mm256_set1_epi32(ROUNDING_CONST);
    let mut store_0 = vld;

    let zeros = _mm256_setzero_si256();

    let px = start_x;

    let bounds_size = bounds.size;

    if bounds_size == 2 {
        let py = start_y;
        let weight = filter.get_unchecked(0..2);
        let v_weight0 = _mm256_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm256_set1_epi32(weight[1] as i32);
        let src_ptr0 = src.get_unchecked(src_stride * py + px);
        let src_ptr1 = src.get_unchecked(src_stride * (py + 1) + px);
        let item_row0 = _mm256_insert_epi8::<0>(_mm256_setzero_si256(), *src_ptr0 as i8);
        let item_row1 = _mm256_insert_epi8::<0>(_mm256_setzero_si256(), *src_ptr1 as i8);

        store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row0, v_weight0));
        store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row1, v_weight1));
    } else if bounds_size == 3 {
        let py = start_y;
        let weight = filter.get_unchecked(0..3);
        let v_weight0 = _mm256_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm256_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm256_set1_epi32(weight[2] as i32);
        let src_ptr0 = src.get_unchecked(src_stride * py + px);
        let src_ptr1 = src.get_unchecked(src_stride * (py + 1) + px);
        let src_ptr2 = src.get_unchecked(src_stride * (py + 2) + px);
        let item_row0 = _mm256_insert_epi8::<0>(_mm256_setzero_si256(), *src_ptr0 as i8);
        let item_row1 = _mm256_insert_epi8::<0>(_mm256_setzero_si256(), *src_ptr1 as i8);
        let item_row2 = _mm256_insert_epi8::<0>(_mm256_setzero_si256(), *src_ptr2 as i8);

        store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row0, v_weight0));
        store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row1, v_weight1));
        store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row2, v_weight2));
    } else if bounds_size == 4 {
        let py = start_y;
        let weight = filter.get_unchecked(0..4);
        let v_weight0 = _mm256_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm256_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm256_set1_epi32(weight[2] as i32);
        let v_weight3 = _mm256_set1_epi32(weight[3] as i32);
        let src_ptr0 = src.get_unchecked(src_stride * py + px);
        let src_ptr1 = src.get_unchecked(src_stride * (py + 1) + px);
        let src_ptr2 = src.get_unchecked(src_stride * (py + 2) + px);
        let src_ptr3 = src.get_unchecked(src_stride * (py + 3) + px);
        let item_row0 = _mm256_insert_epi8::<0>(_mm256_setzero_si256(), *src_ptr0 as i8);
        let item_row1 = _mm256_insert_epi8::<0>(_mm256_setzero_si256(), *src_ptr1 as i8);
        let item_row2 = _mm256_insert_epi8::<0>(_mm256_setzero_si256(), *src_ptr2 as i8);
        let item_row3 = _mm256_insert_epi8::<0>(_mm256_setzero_si256(), *src_ptr3 as i8);

        store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row0, v_weight0));
        store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row1, v_weight1));
        store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row2, v_weight2));
        store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row3, v_weight3));
    } else {
        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = *filter.get_unchecked(j);
            let v_weight = _mm256_set1_epi32(weight as i32);
            let src_ptr = src.get_unchecked(src_stride * py + px);
            let item_row = _mm256_setr_epi32(*src_ptr as i32, 0, 0, 0, 0, 0, 0, 0);

            store_0 = _mm256_add_epi32(store_0, _mm256_madd_epi16(item_row, v_weight));
        }
    }

    store_0 = _mm256_max_epi32(store_0, zeros);

    let low_16 = _mm256_packus_epi32(_mm256_srai_epi32::<PRECISION>(store_0), zeros);

    let item = _mm256_packus_epi16(low_16, low_16);

    let dst_ptr = dst.get_unchecked_mut(px);
    *dst_ptr = _mm256_extract_epi8::<0>(item) as u8;
}

pub(crate) fn convolve_vertical_avx_row(
    dst_width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i16],
) {
    unsafe {
        convolve_vertical_avx_row_impl(dst_width, bounds, src, dst, src_stride, weights);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn convolve_vertical_avx_row_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i16],
) {
    let mut cx = 0usize;
    let total_width = dst.len();

    while cx + 64 < total_width {
        unsafe {
            convolve_vertical_part_avx_64(bounds.start, cx, src, src_stride, dst, weights, bounds);
        }

        cx += 64;
    }

    while cx + 32 < total_width {
        unsafe {
            convolve_vertical_part_avx_32(bounds.start, cx, src, src_stride, dst, weights, bounds);
        }

        cx += 32;
    }

    while cx + 8 < total_width {
        unsafe {
            convolve_vertical_part_8_avx(bounds.start, cx, src, src_stride, dst, weights, bounds);
        }

        cx += 8;
    }

    while cx < total_width {
        unsafe {
            convolve_vertical_part_avx(bounds.start, cx, src, src_stride, dst, weights, bounds);
        }

        cx += 1;
    }
}
