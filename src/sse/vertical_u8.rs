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

use crate::filter_weights::FilterBounds;
use crate::support::{PRECISION, ROUNDING_CONST};

#[inline(always)]
unsafe fn dot_prod(
    store_0: __m128i,
    store_1: __m128i,
    store_2: __m128i,
    store_3: __m128i,
    v: __m128i,
    w: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let interleaved = _mm_unpacklo_epi8(v, zeros);
    let pix = _mm_unpacklo_epi8(interleaved, zeros);
    let store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(pix, w));
    let pix = _mm_unpackhi_epi8(interleaved, zeros);
    let store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(pix, w));

    let interleaved = _mm_unpackhi_epi8(v, zeros);
    let pix = _mm_unpacklo_epi8(interleaved, zeros);
    let store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(pix, w));
    let pix = _mm_unpackhi_epi8(interleaved, zeros);
    let store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(pix, w));

    (store_0, store_1, store_2, store_3)
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_32(
    start_y: usize,
    start_x: usize,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    filter: &[i16],
    bounds: &FilterBounds,
) {
    let zeros = _mm_setzero_si128();
    let vld = _mm_set1_epi32(ROUNDING_CONST);
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

    let mut jj = 0usize;

    while jj < bounds_size.saturating_sub(2) {
        let py = start_y + jj;
        let f_ptr = filter.get_unchecked(jj..).as_ptr() as *const i32;
        let v_weight_2 = _mm_set1_epi32(f_ptr.read_unaligned());
        let src_ptr = src.get_unchecked((src_stride * py + px)..);
        let s_ptr_next = src_ptr.as_ptr().add(src_stride);

        let item_row_0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
        let item_row_1 = _mm_loadu_si128(s_ptr_next as *const __m128i);

        let interleaved = _mm_unpacklo_epi8(item_row_0, item_row_1);
        let pix = _mm_unpacklo_epi8(interleaved, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(pix, v_weight_2));
        let pix = _mm_unpackhi_epi8(interleaved, zeros);
        store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(pix, v_weight_2));

        let interleaved = _mm_unpackhi_epi8(item_row_0, item_row_1);
        let pix = _mm_unpacklo_epi8(interleaved, zeros);
        store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(pix, v_weight_2));
        let pix = _mm_unpackhi_epi8(interleaved, zeros);
        store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(pix, v_weight_2));

        let item_row_0 = _mm_loadu_si128(src_ptr.as_ptr().add(16) as *const __m128i);
        let item_row_1 = _mm_loadu_si128(s_ptr_next.add(16) as *const __m128i);

        let interleaved = _mm_unpacklo_epi8(item_row_0, item_row_1);
        let pix = _mm_unpacklo_epi8(interleaved, zeros);
        store_4 = _mm_add_epi32(store_4, _mm_madd_epi16(pix, v_weight_2));
        let pix = _mm_unpackhi_epi8(interleaved, zeros);
        store_5 = _mm_add_epi32(store_5, _mm_madd_epi16(pix, v_weight_2));

        let interleaved = _mm_unpackhi_epi8(item_row_0, item_row_1);
        let pix = _mm_unpacklo_epi8(interleaved, zeros);
        store_6 = _mm_add_epi32(store_6, _mm_madd_epi16(pix, v_weight_2));
        let pix = _mm_unpackhi_epi8(interleaved, zeros);
        store_7 = _mm_add_epi32(store_7, _mm_madd_epi16(pix, v_weight_2));

        jj += 2;
    }

    for j in jj..bounds_size {
        let py = start_y + j;
        let weight = *filter.get_unchecked(j);
        let v_weight = _mm_set1_epi32(weight as i32);
        let src_ptr = src.get_unchecked((src_stride * py + px)..);
        let item_row_0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
        let item_row_1 = _mm_loadu_si128(src_ptr.as_ptr().add(16) as *const __m128i);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row_0, v_weight);
        (store_4, store_5, store_6, store_7) =
            dot_prod(store_4, store_5, store_6, store_7, item_row_1, v_weight);
    }

    let rgb0 = _mm_packs_epi32(store_0, store_1);
    let rgb2 = _mm_packs_epi32(store_2, store_3);
    let rgb = _mm_packus_epi16(rgb0, rgb2);

    let dst_ptr = dst.get_unchecked_mut(px..);
    _mm_storeu_si128(dst_ptr.as_mut_ptr() as *mut __m128i, rgb);

    let rgb0 = _mm_packs_epi32(store_4, store_5);
    let rgb2 = _mm_packs_epi32(store_6, store_7);
    let rgb = _mm_packus_epi16(rgb0, rgb2);

    let dst_ptr = dst.get_unchecked_mut((px + 16)..);
    _mm_storeu_si128(dst_ptr.as_mut_ptr() as *mut __m128i, rgb);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_16(
    start_y: usize,
    start_x: usize,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    filter: &[i16],
    bounds: &FilterBounds,
) {
    let vld = _mm_set1_epi32(ROUNDING_CONST);
    let mut store_0 = vld;
    let mut store_1 = vld;
    let mut store_2 = vld;
    let mut store_3 = vld;

    let px = start_x;

    let bounds_size = bounds.size;

    for j in 0..bounds_size {
        let py = start_y + j;
        let weight = *filter.get_unchecked(j);
        let v_weight = _mm_set1_epi32(weight as i32);
        let src_ptr = src.get_unchecked((src_stride * py + px)..);
        let item_row = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);

        (store_0, store_1, store_2, store_3) =
            dot_prod(store_0, store_1, store_2, store_3, item_row, v_weight);
    }

    let low_16 = _mm_packs_epi32(
        _mm_srai_epi32::<PRECISION>(store_0),
        _mm_srai_epi32::<PRECISION>(store_1),
    );
    let high_16 = _mm_packs_epi32(
        _mm_srai_epi32::<PRECISION>(store_2),
        _mm_srai_epi32::<PRECISION>(store_3),
    );

    let item = _mm_packus_epi16(low_16, high_16);

    let dst_ptr = dst.get_unchecked_mut(px..);
    _mm_storeu_si128(dst_ptr.as_mut_ptr() as *mut __m128i, item);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_8(
    start_y: usize,
    start_x: usize,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    filter: &[i16],
    bounds: &FilterBounds,
) {
    let vld = _mm_set1_epi32(ROUNDING_CONST);
    let mut store_0 = vld;
    let mut store_1 = vld;

    let zeros = _mm_setzero_si128();

    let px = start_x;

    let bounds_size = bounds.size;

    if bounds_size == 2 {
        let py = start_y;
        let weight = filter.get_unchecked(0..2);
        let v_weight0 = _mm_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm_set1_epi32(weight[1] as i32);
        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let item_row0 = _mm_loadu_si64(src_ptr0.as_ptr());
        let item_row1 = _mm_loadu_si64(src_ptr1.as_ptr());

        let low0 = _mm_unpacklo_epi8(item_row0, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low0), v_weight0));
        store_1 = _mm_add_epi32(
            store_1,
            _mm_madd_epi16(_mm_unpackhi_epi16(low0, zeros), v_weight0),
        );

        let low1 = _mm_unpacklo_epi8(item_row1, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low1), v_weight1));
        store_1 = _mm_add_epi32(
            store_1,
            _mm_madd_epi16(_mm_unpackhi_epi16(low1, zeros), v_weight1),
        );
    } else if bounds_size == 3 {
        let py = start_y;
        let weight = filter.get_unchecked(0..3);
        let v_weight0 = _mm_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm_set1_epi32(weight[2] as i32);
        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
        let item_row0 = _mm_loadu_si64(src_ptr0.as_ptr());
        let item_row1 = _mm_loadu_si64(src_ptr1.as_ptr());
        let item_row2 = _mm_loadu_si64(src_ptr2.as_ptr());

        let low0 = _mm_unpacklo_epi8(item_row0, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low0), v_weight0));
        store_1 = _mm_add_epi32(
            store_1,
            _mm_madd_epi16(_mm_unpackhi_epi16(low0, zeros), v_weight0),
        );

        let low1 = _mm_unpacklo_epi8(item_row1, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low1), v_weight1));
        store_1 = _mm_add_epi32(
            store_1,
            _mm_madd_epi16(_mm_unpackhi_epi16(low1, zeros), v_weight1),
        );

        let low2 = _mm_unpacklo_epi8(item_row2, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low2), v_weight2));
        store_1 = _mm_add_epi32(
            store_1,
            _mm_madd_epi16(_mm_unpackhi_epi16(low2, zeros), v_weight2),
        );
    } else if bounds_size == 4 {
        let py = start_y;
        let weight = filter.get_unchecked(0..4);
        let v_weight0 = _mm_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm_set1_epi32(weight[2] as i32);
        let v_weight3 = _mm_set1_epi32(weight[3] as i32);
        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
        let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);
        let item_row0 = _mm_loadu_si64(src_ptr0.as_ptr());
        let item_row1 = _mm_loadu_si64(src_ptr1.as_ptr());
        let item_row2 = _mm_loadu_si64(src_ptr2.as_ptr());
        let item_row3 = _mm_loadu_si64(src_ptr3.as_ptr());

        let low0 = _mm_unpacklo_epi8(item_row0, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low0), v_weight0));
        store_1 = _mm_add_epi32(
            store_1,
            _mm_madd_epi16(_mm_unpackhi_epi16(low0, zeros), v_weight0),
        );

        let low1 = _mm_unpacklo_epi8(item_row1, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low1), v_weight1));
        store_1 = _mm_add_epi32(
            store_1,
            _mm_madd_epi16(_mm_unpackhi_epi16(low1, zeros), v_weight1),
        );

        let low2 = _mm_unpacklo_epi8(item_row2, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low2), v_weight2));
        store_1 = _mm_add_epi32(
            store_1,
            _mm_madd_epi16(_mm_unpackhi_epi16(low2, zeros), v_weight2),
        );

        let low3 = _mm_unpacklo_epi8(item_row3, zeros);
        store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low3), v_weight3));
        store_1 = _mm_add_epi32(
            store_1,
            _mm_madd_epi16(_mm_unpackhi_epi16(low3, zeros), v_weight3),
        );
    } else {
        for j in 0..bounds_size {
            let py = start_y + j;
            let weight = *filter.get_unchecked(j);
            let v_weight = _mm_set1_epi32(weight as i32);
            let src_ptr = src.get_unchecked((src_stride * py + px)..);
            let item_row = _mm_loadu_si64(src_ptr.as_ptr());

            let low = _mm_unpacklo_epi8(item_row, zeros);
            store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low), v_weight));
            store_1 = _mm_add_epi32(
                store_1,
                _mm_madd_epi16(_mm_unpackhi_epi16(low, zeros), v_weight),
            );
        }
    }

    let low_16 = _mm_packus_epi32(
        _mm_srai_epi32::<PRECISION>(store_0),
        _mm_srai_epi32::<PRECISION>(store_1),
    );

    let item = _mm_packus_epi16(low_16, low_16);

    let dst_ptr = dst.get_unchecked_mut(px..);
    _mm_storeu_si64(dst_ptr.as_mut_ptr(), item);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse(
    start_y: usize,
    start_x: usize,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    filter: &[i16],
    bounds: &FilterBounds,
) {
    let vld = _mm_set1_epi32(ROUNDING_CONST);
    let mut store = vld;

    let px = start_x;

    let bounds_size = bounds.size;

    if bounds_size == 2 {
        let py = start_y;
        let weight = filter.get_unchecked(0..2);
        let v_weight0 = _mm_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm_set1_epi32(weight[1] as i32);
        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let item_row0 =
            _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr0.get_unchecked(0) as i32);
        let item_row1 =
            _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr1.get_unchecked(0) as i32);

        store = _mm_add_epi32(store, _mm_madd_epi16(item_row0, v_weight0));
        store = _mm_add_epi32(store, _mm_madd_epi16(item_row1, v_weight1));
    } else if bounds_size == 3 {
        let py = start_y;
        let weight = filter.get_unchecked(0..3);
        let v_weight0 = _mm_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm_set1_epi32(weight[2] as i32);
        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
        let item_row0 =
            _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr0.get_unchecked(0) as i32);
        let item_row1 =
            _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr1.get_unchecked(0) as i32);
        let item_row2 =
            _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr2.get_unchecked(0) as i32);

        store = _mm_add_epi32(store, _mm_madd_epi16(item_row0, v_weight0));
        store = _mm_add_epi32(store, _mm_madd_epi16(item_row1, v_weight1));
        store = _mm_add_epi32(store, _mm_madd_epi16(item_row2, v_weight2));
    } else if bounds_size == 4 {
        let py = start_y;
        let weight = filter.get_unchecked(0..4);
        let v_weight0 = _mm_set1_epi32(weight[0] as i32);
        let v_weight1 = _mm_set1_epi32(weight[1] as i32);
        let v_weight2 = _mm_set1_epi32(weight[2] as i32);
        let v_weight3 = _mm_set1_epi32(weight[3] as i32);
        let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
        let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
        let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
        let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);
        let item_row0 =
            _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr0.get_unchecked(0) as i32);
        let item_row1 =
            _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr1.get_unchecked(0) as i32);
        let item_row2 =
            _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr2.get_unchecked(0) as i32);
        let item_row3 =
            _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr3.get_unchecked(0) as i32);

        store = _mm_add_epi32(store, _mm_madd_epi16(item_row0, v_weight0));
        store = _mm_add_epi32(store, _mm_madd_epi16(item_row1, v_weight1));
        store = _mm_add_epi32(store, _mm_madd_epi16(item_row2, v_weight2));
        store = _mm_add_epi32(store, _mm_madd_epi16(item_row3, v_weight3));
    } else {
        for j in 0..bounds_size {
            let py = start_y + j;
            let weight = *filter.get_unchecked(j);
            let v_weight = _mm_set1_epi32(weight as i32);
            let src_ptr = src.get_unchecked((src_stride * py + px)..);
            let item_row =
                _mm_insert_epi8::<0>(_mm_setzero_si128(), *src_ptr.get_unchecked(0) as i32);

            store = _mm_add_epi32(store, _mm_madd_epi16(item_row, v_weight));
        }
    }

    let vegi = _mm_srai_epi32::<PRECISION>(store);

    let low_16 = _mm_packus_epi32(vegi, vegi);

    let item = _mm_packus_epi16(low_16, low_16);

    let dst_ptr = dst.get_unchecked_mut(px);
    *dst_ptr = _mm_extract_epi8::<0>(item) as u8;
}

pub(crate) fn convolve_vertical_sse_row(
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

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_vertical_sse_row_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i16],
) {
    let mut cx = 0usize;
    let total_width = dst.len();

    #[cfg(target_arch = "x86_64")]
    while cx + 32 < total_width {
        unsafe {
            convolve_vertical_part_sse_32(bounds.start, cx, src, dst, src_stride, weights, bounds);
        }

        cx += 32;
    }

    while cx + 16 < total_width {
        unsafe {
            convolve_vertical_part_sse_16(bounds.start, cx, src, dst, src_stride, weights, bounds);
        }

        cx += 16;
    }

    while cx + 8 < total_width {
        unsafe {
            convolve_vertical_part_sse_8(bounds.start, cx, src, dst, src_stride, weights, bounds);
        }

        cx += 8;
    }

    while cx < total_width {
        unsafe {
            convolve_vertical_part_sse(bounds.start, cx, src, dst, src_stride, weights, bounds);
        }

        cx += 1;
    }
}
