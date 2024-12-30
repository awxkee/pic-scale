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
use crate::avx2::utils::avx2_pack_u32;
use crate::filter_weights::FilterBounds;
use crate::support::{PRECISION, ROUNDING_CONST};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
/// This is fixed point path for bit-depth's lower or equal to 12
pub(crate) fn convolve_column_lb_avx2_u16(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    unsafe {
        convolve_column_lb_avx_u16_impl(bounds, src, dst, src_stride, weight, bit_depth);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_column_lb_avx_u16_impl(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    assert!((1..=16).contains(&bit_depth));
    let max_colors = (1 << bit_depth) - 1;
    let mut cx = 0usize;

    let bounds_size = bounds.size;

    let zeros = _mm_setzero_si128();
    let zeros256 = _mm256_setzero_si256();
    let initial_store = _mm256_set1_epi32(ROUNDING_CONST);
    let v_max_colors = _mm256_set1_epi16(max_colors);
    let v_max_colors_sse = _mm_set1_epi16(max_colors);

    let v_px = cx;

    let iter32 = dst.chunks_exact_mut(32);

    for (x, dst) in iter32.enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;
        let mut store2 = initial_store;
        let mut store3 = initial_store;

        let v_dx = v_px + x * 32;

        for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
            let py = bounds.start + j;
            let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

            let v_weight = _mm256_set1_epi16(k_weight);

            let item_row0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);
            let item_row1 =
                _mm256_loadu_si256(src_ptr.get_unchecked(16..).as_ptr() as *const __m256i);

            store0 = _mm256_add_epi32(
                store0,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(item_row0, zeros256), v_weight),
            );
            store1 = _mm256_add_epi32(
                store1,
                _mm256_madd_epi16(_mm256_unpackhi_epi16(item_row0, zeros256), v_weight),
            );
            store2 = _mm256_add_epi32(
                store2,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(item_row1, zeros256), v_weight),
            );
            store3 = _mm256_add_epi32(
                store3,
                _mm256_madd_epi16(_mm256_unpackhi_epi16(item_row1, zeros256), v_weight),
            );
        }

        let v_st0 = _mm256_srai_epi32::<PRECISION>(store0);
        let v_st1 = _mm256_srai_epi32::<PRECISION>(store1);
        let v_st2 = _mm256_srai_epi32::<PRECISION>(store2);
        let v_st3 = _mm256_srai_epi32::<PRECISION>(store3);

        let item0 = _mm256_min_epi16(_mm256_packus_epi32(v_st0, v_st1), v_max_colors);
        let item1 = _mm256_min_epi16(_mm256_packus_epi32(v_st2, v_st3), v_max_colors);

        _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, item0);
        _mm256_storeu_si256(
            dst.get_unchecked_mut(16..).as_mut_ptr() as *mut __m256i,
            item1,
        );

        cx = v_dx;
    }

    let v_px = cx;

    let iter32_rem = dst.chunks_exact_mut(32).into_remainder();
    let iter16 = iter32_rem.chunks_exact_mut(16);

    for (x, dst) in iter16.enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;

        let v_dx = v_px + x * 16;

        for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
            let py = bounds.start + j;
            let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

            let v_weight = _mm256_set1_epi16(k_weight);

            let item_row0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);

            store0 = _mm256_add_epi32(
                store0,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(item_row0, zeros256), v_weight),
            );
            store1 = _mm256_add_epi32(
                store1,
                _mm256_madd_epi16(_mm256_unpackhi_epi16(item_row0, zeros256), v_weight),
            );
        }

        let v_st0 = _mm256_srai_epi32::<PRECISION>(store0);
        let v_st1 = _mm256_srai_epi32::<PRECISION>(store1);

        let item0 = _mm256_min_epi16(_mm256_packus_epi32(v_st0, v_st1), v_max_colors);

        _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, item0);

        cx = v_dx;
    }

    let tail16 = dst.chunks_exact_mut(16).into_remainder();
    let iter8 = tail16.chunks_exact_mut(8);

    let v_px = cx;

    for (x, dst) in iter8.enumerate() {
        let mut store0 = initial_store;

        let v_dx = v_px + x * 8;

        for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
            let py = bounds.start + j;
            let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

            let v_weight = _mm256_set1_epi16(k_weight);

            let item_row = _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(
                _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i),
            ));

            store0 = _mm256_add_epi32(
                store0,
                _mm256_madd_epi16(_mm256_unpacklo_epi16(item_row, zeros256), v_weight),
            );
        }

        let v_st0 = _mm256_srai_epi32::<PRECISION>(store0);

        let item = _mm256_min_epi16(avx2_pack_u32(v_st0, zeros256), v_max_colors);
        _mm_storeu_si128(
            dst.as_mut_ptr() as *mut __m128i,
            _mm256_castsi256_si128(item),
        );

        cx = v_dx;
    }

    let tail8 = tail16.chunks_exact_mut(8).into_remainder();
    let iter4 = tail8.chunks_exact_mut(4);

    let v_cx = cx;

    for (x, dst) in iter4.enumerate() {
        let mut store0 = _mm_set1_epi32(ROUNDING_CONST);

        let v_dx = v_cx + x * 4;

        if bounds_size == 2 {
            let weights = weight.get_unchecked(0..2);

            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);

            let py = bounds.start;
            let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
            let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);

            let item_row0 = _mm_loadu_si64(src_ptr0.as_ptr() as *const u8);
            store0 = _mm_add_epi32(
                store0,
                _mm_madd_epi16(_mm_unpacklo_epi16(item_row0, zeros), v_weight0),
            );

            let item_row1 = _mm_loadu_si64(src_ptr1.as_ptr() as *const u8);
            store0 = _mm_add_epi32(
                store0,
                _mm_madd_epi16(_mm_unpacklo_epi16(item_row1, zeros), v_weight1),
            );
        } else if bounds_size == 3 {
            let weights = weight.get_unchecked(0..3);

            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_weight2 = _mm_set1_epi16(weights[2]);

            let py = bounds.start;
            let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
            let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
            let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);

            let item_row0 = _mm_loadu_si64(src_ptr0.as_ptr() as *const u8);
            store0 = _mm_add_epi32(
                store0,
                _mm_madd_epi16(_mm_unpacklo_epi16(item_row0, zeros), v_weight0),
            );

            let item_row1 = _mm_loadu_si64(src_ptr1.as_ptr() as *const u8);
            store0 = _mm_add_epi32(
                store0,
                _mm_madd_epi16(_mm_unpacklo_epi16(item_row1, zeros), v_weight1),
            );

            let item_row2 = _mm_loadu_si64(src_ptr2.as_ptr() as *const u8);
            store0 = _mm_add_epi32(
                store0,
                _mm_madd_epi16(_mm_unpacklo_epi16(item_row2, zeros), v_weight2),
            );
        } else if bounds_size == 4 {
            let weights = weight.get_unchecked(0..4);

            let v_weight0 = _mm_set1_epi16(weights[0]);
            let v_weight1 = _mm_set1_epi16(weights[1]);
            let v_weight2 = _mm_set1_epi16(weights[2]);
            let v_weight3 = _mm_set1_epi16(weights[3]);

            let py = bounds.start;
            let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
            let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
            let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);
            let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + v_dx)..);

            let item_row0 = _mm_loadu_si64(src_ptr0.as_ptr() as *const u8);
            store0 = _mm_add_epi32(
                store0,
                _mm_madd_epi16(_mm_unpacklo_epi16(item_row0, zeros), v_weight0),
            );

            let item_row1 = _mm_loadu_si64(src_ptr1.as_ptr() as *const u8);
            store0 = _mm_add_epi32(
                store0,
                _mm_madd_epi16(_mm_unpacklo_epi16(item_row1, zeros), v_weight1),
            );

            let item_row2 = _mm_loadu_si64(src_ptr2.as_ptr() as *const u8);
            store0 = _mm_add_epi32(
                store0,
                _mm_madd_epi16(_mm_unpacklo_epi16(item_row2, zeros), v_weight2),
            );

            let item_row3 = _mm_loadu_si64(src_ptr3.as_ptr() as *const u8);
            store0 = _mm_add_epi32(
                store0,
                _mm_madd_epi16(_mm_unpacklo_epi16(item_row3, zeros), v_weight3),
            );
        } else {
            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm_set1_epi16(k_weight);

                let item_row = _mm_loadu_si64(src_ptr.as_ptr() as *const u8);

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpacklo_epi16(item_row, zeros), v_weight),
                );
            }
        }

        let v_st = _mm_srai_epi32::<PRECISION>(store0);

        let u_store0 = _mm_min_epi16(_mm_packus_epi32(v_st, v_st), v_max_colors_sse);
        _mm_storeu_si64(dst.as_mut_ptr() as *mut u8, u_store0);

        cx = v_dx;
    }

    let tail4 = tail8.chunks_exact_mut(4).into_remainder();

    let a_px = cx;

    for (x, dst) in tail4.iter_mut().enumerate() {
        let mut store0 = ROUNDING_CONST;

        let v_px = a_px + x;

        if bounds_size == 2 {
            let weights = weight.get_unchecked(0..2);
            let weight0 = weights[0];
            let weight1 = weights[1];

            let py = bounds.start;
            let offset0 = src_stride * py + v_px;
            let src_ptr0 = src.get_unchecked(offset0..(offset0 + 1));
            let offset1 = src_stride * (py + 1) + v_px;
            let src_ptr1 = src.get_unchecked(offset1..(offset1 + 1));

            store0 += src_ptr0[0] as i32 * weight0 as i32;
            store0 += src_ptr1[0] as i32 * weight1 as i32;
        } else if bounds_size == 3 {
            let weights = weight.get_unchecked(0..3);
            let weight0 = weights[0];
            let weight1 = weights[1];
            let weight2 = weights[2];

            let py = bounds.start;
            let offset0 = src_stride * py + v_px;
            let src_ptr0 = src.get_unchecked(offset0..(offset0 + 1));
            let offset1 = src_stride * (py + 1) + v_px;
            let src_ptr1 = src.get_unchecked(offset1..(offset1 + 1));
            let offset2 = src_stride * (py + 2) + v_px;
            let src_ptr2 = src.get_unchecked(offset2..(offset2 + 1));

            store0 += src_ptr0[0] as i32 * weight0 as i32;
            store0 += src_ptr1[0] as i32 * weight1 as i32;
            store0 += src_ptr2[0] as i32 * weight2 as i32;
        } else if bounds_size == 4 {
            let weights = weight.get_unchecked(0..4);
            let weight0 = weights[0];
            let weight1 = weights[1];
            let weight2 = weights[2];
            let weight3 = weights[3];

            let py = bounds.start;
            let offset0 = src_stride * py + v_px;
            let src_ptr0 = src.get_unchecked(offset0..(offset0 + 1));
            let offset1 = src_stride * (py + 1) + v_px;
            let src_ptr1 = src.get_unchecked(offset1..(offset1 + 1));
            let offset2 = src_stride * (py + 2) + v_px;
            let src_ptr2 = src.get_unchecked(offset2..(offset2 + 1));
            let offset3 = src_stride * (py + 3) + v_px;
            let src_ptr3 = src.get_unchecked(offset3..(offset3 + 1));

            store0 += src_ptr0[0] as i32 * weight0 as i32;
            store0 += src_ptr1[0] as i32 * weight1 as i32;
            store0 += src_ptr2[0] as i32 * weight2 as i32;
            store0 += src_ptr3[0] as i32 * weight3 as i32;
        } else {
            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_ptr = src.get_unchecked(offset..(offset + 1));

                store0 += src_ptr[0] as i32 * k_weight as i32;
            }
        }

        *dst = (store0 >> PRECISION).max(0).min(max_colors as i32) as u16;
    }
}
