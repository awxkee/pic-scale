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
use crate::avx512::utils::_mm512_dot16_epi32;
use crate::filter_weights::FilterBounds;
use crate::support::ROUNDING_CONST;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
/// This is fixed point path for bit-depth's lower or equal to 12
pub(crate) fn convolve_column_lb_avx512_u16(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    unsafe {
        if std::arch::is_x86_feature_detected!("avx512vnni") {
            convolve_column_lb_avx512_dot(bounds, src, dst, src_stride, weight, bit_depth);
        } else {
            convolve_column_lb_avx512_reg(bounds, src, dst, src_stride, weight, bit_depth);
        }
    }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn convolve_column_lb_avx512_reg(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    convolve_column_lb_avx512_impl::<false>(bounds, src, dst, src_stride, weight, bit_depth);
}

#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
unsafe fn convolve_column_lb_avx512_dot(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    convolve_column_lb_avx512_impl::<true>(bounds, src, dst, src_stride, weight, bit_depth);
}

#[inline(always)]
unsafe fn convolve_column_lb_avx512_impl<const HAS_DOT: bool>(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    assert!((1..=16).contains(&bit_depth));
    let max_colors = (1 << bit_depth) - 1;

    let bounds_size = bounds.size;

    let zeros = _mm512_setzero_si512();
    let initial_store = _mm512_set1_epi32(ROUNDING_CONST);
    let v_max_colors = _mm512_set1_epi16(max_colors);

    const PRECISION: u32 = 15;

    for (x, dst) in dst.chunks_mut(32).enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;

        let v_dx = x * 32;

        let working_mask: __mmask32 = if dst.len() == 32 {
            0xffff_ffff
        } else {
            0xffff_ffff >> (32 - dst.len())
        };

        if bounds_size == 2 {
            let weights = weight.get_unchecked(0..2);

            let v_weight0 = _mm512_set1_epi16(weights[0]);
            let v_weight1 = _mm512_set1_epi16(weights[1]);

            let py = bounds.start;
            let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
            let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);

            let item_row0 = _mm512_maskz_loadu_epi16(working_mask, src_ptr0.as_ptr() as *const i16);
            let item_row1 = _mm512_maskz_loadu_epi16(working_mask, src_ptr1.as_ptr() as *const i16);

            store0 = _mm512_dot16_epi32::<HAS_DOT>(
                store0,
                _mm512_unpacklo_epi16(item_row0, zeros),
                v_weight0,
            );
            store1 = _mm512_dot16_epi32::<HAS_DOT>(
                store1,
                _mm512_unpackhi_epi16(item_row0, zeros),
                v_weight0,
            );

            store0 = _mm512_dot16_epi32::<HAS_DOT>(
                store0,
                _mm512_unpacklo_epi16(item_row1, zeros),
                v_weight1,
            );
            store1 = _mm512_dot16_epi32::<HAS_DOT>(
                store1,
                _mm512_unpackhi_epi16(item_row1, zeros),
                v_weight1,
            );
        } else if bounds_size == 3 {
            let weights = weight.get_unchecked(0..3);

            let v_weight0 = _mm512_set1_epi16(weights[0]);
            let v_weight1 = _mm512_set1_epi16(weights[1]);
            let v_weight2 = _mm512_set1_epi16(weights[2]);

            let py = bounds.start;
            let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
            let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
            let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);

            let item_row0 = _mm512_maskz_loadu_epi16(working_mask, src_ptr0.as_ptr() as *const i16);
            let item_row1 = _mm512_maskz_loadu_epi16(working_mask, src_ptr1.as_ptr() as *const i16);
            let item_row2 = _mm512_maskz_loadu_epi16(working_mask, src_ptr2.as_ptr() as *const i16);

            store0 = _mm512_dot16_epi32::<HAS_DOT>(
                store0,
                _mm512_unpacklo_epi16(item_row0, zeros),
                v_weight0,
            );
            store1 = _mm512_dot16_epi32::<HAS_DOT>(
                store1,
                _mm512_unpackhi_epi16(item_row0, zeros),
                v_weight0,
            );

            store0 = _mm512_dot16_epi32::<HAS_DOT>(
                store0,
                _mm512_unpacklo_epi16(item_row1, zeros),
                v_weight1,
            );
            store1 = _mm512_dot16_epi32::<HAS_DOT>(
                store1,
                _mm512_unpackhi_epi16(item_row1, zeros),
                v_weight1,
            );

            store0 = _mm512_dot16_epi32::<HAS_DOT>(
                store0,
                _mm512_unpacklo_epi16(item_row2, zeros),
                v_weight2,
            );
            store1 = _mm512_dot16_epi32::<HAS_DOT>(
                store1,
                _mm512_unpackhi_epi16(item_row2, zeros),
                v_weight2,
            );
        } else if bounds_size == 4 {
            let weights = weight.get_unchecked(0..4);

            let v_weight0 = _mm512_set1_epi16(weights[0]);
            let v_weight1 = _mm512_set1_epi16(weights[1]);
            let v_weight2 = _mm512_set1_epi16(weights[2]);
            let v_weight3 = _mm512_set1_epi16(weights[3]);

            let py = bounds.start;
            let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
            let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
            let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);
            let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + v_dx)..);

            let item_row0 = _mm512_maskz_loadu_epi16(working_mask, src_ptr0.as_ptr() as *const i16);
            let item_row1 = _mm512_maskz_loadu_epi16(working_mask, src_ptr1.as_ptr() as *const i16);
            let item_row2 = _mm512_maskz_loadu_epi16(working_mask, src_ptr2.as_ptr() as *const i16);
            let item_row3 = _mm512_maskz_loadu_epi16(working_mask, src_ptr3.as_ptr() as *const i16);

            store0 = _mm512_dot16_epi32::<HAS_DOT>(
                store0,
                _mm512_unpacklo_epi16(item_row0, zeros),
                v_weight0,
            );
            store1 = _mm512_dot16_epi32::<HAS_DOT>(
                store1,
                _mm512_unpackhi_epi16(item_row0, zeros),
                v_weight0,
            );

            store0 = _mm512_dot16_epi32::<HAS_DOT>(
                store0,
                _mm512_unpacklo_epi16(item_row1, zeros),
                v_weight1,
            );
            store1 = _mm512_dot16_epi32::<HAS_DOT>(
                store1,
                _mm512_unpackhi_epi16(item_row1, zeros),
                v_weight1,
            );

            store0 = _mm512_dot16_epi32::<HAS_DOT>(
                store0,
                _mm512_unpacklo_epi16(item_row2, zeros),
                v_weight2,
            );
            store1 = _mm512_dot16_epi32::<HAS_DOT>(
                store1,
                _mm512_unpackhi_epi16(item_row2, zeros),
                v_weight2,
            );

            store0 = _mm512_dot16_epi32::<HAS_DOT>(
                store0,
                _mm512_unpacklo_epi16(item_row3, zeros),
                v_weight3,
            );
            store1 = _mm512_dot16_epi32::<HAS_DOT>(
                store1,
                _mm512_unpackhi_epi16(item_row3, zeros),
                v_weight3,
            );
        } else {
            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm512_set1_epi16(k_weight);

                let item_row0 =
                    _mm512_maskz_loadu_epi16(working_mask, src_ptr.as_ptr() as *const i16);

                store0 = _mm512_dot16_epi32::<HAS_DOT>(
                    store0,
                    _mm512_unpacklo_epi16(item_row0, zeros),
                    v_weight,
                );
                store1 = _mm512_dot16_epi32::<HAS_DOT>(
                    store1,
                    _mm512_unpackhi_epi16(item_row0, zeros),
                    v_weight,
                );
            }
        }

        let v_st0 = _mm512_srai_epi32::<PRECISION>(store0);
        let v_st1 = _mm512_srai_epi32::<PRECISION>(store1);

        let item0 = _mm512_min_epi16(_mm512_packus_epi32(v_st0, v_st1), v_max_colors);

        _mm512_mask_storeu_epi16(dst.as_mut_ptr() as *mut i16, working_mask, item0);
    }
}
