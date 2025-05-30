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
use crate::avx2::utils::{_mm256_fma_ps, avx_combine_epi};
use crate::filter_weights::FilterBounds;
use core::f16;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn convolve_vertical_part_avx_f16<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm256_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j);
            let v_weight = _mm256_broadcast_ss(weight);
            let src_ptr = src.get_unchecked(src_stride * py..).as_ptr();

            let s_ptr = src_ptr.add(px);
            let item_row_0 = _mm256_set1_epi16(s_ptr.read_unaligned().to_bits() as i16);

            store_0 = _mm256_fma_ps::<FMA>(
                store_0,
                _mm256_cvtph_ps(_mm256_castsi256_si128(item_row_0)),
                v_weight,
            );
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();

        const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT;

        let converted = _mm256_cvtps_ph::<ROUNDING_FLAGS>(store_0);
        let first_item = _mm_extract_epi16::<0>(converted) as u16;
        (dst_ptr as *mut u16).write_unaligned(first_item);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_avx_4_f16<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm256_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j);
            let v_weight = _mm256_broadcast_ss(weight);
            let src_ptr = src.get_unchecked(src_stride * py..).as_ptr();

            let s_ptr = src_ptr.add(px);
            let item_row_0 = _mm_loadu_si64(s_ptr as *const u8);

            store_0 = _mm256_fma_ps::<FMA>(store_0, _mm256_cvtph_ps(item_row_0), v_weight);
        }

        const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT;

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        let acc = _mm256_cvtps_ph::<ROUNDING_FLAGS>(store_0);
        _mm_storeu_si64(dst_ptr as *mut u8, acc);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_avx_32_f16<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm256_setzero_ps();
        let mut store_1 = _mm256_setzero_ps();
        let mut store_2 = _mm256_setzero_ps();
        let mut store_3 = _mm256_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j);
            let v_weight = _mm256_broadcast_ss(weight);
            let src_ptr = src.get_unchecked(src_stride * py..).as_ptr();

            let s_ptr = src_ptr.add(px);
            let item_row_0 = _mm256_loadu_si256(s_ptr as *const __m256i);
            let item_row_1 = _mm256_loadu_si256(s_ptr.add(16) as *const __m256i);

            let items0 = _mm256_cvtph_ps(_mm256_castsi256_si128(item_row_0));
            let items1 = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(item_row_0));
            let items2 = _mm256_cvtph_ps(_mm256_castsi256_si128(item_row_1));
            let items3 = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(item_row_1));

            store_0 = _mm256_fma_ps::<FMA>(store_0, items0, v_weight);
            store_1 = _mm256_fma_ps::<FMA>(store_1, items1, v_weight);
            store_2 = _mm256_fma_ps::<FMA>(store_2, items2, v_weight);
            store_3 = _mm256_fma_ps::<FMA>(store_3, items3, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();

        const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT;

        let acc0 = avx_combine_epi(
            _mm256_cvtps_ph::<ROUNDING_FLAGS>(store_0),
            _mm256_cvtps_ph::<ROUNDING_FLAGS>(store_1),
        );
        let acc1 = avx_combine_epi(
            _mm256_cvtps_ph::<ROUNDING_FLAGS>(store_2),
            _mm256_cvtps_ph::<ROUNDING_FLAGS>(store_3),
        );

        _mm256_storeu_si256(dst_ptr as *mut __m256i, acc0);
        _mm256_storeu_si256(dst_ptr.add(16) as *mut __m256i, acc1);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_avx_16_f16<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm256_setzero_ps();
        let mut store_1 = _mm256_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j);
            let v_weight = _mm256_broadcast_ss(weight);
            let src_ptr = src.get_unchecked(src_stride * py..).as_ptr();

            let s_ptr = src_ptr.add(px);
            let item_row = _mm256_loadu_si256(s_ptr as *const __m256i);

            let items0 = _mm256_cvtph_ps(_mm256_castsi256_si128(item_row));
            let items1 = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(item_row));

            store_0 = _mm256_fma_ps::<FMA>(store_0, items0, v_weight);
            store_1 = _mm256_fma_ps::<FMA>(store_1, items1, v_weight);
        }

        const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT;

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        let acc0 = avx_combine_epi(
            _mm256_cvtps_ph::<ROUNDING_FLAGS>(store_0),
            _mm256_cvtps_ph::<ROUNDING_FLAGS>(store_1),
        );
        _mm256_storeu_si256(dst_ptr as *mut __m256i, acc0);
    }
}

pub(crate) fn convolve_vertical_avx_row_f16<const FMA: bool>(
    width: usize,
    bounds: &FilterBounds,
    src: &[f16],
    dst: &mut [f16],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    unsafe {
        if FMA {
            convolve_vertical_avx_row_f16_fma(width, bounds, src, dst, src_stride, weight_ptr);
        } else {
            convolve_vertical_avx_row_f16_regular(width, bounds, src, dst, src_stride, weight_ptr);
        }
    }
}

#[target_feature(enable = "avx2", enable = "f16c")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_vertical_avx_row_f16_regular(
    width: usize,
    bounds: &FilterBounds,
    src: &[f16],
    dst: &mut [f16],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    unsafe {
        convolve_vertical_avx_row_f16_impl::<false>(
            width, bounds, src, dst, src_stride, weight_ptr,
        );
    }
}

#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_vertical_avx_row_f16_fma(
    width: usize,
    bounds: &FilterBounds,
    src: &[f16],
    dst: &mut [f16],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    unsafe {
        convolve_vertical_avx_row_f16_impl::<true>(width, bounds, src, dst, src_stride, weight_ptr);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_avx_row_f16_impl<const FMA: bool>(
    _: usize,
    bounds: &FilterBounds,
    src: &[f16],
    dst: &mut [f16],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    unsafe {
        let mut cx = 0usize;
        let dst_width = dst.len();

        while cx + 32 < dst_width {
            convolve_vertical_part_avx_32_f16::<FMA>(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );

            cx += 32;
        }

        while cx + 16 < dst_width {
            convolve_vertical_part_avx_16_f16::<FMA>(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );

            cx += 16;
        }

        while cx + 4 < dst_width {
            convolve_vertical_part_avx_4_f16::<FMA>(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );

            cx += 4;
        }

        while cx < dst_width {
            convolve_vertical_part_avx_f16::<FMA>(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );

            cx += 1;
        }
    }
}
