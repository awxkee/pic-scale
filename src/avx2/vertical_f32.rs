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

use crate::avx2::utils::_mm256_fma_ps;
use crate::filter_weights::FilterBounds;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_avx_32_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: &[f32],
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm256_setzero_ps();
    let mut store_1 = _mm256_setzero_ps();
    let mut store_2 = _mm256_setzero_ps();
    let mut store_3 = _mm256_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = *filter.get_unchecked(j);
        let v_weight = _mm256_set1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm256_loadu_ps(s_ptr);
        let item_row_1 = _mm256_loadu_ps(s_ptr.add(8));
        let item_row_2 = _mm256_loadu_ps(s_ptr.add(16));
        let item_row_3 = _mm256_loadu_ps(s_ptr.add(24));

        store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, v_weight);
        store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, v_weight);
        store_2 = _mm256_fma_ps::<FMA>(store_2, item_row_2, v_weight);
        store_3 = _mm256_fma_ps::<FMA>(store_3, item_row_3, v_weight);
    }

    let dst_ptr = dst.add(px);
    _mm256_storeu_ps(dst_ptr, store_0);
    _mm256_storeu_ps(dst_ptr.add(8), store_1);
    _mm256_storeu_ps(dst_ptr.add(16), store_2);
    _mm256_storeu_ps(dst_ptr.add(24), store_3);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_avx_16_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: &[f32],
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm256_setzero_ps();
    let mut store_1 = _mm256_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = *filter.get_unchecked(j);
        let v_weight = _mm256_set1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm256_loadu_ps(s_ptr);
        let item_row_1 = _mm256_loadu_ps(s_ptr.add(8));

        store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, v_weight);
        store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, v_weight);
    }

    let dst_ptr = dst.add(px);
    _mm256_storeu_ps(dst_ptr, store_0);
    _mm256_storeu_ps(dst_ptr.add(8), store_1);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_avx_8_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: &[f32],
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm256_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = *filter.get_unchecked(j);
        let v_weight = _mm256_set1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm256_loadu_ps(s_ptr);

        store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, v_weight);
    }

    let dst_ptr = dst.add(px);
    _mm256_storeu_ps(dst_ptr, store_0);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_avx_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: &[f32],
    bounds: &FilterBounds,
) {
    let mut store_0 = _mm256_setzero_ps();

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = *filter.get_unchecked(j);
        let v_weight = _mm256_set1_ps(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row_0 = _mm256_set1_ps(s_ptr.read_unaligned());

        store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, v_weight);
    }

    let dst_ptr = dst.add(px);
    (dst_ptr as *mut i32).write_unaligned(_mm256_extract_epi32::<0>(_mm256_castps_si256(store_0)));
}

#[inline]
pub(crate) fn convolve_vertical_avx_row_f32<const CHANNELS: usize, const FMA: bool>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    src_stride: usize,
    weight_ptr: &[f32],
) {
    unsafe {
        if FMA {
            convolve_vertical_avx_row_f32_fma::<CHANNELS>(
                width,
                bounds,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                src_stride,
                weight_ptr,
            );
        } else {
            convolve_vertical_avx_row_f32_regular::<CHANNELS>(
                width,
                bounds,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                src_stride,
                weight_ptr,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_vertical_avx_row_f32_regular<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    src_stride: usize,
    weight_ptr: &[f32],
) {
    convolve_vertical_avx_row_f32_impl::<CHANNELS, false>(
        width,
        bounds,
        unsafe_source_ptr_0,
        unsafe_destination_ptr_0,
        src_stride,
        weight_ptr,
    );
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn convolve_vertical_avx_row_f32_fma<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    src_stride: usize,
    weight_ptr: &[f32],
) {
    convolve_vertical_avx_row_f32_impl::<CHANNELS, true>(
        width,
        bounds,
        unsafe_source_ptr_0,
        unsafe_destination_ptr_0,
        src_stride,
        weight_ptr,
    );
}

#[inline(always)]
unsafe fn convolve_vertical_avx_row_f32_impl<const CHANNELS: usize, const FMA: bool>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    src_stride: usize,
    weight_ptr: &[f32],
) {
    let mut cx = 0usize;
    let dst_width = CHANNELS * width;

    while cx + 32 < dst_width {
        convolve_vertical_part_avx_32_f32::<FMA>(
            bounds.start,
            cx,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            weight_ptr,
            bounds,
        );

        cx += 32;
    }

    while cx + 16 < dst_width {
        unsafe {
            convolve_vertical_part_avx_16_f32::<FMA>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 16;
    }

    while cx + 8 < dst_width {
        convolve_vertical_part_avx_8_f32::<FMA>(
            bounds.start,
            cx,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            weight_ptr,
            bounds,
        );

        cx += 8;
    }

    while cx < dst_width {
        convolve_vertical_part_avx_f32::<FMA>(
            bounds.start,
            cx,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            weight_ptr,
            bounds,
        );
        cx += 1;
    }
}
