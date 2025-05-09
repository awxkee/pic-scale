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
use crate::sse::_mm_prefer_fma_ps;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn convolve_vertical_part_sse_24_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();
        let mut store_1 = _mm_setzero_ps();
        let mut store_2 = _mm_setzero_ps();
        let mut store_3 = _mm_setzero_ps();
        let mut store_4 = _mm_setzero_ps();
        let mut store_5 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load1_ps(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = _mm_loadu_ps(src_ptr);
            let item_row_1 = _mm_loadu_ps(src_ptr.add(4));
            let item_row_2 = _mm_loadu_ps(src_ptr.add(8));
            let item_row_3 = _mm_loadu_ps(src_ptr.add(12));
            let item_row_4 = _mm_loadu_ps(src_ptr.add(16));
            let item_row_5 = _mm_loadu_ps(src_ptr.add(20));

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
            store_1 = _mm_prefer_fma_ps::<FMA>(store_1, item_row_1, v_weight);
            store_2 = _mm_prefer_fma_ps::<FMA>(store_2, item_row_2, v_weight);
            store_3 = _mm_prefer_fma_ps::<FMA>(store_3, item_row_3, v_weight);
            store_4 = _mm_prefer_fma_ps::<FMA>(store_4, item_row_4, v_weight);
            store_5 = _mm_prefer_fma_ps::<FMA>(store_5, item_row_5, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_ps(dst_ptr, store_0);
        _mm_storeu_ps(dst_ptr.add(4), store_1);
        _mm_storeu_ps(dst_ptr.add(8), store_2);
        _mm_storeu_ps(dst_ptr.add(12), store_3);
        _mm_storeu_ps(dst_ptr.add(16), store_4);
        _mm_storeu_ps(dst_ptr.add(20), store_5);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_sse_16_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();
        let mut store_1 = _mm_setzero_ps();
        let mut store_2 = _mm_setzero_ps();
        let mut store_3 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load1_ps(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = _mm_loadu_ps(src_ptr);
            let item_row_1 = _mm_loadu_ps(src_ptr.add(4));
            let item_row_2 = _mm_loadu_ps(src_ptr.add(8));
            let item_row_3 = _mm_loadu_ps(src_ptr.add(12));

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
            store_1 = _mm_prefer_fma_ps::<FMA>(store_1, item_row_1, v_weight);
            store_2 = _mm_prefer_fma_ps::<FMA>(store_2, item_row_2, v_weight);
            store_3 = _mm_prefer_fma_ps::<FMA>(store_3, item_row_3, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_ps(dst_ptr, store_0);
        _mm_storeu_ps(dst_ptr.add(4), store_1);
        _mm_storeu_ps(dst_ptr.add(8), store_2);
        _mm_storeu_ps(dst_ptr.add(12), store_3);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_sse_8_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();
        let mut store_1 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load1_ps(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = _mm_loadu_ps(src_ptr);
            let item_row_1 = _mm_loadu_ps(src_ptr.add(4));

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
            store_1 = _mm_prefer_fma_ps::<FMA>(store_1, item_row_1, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_ps(dst_ptr, store_0);
        _mm_storeu_ps(dst_ptr.add(4), store_1);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_sse_4_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load1_ps(weight.as_ptr());

            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = _mm_loadu_ps(src_ptr);

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_ps(dst_ptr, store_0);
    }
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_sse_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load_ss(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);

            let item_row_0 = _mm_load_ss(src_ptr.as_ptr());

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_store_ss(dst_ptr, store_0);
    }
}

pub(crate) fn convolve_vertical_rgb_sse_row_f32<const FMA: bool>(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    unsafe {
        if FMA {
            convolve_vertical_rgb_sse_row_f32_fma(width, bounds, src, dst, src_stride, weight_ptr);
        } else {
            convolve_vertical_rgb_sse_row_f32_regular(
                width, bounds, src, dst, src_stride, weight_ptr,
            );
        }
    }
}

#[target_feature(enable = "sse4.1")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_vertical_rgb_sse_row_f32_regular(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    unsafe {
        convolve_vertical_rgb_sse_row_f32_impl::<false>(
            width, bounds, src, dst, src_stride, weight_ptr,
        );
    }
}

#[target_feature(enable = "sse4.1", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_vertical_rgb_sse_row_f32_fma(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    unsafe {
        convolve_vertical_rgb_sse_row_f32_impl::<true>(
            width, bounds, src, dst, src_stride, weight_ptr,
        );
    }
}

#[inline(always)]
unsafe fn convolve_vertical_rgb_sse_row_f32_impl<const FMA: bool>(
    _: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    unsafe {
        let mut cx = 0usize;
        let dst_width = dst.len();

        while cx + 24 < dst_width {
            convolve_vertical_part_sse_24_f32::<FMA>(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );

            cx += 24;
        }

        while cx + 16 < dst_width {
            convolve_vertical_part_sse_16_f32::<FMA>(
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

        while cx + 8 < dst_width {
            convolve_vertical_part_sse_8_f32::<FMA>(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );

            cx += 8;
        }

        while cx + 4 < dst_width {
            convolve_vertical_part_sse_4_f32::<FMA>(
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
            convolve_vertical_part_sse_f32::<FMA>(
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
