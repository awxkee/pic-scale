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
use crate::neon::utils::{prefer_vfmaq_f32, xvld1q_f32_x2};
use crate::neon::utils::{xvld1q_f32_x4, xvst1q_f32_x2, xvst1q_f32_x4};
use std::arch::aarch64::*;

macro_rules! conv_vertical_part_neon_16_f32 {
    ($start_y: expr, $start_x: expr, $src: expr, $src_stride: expr, $dst: expr, $filter: expr, $bounds: expr) => {{
        unsafe {
            let mut store_0 = vdupq_n_f32(0.);
            let mut store_1 = vdupq_n_f32(0.);
            let mut store_2 = vdupq_n_f32(0.);
            let mut store_3 = vdupq_n_f32(0.);

            let px = $start_x;

            for j in 0..$bounds.size {
                let py = $start_y + j;
                let v_weight = vld1q_dup_f32($filter.get_unchecked(j..).as_ptr());
                let src_ptr = $src.get_unchecked($src_stride * py + px..);

                let item_row = xvld1q_f32_x4(src_ptr.as_ptr());

                store_0 = prefer_vfmaq_f32(store_0, item_row.0, v_weight);
                store_1 = prefer_vfmaq_f32(store_1, item_row.1, v_weight);
                store_2 = prefer_vfmaq_f32(store_2, item_row.2, v_weight);
                store_3 = prefer_vfmaq_f32(store_3, item_row.3, v_weight);
            }

            let dst_ptr = $dst.get_unchecked_mut(px..).as_mut_ptr();
            let f_set = float32x4x4_t(store_0, store_1, store_2, store_3);
            xvst1q_f32_x4(dst_ptr, f_set);
        }
    }};
}

macro_rules! conv_vertical_part_neon_32_f32 {
    ($start_y: expr, $start_x: expr, $src: expr, $src_stride: expr, $dst: expr, $filter: expr, $bounds: expr) => {{
        unsafe {
            let mut store_0 = vdupq_n_f32(0.);
            let mut store_1 = vdupq_n_f32(0.);
            let mut store_2 = vdupq_n_f32(0.);
            let mut store_3 = vdupq_n_f32(0.);
            let mut store_4 = vdupq_n_f32(0.);
            let mut store_5 = vdupq_n_f32(0.);
            let mut store_6 = vdupq_n_f32(0.);
            let mut store_7 = vdupq_n_f32(0.);

            let px = $start_x;

            for j in 0..$bounds.size {
                let py = $start_y + j;
                let v_weight = vld1q_dup_f32($filter.get_unchecked(j..).as_ptr());
                let src_ptr = $src.get_unchecked($src_stride * py + px..).as_ptr();

                let item_row_0 = xvld1q_f32_x4(src_ptr);
                let item_row_1 = xvld1q_f32_x4(src_ptr.add(16));

                store_0 = prefer_vfmaq_f32(store_0, item_row_0.0, v_weight);
                store_1 = prefer_vfmaq_f32(store_1, item_row_0.1, v_weight);
                store_2 = prefer_vfmaq_f32(store_2, item_row_0.2, v_weight);
                store_3 = prefer_vfmaq_f32(store_3, item_row_0.3, v_weight);

                store_4 = prefer_vfmaq_f32(store_4, item_row_1.0, v_weight);
                store_5 = prefer_vfmaq_f32(store_5, item_row_1.1, v_weight);
                store_6 = prefer_vfmaq_f32(store_6, item_row_1.2, v_weight);
                store_7 = prefer_vfmaq_f32(store_7, item_row_1.3, v_weight);
            }

            let dst_ptr = $dst.get_unchecked_mut(px..).as_mut_ptr();
            let f_set = float32x4x4_t(store_0, store_1, store_2, store_3);
            xvst1q_f32_x4(dst_ptr, f_set);

            let f_set_1 = float32x4x4_t(store_4, store_5, store_6, store_7);
            xvst1q_f32_x4(dst_ptr.add(16), f_set_1);
        }
    }};
}

#[inline(always)]
fn conv_vertical_part_neon_48_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);
        let mut store_1 = vdupq_n_f32(0.);
        let mut store_2 = vdupq_n_f32(0.);
        let mut store_3 = vdupq_n_f32(0.);

        let mut store_4 = vdupq_n_f32(0.);
        let mut store_5 = vdupq_n_f32(0.);
        let mut store_6 = vdupq_n_f32(0.);
        let mut store_7 = vdupq_n_f32(0.);

        let mut store_8 = vdupq_n_f32(0.);
        let mut store_9 = vdupq_n_f32(0.);
        let mut store_10 = vdupq_n_f32(0.);
        let mut store_11 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let v_weight = vld1q_dup_f32(filter.get_unchecked(j..).as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = xvld1q_f32_x4(src_ptr);
            let item_row_1 = xvld1q_f32_x4(src_ptr.add(16));
            let item_row_2 = xvld1q_f32_x4(src_ptr.add(32));

            store_0 = prefer_vfmaq_f32(store_0, item_row_0.0, v_weight);
            store_1 = prefer_vfmaq_f32(store_1, item_row_0.1, v_weight);
            store_2 = prefer_vfmaq_f32(store_2, item_row_0.2, v_weight);
            store_3 = prefer_vfmaq_f32(store_3, item_row_0.3, v_weight);

            store_4 = prefer_vfmaq_f32(store_4, item_row_1.0, v_weight);
            store_5 = prefer_vfmaq_f32(store_5, item_row_1.1, v_weight);
            store_6 = prefer_vfmaq_f32(store_6, item_row_1.2, v_weight);
            store_7 = prefer_vfmaq_f32(store_7, item_row_1.3, v_weight);

            store_8 = prefer_vfmaq_f32(store_8, item_row_2.0, v_weight);
            store_9 = prefer_vfmaq_f32(store_9, item_row_2.1, v_weight);
            store_10 = prefer_vfmaq_f32(store_10, item_row_2.2, v_weight);
            store_11 = prefer_vfmaq_f32(store_11, item_row_2.3, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        let f_set = float32x4x4_t(store_0, store_1, store_2, store_3);
        xvst1q_f32_x4(dst_ptr, f_set);

        let f_set_1 = float32x4x4_t(store_4, store_5, store_6, store_7);
        xvst1q_f32_x4(dst_ptr.add(16), f_set_1);

        let f_set_2 = float32x4x4_t(store_8, store_9, store_10, store_11);
        xvst1q_f32_x4(dst_ptr.add(32), f_set_2);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_neon_8_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);
        let mut store_1 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1q_dup_f32(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);
            let item_row = xvld1q_f32_x2(src_ptr.as_ptr());

            store_0 = prefer_vfmaq_f32(store_0, item_row.0, v_weight);
            store_1 = prefer_vfmaq_f32(store_1, item_row.1, v_weight);
        }

        let item = float32x4x2_t(store_0, store_1);

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        xvst1q_f32_x2(dst_ptr, item);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_neon_4_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1q_dup_f32(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);

            let item_row = xvld1q_f32_x2(src_ptr.as_ptr());

            store_0 = prefer_vfmaq_f32(store_0, item_row.0, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        vst1q_f32(dst_ptr, store_0);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_neon_1_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1q_dup_f32(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);
            let item_row = vld1q_dup_f32(src_ptr.as_ptr());

            store_0 = prefer_vfmaq_f32(store_0, item_row, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        vst1q_lane_f32::<0>(dst_ptr, store_0);
    }
}

pub(crate) fn convolve_vertical_rgb_neon_row_f32<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    let mut cx = 0usize;
    let dst_width = width * CHANNELS;

    while cx + 48 < dst_width {
        conv_vertical_part_neon_48_f32(bounds.start, cx, src, src_stride, dst, weight_ptr, bounds);

        cx += 48;
    }

    while cx + 32 < dst_width {
        conv_vertical_part_neon_32_f32!(bounds.start, cx, src, src_stride, dst, weight_ptr, bounds);

        cx += 32;
    }

    while cx + 16 < dst_width {
        conv_vertical_part_neon_16_f32!(bounds.start, cx, src, src_stride, dst, weight_ptr, bounds);

        cx += 16;
    }

    while cx + 8 < dst_width {
        unsafe {
            convolve_vertical_part_neon_8_f32(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );
        }

        cx += 8;
    }

    while cx + 4 < dst_width {
        unsafe {
            convolve_vertical_part_neon_4_f32(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );
        }

        cx += 4;
    }

    while cx < dst_width {
        unsafe {
            convolve_vertical_part_neon_1_f32(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );
        }
        cx += 1;
    }
}
