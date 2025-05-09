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
use crate::neon::utils::xvld1q_f32_x2;
use crate::neon::utils::{xvld1q_f32_x4, xvst1q_f32_x2, xvst1q_f32_x4};
use std::arch::aarch64::*;

#[inline(always)]
fn conv_vertical_part_neon_16_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f64],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f64(0.);
        let mut store_1 = vdupq_n_f64(0.);
        let mut store_2 = vdupq_n_f64(0.);
        let mut store_3 = vdupq_n_f64(0.);
        let mut store_4 = vdupq_n_f64(0.);
        let mut store_5 = vdupq_n_f64(0.);
        let mut store_6 = vdupq_n_f64(0.);
        let mut store_7 = vdupq_n_f64(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let v_weight = vld1q_dup_f64(filter.get_unchecked(j..).as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);

            let item_row = xvld1q_f32_x4(src_ptr.as_ptr());

            store_0 = vfmaq_f64(store_0, vcvt_f64_f32(vget_low_f32(item_row.0)), v_weight);
            store_1 = vfmaq_f64(store_1, vcvt_high_f64_f32(item_row.0), v_weight);

            store_2 = vfmaq_f64(store_2, vcvt_f64_f32(vget_low_f32(item_row.1)), v_weight);
            store_3 = vfmaq_f64(store_3, vcvt_high_f64_f32(item_row.1), v_weight);

            store_4 = vfmaq_f64(store_4, vcvt_f64_f32(vget_low_f32(item_row.2)), v_weight);
            store_5 = vfmaq_f64(store_5, vcvt_high_f64_f32(item_row.2), v_weight);

            store_6 = vfmaq_f64(store_6, vcvt_f64_f32(vget_low_f32(item_row.3)), v_weight);
            store_7 = vfmaq_f64(store_7, vcvt_high_f64_f32(item_row.3), v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        let f_set = float32x4x4_t(
            vcombine_f32(vcvt_f32_f64(store_0), vcvt_f32_f64(store_1)),
            vcombine_f32(vcvt_f32_f64(store_2), vcvt_f32_f64(store_3)),
            vcombine_f32(vcvt_f32_f64(store_4), vcvt_f32_f64(store_5)),
            vcombine_f32(vcvt_f32_f64(store_6), vcvt_f32_f64(store_7)),
        );
        xvst1q_f32_x4(dst_ptr, f_set);
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_neon_8_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f64],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f64(0.);
        let mut store_1 = vdupq_n_f64(0.);
        let mut store_2 = vdupq_n_f64(0.);
        let mut store_3 = vdupq_n_f64(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1q_dup_f64(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);
            let item_row = xvld1q_f32_x2(src_ptr.as_ptr());

            store_0 = vfmaq_f64(store_0, vcvt_f64_f32(vget_low_f32(item_row.0)), v_weight);
            store_1 = vfmaq_f64(store_1, vcvt_high_f64_f32(item_row.0), v_weight);

            store_2 = vfmaq_f64(store_2, vcvt_f64_f32(vget_low_f32(item_row.1)), v_weight);
            store_3 = vfmaq_f64(store_3, vcvt_high_f64_f32(item_row.1), v_weight);
        }

        let item = float32x4x2_t(
            vcombine_f32(vcvt_f32_f64(store_0), vcvt_f32_f64(store_1)),
            vcombine_f32(vcvt_f32_f64(store_2), vcvt_f32_f64(store_3)),
        );

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
    filter: &[f64],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f64(0.);
        let mut store_1 = vdupq_n_f64(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1q_dup_f64(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);

            let item_row = xvld1q_f32_x2(src_ptr.as_ptr());

            store_0 = vfmaq_f64(store_0, vcvt_f64_f32(vget_low_f32(item_row.0)), v_weight);
            store_1 = vfmaq_f64(store_1, vcvt_high_f64_f32(item_row.0), v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        vst1q_f32(
            dst_ptr,
            vcombine_f32(vcvt_f32_f64(store_0), vcvt_f32_f64(store_1)),
        );
    }
}

#[inline(always)]
unsafe fn convolve_vertical_part_neon_1_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f64],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdup_n_f64(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1_dup_f64(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);
            let item_row = vld1_dup_f32(src_ptr.as_ptr());

            store_0 = vfma_f64(store_0, vget_low_f64(vcvt_f64_f32(item_row)), v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        vst1_lane_f32::<0>(dst_ptr, vcvt_f32_f64(vcombine_f64(store_0, store_0)));
    }
}

pub(crate) fn convolve_vertical_neon_row_f32_f64(
    _: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f64],
) {
    let mut cx = 0usize;
    let dst_width = dst.len();

    while cx + 16 < dst_width {
        conv_vertical_part_neon_16_f32(bounds.start, cx, src, src_stride, dst, weight_ptr, bounds);

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
