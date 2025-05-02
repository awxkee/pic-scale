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
use std::arch::aarch64::*;

use crate::filter_weights::FilterBounds;
use crate::neon::utils::{xvld1q_u16_x2, xvld1q_u16_x4};
use core::f16;

#[inline(always)]
pub(crate) unsafe fn conv_vertical_part_neon_16_f16(
    start_y: usize,
    start_x: usize,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    filter: &[f16],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);
        let mut store_1 = vdupq_n_f32(0.);
        let mut store_2 = vdupq_n_f32(0.);
        let mut store_3 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let v_weight = vreinterpretq_f16_u16(vld1q_dup_u16(
                filter.get_unchecked(j..).as_ptr() as *const _
            ));
            let src_ptr = src.get_unchecked(src_stride * py..).as_ptr();

            let s_ptr = src_ptr.add(px);
            let item_row = xvld1q_u16_x2(s_ptr as *const _);

            store_0 = vfmlalq_low_f16(store_0, vreinterpretq_f16_u16(item_row.0), v_weight);
            store_1 = vfmlalq_high_f16(store_1, vreinterpretq_f16_u16(item_row.0), v_weight);
            store_2 = vfmlalq_low_f16(store_2, vreinterpretq_f16_u16(item_row.1), v_weight);
            store_3 = vfmlalq_high_f16(store_3, vreinterpretq_f16_u16(item_row.1), v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        let f_set = float16x8x2_t(
            vcombine_f16(vcvt_f16_f32(store_0), vcvt_f16_f32(store_1)),
            vcombine_f16(vcvt_f16_f32(store_2), vcvt_f16_f32(store_3)),
        );
        vst1q_f16(dst_ptr, f_set.0);
        vst1q_f16(dst_ptr.add(8), f_set.1);
    }
}

#[inline(always)]
pub(crate) unsafe fn conv_vertical_part_neon_32_f16(
    start_y: usize,
    start_x: usize,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    filter: &[f16],
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

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let v_weight = vreinterpretq_f16_u16(vld1q_dup_u16(
                filter.get_unchecked(j..).as_ptr() as *const _
            ));
            let src_ptr = src.get_unchecked(src_stride * py..).as_ptr();

            let s_ptr = src_ptr.add(px);
            let item_row = xvld1q_u16_x4(s_ptr as *const _);

            store_0 = vfmlalq_low_f16(store_0, vreinterpretq_f16_u16(item_row.0), v_weight);
            store_1 = vfmlalq_high_f16(store_1, vreinterpretq_f16_u16(item_row.0), v_weight);
            store_2 = vfmlalq_low_f16(store_2, vreinterpretq_f16_u16(item_row.1), v_weight);
            store_3 = vfmlalq_high_f16(store_3, vreinterpretq_f16_u16(item_row.1), v_weight);

            store_4 = vfmlalq_low_f16(store_4, vreinterpretq_f16_u16(item_row.2), v_weight);
            store_5 = vfmlalq_high_f16(store_5, vreinterpretq_f16_u16(item_row.2), v_weight);
            store_6 = vfmlalq_low_f16(store_6, vreinterpretq_f16_u16(item_row.3), v_weight);
            store_7 = vfmlalq_high_f16(store_7, vreinterpretq_f16_u16(item_row.3), v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        let f_set = float16x8x4_t(
            vcombine_f16(vcvt_f16_f32(store_0), vcvt_f16_f32(store_1)),
            vcombine_f16(vcvt_f16_f32(store_2), vcvt_f16_f32(store_3)),
            vcombine_f16(vcvt_f16_f32(store_4), vcvt_f16_f32(store_5)),
            vcombine_f16(vcvt_f16_f32(store_6), vcvt_f16_f32(store_7)),
        );
        vst1q_f16(dst_ptr, f_set.0);
        vst1q_f16(dst_ptr.add(8), f_set.1);
        vst1q_f16(dst_ptr.add(16), f_set.2);
        vst1q_f16(dst_ptr.add(24), f_set.3);
    }
}

#[inline(always)]
pub(crate) unsafe fn conv_vertical_part_neon_48_f16(
    start_y: usize,
    start_x: usize,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    filter: &[f16],
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
            let v_weight = vreinterpretq_f16_u16(vld1q_dup_u16(
                filter.get_unchecked(j..).as_ptr() as *const _
            ));
            let src_ptr = src.get_unchecked(src_stride * py..).as_ptr();

            let s_ptr = src_ptr.add(px);
            let item_row_0 = xvld1q_u16_x4(s_ptr as *const _);
            let item_row_1 = xvld1q_u16_x2(s_ptr.add(32) as *const _);

            store_0 = vfmlalq_low_f16(store_0, vreinterpretq_f16_u16(item_row_0.0), v_weight);
            store_1 = vfmlalq_high_f16(store_1, vreinterpretq_f16_u16(item_row_0.0), v_weight);
            store_2 = vfmlalq_low_f16(store_2, vreinterpretq_f16_u16(item_row_0.1), v_weight);
            store_3 = vfmlalq_high_f16(store_3, vreinterpretq_f16_u16(item_row_0.1), v_weight);

            store_4 = vfmlalq_low_f16(store_4, vreinterpretq_f16_u16(item_row_0.2), v_weight);
            store_5 = vfmlalq_high_f16(store_5, vreinterpretq_f16_u16(item_row_0.2), v_weight);
            store_6 = vfmlalq_low_f16(store_6, vreinterpretq_f16_u16(item_row_0.3), v_weight);
            store_7 = vfmlalq_high_f16(store_7, vreinterpretq_f16_u16(item_row_0.3), v_weight);

            store_8 = vfmlalq_low_f16(store_8, vreinterpretq_f16_u16(item_row_1.0), v_weight);
            store_9 = vfmlalq_high_f16(store_9, vreinterpretq_f16_u16(item_row_1.0), v_weight);
            store_10 = vfmlalq_low_f16(store_10, vreinterpretq_f16_u16(item_row_1.1), v_weight);
            store_11 = vfmlalq_high_f16(store_11, vreinterpretq_f16_u16(item_row_1.1), v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        let f_set = float16x8x4_t(
            vcombine_f16(vcvt_f16_f32(store_0), vcvt_f16_f32(store_1)),
            vcombine_f16(vcvt_f16_f32(store_2), vcvt_f16_f32(store_3)),
            vcombine_f16(vcvt_f16_f32(store_4), vcvt_f16_f32(store_5)),
            vcombine_f16(vcvt_f16_f32(store_6), vcvt_f16_f32(store_7)),
        );
        vst1q_f16(dst_ptr, f_set.0);
        vst1q_f16(dst_ptr.add(8), f_set.1);
        vst1q_f16(dst_ptr.add(16), f_set.2);
        vst1q_f16(dst_ptr.add(24), f_set.3);

        let dst_ptr2 = dst_ptr.add(32);

        let f_set1 = float16x8x2_t(
            vcombine_f16(vcvt_f16_f32(store_8), vcvt_f16_f32(store_9)),
            vcombine_f16(vcvt_f16_f32(store_10), vcvt_f16_f32(store_11)),
        );
        vst1q_f16(dst_ptr2, f_set1.0);
        vst1q_f16(dst_ptr2.add(8), f_set1.1);
    }
}

pub(crate) fn convolve_vertical_rgb_neon_row_f16_fhm(
    w0: usize,
    bounds: &FilterBounds,
    src: &[f16],
    dst: &mut [f16],
    src_stride: usize,
    weight_ptr: &[f16],
) {
    unsafe { convolve_vertical_rgb_neon_row_f16_impl(w0, bounds, src, dst, src_stride, weight_ptr) }
}

#[inline(always)]
unsafe fn convolve_vertical_part_neon_8_f16_fhm<const USE_BLENDING: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    filter: &[f16],
    bounds: &FilterBounds,
    blend_length: usize,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0f32);
        let mut store_1 = vdupq_n_f32(0f32);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let v_weight = vreinterpretq_f16_u16(vld1q_dup_u16(
                filter.get_unchecked(j..).as_ptr() as *const _
            ));
            let src_ptr = src.get_unchecked(src_stride * py..).as_ptr();

            let s_ptr = src_ptr.add(px);
            let item_row = if USE_BLENDING {
                let mut transient: [f16; 8] = [0.; 8];
                std::ptr::copy_nonoverlapping(s_ptr, transient.as_mut_ptr(), blend_length);
                vld1q_f16(transient.as_ptr())
            } else {
                vld1q_f16(s_ptr)
            };

            store_0 = vfmlalq_low_f16(store_0, item_row, v_weight);
            store_1 = vfmlalq_high_f16(store_1, item_row, v_weight);
        }

        let item = vcombine_f16(vcvt_f16_f32(store_0), vcvt_f16_f32(store_1));

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        if USE_BLENDING {
            let mut transient: [f16; 8] = [0.; 8];
            vst1q_f16(transient.as_mut_ptr(), item);
            std::ptr::copy_nonoverlapping(transient.as_ptr(), dst_ptr, blend_length);
        } else {
            vst1q_f16(dst_ptr, item);
        }
    }
}

#[target_feature(enable = "fhm")]
unsafe fn convolve_vertical_rgb_neon_row_f16_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[f16],
    dst: &mut [f16],
    src_stride: usize,
    weight_ptr: &[f16],
) {
    unsafe {
        let mut cx = 0usize;
        let dst_width = dst.len();

        while cx + 48 < dst_width {
            conv_vertical_part_neon_48_f16(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );

            cx += 48;
        }

        while cx + 32 < dst_width {
            conv_vertical_part_neon_32_f16(
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
            conv_vertical_part_neon_16_f16(
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
            convolve_vertical_part_neon_8_f16_fhm::<false>(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
                8,
            );

            cx += 8;
        }

        let left = dst_width - cx;

        if left > 0 {
            convolve_vertical_part_neon_8_f16_fhm::<true>(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
                left,
            );
        }
    }
}
