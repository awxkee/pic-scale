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
use crate::neon::convolve_f16::convolve_vertical_part_neon_8_f16;
use crate::neon::utils::{prefer_vfmaq_f32, xvld1q_u16_x2, xvld1q_u16_x4};
use core::f16;
use std::arch::aarch64::*;

#[inline(always)]
#[cfg(feature = "nightly_f16")]
pub(crate) unsafe fn xvst1q_u16_x2(ptr: *mut u16, x: uint16x8x2_t) {
    unsafe {
        let ptr_u16 = ptr;
        vst1q_u16(ptr_u16, x.0);
        vst1q_u16(ptr_u16.add(8), x.1);
    }
}

#[inline(always)]
#[cfg(feature = "nightly_f16")]
pub(crate) unsafe fn xvst1q_u16_x4(ptr: *const f16, x: uint16x8x4_t) {
    unsafe {
        let ptr_u16 = ptr as *mut u16;
        vst1q_u16(ptr_u16, x.0);
        vst1q_u16(ptr_u16.add(8), x.1);
        vst1q_u16(ptr_u16.add(16), x.2);
        vst1q_u16(ptr_u16.add(24), x.3);
    }
}

macro_rules! conv_vertical_part_neon_16_f16 {
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
                let src_ptr = $src.get_unchecked($src_stride * py..).as_ptr();

                let s_ptr = src_ptr.add(px);
                let item_row = xvld1q_u16_x2(s_ptr as *const _);

                store_0 = prefer_vfmaq_f32(
                    store_0,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(item_row.0))),
                    v_weight,
                );
                store_1 = prefer_vfmaq_f32(
                    store_1,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(item_row.0))),
                    v_weight,
                );
                store_2 = prefer_vfmaq_f32(
                    store_2,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(item_row.1))),
                    v_weight,
                );
                store_3 = prefer_vfmaq_f32(
                    store_3,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(item_row.1))),
                    v_weight,
                );
            }

            let dst_ptr = $dst.get_unchecked_mut(px..).as_mut_ptr();
            let f_set = uint16x8x2_t(
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(store_0)),
                    vreinterpret_u16_f16(vcvt_f16_f32(store_1)),
                ),
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(store_2)),
                    vreinterpret_u16_f16(vcvt_f16_f32(store_3)),
                ),
            );
            xvst1q_u16_x2(dst_ptr as *mut _, f_set);
        }
    }};
}

macro_rules! conv_vertical_part_neon_32_f16 {
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
                let src_ptr = $src.get_unchecked($src_stride * py..).as_ptr();

                let s_ptr = src_ptr.add(px);
                let item_row = xvld1q_u16_x4(s_ptr as *const _);

                store_0 = prefer_vfmaq_f32(
                    store_0,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(item_row.0))),
                    v_weight,
                );
                store_1 = prefer_vfmaq_f32(
                    store_1,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(item_row.0))),
                    v_weight,
                );
                store_2 = prefer_vfmaq_f32(
                    store_2,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(item_row.1))),
                    v_weight,
                );
                store_3 = prefer_vfmaq_f32(
                    store_3,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(item_row.1))),
                    v_weight,
                );

                store_4 = prefer_vfmaq_f32(
                    store_4,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(item_row.2))),
                    v_weight,
                );
                store_5 = prefer_vfmaq_f32(
                    store_5,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(item_row.2))),
                    v_weight,
                );
                store_6 = prefer_vfmaq_f32(
                    store_6,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(item_row.3))),
                    v_weight,
                );
                store_7 = prefer_vfmaq_f32(
                    store_7,
                    vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(item_row.3))),
                    v_weight,
                );
            }

            let dst_ptr = $dst.get_unchecked_mut(px..).as_mut_ptr();
            let f_set = uint16x8x4_t(
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(store_0)),
                    vreinterpret_u16_f16(vcvt_f16_f32(store_1)),
                ),
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(store_2)),
                    vreinterpret_u16_f16(vcvt_f16_f32(store_3)),
                ),
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(store_4)),
                    vreinterpret_u16_f16(vcvt_f16_f32(store_5)),
                ),
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(store_6)),
                    vreinterpret_u16_f16(vcvt_f16_f32(store_7)),
                ),
            );
            xvst1q_u16_x4(dst_ptr as *mut _, f_set);
        }
    }};
}

pub(crate) fn convolve_vertical_rgb_neon_row_f16(
    _: usize,
    bounds: &FilterBounds,
    src: &[f16],
    dst: &mut [f16],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    let mut cx = 0usize;
    let dst_width = dst.len();

    while cx + 32 < dst_width {
        conv_vertical_part_neon_32_f16!(bounds.start, cx, src, src_stride, dst, weight_ptr, bounds);

        cx += 32;
    }

    while cx + 16 < dst_width {
        conv_vertical_part_neon_16_f16!(bounds.start, cx, src, src_stride, dst, weight_ptr, bounds);

        cx += 16;
    }

    while cx + 8 < dst_width {
        unsafe {
            convolve_vertical_part_neon_8_f16::<false>(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
                8,
            );
        }

        cx += 8;
    }

    let left = dst_width - cx;

    if left > 0 {
        unsafe {
            convolve_vertical_part_neon_8_f16::<true>(
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
