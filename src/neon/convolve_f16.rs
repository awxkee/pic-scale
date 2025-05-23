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
use crate::neon::utils::prefer_vfmaq_f32;
use core::f16;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_neon_8_f16<const USE_BLENDING: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    filter: &[f32],
    bounds: &FilterBounds,
    blend_length: usize,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);
        let mut store_1 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1q_dup_f32(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py..).as_ptr();

            let s_ptr = src_ptr.add(px);
            let item_row = if USE_BLENDING {
                let mut transient: [u16; 8] = [0; 8];
                std::ptr::copy_nonoverlapping(
                    s_ptr as *const _,
                    transient.as_mut_ptr(),
                    blend_length,
                );
                vld1q_u16(transient.as_ptr())
            } else {
                vld1q_u16(s_ptr as *const _)
            };

            let p1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(item_row)));
            let p2 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(item_row)));

            store_0 = prefer_vfmaq_f32(store_0, p1, v_weight);
            store_1 = prefer_vfmaq_f32(store_1, p2, v_weight);
        }

        let item = vcombine_u16(
            vreinterpret_u16_f16(vcvt_f16_f32(store_0)),
            vreinterpret_u16_f16(vcvt_f16_f32(store_1)),
        );

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        if USE_BLENDING {
            let mut transient: [u16; 8] = [0; 8];
            vst1q_u16(transient.as_mut_ptr(), item);
            std::ptr::copy_nonoverlapping(transient.as_ptr(), dst_ptr as *mut _, blend_length);
        } else {
            vst1q_u16(dst_ptr as *mut _, item);
        }
    }
}
