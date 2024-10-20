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
use crate::mlaf::mlaf;
use crate::neon::utils::prefer_vfmaq_f32;
use std::arch::aarch64::*;

#[inline(always)]
pub fn convolve_column_u16<const CHANNELS: usize>(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[f32],
    bit_depth: u32,
) {
    unsafe {
        let max_colors = (1 << bit_depth) - 1;
        let mut cx = 0usize;

        let zeros = vdupq_n_f32(0.);

        let v_max_colors = vdupq_n_u32(max_colors);

        let v_px = cx;

        let iter16 = dst.chunks_exact_mut(16);

        for (x, dst) in iter16.enumerate() {
            let mut store0 = zeros;
            let mut store1 = zeros;
            let mut store2 = zeros;
            let mut store3 = zeros;

            for (j, &k_weight) in weight.iter().take(bounds.size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + cx;
                let src_ptr = src.get_unchecked(offset..);

                let v_weight = vdupq_n_f32(k_weight);

                let item_row0 = vld1q_u16(src_ptr.as_ptr());
                let item_row1 = vld1q_u16(src_ptr.as_ptr().add(8));

                let lo0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row0)));
                let hi0 = vcvtq_f32_u32(vmovl_high_u16(item_row0));
                let lo1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row1)));
                let hi1 = vcvtq_f32_u32(vmovl_high_u16(item_row1));
                store0 = prefer_vfmaq_f32(store0, lo0, v_weight);
                store1 = prefer_vfmaq_f32(store1, hi0, v_weight);
                store2 = prefer_vfmaq_f32(store2, lo1, v_weight);
                store3 = prefer_vfmaq_f32(store3, hi1, v_weight);
            }

            let u_store0 = vminq_u32(vcvtaq_u32_f32(vmaxq_f32(store0, zeros)), v_max_colors);
            let u_store1 = vminq_u32(vcvtaq_u32_f32(vmaxq_f32(store1, zeros)), v_max_colors);
            let u_store2 = vminq_u32(vcvtaq_u32_f32(vmaxq_f32(store2, zeros)), v_max_colors);
            let u_store3 = vminq_u32(vcvtaq_u32_f32(vmaxq_f32(store3, zeros)), v_max_colors);

            let item0 = vcombine_u16(vqmovn_u32(u_store0), vqmovn_u32(u_store1));
            vst1q_u16(dst.as_mut_ptr(), item0);
            let item1 = vcombine_u16(vqmovn_u32(u_store2), vqmovn_u32(u_store3));
            vst1q_u16(dst.as_mut_ptr().add(8), item1);

            cx = v_px + x * 16;
        }

        let tail16 = dst.chunks_exact_mut(16).into_remainder();
        let iter8 = tail16.chunks_exact_mut(8);

        let v_px = cx;

        for (x, dst) in iter8.enumerate() {
            let mut store0 = zeros;
            let mut store1 = zeros;

            for (j, &k_weight) in weight.iter().take(bounds.size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + cx;
                let src_ptr = src.get_unchecked(offset..);

                let v_weight = vdupq_n_f32(k_weight);

                let item_row = vld1q_u16(src_ptr.as_ptr());

                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row)));
                let hi = vcvtq_f32_u32(vmovl_high_u16(item_row));
                store0 = prefer_vfmaq_f32(store0, lo, v_weight);
                store1 = prefer_vfmaq_f32(store1, hi, v_weight);
            }

            let u_store0 = vminq_u32(vcvtaq_u32_f32(vmaxq_f32(store0, zeros)), v_max_colors);
            let u_store1 = vminq_u32(vcvtaq_u32_f32(vmaxq_f32(store1, zeros)), v_max_colors);

            let item = vcombine_u16(vqmovn_u32(u_store0), vqmovn_u32(u_store1));
            vst1q_u16(dst.as_mut_ptr(), item);

            cx = v_px + x * 8;
        }

        let tail8 = tail16.chunks_exact_mut(8).into_remainder();
        let iter4 = tail8.chunks_exact_mut(4);

        let v_cx = cx;

        for (x, dst) in iter4.enumerate() {
            let mut store0 = zeros;

            for (j, &k_weight) in weight.iter().take(bounds.size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + cx;
                let src_ptr = src.get_unchecked(offset..);

                let v_weight = vdupq_n_f32(k_weight);

                let item_row = vld1_u16(src_ptr.as_ptr());

                let lo = vcvtq_f32_u32(vmovl_u16(item_row));
                store0 = prefer_vfmaq_f32(store0, lo, v_weight);
            }

            let u_store0 = vminq_u32(vcvtaq_u32_f32(vmaxq_f32(store0, zeros)), v_max_colors);

            vst1_u16(dst.as_mut_ptr(), vqmovn_u32(u_store0));

            cx = v_cx + x * 4;
        }

        let tail4 = tail8.chunks_exact_mut(4).into_remainder();

        let mut a_px = cx;

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = 0.;

            for (j, &k_weight) in weight.iter().take(bounds.size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + a_px;
                let src_ptr = src.get_unchecked(offset..(offset + 1));

                store0 = mlaf(store0, src_ptr[0] as f32, k_weight);
            }

            *dst = store0.round().max(0.).min(max_colors as f32) as u16;

            a_px += x;
        }
    }
}
