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
use crate::neon::utils::{prefer_vfmaq_f32, prefer_vfmaq_lane_f32, prefer_vfmaq_laneq_f32};
use std::arch::aarch64::*;

pub(crate) fn convolve_column_u16(
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
        let mut k_cx = 0usize;

        let bounds_size = bounds.size;

        let zeros = vdupq_n_f32(0.);

        let v_max_colors = vdupq_n_u16(max_colors);

        let v_px = k_cx;

        let iter16 = dst.chunks_exact_mut(16);

        for (x, dst) in iter16.enumerate() {
            let mut store0 = zeros;
            let mut store1 = zeros;
            let mut store2 = zeros;
            let mut store3 = zeros;

            let v_dx = v_px + x * 16;

            if bounds_size == 2 {
                let weights = weight.get_unchecked(0..2);
                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);

                let weights = vld1_f32(weights.as_ptr());

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());
                let item_row1 = vld1q_u16(src_ptr0.as_ptr().add(8));

                let lo0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row0)));
                let hi0 = vcvtq_f32_u32(vmovl_high_u16(item_row0));
                let lo1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row1)));
                let hi1 = vcvtq_f32_u32(vmovl_high_u16(item_row1));
                store0 = prefer_vfmaq_lane_f32::<0>(store0, lo0, weights);
                store1 = prefer_vfmaq_lane_f32::<0>(store1, hi0, weights);
                store2 = prefer_vfmaq_lane_f32::<0>(store2, lo1, weights);
                store3 = prefer_vfmaq_lane_f32::<0>(store3, hi1, weights);

                let item_row10 = vld1q_u16(src_ptr1.as_ptr());
                let item_row11 = vld1q_u16(src_ptr1.as_ptr().add(8));

                let lo10 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row10)));
                let hi10 = vcvtq_f32_u32(vmovl_high_u16(item_row10));
                let lo11 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row11)));
                let hi11 = vcvtq_f32_u32(vmovl_high_u16(item_row11));
                store0 = prefer_vfmaq_lane_f32::<1>(store0, lo10, weights);
                store1 = prefer_vfmaq_lane_f32::<1>(store1, hi10, weights);
                store2 = prefer_vfmaq_lane_f32::<1>(store2, lo11, weights);
                store3 = prefer_vfmaq_lane_f32::<1>(store3, hi11, weights);
            } else if bounds_size == 3 {
                let weights = weight.get_unchecked(0..3);
                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);

                let v_weight0 = vdupq_n_f32(weights[0]);
                let v_weight1 = vdupq_n_f32(weights[1]);
                let v_weight2 = vdupq_n_f32(weights[2]);

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());
                let item_row1 = vld1q_u16(src_ptr0.as_ptr().add(8));

                let lo0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row0)));
                let hi0 = vcvtq_f32_u32(vmovl_high_u16(item_row0));
                let lo1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row1)));
                let hi1 = vcvtq_f32_u32(vmovl_high_u16(item_row1));
                store0 = prefer_vfmaq_f32(store0, lo0, v_weight0);
                store1 = prefer_vfmaq_f32(store1, hi0, v_weight0);
                store2 = prefer_vfmaq_f32(store2, lo1, v_weight0);
                store3 = prefer_vfmaq_f32(store3, hi1, v_weight0);

                let item_row10 = vld1q_u16(src_ptr1.as_ptr());
                let item_row11 = vld1q_u16(src_ptr1.as_ptr().add(8));

                let lo10 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row10)));
                let hi10 = vcvtq_f32_u32(vmovl_high_u16(item_row10));
                let lo11 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row11)));
                let hi11 = vcvtq_f32_u32(vmovl_high_u16(item_row11));
                store0 = prefer_vfmaq_f32(store0, lo10, v_weight1);
                store1 = prefer_vfmaq_f32(store1, hi10, v_weight1);
                store2 = prefer_vfmaq_f32(store2, lo11, v_weight1);
                store3 = prefer_vfmaq_f32(store3, hi11, v_weight1);

                let item_row20 = vld1q_u16(src_ptr2.as_ptr());
                let item_row21 = vld1q_u16(src_ptr2.as_ptr().add(8));

                let lo20 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row20)));
                let hi20 = vcvtq_f32_u32(vmovl_high_u16(item_row20));
                let lo21 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row21)));
                let hi21 = vcvtq_f32_u32(vmovl_high_u16(item_row21));
                store0 = prefer_vfmaq_f32(store0, lo20, v_weight2);
                store1 = prefer_vfmaq_f32(store1, hi20, v_weight2);
                store2 = prefer_vfmaq_f32(store2, lo21, v_weight2);
                store3 = prefer_vfmaq_f32(store3, hi21, v_weight2);
            } else if bounds_size == 4 {
                let weights = weight.get_unchecked(0..4);
                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + v_dx)..);

                let weights = vld1q_f32(weights.as_ptr());

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());
                let item_row1 = vld1q_u16(src_ptr0.as_ptr().add(8));

                let lo0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row0)));
                let hi0 = vcvtq_f32_u32(vmovl_high_u16(item_row0));
                let lo1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row1)));
                let hi1 = vcvtq_f32_u32(vmovl_high_u16(item_row1));
                store0 = prefer_vfmaq_laneq_f32::<0>(store0, lo0, weights);
                store1 = prefer_vfmaq_laneq_f32::<0>(store1, hi0, weights);
                store2 = prefer_vfmaq_laneq_f32::<0>(store2, lo1, weights);
                store3 = prefer_vfmaq_laneq_f32::<0>(store3, hi1, weights);

                let item_row10 = vld1q_u16(src_ptr1.as_ptr());
                let item_row11 = vld1q_u16(src_ptr1.as_ptr().add(8));

                let lo10 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row10)));
                let hi10 = vcvtq_f32_u32(vmovl_high_u16(item_row10));
                let lo11 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row11)));
                let hi11 = vcvtq_f32_u32(vmovl_high_u16(item_row11));
                store0 = prefer_vfmaq_laneq_f32::<1>(store0, lo10, weights);
                store1 = prefer_vfmaq_laneq_f32::<1>(store1, hi10, weights);
                store2 = prefer_vfmaq_laneq_f32::<1>(store2, lo11, weights);
                store3 = prefer_vfmaq_laneq_f32::<1>(store3, hi11, weights);

                let item_row20 = vld1q_u16(src_ptr2.as_ptr());
                let item_row21 = vld1q_u16(src_ptr2.as_ptr().add(8));

                let lo20 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row20)));
                let hi20 = vcvtq_f32_u32(vmovl_high_u16(item_row20));
                let lo21 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row21)));
                let hi21 = vcvtq_f32_u32(vmovl_high_u16(item_row21));
                store0 = prefer_vfmaq_laneq_f32::<2>(store0, lo20, weights);
                store1 = prefer_vfmaq_laneq_f32::<2>(store1, hi20, weights);
                store2 = prefer_vfmaq_laneq_f32::<2>(store2, lo21, weights);
                store3 = prefer_vfmaq_laneq_f32::<2>(store3, hi21, weights);

                let item_row30 = vld1q_u16(src_ptr3.as_ptr());
                let item_row31 = vld1q_u16(src_ptr3.as_ptr().add(8));

                let lo30 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row30)));
                let hi30 = vcvtq_f32_u32(vmovl_high_u16(item_row30));
                let lo31 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row31)));
                let hi31 = vcvtq_f32_u32(vmovl_high_u16(item_row31));
                store0 = prefer_vfmaq_laneq_f32::<3>(store0, lo30, weights);
                store1 = prefer_vfmaq_laneq_f32::<3>(store1, hi30, weights);
                store2 = prefer_vfmaq_laneq_f32::<3>(store2, lo31, weights);
                store3 = prefer_vfmaq_laneq_f32::<3>(store3, hi31, weights);
            } else {
                for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                    let py = bounds.start + j;
                    let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

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
            }
            let u_store0 = vcvtaq_u32_f32(store0);
            let u_store1 = vcvtaq_u32_f32(store1);
            let u_store2 = vcvtaq_u32_f32(store2);
            let u_store3 = vcvtaq_u32_f32(store3);

            let item0 = vminq_u16(
                vcombine_u16(vqmovn_u32(u_store0), vqmovn_u32(u_store1)),
                v_max_colors,
            );
            vst1q_u16(dst.as_mut_ptr(), item0);
            let item1 = vminq_u16(
                vcombine_u16(vqmovn_u32(u_store2), vqmovn_u32(u_store3)),
                v_max_colors,
            );
            vst1q_u16(dst.as_mut_ptr().add(8), item1);

            k_cx = v_dx;
        }

        let tail16 = dst.chunks_exact_mut(16).into_remainder();
        let iter8 = tail16.chunks_exact_mut(8);

        let v_px = k_cx;

        for (x, dst) in iter8.enumerate() {
            let mut store0 = zeros;
            let mut store1 = zeros;

            let v_dx = v_px + x * 8;

            if bounds_size == 2 {
                let weights = weight.get_unchecked(0..2);
                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);

                let weights = vld1_f32(weights.as_ptr());

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());
                let item_row1 = vld1q_u16(src_ptr1.as_ptr());

                let lo0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row0)));
                let hi0 = vcvtq_f32_u32(vmovl_high_u16(item_row0));
                store0 = prefer_vfmaq_lane_f32::<0>(store0, lo0, weights);
                store1 = prefer_vfmaq_lane_f32::<0>(store1, hi0, weights);

                let lo1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row1)));
                let hi1 = vcvtq_f32_u32(vmovl_high_u16(item_row1));
                store0 = prefer_vfmaq_lane_f32::<1>(store0, lo1, weights);
                store1 = prefer_vfmaq_lane_f32::<1>(store1, hi1, weights);
            } else if bounds_size == 3 {
                let weights = weight.get_unchecked(0..3);
                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);

                let v_weight0 = vdupq_n_f32(weights[0]);
                let v_weight1 = vdupq_n_f32(weights[1]);
                let v_weight2 = vdupq_n_f32(weights[2]);

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());
                let item_row1 = vld1q_u16(src_ptr1.as_ptr());
                let item_row2 = vld1q_u16(src_ptr2.as_ptr());

                let lo0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row0)));
                let hi0 = vcvtq_f32_u32(vmovl_high_u16(item_row0));
                store0 = prefer_vfmaq_f32(store0, lo0, v_weight0);
                store1 = prefer_vfmaq_f32(store1, hi0, v_weight0);

                let lo1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row1)));
                let hi1 = vcvtq_f32_u32(vmovl_high_u16(item_row1));
                store0 = prefer_vfmaq_f32(store0, lo1, v_weight1);
                store1 = prefer_vfmaq_f32(store1, hi1, v_weight1);

                let lo2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row2)));
                let hi2 = vcvtq_f32_u32(vmovl_high_u16(item_row2));
                store0 = prefer_vfmaq_f32(store0, lo2, v_weight2);
                store1 = prefer_vfmaq_f32(store1, hi2, v_weight2);
            } else if bounds_size == 4 {
                let weights = weight.get_unchecked(0..4);
                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + v_dx)..);

                let weights = vld1q_f32(weights.as_ptr());

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());
                let item_row1 = vld1q_u16(src_ptr1.as_ptr());
                let item_row2 = vld1q_u16(src_ptr2.as_ptr());
                let item_row3 = vld1q_u16(src_ptr3.as_ptr());

                let lo0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row0)));
                let hi0 = vcvtq_f32_u32(vmovl_high_u16(item_row0));
                store0 = prefer_vfmaq_laneq_f32::<0>(store0, lo0, weights);
                store1 = prefer_vfmaq_laneq_f32::<0>(store1, hi0, weights);

                let lo1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row1)));
                let hi1 = vcvtq_f32_u32(vmovl_high_u16(item_row1));
                store0 = prefer_vfmaq_laneq_f32::<1>(store0, lo1, weights);
                store1 = prefer_vfmaq_laneq_f32::<1>(store1, hi1, weights);

                let lo2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row2)));
                let hi2 = vcvtq_f32_u32(vmovl_high_u16(item_row2));
                store0 = prefer_vfmaq_laneq_f32::<2>(store0, lo2, weights);
                store1 = prefer_vfmaq_laneq_f32::<2>(store1, hi2, weights);

                let lo3 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row3)));
                let hi3 = vcvtq_f32_u32(vmovl_high_u16(item_row3));
                store0 = prefer_vfmaq_laneq_f32::<3>(store0, lo3, weights);
                store1 = prefer_vfmaq_laneq_f32::<3>(store1, hi3, weights);
            } else {
                for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                    let py = bounds.start + j;
                    let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                    let v_weight = vdupq_n_f32(k_weight);

                    let item_row = vld1q_u16(src_ptr.as_ptr());

                    let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(item_row)));
                    let hi = vcvtq_f32_u32(vmovl_high_u16(item_row));
                    store0 = prefer_vfmaq_f32(store0, lo, v_weight);
                    store1 = prefer_vfmaq_f32(store1, hi, v_weight);
                }
            }

            let u_store0 = vcvtaq_u32_f32(store0);
            let u_store1 = vcvtaq_u32_f32(store1);

            let item = vminq_u16(
                vcombine_u16(vqmovn_u32(u_store0), vqmovn_u32(u_store1)),
                v_max_colors,
            );
            vst1q_u16(dst.as_mut_ptr(), item);

            k_cx = v_dx;
        }

        let tail8 = tail16.chunks_exact_mut(8).into_remainder();
        let iter4 = tail8.chunks_exact_mut(4);

        let v_cx = k_cx;

        for (x, dst) in iter4.enumerate() {
            let mut store0 = zeros;

            let v_dx = v_cx + x * 4;

            if bounds_size == 2 {
                let weights = weight.get_unchecked(0..2);
                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);

                let weights = vld1_f32(weights.as_ptr());

                let item_row0 = vld1_u16(src_ptr0.as_ptr());
                let item_row1 = vld1_u16(src_ptr1.as_ptr());

                let lo0 = vcvtq_f32_u32(vmovl_u16(item_row0));
                let lo1 = vcvtq_f32_u32(vmovl_u16(item_row1));
                store0 = prefer_vfmaq_lane_f32::<0>(store0, lo0, weights);
                store0 = prefer_vfmaq_lane_f32::<1>(store0, lo1, weights);
            } else if bounds_size == 3 {
                let weights = weight.get_unchecked(0..3);
                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);

                let v_weight0 = vld1q_dup_f32(weights.as_ptr());
                let v_weight1 = vld1q_dup_f32(weights.as_ptr().add(1));
                let v_weight2 = vld1q_dup_f32(weights.as_ptr().add(2));

                let item_row0 = vld1_u16(src_ptr0.as_ptr());
                let item_row1 = vld1_u16(src_ptr1.as_ptr());
                let item_row2 = vld1_u16(src_ptr2.as_ptr());

                let lo0 = vcvtq_f32_u32(vmovl_u16(item_row0));
                let lo1 = vcvtq_f32_u32(vmovl_u16(item_row1));
                let lo2 = vcvtq_f32_u32(vmovl_u16(item_row2));
                store0 = prefer_vfmaq_f32(store0, lo0, v_weight0);
                store0 = prefer_vfmaq_f32(store0, lo1, v_weight1);
                store0 = prefer_vfmaq_f32(store0, lo2, v_weight2);
            } else if bounds_size == 4 {
                let weights = weight.get_unchecked(0..4);
                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + v_dx)..);

                let weights = vld1q_f32(weights.as_ptr());

                let item_row0 = vld1_u16(src_ptr0.as_ptr());
                let item_row1 = vld1_u16(src_ptr1.as_ptr());
                let item_row2 = vld1_u16(src_ptr2.as_ptr());
                let item_row3 = vld1_u16(src_ptr3.as_ptr());

                let lo0 = vcvtq_f32_u32(vmovl_u16(item_row0));
                let lo1 = vcvtq_f32_u32(vmovl_u16(item_row1));
                let lo2 = vcvtq_f32_u32(vmovl_u16(item_row2));
                let lo3 = vcvtq_f32_u32(vmovl_u16(item_row3));

                store0 = prefer_vfmaq_laneq_f32::<0>(store0, lo0, weights);
                store0 = prefer_vfmaq_laneq_f32::<1>(store0, lo1, weights);
                store0 = prefer_vfmaq_laneq_f32::<2>(store0, lo2, weights);
                store0 = prefer_vfmaq_laneq_f32::<3>(store0, lo3, weights);
            } else {
                for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                    let py = bounds.start + j;
                    let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                    let v_weight = vdupq_n_f32(k_weight);

                    let item_row = vld1_u16(src_ptr.as_ptr());

                    let lo = vcvtq_f32_u32(vmovl_u16(item_row));
                    store0 = prefer_vfmaq_f32(store0, lo, v_weight);
                }
            }

            let u_store0 = vcvtaq_u32_f32(store0);

            vst1_u16(
                dst.as_mut_ptr(),
                vmin_u16(vqmovn_u32(u_store0), vget_low_u16(v_max_colors)),
            );

            k_cx = v_dx;
        }

        let tail4 = tail8.chunks_exact_mut(4).into_remainder();

        let a_px = k_cx;

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = 0.;

            if bounds_size == 2 {
                let weights = weight.get_unchecked(0..2);
                let weight0 = weights[0];
                let weight1 = weights[1];
                let py = bounds.start;
                let v_px = a_px + x;
                let offset0 = src_stride * py + v_px;
                let offset1 = src_stride * (py + 1) + v_px;
                let src_ptr0 = *src.get_unchecked(offset0);
                let src_ptr1 = *src.get_unchecked(offset1);

                store0 = mlaf(store0, src_ptr0 as f32, weight0);
                store0 = mlaf(store0, src_ptr1 as f32, weight1);
            } else if bounds_size == 3 {
                let weights = weight.get_unchecked(0..3);
                let weight0 = weights[0];
                let weight1 = weights[1];
                let weight2 = weights[2];
                let py = bounds.start;
                let v_px = a_px + x;
                let offset0 = src_stride * py + v_px;
                let offset1 = src_stride * (py + 1) + v_px;
                let offset2 = src_stride * (py + 2) + v_px;
                let src_ptr0 = *src.get_unchecked(offset0);
                let src_ptr1 = *src.get_unchecked(offset1);
                let src_ptr2 = *src.get_unchecked(offset2);

                store0 = mlaf(store0, src_ptr0 as f32, weight0);
                store0 = mlaf(store0, src_ptr1 as f32, weight1);
                store0 = mlaf(store0, src_ptr2 as f32, weight2);
            } else if bounds_size == 4 {
                let weights = weight.get_unchecked(0..4);
                let weight0 = weights[0];
                let weight1 = weights[1];
                let weight2 = weights[2];
                let weight3 = weights[3];
                let py = bounds.start;
                let v_px = a_px + x;
                let offset0 = src_stride * py + v_px;
                let offset1 = src_stride * (py + 1) + v_px;
                let offset2 = src_stride * (py + 2) + v_px;
                let offset3 = src_stride * (py + 3) + v_px;
                let src_ptr0 = *src.get_unchecked(offset0);
                let src_ptr1 = *src.get_unchecked(offset1);
                let src_ptr2 = *src.get_unchecked(offset2);
                let src_ptr3 = *src.get_unchecked(offset3);

                store0 = mlaf(store0, src_ptr0 as f32, weight0);
                store0 = mlaf(store0, src_ptr1 as f32, weight1);
                store0 = mlaf(store0, src_ptr2 as f32, weight2);
                store0 = mlaf(store0, src_ptr3 as f32, weight3);
            } else {
                for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                    let py = bounds.start + j;
                    let offset = src_stride * py + a_px + x;
                    let src_ptr = *src.get_unchecked(offset);

                    store0 = mlaf(store0, src_ptr as f32, k_weight);
                }
            }

            *dst = store0.round().max(0.).min(max_colors as f32) as u16;
        }
    }
}
