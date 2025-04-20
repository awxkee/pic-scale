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
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) fn convolve_column_hb_u16(
    w: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i32],
    bit_depth: u32,
) {
    unsafe { convolve_column_hb_impl(w, bounds, src, dst, src_stride, weight, bit_depth) }
}

#[target_feature(enable = "rdm")]
unsafe fn convolve_column_hb_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i32],
    bit_depth: u32,
) {
    unsafe {
        let max_colors = (1u32 << bit_depth) - 1;
        let mut cx = 0usize;

        let bounds_size = bounds.size;

        let initial_store = vdupq_n_s32(0);

        let v_max_colors = vdupq_n_u16(max_colors as u16);

        let v_px = cx;

        let iter16 = dst.chunks_exact_mut(16);

        for (x, dst) in iter16.enumerate() {
            let mut store0 = initial_store;
            let mut store1 = initial_store;
            let mut store2 = initial_store;
            let mut store3 = initial_store;

            let v_dx = v_px + x * 16;

            if bounds_size == 2 {
                let weights = weight.get_unchecked(0..2);
                let mut v_weight = vld1_dup_s32(weights.as_ptr());
                v_weight = vld1_lane_s32::<1>(weights.as_ptr().add(1), v_weight);

                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());
                let item_row1 = vld1q_u16(src_ptr0.as_ptr().add(8));

                store0 = vqrdmlahq_lane_s32::<0>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row0))),
                    v_weight,
                );
                store1 = vqrdmlahq_lane_s32::<0>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row0)),
                    v_weight,
                );
                store2 = vqrdmlahq_lane_s32::<0>(
                    store2,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row1))),
                    v_weight,
                );
                store3 = vqrdmlahq_lane_s32::<0>(
                    store3,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row1)),
                    v_weight,
                );

                let item_row10 = vld1q_u16(src_ptr1.as_ptr());
                let item_row11 = vld1q_u16(src_ptr1.as_ptr().add(8));

                store0 = vqrdmlahq_lane_s32::<1>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row10))),
                    v_weight,
                );
                store1 = vqrdmlahq_lane_s32::<1>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row10)),
                    v_weight,
                );
                store2 = vqrdmlahq_lane_s32::<1>(
                    store2,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row11))),
                    v_weight,
                );
                store3 = vqrdmlahq_lane_s32::<1>(
                    store3,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row11)),
                    v_weight,
                );
            } else if bounds_size == 3 {
                let weights = weight.get_unchecked(0..3);
                let mut v_weight = vld1q_dup_s32(weights.as_ptr());
                v_weight = vld1q_lane_s32::<1>(weights.as_ptr().add(1), v_weight);
                v_weight = vld1q_lane_s32::<2>(weights.as_ptr().add(2), v_weight);

                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());
                let item_row1 = vld1q_u16(src_ptr0.as_ptr().add(8));

                store0 = vqrdmlahq_laneq_s32::<0>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row0))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<0>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row0)),
                    v_weight,
                );
                store2 = vqrdmlahq_laneq_s32::<0>(
                    store2,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row1))),
                    v_weight,
                );
                store3 = vqrdmlahq_laneq_s32::<0>(
                    store3,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row1)),
                    v_weight,
                );

                let item_row10 = vld1q_u16(src_ptr1.as_ptr());
                let item_row11 = vld1q_u16(src_ptr1.as_ptr().add(8));

                store0 = vqrdmlahq_laneq_s32::<1>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row10))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<1>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row10)),
                    v_weight,
                );
                store2 = vqrdmlahq_laneq_s32::<1>(
                    store2,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row11))),
                    v_weight,
                );
                store3 = vqrdmlahq_laneq_s32::<1>(
                    store3,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row11)),
                    v_weight,
                );

                let item_row20 = vld1q_u16(src_ptr2.as_ptr());
                let item_row21 = vld1q_u16(src_ptr2.as_ptr().add(8));

                store0 = vqrdmlahq_laneq_s32::<2>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row20))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<2>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row20)),
                    v_weight,
                );
                store2 = vqrdmlahq_laneq_s32::<2>(
                    store2,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row21))),
                    v_weight,
                );
                store3 = vqrdmlahq_laneq_s32::<2>(
                    store3,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row21)),
                    v_weight,
                );
            } else if bounds_size == 4 {
                let weights = weight.get_unchecked(0..4);

                let v_weight = vld1q_s32(weights.as_ptr());

                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + v_dx)..);

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());
                let item_row1 = vld1q_u16(src_ptr0.as_ptr().add(8));

                store0 = vqrdmlahq_laneq_s32::<0>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row0))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<0>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row0)),
                    v_weight,
                );
                store2 = vqrdmlahq_laneq_s32::<0>(
                    store2,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row1))),
                    v_weight,
                );
                store3 = vqrdmlahq_laneq_s32::<0>(
                    store3,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row1)),
                    v_weight,
                );

                let item_row10 = vld1q_u16(src_ptr1.as_ptr());
                let item_row11 = vld1q_u16(src_ptr1.as_ptr().add(8));

                store0 = vqrdmlahq_laneq_s32::<1>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row10))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<1>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row10)),
                    v_weight,
                );
                store2 = vqrdmlahq_laneq_s32::<1>(
                    store2,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row11))),
                    v_weight,
                );
                store3 = vqrdmlahq_laneq_s32::<1>(
                    store3,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row11)),
                    v_weight,
                );

                let item_row20 = vld1q_u16(src_ptr2.as_ptr());
                let item_row21 = vld1q_u16(src_ptr2.as_ptr().add(8));

                store0 = vqrdmlahq_laneq_s32::<2>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row20))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<2>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row20)),
                    v_weight,
                );
                store2 = vqrdmlahq_laneq_s32::<2>(
                    store2,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row21))),
                    v_weight,
                );
                store3 = vqrdmlahq_laneq_s32::<2>(
                    store3,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row21)),
                    v_weight,
                );

                let item_row30 = vld1q_u16(src_ptr3.as_ptr());
                let item_row31 = vld1q_u16(src_ptr3.as_ptr().add(8));

                store0 = vqrdmlahq_laneq_s32::<3>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row30))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<3>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row30)),
                    v_weight,
                );
                store2 = vqrdmlahq_laneq_s32::<3>(
                    store2,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row31))),
                    v_weight,
                );
                store3 = vqrdmlahq_laneq_s32::<3>(
                    store3,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row31)),
                    v_weight,
                );
            } else {
                for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                    let py = bounds.start + j;
                    let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                    let v_weight = vdupq_n_s32(k_weight);

                    let item_row0 = vld1q_u16(src_ptr.as_ptr());
                    let item_row1 = vld1q_u16(src_ptr.as_ptr().add(8));

                    store0 = vqrdmlahq_s32(
                        store0,
                        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row0))),
                        v_weight,
                    );
                    store1 = vqrdmlahq_s32(
                        store1,
                        vreinterpretq_s32_u32(vmovl_high_u16(item_row0)),
                        v_weight,
                    );
                    store2 = vqrdmlahq_s32(
                        store2,
                        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row1))),
                        v_weight,
                    );
                    store3 = vqrdmlahq_s32(
                        store3,
                        vreinterpretq_s32_u32(vmovl_high_u16(item_row1)),
                        v_weight,
                    );
                }
            }

            let store0 = vqmovun_s32(store0);
            let store1 = vqmovun_s32(store1);
            let store2 = vqmovun_s32(store2);
            let store3 = vqmovun_s32(store3);

            let item0 = vminq_u16(vcombine_u16(store0, store1), v_max_colors);
            let item1 = vminq_u16(vcombine_u16(store2, store3), v_max_colors);

            vst1q_u16(dst.as_mut_ptr(), item0);
            vst1q_u16(dst.as_mut_ptr().add(8), item1);

            cx = v_dx;
        }

        let tail16 = dst.chunks_exact_mut(16).into_remainder();
        let iter8 = tail16.chunks_exact_mut(8);

        let v_px = cx;

        for (x, dst) in iter8.enumerate() {
            let mut store0 = initial_store;
            let mut store1 = initial_store;

            let v_dx = v_px + x * 8;

            if bounds_size == 2 {
                let weights = weight.get_unchecked(0..2);
                let mut v_weight = vld1_dup_s32(weights.as_ptr());
                v_weight = vld1_lane_s32::<1>(weights.as_ptr().add(1), v_weight);

                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());

                store0 = vqrdmlahq_lane_s32::<0>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row0))),
                    v_weight,
                );
                store1 = vqrdmlahq_lane_s32::<0>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row0)),
                    v_weight,
                );

                let item_row1 = vld1q_u16(src_ptr1.as_ptr());

                store0 = vqrdmlahq_lane_s32::<1>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row1))),
                    v_weight,
                );
                store1 = vqrdmlahq_lane_s32::<1>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row1)),
                    v_weight,
                );
            } else if bounds_size == 3 {
                let weights = weight.get_unchecked(0..3);
                let mut v_weight = vld1q_dup_s32(weights.as_ptr());
                v_weight = vld1q_lane_s32::<1>(weights.as_ptr().add(1), v_weight);
                v_weight = vld1q_lane_s32::<2>(weights.as_ptr().add(2), v_weight);

                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());

                store0 = vqrdmlahq_laneq_s32::<0>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row0))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<0>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row0)),
                    v_weight,
                );

                let item_row1 = vld1q_u16(src_ptr1.as_ptr());

                store0 = vqrdmlahq_laneq_s32::<1>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row1))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<1>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row1)),
                    v_weight,
                );

                let item_row2 = vld1q_u16(src_ptr2.as_ptr());

                store0 = vqrdmlahq_laneq_s32::<2>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row2))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<2>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row2)),
                    v_weight,
                );
            } else if bounds_size == 4 {
                let weights = weight.get_unchecked(0..4);
                let v_weight = vld1q_s32(weights.as_ptr());

                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + v_dx)..);

                let item_row0 = vld1q_u16(src_ptr0.as_ptr());

                store0 = vqrdmlahq_laneq_s32::<0>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row0))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<0>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row0)),
                    v_weight,
                );

                let item_row1 = vld1q_u16(src_ptr1.as_ptr());

                store0 = vqrdmlahq_laneq_s32::<1>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row1))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<1>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row1)),
                    v_weight,
                );

                let item_row2 = vld1q_u16(src_ptr2.as_ptr());

                store0 = vqrdmlahq_laneq_s32::<2>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row2))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<2>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row2)),
                    v_weight,
                );

                let item_row3 = vld1q_u16(src_ptr3.as_ptr());

                store0 = vqrdmlahq_laneq_s32::<3>(
                    store0,
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row3))),
                    v_weight,
                );
                store1 = vqrdmlahq_laneq_s32::<3>(
                    store1,
                    vreinterpretq_s32_u32(vmovl_high_u16(item_row3)),
                    v_weight,
                );
            } else {
                for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                    let py = bounds.start + j;
                    let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                    let v_weight = vdupq_n_s32(k_weight);

                    let item_row = vld1q_u16(src_ptr.as_ptr());

                    store0 = vqrdmlahq_s32(
                        store0,
                        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row))),
                        v_weight,
                    );
                    store1 = vqrdmlahq_s32(
                        store1,
                        vreinterpretq_s32_u32(vmovl_high_u16(item_row)),
                        v_weight,
                    );
                }
            }

            let item = vminq_u16(
                vcombine_u16(vqmovun_s32(store0), vqmovun_s32(store1)),
                v_max_colors,
            );
            vst1q_u16(dst.as_mut_ptr(), item);

            cx = v_dx;
        }

        let tail8 = tail16.chunks_exact_mut(8).into_remainder();
        let iter4 = tail8.chunks_exact_mut(4);

        let v_cx = cx;

        for (x, dst) in iter4.enumerate() {
            let mut store0 = initial_store;

            let v_dx = v_cx + x * 4;

            if bounds_size == 2 {
                let weights = weight.get_unchecked(0..2);
                let mut v_weight = vld1_dup_s32(weights.as_ptr());
                v_weight = vld1_lane_s32::<1>(weights.as_ptr().add(1), v_weight);

                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);

                let item_row0 = vreinterpret_s16_u16(vld1_u16(src_ptr0.as_ptr()));
                store0 = vqrdmlahq_lane_s32::<0>(store0, vmovl_s16(item_row0), v_weight);

                let item_row1 = vreinterpret_s16_u16(vld1_u16(src_ptr1.as_ptr()));
                store0 = vqrdmlahq_lane_s32::<1>(store0, vmovl_s16(item_row1), v_weight);
            } else if bounds_size == 3 {
                let weights = weight.get_unchecked(0..3);
                let mut v_weight = vld1q_dup_s32(weights.as_ptr());
                v_weight = vld1q_lane_s32::<1>(weights.as_ptr().add(1), v_weight);
                v_weight = vld1q_lane_s32::<2>(weights.as_ptr().add(2), v_weight);

                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);

                let item_row0 = vreinterpret_s16_u16(vld1_u16(src_ptr0.as_ptr()));
                store0 = vqrdmlahq_laneq_s32::<0>(store0, vmovl_s16(item_row0), v_weight);

                let item_row1 = vreinterpret_s16_u16(vld1_u16(src_ptr1.as_ptr()));
                store0 = vqrdmlahq_laneq_s32::<1>(store0, vmovl_s16(item_row1), v_weight);

                let item_row2 = vreinterpret_s16_u16(vld1_u16(src_ptr2.as_ptr()));
                store0 = vqrdmlahq_laneq_s32::<2>(store0, vmovl_s16(item_row2), v_weight);
            } else if bounds_size == 4 {
                let weights = weight.get_unchecked(0..4);
                let v_weight = vld1q_s32(weights.as_ptr());

                let py = bounds.start;
                let src_ptr0 = src.get_unchecked((src_stride * py + v_dx)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_dx)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_dx)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + v_dx)..);

                let item_row0 = vreinterpret_s16_u16(vld1_u16(src_ptr0.as_ptr()));
                store0 = vqrdmlahq_laneq_s32::<0>(store0, vmovl_s16(item_row0), v_weight);

                let item_row1 = vreinterpret_s16_u16(vld1_u16(src_ptr1.as_ptr()));
                store0 = vqrdmlahq_laneq_s32::<1>(store0, vmovl_s16(item_row1), v_weight);

                let item_row2 = vreinterpret_s16_u16(vld1_u16(src_ptr2.as_ptr()));
                store0 = vqrdmlahq_laneq_s32::<2>(store0, vmovl_s16(item_row2), v_weight);

                let item_row3 = vreinterpret_s16_u16(vld1_u16(src_ptr3.as_ptr()));
                store0 = vqrdmlahq_laneq_s32::<3>(store0, vmovl_s16(item_row3), v_weight);
            } else {
                for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                    let py = bounds.start + j;
                    let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                    let v_weight = vdupq_n_s32(k_weight);

                    let item_row = vreinterpret_s16_u16(vld1_u16(src_ptr.as_ptr()));

                    store0 = vqrdmlahq_s32(store0, vmovl_s16(item_row), v_weight);
                }
            }

            let u_store0 = vmin_u16(vqmovun_s32(store0), vget_low_u16(v_max_colors));
            vst1_u16(dst.as_mut_ptr(), u_store0);

            cx = v_dx;
        }

        let tail4 = tail8.chunks_exact_mut(4).into_remainder();

        let a_px = cx;

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = 0;

            let v_px = a_px + x;

            if bounds_size == 2 {
                let weights = weight.get_unchecked(0..2);
                let weight0 = weights[0];
                let weight1 = weights[1];

                let py = bounds.start;
                let offset0 = src_stride * py + v_px;
                let src_ptr0 = src.get_unchecked(offset0..(offset0 + 1));
                let offset1 = src_stride * (py + 1) + v_px;
                let src_ptr1 = src.get_unchecked(offset1..(offset1 + 1));

                store0 += src_ptr0[0] as i64 * weight0 as i64;
                store0 += src_ptr1[0] as i64 * weight1 as i64;
            } else if bounds_size == 3 {
                let weights = weight.get_unchecked(0..3);
                let weight0 = weights[0];
                let weight1 = weights[1];
                let weight2 = weights[2];

                let py = bounds.start;
                let offset0 = src_stride * py + v_px;
                let src_ptr0 = src.get_unchecked(offset0..(offset0 + 1));
                let offset1 = src_stride * (py + 1) + v_px;
                let src_ptr1 = src.get_unchecked(offset1..(offset1 + 1));
                let offset2 = src_stride * (py + 2) + v_px;
                let src_ptr2 = src.get_unchecked(offset2..(offset2 + 1));

                store0 += src_ptr0[0] as i64 * weight0 as i64;
                store0 += src_ptr1[0] as i64 * weight1 as i64;
                store0 += src_ptr2[0] as i64 * weight2 as i64;
            } else if bounds_size == 4 {
                let weights = weight.get_unchecked(0..4);
                let weight0 = weights[0];
                let weight1 = weights[1];
                let weight2 = weights[2];
                let weight3 = weights[3];

                let py = bounds.start;
                let offset0 = src_stride * py + v_px;
                let src_ptr0 = src.get_unchecked(offset0..(offset0 + 1));
                let offset1 = src_stride * (py + 1) + v_px;
                let src_ptr1 = src.get_unchecked(offset1..(offset1 + 1));
                let offset2 = src_stride * (py + 2) + v_px;
                let src_ptr2 = src.get_unchecked(offset2..(offset2 + 1));
                let offset3 = src_stride * (py + 3) + v_px;
                let src_ptr3 = src.get_unchecked(offset3..(offset3 + 1));

                store0 += src_ptr0[0] as i64 * weight0 as i64;
                store0 += src_ptr1[0] as i64 * weight1 as i64;
                store0 += src_ptr2[0] as i64 * weight2 as i64;
                store0 += src_ptr3[0] as i64 * weight3 as i64;
            } else {
                for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                    let py = bounds.start + j;
                    let offset = src_stride * py + v_px;
                    let src_ptr = src.get_unchecked(offset..(offset + 1));

                    store0 += src_ptr[0] as i64 * k_weight as i64;
                }
            }

            const R: i64 = 1 << 30;
            *dst = ((store0 + R) >> 31).max(0).min(max_colors as i64) as u16;
        }
    }
}
