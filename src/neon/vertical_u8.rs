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
use crate::neon::utils::{
    vxmlal_high_lane_s16, vxmlal_high_s16, vxmlal_lane_s16, vxmlal_s16, xvld1q_u8_x2, xvld1q_u8_x4,
    xvst1q_u8_x2, xvst1q_u8_x4,
};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn pack_weights<const PRECISION: i32>(
    store_0: int32x4_t,
    store_1: int32x4_t,
    store_2: int32x4_t,
    store_3: int32x4_t,
) -> uint8x16_t {
    unsafe {
        let low_u16 = vcombine_u16(
            vqshrun_n_s32::<PRECISION>(store_0),
            vqshrun_n_s32::<PRECISION>(store_1),
        );
        let high_u16 = vcombine_u16(
            vqshrun_n_s32::<PRECISION>(store_2),
            vqshrun_n_s32::<PRECISION>(store_3),
        );
        vcombine_u8(vqmovn_u16(low_u16), vqmovn_u16(high_u16))
    }
}

#[must_use]
#[inline(always)]
unsafe fn accumulate_4_into<const D: bool>(
    item: uint8x16_t,
    store_0: int32x4_t,
    store_1: int32x4_t,
    store_2: int32x4_t,
    store_3: int32x4_t,
    weight: int16x8_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    unsafe {
        let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(item)));
        let high = vreinterpretq_s16_u16(vmovl_high_u8(item));

        let store_0 = vxmlal_s16::<D>(store_0, vget_low_s16(low), vget_low_s16(weight));
        let store_1 = vxmlal_high_s16::<D>(store_1, low, weight);
        let store_2 = vxmlal_s16::<D>(store_2, vget_low_s16(high), vget_low_s16(weight));
        let store_3 = vxmlal_high_s16::<D>(store_3, high, weight);
        (store_0, store_1, store_2, store_3)
    }
}

#[must_use]
#[inline(always)]
unsafe fn accumulate_4_into_lane<const D: bool, const W: i32>(
    item: uint8x16_t,
    store_0: int32x4_t,
    store_1: int32x4_t,
    store_2: int32x4_t,
    store_3: int32x4_t,
    weight: int16x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    unsafe {
        let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(item)));
        let high = vreinterpretq_s16_u16(vmovl_high_u8(item));

        let store_0 = vxmlal_lane_s16::<D, W>(store_0, vget_low_s16(low), weight);
        let store_1 = vxmlal_high_lane_s16::<D, W>(store_1, low, weight);
        let store_2 = vxmlal_lane_s16::<D, W>(store_2, vget_low_s16(high), weight);
        let store_3 = vxmlal_high_lane_s16::<D, W>(store_3, high, weight);
        (store_0, store_1, store_2, store_3)
    }
}

pub(crate) fn convolve_vertical_neon_i32_precision(
    width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    convolve_vertical_neon_row_full::<false, { crate::support::PRECISION }>(
        width, bounds, src, dst, src_stride, weight,
    );
}

pub(crate) fn convolve_vertical_neon_i32_precision_d(
    width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    convolve_vertical_neon_row_full::<true, 16>(width, bounds, src, dst, src_stride, weight);
}

fn convolve_vertical_neon_row_full<const D: bool, const PRECISION: i32>(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    let mut cx = 0usize;
    let rnd_const: i32 = 1 << (PRECISION - 1);

    unsafe {
        let iter_64 = dst.chunks_exact_mut(64);

        let bounds_size = bounds.size;

        for dst in iter_64 {
            let vld = vdupq_n_s32(rnd_const);
            let mut store_0 = vld;
            let mut store_1 = vld;
            let mut store_2 = vld;
            let mut store_3 = vld;

            let mut store_4 = vld;
            let mut store_5 = vld;
            let mut store_6 = vld;
            let mut store_7 = vld;

            let mut store_8 = vld;
            let mut store_9 = vld;
            let mut store_10 = vld;
            let mut store_11 = vld;

            let mut store_12 = vld;
            let mut store_13 = vld;
            let mut store_14 = vld;
            let mut store_15 = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight = vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);

                let items0 = xvld1q_u8_x4(src_ptr0.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                    items0.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 0>(
                    items0.1, store_4, store_5, store_6, store_7, v_weight,
                );
                (store_8, store_9, store_10, store_11) = accumulate_4_into_lane::<D, 0>(
                    items0.2, store_8, store_9, store_10, store_11, v_weight,
                );
                (store_12, store_13, store_14, store_15) = accumulate_4_into_lane::<D, 0>(
                    items0.3, store_12, store_13, store_14, store_15, v_weight,
                );

                let items1 = xvld1q_u8_x4(src_ptr1.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                    items1.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 1>(
                    items1.1, store_4, store_5, store_6, store_7, v_weight,
                );
                (store_8, store_9, store_10, store_11) = accumulate_4_into_lane::<D, 1>(
                    items1.2, store_8, store_9, store_10, store_11, v_weight,
                );
                (store_12, store_13, store_14, store_15) = accumulate_4_into_lane::<D, 1>(
                    items1.3, store_12, store_13, store_14, store_15, v_weight,
                );
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let mut v_weight =
                    vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                v_weight = vld1_lane_s16::<2>(weight.as_ptr().add(2), v_weight);
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);

                let items0 = xvld1q_u8_x4(src_ptr0.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                    items0.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 0>(
                    items0.1, store_4, store_5, store_6, store_7, v_weight,
                );
                (store_8, store_9, store_10, store_11) = accumulate_4_into_lane::<D, 0>(
                    items0.2, store_8, store_9, store_10, store_11, v_weight,
                );
                (store_12, store_13, store_14, store_15) = accumulate_4_into_lane::<D, 0>(
                    items0.3, store_12, store_13, store_14, store_15, v_weight,
                );

                let items1 = xvld1q_u8_x4(src_ptr1.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                    items1.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 1>(
                    items1.1, store_4, store_5, store_6, store_7, v_weight,
                );
                (store_8, store_9, store_10, store_11) = accumulate_4_into_lane::<D, 1>(
                    items1.2, store_8, store_9, store_10, store_11, v_weight,
                );
                (store_12, store_13, store_14, store_15) = accumulate_4_into_lane::<D, 1>(
                    items1.3, store_12, store_13, store_14, store_15, v_weight,
                );

                let items2 = xvld1q_u8_x4(src_ptr2.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 2>(
                    items2.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 2>(
                    items2.1, store_4, store_5, store_6, store_7, v_weight,
                );
                (store_8, store_9, store_10, store_11) = accumulate_4_into_lane::<D, 2>(
                    items2.2, store_8, store_9, store_10, store_11, v_weight,
                );
                (store_12, store_13, store_14, store_15) = accumulate_4_into_lane::<D, 2>(
                    items2.3, store_12, store_13, store_14, store_15, v_weight,
                );
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight = vld1_s16(weight.as_ptr());
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);

                let items0 = xvld1q_u8_x4(src_ptr0.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                    items0.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 0>(
                    items0.1, store_4, store_5, store_6, store_7, v_weight,
                );
                (store_8, store_9, store_10, store_11) = accumulate_4_into_lane::<D, 0>(
                    items0.2, store_8, store_9, store_10, store_11, v_weight,
                );
                (store_12, store_13, store_14, store_15) = accumulate_4_into_lane::<D, 0>(
                    items0.3, store_12, store_13, store_14, store_15, v_weight,
                );

                let items1 = xvld1q_u8_x4(src_ptr1.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                    items1.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 1>(
                    items1.1, store_4, store_5, store_6, store_7, v_weight,
                );
                (store_8, store_9, store_10, store_11) = accumulate_4_into_lane::<D, 1>(
                    items1.2, store_8, store_9, store_10, store_11, v_weight,
                );
                (store_12, store_13, store_14, store_15) = accumulate_4_into_lane::<D, 1>(
                    items1.3, store_12, store_13, store_14, store_15, v_weight,
                );

                let items2 = xvld1q_u8_x4(src_ptr2.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 2>(
                    items2.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 2>(
                    items2.1, store_4, store_5, store_6, store_7, v_weight,
                );
                (store_8, store_9, store_10, store_11) = accumulate_4_into_lane::<D, 2>(
                    items2.2, store_8, store_9, store_10, store_11, v_weight,
                );
                (store_12, store_13, store_14, store_15) = accumulate_4_into_lane::<D, 2>(
                    items2.3, store_12, store_13, store_14, store_15, v_weight,
                );

                let items3 = xvld1q_u8_x4(src_ptr3.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 3>(
                    items3.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 3>(
                    items3.1, store_4, store_5, store_6, store_7, v_weight,
                );
                (store_8, store_9, store_10, store_11) = accumulate_4_into_lane::<D, 3>(
                    items3.2, store_8, store_9, store_10, store_11, v_weight,
                );
                (store_12, store_13, store_14, store_15) = accumulate_4_into_lane::<D, 3>(
                    items3.3, store_12, store_13, store_14, store_15, v_weight,
                );
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let items = xvld1q_u8_x4(src_ptr.as_ptr());

                    (store_0, store_1, store_2, store_3) = accumulate_4_into::<D>(
                        items.0, store_0, store_1, store_2, store_3, v_weight,
                    );
                    (store_4, store_5, store_6, store_7) = accumulate_4_into::<D>(
                        items.1, store_4, store_5, store_6, store_7, v_weight,
                    );
                    (store_8, store_9, store_10, store_11) = accumulate_4_into::<D>(
                        items.2, store_8, store_9, store_10, store_11, v_weight,
                    );
                    (store_12, store_13, store_14, store_15) = accumulate_4_into::<D>(
                        items.3, store_12, store_13, store_14, store_15, v_weight,
                    );
                }
            }

            let item_0 = pack_weights::<PRECISION>(store_0, store_1, store_2, store_3);
            let item_1 = pack_weights::<PRECISION>(store_4, store_5, store_6, store_7);
            let item_2 = pack_weights::<PRECISION>(store_8, store_9, store_10, store_11);
            let item_3 = pack_weights::<PRECISION>(store_12, store_13, store_14, store_15);

            let dst_items = uint8x16x4_t(item_0, item_1, item_2, item_3);
            xvst1q_u8_x4(dst.as_mut_ptr(), dst_items);

            cx += 64;
        }

        let mut rem = dst.chunks_exact_mut(64).into_remainder();
        let iter_32 = rem.chunks_exact_mut(32);

        for dst in iter_32 {
            let vld = vdupq_n_s32(rnd_const);
            let mut store_0 = vld;
            let mut store_1 = vld;
            let mut store_2 = vld;
            let mut store_3 = vld;
            let mut store_4 = vld;
            let mut store_5 = vld;
            let mut store_6 = vld;
            let mut store_7 = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight = vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let items0 = xvld1q_u8_x2(src_ptr0.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                    items0.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 0>(
                    items0.1, store_4, store_5, store_6, store_7, v_weight,
                );

                let items1 = xvld1q_u8_x2(src_ptr1.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                    items1.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 1>(
                    items1.1, store_4, store_5, store_6, store_7, v_weight,
                );
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let mut v_weight =
                    vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                v_weight = vld1_lane_s16::<2>(weight.as_ptr().add(2), v_weight);
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let items0 = xvld1q_u8_x2(src_ptr0.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                    items0.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 0>(
                    items0.1, store_4, store_5, store_6, store_7, v_weight,
                );

                let items1 = xvld1q_u8_x2(src_ptr1.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                    items1.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 1>(
                    items1.1, store_4, store_5, store_6, store_7, v_weight,
                );

                let items2 = xvld1q_u8_x2(src_ptr2.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 2>(
                    items2.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 2>(
                    items2.1, store_4, store_5, store_6, store_7, v_weight,
                );
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight = vld1_s16(weight.as_ptr());
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);
                let items0 = xvld1q_u8_x2(src_ptr0.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                    items0.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 0>(
                    items0.1, store_4, store_5, store_6, store_7, v_weight,
                );

                let items1 = xvld1q_u8_x2(src_ptr1.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                    items1.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 1>(
                    items1.1, store_4, store_5, store_6, store_7, v_weight,
                );

                let items2 = xvld1q_u8_x2(src_ptr2.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 2>(
                    items2.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 2>(
                    items2.1, store_4, store_5, store_6, store_7, v_weight,
                );

                let items3 = xvld1q_u8_x2(src_ptr3.as_ptr());

                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 3>(
                    items3.0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 3>(
                    items3.1, store_4, store_5, store_6, store_7, v_weight,
                );
            } else {
                for j in 0..bounds.size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let items = xvld1q_u8_x2(src_ptr.as_ptr());

                    (store_0, store_1, store_2, store_3) = accumulate_4_into::<D>(
                        items.0, store_0, store_1, store_2, store_3, v_weight,
                    );
                    (store_4, store_5, store_6, store_7) = accumulate_4_into::<D>(
                        items.1, store_4, store_5, store_6, store_7, v_weight,
                    );
                }
            }

            let item_0 = pack_weights::<PRECISION>(store_0, store_1, store_2, store_3);
            let item_1 = pack_weights::<PRECISION>(store_4, store_5, store_6, store_7);

            let dst_items = uint8x16x2_t(item_0, item_1);
            xvst1q_u8_x2(dst.as_mut_ptr(), dst_items);

            cx += 32;
        }

        rem = rem.chunks_exact_mut(32).into_remainder();
        let iter_16 = rem.chunks_exact_mut(16);

        for dst in iter_16 {
            let vld = vdupq_n_s32(rnd_const);
            let mut store_0 = vld;
            let mut store_1 = vld;
            let mut store_2 = vld;
            let mut store_3 = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight = vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let item_row0 = vld1q_u8(src_ptr0.as_ptr());
                let item_row1 = vld1q_u8(src_ptr1.as_ptr());
                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                    item_row0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                    item_row1, store_0, store_1, store_2, store_3, v_weight,
                );
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let mut v_weight =
                    vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                v_weight = vld1_lane_s16::<2>(weight.as_ptr().add(2), v_weight);
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let item_row0 = vld1q_u8(src_ptr0.as_ptr());
                let item_row1 = vld1q_u8(src_ptr1.as_ptr());
                let item_row2 = vld1q_u8(src_ptr2.as_ptr());
                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                    item_row0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                    item_row1, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 2>(
                    item_row2, store_0, store_1, store_2, store_3, v_weight,
                );
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight = vld1_s16(weight.as_ptr());
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);
                let item_row0 = vld1q_u8(src_ptr0.as_ptr());
                let item_row1 = vld1q_u8(src_ptr1.as_ptr());
                let item_row2 = vld1q_u8(src_ptr2.as_ptr());
                let item_row3 = vld1q_u8(src_ptr3.as_ptr());
                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                    item_row0, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                    item_row1, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 2>(
                    item_row2, store_0, store_1, store_2, store_3, v_weight,
                );
                (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 3>(
                    item_row3, store_0, store_1, store_2, store_3, v_weight,
                );
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let item_row = vld1q_u8(src_ptr.as_ptr());
                    (store_0, store_1, store_2, store_3) = accumulate_4_into::<D>(
                        item_row, store_0, store_1, store_2, store_3, v_weight,
                    );
                }
            }

            let item = pack_weights::<PRECISION>(store_0, store_1, store_2, store_3);

            vst1q_u8(dst.as_mut_ptr(), item);

            cx += 16;
        }

        rem = rem.chunks_exact_mut(16).into_remainder();
        let iter_8 = rem.chunks_exact_mut(8);

        for dst in iter_8 {
            let vld = vdupq_n_s32(rnd_const);
            let mut store_0 = vld;
            let mut store_1 = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight = vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let item_row0 = vld1_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_u8(src_ptr1.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                store_0 = vxmlal_lane_s16::<D, 0>(store_0, vget_low_s16(low0), v_weight);
                store_1 = vxmlal_high_lane_s16::<D, 0>(store_1, low0, v_weight);
                store_0 = vxmlal_lane_s16::<D, 1>(store_0, vget_low_s16(low1), v_weight);
                store_1 = vxmlal_high_lane_s16::<D, 1>(store_1, low1, v_weight);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let mut v_weight =
                    vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                v_weight = vld1_lane_s16::<2>(weight.as_ptr().add(2), v_weight);
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let item_row0 = vld1_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_u8(src_ptr1.as_ptr());
                let item_row2 = vld1_u8(src_ptr2.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                let low2 = vreinterpretq_s16_u16(vmovl_u8(item_row2));
                store_0 = vxmlal_lane_s16::<D, 0>(store_0, vget_low_s16(low0), v_weight);
                store_1 = vxmlal_high_lane_s16::<D, 0>(store_1, low0, v_weight);
                store_0 = vxmlal_lane_s16::<D, 1>(store_0, vget_low_s16(low1), v_weight);
                store_1 = vxmlal_high_lane_s16::<D, 1>(store_1, low1, v_weight);
                store_0 = vxmlal_lane_s16::<D, 2>(store_0, vget_low_s16(low2), v_weight);
                store_1 = vxmlal_high_lane_s16::<D, 3>(store_1, low2, v_weight);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight = vld1_s16(weight.as_ptr());
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);
                let item_row0 = vld1_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_u8(src_ptr1.as_ptr());
                let item_row2 = vld1_u8(src_ptr2.as_ptr());
                let item_row3 = vld1_u8(src_ptr3.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                let low2 = vreinterpretq_s16_u16(vmovl_u8(item_row2));
                let low3 = vreinterpretq_s16_u16(vmovl_u8(item_row3));
                store_0 = vxmlal_lane_s16::<D, 0>(store_0, vget_low_s16(low0), v_weight);
                store_1 = vxmlal_high_lane_s16::<D, 0>(store_1, low0, v_weight);
                store_0 = vxmlal_lane_s16::<D, 1>(store_0, vget_low_s16(low1), v_weight);
                store_1 = vxmlal_high_lane_s16::<D, 1>(store_1, low1, v_weight);
                store_0 = vxmlal_lane_s16::<D, 2>(store_0, vget_low_s16(low2), v_weight);
                store_1 = vxmlal_high_lane_s16::<D, 2>(store_1, low2, v_weight);
                store_0 = vxmlal_lane_s16::<D, 3>(store_0, vget_low_s16(low3), v_weight);
                store_1 = vxmlal_high_lane_s16::<D, 3>(store_1, low3, v_weight);
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let item_row = vld1_u8(src_ptr.as_ptr());

                    let low = vreinterpretq_s16_u16(vmovl_u8(item_row));
                    store_0 = vxmlal_s16::<D>(store_0, vget_low_s16(low), vget_low_s16(v_weight));
                    store_1 = vxmlal_high_s16::<D>(store_1, low, v_weight);
                }
            }

            let low_u16 = vcombine_u16(
                vqshrun_n_s32::<PRECISION>(store_0),
                vqshrun_n_s32::<PRECISION>(store_1),
            );

            let item = vqmovn_u16(low_u16);

            vst1_u8(dst.as_mut_ptr(), item);

            cx += 8;
        }

        rem = rem.chunks_exact_mut(8).into_remainder();
        let iter_1 = rem.iter_mut();

        for dst in iter_1 {
            let vld = vdupq_n_s32(rnd_const);
            let mut store = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight = vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let item_row0 = vld1_dup_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_dup_u8(src_ptr1.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                store = vxmlal_lane_s16::<D, 0>(store, vget_low_s16(low0), v_weight);
                store = vxmlal_lane_s16::<D, 1>(store, vget_low_s16(low1), v_weight);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let mut v_weight =
                    vreinterpret_s16_s32(vld1_dup_s32(weight.as_ptr() as *const i32));
                v_weight = vld1_lane_s16::<2>(weight.as_ptr().add(2), v_weight);
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let item_row0 = vld1_dup_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_dup_u8(src_ptr1.as_ptr());
                let item_row2 = vld1_dup_u8(src_ptr2.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                let low2 = vreinterpretq_s16_u16(vmovl_u8(item_row2));
                store = vxmlal_lane_s16::<D, 0>(store, vget_low_s16(low0), v_weight);
                store = vxmlal_lane_s16::<D, 1>(store, vget_low_s16(low1), v_weight);
                store = vxmlal_lane_s16::<D, 2>(store, vget_low_s16(low2), v_weight);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight = vld1_s16(weight.as_ptr());
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);
                let item_row0 = vld1_dup_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_dup_u8(src_ptr1.as_ptr());
                let item_row2 = vld1_dup_u8(src_ptr2.as_ptr());
                let item_row3 = vld1_dup_u8(src_ptr3.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                let low2 = vreinterpretq_s16_u16(vmovl_u8(item_row2));
                let low3 = vreinterpretq_s16_u16(vmovl_u8(item_row3));
                store = vxmlal_lane_s16::<D, 0>(store, vget_low_s16(low0), v_weight);
                store = vxmlal_lane_s16::<D, 1>(store, vget_low_s16(low1), v_weight);
                store = vxmlal_lane_s16::<D, 2>(store, vget_low_s16(low2), v_weight);
                store = vxmlal_lane_s16::<D, 3>(store, vget_low_s16(low3), v_weight);
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let item_row = vld1_dup_u8(src_ptr.as_ptr());

                    let low = vreinterpretq_s16_u16(vmovl_u8(item_row));
                    store = vxmlal_s16::<D>(store, vget_low_s16(low), vget_low_s16(v_weight));
                }
            }

            let shrank_store = vqshrun_n_s32::<PRECISION>(store);

            let low_16 = vcombine_u16(shrank_store, shrank_store);

            let item = vqmovn_u16(low_16);
            vst1_lane_u8::<0>(dst, item);
            cx += 1;
        }
    }
}
