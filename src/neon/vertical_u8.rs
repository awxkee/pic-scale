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
    vxmlal_high_lane_s16, vxmlal_high_s16, vxmlal_lane_s16, vxmlal_s16, xvld1q_u8_x2, xvst1q_u8_x2,
};
use std::arch::aarch64::*;

#[inline(always)]
fn pack_weights<const PRECISION: i32>(
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
fn accumulate_4_into<const D: bool>(
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
fn accumulate_4_into_lane<const D: bool, const W: i32>(
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
    _: u32,
) {
    convolve_vertical_neon_row_full::<false, { crate::support::PRECISION }>(
        width, bounds, src, dst, src_stride, weight,
    );
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_32_items<const D: bool, const PRECISION: i32>(
    chunks: &mut [[u8; 32]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weights: &[i16],
    cx: usize,
) -> usize {
    let rnd_const: i32 = 1 << (PRECISION - 1);
    let vld = vdupq_n_s32(rnd_const);

    let mut cx = cx;

    for dst in chunks {
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;
        let mut store_4 = vld;
        let mut store_5 = vld;
        let mut store_6 = vld;
        let mut store_7 = vld;
        let px = cx;

        let mut j = 0usize;

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let weights = unsafe { vld1_s16(weights.get_unchecked(j..).as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let items0 = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };
            let items1 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let items2 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let items3 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };

            (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                items0.0, store_0, store_1, store_2, store_3, weights,
            );
            (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 0>(
                items0.1, store_4, store_5, store_6, store_7, weights,
            );

            (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                items1.0, store_0, store_1, store_2, store_3, weights,
            );
            (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 1>(
                items1.1, store_4, store_5, store_6, store_7, weights,
            );

            (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 2>(
                items2.0, store_0, store_1, store_2, store_3, weights,
            );
            (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 2>(
                items2.1, store_4, store_5, store_6, store_7, weights,
            );

            (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 3>(
                items3.0, store_0, store_1, store_2, store_3, weights,
            );
            (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 3>(
                items3.1, store_4, store_5, store_6, store_7, weights,
            );

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let weights = unsafe {
                vreinterpret_s16_u32(vld1_lane_u32::<0>(
                    weights.get_unchecked(j..).as_ptr().cast(),
                    vdup_n_u32(0),
                ))
            };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let items0 = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };
            let items1 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride..).as_ptr()) };

            (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 0>(
                items0.0, store_0, store_1, store_2, store_3, weights,
            );
            (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 0>(
                items0.1, store_4, store_5, store_6, store_7, weights,
            );

            (store_0, store_1, store_2, store_3) = accumulate_4_into_lane::<D, 1>(
                items1.0, store_0, store_1, store_2, store_3, weights,
            );
            (store_4, store_5, store_6, store_7) = accumulate_4_into_lane::<D, 1>(
                items1.1, store_4, store_5, store_6, store_7, weights,
            );

            j += 2;
        }

        for j in j..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weights.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s16(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let items = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };

            (store_0, store_1, store_2, store_3) =
                accumulate_4_into::<D>(items.0, store_0, store_1, store_2, store_3, v_weight);
            (store_4, store_5, store_6, store_7) =
                accumulate_4_into::<D>(items.1, store_4, store_5, store_6, store_7, v_weight);
        }

        let item_0 = pack_weights::<PRECISION>(store_0, store_1, store_2, store_3);
        let item_1 = pack_weights::<PRECISION>(store_4, store_5, store_6, store_7);

        let dst_items = uint8x16x2_t(item_0, item_1);
        unsafe {
            xvst1q_u8_x2(dst.as_mut_ptr(), dst_items);
        }

        cx += 32;
    }
    cx
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_16_items<const D: bool, const PRECISION: i32>(
    chunks: &mut [[u8; 16]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weight: &[i16],
    cx: usize,
) -> usize {
    let rnd_const: i32 = 1 << (PRECISION - 1);
    let vld = vdupq_n_s32(rnd_const);
    let mut cx = cx;

    for dst in chunks {
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;

        let px = cx;

        for j in 0..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weight.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s16(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1q_u8(src_ptr.as_ptr()) };
            (store_0, store_1, store_2, store_3) =
                accumulate_4_into::<D>(item_row, store_0, store_1, store_2, store_3, v_weight);
        }

        let item = pack_weights::<PRECISION>(store_0, store_1, store_2, store_3);

        unsafe {
            vst1q_u8(dst.as_mut_ptr(), item);
        }

        cx += 16;
    }
    cx
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_8_items<const D: bool, const PRECISION: i32>(
    chunks: &mut [[u8; 8]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weight: &[i16],
    cx: usize,
) -> usize {
    let rnd_const: i32 = 1 << (PRECISION - 1);
    let vld = vdupq_n_s32(rnd_const);
    let mut cx = cx;

    for dst in chunks {
        let mut store_0 = vld;
        let mut store_1 = vld;

        let px = cx;

        for j in 0..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weight.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s16(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1_u8(src_ptr.as_ptr()) };

            let low = vreinterpretq_s16_u16(vmovl_u8(item_row));
            store_0 = vxmlal_s16::<D>(store_0, vget_low_s16(low), vget_low_s16(v_weight));
            store_1 = vxmlal_high_s16::<D>(store_1, low, v_weight);
        }

        let low_u16 = vcombine_u16(
            vqshrun_n_s32::<PRECISION>(store_0),
            vqshrun_n_s32::<PRECISION>(store_1),
        );

        let item = vqmovn_u16(low_u16);

        unsafe {
            vst1_u8(dst.as_mut_ptr(), item);
        }

        cx += 8;
    }
    cx
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_items<const D: bool, const PRECISION: i32>(
    chunks: &mut [u8],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weight: &[i16],
    cx: usize,
) {
    let rnd_const: i32 = 1 << (PRECISION - 1);
    let vld = vdupq_n_s32(rnd_const);
    let mut cx = cx;

    #[allow(clippy::explicit_counter_loop)]
    for dst in chunks {
        let mut store = vld;

        let px = cx;

        for j in 0..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weight.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s16(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1_dup_u8(src_ptr.as_ptr()) };

            let low = vreinterpretq_s16_u16(vmovl_u8(item_row));
            store = vxmlal_s16::<D>(store, vget_low_s16(low), vget_low_s16(v_weight));
        }

        let shrank_store = vqshrun_n_s32::<PRECISION>(store);

        let low_16 = vcombine_u16(shrank_store, shrank_store);

        let item = vqmovn_u16(low_16);
        unsafe {
            vst1_lane_u8::<0>(dst, item);
        }
        cx += 1;
    }
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
    unsafe {
        cx = convolve_32_items::<D, PRECISION>(
            dst.as_chunks_mut::<32>().0,
            bounds,
            src,
            src_stride,
            weight,
            cx,
        );

        let mut rem = dst.as_chunks_mut::<32>().1;

        cx = convolve_16_items::<D, PRECISION>(
            rem.as_chunks_mut::<16>().0,
            bounds,
            src,
            src_stride,
            weight,
            cx,
        );

        rem = rem.as_chunks_mut::<16>().1;
        cx = convolve_8_items::<D, PRECISION>(
            rem.as_chunks_mut::<8>().0,
            bounds,
            src,
            src_stride,
            weight,
            cx,
        );

        rem = rem.as_chunks_mut::<8>().1;
        convolve_items::<D, PRECISION>(rem, bounds, src, src_stride, weight, cx);
    }
}
