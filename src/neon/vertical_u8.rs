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
use crate::support::{PRECISION, ROUNDING_CONST};
use std::arch::aarch64::*;

macro_rules! pack_weights {
    ($store_0: expr, $store_1: expr, $store_2: expr, $store_3: expr) => {{
        let zeros = vdupq_n_s16(0);
        let low_s16 = vcombine_s16(
            vqshrn_n_s32::<PRECISION>($store_0),
            vqshrn_n_s32::<PRECISION>($store_1),
        );
        let high_s16 = vcombine_s16(
            vqshrn_n_s32::<PRECISION>($store_2),
            vqshrn_n_s32::<PRECISION>($store_3),
        );
        let low_16 = vreinterpretq_u16_s16(vmaxq_s16(low_s16, zeros));
        let high_16 = vreinterpretq_u16_s16(vmaxq_s16(high_s16, zeros));
        vcombine_u8(vqmovn_u16(low_16), vqmovn_u16(high_16))
    }};
}

macro_rules! accumulate_4_into {
    ($item: expr,$store_0: expr, $store_1: expr, $store_2: expr, $store_3: expr, $weight: expr) => {{
        let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8($item)));
        let high = vreinterpretq_s16_u16(vmovl_high_u8($item));

        $store_0 = vmlal_s16($store_0, vget_low_s16(low), vget_low_s16($weight));
        $store_1 = vmlal_high_s16($store_1, low, $weight);
        $store_2 = vmlal_s16($store_2, vget_low_s16(high), vget_low_s16($weight));
        $store_3 = vmlal_high_s16($store_3, high, $weight);
    }};
}

pub fn convolve_vertical_neon_i16_precision(
    width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    unsafe {
        convolve_vertical_neon_row_upper(width, bounds, src, dst, src_stride, weight);
    }
}

pub fn convolve_vertical_neon_i32_precision(
    width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    convolve_vertical_neon_row_full(width, bounds, src, dst, src_stride, weight);
}

#[inline(always)]
unsafe fn vdot<const SCALE: i32>(
    store0: int16x8_t,
    store1: int16x8_t,
    row: uint8x16_t,
    weight: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let lo0 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(vget_low_u8(row)));
    let store0 = vqrdmlahq_s16(store0, lo0, weight);
    let hi0 = vreinterpretq_s16_u16(vshll_high_n_u8::<SCALE>(row));
    let store1 = vqrdmlahq_s16(store1, hi0, weight);
    (store0, store1)
}

#[target_feature(enable = "rdm")]
unsafe fn convolve_vertical_neon_row_upper(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    let mut cx = 0usize;

    unsafe {
        let zeros = vdupq_n_s16(0);
        let iter_64 = dst.chunks_exact_mut(64);

        let bounds_size = bounds.size;
        const SCALE: i32 = 6;
        const R_SHR_SCALE: i32 = SCALE;
        const ROUNDING: i16 = 1 << (SCALE - 1);

        for dst in iter_64 {
            let vld = vdupq_n_s16(ROUNDING);

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
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);

                let items0 = vld1q_u8_x4(src_ptr0.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items0.0, v_weight0);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items0.1, v_weight0);
                (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items0.2, v_weight0);
                (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items0.3, v_weight0);

                let items1 = vld1q_u8_x4(src_ptr1.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items1.0, v_weight1);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items1.1, v_weight1);
                (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items1.2, v_weight1);
                (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items1.3, v_weight1);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);

                let items0 = vld1q_u8_x4(src_ptr0.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items0.0, v_weight0);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items0.1, v_weight0);
                (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items0.2, v_weight0);
                (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items0.3, v_weight0);

                let items1 = vld1q_u8_x4(src_ptr1.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items1.0, v_weight1);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items1.1, v_weight1);
                (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items1.2, v_weight1);
                (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items1.3, v_weight1);

                let items2 = vld1q_u8_x4(src_ptr2.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items2.0, v_weight2);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items2.1, v_weight2);
                (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items2.2, v_weight2);
                (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items2.3, v_weight2);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);

                let items0 = vld1q_u8_x4(src_ptr0.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items0.0, v_weight0);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items0.1, v_weight0);
                (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items0.2, v_weight0);
                (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items0.3, v_weight0);

                let items1 = vld1q_u8_x4(src_ptr1.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items1.0, v_weight1);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items1.1, v_weight1);
                (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items1.2, v_weight1);
                (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items1.3, v_weight1);

                let items2 = vld1q_u8_x4(src_ptr2.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items2.0, v_weight2);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items2.1, v_weight2);
                (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items2.2, v_weight2);
                (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items2.3, v_weight2);

                let items3 = vld1q_u8_x4(src_ptr3.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items3.0, v_weight3);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items3.1, v_weight3);
                (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items3.2, v_weight3);
                (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items3.3, v_weight3);
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let items = vld1q_u8_x4(src_ptr.as_ptr());

                    (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items.0, v_weight);
                    (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items.1, v_weight);
                    (store_4, store_5) = vdot::<SCALE>(store_4, store_5, items.2, v_weight);
                    (store_6, store_7) = vdot::<SCALE>(store_6, store_7, items.3, v_weight);
                }
            }

            store_0 = vmaxq_s16(store_0, zeros);
            store_1 = vmaxq_s16(store_1, zeros);
            store_2 = vmaxq_s16(store_2, zeros);
            store_3 = vmaxq_s16(store_3, zeros);
            store_4 = vmaxq_s16(store_4, zeros);
            store_5 = vmaxq_s16(store_5, zeros);
            store_6 = vmaxq_s16(store_6, zeros);
            store_7 = vmaxq_s16(store_7, zeros);

            let item00 = vqshrun_n_s16::<R_SHR_SCALE>(store_0);
            let item01 = vqshrun_n_s16::<R_SHR_SCALE>(store_1);
            let item10 = vqshrun_n_s16::<R_SHR_SCALE>(store_2);
            let item11 = vqshrun_n_s16::<R_SHR_SCALE>(store_3);
            let item20 = vqshrun_n_s16::<R_SHR_SCALE>(store_4);
            let item21 = vqshrun_n_s16::<R_SHR_SCALE>(store_5);
            let item30 = vqshrun_n_s16::<R_SHR_SCALE>(store_6);
            let item31 = vqshrun_n_s16::<R_SHR_SCALE>(store_7);
            let item0 = vcombine_u8(item00, item01);
            let item1 = vcombine_u8(item10, item11);
            let item2 = vcombine_u8(item20, item21);
            let item3 = vcombine_u8(item30, item31);

            let dst_items = uint8x16x4_t(item0, item1, item2, item3);
            vst1q_u8_x4(dst.as_mut_ptr(), dst_items);

            cx += 64;
        }

        let mut rem = dst.chunks_exact_mut(64).into_remainder();
        let iter_32 = rem.chunks_exact_mut(32);

        for dst in iter_32 {
            let vld = vdupq_n_s16(ROUNDING);
            let mut store_0 = vld;
            let mut store_1 = vld;
            let mut store_2 = vld;
            let mut store_3 = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);

                let items0 = vld1q_u8_x2(src_ptr0.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items0.0, v_weight0);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items0.1, v_weight0);

                let items1 = vld1q_u8_x2(src_ptr1.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items1.0, v_weight1);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items1.1, v_weight1);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);

                let items0 = vld1q_u8_x2(src_ptr0.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items0.0, v_weight0);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items0.1, v_weight0);

                let items1 = vld1q_u8_x2(src_ptr1.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items1.0, v_weight1);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items1.1, v_weight1);

                let items2 = vld1q_u8_x2(src_ptr2.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items2.0, v_weight2);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items2.1, v_weight2);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);

                let items0 = vld1q_u8_x2(src_ptr0.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items0.0, v_weight0);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items0.1, v_weight0);

                let items1 = vld1q_u8_x2(src_ptr1.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items1.0, v_weight1);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items1.1, v_weight1);

                let items2 = vld1q_u8_x2(src_ptr2.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items2.0, v_weight2);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items2.1, v_weight2);

                let items3 = vld1q_u8_x2(src_ptr3.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items3.0, v_weight3);
                (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items3.1, v_weight3);
            } else {
                for j in 0..bounds.size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let items = vld1q_u8_x2(src_ptr.as_ptr());

                    (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items.0, v_weight);
                    (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items.1, v_weight);
                }
            }

            store_0 = vmaxq_s16(store_0, zeros);
            store_1 = vmaxq_s16(store_1, zeros);
            store_2 = vmaxq_s16(store_2, zeros);
            store_3 = vmaxq_s16(store_3, zeros);

            let item00 = vqshrun_n_s16::<R_SHR_SCALE>(store_0);
            let item01 = vqshrun_n_s16::<R_SHR_SCALE>(store_1);
            let item10 = vqshrun_n_s16::<R_SHR_SCALE>(store_2);
            let item11 = vqshrun_n_s16::<R_SHR_SCALE>(store_3);
            let item0 = vcombine_u8(item00, item01);
            let item1 = vcombine_u8(item10, item11);

            let dst_items = uint8x16x2_t(item0, item1);
            vst1q_u8_x2(dst.as_mut_ptr(), dst_items);

            cx += 32;
        }

        rem = rem.chunks_exact_mut(32).into_remainder();
        let iter_16 = rem.chunks_exact_mut(16);

        for dst in iter_16 {
            let vld = vdupq_n_s16(ROUNDING);
            let mut store_0 = vld;
            let mut store_1 = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);

                let item0 = vld1q_u8(src_ptr0.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item0, v_weight0);

                let item1 = vld1q_u8(src_ptr1.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item1, v_weight1);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);

                let item0 = vld1q_u8(src_ptr0.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item0, v_weight0);

                let item1 = vld1q_u8(src_ptr1.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item1, v_weight1);

                let item2 = vld1q_u8(src_ptr2.as_ptr());

                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item2, v_weight2);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);

                let item0 = vld1q_u8(src_ptr0.as_ptr());
                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item0, v_weight0);

                let item1 = vld1q_u8(src_ptr1.as_ptr());
                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item1, v_weight1);

                let item2 = vld1q_u8(src_ptr2.as_ptr());
                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item2, v_weight2);

                let item3 = vld1q_u8(src_ptr3.as_ptr());
                (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item3, v_weight3);
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let item_row = vld1q_u8(src_ptr.as_ptr());

                    (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item_row, v_weight);
                }
            }

            store_0 = vmaxq_s16(store_0, zeros);
            store_1 = vmaxq_s16(store_1, zeros);

            let item0 = vqshrun_n_s16::<R_SHR_SCALE>(store_0);
            let item1 = vqshrun_n_s16::<R_SHR_SCALE>(store_1);

            vst1q_u8(dst.as_mut_ptr(), vcombine_u8(item0, item1));

            cx += 16;
        }

        rem = rem.chunks_exact_mut(16).into_remainder();
        let iter_8 = rem.chunks_exact_mut(8);

        for dst in iter_8 {
            let vld = vdupq_n_s16(ROUNDING);
            let mut store_0 = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);

                let item0 = vld1_u8(src_ptr0.as_ptr());
                let low0 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item0));
                store_0 = vqrdmlahq_s16(store_0, low0, v_weight0);

                let item1 = vld1_u8(src_ptr1.as_ptr());
                let low1 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item1));
                store_0 = vqrdmlahq_s16(store_0, low1, v_weight1);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);

                let item0 = vld1_u8(src_ptr0.as_ptr());
                let low0 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item0));
                store_0 = vqrdmlahq_s16(store_0, low0, v_weight0);

                let item1 = vld1_u8(src_ptr1.as_ptr());
                let low1 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item1));
                store_0 = vqrdmlahq_s16(store_0, low1, v_weight1);

                let item2 = vld1_u8(src_ptr2.as_ptr());
                let low2 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item2));
                store_0 = vqrdmlahq_s16(store_0, low2, v_weight2);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);

                let item0 = vld1_u8(src_ptr0.as_ptr());
                let low0 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item0));
                store_0 = vqrdmlahq_s16(store_0, low0, v_weight0);

                let item1 = vld1_u8(src_ptr1.as_ptr());
                let low1 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item1));
                store_0 = vqrdmlahq_s16(store_0, low1, v_weight1);

                let item2 = vld1_u8(src_ptr2.as_ptr());
                let low2 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item2));
                store_0 = vqrdmlahq_s16(store_0, low2, v_weight2);

                let item3 = vld1_u8(src_ptr3.as_ptr());
                let low3 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item3));
                store_0 = vqrdmlahq_s16(store_0, low3, v_weight3);
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let item_row = vld1_u8(src_ptr.as_ptr());

                    let low = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item_row));
                    store_0 = vqrdmlahq_s16(store_0, low, v_weight);
                }
            }

            store_0 = vmaxq_s16(store_0, zeros);

            let item = vqshrun_n_s16::<R_SHR_SCALE>(store_0);
            vst1_u8(dst.as_mut_ptr(), item);

            cx += 8;
        }

        rem = rem.chunks_exact_mut(8).into_remainder();
        let iter_1 = rem.iter_mut();

        for dst in iter_1 {
            let vld = vdupq_n_s16(ROUNDING);
            let mut store = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);

                let items0 = vld1_dup_u8(src_ptr0.as_ptr());
                let low0 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(items0));
                store = vqrdmlahq_s16(store, low0, v_weight0);

                let items1 = vld1_dup_u8(src_ptr1.as_ptr());
                let low1 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(items1));
                store = vqrdmlahq_s16(store, low1, v_weight1);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);

                let items0 = vld1_dup_u8(src_ptr0.as_ptr());
                let low0 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(items0));
                store = vqrdmlahq_s16(store, low0, v_weight0);

                let items1 = vld1_dup_u8(src_ptr1.as_ptr());
                let low1 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(items1));
                store = vqrdmlahq_s16(store, low1, v_weight1);

                let items2 = vld1_dup_u8(src_ptr2.as_ptr());
                let low2 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(items2));
                store = vqrdmlahq_s16(store, low2, v_weight2);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);

                let items0 = vld1_dup_u8(src_ptr0.as_ptr());
                let low0 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(items0));
                store = vqrdmlahq_s16(store, low0, v_weight0);

                let items1 = vld1_dup_u8(src_ptr1.as_ptr());
                let low1 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(items1));
                store = vqrdmlahq_s16(store, low1, v_weight1);

                let items2 = vld1_dup_u8(src_ptr2.as_ptr());
                let low2 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(items2));
                store = vqrdmlahq_s16(store, low2, v_weight2);

                let items3 = vld1_dup_u8(src_ptr3.as_ptr());
                let low3 = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(items3));
                store = vqrdmlahq_s16(store, low3, v_weight3);
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let item_row = vld1_dup_u8(src_ptr.as_ptr());

                    let low = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(item_row));
                    store = vqrdmlahq_s16(store, low, v_weight);
                }
            }

            store = vmaxq_s16(store, zeros);

            let shrinked_store = vqshrun_n_s16::<R_SHR_SCALE>(store);
            let value = vget_lane_u8::<0>(shrinked_store);
            *dst = value;
            cx += 1;
        }
    }
}

fn convolve_vertical_neon_row_full(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    let mut cx = 0usize;

    unsafe {
        let iter_64 = dst.chunks_exact_mut(64);

        let bounds_size = bounds.size;

        for dst in iter_64 {
            let vld = vdupq_n_s32(ROUNDING_CONST);
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
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);

                let items0 = vld1q_u8_x4(src_ptr0.as_ptr());

                accumulate_4_into!(items0.0, store_0, store_1, store_2, store_3, v_weight0);
                accumulate_4_into!(items0.1, store_4, store_5, store_6, store_7, v_weight0);
                accumulate_4_into!(items0.2, store_8, store_9, store_10, store_11, v_weight0);
                accumulate_4_into!(items0.3, store_12, store_13, store_14, store_15, v_weight0);

                let items1 = vld1q_u8_x4(src_ptr1.as_ptr());

                accumulate_4_into!(items1.0, store_0, store_1, store_2, store_3, v_weight1);
                accumulate_4_into!(items1.1, store_4, store_5, store_6, store_7, v_weight1);
                accumulate_4_into!(items1.2, store_8, store_9, store_10, store_11, v_weight1);
                accumulate_4_into!(items1.3, store_12, store_13, store_14, store_15, v_weight1);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);

                let items0 = vld1q_u8_x4(src_ptr0.as_ptr());

                accumulate_4_into!(items0.0, store_0, store_1, store_2, store_3, v_weight0);
                accumulate_4_into!(items0.1, store_4, store_5, store_6, store_7, v_weight0);
                accumulate_4_into!(items0.2, store_8, store_9, store_10, store_11, v_weight0);
                accumulate_4_into!(items0.3, store_12, store_13, store_14, store_15, v_weight0);

                let items1 = vld1q_u8_x4(src_ptr1.as_ptr());

                accumulate_4_into!(items1.0, store_0, store_1, store_2, store_3, v_weight1);
                accumulate_4_into!(items1.1, store_4, store_5, store_6, store_7, v_weight1);
                accumulate_4_into!(items1.2, store_8, store_9, store_10, store_11, v_weight1);
                accumulate_4_into!(items1.3, store_12, store_13, store_14, store_15, v_weight1);

                let items2 = vld1q_u8_x4(src_ptr2.as_ptr());

                accumulate_4_into!(items2.0, store_0, store_1, store_2, store_3, v_weight2);
                accumulate_4_into!(items2.1, store_4, store_5, store_6, store_7, v_weight2);
                accumulate_4_into!(items2.2, store_8, store_9, store_10, store_11, v_weight2);
                accumulate_4_into!(items2.3, store_12, store_13, store_14, store_15, v_weight2);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);

                let items0 = vld1q_u8_x4(src_ptr0.as_ptr());

                accumulate_4_into!(items0.0, store_0, store_1, store_2, store_3, v_weight0);
                accumulate_4_into!(items0.1, store_4, store_5, store_6, store_7, v_weight0);
                accumulate_4_into!(items0.2, store_8, store_9, store_10, store_11, v_weight0);
                accumulate_4_into!(items0.3, store_12, store_13, store_14, store_15, v_weight0);

                let items1 = vld1q_u8_x4(src_ptr1.as_ptr());

                accumulate_4_into!(items1.0, store_0, store_1, store_2, store_3, v_weight1);
                accumulate_4_into!(items1.1, store_4, store_5, store_6, store_7, v_weight1);
                accumulate_4_into!(items1.2, store_8, store_9, store_10, store_11, v_weight1);
                accumulate_4_into!(items1.3, store_12, store_13, store_14, store_15, v_weight1);

                let items2 = vld1q_u8_x4(src_ptr2.as_ptr());

                accumulate_4_into!(items2.0, store_0, store_1, store_2, store_3, v_weight2);
                accumulate_4_into!(items2.1, store_4, store_5, store_6, store_7, v_weight2);
                accumulate_4_into!(items2.2, store_8, store_9, store_10, store_11, v_weight2);
                accumulate_4_into!(items2.3, store_12, store_13, store_14, store_15, v_weight2);

                let items3 = vld1q_u8_x4(src_ptr3.as_ptr());

                accumulate_4_into!(items3.0, store_0, store_1, store_2, store_3, v_weight3);
                accumulate_4_into!(items3.1, store_4, store_5, store_6, store_7, v_weight3);
                accumulate_4_into!(items3.2, store_8, store_9, store_10, store_11, v_weight3);
                accumulate_4_into!(items3.3, store_12, store_13, store_14, store_15, v_weight3);
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let items = vld1q_u8_x4(src_ptr.as_ptr());

                    accumulate_4_into!(items.0, store_0, store_1, store_2, store_3, v_weight);
                    accumulate_4_into!(items.1, store_4, store_5, store_6, store_7, v_weight);
                    accumulate_4_into!(items.2, store_8, store_9, store_10, store_11, v_weight);
                    accumulate_4_into!(items.3, store_12, store_13, store_14, store_15, v_weight);
                }
            }

            let item_0 = pack_weights!(store_0, store_1, store_2, store_3);
            let item_1 = pack_weights!(store_4, store_5, store_6, store_7);
            let item_2 = pack_weights!(store_8, store_9, store_10, store_11);
            let item_3 = pack_weights!(store_12, store_13, store_14, store_15);

            let dst_items = uint8x16x4_t(item_0, item_1, item_2, item_3);
            vst1q_u8_x4(dst.as_mut_ptr(), dst_items);

            cx += 64;
        }

        let mut rem = dst.chunks_exact_mut(64).into_remainder();
        let iter_32 = rem.chunks_exact_mut(32);

        for dst in iter_32 {
            let vld = vdupq_n_s32(ROUNDING_CONST);
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
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let items0 = vld1q_u8_x2(src_ptr0.as_ptr());

                accumulate_4_into!(items0.0, store_0, store_1, store_2, store_3, v_weight0);
                accumulate_4_into!(items0.1, store_4, store_5, store_6, store_7, v_weight0);

                let items1 = vld1q_u8_x2(src_ptr1.as_ptr());

                accumulate_4_into!(items1.0, store_0, store_1, store_2, store_3, v_weight1);
                accumulate_4_into!(items1.1, store_4, store_5, store_6, store_7, v_weight1);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let items0 = vld1q_u8_x2(src_ptr0.as_ptr());

                accumulate_4_into!(items0.0, store_0, store_1, store_2, store_3, v_weight0);
                accumulate_4_into!(items0.1, store_4, store_5, store_6, store_7, v_weight0);

                let items1 = vld1q_u8_x2(src_ptr1.as_ptr());

                accumulate_4_into!(items1.0, store_0, store_1, store_2, store_3, v_weight1);
                accumulate_4_into!(items1.1, store_4, store_5, store_6, store_7, v_weight1);

                let items2 = vld1q_u8_x2(src_ptr2.as_ptr());

                accumulate_4_into!(items2.0, store_0, store_1, store_2, store_3, v_weight2);
                accumulate_4_into!(items2.1, store_4, store_5, store_6, store_7, v_weight2);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);
                let items0 = vld1q_u8_x2(src_ptr0.as_ptr());

                accumulate_4_into!(items0.0, store_0, store_1, store_2, store_3, v_weight0);
                accumulate_4_into!(items0.1, store_4, store_5, store_6, store_7, v_weight0);

                let items1 = vld1q_u8_x2(src_ptr1.as_ptr());

                accumulate_4_into!(items1.0, store_0, store_1, store_2, store_3, v_weight1);
                accumulate_4_into!(items1.1, store_4, store_5, store_6, store_7, v_weight1);

                let items2 = vld1q_u8_x2(src_ptr2.as_ptr());

                accumulate_4_into!(items2.0, store_0, store_1, store_2, store_3, v_weight2);
                accumulate_4_into!(items2.1, store_4, store_5, store_6, store_7, v_weight2);

                let items3 = vld1q_u8_x2(src_ptr3.as_ptr());

                accumulate_4_into!(items3.0, store_0, store_1, store_2, store_3, v_weight3);
                accumulate_4_into!(items3.1, store_4, store_5, store_6, store_7, v_weight3);
            } else {
                for j in 0..bounds.size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let items = vld1q_u8_x2(src_ptr.as_ptr());

                    accumulate_4_into!(items.0, store_0, store_1, store_2, store_3, v_weight);
                    accumulate_4_into!(items.1, store_4, store_5, store_6, store_7, v_weight);
                }
            }

            let item_0 = pack_weights!(store_0, store_1, store_2, store_3);
            let item_1 = pack_weights!(store_4, store_5, store_6, store_7);

            let dst_items = uint8x16x2_t(item_0, item_1);
            vst1q_u8_x2(dst.as_mut_ptr(), dst_items);

            cx += 32;
        }

        rem = rem.chunks_exact_mut(32).into_remainder();
        let iter_16 = rem.chunks_exact_mut(16);

        for dst in iter_16 {
            let vld = vdupq_n_s32(ROUNDING_CONST);
            let mut store_0 = vld;
            let mut store_1 = vld;
            let mut store_2 = vld;
            let mut store_3 = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let item_row0 = vld1q_u8(src_ptr0.as_ptr());
                let item_row1 = vld1q_u8(src_ptr1.as_ptr());
                accumulate_4_into!(item_row0, store_0, store_1, store_2, store_3, v_weight0);
                accumulate_4_into!(item_row1, store_0, store_1, store_2, store_3, v_weight1);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let item_row0 = vld1q_u8(src_ptr0.as_ptr());
                let item_row1 = vld1q_u8(src_ptr1.as_ptr());
                let item_row2 = vld1q_u8(src_ptr2.as_ptr());
                accumulate_4_into!(item_row0, store_0, store_1, store_2, store_3, v_weight0);
                accumulate_4_into!(item_row1, store_0, store_1, store_2, store_3, v_weight1);
                accumulate_4_into!(item_row2, store_0, store_1, store_2, store_3, v_weight2);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + px)..);
                let item_row0 = vld1q_u8(src_ptr0.as_ptr());
                let item_row1 = vld1q_u8(src_ptr1.as_ptr());
                let item_row2 = vld1q_u8(src_ptr2.as_ptr());
                let item_row3 = vld1q_u8(src_ptr3.as_ptr());
                accumulate_4_into!(item_row0, store_0, store_1, store_2, store_3, v_weight0);
                accumulate_4_into!(item_row1, store_0, store_1, store_2, store_3, v_weight1);
                accumulate_4_into!(item_row2, store_0, store_1, store_2, store_3, v_weight2);
                accumulate_4_into!(item_row3, store_0, store_1, store_2, store_3, v_weight3);
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let item_row = vld1q_u8(src_ptr.as_ptr());
                    accumulate_4_into!(item_row, store_0, store_1, store_2, store_3, v_weight);
                }
            }

            let item = pack_weights!(store_0, store_1, store_2, store_3);

            vst1q_u8(dst.as_mut_ptr(), item);

            cx += 16;
        }

        rem = rem.chunks_exact_mut(16).into_remainder();
        let iter_8 = rem.chunks_exact_mut(8);

        for dst in iter_8 {
            let vld = vdupq_n_s32(ROUNDING_CONST);
            let mut store_0 = vld;
            let mut store_1 = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let item_row0 = vld1_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_u8(src_ptr1.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                store_0 = vmlal_s16(store_0, vget_low_s16(low0), vget_low_s16(v_weight0));
                store_1 = vmlal_high_s16(store_1, low0, v_weight0);
                store_0 = vmlal_s16(store_0, vget_low_s16(low1), vget_low_s16(v_weight1));
                store_1 = vmlal_high_s16(store_1, low1, v_weight1);
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let item_row0 = vld1_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_u8(src_ptr1.as_ptr());
                let item_row2 = vld1_u8(src_ptr2.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                let low2 = vreinterpretq_s16_u16(vmovl_u8(item_row2));
                store_0 = vmlal_s16(store_0, vget_low_s16(low0), vget_low_s16(v_weight0));
                store_1 = vmlal_high_s16(store_1, low0, v_weight0);
                store_0 = vmlal_s16(store_0, vget_low_s16(low1), vget_low_s16(v_weight1));
                store_1 = vmlal_high_s16(store_1, low1, v_weight1);
                store_0 = vmlal_s16(store_0, vget_low_s16(low2), vget_low_s16(v_weight2));
                store_1 = vmlal_high_s16(store_1, low2, v_weight2);
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
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
                store_0 = vmlal_s16(store_0, vget_low_s16(low0), vget_low_s16(v_weight0));
                store_1 = vmlal_high_s16(store_1, low0, v_weight0);
                store_0 = vmlal_s16(store_0, vget_low_s16(low1), vget_low_s16(v_weight1));
                store_1 = vmlal_high_s16(store_1, low1, v_weight1);
                store_0 = vmlal_s16(store_0, vget_low_s16(low2), vget_low_s16(v_weight2));
                store_1 = vmlal_high_s16(store_1, low2, v_weight2);
                store_0 = vmlal_s16(store_0, vget_low_s16(low3), vget_low_s16(v_weight3));
                store_1 = vmlal_high_s16(store_1, low3, v_weight3);
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let item_row = vld1_u8(src_ptr.as_ptr());

                    let low = vreinterpretq_s16_u16(vmovl_u8(item_row));
                    store_0 = vmlal_s16(store_0, vget_low_s16(low), vget_low_s16(v_weight));
                    store_1 = vmlal_high_s16(store_1, low, v_weight);
                }
            }

            let zeros = vdupq_n_s16(0);

            let low_s16 = vcombine_s16(
                vqshrn_n_s32::<PRECISION>(store_0),
                vqshrn_n_s32::<PRECISION>(store_1),
            );
            let low_16 = vreinterpretq_u16_s16(vmaxq_s16(low_s16, zeros));

            let item = vqmovn_u16(low_16);

            vst1_u8(dst.as_mut_ptr(), item);

            cx += 8;
        }

        rem = rem.chunks_exact_mut(8).into_remainder();
        let iter_1 = rem.iter_mut();

        for dst in iter_1 {
            let vld = vdupq_n_s32(ROUNDING_CONST);
            let mut store = vld;

            let px = cx;

            if bounds_size == 2 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..2);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let item_row0 = vld1_dup_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_dup_u8(src_ptr1.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                store = vmlal_s16(store, vget_low_s16(low0), vget_low_s16(v_weight0));
                store = vmlal_s16(store, vget_low_s16(low1), vget_low_s16(v_weight1));
            } else if bounds_size == 3 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..3);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let src_ptr0 = src.get_unchecked((src_stride * py + px)..);
                let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + px)..);
                let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + px)..);
                let item_row0 = vld1_dup_u8(src_ptr0.as_ptr());
                let item_row1 = vld1_dup_u8(src_ptr1.as_ptr());
                let item_row2 = vld1_dup_u8(src_ptr2.as_ptr());

                let low0 = vreinterpretq_s16_u16(vmovl_u8(item_row0));
                let low1 = vreinterpretq_s16_u16(vmovl_u8(item_row1));
                let low2 = vreinterpretq_s16_u16(vmovl_u8(item_row2));
                store = vmlal_s16(store, vget_low_s16(low0), vget_low_s16(v_weight0));
                store = vmlal_s16(store, vget_low_s16(low1), vget_low_s16(v_weight1));
                store = vmlal_s16(store, vget_low_s16(low2), vget_low_s16(v_weight2));
            } else if bounds_size == 4 {
                let py = bounds.start;
                let weight = weight.get_unchecked(0..4);
                let v_weight0 = vld1q_dup_s16(weight.as_ptr());
                let v_weight1 = vld1q_dup_s16(weight.as_ptr().add(1));
                let v_weight2 = vld1q_dup_s16(weight.as_ptr().add(2));
                let v_weight3 = vld1q_dup_s16(weight.as_ptr().add(3));
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
                store = vmlal_s16(store, vget_low_s16(low0), vget_low_s16(v_weight0));
                store = vmlal_s16(store, vget_low_s16(low1), vget_low_s16(v_weight1));
                store = vmlal_s16(store, vget_low_s16(low2), vget_low_s16(v_weight2));
                store = vmlal_s16(store, vget_low_s16(low3), vget_low_s16(v_weight3));
            } else {
                for j in 0..bounds_size {
                    let py = bounds.start + j;
                    let weight = weight.get_unchecked(j..);
                    let v_weight = vld1q_dup_s16(weight.as_ptr());
                    let src_ptr = src.get_unchecked((src_stride * py + px)..);
                    let item_row = vld1_dup_u8(src_ptr.as_ptr());

                    let low = vreinterpretq_s16_u16(vmovl_u8(item_row));
                    store = vmlal_s16(store, vget_low_s16(low), vget_low_s16(v_weight));
                }
            }

            let zeros = vdupq_n_s32(0);

            store = vmaxq_s32(store, zeros);

            let shrinked_store = vqshrun_n_s32::<PRECISION>(store);

            let low_16 = vcombine_u16(shrinked_store, shrinked_store);

            let item = vqmovn_u16(low_16);

            let value = vget_lane_u8::<0>(item);
            *dst = value;
            cx += 1;
        }
    }
}
