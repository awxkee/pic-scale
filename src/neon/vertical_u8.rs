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

macro_rules! consume_64_u8 {
    ($start_y: expr,$start_x: expr, $src: expr, $src_stride: expr, $dst: expr, $filter: expr, $bounds: expr) => {{
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

        let px = $start_x;

        for j in 0..$bounds.size {
            let py = $start_y + j;
            let weight = $filter.add(j);
            let v_weight = vld1q_dup_s16(weight);
            let src_ptr = $src.add($src_stride * py);

            let s_ptr = src_ptr.add(px);
            let items = vld1q_u8_x4(s_ptr);

            accumulate_4_into!(items.0, store_0, store_1, store_2, store_3, v_weight);
            accumulate_4_into!(items.1, store_4, store_5, store_6, store_7, v_weight);
            accumulate_4_into!(items.2, store_8, store_9, store_10, store_11, v_weight);
            accumulate_4_into!(items.3, store_12, store_13, store_14, store_15, v_weight);
        }

        let item_0 = pack_weights!(store_0, store_1, store_2, store_3);
        let item_1 = pack_weights!(store_4, store_5, store_6, store_7);
        let item_2 = pack_weights!(store_8, store_9, store_10, store_11);
        let item_3 = pack_weights!(store_12, store_13, store_14, store_15);

        let dst_ptr = $dst.add(px);

        let dst_items = uint8x16x4_t(item_0, item_1, item_2, item_3);
        vst1q_u8_x4(dst_ptr, dst_items);
    }};
}

macro_rules! consume_32_u8 {
    ($start_y: expr,$start_x: expr, $src: expr, $src_stride: expr, $dst: expr, $filter: expr, $bounds: expr) => {{
        let vld = vdupq_n_s32(ROUNDING_CONST);
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;
        let mut store_4 = vld;
        let mut store_5 = vld;
        let mut store_6 = vld;
        let mut store_7 = vld;

        let px = $start_x;

        for j in 0..$bounds.size {
            let py = $start_y + j;
            let weight = $filter.add(j);
            let v_weight = vld1q_dup_s16(weight);
            let src_ptr = $src.add($src_stride * py);

            let s_ptr = src_ptr.add(px);
            let items = vld1q_u8_x2(s_ptr);

            accumulate_4_into!(items.0, store_0, store_1, store_2, store_3, v_weight);
            accumulate_4_into!(items.1, store_4, store_5, store_6, store_7, v_weight);
        }

        let item_0 = pack_weights!(store_0, store_1, store_2, store_3);
        let item_1 = pack_weights!(store_4, store_5, store_6, store_7);

        let dst_ptr = $dst.add(px);

        let dst_items = uint8x16x2_t(item_0, item_1);
        vst1q_u8_x2(dst_ptr, dst_items);
    }};
}

macro_rules! consume_16_u8 {
    ($start_y: expr,$start_x: expr, $src: expr, $src_stride: expr, $dst: expr, $filter: expr, $bounds: expr) => {{
        let vld = vdupq_n_s32(ROUNDING_CONST);
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;

        let px = $start_x;

        for j in 0..$bounds.size {
            let py = $start_y + j;
            let weight = $filter.add(j);
            let v_weight = vld1q_dup_s16(weight);
            let src_ptr = $src.add($src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row = vld1q_u8(s_ptr);
            accumulate_4_into!(item_row, store_0, store_1, store_2, store_3, v_weight);
        }

        let item = pack_weights!(store_0, store_1, store_2, store_3);

        let dst_ptr = $dst.add(px);
        vst1q_u8(dst_ptr, item);
    }};
}

macro_rules! consume_u8_8 {
    ($start_y: expr,$start_x: expr, $src: expr, $src_stride: expr, $dst: expr, $filter: expr, $bounds: expr) => {{
        let vld = vdupq_n_s32(ROUNDING_CONST);
        let mut store_0 = vld;
        let mut store_1 = vld;

        let px = $start_x;

        for j in 0..$bounds.size {
            let py = $start_y + j;
            let weight = $filter.add(j);
            let v_weight = vld1q_dup_s16(weight);
            let src_ptr = $src.add($src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row = vld1_u8(s_ptr);

            let low = vreinterpretq_s16_u16(vmovl_u8(item_row));
            store_0 = vmlal_s16(store_0, vget_low_s16(low), vget_low_s16(v_weight));
            store_1 = vmlal_high_s16(store_1, low, v_weight);
        }

        let zeros = vdupq_n_s16(0);

        let low_s16 = vcombine_s16(
            vqshrn_n_s32::<PRECISION>(store_0),
            vqshrn_n_s32::<PRECISION>(store_1),
        );
        let low_16 = vreinterpretq_u16_s16(vmaxq_s16(low_s16, zeros));

        let item = vqmovn_u16(low_16);

        let dst_ptr = $dst.add(px);
        vst1_u8(dst_ptr, item);
    }};
}

macro_rules! consume_u8_1 {
    ($start_y: expr, $start_x: expr, $src: expr, $src_stride: expr, $dst: expr, $filter: expr, $bounds: expr) => {{
        let vld = vdupq_n_s32(ROUNDING_CONST);
        let mut store = vld;

        let px = $start_x;

        for j in 0..$bounds.size {
            let py = $start_y + j;
            let weight = $filter.add(j);
            let v_weight = vld1q_dup_s16(weight);
            let src_ptr = $src.add($src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row = vld1_dup_u8(s_ptr);

            let low = vreinterpretq_s16_u16(vmovl_u8(item_row));
            store = vmlal_s16(store, vget_low_s16(low), vget_low_s16(v_weight));
        }

        let zeros = vdupq_n_s32(0);

        store = vmaxq_s32(store, zeros);

        let shrinked_store = vqshrun_n_s32::<PRECISION>(store);

        let low_16 = vcombine_u16(shrinked_store, shrinked_store);

        let item = vqmovn_u16(low_16);

        let dst_ptr = $dst.add(px);
        let value = vget_lane_u8::<0>(item);
        dst_ptr.write_unaligned(value);
    }};
}

#[inline]
pub fn convolve_vertical_neon_row<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
    src_stride: usize,
    weight_ptr: *const i16,
) {
    let mut cx = 0usize;
    let dst_width = width * CHANNELS;

    while cx + 64 < dst_width {
        unsafe {
            consume_64_u8!(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds
            );
        }

        cx += 64;
    }

    while cx + 32 < dst_width {
        unsafe {
            consume_32_u8!(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds
            );
        }

        cx += 32;
    }

    while cx + 16 < dst_width {
        unsafe {
            consume_16_u8!(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds
            );
        }

        cx += 16;
    }

    while cx + 8 < dst_width {
        unsafe {
            consume_u8_8!(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds
            );
        }

        cx += 8;
    }

    while cx < dst_width {
        unsafe {
            consume_u8_1!(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds
            );
        }
        cx += 1;
    }
}
