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
use crate::support::{PRECISION, ROUNDING_APPROX};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn consume_u16_8(
    start_y: usize,
    start_x: usize,
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    filter: *const i16,
    bounds: &FilterBounds,
    max_colors: i32,
) {
    let vld = vdupq_n_s64(ROUNDING_APPROX as i64);
    let mut store_0 = vld;
    let mut store_1 = vld;
    let mut store_2 = vld;
    let mut store_3 = vld;

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j);
        let v_weight = vmovl_s16(vld1_dup_s16(weight));
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = vld1q_u16(s_ptr);

        let item_low = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(item_row)));
        let item_high = vreinterpretq_s32_u32(vmovl_high_u16(item_row));

        store_0 = vmlal_s32(store_0, vget_low_s32(item_low), vget_low_s32(v_weight));
        store_1 = vmlal_high_s32(store_1, item_low, v_weight);

        store_2 = vmlal_s32(store_2, vget_low_s32(item_high), vget_low_s32(v_weight));
        store_3 = vmlal_high_s32(store_3, item_high, v_weight);
    }

    let zeros = vdupq_n_s32(0);
    let v_max_colors = vdupq_n_s32(max_colors);
    let n_store_0 = vqshrn_n_s64::<PRECISION>(store_0);
    let n_store_1 = vqshrn_n_s64::<PRECISION>(store_1);
    let n_store_2 = vqshrn_n_s64::<PRECISION>(store_2);
    let n_store_3 = vqshrn_n_s64::<PRECISION>(store_3);

    let mut new_store_0 = vcombine_s32(n_store_0, n_store_1);
    new_store_0 = vminq_s32(vmaxq_s32(new_store_0, zeros), v_max_colors);

    let mut new_store_1 = vcombine_s32(n_store_2, n_store_3);
    new_store_1 = vminq_s32(vmaxq_s32(new_store_1, zeros), v_max_colors);

    let store_u16 = vcombine_u16(
        vmovn_u32(vreinterpretq_u32_s32(new_store_0)),
        vmovn_u32(vreinterpretq_u32_s32(new_store_1)),
    );

    let dst_ptr = dst.add(px);
    vst1q_u16(dst_ptr, store_u16);
}

#[inline(always)]
unsafe fn consume_u16_4(
    start_y: usize,
    start_x: usize,
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    filter: *const i16,
    bounds: &FilterBounds,
    max_colors: i32,
) {
    let vld = vdupq_n_s64(ROUNDING_APPROX as i64);
    let mut store_0 = vld;
    let mut store_1 = vld;

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j);
        let v_weight = vmovl_s16(vld1_dup_s16(weight));
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = vld1_u16(s_ptr);

        let item_row_rescaled = vreinterpretq_s32_u32(vmovl_u16(item_row));

        store_0 = vmlal_s32(
            store_0,
            vget_low_s32(item_row_rescaled),
            vget_low_s32(v_weight),
        );
        store_1 = vmlal_high_s32(store_1, item_row_rescaled, v_weight);
    }

    let zeros = vdupq_n_s32(0);
    let v_max_colors = vdupq_n_s32(max_colors);
    let n_store_0 = vqshrn_n_s64::<PRECISION>(store_0);
    let n_store_1 = vqshrn_n_s64::<PRECISION>(store_1);
    let mut new_store = vcombine_s32(n_store_0, n_store_1);
    new_store = vminq_s32(vmaxq_s32(new_store, zeros), v_max_colors);

    let store_u16 = vmovn_u32(vreinterpretq_u32_s32(new_store));

    let dst_ptr = dst.add(px);
    vst1_u16(dst_ptr, store_u16);
}

#[inline(always)]
unsafe fn consume_u16_1(
    start_y: usize,
    start_x: usize,
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    filter: *const i16,
    bounds: &FilterBounds,
    max_colors: i32,
) {
    let vld = vdupq_n_s64(ROUNDING_APPROX as i64);
    let mut store = vld;

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = filter.add(j);
        let v_weight = vmovl_s16(vld1_dup_s16(weight));
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = vld1_dup_u16(s_ptr);

        let low = vreinterpretq_s32_u32(vmovl_u16(item_row));
        store = vmlal_s32(store, vget_low_s32(low), vget_low_s32(v_weight));
    }

    let zeros = vdup_n_s32(0);
    let v_max_colors = vdup_n_s32(max_colors);

    let mut shrinked_store = vqshrn_n_s64::<PRECISION>(store);
    shrinked_store = vmin_s32(vmax_s32(shrinked_store, zeros), v_max_colors);
    let dst_ptr = dst.add(px);
    let value = vget_lane_s32::<0>(shrinked_store);
    dst_ptr.write_unaligned(value as u16);
}

#[inline(always)]
pub fn convolve_vertical_rgb_neon_row_u16<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const u16,
    unsafe_destination_ptr_0: *mut u16,
    src_stride: usize,
    weight_ptr: *const i16,
    bit_depth: usize,
) {
    let max_colors = 2i32.pow(bit_depth as u32) - 1i32;
    let mut cx = 0usize;
    let dst_width = width * CHANNELS;

    while cx + 8 < dst_width {
        unsafe {
            consume_u16_8(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
                max_colors,
            );
        }

        cx += 8;
    }

    while cx + 4 < dst_width {
        unsafe {
            consume_u16_4(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
                max_colors,
            );
        }

        cx += 4;
    }

    while cx < dst_width {
        unsafe {
            consume_u16_1(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
                max_colors,
            );
        }
        cx += 1;
    }
}
