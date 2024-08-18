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

use crate::filter_weights::FilterWeights;
use crate::neon::utils::load_3b_as_u16x4;
use crate::support::{PRECISION, ROUNDING_APPROX};
use std::arch::aarch64::*;

macro_rules! conv_horiz_rgba_5_u8 {
    ($start_x: expr, $src: expr, $w0: expr, $w1: expr, $w2: expr, $w3: expr, $w4: expr, $store: expr, $shuffle: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let pixel = vld1q_u8(src_ptr);
        let rgb_pixel = vqtbl1q_u8(pixel, $shuffle);
        let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb_pixel));
        let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb_pixel)));
        let highest = vreinterpret_s16_u16(vget_high_u16(vmovl_high_u8(pixel)));

        let mut acc = vmlal_high_s16($store, hi, $w3);
        acc = vmlal_s16(acc, vget_low_s16(hi), $w2);
        acc = vmlal_high_s16(acc, lo, $w1);
        acc = vmlal_s16(acc, vget_low_s16(lo), $w0);
        acc = vmlal_s16(acc, highest, $w4);
        acc
    }};
}

macro_rules! conv_horiz_rgba_4_u8 {
    ($start_x: expr, $src: expr, $w0: expr, $w1: expr, $w2: expr, $w3: expr, $store: expr, $shuffle: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let mut rgb_pixel = vld1q_u8(src_ptr);
        rgb_pixel = vqtbl1q_u8(rgb_pixel, $shuffle);
        let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb_pixel));
        let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb_pixel)));

        let acc = vmlal_high_s16($store, hi, $w3);
        let acc = vmlal_s16(acc, vget_low_s16(hi), $w2);
        let acc = vmlal_high_s16(acc, lo, $w1);
        let acc = vmlal_s16(acc, vget_low_s16(lo), $w0);
        acc
    }};
}

macro_rules! conv_horiz_rgba_2_u8 {
    ($start_x: expr, $src: expr, $w0: expr, $w1: expr, $store: expr, $shuffle: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let mut rgb_pixel = vld1_u8(src_ptr);
        rgb_pixel = vtbl1_u8(rgb_pixel, $shuffle);
        let wide = vreinterpretq_s16_u16(vmovl_u8(rgb_pixel));

        let acc = vmlal_high_s16($store, wide, $w1);
        let acc = vmlal_s16(acc, vget_low_s16(wide), $w0);
        acc
    }};
}

macro_rules! conv_horiz_rgba_1_u8 {
    ($start_x: expr, $src: expr, $w0: expr, $store: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);
        let rgb_pixel = load_3b_as_u16x4(src_ptr);
        let lo = vreinterpret_s16_u16(rgb_pixel);
        let acc = vmlal_s16($store, lo, $w0);
        acc
    }};
}

macro_rules! write_accumulator_u8 {
    ($store: expr, $dst: expr) => {{
        let zeros = vdupq_n_s32(0i32);
        let store_16 = vqshrun_n_s32::<PRECISION>(vmaxq_s32($store, zeros));
        let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));
        let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
        let bytes = pixel.to_le_bytes();
        let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
        ($dst as *mut u16).write_unaligned(first_byte);
        $dst.add(2).write_unaligned(bytes[2]);
    }};
}

pub fn convolve_horizontal_rgb_neon_rows_4(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    unsafe {
        let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
        let shuffle_1 = vld1_u8(shuf_table_1.as_ptr());
        let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
        let shuffle_2 = vld1_u8(shuf_table_2.as_ptr());
        let shuffle = vcombine_u8(shuffle_1, shuffle_2);

        // (r0 g0 b0 r1) (g2 b2 r3 g3) (b3 r4 g4 b4) (r5 g5 b5 r6)

        let mut filter_offset = 0usize;
        let weights_ptr = approx_weights.weights.as_ptr();
        const CHANNELS: usize = 3;
        let init = vdupq_n_s32(ROUNDING_APPROX);
        for x in 0..dst_width {
            let bounds = approx_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = init;
            let mut store_1 = init;
            let mut store_2 = init;
            let mut store_3 = init;

            while jx + 5 < bounds.size && bounds.start + jx + 6 < src_width {
                let bnds = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1_s16(ptr);
                let w0 = vdup_lane_s16::<0>(weights);
                let w1 = vdupq_lane_s16::<1>(weights);
                let w2 = vdup_lane_s16::<2>(weights);
                let w3 = vdupq_lane_s16::<3>(weights);
                let w4 = vld1_dup_s16(ptr.add(4));
                let ptr_0 = unsafe_source_ptr_0;
                store_0 = conv_horiz_rgba_5_u8!(bnds, ptr_0, w0, w1, w2, w3, w4, store_0, shuffle);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_5_u8!(bnds, ptr_1, w0, w1, w2, w3, w4, store_1, shuffle);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_5_u8!(bnds, ptr_2, w0, w1, w2, w3, w4, store_2, shuffle);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_5_u8!(bnds, ptr_3, w0, w1, w2, w3, w4, store_3, shuffle);
                jx += 5;
            }

            while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1_s16(ptr);
                let w0 = vdup_lane_s16::<0>(weights);
                let w1 = vdupq_lane_s16::<1>(weights);
                let w2 = vdup_lane_s16::<2>(weights);
                let w3 = vdupq_lane_s16::<3>(weights);
                let ptr_0 = unsafe_source_ptr_0;
                store_0 =
                    conv_horiz_rgba_4_u8!(bounds_start, ptr_0, w0, w1, w2, w3, store_0, shuffle);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 =
                    conv_horiz_rgba_4_u8!(bounds_start, ptr_1, w0, w1, w2, w3, store_1, shuffle);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 =
                    conv_horiz_rgba_4_u8!(bounds_start, ptr_2, w0, w1, w2, w3, store_2, shuffle);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 =
                    conv_horiz_rgba_4_u8!(bounds_start, ptr_3, w0, w1, w2, w3, store_3, shuffle);
                jx += 4;
            }

            while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bnds = bounds.start + jx;
                let w0 = vld1_dup_s16(ptr);
                let w1 = vld1q_dup_s16(ptr.add(1));
                store_0 =
                    conv_horiz_rgba_2_u8!(bnds, unsafe_source_ptr_0, w0, w1, store_0, shuffle_1);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_2_u8!(bnds, ptr_1, w0, w1, store_1, shuffle_1);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_2_u8!(bnds, ptr_2, w0, w1, store_2, shuffle_1);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_2_u8!(bnds, ptr_3, w0, w1, store_3, shuffle_1);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bnds = bounds.start + jx;
                let weight0 = vld1_dup_s16(ptr);
                store_0 = conv_horiz_rgba_1_u8!(bnds, unsafe_source_ptr_0, weight0, store_0);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_1_u8!(bnds, ptr_1, weight0, store_1);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_1_u8!(bnds, ptr_2, weight0, store_2);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_1_u8!(bnds, ptr_3, weight0, store_3);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            write_accumulator_u8!(store_0, dest_ptr);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            write_accumulator_u8!(store_1, dest_ptr);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            write_accumulator_u8!(store_2, dest_ptr);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            write_accumulator_u8!(store_3, dest_ptr);

            filter_offset += approx_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgb_neon_row_one(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    unsafe {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;
        let weights_ptr = approx_weights.weights.as_ptr();

        let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
        let shuffle_1 = vld1_u8(shuf_table_1.as_ptr());
        let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
        let shuffle_2 = vld1_u8(shuf_table_2.as_ptr());
        let shuffle = vcombine_u8(shuffle_1, shuffle_2);

        for x in 0..dst_width {
            let bounds = approx_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vdupq_n_s32(ROUNDING_APPROX);

            while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1_s16(ptr);
                let w0 = vdup_lane_s16::<0>(weights);
                let w1 = vdupq_lane_s16::<1>(weights);
                let w2 = vdup_lane_s16::<2>(weights);
                let w3 = vdupq_lane_s16::<3>(weights);
                let ptr_0 = unsafe_source_ptr_0;
                store = conv_horiz_rgba_4_u8!(bounds_start, ptr_0, w0, w1, w2, w3, store, shuffle);
                jx += 4;
            }

            while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bnds = bounds.start + jx;
                let w0 = vld1_dup_s16(ptr);
                let w1 = vld1q_dup_s16(ptr.add(1));
                store = conv_horiz_rgba_2_u8!(bnds, unsafe_source_ptr_0, w0, w1, store, shuffle_1);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vld1_dup_s16(ptr);
                let bnds = bounds.start + jx;
                store = conv_horiz_rgba_1_u8!(bnds, unsafe_source_ptr_0, weight0, store);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            write_accumulator_u8!(store, dest_ptr);

            filter_offset += approx_weights.aligned_size;
        }
    }
}
