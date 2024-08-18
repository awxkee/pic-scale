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
use crate::support::PRECISION;
use crate::support::ROUNDING_APPROX;
use std::arch::aarch64::*;
use crate::neon::utils::load_4b_as_u16x4;

macro_rules! conv_horiz_rgba_12_u8 {
    ($start_x: expr, $src: expr, $weights_ptr: expr, $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgba_pixel = vld1q_u8_x3(src_ptr);
        let weights = vld1q_s16($weights_ptr);
        let weights_1 = vld1_s16($weights_ptr.add(8));

        let hi0 = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.0));
        let lo0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.0)));
        let hi1 = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.1));
        let lo1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.1)));
        let hi2 = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.2));
        let lo3 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.2)));

        let w0 = vdupq_laneq_s16::<3>(weights);
        let mut acc = vmlal_high_s16($store, hi0, w0);
        let w1 = vdup_laneq_s16::<2>(weights);
        acc = vmlal_s16(acc, vget_low_s16(hi0), w1);
        let w2 = vdupq_laneq_s16::<1>(weights);
        acc = vmlal_high_s16(acc, lo0, w2);
        let w3 = vdup_laneq_s16::<0>(weights);
        acc = vmlal_s16(acc, vget_low_s16(lo0), w3);

        let w4 = vdupq_laneq_s16::<7>(weights);
        acc = vmlal_high_s16(acc, hi1, w4);
        let w5 = vdup_laneq_s16::<6>(weights);
        acc = vmlal_s16(acc, vget_low_s16(hi1), w5);
        let w6 = vdupq_laneq_s16::<5>(weights);
        acc = vmlal_high_s16(acc, lo1, w6);
        let w7 = vdup_laneq_s16::<4>(weights);
        acc = vmlal_s16(acc, vget_low_s16(lo1), w7);

        let w0 = vdupq_lane_s16::<3>(weights_1);
        acc = vmlal_high_s16(acc, hi2, w0);
        let w1 = vdup_lane_s16::<2>(weights_1);
        acc = vmlal_s16(acc, vget_low_s16(hi2), w1);
        let w2 = vdupq_lane_s16::<1>(weights_1);
        acc = vmlal_high_s16(acc, lo3, w2);
        let w3 = vdup_lane_s16::<0>(weights_1);
        acc = vmlal_s16(acc, vget_low_s16(lo3), w3);

        acc
    }};
}

macro_rules! conv_horiz_rgba_8_u8 {
    ($start_x: expr, $src: expr, $set1: expr, $set2: expr, $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgba_pixel = vld1q_u8_x2(src_ptr);

        let hi0 = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.0));
        let lo0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.0)));
        let hi1 = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.1));
        let lo1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.1)));

        let mut acc = vmlal_high_s16($store, hi0, $set1.3);
        acc = vmlal_s16(acc, vget_low_s16(hi0), vget_low_s16($set1.2));
        acc = vmlal_high_s16(acc, lo0, $set1.1);
        acc = vmlal_s16(acc, vget_low_s16(lo0), vget_low_s16($set1.0));

        acc = vmlal_high_s16(acc, hi1, $set2.3);
        acc = vmlal_s16(acc, vget_low_s16(hi1), vget_low_s16($set2.2));
        acc = vmlal_high_s16(acc, lo1, $set2.1);
        acc = vmlal_s16(acc, vget_low_s16(lo1), vget_low_s16($set2.0));
        acc
    }};
}

macro_rules! conv_horiz_rgba_4_u8 {
    ($start_x: expr, $src: expr, $w0: expr, $w1: expr, $w2: expr, $w3: expr, $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgba_pixel = vld1q_u8(src_ptr);

        let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel));
        let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel)));

        let acc = vmlal_high_s16($store, hi, $w3);
        let acc = vmlal_s16(acc, vget_low_s16(hi), $w2);
        let acc = vmlal_high_s16(acc, lo, $w1);
        let acc = vmlal_s16(acc, vget_low_s16(lo), $w0);
        acc
    }};
}

macro_rules! conv_horiz_rgba_2_u8 {
    ($start_x: expr, $src: expr, $w0: expr, $w1: expr, $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel = vld1_u8(src_ptr);
        let wide = vreinterpretq_s16_u16(vmovl_u8(rgb_pixel));

        let acc = vmlal_high_s16($store, wide, $w1);
        let acc = vmlal_s16(acc, vget_low_s16(wide), $w0);
        acc
    }};
}

macro_rules! conv_horiz_rgba_1_u8 {
    ($start_x: expr, $src: expr, $w0: expr,$store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);
        let rgba_pixel = load_4b_as_u16x4(src_ptr);
        let lo = vreinterpret_s16_u16(rgba_pixel);

        let acc = vmlal_s16($store, lo, $w0);
        acc
    }};
}

pub fn convolve_horizontal_rgba_neon_rows_4_u8(
    dst_width: usize,
    _: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = approx_weights.weights.as_ptr();
        const CHANNELS: usize = 4;
        let zeros = vdupq_n_s32(0i32);
        let init = vdupq_n_s32(ROUNDING_APPROX);
        for x in 0..dst_width {
            let bounds = approx_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = init;
            let mut store_1 = init;
            let mut store_2 = init;
            let mut store_3 = init;

            while jx + 12 < bounds.size {
                let bounds_start = bounds.start + jx;
                let weights_ptr = weights_ptr.add(jx + filter_offset);
                let ptr_0 = unsafe_source_ptr_0;
                store_0 = conv_horiz_rgba_12_u8!(bounds_start, ptr_0, weights_ptr, store_0);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_12_u8!(bounds_start, ptr_1, weights_ptr, store_1);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_12_u8!(bounds_start, ptr_2, weights_ptr, store_2);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_12_u8!(bounds_start, ptr_3, weights_ptr, store_3);
                jx += 12;
            }

            while jx + 8 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights_set = vld1q_s16(ptr);
                let w0 = vdupq_laneq_s16::<0>(weights_set);
                let w1 = vdupq_laneq_s16::<1>(weights_set);
                let w2 = vdupq_laneq_s16::<2>(weights_set);
                let w3 = vdupq_laneq_s16::<3>(weights_set);
                let w4 = vdupq_laneq_s16::<4>(weights_set);
                let w5 = vdupq_laneq_s16::<5>(weights_set);
                let w6 = vdupq_laneq_s16::<6>(weights_set);
                let w7 = vdupq_laneq_s16::<7>(weights_set);
                let set1 = (w0, w1, w2, w3);
                let set2 = (w4, w5, w6, w7);
                let ptr_0 = unsafe_source_ptr_0;
                store_0 = conv_horiz_rgba_8_u8!(bounds_start, ptr_0, set1, set2, store_0);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_8_u8!(bounds_start, ptr_1, set1, set2, store_1);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_8_u8!(bounds_start, ptr_2, set1, set2, store_2);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_8_u8!(bounds_start, ptr_3, set1, set2, store_3);
                jx += 8;
            }

            while jx + 4 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1_s16(ptr);
                let w0 = vdup_lane_s16::<0>(weights);
                let w1 = vdupq_lane_s16::<1>(weights);
                let w2 = vdup_lane_s16::<2>(weights);
                let w3 = vdupq_lane_s16::<3>(weights);
                let ptr_0 = unsafe_source_ptr_0;
                store_0 = conv_horiz_rgba_4_u8!(bounds_start, ptr_0, w0, w1, w2, w3, store_0);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_4_u8!(bounds_start, ptr_1, w0, w1, w2, w3, store_1);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_4_u8!(bounds_start, ptr_2, w0, w1, w2, w3, store_2);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_4_u8!(bounds_start, ptr_3, w0, w1, w2, w3, store_3);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let w0 = vld1_dup_s16(ptr);
                let w1 = vld1q_dup_s16(ptr.add(1));
                let ptr_0 = unsafe_source_ptr_0;
                store_0 = conv_horiz_rgba_2_u8!(bounds_start, ptr_0, w0, w1, store_0);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_2_u8!(bounds_start, ptr_1, w0, w1, store_1);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_2_u8!(bounds_start, ptr_2, w0, w1, store_2);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_2_u8!(bounds_start, ptr_3, w0, w1, store_3);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(ptr);
                let ptr_0 = unsafe_source_ptr_0;
                store_0 = conv_horiz_rgba_1_u8!(bounds_start, ptr_0, weight0, store_0);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_1_u8!(bounds_start, ptr_1, weight0, store_1);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_1_u8!(bounds_start, ptr_2, weight0, store_2);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_1_u8!(bounds_start, ptr_3, weight0, store_3);
                jx += 1;
            }

            let store_16_0 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_0, zeros));
            let store_16_1 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_1, zeros));
            let store_16_2 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_2, zeros));
            let store_16_3 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_3, zeros));

            let store_16_8_0 = vqmovn_u16(vcombine_u16(store_16_0, store_16_0));
            let store_16_8_1 = vqmovn_u16(vcombine_u16(store_16_1, store_16_1));
            let store_16_8_2 = vqmovn_u16(vcombine_u16(store_16_2, store_16_2));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16_3, store_16_3));

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8_0));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8_1));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8_2));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            filter_offset += approx_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgba_neon_row(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;

        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vdupq_n_s32(ROUNDING_APPROX);

            while jx + 8 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights_set = vld1q_s16(ptr);
                let w0 = vdupq_laneq_s16::<0>(weights_set);
                let w1 = vdupq_laneq_s16::<1>(weights_set);
                let w2 = vdupq_laneq_s16::<2>(weights_set);
                let w3 = vdupq_laneq_s16::<3>(weights_set);
                let w4 = vdupq_laneq_s16::<4>(weights_set);
                let w5 = vdupq_laneq_s16::<5>(weights_set);
                let w6 = vdupq_laneq_s16::<6>(weights_set);
                let w7 = vdupq_laneq_s16::<7>(weights_set);
                let set1 = (w0, w1, w2, w3);
                let set2 = (w4, w5, w6, w7);
                let ptr_0 = unsafe_source_ptr_0;
                store = conv_horiz_rgba_8_u8!(bounds_start, ptr_0, set1, set2, store);
                jx += 8;
            }

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1_s16(ptr);
                let weight0 = vdup_lane_s16::<0>(weights);
                let weight1 = vdupq_lane_s16::<1>(weights);
                let weight2 = vdup_lane_s16::<2>(weights);
                let weight3 = vdupq_lane_s16::<3>(weights);
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_4_u8!(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store
                );
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(ptr);
                let weight1 = vld1q_dup_s16(ptr.add(1));
                store = conv_horiz_rgba_2_u8!(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    store
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vld1_dup_s16(ptr);
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_1_u8!(bounds_start, unsafe_source_ptr_0, weight0, store);
                jx += 1;
            }

            let store_16 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store, vdupq_n_s32(0i32)));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            let value = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(value);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
