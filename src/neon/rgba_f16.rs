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
use crate::neon::f16_utils::xvcvt_f16_f32;
use crate::neon::utils::prefer_vfmaq_f32;
use crate::neon::{
    xvcvt_f32_f16, xvget_high_f16, xvget_low_f16, xvld_f16, xvldq_f16, xvldq_f16_x2, xvldq_f16_x4,
    xvst_f16,
};
use std::arch::aarch64::*;

macro_rules! conv_horiz_rgba_8_f16 {
    ($start_x: expr, $src: expr, $set1: expr, $set2: expr, $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel = xvldq_f16_x4(src_ptr);

        let mut acc = prefer_vfmaq_f32($store, xvcvt_f32_f16(xvget_low_f16(rgb_pixel.0)), $set1.0);
        acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_high_f16(rgb_pixel.0)), $set1.1);
        acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_low_f16(rgb_pixel.1)), $set1.2);
        acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_high_f16(rgb_pixel.1)), $set1.3);
        acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_low_f16(rgb_pixel.2)), $set2.0);
        acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_high_f16(rgb_pixel.2)), $set2.1);
        acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_low_f16(rgb_pixel.3)), $set2.2);
        acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_high_f16(rgb_pixel.3)), $set2.3);
        acc
    }};
}

macro_rules! conv_horiz_rgba_4_f16 {
    ($start_x: expr, $src: expr, $set1: expr,  $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel = xvldq_f16_x2(src_ptr);

        let acc = prefer_vfmaq_f32($store, xvcvt_f32_f16(xvget_low_f16(rgb_pixel.0)), $set1.0);
        let acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_high_f16(rgb_pixel.0)), $set1.1);
        let acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_low_f16(rgb_pixel.1)), $set1.2);
        let acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_high_f16(rgb_pixel.0)), $set1.3);
        acc
    }};
}

macro_rules! conv_horiz_rgba_2_f32 {
    ($start_x: expr, $src: expr, $set: expr,  $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel = xvldq_f16(src_ptr);

        let mut acc = prefer_vfmaq_f32($store, xvcvt_f32_f16(xvget_low_f16(rgb_pixel)), $set.0);
        acc = prefer_vfmaq_f32(acc, xvcvt_f32_f16(xvget_high_f16(rgb_pixel)), $set.1);
        acc
    }};
}

macro_rules! conv_horiz_rgba_1_f16 {
    ($start_x: expr, $src: expr, $set: expr,  $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);
        let rgb_pixel = xvld_f16(src_ptr);
        let acc = prefer_vfmaq_f32($store, xvcvt_f32_f16(rgb_pixel), $set);
        acc
    }};
}

pub fn convolve_horizontal_rgba_neon_row_one_f16(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const half::f16,
    unsafe_destination_ptr_0: *mut half::f16,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vdupq_n_f32(0f32);

            while jx + 4 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vld1q_f32(ptr);
                let w0 = vdupq_n_f32(vgetq_lane_f32::<0>(read_weights));
                let w1 = vdupq_n_f32(vgetq_lane_f32::<1>(read_weights));
                let w2 = vdupq_n_f32(vgetq_lane_f32::<2>(read_weights));
                let w3 = vdupq_n_f32(vgetq_lane_f32::<3>(read_weights));
                let set1 = (w0, w1, w2, w3);
                store = conv_horiz_rgba_4_f16!(bounds_start, unsafe_source_ptr_0, set1, store);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vdupq_n_f32(ptr.read_unaligned());
                let weight1 = vdupq_n_f32(ptr.add(1).read_unaligned());
                let set = (weight0, weight1);
                store = conv_horiz_rgba_2_f32!(bounds_start, unsafe_source_ptr_0, set, store);
                jx += 2;
            }

            while jx < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vdupq_n_f32(ptr.read_unaligned());
                store = conv_horiz_rgba_1_f16!(bounds_start, unsafe_source_ptr_0, weight0, store);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            xvst_f16(dest_ptr, xvcvt_f16_f32(store));

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgba_neon_rows_4_f16(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const half::f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut half::f16,
    dst_stride: usize,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let zeros = vdupq_n_f32(0f32);
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 8 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vld1q_f32_x2(ptr);
                let w0 = vdupq_laneq_f32::<0>(read_weights.0);
                let w1 = vdupq_laneq_f32::<1>(read_weights.0);
                let w2 = vdupq_laneq_f32::<2>(read_weights.0);
                let w3 = vdupq_laneq_f32::<3>(read_weights.0);
                let w4 = vdupq_laneq_f32::<0>(read_weights.1);
                let w5 = vdupq_laneq_f32::<1>(read_weights.1);
                let w6 = vdupq_laneq_f32::<2>(read_weights.1);
                let w7 = vdupq_laneq_f32::<3>(read_weights.1);
                let set1 = (w0, w1, w2, w3);
                let set2 = (w4, w5, w6, w7);
                let bounds_start = bounds.start + jx;
                store_0 =
                    conv_horiz_rgba_8_f16!(bounds_start, unsafe_source_ptr_0, set1, set2, store_0);
                let s_ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_8_f16!(bounds_start, s_ptr_1, set1, set2, store_1);
                let s_ptr2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_8_f16!(bounds_start, s_ptr2, set1, set2, store_2);
                let s_ptr3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_8_f16!(bounds_start, s_ptr3, set1, set2, store_3);
                jx += 8;
            }

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vld1q_f32(ptr);
                let w0 = vdupq_laneq_f32::<0>(read_weights);
                let w1 = vdupq_laneq_f32::<1>(read_weights);
                let w2 = vdupq_laneq_f32::<2>(read_weights);
                let w3 = vdupq_laneq_f32::<3>(read_weights);
                let set1 = (w0, w1, w2, w3);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_rgba_4_f16!(bounds_start, unsafe_source_ptr_0, set1, store_0);
                let s_ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_4_f16!(bounds_start, s_ptr_1, set1, store_1);
                let s_ptr2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_4_f16!(bounds_start, s_ptr2, set1, store_2);
                let s_ptr3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_4_f16!(bounds_start, s_ptr3, set1, store_3);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vld1_f32(ptr);
                let w0 = vdupq_lane_f32::<0>(read_weights);
                let w1 = vdupq_lane_f32::<1>(read_weights);
                let set = (w0, w1);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_rgba_2_f32!(bounds_start, unsafe_source_ptr_0, set, store_0);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_2_f32!(bounds_start, ptr_1, set, store_1);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_2_f32!(bounds_start, ptr_2, set, store_2);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_2_f32!(bounds_start, ptr_3, set, store_3);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vld1q_dup_f32(ptr);
                let bounds_start = bounds.start + jx;
                store_0 =
                    conv_horiz_rgba_1_f16!(bounds_start, unsafe_source_ptr_0, weight0, store_0);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_rgba_1_f16!(bounds_start, ptr_1, weight0, store_1);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_rgba_1_f16!(bounds_start, ptr_2, weight0, store_2);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_rgba_1_f16!(bounds_start, ptr_3, weight0, store_3);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            xvst_f16(dest_ptr, xvcvt_f16_f32(store_0));

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            xvst_f16(dest_ptr, xvcvt_f16_f32(store_1));

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            xvst_f16(dest_ptr, xvcvt_f16_f32(store_2));

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            xvst_f16(dest_ptr, xvcvt_f16_f32(store_3));

            filter_offset += filter_weights.aligned_size;
        }
    }
}
