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

use std::arch::aarch64::*;

use crate::filter_weights::FilterWeights;
use crate::neon::f16_utils::{xvcombine_f16, xvcvt_f16_f32, xvfmla_f16, xvzeros_f16};
use crate::neon::{
    xvdup_lane_f16, xvdup_laneq_f16, xvget_high_f16, xvget_low_f16, xvld_f16, xvldq_f16,
    xvldq_f16_x2, xvldq_f16_x4, xvst_f16,
};

macro_rules! conv_horiz_rgba_8_f16 {
    ($start_x: expr, $src: expr, $set1: expr, $set2: expr, $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel = xvldq_f16_x4(src_ptr);

        let mut acc = xvfmla_f16($store, xvget_low_f16(rgb_pixel.0), $set1.0);
        acc = xvfmla_f16(acc, xvget_high_f16(rgb_pixel.0), $set1.1);
        acc = xvfmla_f16(acc, xvget_low_f16(rgb_pixel.1), $set1.2);
        acc = xvfmla_f16(acc, xvget_high_f16(rgb_pixel.1), $set1.3);
        acc = xvfmla_f16(acc, xvget_low_f16(rgb_pixel.2), $set2.0);
        acc = xvfmla_f16(acc, xvget_high_f16(rgb_pixel.2), $set2.1);
        acc = xvfmla_f16(acc, xvget_low_f16(rgb_pixel.3), $set2.2);
        acc = xvfmla_f16(acc, xvget_high_f16(rgb_pixel.3), $set2.3);
        acc
    }};
}

macro_rules! conv_horiz_rgba_4_f16 {
    ($start_x: expr, $src: expr, $set1: expr,  $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel = xvldq_f16_x2(src_ptr);

        let acc = xvfmla_f16($store, xvget_low_f16(rgb_pixel.0), $set1.0);
        let acc = xvfmla_f16(acc, xvget_high_f16(rgb_pixel.0), $set1.1);
        let acc = xvfmla_f16(acc, xvget_low_f16(rgb_pixel.1), $set1.2);
        let acc = xvfmla_f16(acc, xvget_high_f16(rgb_pixel.0), $set1.3);
        acc
    }};
}

macro_rules! conv_horiz_rgba_2_f32 {
    ($start_x: expr, $src: expr, $set: expr,  $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel = xvldq_f16(src_ptr);

        let mut acc = xvfmla_f16($store, xvget_low_f16(rgb_pixel), $set.0);
        acc = xvfmla_f16(acc, xvget_high_f16(rgb_pixel), $set.1);
        acc
    }};
}

macro_rules! conv_horiz_rgba_1_f16 {
    ($start_x: expr, $src: expr, $set: expr,  $store: expr) => {{
        const COMPONENTS: usize = 4;
        let src_ptr = $src.add($start_x * COMPONENTS);
        let rgb_pixel = xvld_f16(src_ptr);
        let acc = xvfmla_f16($store, rgb_pixel, $set);
        acc
    }};
}

pub fn xconvolve_horizontal_rgba_neon_row_one_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const half::f16,
    unsafe_destination_ptr_0: *mut half::f16,
) {
    unsafe {
        xconvolve_horizontal_rgba_neon_row_one_f16_impl(
            dst_width,
            src_width,
            filter_weights,
            unsafe_source_ptr_0,
            unsafe_destination_ptr_0,
        );
    }
}

#[target_feature(enable = "fp16")]
unsafe fn xconvolve_horizontal_rgba_neon_row_one_f16_impl(
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
            let mut store = xvzeros_f16();

            while jx + 4 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = xvcvt_f16_f32(vld1q_f32(ptr));
                let w0 = xvdup_lane_f16::<0>(read_weights);
                let w1 = xvdup_lane_f16::<1>(read_weights);
                let w2 = xvdup_lane_f16::<2>(read_weights);
                let w3 = xvdup_lane_f16::<3>(read_weights);
                let set1 = (w0, w1, w2, w3);
                store = conv_horiz_rgba_4_f16!(bounds_start, unsafe_source_ptr_0, set1, store);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights_h = vld1_f32(ptr);
                let read_weights = xvcvt_f16_f32(vcombine_f32(read_weights_h, read_weights_h));
                let w0 = xvdup_lane_f16::<0>(read_weights);
                let w1 = xvdup_lane_f16::<1>(read_weights);
                let set = (w0, w1);
                store = conv_horiz_rgba_2_f32!(bounds_start, unsafe_source_ptr_0, set, store);
                jx += 2;
            }

            while jx < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = xvcvt_f16_f32(vld1q_dup_f32(ptr));
                store = conv_horiz_rgba_1_f16!(bounds_start, unsafe_source_ptr_0, weight0, store);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            xvst_f16(dest_ptr, store);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub fn xconvolve_horizontal_rgba_neon_rows_4_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const half::f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut half::f16,
    dst_stride: usize,
) {
    unsafe {
        xconvolve_horizontal_rgba_neon_rows_4_f16_impl(
            dst_width,
            src_width,
            filter_weights,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            dst_stride,
        );
    }
}

#[target_feature(enable = "fp16")]
unsafe fn xconvolve_horizontal_rgba_neon_rows_4_f16_impl(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const half::f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut half::f16,
    dst_stride: usize,
) {
    const CHANNELS: usize = 4;
    let mut filter_offset = 0usize;

    let weights_ptr = filter_weights.weights.as_ptr();

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store_0 = xvzeros_f16();
        let mut store_1 = xvzeros_f16();
        let mut store_2 = xvzeros_f16();
        let mut store_3 = xvzeros_f16();

        while jx + 8 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights_h = vld1q_f32_x2(ptr);
            let read_weights = xvcombine_f16(
                xvcvt_f16_f32(read_weights_h.0),
                xvcvt_f16_f32(read_weights_h.1),
            );
            let w0 = xvdup_laneq_f16::<0>(read_weights);
            let w1 = xvdup_laneq_f16::<1>(read_weights);
            let w2 = xvdup_laneq_f16::<2>(read_weights);
            let w3 = xvdup_laneq_f16::<3>(read_weights);
            let w4 = xvdup_laneq_f16::<4>(read_weights);
            let w5 = xvdup_laneq_f16::<5>(read_weights);
            let w6 = xvdup_laneq_f16::<6>(read_weights);
            let w7 = xvdup_laneq_f16::<7>(read_weights);
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
            let read_weights = xvcvt_f16_f32(vld1q_f32(ptr));
            let w0 = xvdup_lane_f16::<0>(read_weights);
            let w1 = xvdup_lane_f16::<1>(read_weights);
            let w2 = xvdup_lane_f16::<2>(read_weights);
            let w3 = xvdup_lane_f16::<3>(read_weights);
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
            let read_weights_h = vld1_f32(ptr);
            let read_weights = xvcvt_f16_f32(vcombine_f32(read_weights_h, read_weights_h));
            let w0 = xvdup_lane_f16::<0>(read_weights);
            let w1 = xvdup_lane_f16::<1>(read_weights);
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
            let weight0 = xvcvt_f16_f32(vld1q_dup_f32(ptr));
            let bounds_start = bounds.start + jx;
            store_0 = conv_horiz_rgba_1_f16!(bounds_start, unsafe_source_ptr_0, weight0, store_0);
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
        xvst_f16(dest_ptr, store_0);

        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
        xvst_f16(dest_ptr, store_1);

        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
        xvst_f16(dest_ptr, store_2);

        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
        xvst_f16(dest_ptr, store_3);

        filter_offset += filter_weights.aligned_size;
    }
}
