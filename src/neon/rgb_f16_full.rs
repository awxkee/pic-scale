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

use half::f16;

use crate::filter_weights::FilterWeights;
use crate::neon::*;

macro_rules! write_rgb_f16 {
    ($store: expr, $dest_ptr: expr) => {{
        let cvt = xreinterpret_u16_f16($store);
        let l1 = vget_lane_u32::<0>(vreinterpret_u32_u16(cvt));
        let l3 = vget_lane_u16::<2>(cvt);
        ($dest_ptr as *mut u32).write_unaligned(l1);
        ($dest_ptr as *mut u16).add(2).write_unaligned(l3);
    }};
}

macro_rules! conv_horiz_5_rgb_f16 {
    ($start_x: expr, $src: expr, $set: expr, $store: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel_s = xvldq_f16_x2(src_ptr);
        let rgb_first_u = vget_low_u16(xreinterpretq_u16_f16(rgb_pixel_s.0));
        let rgb_first = xreinterpret_f16_u16(rgb_first_u);
        let rgb_second_u = vext_u16::<3>(
            vget_low_u16(xreinterpretq_u16_f16(rgb_pixel_s.0)),
            vget_high_u16(xreinterpretq_u16_f16(rgb_pixel_s.0)),
        );
        let rgb_second = xreinterpret_f16_u16(rgb_second_u);

        let rgb_third_u = vext_u16::<2>(
            vget_high_u16(xreinterpretq_u16_f16(rgb_pixel_s.0)),
            vget_low_u16(xreinterpretq_u16_f16(rgb_pixel_s.1)),
        );
        let rgb_third = xreinterpret_f16_u16(rgb_third_u);

        let rgb_fourth_u = vext_u16::<1>(
            vget_low_u16(xreinterpretq_u16_f16(rgb_pixel_s.1)),
            vget_high_u16(xreinterpretq_u16_f16(rgb_pixel_s.1)),
        );
        let rgb_fourth = xreinterpret_f16_u16(rgb_fourth_u);

        let rgb_fifth = xvget_high_f16(rgb_pixel_s.1);

        let mut acc = xvfmla_f16($store, rgb_first, $set.0);
        acc = xvfmla_f16(acc, rgb_second, $set.1);
        acc = xvfmla_f16(acc, rgb_third, $set.2);
        acc = xvfmla_f16(acc, rgb_fourth, $set.3);
        acc = xvfmla_f16(acc, rgb_fifth, $set.4);
        acc
    }};
}

macro_rules! conv_horiz_4_rgb_f16 {
    ($start_x: expr, $src: expr, $set: expr, $store: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel_s = xvldq_f16_x2(src_ptr);
        let rgb_first_u = vget_low_u16(xreinterpretq_u16_f16(rgb_pixel_s.0));
        let rgb_first = xreinterpret_f16_u16(rgb_first_u);
        let rgb_second_u = vext_u16::<3>(
            vget_low_u16(xreinterpretq_u16_f16(rgb_pixel_s.0)),
            vget_high_u16(xreinterpretq_u16_f16(rgb_pixel_s.0)),
        );
        let rgb_second = xreinterpret_f16_u16(rgb_second_u);

        let rgb_third_u = vext_u16::<2>(
            vget_high_u16(xreinterpretq_u16_f16(rgb_pixel_s.0)),
            vget_low_u16(xreinterpretq_u16_f16(rgb_pixel_s.1)),
        );
        let rgb_third = xreinterpret_f16_u16(rgb_third_u);

        let rgb_fourth_u = vext_u16::<1>(
            vget_low_u16(xreinterpretq_u16_f16(rgb_pixel_s.1)),
            vget_high_u16(xreinterpretq_u16_f16(rgb_pixel_s.1)),
        );
        let rgb_fourth = xreinterpret_f16_u16(rgb_fourth_u);

        let acc = xvfmla_f16($store, rgb_first, $set.0);
        let acc = xvfmla_f16(acc, rgb_second, $set.1);
        let acc = xvfmla_f16(acc, rgb_third, $set.2);
        let acc = xvfmla_f16(acc, rgb_fourth, $set.3);
        acc
    }};
}

macro_rules! conv_horiz_2_rgb_f16 {
    ($start_x: expr, $src: expr, $set: expr, $store: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        const ZEROS_F16: f16 = half::f16::from_bits(0);

        let rgb_pixel = xvld_f16(src_ptr);
        let second_part: [half::f16; 4] = [
            src_ptr.add(4).read_unaligned(),
            src_ptr.add(5).read_unaligned(),
            ZEROS_F16,
            ZEROS_F16,
        ];
        let second_px = xvld_f16(second_part.as_ptr());

        let mut rgb_first_u = xreinterpret_u16_f16(rgb_pixel);
        rgb_first_u = vset_lane_u16::<3>(0, rgb_first_u);
        let rgb_first = xreinterpret_f16_u16(rgb_first_u);
        let mut rgb_second_u = vext_u16::<3>(
            xreinterpret_u16_f16(rgb_pixel),
            xreinterpret_u16_f16(second_px),
        );
        rgb_second_u = vset_lane_u16::<3>(0, rgb_second_u);
        let rgb_second = xreinterpret_f16_u16(rgb_second_u);

        let acc = xvfmla_f16($store, rgb_first, $set.0);
        let acc = xvfmla_f16(acc, rgb_second, $set.1);
        acc
    }};
}

macro_rules! conv_horiz_1_rgb_f16 {
    ($start_x: expr, $src: expr, $weight: expr, $store: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        const ZEROS_F16: f16 = half::f16::from_bits(0);

        let transient: [half::f16; 4] = [
            src_ptr.read_unaligned(),
            src_ptr.add(1).read_unaligned(),
            src_ptr.add(2).read_unaligned(),
            ZEROS_F16,
        ];
        let rgb_pixel = xvld_f16(transient.as_ptr());

        let acc = xvfmla_f16($store, rgb_pixel, $weight);
        acc
    }};
}

pub fn xconvolve_horizontal_rgb_neon_rows_4_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f16,
    dst_stride: usize,
) {
    unsafe {
        xconvolve_horizontal_rgb_neon_rows_4_f16_impl(
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
unsafe fn xconvolve_horizontal_rgb_neon_rows_4_f16_impl(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f16,
    dst_stride: usize,
) {
    unsafe {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;

        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = xvzeros_f16();
            let mut store_1 = xvzeros_f16();
            let mut store_2 = xvzeros_f16();
            let mut store_3 = xvzeros_f16();

            while jx + 5 < bounds.size && bounds.start + jx + 6 < src_width {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = xvcvt_f16_f32(vld1q_f32(ptr));
                let w0 = xvdup_lane_f16::<0>(read_weights);
                let w1 = xvdup_lane_f16::<1>(read_weights);
                let w2 = xvdup_lane_f16::<2>(read_weights);
                let w3 = xvdup_lane_f16::<3>(read_weights);
                let w4 = xvcvt_f16_f32(vld1q_dup_f32(ptr.add(4)));
                let set = (w0, w1, w2, w3, w4);
                let b_start = bounds_start;
                store_0 = conv_horiz_5_rgb_f16!(b_start, unsafe_source_ptr_0, set, store_0);
                let s_ptr1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_5_rgb_f16!(b_start, s_ptr1, set, store_1);
                let s_ptr2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_5_rgb_f16!(b_start, s_ptr2, set, store_2);
                let s_ptr3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_5_rgb_f16!(b_start, s_ptr3, set, store_3);
                jx += 5;
            }

            while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = xvcvt_f16_f32(vld1q_f32(ptr));
                let w0 = xvdup_lane_f16::<0>(read_weights);
                let w1 = xvdup_lane_f16::<1>(read_weights);
                let w2 = xvdup_lane_f16::<2>(read_weights);
                let w3 = xvdup_lane_f16::<3>(read_weights);
                let set = (w0, w1, w2, w3);
                store_0 = conv_horiz_4_rgb_f16!(bounds_start, unsafe_source_ptr_0, set, store_0);
                let s_ptr1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_4_rgb_f16!(bounds_start, s_ptr1, set, store_1);
                let s_ptr2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_4_rgb_f16!(bounds_start, s_ptr2, set, store_2);
                let s_ptr = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_4_rgb_f16!(bounds_start, s_ptr, set, store_3);
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
                store_0 = conv_horiz_2_rgb_f16!(bounds_start, unsafe_source_ptr_0, set, store_0);
                let s_ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_2_rgb_f16!(bounds_start, s_ptr_1, set, store_1);
                let s_ptr2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_2_rgb_f16!(bounds_start, s_ptr2, set, store_2);
                let s_ptr3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_2_rgb_f16!(bounds_start, s_ptr3, set, store_3);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = xvcvt_f16_f32(vld1q_dup_f32(ptr));
                store_0 =
                    conv_horiz_1_rgb_f16!(bounds_start, unsafe_source_ptr_0, weight0, store_0);
                let s_ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_1_rgb_f16!(bounds_start, s_ptr_1, weight0, store_1);
                let s_ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_1_rgb_f16!(bounds_start, s_ptr_2, weight0, store_2);
                let s_ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_1_rgb_f16!(bounds_start, s_ptr_3, weight0, store_3);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            write_rgb_f16!(store_0, dest_ptr);

            let dest_ptr_1 = unsafe_destination_ptr_0.add(px + dst_stride);
            write_rgb_f16!(store_1, dest_ptr_1);

            let dest_ptr_2 = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            write_rgb_f16!(store_2, dest_ptr_2);

            let dest_ptr_3 = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            write_rgb_f16!(store_3, dest_ptr_3);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub fn xconvolve_horizontal_rgb_neon_row_one_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const half::f16,
    unsafe_destination_ptr_0: *mut half::f16,
) {
    unsafe {
        xconvolve_horizontal_rgb_neon_row_one_f16_impl(
            dst_width,
            src_width,
            filter_weights,
            unsafe_source_ptr_0,
            unsafe_destination_ptr_0,
        );
    }
}

#[target_feature(enable = "fp16")]
unsafe fn xconvolve_horizontal_rgb_neon_row_one_f16_impl(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const half::f16,
    unsafe_destination_ptr_0: *mut half::f16,
) {
    const CHANNELS: usize = 3;
    let weights_ptr = filter_weights.weights.as_ptr();
    let mut filter_offset = 0usize;

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store = xvzeros_f16();

        while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights = xvcvt_f16_f32(vld1q_f32(ptr));
            let w0 = xvdup_lane_f16::<0>(read_weights);
            let w1 = xvdup_lane_f16::<1>(read_weights);
            let w2 = xvdup_lane_f16::<2>(read_weights);
            let w3 = xvdup_lane_f16::<3>(read_weights);
            let set = (w0, w1, w2, w3);

            store = conv_horiz_4_rgb_f16!(bounds_start, unsafe_source_ptr_0, set, store);
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
            store = conv_horiz_2_rgb_f16!(bounds_start, unsafe_source_ptr_0, set, store);
            jx += 2;
        }

        while jx < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight0 = xvcvt_f16_f32(vld1q_dup_f32(ptr));
            let bounds_start = bounds.start + jx;
            store = conv_horiz_1_rgb_f16!(bounds_start, unsafe_source_ptr_0, weight0, store);
            jx += 1;
        }

        let px = x * CHANNELS;
        let dest_ptr = unsafe_destination_ptr_0.add(px);
        write_rgb_f16!(store, dest_ptr);

        filter_offset += filter_weights.aligned_size;
    }
}
