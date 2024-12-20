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
use crate::neon::utils::prefer_vfmaq_laneq_f32;
use crate::neon::utils::{prefer_vfmaq_f32, prefer_vfmaq_lane_f32};

macro_rules! write_rgb_f32 {
    ($store: expr, $dest_ptr: expr) => {{
        let l1 = vgetq_lane_u64::<0>(vreinterpretq_u64_f32($store));
        let l3 = vgetq_lane_f32::<2>($store);
        ($dest_ptr as *mut u64).write_unaligned(l1);
        $dest_ptr.add(2).write_unaligned(l3);
    }};
}

macro_rules! conv_horiz_4_rgb_f32 {
    ($start_x: expr, $src: expr, $weights: expr, $store: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let rgb_pixel_0 = vld1q_f32(src_ptr);
        let rgb_pixel_1 = vld1q_f32(src_ptr.add(3));
        let rgb_pixel_2 = vld1q_f32(src_ptr.add(6));
        let rgb_pixel_3 = vld1q_f32(
            [
                src_ptr.add(9).read_unaligned(),
                src_ptr.add(10).read_unaligned(),
                src_ptr.add(11).read_unaligned(),
                0f32,
            ]
            .as_ptr(),
        );

        let acc = prefer_vfmaq_laneq_f32::<0>($store, rgb_pixel_0, $weights);
        let acc = prefer_vfmaq_laneq_f32::<1>(acc, rgb_pixel_1, $weights);
        let acc = prefer_vfmaq_laneq_f32::<2>(acc, rgb_pixel_2, $weights);
        let acc = prefer_vfmaq_laneq_f32::<3>(acc, rgb_pixel_3, $weights);
        acc
    }};
}

macro_rules! conv_horiz_2_rgb_f32 {
    ($start_x: expr, $src: expr, $set: expr, $store: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let mut rgb_pixel_0 = vld1q_f32(src_ptr);
        rgb_pixel_0 = vsetq_lane_f32::<3>(0., rgb_pixel_0);
        let rgb_pixel_1 = vld1q_f32(
            [
                src_ptr.add(3).read_unaligned(),
                src_ptr.add(4).read_unaligned(),
                src_ptr.add(5).read_unaligned(),
                0f32,
            ]
            .as_ptr(),
        );

        let acc = prefer_vfmaq_lane_f32::<0>($store, rgb_pixel_0, $set);
        let acc = prefer_vfmaq_lane_f32::<1>(acc, rgb_pixel_1, $set);
        acc
    }};
}

macro_rules! conv_horiz_1_rgb_f32 {
    ($start_x: expr, $src: expr, $weight: expr, $store: expr) => {{
        const COMPONENTS: usize = 3;
        let src_ptr = $src.add($start_x * COMPONENTS);

        let transient: [f32; 4] = [
            src_ptr.read_unaligned(),
            src_ptr.add(1).read_unaligned(),
            src_ptr.add(2).read_unaligned(),
            0f32,
        ];
        let rgb_pixel = vld1q_f32(transient.as_ptr());

        let acc = prefer_vfmaq_f32($store, rgb_pixel, $weight);
        acc
    }};
}

pub(crate) fn convolve_horizontal_rgb_neon_rows_4_f32(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;

        let zeros = vdupq_n_f32(0.);

        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 4 < bounds.size && bounds.start + jx + 5 < src_width {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vld1q_f32(ptr);
                store_0 = conv_horiz_4_rgb_f32!(bounds_start, src.as_ptr(), read_weights, store_0);
                let s_ptr1 = src.get_unchecked(src_stride..).as_ptr();
                store_1 = conv_horiz_4_rgb_f32!(bounds_start, s_ptr1, read_weights, store_1);
                let s_ptr2 = src.get_unchecked(src_stride * 2..).as_ptr();
                store_2 = conv_horiz_4_rgb_f32!(bounds_start, s_ptr2, read_weights, store_2);
                let s_ptr = src.get_unchecked(src_stride * 3..).as_ptr();
                store_3 = conv_horiz_4_rgb_f32!(bounds_start, s_ptr, read_weights, store_3);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vld1_f32(ptr);
                store_0 = conv_horiz_2_rgb_f32!(bounds_start, src.as_ptr(), read_weights, store_0);
                let s_ptr_1 = src.get_unchecked(src_stride..).as_ptr();
                store_1 = conv_horiz_2_rgb_f32!(bounds_start, s_ptr_1, read_weights, store_1);
                let s_ptr2 = src.get_unchecked(src_stride * 2..).as_ptr();
                store_2 = conv_horiz_2_rgb_f32!(bounds_start, s_ptr2, read_weights, store_2);
                let s_ptr3 = src.get_unchecked(src_stride * 3..).as_ptr();
                store_3 = conv_horiz_2_rgb_f32!(bounds_start, s_ptr3, read_weights, store_3);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vld1q_dup_f32(ptr);
                store_0 = conv_horiz_1_rgb_f32!(bounds_start, src.as_ptr(), weight0, store_0);
                let s_ptr_1 = src.get_unchecked(src_stride..).as_ptr();
                store_1 = conv_horiz_1_rgb_f32!(bounds_start, s_ptr_1, weight0, store_1);
                let s_ptr_2 = src.get_unchecked(src_stride * 2..).as_ptr();
                store_2 = conv_horiz_1_rgb_f32!(bounds_start, s_ptr_2, weight0, store_2);
                let s_ptr_3 = src.get_unchecked(src_stride * 3..).as_ptr();
                store_3 = conv_horiz_1_rgb_f32!(bounds_start, s_ptr_3, weight0, store_3);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            write_rgb_f32!(store_0, dest_ptr);

            let dest_ptr_1 = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
            write_rgb_f32!(store_1, dest_ptr_1);

            let dest_ptr_2 = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
            write_rgb_f32!(store_2, dest_ptr_2);

            let dest_ptr_3 = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
            write_rgb_f32!(store_3, dest_ptr_3);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_row_one_f32(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        const CHANNELS: usize = 3;
        let weights_ptr = filter_weights.weights.as_ptr();
        let mut filter_offset = 0usize;

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vdupq_n_f32(0f32);

            while jx + 4 < bounds.size && bounds.start + jx + 5 < src_width {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vld1q_f32(ptr);
                store = conv_horiz_4_rgb_f32!(bounds_start, src.as_ptr(), read_weights, store);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vld1_f32(ptr);
                store = conv_horiz_2_rgb_f32!(bounds_start, src.as_ptr(), read_weights, store);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vld1q_dup_f32(ptr);
                let bounds_start = bounds.start + jx;
                store = conv_horiz_1_rgb_f32!(bounds_start, src.as_ptr(), weight0, store);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            write_rgb_f32!(store, dest_ptr);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
