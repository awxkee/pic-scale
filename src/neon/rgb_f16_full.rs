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

use core::f16;

use crate::filter_weights::FilterWeights;

macro_rules! write_rgb_f16 {
    ($store: expr, $dest_ptr: expr) => {{
        let cvt = vreinterpret_u16_f16($store);
        let l1 = vget_lane_u32::<0>(vreinterpret_u32_u16(cvt));
        let l3 = vget_lane_u16::<2>(cvt);
        ($dest_ptr as *mut u32).write_unaligned(l1);
        ($dest_ptr as *mut u16).add(2).write_unaligned(l3);
    }};
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_4_rgb_f16(
    start_x: usize,
    src: &[f16],
    set: float16x4_t,
    store: float16x4_t,
) -> float16x4_t {
    unsafe {
        const COMPONENTS: usize = 3;
        let src_ptr = src.get_unchecked(start_x * COMPONENTS..).as_ptr();

        let v0 = vld1q_u16(src_ptr as *const _);
        let v1 = vcombine_u16(vld1_u16(src_ptr as *const _), vdup_n_u16(0));

        let rgb_pixel_s = uint16x8x2_t(v0, v1);
        let rgb_first_u = vget_low_u16(rgb_pixel_s.0);
        let rgb_first = rgb_first_u;
        let rgb_second_u = vext_u16::<3>(vget_low_u16(rgb_pixel_s.0), vget_high_u16(rgb_pixel_s.0));
        let rgb_second = rgb_second_u;

        let rgb_third_u = vext_u16::<2>(vget_high_u16(rgb_pixel_s.0), vget_low_u16(rgb_pixel_s.1));
        let rgb_third = rgb_third_u;

        let rgb_fourth_u = vext_u16::<1>(vget_low_u16(rgb_pixel_s.1), vget_high_u16(rgb_pixel_s.1));
        let rgb_fourth = vreinterpret_f16_u16(rgb_fourth_u);

        let acc = vfma_lane_f16::<0>(store, vreinterpret_f16_u16(rgb_first), set);
        let acc = vfma_lane_f16::<1>(acc, vreinterpret_f16_u16(rgb_second), set);
        let acc = vfma_lane_f16::<2>(acc, vreinterpret_f16_u16(rgb_third), set);
        vfma_lane_f16::<3>(acc, rgb_fourth, set)
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_2_rgb_f16(
    start_x: usize,
    src: &[f16],
    set: float16x4_t,
    store: float16x4_t,
) -> float16x4_t {
    unsafe {
        const COMPONENTS: usize = 3;
        let src_ptr = src.get_unchecked(start_x * COMPONENTS..).as_ptr();

        let rgb_pixel = vld1_u16(src_ptr as *const _);
        let second_px = vreinterpret_u16_u32(vld1_lane_u32::<0>(
            src_ptr.add(4) as *const _,
            vdup_n_u32(0),
        ));

        let mut rgb_first_u = rgb_pixel;
        rgb_first_u = vset_lane_u16::<3>(0, rgb_first_u);
        let rgb_first = rgb_first_u;
        let mut rgb_second_u = vext_u16::<3>(rgb_pixel, second_px);
        rgb_second_u = vset_lane_u16::<3>(0, rgb_second_u);
        let rgb_second = vreinterpret_f16_u16(rgb_second_u);

        let acc = vfma_lane_f16::<0>(store, vreinterpret_f16_u16(rgb_first), set);
        vfma_lane_f16::<1>(acc, rgb_second, set)
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_1_rgb_f16(
    start_x: usize,
    src: &[f16],
    set: float16x4_t,
    store: float16x4_t,
) -> float16x4_t {
    unsafe {
        const COMPONENTS: usize = 3;
        let src_ptr = src.get_unchecked(start_x * COMPONENTS..).as_ptr();

        let mut rgb_pixel_u =
            vreinterpret_u16_u32(vld1_lane_u32::<0>(src_ptr as *const _, vdup_n_u32(0)));
        rgb_pixel_u = vld1_lane_u16::<2>(src_ptr as *const _, rgb_pixel_u);

        let rgb_pixel = vreinterpret_f16_u16(rgb_pixel_u);
        vfma_f16(store, rgb_pixel, set)
    }
}

pub(crate) fn xconvolve_horizontal_rgb_neon_rows_4_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    unsafe {
        xconvolve_horizontal_rgb_neon_rows_4_f16_impl(
            dst_width,
            src_width,
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "fp16")]
unsafe fn xconvolve_horizontal_rgb_neon_rows_4_f16_impl(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    unsafe {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;

        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = vdup_n_f16(0.);
            let mut store_1 = vdup_n_f16(0.);
            let mut store_2 = vdup_n_f16(0.);
            let mut store_3 = vdup_n_f16(0.);

            while jx + 4 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vcvt_f16_f32(vld1q_f32(ptr));
                store_0 = conv_horiz_4_rgb_f16(bounds_start, src, read_weights, store_0);
                let s_ptr1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_4_rgb_f16(bounds_start, s_ptr1, read_weights, store_1);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_4_rgb_f16(bounds_start, s_ptr2, read_weights, store_2);
                let s_ptr = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_4_rgb_f16(bounds_start, s_ptr, read_weights, store_3);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights_h = vld1_f32(ptr);
                let read_weights = vcvt_f16_f32(vcombine_f32(read_weights_h, read_weights_h));
                store_0 = conv_horiz_2_rgb_f16(bounds_start, src, read_weights, store_0);
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_2_rgb_f16(bounds_start, s_ptr_1, read_weights, store_1);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_2_rgb_f16(bounds_start, s_ptr2, read_weights, store_2);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_2_rgb_f16(bounds_start, s_ptr3, read_weights, store_3);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vcvt_f16_f32(vld1q_dup_f32(ptr));
                store_0 = conv_horiz_1_rgb_f16(bounds_start, src, weight0, store_0);
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_1_rgb_f16(bounds_start, s_ptr_1, weight0, store_1);
                let s_ptr_2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_1_rgb_f16(bounds_start, s_ptr_2, weight0, store_2);
                let s_ptr_3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_1_rgb_f16(bounds_start, s_ptr_3, weight0, store_3);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            write_rgb_f16!(store_0, dest_ptr);

            let dest_ptr_1 = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
            write_rgb_f16!(store_1, dest_ptr_1);

            let dest_ptr_2 = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
            write_rgb_f16!(store_2, dest_ptr_2);

            let dest_ptr_3 = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
            write_rgb_f16!(store_3, dest_ptr_3);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn xconvolve_horizontal_rgb_neon_row_one_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    unsafe {
        xconvolve_horizontal_rgb_neon_row_one_f16_impl(
            dst_width,
            src_width,
            filter_weights,
            src,
            dst,
        );
    }
}

#[target_feature(enable = "fp16")]
unsafe fn xconvolve_horizontal_rgb_neon_row_one_f16_impl(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    unsafe {
        const CHANNELS: usize = 3;
        let weights_ptr = filter_weights.weights.as_ptr();
        let mut filter_offset = 0usize;

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vdup_n_f16(0.);

            while jx + 4 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = vcvt_f16_f32(vld1q_f32(ptr));
                store = conv_horiz_4_rgb_f16(bounds_start, src, read_weights, store);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights_h = vld1_f32(ptr);
                let read_weights = vcvt_f16_f32(vcombine_f32(read_weights_h, read_weights_h));
                store = conv_horiz_2_rgb_f16(bounds_start, src, read_weights, store);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vcvt_f16_f32(vld1q_dup_f32(ptr));
                let bounds_start = bounds.start + jx;
                store = conv_horiz_1_rgb_f16(bounds_start, src, weight0, store);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            write_rgb_f16!(store, dest_ptr);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
