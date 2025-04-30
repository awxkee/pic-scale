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
use crate::neon::utils::xvld1q_f32_x2;
use core::f16;
use std::arch::aarch64::*;

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_8_rgba_f16(
    start_x: usize,
    src: &[f16],
    weights: float16x8_t,
    store: float16x4_t,
) -> float16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..);

    let rgb_pixel = vld1q_u16_x4(src_ptr.as_ptr() as *const _);

    let mut acc = vfma_laneq_f16::<0>(
        store,
        vreinterpret_f16_u16(vget_low_u16(rgb_pixel.0)),
        weights,
    );
    acc = vfma_laneq_f16::<1>(
        acc,
        vreinterpret_f16_u16(vget_high_u16(rgb_pixel.0)),
        weights,
    );
    acc = vfma_laneq_f16::<2>(
        acc,
        vreinterpret_f16_u16(vget_low_u16(rgb_pixel.1)),
        weights,
    );
    acc = vfma_laneq_f16::<3>(
        acc,
        vreinterpret_f16_u16(vget_high_u16(rgb_pixel.1)),
        weights,
    );
    acc = vfma_laneq_f16::<4>(
        acc,
        vreinterpret_f16_u16(vget_low_u16(rgb_pixel.2)),
        weights,
    );
    acc = vfma_laneq_f16::<5>(
        acc,
        vreinterpret_f16_u16(vget_high_u16(rgb_pixel.2)),
        weights,
    );
    acc = vfma_laneq_f16::<6>(
        acc,
        vreinterpret_f16_u16(vget_low_u16(rgb_pixel.3)),
        weights,
    );
    acc = vfma_laneq_f16::<7>(
        acc,
        vreinterpret_f16_u16(vget_high_u16(rgb_pixel.3)),
        weights,
    );
    acc
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_4_rgba_f16(
    start_x: usize,
    src: &[f16],
    weights: float16x4_t,
    store: float16x4_t,
) -> float16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..);

    let rgb_pixel = vld1q_u16_x2(src_ptr.as_ptr() as *const _);

    let acc = vfma_lane_f16::<0>(
        store,
        vreinterpret_f16_u16(vget_low_u16(rgb_pixel.0)),
        weights,
    );
    let acc = vfma_lane_f16::<1>(
        acc,
        vreinterpret_f16_u16(vget_high_u16(rgb_pixel.0)),
        weights,
    );
    let acc = vfma_lane_f16::<2>(
        acc,
        vreinterpret_f16_u16(vget_low_u16(rgb_pixel.1)),
        weights,
    );
    vfma_lane_f16::<3>(
        acc,
        vreinterpret_f16_u16(vget_high_u16(rgb_pixel.0)),
        weights,
    )
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_2_f32(
    start_x: usize,
    src: &[f16],
    weights: float16x4_t,
    store: float16x4_t,
) -> float16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..);

    let rgb_pixel = vld1q_f16(src_ptr.as_ptr());

    let acc = vfma_lane_f16::<0>(store, vget_low_f16(rgb_pixel), weights);
    vfma_lane_f16::<1>(acc, vget_high_f16(rgb_pixel), weights)
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_1_f16(
    start_x: usize,
    src: &[f16],
    weights: float16x4_t,
    store: float16x4_t,
) -> float16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..);
    let rgb_pixel = vld1_f16(src_ptr.as_ptr());
    vfma_f16(store, rgb_pixel, weights)
}

pub(crate) fn xconvolve_horizontal_rgba_neon_row_one_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    unsafe {
        xconvolve_horizontal_rgba_neon_row_one_f16_impl(
            dst_width,
            src_width,
            filter_weights,
            src,
            dst,
        );
    }
}

#[target_feature(enable = "fp16")]
unsafe fn xconvolve_horizontal_rgba_neon_row_one_f16_impl(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    const CHANNELS: usize = 4;
    let mut filter_offset = 0usize;
    let weights_ptr = filter_weights.weights.as_ptr();

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store = vdup_n_f16(0.);

        while jx + 4 < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights = vcvt_f16_f32(vld1q_f32(ptr));
            store = conv_horiz_4_rgba_f16(bounds_start, src, read_weights, store);
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights_h = vld1_f32(ptr);
            let read_weights = vcvt_f16_f32(vcombine_f32(read_weights_h, read_weights_h));
            store = conv_horiz_rgba_2_f32(bounds_start, src, read_weights, store);
            jx += 2;
        }

        while jx < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight0 = vcvt_f16_f32(vld1q_dup_f32(ptr));
            store = conv_horiz_rgba_1_f16(bounds_start, src, weight0, store);
            jx += 1;
        }

        let px = x * CHANNELS;
        let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        vst1_f16(dest_ptr, store);

        filter_offset += filter_weights.aligned_size;
    }
}

pub(crate) fn xconvolve_horizontal_rgba_neon_rows_4_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: &[f16],
    src_stride: usize,
    unsafe_destination_ptr_0: &mut [f16],
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
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    const CHANNELS: usize = 4;
    let mut filter_offset = 0usize;

    let weights_ptr = filter_weights.weights.as_ptr();

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store_0 = vdup_n_f16(0.);
        let mut store_1 = vdup_n_f16(0.);
        let mut store_2 = vdup_n_f16(0.);
        let mut store_3 = vdup_n_f16(0.);

        while jx + 8 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights_h = xvld1q_f32_x2(ptr);
            let read_weights = vcombine_f16(
                vcvt_f16_f32(read_weights_h.0),
                vcvt_f16_f32(read_weights_h.1),
            );
            let bounds_start = bounds.start + jx;
            store_0 = conv_horiz_8_rgba_f16(bounds_start, src, read_weights, store_0);
            let s_ptr_1 = src.get_unchecked(src_stride..);
            store_1 = conv_horiz_8_rgba_f16(bounds_start, s_ptr_1, read_weights, store_1);
            let s_ptr2 = src.get_unchecked(src_stride * 2..);
            store_2 = conv_horiz_8_rgba_f16(bounds_start, s_ptr2, read_weights, store_2);
            let s_ptr3 = src.get_unchecked(src_stride * 3..);
            store_3 = conv_horiz_8_rgba_f16(bounds_start, s_ptr3, read_weights, store_3);
            jx += 8;
        }

        while jx + 4 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights = vcvt_f16_f32(vld1q_f32(ptr));
            let bounds_start = bounds.start + jx;
            store_0 = conv_horiz_4_rgba_f16(bounds_start, src, read_weights, store_0);
            let s_ptr_1 = src.get_unchecked(src_stride..);
            store_1 = conv_horiz_4_rgba_f16(bounds_start, s_ptr_1, read_weights, store_1);
            let s_ptr2 = src.get_unchecked(src_stride * 2..);
            store_2 = conv_horiz_4_rgba_f16(bounds_start, s_ptr2, read_weights, store_2);
            let s_ptr3 = src.get_unchecked(src_stride * 3..);
            store_3 = conv_horiz_4_rgba_f16(bounds_start, s_ptr3, read_weights, store_3);
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights_h = vld1_f32(ptr);
            let read_weights = vcvt_f16_f32(vcombine_f32(read_weights_h, read_weights_h));
            let bounds_start = bounds.start + jx;
            store_0 = conv_horiz_rgba_2_f32(bounds_start, src, read_weights, store_0);
            let ptr_1 = src.get_unchecked(src_stride..);
            store_1 = conv_horiz_rgba_2_f32(bounds_start, ptr_1, read_weights, store_1);
            let ptr_2 = src.get_unchecked(src_stride * 2..);
            store_2 = conv_horiz_rgba_2_f32(bounds_start, ptr_2, read_weights, store_2);
            let ptr_3 = src.get_unchecked(src_stride * 3..);
            store_3 = conv_horiz_rgba_2_f32(bounds_start, ptr_3, read_weights, store_3);
            jx += 2;
        }

        while jx < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight0 = vcvt_f16_f32(vld1q_dup_f32(ptr));
            let bounds_start = bounds.start + jx;
            store_0 = conv_horiz_rgba_1_f16(bounds_start, src, weight0, store_0);
            let ptr_1 = src.get_unchecked(src_stride..);
            store_1 = conv_horiz_rgba_1_f16(bounds_start, ptr_1, weight0, store_1);
            let ptr_2 = src.get_unchecked(src_stride * 2..);
            store_2 = conv_horiz_rgba_1_f16(bounds_start, ptr_2, weight0, store_2);
            let ptr_3 = src.get_unchecked(src_stride * 3..);
            store_3 = conv_horiz_rgba_1_f16(bounds_start, ptr_3, weight0, store_3);
            jx += 1;
        }

        let px = x * CHANNELS;
        let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        vst1_f16(dest_ptr, store_0);

        let dest_ptr = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
        vst1_f16(dest_ptr, store_1);

        let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
        vst1_f16(dest_ptr, store_2);

        let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
        vst1_f16(dest_ptr, store_3);

        filter_offset += filter_weights.aligned_size;
    }
}
