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
use crate::neon::utils::{xvld1q_u16_x2, xvld1q_u16_x4};
use core::f16;
use std::arch::aarch64::*;

#[must_use]
#[inline]
#[target_feature(enable = "fp16,fhm")]
fn conv_horiz_rgba_8_f16(
    start_x: usize,
    src: &[f16],
    w: float16x8_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked(start_x * CN..).as_ptr();

        let rgb_pixel = xvld1q_u16_x4(src_ptr.cast());

        let mut acc = vfmlalq_laneq_low_f16::<0>(store, vreinterpretq_f16_u16(rgb_pixel.0), w);
        acc = vfmlalq_laneq_high_f16::<1>(acc, vreinterpretq_f16_u16(rgb_pixel.0), w);
        acc = vfmlalq_laneq_low_f16::<2>(acc, vreinterpretq_f16_u16(rgb_pixel.1), w);
        acc = vfmlalq_laneq_high_f16::<3>(acc, vreinterpretq_f16_u16(rgb_pixel.1), w);
        acc = vfmlalq_laneq_low_f16::<4>(acc, vreinterpretq_f16_u16(rgb_pixel.2), w);
        acc = vfmlalq_laneq_high_f16::<5>(acc, vreinterpretq_f16_u16(rgb_pixel.2), w);
        acc = vfmlalq_laneq_low_f16::<6>(acc, vreinterpretq_f16_u16(rgb_pixel.3), w);
        vfmlalq_laneq_high_f16::<7>(acc, vreinterpretq_f16_u16(rgb_pixel.3), w)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "fp16,fhm")]
fn conv_horiz_rgba_4_f16(
    start_x: usize,
    src: &[f16],
    set1: float16x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked(start_x * CN..).as_ptr();

        let rgb_pixel = xvld1q_u16_x2(src_ptr.cast());

        let acc = vfmlalq_lane_low_f16::<0>(store, vreinterpretq_f16_u16(rgb_pixel.0), set1);
        let acc = vfmlalq_lane_high_f16::<1>(acc, vreinterpretq_f16_u16(rgb_pixel.0), set1);
        let acc = vfmlalq_lane_low_f16::<2>(acc, vreinterpretq_f16_u16(rgb_pixel.1), set1);
        vfmlalq_lane_high_f16::<3>(acc, vreinterpretq_f16_u16(rgb_pixel.0), set1)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "fp16,fhm")]
fn conv_horiz_rgba_2_f32(
    start_x: usize,
    src: &[f16],
    set: float16x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked(start_x * CN..).as_ptr();

        let rgb_pixel = vld1q_u16(src_ptr.cast());

        let acc = vfmlalq_lane_low_f16::<0>(store, vreinterpretq_f16_u16(rgb_pixel), set);
        vfmlalq_lane_high_f16::<1>(acc, vreinterpretq_f16_u16(rgb_pixel), set)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "fp16,fhm")]
fn conv_horiz_rgba_1_f16(
    start_x: usize,
    src: &[f16],
    set: float16x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked(start_x * CN..).as_ptr();
        let rgb_pixel = vld1_u16(src_ptr.cast());
        vfmlalq_lane_low_f16::<0>(
            store,
            vreinterpretq_f16_u16(vcombine_u16(rgb_pixel, rgb_pixel)),
            set,
        )
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_row_one_f16_fhm(
    src: &[f16],
    dst: &mut [f16],
    filter_weights: &FilterWeights<f16>,
    _: u32,
) {
    unsafe { convolve_horizontal_rgba_neon_row_one_f16_impl(filter_weights, src, dst) }
}

#[target_feature(enable = "fhm")]
fn convolve_horizontal_rgba_neon_row_one_f16_impl(
    filter_weights: &FilterWeights<f16>,
    src: &[f16],
    dst: &mut [f16],
) {
    unsafe {
        const CN: usize = 4;

        for ((dst, bounds), weights) in dst
            .as_chunks_mut::<CN>()
            .0
            .iter_mut()
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut jx = 0usize;
            let mut store = vdupq_n_f32(0f32);

            while jx + 4 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let w_s = weights.get_unchecked(jx);
                let read_weights = vld1_f16(w_s);
                store = conv_horiz_rgba_4_f16(bounds_start, src, read_weights, store);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let w_s = weights.get_unchecked(jx..);
                let read_weights =
                    vreinterpret_f16_u16(vreinterpret_u16_u32(vld1_dup_u32(w_s.as_ptr().cast())));
                store = conv_horiz_rgba_2_f32(bounds_start, src, read_weights, store);
                jx += 2;
            }

            while jx < bounds.size {
                let bounds_start = bounds.start + jx;
                let w_s = weights.get_unchecked(jx..);
                let weight0 = vreinterpret_f16_u16(vld1_dup_u16(w_s.as_ptr().cast()));
                store = conv_horiz_rgba_1_f16(bounds_start, src, weight0, store);
                jx += 1;
            }

            vst1_f16(dst.as_mut_ptr(), vcvt_f16_f32(store));
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_rows_4_f16_fhm(
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f16>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgba_neon_rows_4_f16_impl(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        )
    }
}

#[target_feature(enable = "fhm")]
fn convolve_horizontal_rgba_neon_rows_4_f16_impl(
    filter_weights: &FilterWeights<f16>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    unsafe {
        const CN: usize = 4;

        let zeros = vdupq_n_f32(0f32);

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
        let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
        let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
        let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

        for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
            .iter_mut()
            .zip(iter_row1.iter_mut())
            .zip(iter_row2.iter_mut())
            .zip(iter_row3.iter_mut())
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 8 <= bounds.size {
                let w_s = weights.get_unchecked(jx);
                let read_weights = vld1q_f16(w_s);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_rgba_8_f16(bounds_start, src, read_weights, store_0);
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_rgba_8_f16(bounds_start, s_ptr_1, read_weights, store_1);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_rgba_8_f16(bounds_start, s_ptr2, read_weights, store_2);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_rgba_8_f16(bounds_start, s_ptr3, read_weights, store_3);
                jx += 8;
            }

            while jx + 4 <= bounds.size {
                let w_s = weights.get_unchecked(jx);
                let read_weights = vld1_f16(w_s);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_rgba_4_f16(bounds_start, src, read_weights, store_0);
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_rgba_4_f16(bounds_start, s_ptr_1, read_weights, store_1);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_rgba_4_f16(bounds_start, s_ptr2, read_weights, store_2);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_rgba_4_f16(bounds_start, s_ptr3, read_weights, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w_s = weights.get_unchecked(jx..);
                let read_weights =
                    vreinterpret_f16_u16(vreinterpret_u16_u32(vld1_dup_u32(w_s.as_ptr().cast())));
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
                let w_s = weights.get_unchecked(jx..);
                let weight0 = vreinterpret_f16_u16(vld1_dup_u16(w_s.as_ptr().cast()));
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

            vst1_f16(chunk0.as_mut_ptr(), vcvt_f16_f32(store_0));
            vst1_f16(chunk1.as_mut_ptr(), vcvt_f16_f32(store_1));
            vst1_f16(chunk2.as_mut_ptr(), vcvt_f16_f32(store_2));
            vst1_f16(chunk3.as_mut_ptr(), vcvt_f16_f32(store_3));
        }
    }
}
