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
use core::f16;
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn write_rgb_f16(store: float32x4_t, dest_ptr: &mut [f16; 3]) {
    let cvt = vreinterpret_u16_f16(vcvt_f16_f32(store));
    unsafe {
        vst1_lane_u32::<0>(dest_ptr.as_mut_ptr().cast(), vreinterpret_u32_u16(cvt));
        vst1_lane_u16::<2>(dest_ptr[2..].as_mut_ptr().cast(), cvt);
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "neon,fp16,fhm")]
fn conv_horiz_4_rgb_f16(
    start_x: usize,
    src: &[f16],
    w: float16x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked(start_x * CN..).as_ptr();

        let rgb_pixel_s = vld1q_u16(src_ptr.cast());
        let rgb_pixel_n = vld1_u16(src_ptr.add(8).cast());

        let rgb_first_u = vget_low_u16(rgb_pixel_s);
        let rgb_first = rgb_first_u;
        let rgb_second_u = vext_u16::<3>(vget_low_u16(rgb_pixel_s), vget_high_u16(rgb_pixel_s));
        let rgb_second = rgb_second_u;

        let rgb_third_u = vext_u16::<2>(vget_high_u16(rgb_pixel_s), rgb_pixel_n);
        let rgb_third = rgb_third_u;

        let rgb_fourth_u = vext_u16::<1>(rgb_pixel_n, rgb_pixel_n);
        let rgb_fourth = rgb_fourth_u;

        let f0 = vreinterpretq_f16_u16(vcombine_u16(rgb_first, rgb_second));
        let f1 = vreinterpretq_f16_u16(vcombine_u16(rgb_third, rgb_fourth));

        let acc = vfmlalq_lane_low_f16::<0>(store, f0, w);
        let acc = vfmlalq_lane_high_f16::<1>(acc, f0, w);
        let acc = vfmlalq_lane_low_f16::<2>(acc, f1, w);
        vfmlalq_lane_high_f16::<3>(acc, f1, w)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "neon,fp16,fhm")]
fn conv_horiz_2_rgb_f16(
    start_x: usize,
    src: &[f16],
    w: float16x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked(start_x * CN..).as_ptr();

        let rgb_pixel = vld1_u16(src_ptr.cast());
        let second_px =
            vreinterpret_u16_u32(vld1_lane_u32::<0>(src_ptr.add(4).cast(), vdup_n_u32(0)));

        let rgb_first_u = rgb_pixel;
        let rgb_first = rgb_first_u;
        let rgb_second_u = vext_u16::<3>(rgb_pixel, second_px);
        let rgb_second = rgb_second_u;

        let f0 = vreinterpretq_f16_u16(vcombine_u16(rgb_first, rgb_second));

        let acc = vfmlalq_lane_low_f16::<0>(store, f0, w);
        vfmlalq_lane_high_f16::<1>(acc, f0, w)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "neon,fp16,fhm")]
fn conv_horiz_1_rgb_f16(
    start_x: usize,
    src: &[f16],
    w: float16x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked(start_x * CN..).as_ptr();

        let mut fq = vreinterpret_u16_u32(vld1_lane_u32::<0>(src_ptr.cast(), vdup_n_u32(0)));
        fq = vld1_lane_u16::<2>(src_ptr.add(2).cast(), fq);

        let rgb_pixel = vreinterpret_f16_u16(fq);

        vfmlalq_lane_low_f16::<0>(store, vcombine_f16(rgb_pixel, rgb_pixel), w)
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_rows_4_f16_fhm(
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f16>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_neon_rows_4_f16_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        )
    }
}

#[target_feature(enable = "fp16,fhm")]
fn convolve_horizontal_rgb_neon_rows_4_f16_impl(
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f16>,
) {
    unsafe {
        const CN: usize = 3;

        let zeros = vdupq_n_f32(0.);

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

            while jx + 4 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let w_s = weights.get_unchecked(jx);
                let read_weights = vld1_f16(w_s);
                store_0 = conv_horiz_4_rgb_f16(bounds_start, src, read_weights, store_0);
                let s_ptr1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_4_rgb_f16(bounds_start, s_ptr1, read_weights, store_1);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_4_rgb_f16(bounds_start, s_ptr2, read_weights, store_2);
                let s_ptr = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_4_rgb_f16(bounds_start, s_ptr, read_weights, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let w_s = weights.get_unchecked(jx..);
                let read_weights =
                    vreinterpret_f16_u16(vreinterpret_u16_u32(vld1_dup_u32(w_s.as_ptr().cast())));
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
                let w_s = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let weight0 = vreinterpret_f16_u16(vld1_dup_u16(w_s.as_ptr().cast()));
                store_0 = conv_horiz_1_rgb_f16(bounds_start, src, weight0, store_0);
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_1_rgb_f16(bounds_start, s_ptr_1, weight0, store_1);
                let s_ptr_2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_1_rgb_f16(bounds_start, s_ptr_2, weight0, store_2);
                let s_ptr_3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_1_rgb_f16(bounds_start, s_ptr_3, weight0, store_3);
                jx += 1;
            }

            write_rgb_f16(store_0, chunk0);
            write_rgb_f16(store_1, chunk1);
            write_rgb_f16(store_2, chunk2);
            write_rgb_f16(store_3, chunk3);
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_row_one_f16_fhm(
    src: &[f16],
    dst: &mut [f16],
    filter_weights: &FilterWeights<f16>,
    _: u32,
) {
    unsafe { convolve_horizontal_rgb_neon_row_one_f16_impl(src, dst, filter_weights) }
}

#[target_feature(enable = "fhm")]
fn convolve_horizontal_rgb_neon_row_one_f16_impl(
    src: &[f16],
    dst: &mut [f16],
    filter_weights: &FilterWeights<f16>,
) {
    unsafe {
        const CN: usize = 3;

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
                store = conv_horiz_4_rgb_f16(bounds_start, src, read_weights, store);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let w_s = weights.get_unchecked(jx..);
                let read_weights =
                    vreinterpret_f16_u16(vreinterpret_u16_u32(vld1_dup_u32(w_s.as_ptr().cast())));
                store = conv_horiz_2_rgb_f16(bounds_start, src, read_weights, store);
                jx += 2;
            }

            while jx < bounds.size {
                let w_s = weights.get_unchecked(jx..);
                let weight0 = vreinterpret_f16_u16(vld1_dup_u16(w_s.as_ptr().cast()));
                let bounds_start = bounds.start + jx;
                store = conv_horiz_1_rgb_f16(bounds_start, src, weight0, store);
                jx += 1;
            }

            write_rgb_f16(store, dst);
        }
    }
}
