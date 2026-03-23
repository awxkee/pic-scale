/*
 * Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use std::arch::aarch64::*;

#[must_use]
#[inline]
#[target_feature(enable = "neon")]
fn conv_horiz_rgba_1_u16(
    start_x: usize,
    src: &[u16],
    w0: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let vl0 = vreinterpret_u16_u32(vld1_lane_u32::<0>(src_ptr.as_ptr().cast(), vdup_n_u32(0)));
        let rgba_pixel = vld1_lane_u16::<2>(src_ptr.get_unchecked(2..).as_ptr(), vl0);
        let lo = vcvtq_f32_u32(vmovl_u16(rgba_pixel));
        vfmaq_f32(store, lo, w0)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "neon")]
fn conv_horiz_rgba_2_u16(
    start_x: usize,
    src: &[u16],
    w0: float32x4_t,
    w1: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);

        let l1 = vld1_u16(src_ptr.as_ptr());
        let l2 = vld1_lane_u32::<0>(src_ptr.get_unchecked(4..).as_ptr().cast(), vdup_n_u32(0));

        let hi = vext_u16::<3>(l1, vreinterpret_u16_u32(l2));

        let acc = vfmaq_f32(store, vcvtq_f32_u32(vmovl_u16(l1)), w1);
        vfmaq_f32(acc, vcvtq_f32_u32(vmovl_u16(hi)), w0)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "neon")]
fn conv_horiz_rgba_4_u16(
    start_x: usize,
    src: &[u16],
    weights: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);

        let part0 = vld1q_u16(src_ptr.as_ptr());
        let part1 = vld1_u16(src_ptr.get_unchecked(8..).as_ptr());

        let second = vext_u16::<3>(vget_low_u16(part0), vget_high_u16(part0));
        let third = vext_u16::<2>(vget_high_u16(part0), part1);
        let fourth = vext_u16::<1>(part1, part1);

        let acc = vfmaq_laneq_f32::<3>(store, vcvtq_f32_u32(vmovl_u16(fourth)), weights);
        let acc = vfmaq_laneq_f32::<2>(acc, vcvtq_f32_u32(vmovl_u16(third)), weights);
        let acc = vfmaq_laneq_f32::<1>(acc, vcvtq_f32_u32(vmovl_u16(second)), weights);
        vfmaq_laneq_f32::<0>(acc, vcvtq_f32_u32(vmovl_u16(vget_low_u16(part0))), weights)
    }
}

#[inline(always)]
fn set_pixel(ptr: &mut [u16], pixel: uint16x4_t) {
    unsafe {
        vst1_lane_u32::<0>(ptr.as_mut_ptr().cast(), vreinterpret_u32_u16(pixel));
        vst1_lane_u16::<2>(ptr.get_unchecked_mut(2..).as_mut_ptr(), pixel);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_f32_to_u16(store: float32x4_t, v_max_colors: uint16x4_t) -> uint16x4_t {
    let rounded = vcvtaq_u32_f32(store);
    let narrow = vqmovn_u32(rounded);
    vmin_u16(narrow, v_max_colors)
}

pub(crate) fn convolve_horizontal_rgb_neon_rows_4_u16_f32(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgb_neon_rows_4_hb_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        )
    }
}

#[target_feature(enable = "neon")]
fn convolve_horizontal_rgb_neon_rows_4_hb_impl(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    const CN: usize = 3;
    let init = vdupq_n_f32(0.0f32);
    let v_max_colors = vdup_n_u16(((1u32 << bit_depth) - 1) as u16);

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
        unsafe {
            let mut jx = 0usize;
            let mut store_0 = init;
            let mut store_1 = init;
            let mut store_2 = init;
            let mut store_3 = init;

            let bounds_size = bounds.size;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = vld1q_f32(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_4_u16(bounds_start, src0, ws, store_0);
                store_1 = conv_horiz_rgba_4_u16(bounds_start, src1, ws, store_1);
                store_2 = conv_horiz_rgba_4_u16(bounds_start, src2, ws, store_2);
                store_3 = conv_horiz_rgba_4_u16(bounds_start, src3, ws, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vdupq_n_f32(*w_ptr.get_unchecked(0));
                let w1 = vdupq_n_f32(*w_ptr.get_unchecked(1));
                store_0 = conv_horiz_rgba_2_u16(bounds_start, src0, w0, w1, store_0);
                store_1 = conv_horiz_rgba_2_u16(bounds_start, src1, w0, w1, store_1);
                store_2 = conv_horiz_rgba_2_u16(bounds_start, src2, w0, w1, store_2);
                store_3 = conv_horiz_rgba_2_u16(bounds_start, src3, w0, w1, store_3);
                jx += 2;
            }

            while jx < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = vdupq_n_f32(*w_ptr.get_unchecked(0));
                store_0 = conv_horiz_rgba_1_u16(bounds_start, src0, weight0, store_0);
                store_1 = conv_horiz_rgba_1_u16(bounds_start, src1, weight0, store_1);
                store_2 = conv_horiz_rgba_1_u16(bounds_start, src2, weight0, store_2);
                store_3 = conv_horiz_rgba_1_u16(bounds_start, src3, weight0, store_3);
                jx += 1;
            }

            set_pixel(chunk0, store_f32_to_u16(store_0, v_max_colors));
            set_pixel(chunk1, store_f32_to_u16(store_1, v_max_colors));
            set_pixel(chunk2, store_f32_to_u16(store_2, v_max_colors));
            set_pixel(chunk3, store_f32_to_u16(store_3, v_max_colors));
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_u16_row_f32(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgb_neon_u16_hb_impl(src, dst, filter_weights, bit_depth);
    }
}

#[target_feature(enable = "neon")]
fn convolve_horizontal_rgb_neon_u16_hb_impl(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    const CN: usize = 3;
    let v_max_colors = vdup_n_u16(((1u32 << bit_depth) - 1) as u16);

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
        unsafe {
            let bounds_size = bounds.size;
            let mut jx = 0usize;
            let mut store = vdupq_n_f32(0.0f32);

            while jx + 4 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let ws = vld1q_f32(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_4_u16(bounds_start, src, ws, store);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = vdupq_n_f32(*w_ptr.get_unchecked(0));
                let w1 = vdupq_n_f32(*w_ptr.get_unchecked(1));
                store = conv_horiz_rgba_2_u16(bounds_start, src, w0, w1, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = vdupq_n_f32(*w_ptr.get_unchecked(0));
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_1_u16(bounds_start, src, weight0, store);
                jx += 1;
            }

            set_pixel(dst, store_f32_to_u16(store, v_max_colors));
        }
    }
}
