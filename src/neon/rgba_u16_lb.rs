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
use std::arch::aarch64::*;

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_1_u16(
    start_x: usize,
    src: &[u16],
    w0: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
        let rgba_pixel = vld1_u16(src_ptr.as_ptr());
        let lo = vreinterpret_s16_u16(rgba_pixel);
        vqdmlal_s16(store, lo, w0)
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_2_u16(
    start_x: usize,
    src: &[u16],
    w0: int16x4_t,
    w1: int16x8_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let rgb_pixel = vld1q_u16(src_ptr.as_ptr());
        let wide = vreinterpretq_s16_u16(rgb_pixel);

        let acc = vqdmlal_high_s16(store, wide, w1);
        vqdmlal_s16(acc, vget_low_s16(wide), w0)
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_4_u16(
    start_x: usize,
    src: &[u16],
    weights: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let rgba_pixel = xvld1q_u16_x2(src_ptr.as_ptr());

        let hi = vreinterpretq_s16_u16(rgba_pixel.1);
        let lo = vreinterpretq_s16_u16(rgba_pixel.0);

        let acc = vqdmlal_high_lane_s16::<3>(store, hi, weights);
        let acc = vqdmlal_lane_s16::<2>(acc, vget_low_s16(hi), weights);
        let acc = vqdmlal_high_lane_s16::<1>(acc, lo, weights);
        vqdmlal_lane_s16::<0>(acc, vget_low_s16(lo), weights)
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_8_u16(
    start_x: usize,
    src: &[u16],
    weights: int16x8_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const COMPONENTS: usize = 4;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let rgba_pixel = xvld1q_u16_x4(src_ptr.as_ptr());

        let hi0 = vreinterpretq_s16_u16(rgba_pixel.1);
        let lo0 = vreinterpretq_s16_u16(rgba_pixel.0);
        let hi1 = vreinterpretq_s16_u16(rgba_pixel.3);
        let lo1 = vreinterpretq_s16_u16(rgba_pixel.2);

        let mut acc = vqdmlal_high_laneq_s16::<3>(store, hi0, weights);
        acc = vqdmlal_laneq_s16::<2>(acc, vget_low_s16(hi0), weights);
        acc = vqdmlal_high_laneq_s16::<1>(acc, lo0, weights);
        acc = vqdmlal_laneq_s16::<0>(acc, vget_low_s16(lo0), weights);

        acc = vqdmlal_high_laneq_s16::<7>(acc, hi1, weights);
        acc = vqdmlal_laneq_s16::<6>(acc, vget_low_s16(hi1), weights);
        acc = vqdmlal_high_laneq_s16::<5>(acc, lo1, weights);
        acc = vqdmlal_laneq_s16::<4>(acc, vget_low_s16(lo1), weights);
        acc
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_rows_4_lb_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        const CHANNELS: usize = 4;
        const PRECISION: i32 = 16;
        const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
        let init = vdupq_n_s32(ROUNDING_CONST);

        let v_max_colors = vdup_n_u16((1 << bit_depth) - 1);

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
        let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
        let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
        let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

        for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
            .zip(iter_row1)
            .zip(iter_row2)
            .zip(iter_row3)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
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

            while jx + 8 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = vld1q_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_8_u16(bounds_start, src0, weights_set, store_0);
                store_1 = conv_horiz_rgba_8_u16(bounds_start, src1, weights_set, store_1);
                store_2 = conv_horiz_rgba_8_u16(bounds_start, src2, weights_set, store_2);
                store_3 = conv_horiz_rgba_8_u16(bounds_start, src3, weights_set, store_3);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_4_u16(bounds_start, src0, weights, store_0);
                store_1 = conv_horiz_rgba_4_u16(bounds_start, src1, weights, store_1);
                store_2 = conv_horiz_rgba_4_u16(bounds_start, src2, weights, store_2);
                store_3 = conv_horiz_rgba_4_u16(bounds_start, src3, weights, store_3);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = vld1_dup_s16(w_ptr.as_ptr());
                let w1 = vld1q_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
                store_0 = conv_horiz_rgba_2_u16(bounds_start, src0, w0, w1, store_0);
                store_1 = conv_horiz_rgba_2_u16(bounds_start, src1, w0, w1, store_1);
                store_2 = conv_horiz_rgba_2_u16(bounds_start, src2, w0, w1, store_2);
                store_3 = conv_horiz_rgba_2_u16(bounds_start, src3, w0, w1, store_3);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_1_u16(bounds_start, src0, weight0, store_0);
                store_1 = conv_horiz_rgba_1_u16(bounds_start, src1, weight0, store_1);
                store_2 = conv_horiz_rgba_1_u16(bounds_start, src2, weight0, store_2);
                store_3 = conv_horiz_rgba_1_u16(bounds_start, src3, weight0, store_3);
                jx += 1;
            }

            let j0 = vqshrun_n_s32::<PRECISION>(store_0);
            let j1 = vqshrun_n_s32::<PRECISION>(store_1);
            let j2 = vqshrun_n_s32::<PRECISION>(store_2);
            let j3 = vqshrun_n_s32::<PRECISION>(store_3);

            let store_16_0 = vmin_u16(j0, v_max_colors);
            let store_16_1 = vmin_u16(j1, v_max_colors);
            let store_16_2 = vmin_u16(j2, v_max_colors);
            let store_16_3 = vmin_u16(j3, v_max_colors);

            vst1_u16(chunk0.as_mut_ptr(), store_16_0);
            vst1_u16(chunk1.as_mut_ptr(), store_16_1);
            vst1_u16(chunk2.as_mut_ptr(), store_16_2);
            vst1_u16(chunk3.as_mut_ptr(), store_16_3);
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_u16_lb_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        const CHANNELS: usize = 4;

        let v_max_colors = vdup_n_u16((1 << bit_depth) - 1);

        const PRECISION: i32 = 16;
        const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

        for ((dst, bounds), weights) in dst
            .chunks_exact_mut(CHANNELS)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let bounds_size = bounds.size;
            let mut jx = 0usize;
            let mut store = vdupq_n_s32(ROUNDING_CONST);

            while jx + 8 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = vld1q_s16(w_ptr.as_ptr());
                store = conv_horiz_rgba_8_u16(bounds_start, src, weights_set, store);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_4_u16(bounds_start, src, weights, store);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let weight1 = vld1q_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
                store = conv_horiz_rgba_2_u16(bounds_start, src, weight0, weight1, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_1_u16(bounds_start, src, weight0, store);
                jx += 1;
            }

            let store_16_0 = vmin_u16(vqshrun_n_s32::<PRECISION>(store), v_max_colors);

            vst1_u16(dst.as_mut_ptr(), store_16_0);
        }
    }
}
