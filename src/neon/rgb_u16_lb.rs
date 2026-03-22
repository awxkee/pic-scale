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
use std::arch::aarch64::*;

#[must_use]
#[inline(always)]
fn conv_horiz_rgba_1_u16(
    start_x: usize,
    src: &[u16],
    w0: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let vl0 = vreinterpret_u16_u32(vld1_lane_u32::<0>(
            src_ptr.as_ptr() as *const _,
            vdup_n_u32(0),
        ));
        let rgba_pixel = vld1_lane_u16::<2>(src_ptr.get_unchecked(2..).as_ptr() as *const _, vl0);
        let lo = vreinterpret_s16_u16(rgba_pixel);
        vmlal_s16(store, lo, w0)
    }
}

#[must_use]
#[inline(always)]
fn conv_horiz_rgba_2_u16(
    start_x: usize,
    src: &[u16],
    w0: int16x4_t,
    w1: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);

        let l1 = vld1_u16(src_ptr.as_ptr());
        let l2 = vld1_lane_u32::<0>(src_ptr.get_unchecked(4..).as_ptr().cast(), vdup_n_u32(0));

        let hi = vext_u16::<3>(l1, vreinterpret_u16_u32(l2));

        let acc = vmlal_s16(store, vreinterpret_s16_u16(hi), w1);
        vmlal_s16(acc, vreinterpret_s16_u16(l1), w0)
    }
}

#[must_use]
#[inline(always)]
fn conv_horiz_rgba_4_u16(
    start_x: usize,
    src: &[u16],
    weights: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);

        let part0 = vld1q_u16(src_ptr.as_ptr());
        let part1 = vld1_u16(src_ptr.get_unchecked(8..).as_ptr());

        let second = vext_u16::<3>(vget_low_u16(part0), vget_high_u16(part0));
        let third = vext_u16::<2>(vget_high_u16(part0), part1);
        let fourth = vext_u16::<1>(part1, part1);

        let acc = vmlal_lane_s16::<3>(store, vreinterpret_s16_u16(fourth), weights);
        let acc = vmlal_lane_s16::<2>(acc, vreinterpret_s16_u16(third), weights);
        let acc = vmlal_lane_s16::<1>(acc, vreinterpret_s16_u16(second), weights);
        vmlal_lane_s16::<0>(acc, vreinterpret_s16_u16(vget_low_u16(part0)), weights)
    }
}

#[must_use]
#[inline(always)]
fn conv_horiz_rgba_8_u16(
    start_x: usize,
    src: &[u16],
    weights: int16x8_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);

        let part0 = vld1q_u16(src_ptr.as_ptr()); // [r0,g0,b0,r1,g1,b1,r2,g2]
        let part1 = vld1q_u16(src_ptr.get_unchecked(8..).as_ptr()); // [b2,r3,g3,b3,r4,g4,b4,r5]
        let part2 = vld1q_u16(src_ptr.get_unchecked(16..).as_ptr()); // [g5,b5,r6,g6,b6,r7,g7,b7]

        // pixel 0: [r0,g0,b0,_] = low(part0)
        // pixel 1: [r1,g1,b1,_] = vext<3>(low(part0), high(part0))
        // pixel 2: [r2,g2,b2,_] = vext<2>(high(part0), low(part1))
        // pixel 3: [r3,g3,b3,_] = vext<1>(low(part1), high(part1))
        let p1 = vext_u16::<3>(vget_low_u16(part0), vget_high_u16(part0));
        let p2 = vext_u16::<2>(vget_high_u16(part0), vget_low_u16(part1));
        let p3 = vext_u16::<1>(vget_low_u16(part1), vget_high_u16(part1));

        let mut acc =
            vmlal_laneq_s16::<0>(store, vreinterpret_s16_u16(vget_low_u16(part0)), weights);
        acc = vmlal_laneq_s16::<1>(acc, vreinterpret_s16_u16(p1), weights);
        acc = vmlal_laneq_s16::<2>(acc, vreinterpret_s16_u16(p2), weights);
        acc = vmlal_laneq_s16::<3>(acc, vreinterpret_s16_u16(p3), weights);

        // pixel 4: [r4,g4,b4,_] = vext<0>(high(part1), high(part1)) = high(part1) directly
        // pixel 5: [r5,g5,b5,_] = vext<3>(high(part1), low(part2))
        // pixel 6: [r6,g6,b6,_] = vext<2>(low(part2), high(part2))
        // pixel 7: [r7,g7,b7,_] = vext<1>(high(part2), high(part2))
        let p5 = vext_u16::<3>(vget_high_u16(part1), vget_low_u16(part2));
        let p6 = vext_u16::<2>(vget_low_u16(part2), vget_high_u16(part2));
        let p7 = vext_u16::<1>(vget_high_u16(part2), vget_high_u16(part2));

        acc = vmlal_laneq_s16::<4>(acc, vreinterpret_s16_u16(vget_high_u16(part1)), weights);
        acc = vmlal_laneq_s16::<5>(acc, vreinterpret_s16_u16(p5), weights);
        acc = vmlal_laneq_s16::<6>(acc, vreinterpret_s16_u16(p6), weights);
        acc = vmlal_laneq_s16::<7>(acc, vreinterpret_s16_u16(p7), weights);
        acc
    }
}

#[inline(always)]
fn set_pixel(ptr: &mut [u16], pixel: uint16x4_t) {
    unsafe {
        vst1_lane_u32::<0>(ptr.as_mut_ptr() as *mut _, vreinterpret_u32_u16(pixel));
        vst1_lane_u16::<2>(ptr.get_unchecked_mut(2..).as_mut_ptr(), pixel);
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_rows_4_lb_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        const CN: usize = 3;
        const PRECISION: i32 = 15;
        const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
        let init = vdupq_n_s32(ROUNDING_CONST);

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
            let mut jx = 0usize;
            let mut store_0 = init;
            let mut store_1 = init;
            let mut store_2 = init;
            let mut store_3 = init;

            let bounds_size = bounds.size;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src0.get_unchecked(src_stride * 2..);
            let src3 = src0.get_unchecked(src_stride * 3..);

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1q_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_8_u16(bounds_start, src0, weights, store_0);
                store_1 = conv_horiz_rgba_8_u16(bounds_start, src1, weights, store_1);
                store_2 = conv_horiz_rgba_8_u16(bounds_start, src2, weights, store_2);
                store_3 = conv_horiz_rgba_8_u16(bounds_start, src3, weights, store_3);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_4_u16(bounds_start, src0, weights, store_0);
                store_1 = conv_horiz_rgba_4_u16(bounds_start, src1, weights, store_1);
                store_2 = conv_horiz_rgba_4_u16(bounds_start, src2, weights, store_2);
                store_3 = conv_horiz_rgba_4_u16(bounds_start, src3, weights, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = vld1_dup_s16(w_ptr.as_ptr());
                let w1 = vld1_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
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

            set_pixel(chunk0, store_16_0);
            set_pixel(chunk1, store_16_1);
            set_pixel(chunk2, store_16_2);
            set_pixel(chunk3, store_16_3);
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_u16_lb_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        const CN: usize = 3;

        let v_max_colors = vdup_n_u16(((1u32 << bit_depth) - 1) as u16);

        const PRECISION: i32 = 15;
        const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

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
            let bounds_size = bounds.size;
            let mut jx = 0usize;
            let mut store = vdupq_n_s32(ROUNDING_CONST);

            while jx + 8 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1q_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_8_u16(bounds_start, src, weights, store);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_4_u16(bounds_start, src, weights, store);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let weight1 = vld1_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
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

            set_pixel(dst, store_16_0);
        }
    }
}
