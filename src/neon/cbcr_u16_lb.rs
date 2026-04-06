/*
 * Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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
fn ga_horiz_1(start_x: usize, src: &[u16], w0: int16x4_t, store: int32x4_t) -> int32x4_t {
    unsafe {
        const CN: usize = 2;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let pixel = vld1_lane_u32::<0>(src_ptr.as_ptr().cast(), vdup_n_u32(0));
        let s = vreinterpret_s16_u32(pixel);
        vmlal_s16(store, s, w0)
    }
}

#[must_use]
#[inline(always)]
fn ga_horiz_2(start_x: usize, src: &[u16], w0: int16x4_t, store: int32x4_t) -> int32x4_t {
    unsafe {
        const CN: usize = 2;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let pixel = vld1_u16(src_ptr.as_ptr());
        let s = vreinterpret_s16_u16(pixel);

        vmlal_s16(store, s, w0)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn vadd_s16_p_lo_hi(q: int32x4_t) -> int32x4_t {
    let qz = vextq_s32::<2>(q, q);
    vaddq_s32(q, qz)
}

#[must_use]
#[inline(always)]
fn ga_horiz_4(start_x: usize, src: &[u16], weights: int16x4_t, store: int32x4_t) -> int32x4_t {
    unsafe {
        const CN: usize = 2;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let pixel = vld1q_u16(src_ptr.as_ptr());
        let s = vreinterpretq_s16_u16(pixel);

        let wlo = vzip1_s16(weights, weights);
        let whi = vzip2_s16(weights, weights);
        let acc = vmlal_s16(store, vget_low_s16(s), wlo);
        vmlal_high_s16(acc, s, vcombine_s16(wlo, whi))
    }
}

#[must_use]
#[inline(always)]
fn ga_horiz_8(start_x: usize, src: &[u16], weights: int16x8_t, store: int32x4_t) -> int32x4_t {
    unsafe {
        const CN: usize = 2;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let lo_raw = vld1q_u16(src_ptr.as_ptr());
        let hi_raw = vld1q_u16(src_ptr.get_unchecked(8..).as_ptr());

        let s_lo = vreinterpretq_s16_u16(lo_raw);
        let s_hi = vreinterpretq_s16_u16(hi_raw);

        let w_lo = vget_low_s16(weights);
        let w_hi = vget_high_s16(weights);

        let wlo0 = vzip1_s16(w_lo, w_lo);
        let wlo1 = vzip2_s16(w_lo, w_lo);
        let whi0 = vzip1_s16(w_hi, w_hi);
        let whi1 = vzip2_s16(w_hi, w_hi);

        let acc = vmlal_s16(store, vget_low_s16(s_lo), wlo0);
        let acc = vmlal_high_s16(acc, s_lo, vcombine_s16(wlo0, wlo1));
        let acc = vmlal_s16(acc, vget_low_s16(s_hi), whi0);
        vmlal_high_s16(acc, s_hi, vcombine_s16(whi0, whi1))
    }
}

pub(crate) fn convolve_horizontal_gray_alpha_neon_rows_4_lb_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        const CN: usize = 2;
        const PRECISION: i32 = 15;
        const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
        let init = vld1q_s32([ROUNDING_CONST, ROUNDING_CONST, 0, 0].as_ptr());

        let v_max_colors = vdup_n_u16((1 << bit_depth) - 1);

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
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = vld1q_s16(w_ptr.as_ptr());
                store_0 = ga_horiz_8(bounds_start, src0, weights_set, store_0);
                store_1 = ga_horiz_8(bounds_start, src1, weights_set, store_1);
                store_2 = ga_horiz_8(bounds_start, src2, weights_set, store_2);
                store_3 = ga_horiz_8(bounds_start, src3, weights_set, store_3);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w = vld1_s16(w_ptr.as_ptr());
                store_0 = ga_horiz_4(bounds_start, src0, w, store_0);
                store_1 = ga_horiz_4(bounds_start, src1, w, store_1);
                store_2 = ga_horiz_4(bounds_start, src2, w, store_2);
                store_3 = ga_horiz_4(bounds_start, src3, w, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let mut w0 =
                    vreinterpret_s16_s32(vld1_lane_s32::<0>(w_ptr.as_ptr().cast(), vdup_n_s32(0)));
                w0 = vzip1_s16(w0, w0);
                store_0 = ga_horiz_2(bounds_start, src0, w0, store_0);
                store_1 = ga_horiz_2(bounds_start, src1, w0, store_1);
                store_2 = ga_horiz_2(bounds_start, src2, w0, store_2);
                store_3 = ga_horiz_2(bounds_start, src3, w0, store_3);
                jx += 2;
            }

            store_0 = vadd_s16_p_lo_hi(store_0);
            store_1 = vadd_s16_p_lo_hi(store_1);
            store_2 = vadd_s16_p_lo_hi(store_2);
            store_3 = vadd_s16_p_lo_hi(store_3);

            while jx < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1_dup_s16(w_ptr.as_ptr());
                store_0 = ga_horiz_1(bounds_start, src0, w0, store_0);
                store_1 = ga_horiz_1(bounds_start, src1, w0, store_1);
                store_2 = ga_horiz_1(bounds_start, src2, w0, store_2);
                store_3 = ga_horiz_1(bounds_start, src3, w0, store_3);
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

            vst1_lane_u32::<0>(chunk0.as_mut_ptr().cast(), vreinterpret_u32_u16(store_16_0));
            vst1_lane_u32::<0>(chunk1.as_mut_ptr().cast(), vreinterpret_u32_u16(store_16_1));
            vst1_lane_u32::<0>(chunk2.as_mut_ptr().cast(), vreinterpret_u32_u16(store_16_2));
            vst1_lane_u32::<0>(chunk3.as_mut_ptr().cast(), vreinterpret_u32_u16(store_16_3));
        }
    }
}

pub(crate) fn convolve_horizontal_gray_alpha_neon_u16_lb_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        const CN: usize = 2;
        const PRECISION: i32 = 15;
        const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
        let init = vld1q_s32([ROUNDING_CONST, ROUNDING_CONST, 0, 0].as_ptr());

        let v_max_colors = vdup_n_u16((1 << bit_depth) - 1);

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
            let mut store = init;

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = vld1q_s16(w_ptr.as_ptr());
                store = ga_horiz_8(bounds_start, src, weights_set, store);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w = vld1_s16(w_ptr.as_ptr());
                store = ga_horiz_4(bounds_start, src, w, store);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let mut w0 =
                    vreinterpret_s16_s32(vld1_lane_s32::<0>(w_ptr.as_ptr().cast(), vdup_n_s32(0)));
                w0 = vzip1_s16(w0, w0);
                store = ga_horiz_2(bounds_start, src, w0, store);
                jx += 2;
            }

            store = vadd_s16_p_lo_hi(store);

            while jx < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1_dup_s16(w_ptr.as_ptr());
                store = ga_horiz_1(bounds_start, src, w0, store);
                jx += 1;
            }

            let store_16 = vmin_u16(vqshrun_n_s32::<PRECISION>(store), v_max_colors);

            vst1_lane_u32::<0>(dst.as_mut_ptr().cast(), vreinterpret_u32_u16(store_16));
        }
    }
}
