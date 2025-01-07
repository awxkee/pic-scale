/*
 * Copyright (c) Radzivon Bartoshyk 01/2025. All rights reserved.
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
use crate::neon::utils::{load_12b_as_u8x16, load_3b_as_u8x16, load_6b_as_u8x16};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u8(
    start_x: usize,
    src: &[u8],
    weights0: int8x16_t,
    weights1: int8x16_t,
    shuffle: uint8x16_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let pixel_lo = vld1q_u8(src_ptr.as_ptr());
    let pixel_hi = vcombine_u8(vld1_u8(src_ptr.get_unchecked(16..).as_ptr()), vdup_n_u8(0));
    let created_new = vextq_u8::<12>(pixel_lo, pixel_hi);
    let pixel0 = vqtbl1q_u8(pixel_lo, shuffle);
    let pixel1 = vqtbl1q_u8(created_new, shuffle);
    let v0 = vusdotq_s32(store, pixel0, weights0);
    vusdotq_s32(v0, pixel1, weights1)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_4_u8(
    start_x: usize,
    src: &[u8],
    weights: int8x16_t,
    shuffle: uint8x16_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let pixel = vqtbl1q_u8(load_12b_as_u8x16(src_ptr.as_ptr()), shuffle);
    vusdotq_s32(store, pixel, weights)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_2_u8(
    start_x: usize,
    src: &[u8],
    weights: int8x16_t,
    shuffle: uint8x16_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgb_pixel = vqtbl1q_u8(load_6b_as_u8x16(src_ptr.as_ptr()), shuffle);
    vusdotq_s32(store, rgb_pixel, weights)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_1_u8(
    start_x: usize,
    src: &[u8],
    w0: int8x16_t,
    shuf: uint8x16_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgb_pixel = vqtbl1q_u8(load_3b_as_u8x16(src_ptr.as_ptr()), shuf);
    vusdotq_s32(store, rgb_pixel, w0)
}

#[inline(always)]
unsafe fn write_accumulator_u8(store: int32x4_t, dst: &mut [u8]) {
    let store_16 = vqshrun_n_s32::<7>(store);
    let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));
    vst1_lane_u16::<0>(
        dst.as_mut_ptr() as *mut u16,
        vreinterpret_u16_u8(store_16_8),
    );
    vst1_lane_u8::<2>(dst.as_mut_ptr().add(2), store_16_8);
}

pub(crate) fn convolve_horizontal_rgb_neon_dot_rows_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgb_neon_dot_rows_4_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve_horizontal_rgb_neon_dot_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    let shuffle_v_table: [u8; 16] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255];
    let shuffle_v = vld1q_u8(shuffle_v_table.as_ptr());
    let weights_shuffle_table: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    let weights_shuffle = vld1q_u8(weights_shuffle_table.as_ptr());
    let weights_shuffle_table1: [u8; 16] = [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7];
    let weights_shuffle1 = vld1q_u8(weights_shuffle_table1.as_ptr());

    // (r0 g0 b0 r1) (g2 b2 r3 g3) (b3 r4 g4 b4) (r5 g5 b5 r6)

    const CHANNELS: usize = 3;
    const ROUNDING_CONST: i32 = 1 << 6;
    let init = vdupq_n_s32(ROUNDING_CONST);
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

        let src0 = src;
        let src1 = src0.get_unchecked(src_stride..);
        let src2 = src1.get_unchecked(src_stride..);
        let src3 = src2.get_unchecked(src_stride..);

        while jx + 8 < bounds.size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let weights = vreinterpretq_s8_s64(vld1q_dup_s64(w_ptr.as_ptr() as *const _));
            let weights0 = vqtbl1q_s8(weights, weights_shuffle);
            let weights1 = vqtbl1q_s8(weights, weights_shuffle1);

            store_0 =
                conv_horiz_rgba_8_u8(bounds_start, src0, weights0, weights1, shuffle_v, store_0);
            store_1 =
                conv_horiz_rgba_8_u8(bounds_start, src1, weights0, weights1, shuffle_v, store_1);
            store_2 =
                conv_horiz_rgba_8_u8(bounds_start, src2, weights0, weights1, shuffle_v, store_2);
            store_3 =
                conv_horiz_rgba_8_u8(bounds_start, src3, weights0, weights1, shuffle_v, store_3);
            jx += 8;
        }

        while jx + 4 < bounds.size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let weights = vqtbl1q_s8(
                vreinterpretq_s8_s32(vld1q_dup_s32(w_ptr.as_ptr() as *const _)),
                weights_shuffle,
            );
            store_0 = conv_horiz_rgba_4_u8(bounds_start, src0, weights, shuffle_v, store_0);
            store_1 = conv_horiz_rgba_4_u8(bounds_start, src1, weights, shuffle_v, store_1);
            store_2 = conv_horiz_rgba_4_u8(bounds_start, src2, weights, shuffle_v, store_2);
            store_3 = conv_horiz_rgba_4_u8(bounds_start, src3, weights, shuffle_v, store_3);
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bnds = bounds.start + jx;
            let v_weight = vqtbl1q_s8(
                vreinterpretq_s8_s16(vld1q_dup_s16(w_ptr.as_ptr() as *const _)),
                weights_shuffle,
            );
            store_0 = conv_horiz_rgba_2_u8(bnds, src0, v_weight, shuffle_v, store_0);
            store_1 = conv_horiz_rgba_2_u8(bnds, src1, v_weight, shuffle_v, store_1);
            store_2 = conv_horiz_rgba_2_u8(bnds, src2, v_weight, shuffle_v, store_2);
            store_3 = conv_horiz_rgba_2_u8(bnds, src3, v_weight, shuffle_v, store_3);
            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let bnds = bounds.start + jx;
            let weight0 = vqtbl1q_s8(vld1q_dup_s8(w_ptr.as_ptr()), weights_shuffle);
            store_0 = conv_horiz_rgba_1_u8(bnds, src0, weight0, shuffle_v, store_0);
            store_1 = conv_horiz_rgba_1_u8(bnds, src1, weight0, shuffle_v, store_1);
            store_2 = conv_horiz_rgba_1_u8(bnds, src2, weight0, shuffle_v, store_2);
            store_3 = conv_horiz_rgba_1_u8(bnds, src3, weight0, shuffle_v, store_3);
            jx += 1;
        }

        write_accumulator_u8(store_0, chunk0);
        write_accumulator_u8(store_1, chunk1);
        write_accumulator_u8(store_2, chunk2);
        write_accumulator_u8(store_3, chunk3);
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_dot_row_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgb_neon_dot_row_one_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve_horizontal_rgb_neon_dot_row_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    const CHANNELS: usize = 3;
    const ROUNDING_CONST: i32 = 1 << 6;

    let shuffle_v_table: [u8; 16] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255];
    let shuffle_v = vld1q_u8(shuffle_v_table.as_ptr());
    let weights_shuffle_table: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    let weights_shuffle = vld1q_u8(weights_shuffle_table.as_ptr());
    let weights_shuffle_table1: [u8; 16] = [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7];
    let weights_shuffle1 = vld1q_u8(weights_shuffle_table1.as_ptr());

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
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let weights = vreinterpretq_s8_s64(vld1q_dup_s64(w_ptr.as_ptr() as *const _));
            let weights0 = vqtbl1q_s8(weights, weights_shuffle);
            let weights1 = vqtbl1q_s8(weights, weights_shuffle1);
            store = conv_horiz_rgba_8_u8(bounds_start, src, weights0, weights1, shuffle_v, store);
            jx += 8;
        }

        while jx + 4 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let weights = vqtbl1q_s8(
                vreinterpretq_s8_s32(vld1q_dup_s32(w_ptr.as_ptr() as *const _)),
                weights_shuffle,
            );
            store = conv_horiz_rgba_4_u8(bounds_start, src, weights, shuffle_v, store);
            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let v_weight = vqtbl1q_s8(
                vreinterpretq_s8_s16(vld1q_dup_s16(w_ptr.as_ptr() as *const _)),
                weights_shuffle,
            );
            store = conv_horiz_rgba_2_u8(bounds_start, src, v_weight, shuffle_v, store);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weight0 = vqtbl1q_s8(vld1q_dup_s8(w_ptr.as_ptr()), weights_shuffle);
            let bnds = bounds.start + jx;
            store = conv_horiz_rgba_1_u8(bnds, src, weight0, shuffle_v, store);
            jx += 1;
        }

        write_accumulator_u8(store, dst);
    }
}
