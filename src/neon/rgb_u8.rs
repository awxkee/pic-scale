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
use crate::neon::utils::load_3b_as_u16x4;
use crate::support::{PRECISION, ROUNDING_CONST};
use std::arch::aarch64::*;

#[inline]
unsafe fn conv_horiz_rgba_4_u8(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    w1: int16x8_t,
    w2: int16x4_t,
    w3: int16x8_t,
    store: int32x4_t,
    shuffle: uint8x16_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let px_lo = vld1_u8(src_ptr.as_ptr());
    let px_hi_part = vset_lane_u32::<0>(
        (src_ptr.get_unchecked(8..).as_ptr() as *const u32).read_unaligned(),
        vdup_n_u32(0),
    );

    let mut rgb_pixel = vcombine_u8(px_lo, vreinterpret_u8_u32(px_hi_part));
    rgb_pixel = vqtbl1q_u8(rgb_pixel, shuffle);
    let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb_pixel));
    let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb_pixel)));

    let acc = vmlal_high_s16(store, hi, w3);
    let acc = vmlal_s16(acc, vget_low_s16(hi), w2);
    let acc = vmlal_high_s16(acc, lo, w1);
    vmlal_s16(acc, vget_low_s16(lo), w0)
}

#[inline]
unsafe fn conv_horiz_rgba_2_u8(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    w1: int16x8_t,
    store: int32x4_t,
    shuffle: uint8x8_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let mut rgb_pixel = vdup_n_u32(0);
    rgb_pixel = vset_lane_u32::<0>((src_ptr.as_ptr() as *const u32).read_unaligned(), rgb_pixel);
    rgb_pixel = vreinterpret_u32_u16(vset_lane_u16::<2>(
        (src_ptr.get_unchecked(4..).as_ptr() as *const u16).read_unaligned(),
        vreinterpret_u16_u32(rgb_pixel),
    ));
    rgb_pixel = vreinterpret_u32_u8(vtbl1_u8(vreinterpret_u8_u32(rgb_pixel), shuffle));

    let wide = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(rgb_pixel)));

    let acc = vmlal_high_s16(store, wide, w1);
    vmlal_s16(acc, vget_low_s16(wide), w0)
}

#[inline]
unsafe fn conv_horiz_rgba_1_u8(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgb_pixel = load_3b_as_u16x4(src_ptr.as_ptr());
    let lo = vreinterpret_s16_u16(rgb_pixel);
    vmlal_s16(store, lo, w0)
}

#[inline]
unsafe fn write_accumulator_u8(store: int32x4_t, dst: &mut [u8]) {
    let zeros = vdupq_n_s32(0i32);
    let store_16 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store, zeros));
    let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));
    let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
    let bytes = pixel.to_le_bytes();
    let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
    (dst.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
    *dst.get_unchecked_mut(2) = bytes[2];
}

pub fn convolve_horizontal_rgb_neon_rows_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
        let shuffle_1 = vld1_u8(shuf_table_1.as_ptr());
        let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
        let shuffle_2 = vld1_u8(shuf_table_2.as_ptr());
        let shuffle = vcombine_u8(shuffle_1, shuffle_2);

        // (r0 g0 b0 r1) (g2 b2 r3 g3) (b3 r4 g4 b4) (r5 g5 b5 r6)

        const CHANNELS: usize = 3;
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

            while jx + 4 < bounds.size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                let w0 = vdup_lane_s16::<0>(weights);
                let w1 = vdupq_lane_s16::<1>(weights);
                let w2 = vdup_lane_s16::<2>(weights);
                let w3 = vdupq_lane_s16::<3>(weights);
                store_0 =
                    conv_horiz_rgba_4_u8(bounds_start, src0, w0, w1, w2, w3, store_0, shuffle);
                store_1 =
                    conv_horiz_rgba_4_u8(bounds_start, src1, w0, w1, w2, w3, store_1, shuffle);
                store_2 =
                    conv_horiz_rgba_4_u8(bounds_start, src2, w0, w1, w2, w3, store_2, shuffle);
                store_3 =
                    conv_horiz_rgba_4_u8(bounds_start, src3, w0, w1, w2, w3, store_3, shuffle);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bnds = bounds.start + jx;
                let w0 = vld1_dup_s16(w_ptr.as_ptr());
                let w1 = vld1q_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
                store_0 = conv_horiz_rgba_2_u8(bnds, src0, w0, w1, store_0, shuffle_1);
                store_1 = conv_horiz_rgba_2_u8(bnds, src1, w0, w1, store_1, shuffle_1);
                store_2 = conv_horiz_rgba_2_u8(bnds, src2, w0, w1, store_2, shuffle_1);
                store_3 = conv_horiz_rgba_2_u8(bnds, src3, w0, w1, store_3, shuffle_1);
                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let bnds = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_1_u8(bnds, src0, weight0, store_0);
                store_1 = conv_horiz_rgba_1_u8(bnds, src1, weight0, store_1);
                store_2 = conv_horiz_rgba_1_u8(bnds, src2, weight0, store_2);
                store_3 = conv_horiz_rgba_1_u8(bnds, src3, weight0, store_3);
                jx += 1;
            }

            write_accumulator_u8(store_0, chunk0);
            write_accumulator_u8(store_1, chunk1);
            write_accumulator_u8(store_2, chunk2);
            write_accumulator_u8(store_3, chunk3);
        }
    }
}

pub fn convolve_horizontal_rgb_neon_row_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CHANNELS: usize = 3;

        let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
        let shuffle_1 = vld1_u8(shuf_table_1.as_ptr());
        let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
        let shuffle_2 = vld1_u8(shuf_table_2.as_ptr());
        let shuffle = vcombine_u8(shuffle_1, shuffle_2);

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

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                let w0 = vdup_lane_s16::<0>(weights);
                let w1 = vdupq_lane_s16::<1>(weights);
                let w2 = vdup_lane_s16::<2>(weights);
                let w3 = vdupq_lane_s16::<3>(weights);
                store = conv_horiz_rgba_4_u8(bounds_start, src, w0, w1, w2, w3, store, shuffle);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let weight1 = vld1q_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
                store = conv_horiz_rgba_2_u8(bounds_start, src, weight0, weight1, store, shuffle_1);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let bnds = bounds.start + jx;
                store = conv_horiz_rgba_1_u8(bnds, src, weight0, store);
                jx += 1;
            }

            write_accumulator_u8(store, dst);
        }
    }
}
