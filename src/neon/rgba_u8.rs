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
use crate::neon::utils::{load_4b_as_u16x4, xvld1q_u8_x2};
use crate::support::PRECISION;
use crate::support::ROUNDING_CONST;
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u8(
    start_x: usize,
    src: &[u8],
    weights: int16x8_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = xvld1q_u8_x2(src_ptr.as_ptr());

    let hi0 = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.0));
    let lo0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.0)));
    let hi1 = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.1));
    let lo1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.1)));

    let mut acc = vmlal_high_laneq_s16::<3>(store, hi0, weights);
    acc = vmlal_laneq_s16::<2>(acc, vget_low_s16(hi0), weights);
    acc = vmlal_high_laneq_s16::<1>(acc, lo0, weights);
    acc = vmlal_laneq_s16::<0>(acc, vget_low_s16(lo0), weights);

    acc = vmlal_high_laneq_s16::<7>(acc, hi1, weights);
    acc = vmlal_laneq_s16::<6>(acc, vget_low_s16(hi1), weights);
    acc = vmlal_high_laneq_s16::<5>(acc, lo1, weights);
    acc = vmlal_laneq_s16::<4>(acc, vget_low_s16(lo1), weights);
    acc
}

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    weights: int16x8_t,
    store: int16x4_t,
) -> int16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = xvld1q_u8_x2(src_ptr.as_ptr());

    let hi0 = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.0)));
    let lo0 = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.0))));
    let hi1 = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.1)));
    let lo1 = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.1))));

    let mut product = vqrdmlah_laneq_s16::<3>(store, vget_high_s16(hi0), weights);
    product = vqrdmlah_laneq_s16::<2>(product, vget_low_s16(hi0), weights);
    product = vqrdmlah_laneq_s16::<1>(product, vget_high_s16(lo0), weights);
    product = vqrdmlah_laneq_s16::<0>(product, vget_low_s16(lo0), weights);
    product = vqrdmlah_laneq_s16::<7>(product, vget_high_s16(hi1), weights);
    product = vqrdmlah_laneq_s16::<6>(product, vget_low_s16(hi1), weights);
    product = vqrdmlah_laneq_s16::<5>(product, vget_high_s16(lo1), weights);
    product = vqrdmlah_laneq_s16::<4>(product, vget_low_s16(lo1), weights);
    product
}

#[inline(always)]
unsafe fn conv_horiz_rgba_2_u8(
    start_x: usize,
    src: &[u8],
    weights: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgb_pixel = vld1_u8(src_ptr.as_ptr());
    let wide = vreinterpretq_s16_u16(vmovl_u8(rgb_pixel));

    let acc = vmlal_high_lane_s16::<1>(store, wide, weights);
    vmlal_lane_s16::<0>(acc, vget_low_s16(wide), weights)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_2_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    weights: int16x4_t,
    store: int16x4_t,
) -> int16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgb_pixel = vld1_u8(src_ptr.as_ptr());
    let wide = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_u8(rgb_pixel)));

    let product = vqrdmlah_lane_s16::<0>(store, vget_low_s16(wide), weights);
    vqrdmlah_lane_s16::<1>(product, vget_high_s16(wide), weights)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_4_u8(
    start_x: usize,
    src: &[u8],
    weights: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = vld1q_u8(src_ptr.as_ptr());

    let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel));
    let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel)));

    let acc = vmlal_high_lane_s16::<3>(store, hi, weights);
    let acc = vmlal_lane_s16::<2>(acc, vget_low_s16(hi), weights);
    let acc = vmlal_high_lane_s16::<1>(acc, lo, weights);
    vmlal_lane_s16::<0>(acc, vget_low_s16(lo), weights)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_4_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    weights: int16x4_t,
    store: int16x4_t,
) -> int16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = vld1q_u8(src_ptr.as_ptr());

    let hi = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel)));
    let lo = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel))));

    let mut product = vqrdmlah_lane_s16::<3>(store, vget_high_s16(hi), weights);
    product = vqrdmlah_lane_s16::<2>(product, vget_low_s16(hi), weights);
    product = vqrdmlah_lane_s16::<1>(product, vget_high_s16(lo), weights);
    product = vqrdmlah_lane_s16::<0>(product, vget_low_s16(lo), weights);
    product
}

#[inline(always)]
unsafe fn conv_horiz_rgba_1_u8(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel = load_4b_as_u16x4(src_ptr.as_ptr());
    let lo = vreinterpret_s16_u16(rgba_pixel);
    vmlal_s16(store, lo, w0)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_1_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    store: int16x4_t,
) -> int16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel = vshl_n_u16::<SCALE>(load_4b_as_u16x4(src_ptr.as_ptr()));
    let lo = vreinterpret_s16_u16(rgba_pixel);
    vqrdmlah_s16(store, lo, w0)
}

pub(crate) fn convolve_horizontal_rgba_neon_rows_4_u8_i16(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_neon_rows_4_u8_i16_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

/// Slightly lower precision scale option
#[target_feature(enable = "rdm")]
unsafe fn convolve_horizontal_rgba_neon_rows_4_u8_i16_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 4;
    const SCALE: i32 = 6;
    const ROUNDING: i16 = 1 << (SCALE - 1);
    let init = vdup_n_s16(ROUNDING);

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

        let bounds_size = bounds.size;

        let mut store_0 = init;
        let mut store_1 = init;
        let mut store_2 = init;
        let mut store_3 = init;

        let src0 = src;
        let src1 = src0.get_unchecked(src_stride..);
        let src2 = src1.get_unchecked(src_stride..);
        let src3 = src2.get_unchecked(src_stride..);

        while jx + 8 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 8));
            let weights_set = vld1q_s16(w_ptr.as_ptr());
            store_0 = conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src0, weights_set, store_0);
            store_1 = conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src1, weights_set, store_1);
            store_2 = conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src2, weights_set, store_2);
            store_3 = conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src3, weights_set, store_3);
            jx += 8;
        }

        while jx + 4 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let weights = vld1_s16(w_ptr.as_ptr());
            store_0 = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src0, weights, store_0);
            store_1 = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src1, weights, store_1);
            store_2 = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src2, weights, store_2);
            store_3 = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src3, weights, store_3);
            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let mut v_weight = vld1_dup_s16(w_ptr.as_ptr());
            v_weight = vld1_lane_s16::<1>(w_ptr.as_ptr().add(1), v_weight);
            store_0 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src0, v_weight, store_0);
            store_1 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src1, v_weight, store_1);
            store_2 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src2, v_weight, store_2);
            store_3 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src3, v_weight, store_3);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let bounds_start = bounds.start + jx;
            let weight0 = vld1_dup_s16(w_ptr.as_ptr());
            store_0 = conv_horiz_rgba_1_u8_i16::<SCALE>(bounds_start, src0, weight0, store_0);
            store_1 = conv_horiz_rgba_1_u8_i16::<SCALE>(bounds_start, src1, weight0, store_1);
            store_2 = conv_horiz_rgba_1_u8_i16::<SCALE>(bounds_start, src2, weight0, store_2);
            store_3 = conv_horiz_rgba_1_u8_i16::<SCALE>(bounds_start, src3, weight0, store_3);
            jx += 1;
        }

        let store_16_0 = vshr_n_s16::<SCALE>(store_0);
        let store_16_1 = vshr_n_s16::<SCALE>(store_1);
        let store_16_2 = vshr_n_s16::<SCALE>(store_2);
        let store_16_3 = vshr_n_s16::<SCALE>(store_3);

        let store_16_8_0 = vqmovun_s16(vcombine_s16(store_16_0, store_16_0));
        let store_16_8_1 = vqmovun_s16(vcombine_s16(store_16_1, store_16_1));
        let store_16_8_2 = vqmovun_s16(vcombine_s16(store_16_2, store_16_2));
        let store_16_8 = vqmovun_s16(vcombine_s16(store_16_3, store_16_3));

        let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8_0));
        let dest_ptr_32 = chunk0.as_mut_ptr() as *mut u32;
        dest_ptr_32.write_unaligned(pixel);

        let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8_1));
        let dest_ptr_32 = chunk1.as_mut_ptr() as *mut u32;
        dest_ptr_32.write_unaligned(pixel);

        let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8_2));
        let dest_ptr_32 = chunk2.as_mut_ptr() as *mut u32;
        dest_ptr_32.write_unaligned(pixel);

        let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
        let dest_ptr_32 = chunk3.as_mut_ptr() as *mut u32;
        dest_ptr_32.write_unaligned(pixel);
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_rows_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CHANNELS: usize = 4;
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

            let bounds_size = bounds.size;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 8 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..(jx + 8));
                let weights_set = vld1q_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_8_u8(bounds_start, src0, weights_set, store_0);
                store_1 = conv_horiz_rgba_8_u8(bounds_start, src1, weights_set, store_1);
                store_2 = conv_horiz_rgba_8_u8(bounds_start, src2, weights_set, store_2);
                store_3 = conv_horiz_rgba_8_u8(bounds_start, src3, weights_set, store_3);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_4_u8(bounds_start, src0, weights, store_0);
                store_1 = conv_horiz_rgba_4_u8(bounds_start, src1, weights, store_1);
                store_2 = conv_horiz_rgba_4_u8(bounds_start, src2, weights, store_2);
                store_3 = conv_horiz_rgba_4_u8(bounds_start, src3, weights, store_3);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bounds_start = bounds.start + jx;
                let mut v_weight = vld1_dup_s16(w_ptr.as_ptr());
                v_weight = vld1_lane_s16::<1>(w_ptr.as_ptr().add(1), v_weight);
                store_0 = conv_horiz_rgba_2_u8(bounds_start, src0, v_weight, store_0);
                store_1 = conv_horiz_rgba_2_u8(bounds_start, src1, v_weight, store_1);
                store_2 = conv_horiz_rgba_2_u8(bounds_start, src2, v_weight, store_2);
                store_3 = conv_horiz_rgba_2_u8(bounds_start, src3, v_weight, store_3);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_1_u8(bounds_start, src0, weight0, store_0);
                store_1 = conv_horiz_rgba_1_u8(bounds_start, src1, weight0, store_1);
                store_2 = conv_horiz_rgba_1_u8(bounds_start, src2, weight0, store_2);
                store_3 = conv_horiz_rgba_1_u8(bounds_start, src3, weight0, store_3);
                jx += 1;
            }

            let store_16_0 = vqshrun_n_s32::<PRECISION>(store_0);
            let store_16_1 = vqshrun_n_s32::<PRECISION>(store_1);
            let store_16_2 = vqshrun_n_s32::<PRECISION>(store_2);
            let store_16_3 = vqshrun_n_s32::<PRECISION>(store_3);

            let store_16_8_0 = vqmovn_u16(vcombine_u16(store_16_0, store_16_0));
            let store_16_8_1 = vqmovn_u16(vcombine_u16(store_16_1, store_16_1));
            let store_16_8_2 = vqmovn_u16(vcombine_u16(store_16_2, store_16_2));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16_3, store_16_3));

            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8_0));
            let dest_ptr_32 = chunk0.as_mut_ptr() as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8_1));
            let dest_ptr_32 = chunk1.as_mut_ptr() as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8_2));
            let dest_ptr_32 = chunk2.as_mut_ptr() as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = chunk3.as_mut_ptr() as *mut u32;
            dest_ptr_32.write_unaligned(pixel);
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_row(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CHANNELS: usize = 4;

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
                let w_ptr = weights.get_unchecked(jx..(jx + 8));
                let weights_set = vld1q_s16(w_ptr.as_ptr());
                store = conv_horiz_rgba_8_u8(bounds_start, src, weights_set, store);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_4_u8(bounds_start, src, weights, store);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bounds_start = bounds.start + jx;
                let mut v_weight = vld1_dup_s16(w_ptr.as_ptr());
                v_weight = vld1_lane_s16::<1>(w_ptr.as_ptr().add(1), v_weight);
                store = conv_horiz_rgba_2_u8(bounds_start, src, v_weight, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_1_u8(bounds_start, src, weight0, store);
                jx += 1;
            }

            let store_16 = vqshrun_n_s32::<PRECISION>(store);
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            let value = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dst.as_mut_ptr() as *mut u32;
            dest_ptr_32.write_unaligned(value);
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_row_i16(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_neon_row_i16_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "rdm")]
unsafe fn convolve_horizontal_rgba_neon_row_i16_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    const SCALE: i32 = 6;
    const ROUNDING: i16 = 1 << (SCALE - 1);
    const CHANNELS: usize = 4;

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
        let mut store = vdup_n_s16(ROUNDING);

        while jx + 8 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 8));
            let weights_set = vld1q_s16(w_ptr.as_ptr());
            store = conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src, weights_set, store);
            jx += 8;
        }

        while jx + 4 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let weights = vld1_s16(w_ptr.as_ptr());
            let bounds_start = bounds.start + jx;
            store = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src, weights, store);
            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let mut v_weight = vld1_dup_s16(w_ptr.as_ptr());
            v_weight = vld1_lane_s16::<1>(w_ptr.as_ptr().add(1), v_weight);
            store = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src, v_weight, store);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weight0 = vld1_dup_s16(w_ptr.as_ptr());
            let bounds_start = bounds.start + jx;
            store = conv_horiz_rgba_1_u8_i16::<SCALE>(bounds_start, src, weight0, store);
            jx += 1;
        }

        let store_16 = vshr_n_s16::<SCALE>(store);
        let store_16_8 = vqmovun_s16(vcombine_s16(store_16, store_16));

        let value = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
        let dest_ptr_32 = dst.as_mut_ptr() as *mut u32;
        dest_ptr_32.write_unaligned(value);
    }
}
