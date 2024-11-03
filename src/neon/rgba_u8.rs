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
use crate::neon::utils::load_4b_as_u16x4;
use crate::support::PRECISION;
use crate::support::ROUNDING_CONST;
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u8(
    start_x: usize,
    src: &[u8],
    set1: (int16x8_t, int16x8_t, int16x8_t, int16x8_t),
    set2: (int16x8_t, int16x8_t, int16x8_t, int16x8_t),
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = vld1q_u8_x2(src_ptr.as_ptr());

    let hi0 = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.0));
    let lo0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.0)));
    let hi1 = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.1));
    let lo1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.1)));

    let mut acc = vmlal_high_s16(store, hi0, set1.3);
    acc = vmlal_s16(acc, vget_low_s16(hi0), vget_low_s16(set1.2));
    acc = vmlal_high_s16(acc, lo0, set1.1);
    acc = vmlal_s16(acc, vget_low_s16(lo0), vget_low_s16(set1.0));

    acc = vmlal_high_s16(acc, hi1, set2.3);
    acc = vmlal_s16(acc, vget_low_s16(hi1), vget_low_s16(set2.2));
    acc = vmlal_high_s16(acc, lo1, set2.1);
    acc = vmlal_s16(acc, vget_low_s16(lo1), vget_low_s16(set2.0));
    acc
}

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    set1: (int16x4_t, int16x4_t, int16x4_t, int16x4_t),
    set2: (int16x4_t, int16x4_t, int16x4_t, int16x4_t),
    store: int16x4_t,
) -> int16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = vld1q_u8_x2(src_ptr.as_ptr());

    let hi0 = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.0)));
    let lo0 = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.0))));
    let hi1 = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel.1)));
    let lo1 = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel.1))));

    let hi_v = vqrdmulhq_s16(hi0, vcombine_s16(set1.2, set1.3));
    let mut product = vqrdmlahq_s16(hi_v, lo0, vcombine_s16(set1.0, set1.1));
    product = vqrdmlahq_s16(product, hi1, vcombine_s16(set2.2, set2.3));
    product = vqrdmlahq_s16(product, lo1, vcombine_s16(set2.0, set2.1));

    vadd_s16(
        vadd_s16(store, vget_low_s16(product)),
        vget_high_s16(product),
    )
}

#[inline]
unsafe fn conv_horiz_rgba_2_u8(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    w1: int16x8_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgb_pixel = vld1_u8(src_ptr.as_ptr());
    let wide = vreinterpretq_s16_u16(vmovl_u8(rgb_pixel));

    let acc = vmlal_high_s16(store, wide, w1);
    vmlal_s16(acc, vget_low_s16(wide), w0)
}

#[inline]
unsafe fn conv_horiz_rgba_2_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    w1: int16x4_t,
    store: int16x4_t,
) -> int16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgb_pixel = vld1_u8(src_ptr.as_ptr());
    let wide = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_u8(rgb_pixel)));

    let product = vqrdmulhq_s16(wide, vcombine_s16(w0, w1));

    vadd_s16(
        vadd_s16(store, vget_low_s16(product)),
        vget_high_s16(product),
    )
}

#[inline]
unsafe fn conv_horiz_rgba_4_u8(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    w1: int16x8_t,
    w2: int16x4_t,
    w3: int16x8_t,
    store: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = vld1q_u8(src_ptr.as_ptr());

    let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel));
    let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel)));

    let acc = vmlal_high_s16(store, hi, w3);
    let acc = vmlal_s16(acc, vget_low_s16(hi), w2);
    let acc = vmlal_high_s16(acc, lo, w1);
    vmlal_s16(acc, vget_low_s16(lo), w0)
}

#[inline]
unsafe fn conv_horiz_rgba_4_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    w1: int16x4_t,
    w2: int16x4_t,
    w3: int16x4_t,
    store: int16x4_t,
) -> int16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = vld1q_u8(src_ptr.as_ptr());

    let hi = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel)));
    let lo = vshlq_n_s16::<SCALE>(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel))));

    let hi_v = vqrdmulhq_s16(hi, vcombine_s16(w2, w3));
    let product = vqrdmlahq_s16(hi_v, lo, vcombine_s16(w0, w1));

    vadd_s16(
        vadd_s16(store, vget_low_s16(product)),
        vget_high_s16(product),
    )
}

#[inline]
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

#[inline]
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

/// Slightly lower precision scale option
pub fn convolve_horizontal_rgba_neon_rows_4_u8_i16(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CHANNELS: usize = 4;
        const SCALE: i32 = 6;
        const ROUNDING: i16 = 1 << (SCALE - 1);
        let zeros = vdup_n_s16(0i16);
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
                let w0 = vdup_laneq_s16::<0>(weights_set);
                let w1 = vdup_laneq_s16::<1>(weights_set);
                let w2 = vdup_laneq_s16::<2>(weights_set);
                let w3 = vdup_laneq_s16::<3>(weights_set);
                let w4 = vdup_laneq_s16::<4>(weights_set);
                let w5 = vdup_laneq_s16::<5>(weights_set);
                let w6 = vdup_laneq_s16::<6>(weights_set);
                let w7 = vdup_laneq_s16::<7>(weights_set);
                let set1 = (w0, w1, w2, w3);
                let set2 = (w4, w5, w6, w7);
                store_0 =
                    conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src0, set1, set2, store_0);
                store_1 =
                    conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src1, set1, set2, store_1);
                store_2 =
                    conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src2, set1, set2, store_2);
                store_3 =
                    conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src3, set1, set2, store_3);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                let w0 = vdup_lane_s16::<0>(weights);
                let w1 = vdup_lane_s16::<1>(weights);
                let w2 = vdup_lane_s16::<2>(weights);
                let w3 = vdup_lane_s16::<3>(weights);
                store_0 =
                    conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src0, w0, w1, w2, w3, store_0);
                store_1 =
                    conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src1, w0, w1, w2, w3, store_1);
                store_2 =
                    conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src2, w0, w1, w2, w3, store_2);
                store_3 =
                    conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src3, w0, w1, w2, w3, store_3);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bounds_start = bounds.start + jx;
                let w0 = vld1_dup_s16(w_ptr.as_ptr());
                let w1 = vld1_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
                store_0 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src0, w0, w1, store_0);
                store_1 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src1, w0, w1, store_1);
                store_2 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src2, w0, w1, store_2);
                store_3 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src3, w0, w1, store_3);
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

            let store_16_0 = vreinterpret_u16_s16(vshr_n_s16::<SCALE>(vmax_s16(store_0, zeros)));
            let store_16_1 = vreinterpret_u16_s16(vshr_n_s16::<SCALE>(vmax_s16(store_1, zeros)));
            let store_16_2 = vreinterpret_u16_s16(vshr_n_s16::<SCALE>(vmax_s16(store_2, zeros)));
            let store_16_3 = vreinterpret_u16_s16(vshr_n_s16::<SCALE>(vmax_s16(store_3, zeros)));

            let store_16_8_0 = vmovn_u16(vcombine_u16(store_16_0, store_16_0));
            let store_16_8_1 = vmovn_u16(vcombine_u16(store_16_1, store_16_1));
            let store_16_8_2 = vmovn_u16(vcombine_u16(store_16_2, store_16_2));
            let store_16_8 = vmovn_u16(vcombine_u16(store_16_3, store_16_3));

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

pub fn convolve_horizontal_rgba_neon_rows_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let zeros = vdupq_n_s32(0i32);
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
                let w0 = vdupq_laneq_s16::<0>(weights_set);
                let w1 = vdupq_laneq_s16::<1>(weights_set);
                let w2 = vdupq_laneq_s16::<2>(weights_set);
                let w3 = vdupq_laneq_s16::<3>(weights_set);
                let w4 = vdupq_laneq_s16::<4>(weights_set);
                let w5 = vdupq_laneq_s16::<5>(weights_set);
                let w6 = vdupq_laneq_s16::<6>(weights_set);
                let w7 = vdupq_laneq_s16::<7>(weights_set);
                let set1 = (w0, w1, w2, w3);
                let set2 = (w4, w5, w6, w7);
                store_0 = conv_horiz_rgba_8_u8(bounds_start, src0, set1, set2, store_0);
                store_1 = conv_horiz_rgba_8_u8(bounds_start, src1, set1, set2, store_1);
                store_2 = conv_horiz_rgba_8_u8(bounds_start, src2, set1, set2, store_2);
                store_3 = conv_horiz_rgba_8_u8(bounds_start, src3, set1, set2, store_3);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                let w0 = vdup_lane_s16::<0>(weights);
                let w1 = vdupq_lane_s16::<1>(weights);
                let w2 = vdup_lane_s16::<2>(weights);
                let w3 = vdupq_lane_s16::<3>(weights);
                store_0 = conv_horiz_rgba_4_u8(bounds_start, src0, w0, w1, w2, w3, store_0);
                store_1 = conv_horiz_rgba_4_u8(bounds_start, src1, w0, w1, w2, w3, store_1);
                store_2 = conv_horiz_rgba_4_u8(bounds_start, src2, w0, w1, w2, w3, store_2);
                store_3 = conv_horiz_rgba_4_u8(bounds_start, src3, w0, w1, w2, w3, store_3);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bounds_start = bounds.start + jx;
                let w0 = vld1_dup_s16(w_ptr.as_ptr());
                let w1 = vld1q_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
                store_0 = conv_horiz_rgba_2_u8(bounds_start, src0, w0, w1, store_0);
                store_1 = conv_horiz_rgba_2_u8(bounds_start, src1, w0, w1, store_1);
                store_2 = conv_horiz_rgba_2_u8(bounds_start, src2, w0, w1, store_2);
                store_3 = conv_horiz_rgba_2_u8(bounds_start, src3, w0, w1, store_3);
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

            let store_16_0 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_0, zeros));
            let store_16_1 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_1, zeros));
            let store_16_2 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_2, zeros));
            let store_16_3 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_3, zeros));

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

pub fn convolve_horizontal_rgba_neon_row(
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
                let w0 = vdupq_laneq_s16::<0>(weights_set);
                let w1 = vdupq_laneq_s16::<1>(weights_set);
                let w2 = vdupq_laneq_s16::<2>(weights_set);
                let w3 = vdupq_laneq_s16::<3>(weights_set);
                let w4 = vdupq_laneq_s16::<4>(weights_set);
                let w5 = vdupq_laneq_s16::<5>(weights_set);
                let w6 = vdupq_laneq_s16::<6>(weights_set);
                let w7 = vdupq_laneq_s16::<7>(weights_set);
                let set1 = (w0, w1, w2, w3);
                let set2 = (w4, w5, w6, w7);
                store = conv_horiz_rgba_8_u8(bounds_start, src, set1, set2, store);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                let weight0 = vdup_lane_s16::<0>(weights);
                let weight1 = vdupq_lane_s16::<1>(weights);
                let weight2 = vdup_lane_s16::<2>(weights);
                let weight3 = vdupq_lane_s16::<3>(weights);
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_4_u8(
                    bounds_start,
                    src,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store,
                );
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let weight1 = vld1q_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
                store = conv_horiz_rgba_2_u8(bounds_start, src, weight0, weight1, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_1_u8(bounds_start, src, weight0, store);
                jx += 1;
            }

            let store_16 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store, vdupq_n_s32(0i32)));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            let value = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dst.as_mut_ptr() as *mut u32;
            dest_ptr_32.write_unaligned(value);
        }
    }
}

pub fn convolve_horizontal_rgba_neon_row_i16(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const SCALE: i32 = 6;
        const ROUNDING: i16 = 1 << (SCALE - 1);
        let zeros = vdup_n_s16(0i16);
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
                let w0 = vdup_laneq_s16::<0>(weights_set);
                let w1 = vdup_laneq_s16::<1>(weights_set);
                let w2 = vdup_laneq_s16::<2>(weights_set);
                let w3 = vdup_laneq_s16::<3>(weights_set);
                let w4 = vdup_laneq_s16::<4>(weights_set);
                let w5 = vdup_laneq_s16::<5>(weights_set);
                let w6 = vdup_laneq_s16::<6>(weights_set);
                let w7 = vdup_laneq_s16::<7>(weights_set);
                let set1 = (w0, w1, w2, w3);
                let set2 = (w4, w5, w6, w7);
                store = conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src, set1, set2, store);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                let weight0 = vdup_lane_s16::<0>(weights);
                let weight1 = vdup_lane_s16::<1>(weights);
                let weight2 = vdup_lane_s16::<2>(weights);
                let weight3 = vdup_lane_s16::<3>(weights);
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_4_u8_i16::<SCALE>(
                    bounds_start,
                    src,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store,
                );
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let weight1 = vld1_dup_s16(w_ptr.get_unchecked(1..).as_ptr());
                store =
                    conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src, weight0, weight1, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_1_u8_i16::<SCALE>(bounds_start, src, weight0, store);
                jx += 1;
            }

            let store_16 = vreinterpret_u16_s16(vshr_n_s16::<SCALE>(vmax_s16(store, zeros)));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            let value = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dst.as_mut_ptr() as *mut u32;
            dest_ptr_32.write_unaligned(value);
        }
    }
}
