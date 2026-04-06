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
#[inline]
#[target_feature(enable = "rdm")]
fn conv_horiz_1_s16(start_x: usize, src: &[i16], w0: int32x4_t, store: int32x4_t) -> int32x4_t {
    unsafe {
        const CN: usize = 1;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let px = vld1_lane_s16::<0>(src_ptr.as_ptr(), vdup_n_s16(0));
        let lo = vshll_n_s16::<6>(px);
        vqrdmlahq_s32(store, lo, w0)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "rdm")]
fn conv_horiz_2_s16(start_x: usize, src: &[i16], w0: int32x4_t, store: int32x4_t) -> int32x4_t {
    unsafe {
        const CN: usize = 1;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let px = vreinterpret_s16_u32(vld1_lane_u32::<0>(src_ptr.as_ptr().cast(), vdup_n_u32(0)));
        vqrdmlahq_s32(store, vshll_n_s16::<6>(px), w0)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "rdm")]
fn conv_horiz_4_s16(
    start_x: usize,
    src: &[i16],
    weights: int32x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const CN: usize = 1;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let px = vld1_s16(src_ptr.as_ptr());
        vqrdmlahq_s32(store, vshll_n_s16::<6>(px), weights)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "rdm")]
fn conv_horiz_8_s16(
    start_x: usize,
    src: &[i16],
    weights: (int32x4_t, int32x4_t),
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const CN: usize = 1;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let pixels = vld1q_s16(src_ptr.as_ptr());
        let acc = vqrdmlahq_s32(store, vshll_high_n_s16::<6>(pixels), weights.1);
        vqrdmlahq_s32(acc, vshll_n_s16::<6>(vget_low_s16(pixels)), weights.0)
    }
}

pub(crate) fn convolve_horizontal_plane_neon_rows_4_hb_s16(
    src: &[i16],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_plane_neon_rows_4_hb_s16_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        )
    }
}

#[target_feature(enable = "rdm")]
fn convolve_horizontal_plane_neon_rows_4_hb_s16_impl(
    src: &[i16],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i32>,
    bit_depth: u32,
) {
    unsafe {
        let init = vld1q_s32([1i32 << 5, 0, 0, 0].as_ptr());

        let v_max_colors = vdup_n_s16(((1i32 << (bit_depth - 1)) - 1) as i16);
        let v_min_colors = vdup_n_s16((-(1i32 << (bit_depth - 1))) as i16);

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.iter_mut();
        let iter_row1 = row1_ref.iter_mut();
        let iter_row2 = row2_ref.iter_mut();
        let iter_row3 = row3_ref.iter_mut();

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

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = (
                    vld1q_s32(w_ptr.as_ptr()),
                    vld1q_s32(w_ptr.get_unchecked(4..).as_ptr()),
                );
                store_0 = conv_horiz_8_s16(bounds_start, src0, weights_set, store_0);
                store_1 = conv_horiz_8_s16(bounds_start, src1, weights_set, store_1);
                store_2 = conv_horiz_8_s16(bounds_start, src2, weights_set, store_2);
                store_3 = conv_horiz_8_s16(bounds_start, src3, weights_set, store_3);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = vld1q_s32(w_ptr.as_ptr());
                store_0 = conv_horiz_4_s16(bounds_start, src0, ws, store_0);
                store_1 = conv_horiz_4_s16(bounds_start, src1, ws, store_1);
                store_2 = conv_horiz_4_s16(bounds_start, src2, ws, store_2);
                store_3 = conv_horiz_4_s16(bounds_start, src3, ws, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = vcombine_s32(vld1_s32(w_ptr.as_ptr()), vdup_n_s32(0));
                store_0 = conv_horiz_2_s16(bounds_start, src0, w0, store_0);
                store_1 = conv_horiz_2_s16(bounds_start, src1, w0, store_1);
                store_2 = conv_horiz_2_s16(bounds_start, src2, w0, store_2);
                store_3 = conv_horiz_2_s16(bounds_start, src3, w0, store_3);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1q_dup_s32(w_ptr.as_ptr());
                store_0 = conv_horiz_1_s16(bounds_start, src0, weight0, store_0);
                store_1 = conv_horiz_1_s16(bounds_start, src1, weight0, store_1);
                store_2 = conv_horiz_1_s16(bounds_start, src2, weight0, store_2);
                store_3 = conv_horiz_1_s16(bounds_start, src3, weight0, store_3);
                jx += 1;
            }

            let packed = vpaddq_s32(vpaddq_s32(store_0, store_1), vpaddq_s32(store_2, store_3));
            let narrowed = vqshrn_n_s32::<6>(packed);
            let clamped = vmin_s16(vmax_s16(narrowed, v_min_colors), v_max_colors);

            vst1_lane_s16::<0>(chunk0, clamped);
            vst1_lane_s16::<1>(chunk1, clamped);
            vst1_lane_s16::<2>(chunk2, clamped);
            vst1_lane_s16::<3>(chunk3, clamped);
        }
    }
}

pub(crate) fn convolve_horizontal_plane_neon_s16_hb_row(
    src: &[i16],
    dst: &mut [i16],
    filter_weights: &FilterWeights<i32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_plane_neon_s16_hb_impl(src, dst, filter_weights, bit_depth);
    }
}

#[target_feature(enable = "rdm")]
fn convolve_horizontal_plane_neon_s16_hb_impl(
    src: &[i16],
    dst: &mut [i16],
    filter_weights: &FilterWeights<i32>,
    bit_depth: u32,
) {
    unsafe {
        let v_max_colors = vdup_n_s16(((1i32 << (bit_depth - 1)) - 1) as i16);
        let v_min_colors = vdup_n_s16((-(1i32 << (bit_depth - 1))) as i16);

        let init = vld1q_s32([1i32 << 5, 0, 0, 0].as_ptr());

        for ((dst, bounds), weights) in dst.iter_mut().zip(filter_weights.bounds.iter()).zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        ) {
            let bounds_size = bounds.size;
            let mut jx = 0usize;
            let mut store = init;

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = (
                    vld1q_s32(w_ptr.as_ptr()),
                    vld1q_s32(w_ptr.get_unchecked(4..).as_ptr()),
                );
                store = conv_horiz_8_s16(bounds_start, src, weights_set, store);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let ws = vld1q_s32(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_4_s16(bounds_start, src, ws, store);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = vcombine_s32(vld1_s32(w_ptr.as_ptr()), vdup_n_s32(0));
                store = conv_horiz_2_s16(bounds_start, src, w0, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = vld1q_dup_s32(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_1_s16(bounds_start, src, weight0, store);
                jx += 1;
            }

            let packed = vpaddq_s32(vpaddq_s32(store, vdupq_n_s32(0)), vdupq_n_s32(0));
            let narrowed = vqshrn_n_s32::<6>(packed);
            let clamped = vmin_s16(vmax_s16(narrowed, v_min_colors), v_max_colors);

            vst1_lane_s16::<0>(dst, clamped);
        }
    }
}
