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
unsafe fn conv_horiz_1_u16(
    start_x: usize,
    src: &[u16],
    w0: int32x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const COMPONENTS: usize = 1;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
        let px = vld1_lane_u16::<0>(src_ptr.as_ptr() as *const _, vdup_n_u16(0));
        let lo = vreinterpretq_s32_u32(vshll_n_u16::<6>(px));
        vqrdmlahq_s32(store, lo, w0)
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_2_u16(
    start_x: usize,
    src: &[u16],
    w0: int32x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const COMPONENTS: usize = 1;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let px = vreinterpret_u16_u32(vld1_lane_u32::<0>(
            src_ptr.as_ptr() as *const _,
            vdup_n_u32(0),
        ));

        vqrdmlahq_s32(store, vreinterpretq_s32_u32(vshll_n_u16::<6>(px)), w0)
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_4_u16(
    start_x: usize,
    src: &[u16],
    weights: int32x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const COMPONENTS: usize = 1;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let px = vld1_u16(src_ptr.as_ptr());

        vqrdmlahq_s32(store, vreinterpretq_s32_u32(vshll_n_u16::<6>(px)), weights)
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_8_u16(
    start_x: usize,
    src: &[u16],
    weights: (int32x4_t, int32x4_t),
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const COMPONENTS: usize = 1;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let pixels = vld1q_u16(src_ptr.as_ptr());

        let acc = vqrdmlahq_s32(
            store,
            vreinterpretq_s32_u32(vshll_high_n_u16::<6>(pixels)),
            weights.1,
        );
        vqrdmlahq_s32(
            acc,
            vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(pixels))),
            weights.0,
        )
    }
}

pub(crate) fn convolve_horizontal_plane_neon_rows_4_hb_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_plane_neon_rows_4_hb_impl(
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
unsafe fn convolve_horizontal_plane_neon_rows_4_hb_impl(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i32>,
    bit_depth: u32,
) {
    unsafe {
        let init = vdupq_n_s32(1 << 5);

        let v_max_colors = (1u32 << bit_depth) - 1;

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

            while jx + 8 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = (
                    vld1q_s32(w_ptr.as_ptr()),
                    vld1q_s32(w_ptr.get_unchecked(4..).as_ptr()),
                );
                store_0 = conv_horiz_8_u16(bounds_start, src0, weights_set, store_0);
                store_1 = conv_horiz_8_u16(bounds_start, src1, weights_set, store_1);
                store_2 = conv_horiz_8_u16(bounds_start, src2, weights_set, store_2);
                store_3 = conv_horiz_8_u16(bounds_start, src3, weights_set, store_3);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1q_s32(w_ptr.as_ptr());
                store_0 = conv_horiz_4_u16(bounds_start, src0, weights, store_0);
                store_1 = conv_horiz_4_u16(bounds_start, src1, weights, store_1);
                store_2 = conv_horiz_4_u16(bounds_start, src2, weights, store_2);
                store_3 = conv_horiz_4_u16(bounds_start, src3, weights, store_3);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = vcombine_s32(vld1_s32(w_ptr.as_ptr()), vdup_n_s32(0));
                store_0 = conv_horiz_2_u16(bounds_start, src0, w0, store_0);
                store_1 = conv_horiz_2_u16(bounds_start, src1, w0, store_1);
                store_2 = conv_horiz_2_u16(bounds_start, src2, w0, store_2);
                store_3 = conv_horiz_2_u16(bounds_start, src3, w0, store_3);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1q_dup_s32(w_ptr.as_ptr());
                store_0 = conv_horiz_1_u16(bounds_start, src0, weight0, store_0);
                store_1 = conv_horiz_1_u16(bounds_start, src1, weight0, store_1);
                store_2 = conv_horiz_1_u16(bounds_start, src2, weight0, store_2);
                store_3 = conv_horiz_1_u16(bounds_start, src3, weight0, store_3);
                jx += 1;
            }

            let store_16_0 = ((vaddvq_s32(store_0).max(0) as u32) >> 6).min(v_max_colors);
            let store_16_1 = ((vaddvq_s32(store_1).max(0) as u32) >> 6).min(v_max_colors);
            let store_16_2 = ((vaddvq_s32(store_2).max(0) as u32) >> 6).min(v_max_colors);
            let store_16_3 = ((vaddvq_s32(store_3).max(0) as u32) >> 6).min(v_max_colors);

            *chunk0 = store_16_0 as u16;
            *chunk1 = store_16_1 as u16;
            *chunk2 = store_16_2 as u16;
            *chunk3 = store_16_3 as u16;
        }
    }
}

pub(crate) fn convolve_horizontal_plane_neon_u16_hb_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_plane_neon_u16_hb_impl(src, dst, filter_weights, bit_depth);
    }
}

#[target_feature(enable = "rdm")]
unsafe fn convolve_horizontal_plane_neon_u16_hb_impl(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i32>,
    bit_depth: u32,
) {
    unsafe {
        let v_max_colors = (1u32 << bit_depth) - 1;

        for ((dst, bounds), weights) in dst.iter_mut().zip(filter_weights.bounds.iter()).zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        ) {
            let bounds_size = bounds.size;
            let mut jx = 0usize;
            let mut store = vdupq_n_s32(1 << 5);

            while jx + 8 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = (
                    vld1q_s32(w_ptr.as_ptr()),
                    vld1q_s32(w_ptr.get_unchecked(4..).as_ptr()),
                );
                store = conv_horiz_8_u16(bounds_start, src, weights_set, store);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1q_s32(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_4_u16(bounds_start, src, weights, store);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = vcombine_s32(vld1_s32(w_ptr.as_ptr()), vdup_n_s32(0));
                store = conv_horiz_2_u16(bounds_start, src, w0, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = vld1q_dup_s32(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_1_u16(bounds_start, src, weight0, store);
                jx += 1;
            }

            let store_16_0 = (((vaddvq_s32(store)) >> 6).max(0) as u32).min(v_max_colors);

            *dst = store_16_0 as u16;
        }
    }
}
