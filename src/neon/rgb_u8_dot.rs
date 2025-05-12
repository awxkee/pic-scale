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

#[inline(always)]
unsafe fn write_accumulator_u8(store: int32x4_t, dst: &mut [u8]) {
    unsafe {
        let store_16 = vqshrun_n_s32::<7>(store);
        let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));
        vst1_lane_u16::<0>(
            dst.as_mut_ptr() as *mut u16,
            vreinterpret_u16_u8(store_16_8),
        );
        vst1_lane_u8::<2>(dst.get_unchecked_mut(2..).as_mut_ptr(), store_16_8);
    }
}

#[inline(always)]
unsafe fn load_3b_as_u8x16(src_ptr: &[u8]) -> uint8x16_t {
    unsafe {
        let v = vreinterpretq_u8_u16(vld1q_lane_u16::<0>(
            src_ptr.as_ptr() as *const u16,
            vdupq_n_u16(0),
        ));
        vld1q_lane_u8::<2>(src_ptr.get_unchecked(2..).as_ptr(), v)
    }
}

#[inline(always)]
unsafe fn load_2x3b_as_u8x16(src_ptr: &[u8]) -> uint8x16_t {
    unsafe {
        let mut rgb_pixel = vld1q_lane_u32::<0>(src_ptr.as_ptr() as *const u32, vdupq_n_u32(0));
        rgb_pixel = vreinterpretq_u32_u16(vld1q_lane_u16::<2>(
            src_ptr.get_unchecked(4..).as_ptr() as *const u16,
            vreinterpretq_u16_u32(rgb_pixel),
        ));
        vreinterpretq_u8_u32(rgb_pixel)
    }
}

#[inline(always)]
unsafe fn load_4x3b_as_u8x16(src_ptr: &[u8]) -> uint8x16_t {
    unsafe {
        let px_lo = vld1_u8(src_ptr.as_ptr());
        let px_hi_part = vld1_lane_u32::<0>(
            src_ptr.get_unchecked(8..).as_ptr() as *const u32,
            vdup_n_u32(0),
        );
        vcombine_u8(px_lo, vreinterpret_u8_u32(px_hi_part))
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_rows_4_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgb_neon_rows_4_impl(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve_horizontal_rgb_neon_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        let tbl: [u8; 16] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255];
        let v_tbl = vld1q_u8(tbl.as_ptr());
        let weights_tbl: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let v_weights = vld1q_u8(weights_tbl.as_ptr());

        // (r0 g0 b0 r1) (g2 b2 r3 g3) (b3 r4 g4 b4) (r5 g5 b5 r6)

        let rnd_const: i32 = 1 << 6;

        const CN: usize = 3;
        let init = vdupq_n_s32(rnd_const);
        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.chunks_exact_mut(CN);
        let iter_row1 = row1_ref.chunks_exact_mut(CN);
        let iter_row2 = row2_ref.chunks_exact_mut(CN);
        let iter_row3 = row3_ref.chunks_exact_mut(CN);

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
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s32(vld1q_lane_s32::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s32(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);

                let rgb_pixel0 = load_4x3b_as_u8x16(src0.get_unchecked((bounds_start * CN)..));
                let rgb_pixel1 = load_4x3b_as_u8x16(src1.get_unchecked((bounds_start * CN)..));
                let rgb_pixel2 = load_4x3b_as_u8x16(src2.get_unchecked((bounds_start * CN)..));
                let rgb_pixel3 = load_4x3b_as_u8x16(src3.get_unchecked((bounds_start * CN)..));
                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgb_pixel0, v_tbl), v_weight);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgb_pixel1, v_tbl), v_weight);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgb_pixel2, v_tbl), v_weight);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgb_pixel3, v_tbl), v_weight);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let v_weight = vqtbl1q_s8(
                    vreinterpretq_s8_s16(vld1q_dup_s16(w_ptr.as_ptr() as *const _)),
                    v_weights,
                );
                let rgb_pixel0 = load_2x3b_as_u8x16(src0.get_unchecked((bounds_start * CN)..));
                let rgb_pixel1 = load_2x3b_as_u8x16(src1.get_unchecked((bounds_start * CN)..));
                let rgb_pixel2 = load_2x3b_as_u8x16(src2.get_unchecked((bounds_start * CN)..));
                let rgb_pixel3 = load_2x3b_as_u8x16(src3.get_unchecked((bounds_start * CN)..));
                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgb_pixel0, v_tbl), v_weight);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgb_pixel1, v_tbl), v_weight);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgb_pixel2, v_tbl), v_weight);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgb_pixel3, v_tbl), v_weight);
                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let weight0 = vld1q_dup_s8(w_ptr.as_ptr());
                let rgb_pixel0 = load_3b_as_u8x16(src0.get_unchecked((bounds_start * CN)..));
                let rgb_pixel1 = load_3b_as_u8x16(src1.get_unchecked((bounds_start * CN)..));
                let rgb_pixel2 = load_3b_as_u8x16(src2.get_unchecked((bounds_start * CN)..));
                let rgb_pixel3 = load_3b_as_u8x16(src3.get_unchecked((bounds_start * CN)..));
                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgb_pixel0, v_tbl), weight0);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgb_pixel1, v_tbl), weight0);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgb_pixel2, v_tbl), weight0);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgb_pixel3, v_tbl), weight0);
                jx += 1;
            }

            write_accumulator_u8(store_0, chunk0);
            write_accumulator_u8(store_1, chunk1);
            write_accumulator_u8(store_2, chunk2);
            write_accumulator_u8(store_3, chunk3);
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_row_one_dot(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgb_neon_row_one_impl_dot(src, dst, filter_weights);
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve_horizontal_rgb_neon_row_one_impl_dot(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        const CN: usize = 3;

        let tbl: [u8; 16] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255];
        let v_tbl = vld1q_u8(tbl.as_ptr());
        let weights_tbl: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let v_weights = vld1q_u8(weights_tbl.as_ptr());

        let rnd_const: i32 = 1 << 6;

        for ((dst, bounds), weights) in dst
            .chunks_exact_mut(CN)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let bounds_size = bounds.size;

            let mut jx = 0usize;
            let mut store = vdupq_n_s32(rnd_const);

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s32(vld1q_lane_s32::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s32(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);

                let src_ptr = src.get_unchecked((bounds_start * CN)..);
                let rgb_pixel = load_4x3b_as_u8x16(src_ptr);
                store = vusdotq_s32(store, vqtbl1q_u8(rgb_pixel, v_tbl), v_weight);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let v_weight = vqtbl1q_s8(
                    vreinterpretq_s8_s16(vld1q_dup_s16(w_ptr.as_ptr() as *const _)),
                    v_weights,
                );

                let src_ptr = src.get_unchecked((bounds_start * CN)..);
                let rgb_pixel = load_2x3b_as_u8x16(src_ptr);
                store = vusdotq_s32(store, vqtbl1q_u8(rgb_pixel, v_tbl), v_weight);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = vld1q_dup_s8(w_ptr.as_ptr());
                let start = bounds.start + jx;

                let src_ptr = src.get_unchecked((start * CN)..);
                let rgb_pixel = load_3b_as_u8x16(src_ptr);
                store = vusdotq_s32(store, vqtbl1q_u8(rgb_pixel, v_tbl), weight0);
                jx += 1;
            }

            write_accumulator_u8(store, dst);
        }
    }
}
