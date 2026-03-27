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
pub(crate) unsafe fn load_4b_as_u8x16(src_ptr: &[u8]) -> uint8x16_t {
    unsafe {
        vreinterpretq_u8_u32(vld1q_lane_u32::<0>(
            src_ptr.as_ptr() as *const u32,
            vdupq_n_u32(0),
        ))
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_rows_4_u8_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgba_neon_rows_4_u8_impl_dot(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[target_feature(enable = "i8mm")]
fn convolve_horizontal_rgba_neon_rows_4_u8_impl_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        const CN: usize = 4;

        let rnd_const: i32 = 1 << (7 - 1);
        let init = vdupq_n_s32(rnd_const);

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
        let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
        let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
        let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

        let tbl: [u8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
        let v_tbl = vld1q_u8(tbl.as_ptr());
        let weights_tbl: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let v_weights = vld1q_u8(weights_tbl.as_ptr());
        let weights_tbl1: [u8; 16] = [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7];
        let v_weights_hi = vld1q_u8(weights_tbl1.as_ptr());
        let weights_tbl2: [u8; 16] = [8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11];
        let v_weights_hi2 = vld1q_u8(weights_tbl2.as_ptr());
        let weights_tbl3: [u8; 16] = [
            12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15,
        ];
        let v_weights_hi3 = vld1q_u8(weights_tbl3.as_ptr());

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

            while jx + 16 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vreinterpretq_s8_s16(vld1q_s16(w_ptr.as_ptr().cast()));
                let w0 = vqtbl1q_s8(weights, v_weights);
                let w1 = vqtbl1q_s8(weights, v_weights_hi);
                let w2 = vqtbl1q_s8(weights, v_weights_hi2);
                let w3 = vqtbl1q_s8(weights, v_weights_hi3);

                let rgba_pixel0 = vld1q_u8(src0.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel1 = vld1q_u8(src1.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel2 = vld1q_u8(src2.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel3 = vld1q_u8(src3.get_unchecked((bounds_start * CN)..).as_ptr());

                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgba_pixel0, v_tbl), w0);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgba_pixel1, v_tbl), w0);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgba_pixel2, v_tbl), w0);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgba_pixel3, v_tbl), w0);

                let rgba_pixel0 = vld1q_u8(src0.get_unchecked((bounds_start * CN) + 16..).as_ptr());
                let rgba_pixel1 = vld1q_u8(src1.get_unchecked((bounds_start * CN) + 16..).as_ptr());
                let rgba_pixel2 = vld1q_u8(src2.get_unchecked((bounds_start * CN) + 16..).as_ptr());
                let rgba_pixel3 = vld1q_u8(src3.get_unchecked((bounds_start * CN) + 16..).as_ptr());

                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgba_pixel0, v_tbl), w1);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgba_pixel1, v_tbl), w1);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgba_pixel2, v_tbl), w1);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgba_pixel3, v_tbl), w1);

                let rgba_pixel0 = vld1q_u8(src0.get_unchecked((bounds_start * CN) + 32..).as_ptr());
                let rgba_pixel1 = vld1q_u8(src1.get_unchecked((bounds_start * CN) + 32..).as_ptr());
                let rgba_pixel2 = vld1q_u8(src2.get_unchecked((bounds_start * CN) + 32..).as_ptr());
                let rgba_pixel3 = vld1q_u8(src3.get_unchecked((bounds_start * CN) + 32..).as_ptr());

                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgba_pixel0, v_tbl), w2);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgba_pixel1, v_tbl), w2);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgba_pixel2, v_tbl), w2);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgba_pixel3, v_tbl), w2);

                let rgba_pixel0 = vld1q_u8(src0.get_unchecked((bounds_start * CN) + 48..).as_ptr());
                let rgba_pixel1 = vld1q_u8(src1.get_unchecked((bounds_start * CN) + 48..).as_ptr());
                let rgba_pixel2 = vld1q_u8(src2.get_unchecked((bounds_start * CN) + 48..).as_ptr());
                let rgba_pixel3 = vld1q_u8(src3.get_unchecked((bounds_start * CN) + 48..).as_ptr());

                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgba_pixel0, v_tbl), w3);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgba_pixel1, v_tbl), w3);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgba_pixel2, v_tbl), w3);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgba_pixel3, v_tbl), w3);

                jx += 16;
            }

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vcombine_s8(vld1_s8(w_ptr.as_ptr().cast()), vdup_n_s8(0));
                let w0 = vqtbl1q_s8(weights, v_weights);
                let w1 = vqtbl1q_s8(weights, v_weights_hi);

                let rgba_pixel0 = vld1q_u8(src0.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel1 = vld1q_u8(src1.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel2 = vld1q_u8(src2.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel3 = vld1q_u8(src3.get_unchecked((bounds_start * CN)..).as_ptr());

                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgba_pixel0, v_tbl), w0);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgba_pixel1, v_tbl), w0);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgba_pixel2, v_tbl), w0);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgba_pixel3, v_tbl), w0);

                let rgba_pixel0 = vld1q_u8(src0.get_unchecked((bounds_start * CN) + 16..).as_ptr());
                let rgba_pixel1 = vld1q_u8(src1.get_unchecked((bounds_start * CN) + 16..).as_ptr());
                let rgba_pixel2 = vld1q_u8(src2.get_unchecked((bounds_start * CN) + 16..).as_ptr());
                let rgba_pixel3 = vld1q_u8(src3.get_unchecked((bounds_start * CN) + 16..).as_ptr());

                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgba_pixel0, v_tbl), w1);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgba_pixel1, v_tbl), w1);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgba_pixel2, v_tbl), w1);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgba_pixel3, v_tbl), w1);

                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s32(vld1q_lane_s32::<0>(
                    w_ptr.as_ptr().cast(),
                    vdupq_n_s32(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);

                let rgba_pixel0 = vld1q_u8(src0.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel1 = vld1q_u8(src1.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel2 = vld1q_u8(src2.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel3 = vld1q_u8(src3.get_unchecked((bounds_start * CN)..).as_ptr());

                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgba_pixel0, v_tbl), v_weight);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgba_pixel1, v_tbl), v_weight);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgba_pixel2, v_tbl), v_weight);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgba_pixel3, v_tbl), v_weight);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let mut v_weight = vreinterpretq_s8_s16(vld1q_lane_s16::<0>(
                    w_ptr.as_ptr().cast(),
                    vdupq_n_s16(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);

                let rgba_pixel0 = vld1_u8(src0.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel1 = vld1_u8(src1.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel2 = vld1_u8(src2.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel3 = vld1_u8(src3.get_unchecked((bounds_start * CN)..).as_ptr());

                store_0 = vusdotq_s32(
                    store_0,
                    vqtbl1q_u8(vcombine_u8(rgba_pixel0, vdup_n_u8(0)), v_tbl),
                    v_weight,
                );
                store_1 = vusdotq_s32(
                    store_1,
                    vqtbl1q_u8(vcombine_u8(rgba_pixel1, vdup_n_u8(0)), v_tbl),
                    v_weight,
                );
                store_2 = vusdotq_s32(
                    store_2,
                    vqtbl1q_u8(vcombine_u8(rgba_pixel2, vdup_n_u8(0)), v_tbl),
                    v_weight,
                );
                store_3 = vusdotq_s32(
                    store_3,
                    vqtbl1q_u8(vcombine_u8(rgba_pixel3, vdup_n_u8(0)), v_tbl),
                    v_weight,
                );
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let weight0 = vld1q_dup_s8(w_ptr.as_ptr());

                let rgba_pixel0 = load_4b_as_u8x16(src0.get_unchecked((bounds_start * CN)..));
                let rgba_pixel1 = load_4b_as_u8x16(src1.get_unchecked((bounds_start * CN)..));
                let rgba_pixel2 = load_4b_as_u8x16(src2.get_unchecked((bounds_start * CN)..));
                let rgba_pixel3 = load_4b_as_u8x16(src3.get_unchecked((bounds_start * CN)..));

                store_0 = vusdotq_s32(store_0, vqtbl1q_u8(rgba_pixel0, v_tbl), weight0);
                store_1 = vusdotq_s32(store_1, vqtbl1q_u8(rgba_pixel1, v_tbl), weight0);
                store_2 = vusdotq_s32(store_2, vqtbl1q_u8(rgba_pixel2, v_tbl), weight0);
                store_3 = vusdotq_s32(store_3, vqtbl1q_u8(rgba_pixel3, v_tbl), weight0);
                jx += 1;
            }

            let store_16_0 = vqshrun_n_s32::<7>(store_0);
            let store_16_1 = vqshrun_n_s32::<7>(store_1);
            let store_16_2 = vqshrun_n_s32::<7>(store_2);
            let store_16_3 = vqshrun_n_s32::<7>(store_3);

            let store_16_8_0 = vqmovn_u16(vcombine_u16(store_16_0, store_16_1));
            let store_16_8_1 = vqmovn_u16(vcombine_u16(store_16_2, store_16_3));

            vst1_lane_u32::<0>(
                chunk0.as_mut_ptr().cast(),
                vreinterpret_u32_u8(store_16_8_0),
            );
            vst1_lane_u32::<1>(
                chunk1.as_mut_ptr().cast(),
                vreinterpret_u32_u8(store_16_8_0),
            );
            vst1_lane_u32::<0>(
                chunk2.as_mut_ptr().cast(),
                vreinterpret_u32_u8(store_16_8_1),
            );
            vst1_lane_u32::<1>(
                chunk3.as_mut_ptr().cast(),
                vreinterpret_u32_u8(store_16_8_1),
            );
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_row_dot(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgba_neon_row_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "i8mm")]
fn convolve_horizontal_rgba_neon_row_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        const CN: usize = 4;
        let rnd_const: i32 = 1 << (7 - 1);

        let tbl: [u8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
        let v_tbl = vld1q_u8(tbl.as_ptr());
        let weights_tbl: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let v_weights = vld1q_u8(weights_tbl.as_ptr());
        let weights_tbl1: [u8; 16] = [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7];
        let v_weights_hi = vld1q_u8(weights_tbl1.as_ptr());

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
            let mut store = vdupq_n_s32(rnd_const);

            while jx + 8 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let v_weight = vcombine_s8(vld1_s8(w_ptr.as_ptr().cast()), vdup_n_s8(0));
                let weights_lo = vqtbl1q_s8(v_weight, v_weights);
                let weights_hi = vqtbl1q_s8(v_weight, v_weights_hi);
                let bounds_start = bounds.start + jx;
                let rgba_pixel0 = vld1q_u8(src.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgba_pixel1 = vld1q_u8(src.get_unchecked((bounds_start * CN + 16)..).as_ptr());

                store = vusdotq_s32(store, vqtbl1q_u8(rgba_pixel0, v_tbl), weights_lo);
                store = vusdotq_s32(store, vqtbl1q_u8(rgba_pixel1, v_tbl), weights_hi);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s32(vld1q_lane_s32::<0>(
                    w_ptr.as_ptr().cast(),
                    vdupq_n_s32(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let bounds_start = bounds.start + jx;
                let rgba_pixel = vld1q_u8(src.get_unchecked((bounds_start * CN)..).as_ptr());

                store = vusdotq_s32(store, vqtbl1q_u8(rgba_pixel, v_tbl), v_weight);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let mut v_weight = vreinterpretq_s8_s16(vld1q_lane_s16::<0>(
                    w_ptr.as_ptr().cast(),
                    vdupq_n_s16(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let rgba_pixel0 = vld1_u8(src.get_unchecked((bounds_start * CN)..).as_ptr());
                store = vusdotq_s32(
                    store,
                    vqtbl1q_u8(vcombine_u8(rgba_pixel0, vdup_n_u8(0)), v_tbl),
                    v_weight,
                );
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = vld1q_dup_s8(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                let rgba_pixel0 = load_4b_as_u8x16(src.get_unchecked((bounds_start * CN)..));
                store = vusdotq_s32(store, vqtbl1q_u8(rgba_pixel0, v_tbl), weight0);
                jx += 1;
            }

            let store_16 = vqshrun_n_s32::<7>(store);
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            vst1_lane_u32::<0>(dst.as_mut_ptr().cast(), vreinterpret_u32_u8(store_16_8));
        }
    }
}
