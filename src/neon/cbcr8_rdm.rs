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
use crate::neon::utils::{expand8_high_to_14, expand8_to_14};
use std::arch::aarch64::*;

#[must_use]
#[inline(always)]
unsafe fn accumulate_8_horiz(
    store: int16x8_t,
    ptr: *const u8,
    w0: int16x8_t,
    w1: int16x8_t,
) -> int16x8_t {
    unsafe {
        let pixel_colors = vld1q_u8(ptr);
        let lo = expand8_to_14(vget_low_u8(pixel_colors));
        let hi = expand8_high_to_14(pixel_colors);
        let p = vqrdmlahq_s16(store, lo, w0);
        vqrdmlahq_s16(p, hi, w1)
    }
}

#[must_use]
#[inline(always)]
unsafe fn accumulate_4_horiz(store: int16x8_t, ptr: *const u8, weights: int16x8_t) -> int16x8_t {
    unsafe {
        let pixel_colors = vld1_u8(ptr);
        let lo = expand8_to_14(pixel_colors);
        vqrdmlahq_s16(store, lo, weights)
    }
}

#[must_use]
#[inline(always)]
unsafe fn accumulate_1_horiz(store: int16x8_t, ptr: *const u8, weights: int16x8_t) -> int16x8_t {
    unsafe {
        let pixel_colors =
            vreinterpret_u8_u16(vld1_lane_u16::<0>(ptr as *const u16, vdup_n_u16(0)));
        let lo = expand8_to_14(pixel_colors);
        vqrdmlahq_s16(store, lo, weights)
    }
}

#[inline(always)]
unsafe fn store_cbcr<const PRECISION: i32>(
    ptr: *mut u8,
    store: int16x8_t,
    reduction_shuffle: uint8x8_t,
) {
    unsafe {
        let m0 = vadd_s16(vget_low_s16(store), vget_high_s16(store));
        let m_shuf = vreinterpret_s16_u8(vtbl1_u8(vreinterpret_u8_s16(m0), reduction_shuffle));
        let m1 = vadd_s16(m0, m_shuf);
        let m0 = vqshrun_n_s16::<PRECISION>(vcombine_s16(m1, m1));
        let v0 = vreinterpret_u16_u8(m0);
        vst1_lane_u16::<0>(ptr as *mut u16, v0);
    }
}

pub(crate) fn convolve_horizontal_cbcr_neon_rows_rdm_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_cbcr_neon_rows_4_u8_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[target_feature(enable = "rdm")]
unsafe fn convolve_horizontal_cbcr_neon_rows_4_u8_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        const CN: usize = 2;

        let iter_row0 = row0_ref.chunks_exact_mut(CN);
        let iter_row1 = row1_ref.chunks_exact_mut(CN);
        let iter_row2 = row2_ref.chunks_exact_mut(CN);
        let iter_row3 = row3_ref.chunks_exact_mut(CN);

        const PRECISION: i32 = 6;
        const ROUNDING_CONST: i16 = 1 << (PRECISION - 1);
        let base_val = {
            let j = vdupq_n_s16(0);
            let p = vsetq_lane_s16::<0>(ROUNDING_CONST, j);
            vsetq_lane_s16::<1>(ROUNDING_CONST, p)
        };

        let weights_distribute_table: [u8; 16] = [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7];
        let weights_shuffle = vld1q_u8(weights_distribute_table.as_ptr());
        let weights_distribute_table1: [u8; 16] =
            [8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 14, 15];
        let weights_shuffle1 = vld1q_u8(weights_distribute_table1.as_ptr());
        let reduction_table: [u8; 8] = [4, 5, 6, 7, 255, 255, 255, 255];
        let reduction_shuffle = vld1_u8(reduction_table.as_ptr());

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
            let mut store0 = base_val;
            let mut store1 = base_val;
            let mut store2 = base_val;
            let mut store3 = base_val;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 8 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1q_s16(w_ptr.as_ptr());
                let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(
                    vreinterpretq_u8_s16(weights),
                    weights_shuffle,
                ));
                let w1 = vreinterpretq_s16_u8(vqtbl1q_u8(
                    vreinterpretq_u8_s16(weights),
                    weights_shuffle1,
                ));
                let bounds_start = (bounds.start + jx) * CN;

                let src_ptr = src0.get_unchecked(bounds_start..);
                store0 = accumulate_8_horiz(store0, src_ptr.as_ptr(), w0, w1);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                store1 = accumulate_8_horiz(store1, src_ptr1.as_ptr(), w0, w1);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                store2 = accumulate_8_horiz(store2, src_ptr2.as_ptr(), w0, w1);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                store3 = accumulate_8_horiz(store3, src_ptr3.as_ptr(), w0, w1);

                jx += 8;
            }

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vreinterpretq_s16_u8(vqtbl1q_u8(
                    vreinterpretq_u8_s16(vcombine_s16(vld1_s16(w_ptr.as_ptr()), vdup_n_s16(0))),
                    weights_shuffle,
                ));
                let bounds_start = (bounds.start + jx) * CN;

                let src_ptr = src0.get_unchecked(bounds_start..);
                store0 = accumulate_4_horiz(store0, src_ptr.as_ptr(), weights);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                store1 = accumulate_4_horiz(store1, src_ptr1.as_ptr(), weights);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                store2 = accumulate_4_horiz(store2, src_ptr2.as_ptr(), weights);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                store3 = accumulate_4_horiz(store3, src_ptr3.as_ptr(), weights);

                jx += 4;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight = vreinterpretq_s16_u8(vqtbl1q_u8(
                    vreinterpretq_u8_s16(vcombine_s16(
                        vld1_lane_s16::<0>(w_ptr.as_ptr(), vdup_n_s16(0)),
                        vdup_n_s16(0),
                    )),
                    weights_shuffle,
                ));
                let bounds_start = (bounds.start + jx) * CN;

                let src_ptr = src0.get_unchecked(bounds_start..);
                store0 = accumulate_1_horiz(store0, src_ptr.as_ptr(), weight);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                store1 = accumulate_1_horiz(store1, src_ptr1.as_ptr(), weight);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                store2 = accumulate_1_horiz(store2, src_ptr2.as_ptr(), weight);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                store3 = accumulate_1_horiz(store3, src_ptr3.as_ptr(), weight);

                jx += 1;
            }

            store_cbcr::<PRECISION>(chunk0.as_mut_ptr(), store0, reduction_shuffle);
            store_cbcr::<PRECISION>(chunk1.as_mut_ptr(), store1, reduction_shuffle);
            store_cbcr::<PRECISION>(chunk2.as_mut_ptr(), store2, reduction_shuffle);
            store_cbcr::<PRECISION>(chunk3.as_mut_ptr(), store3, reduction_shuffle);
        }
    }
}

pub fn convolve_horizontal_cbcr_neon_rdm_row(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_cbcr_neon_rdm_row_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "rdm")]
unsafe fn convolve_horizontal_cbcr_neon_rdm_row_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CN: usize = 2;
        const PRECISION: i32 = 6;
        const ROUNDING_CONST: i16 = 1 << (PRECISION - 1);

        let base_val = {
            let j = vdupq_n_s16(0);
            let p = vsetq_lane_s16::<0>(ROUNDING_CONST, j);
            vsetq_lane_s16::<1>(ROUNDING_CONST, p)
        };

        let weights_distribute_table: [u8; 16] = [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7];
        let weights_shuffle = vld1q_u8(weights_distribute_table.as_ptr());
        let weights_distribute_table1: [u8; 16] =
            [8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 14, 15];
        let weights_shuffle1 = vld1q_u8(weights_distribute_table1.as_ptr());
        let reduction_table: [u8; 8] = [4, 5, 6, 7, 255, 255, 255, 255];
        let reduction_shuffle = vld1_u8(reduction_table.as_ptr());

        for ((dst, bounds), weights) in dst
            .chunks_exact_mut(2)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let bounds_size = bounds.size;

            let mut jx = 0usize;
            let mut store = base_val;

            while jx + 8 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1q_s16(w_ptr.as_ptr());
                let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(
                    vreinterpretq_u8_s16(weights),
                    weights_shuffle,
                ));
                let w1 = vreinterpretq_s16_u8(vqtbl1q_u8(
                    vreinterpretq_u8_s16(weights),
                    weights_shuffle1,
                ));
                let bounds_start = (bounds.start + jx) * CN;

                let src_ptr = src.get_unchecked(bounds_start..).as_ptr();
                store = accumulate_8_horiz(store, src_ptr, w0, w1);

                jx += 8;
            }

            while jx + 4 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vreinterpretq_s16_u8(vqtbl1q_u8(
                    vreinterpretq_u8_s16(vcombine_s16(vld1_s16(w_ptr.as_ptr()), vdup_n_s16(0))),
                    weights_shuffle,
                ));
                let bounds_start = (bounds.start + jx) * CN;

                let src_ptr = src.get_unchecked(bounds_start..).as_ptr();
                store = accumulate_4_horiz(store, src_ptr, weights);

                jx += 4;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight = vreinterpretq_s16_u8(vqtbl1q_u8(
                    vreinterpretq_u8_s16(vcombine_s16(
                        vld1_lane_s16::<0>(w_ptr.as_ptr(), vdup_n_s16(0)),
                        vdup_n_s16(0),
                    )),
                    weights_shuffle,
                ));
                let bounds_start = (bounds.start + jx) * CN;
                let src_ptr = src.get_unchecked(bounds_start..).as_ptr();
                store = accumulate_1_horiz(store, src_ptr, weight);
                jx += 1;
            }

            store_cbcr::<PRECISION>(dst.as_mut_ptr(), store, reduction_shuffle);
        }
    }
}
