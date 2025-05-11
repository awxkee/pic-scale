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

#[inline]
unsafe fn ld1(src: &[u8]) -> uint8x16_t {
    unsafe {
        vreinterpretq_u8_u16(vld1q_lane_u16::<0>(
            src.as_ptr() as *const u16,
            vdupq_n_u16(0),
        ))
    }
}

#[inline]
unsafe fn ld2(src: &[u8]) -> uint8x16_t {
    unsafe {
        vreinterpretq_u8_u32(vld1q_lane_u32::<0>(
            src.as_ptr() as *const u32,
            vdupq_n_u32(0),
        ))
    }
}

#[inline]
unsafe fn ld4(src: &[u8]) -> uint8x16_t {
    unsafe { vcombine_u8(vld1_u8(src.as_ptr()), vdup_n_u8(0)) }
}

#[inline(always)]
unsafe fn store_cbcr(ptr: &mut [u8], store: int32x2_t) {
    unsafe {
        let m0 = vqshrun_n_s32::<7>(vcombine_s32(store, vdup_n_s32(0)));
        let v0 = vreinterpret_u16_u8(vqmovn_u16(vcombine_u16(m0, m0)));
        vst1_lane_u16::<0>(ptr.as_mut_ptr() as *mut u16, v0);
    }
}

pub(crate) fn convolve_horizontal_cbcr_neon_rows_dot_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_cbcr_neon_rows_4_u8_impl_dot(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve_horizontal_cbcr_neon_rows_4_u8_impl_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
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

        static ST: [i32; 2] = [1 << 6, 1 << 6];
        let base_val = vld1_s32(ST.as_ptr());

        let tbl: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let v_tbl = vld1_u8(tbl.as_ptr());
        let weights_tbl: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let v_weights = vld1q_u8(weights_tbl.as_ptr());

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

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s32(vld1q_lane_s32::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s32(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let bounds_start = bounds.start + jx;

                let src_ptr0 = src0.get_unchecked(bounds_start * CN..);
                let src_ptr1 = src1.get_unchecked(bounds_start * CN..);
                let src_ptr2 = src2.get_unchecked(bounds_start * CN..);
                let src_ptr3 = src3.get_unchecked(bounds_start * CN..);

                store0 = vusdot_s32(
                    store0,
                    vqtbl1_u8(ld4(src_ptr0), v_tbl),
                    vget_low_s8(v_weight),
                );
                store1 = vusdot_s32(
                    store1,
                    vqtbl1_u8(ld4(src_ptr1), v_tbl),
                    vget_low_s8(v_weight),
                );
                store2 = vusdot_s32(
                    store2,
                    vqtbl1_u8(ld4(src_ptr2), v_tbl),
                    vget_low_s8(v_weight),
                );
                store3 = vusdot_s32(
                    store3,
                    vqtbl1_u8(ld4(src_ptr3), v_tbl),
                    vget_low_s8(v_weight),
                );

                jx += 4;
            }

            while jx + 2 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s16(vld1q_lane_s16::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s16(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let bounds_start = bounds.start + jx;

                let src_ptr0 = src0.get_unchecked(bounds_start * CN..);
                let src_ptr1 = src1.get_unchecked(bounds_start * CN..);
                let src_ptr2 = src2.get_unchecked(bounds_start * CN..);
                let src_ptr3 = src3.get_unchecked(bounds_start * CN..);

                store0 = vusdot_s32(
                    store0,
                    vqtbl1_u8(ld2(src_ptr0), v_tbl),
                    vget_low_s8(v_weight),
                );
                store1 = vusdot_s32(
                    store1,
                    vqtbl1_u8(ld2(src_ptr1), v_tbl),
                    vget_low_s8(v_weight),
                );
                store2 = vusdot_s32(
                    store2,
                    vqtbl1_u8(ld2(src_ptr2), v_tbl),
                    vget_low_s8(v_weight),
                );
                store3 = vusdot_s32(
                    store3,
                    vqtbl1_u8(ld2(src_ptr3), v_tbl),
                    vget_low_s8(v_weight),
                );

                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1_dup_s8(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;

                let src_ptr0 = src0.get_unchecked(bounds_start * CN..);
                let src_ptr1 = src1.get_unchecked(bounds_start * CN..);
                let src_ptr2 = src2.get_unchecked(bounds_start * CN..);
                let src_ptr3 = src3.get_unchecked(bounds_start * CN..);

                store0 = vusdot_s32(store0, vqtbl1_u8(ld1(src_ptr0), v_tbl), w0);
                store1 = vusdot_s32(store1, vqtbl1_u8(ld1(src_ptr1), v_tbl), w0);
                store2 = vusdot_s32(store2, vqtbl1_u8(ld1(src_ptr2), v_tbl), w0);
                store3 = vusdot_s32(store3, vqtbl1_u8(ld1(src_ptr3), v_tbl), w0);

                jx += 1;
            }

            store_cbcr(chunk0, store0);
            store_cbcr(chunk1, store1);
            store_cbcr(chunk2, store2);
            store_cbcr(chunk3, store3);
        }
    }
}

pub fn convolve_horizontal_cbcr_neon_dot_row(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_cbcr_neon_rdm_row_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve_horizontal_cbcr_neon_rdm_row_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        const CN: usize = 2;

        static ST: [i32; 2] = [1 << 6, 1 << 6];
        let base_val = vld1_s32(ST.as_ptr());

        let tbl: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let v_tbl = vld1_u8(tbl.as_ptr());
        let weights_tbl: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let v_weights = vld1q_u8(weights_tbl.as_ptr());

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

            while jx + 4 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s32(vld1q_lane_s32::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s32(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let bounds_start = bounds.start + jx;

                let src_ptr = src.get_unchecked(bounds_start * CN..);
                store = vusdot_s32(store, vqtbl1_u8(ld4(src_ptr), v_tbl), vget_low_s8(v_weight));

                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s16(vld1q_lane_s16::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s16(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let bounds_start = bounds.start + jx;

                let src_ptr = src.get_unchecked(bounds_start * CN..);
                store = vusdot_s32(store, vqtbl1_u8(ld2(src_ptr), v_tbl), vget_low_s8(v_weight));

                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1_dup_s8(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                let src_ptr = src.get_unchecked(bounds_start * CN..);
                store = vusdot_s32(store, vqtbl1_u8(ld1(src_ptr), v_tbl), w0);
                jx += 1;
            }

            store_cbcr(dst, store);
        }
    }
}
