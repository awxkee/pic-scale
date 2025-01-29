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
use crate::neon::ar30::{
    vextract_ar30, vld1_ar30_s16, vunzip_3_ar30_separate, vunzips_4_ar30_separate,
};
use std::arch::aarch64::*;

#[inline]
unsafe fn conv_horiz_rgba_1_u8_i16<const AR_TYPE: usize, const AR_ORDER: usize>(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    let src_ptr = src.get_unchecked(start_x * 4..);
    let ld = vld1_ar30_s16::<AR_TYPE, AR_ORDER>(src_ptr);
    vqdmlal_lane_s16::<0>(store, ld, w0)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u8_i16<const AR_TYPE: usize, const AR_ORDER: usize>(
    start_x: usize,
    src: &[u8],
    w: int16x8_t,
    store: int32x4_t,
) -> int32x4_t {
    let src_ptr = src.get_unchecked(start_x * 4..);

    let rgba_pixel =
        vunzip_3_ar30_separate::<AR_TYPE, AR_ORDER>(vld1q_u32_x2(src_ptr.as_ptr() as *const _));

    let mut v = vqdmlal_laneq_s16::<0>(store, vget_low_s16(rgba_pixel.0), w);
    v = vqdmlal_high_laneq_s16::<1>(v, rgba_pixel.1, w);
    v = vqdmlal_laneq_s16::<2>(v, vget_low_s16(rgba_pixel.1), w);
    v = vqdmlal_high_laneq_s16::<3>(v, rgba_pixel.1, w);
    v = vqdmlal_laneq_s16::<4>(v, vget_low_s16(rgba_pixel.2), w);
    v = vqdmlal_high_laneq_s16::<5>(v, rgba_pixel.2, w);
    v = vqdmlal_laneq_s16::<6>(v, vget_low_s16(rgba_pixel.3), w);
    vqdmlal_high_laneq_s16::<7>(v, rgba_pixel.3, w)
}

#[inline]
unsafe fn conv_horiz_rgba_4_u8_i16<const AR_TYPE: usize, const AR_ORDER: usize>(
    start_x: usize,
    src: &[u8],
    w: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    let src_ptr = src.get_unchecked(start_x * 4..);

    let rgba_pixel =
        vunzips_4_ar30_separate::<AR_TYPE, AR_ORDER>(vld1q_u32(src_ptr.as_ptr() as *const _));

    let mut v = vqdmlal_lane_s16::<0>(store, vget_low_s16(rgba_pixel.0), w);
    v = vqdmlal_high_lane_s16::<1>(v, rgba_pixel.1, w);
    v = vqdmlal_lane_s16::<2>(v, vget_low_s16(rgba_pixel.1), w);
    vqdmlal_high_lane_s16::<3>(v, rgba_pixel.1, w)
}

pub(crate) fn neon_convolve_horizontal_rgba_rows_4_ar30<
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        neon_convolve_horizontal_rgba_rows_4_impl::<AR_TYPE, AR_ORDER>(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

unsafe fn neon_convolve_horizontal_rgba_rows_4_impl<const AR_TYPE: usize, const AR_ORDER: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const PRECISION: i32 = 16;
        const ROUNDING: i32 = (1 << (PRECISION - 1)) - 1;

        let init = vdupq_n_s32(ROUNDING);

        let v_cut_off = vdup_n_u16(1023);

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.chunks_exact_mut(4);
        let iter_row1 = row1_ref.chunks_exact_mut(4);
        let iter_row2 = row2_ref.chunks_exact_mut(4);
        let iter_row3 = row3_ref.chunks_exact_mut(4);

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
                store_0 = conv_horiz_rgba_8_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src0,
                    weights_set,
                    store_0,
                );
                store_1 = conv_horiz_rgba_8_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src1,
                    weights_set,
                    store_1,
                );
                store_2 = conv_horiz_rgba_8_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src2,
                    weights_set,
                    store_2,
                );
                store_3 = conv_horiz_rgba_8_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src3,
                    weights_set,
                    store_3,
                );
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_4_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src0,
                    weights,
                    store_0,
                );
                store_1 = conv_horiz_rgba_4_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src1,
                    weights,
                    store_1,
                );
                store_2 = conv_horiz_rgba_4_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src2,
                    weights,
                    store_2,
                );
                store_3 = conv_horiz_rgba_4_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src3,
                    weights,
                    store_3,
                );
                jx += 4;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_1_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src0,
                    weight0,
                    store_0,
                );
                store_1 = conv_horiz_rgba_1_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src1,
                    weight0,
                    store_1,
                );
                store_2 = conv_horiz_rgba_1_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src2,
                    weight0,
                    store_2,
                );
                store_3 = conv_horiz_rgba_1_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src3,
                    weight0,
                    store_3,
                );
                jx += 1;
            }

            let store_0 = vqshrun_n_s32::<PRECISION>(store_0);
            let store_1 = vqshrun_n_s32::<PRECISION>(store_1);
            let store_2 = vqshrun_n_s32::<PRECISION>(store_2);
            let store_3 = vqshrun_n_s32::<PRECISION>(store_3);

            let store_16_0 = vmin_u16(store_0, v_cut_off);
            let store_16_1 = vmin_u16(store_1, v_cut_off);
            let store_16_2 = vmin_u16(store_2, v_cut_off);
            let store_16_3 = vmin_u16(store_3, v_cut_off);

            let packed0 = vextract_ar30::<AR_TYPE, AR_ORDER>(store_16_0).to_ne_bytes();
            chunk0[0] = packed0[0];
            chunk0[1] = packed0[1];
            chunk0[2] = packed0[2];
            chunk0[3] = packed0[3];
            let packed1 = vextract_ar30::<AR_TYPE, AR_ORDER>(store_16_1).to_ne_bytes();
            chunk1[0] = packed1[0];
            chunk1[1] = packed1[1];
            chunk1[2] = packed1[2];
            chunk1[3] = packed1[3];
            let packed2 = vextract_ar30::<AR_TYPE, AR_ORDER>(store_16_2).to_ne_bytes();
            chunk2[0] = packed2[0];
            chunk2[1] = packed2[1];
            chunk2[2] = packed2[2];
            chunk2[3] = packed2[3];
            let packed3 = vextract_ar30::<AR_TYPE, AR_ORDER>(store_16_3).to_ne_bytes();
            chunk3[0] = packed3[0];
            chunk3[1] = packed3[1];
            chunk3[2] = packed3[2];
            chunk3[3] = packed3[3];
        }
    }
}

pub(crate) fn neon_convolve_horizontal_rgba_rows_ar30<
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        neon_convolve_horizontal_rgba_row_impl::<AR_TYPE, AR_ORDER>(src, dst, filter_weights);
    }
}

unsafe fn neon_convolve_horizontal_rgba_row_impl<const AR_TYPE: usize, const AR_ORDER: usize>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const PRECISION: i32 = 16;
        const ROUNDING: i32 = (1 << (PRECISION - 1)) - 1;

        let init = vdupq_n_s32(ROUNDING);

        let v_cut_off = vdup_n_u16(1023);

        for ((chunk0, &bounds), weights) in dst
            .chunks_exact_mut(4)
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

            let src0 = src;

            while jx + 8 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..(jx + 8));
                let weights_set = vld1q_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_8_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src0,
                    weights_set,
                    store_0,
                );
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = vld1_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_4_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src0,
                    weights,
                    store_0,
                );
                jx += 4;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_1_u8_i16::<AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src0,
                    weight0,
                    store_0,
                );
                jx += 1;
            }

            let store_0 = vqshrun_n_s32::<PRECISION>(store_0);

            let store_16_0 = vmin_u16(store_0, v_cut_off);

            let packed0 = vextract_ar30::<AR_TYPE, AR_ORDER>(store_16_0).to_ne_bytes();
            chunk0[0] = packed0[0];
            chunk0[1] = packed0[1];
            chunk0[2] = packed0[2];
            chunk0[3] = packed0[3];
        }
    }
}
