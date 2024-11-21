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
    vextract_ar30, vld1_ar30_s16, vunzip_4_ar30_separate, vunzips_4_ar30_separate,
};
use std::arch::aarch64::*;

#[inline]
unsafe fn conv_horiz_rgba_1_u8_i16<
    const SCALE: i32,
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    start_x: usize,
    src: &[u32],
    w0: int16x4_t,
    store: int16x4_t,
) -> int16x4_t {
    let src_ptr = src.get_unchecked(start_x..);
    let ld = vld1_ar30_s16::<AR_TYPE, AR_ORDER>(src_ptr);
    let rgba_pixel = vshl_n_s16::<SCALE>(ld);
    vqrdmlah_s16(store, rgba_pixel, w0)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u8_i16<
    const SCALE: i32,
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    start_x: usize,
    src: &[u32],
    set1: (int16x4_t, int16x4_t, int16x4_t, int16x4_t),
    set2: (int16x4_t, int16x4_t, int16x4_t, int16x4_t),
    store: int16x4_t,
) -> int16x4_t {
    let src_ptr = src.get_unchecked(start_x..);

    let rgba_pixel = vunzip_4_ar30_separate::<AR_TYPE, AR_ORDER>(vld1q_u32_x2(src_ptr.as_ptr()));

    let hi0 = vshlq_n_s16::<SCALE>(rgba_pixel.1);
    let lo0 = vshlq_n_s16::<SCALE>(rgba_pixel.0);
    let hi1 = vshlq_n_s16::<SCALE>(rgba_pixel.3);
    let lo1 = vshlq_n_s16::<SCALE>(rgba_pixel.2);

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
unsafe fn conv_horiz_rgba_4_u8_i16<
    const SCALE: i32,
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    start_x: usize,
    src: &[u32],
    w0: int16x4_t,
    w1: int16x4_t,
    w2: int16x4_t,
    w3: int16x4_t,
    store: int16x4_t,
) -> int16x4_t {
    let src_ptr = src.get_unchecked(start_x..);

    let rgba_pixel = vunzips_4_ar30_separate::<AR_TYPE, AR_ORDER>(vld1q_u32(src_ptr.as_ptr()));

    let hi = vshlq_n_s16::<SCALE>(rgba_pixel.1);
    let lo = vshlq_n_s16::<SCALE>(rgba_pixel.0);

    let hi_v = vqrdmulhq_s16(hi, vcombine_s16(w2, w3));
    let product = vqrdmlahq_s16(hi_v, lo, vcombine_s16(w0, w1));

    vadd_s16(
        vadd_s16(store, vget_low_s16(product)),
        vget_high_s16(product),
    )
}

pub(crate) fn neon_convolve_horizontal_rgba_rows_4_ar30<
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    src: &[u32],
    src_stride: usize,
    dst: &mut [u32],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const SCALE: i32 = 4;
        const ROUNDING: i16 = 1 << (SCALE - 1);
        let zeros = vdup_n_s16(0i16);
        const ALPHA_ROUNDING: i16 = 1 << (SCALE as i16 + 7);
        let init = vld1_s16([ROUNDING, ROUNDING, ROUNDING, ALPHA_ROUNDING].as_ptr());

        let v_cut_off = vld1_s16([1023, 1023, 1023, 3].as_ptr());

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.iter_mut();
        let iter_row1 = row1_ref.iter_mut();
        let iter_row2 = row2_ref.iter_mut();
        let iter_row3 = row3_ref.iter_mut();

        let v_shl_back = vld1_s16(
            [
                -SCALE as i16,
                -SCALE as i16,
                -SCALE as i16,
                -(SCALE as i16 + 8),
            ]
            .as_ptr(),
        );

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
                store_0 = conv_horiz_rgba_8_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src0,
                    set1,
                    set2,
                    store_0,
                );
                store_1 = conv_horiz_rgba_8_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src1,
                    set1,
                    set2,
                    store_1,
                );
                store_2 = conv_horiz_rgba_8_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src2,
                    set1,
                    set2,
                    store_2,
                );
                store_3 = conv_horiz_rgba_8_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src3,
                    set1,
                    set2,
                    store_3,
                );
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
                store_0 = conv_horiz_rgba_4_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src0,
                    w0,
                    w1,
                    w2,
                    w3,
                    store_0,
                );
                store_1 = conv_horiz_rgba_4_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src1,
                    w0,
                    w1,
                    w2,
                    w3,
                    store_1,
                );
                store_2 = conv_horiz_rgba_4_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src2,
                    w0,
                    w1,
                    w2,
                    w3,
                    store_2,
                );
                store_3 = conv_horiz_rgba_4_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src3,
                    w0,
                    w1,
                    w2,
                    w3,
                    store_3,
                );
                jx += 4;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_1_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src0,
                    weight0,
                    store_0,
                );
                store_1 = conv_horiz_rgba_1_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src1,
                    weight0,
                    store_1,
                );
                store_2 = conv_horiz_rgba_1_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src2,
                    weight0,
                    store_2,
                );
                store_3 = conv_horiz_rgba_1_u8_i16::<SCALE, AR_TYPE, AR_ORDER>(
                    bounds_start,
                    src3,
                    weight0,
                    store_3,
                );
                jx += 1;
            }

            let store_16_0 = vreinterpret_u16_s16(vmin_s16(
                vshl_s16(vmax_s16(store_0, zeros), v_shl_back),
                v_cut_off,
            ));
            let store_16_1 = vreinterpret_u16_s16(vmin_s16(
                vshl_s16(vmax_s16(store_1, zeros), v_shl_back),
                v_cut_off,
            ));
            let store_16_2 = vreinterpret_u16_s16(vmin_s16(
                vshl_s16(vmax_s16(store_2, zeros), v_shl_back),
                v_cut_off,
            ));
            let store_16_3 = vreinterpret_u16_s16(vmin_s16(
                vshl_s16(vmax_s16(store_3, zeros), v_shl_back),
                v_cut_off,
            ));

            let packed0 = vextract_ar30::<AR_TYPE, AR_ORDER>(store_16_0);
            *chunk0 = packed0;
            let packed1 = vextract_ar30::<AR_TYPE, AR_ORDER>(store_16_1);
            *chunk1 = packed1;
            let packed2 = vextract_ar30::<AR_TYPE, AR_ORDER>(store_16_2);
            *chunk2 = packed2;
            let packed3 = vextract_ar30::<AR_TYPE, AR_ORDER>(store_16_3);
            *chunk3 = packed3;
        }
    }
}
