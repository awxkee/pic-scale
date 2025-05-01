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
use crate::neon::utils::{expand8_high_to_14, expand8_to_14, load_4b_as_u8x8, xvld1q_u8_x2};
use std::arch::aarch64::*;

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_8_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    w0: int16x8_t,
    w1: int16x8_t,
    w2: int16x8_t,
    w3: int16x8_t,
    store: int16x8_t,
) -> int16x8_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = xvld1q_u8_x2(src_ptr.as_ptr());

    let hi0 = expand8_high_to_14(rgba_pixel.0);
    let lo0 = expand8_to_14(vget_low_u8(rgba_pixel.0));
    let hi1 = expand8_high_to_14(rgba_pixel.1);
    let lo1 = expand8_to_14(vget_low_u8(rgba_pixel.1));

    let mut p = vqrdmlahq_s16(store, lo0, w0);
    p = vqrdmlahq_s16(p, hi0, w1);
    p = vqrdmlahq_s16(p, lo1, w2);
    vqrdmlahq_s16(p, hi1, w3)
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_2_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    weights: int16x8_t,
    store: int16x8_t,
) -> int16x8_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgb_pixel = vld1_u8(src_ptr.as_ptr());
    let wide = expand8_to_14(rgb_pixel);

    vqrdmlahq_s16(store, wide, weights)
}

#[inline(always)]
unsafe fn conv_horiz_rgba_4_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    w0: int16x8_t,
    w1: int16x8_t,
    store: int16x8_t,
) -> int16x8_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = vld1q_u8(src_ptr.as_ptr());

    let hi = expand8_high_to_14(rgba_pixel);
    let lo = expand8_to_14(vget_low_u8(rgba_pixel));

    let p = vqrdmlahq_s16(store, lo, w0);
    vqrdmlahq_s16(p, hi, w1)
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_1_u8_i16<const SCALE: i32>(
    start_x: usize,
    src: &[u8],
    w0: int16x4_t,
    store: int16x4_t,
) -> int16x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel = expand8_to_14(load_4b_as_u8x8(src_ptr.as_ptr()));
    vqrdmlah_s16(store, vget_low_s16(rgba_pixel), w0)
}

/// Checking NEON `rdm` availability is required before a call.
///
/// RDM feature has slightly lower precision and won't work really well on huge kernel which
/// edges fades out fast. Therefore, it would be reasonable to avoid using feature for huge downscaling.
///
/// # Safety
/// - Check `rdm` availability before the call.
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
///
/// # Safety
/// - Check `rdm` availability before the call.
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

    let weights_distribute: [u8; 16] = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3];
    let v_w_distribute0 = vld1q_u8(weights_distribute.as_ptr());
    let weights_distribute1: [u8; 16] = [4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7];
    let v_w_distribute1 = vld1q_u8(weights_distribute1.as_ptr());
    let weights_distribute2: [u8; 16] = [8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11];
    let v_w_distribute2 = vld1q_u8(weights_distribute2.as_ptr());
    let weights_distribute3: [u8; 16] = [
        12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    let v_w_distribute3 = vld1q_u8(weights_distribute3.as_ptr());

    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
    let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
    let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
    let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

    let initial_val = vcombine_s16(vdup_n_s16(ROUNDING), vdup_n_s16(0));

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

        let mut store_0 = initial_val;
        let mut store_1 = initial_val;
        let mut store_2 = initial_val;
        let mut store_3 = initial_val;

        let src0 = src;
        let src1 = src0.get_unchecked(src_stride..);
        let src2 = src1.get_unchecked(src_stride..);
        let src3 = src2.get_unchecked(src_stride..);

        while jx + 8 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..);
            let weights_set = vld1q_s16(w_ptr.as_ptr());

            let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(weights_set),
                v_w_distribute0,
            ));
            let w1 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(weights_set),
                v_w_distribute1,
            ));
            let w2 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(weights_set),
                v_w_distribute2,
            ));
            let w3 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(weights_set),
                v_w_distribute3,
            ));

            store_0 =
                conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src0, w0, w1, w2, w3, store_0);
            store_1 =
                conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src1, w0, w1, w2, w3, store_1);
            store_2 =
                conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src2, w0, w1, w2, w3, store_2);
            store_3 =
                conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src3, w0, w1, w2, w3, store_3);
            jx += 8;
        }

        while jx + 4 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..);
            let weights = vld1_s16(w_ptr.as_ptr());

            let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(vcombine_s16(weights, vdup_n_s16(0))),
                v_w_distribute0,
            ));
            let w1 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(vcombine_s16(weights, vdup_n_s16(0))),
                v_w_distribute1,
            ));

            store_0 = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src0, w0, w1, store_0);
            store_1 = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src1, w0, w1, store_1);
            store_2 = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src2, w0, w1, store_2);
            store_3 = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src3, w0, w1, store_3);
            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..);
            let bounds_start = bounds.start + jx;
            let v_weight = vreinterpret_s16_s32(vld1_dup_s32(w_ptr.as_ptr() as *const i32));
            let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(vcombine_s16(v_weight, vdup_n_s16(0))),
                v_w_distribute0,
            ));
            store_0 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src0, w0, store_0);
            store_1 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src1, w0, store_1);
            store_2 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src2, w0, store_2);
            store_3 = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src3, w0, store_3);
            jx += 2;
        }

        let mut store_0 = vadd_s16(vget_low_s16(store_0), vget_high_s16(store_0));
        let mut store_1 = vadd_s16(vget_low_s16(store_1), vget_high_s16(store_1));
        let mut store_2 = vadd_s16(vget_low_s16(store_2), vget_high_s16(store_2));
        let mut store_3 = vadd_s16(vget_low_s16(store_3), vget_high_s16(store_3));

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..);
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

        vst1_lane_u32::<0>(
            chunk0.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8_0),
        );
        vst1_lane_u32::<0>(
            chunk1.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8_1),
        );
        vst1_lane_u32::<0>(
            chunk2.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8_2),
        );
        vst1_lane_u32::<0>(
            chunk3.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8),
        );
    }
}

/// Checking NEON `rdm` availability is required before a call.
///
/// RDM feature has slightly lower precision and won't work really well on huge kernel which
/// edges fades out fast. Therefore, it would be reasonable to avoid using feature for huge downscaling.
///
/// # Safety
/// - Check `rdm` availability before the call.
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

    let weights_distribute: [u8; 16] = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3];
    let v_w_distribute0 = vld1q_u8(weights_distribute.as_ptr());
    let weights_distribute1: [u8; 16] = [4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7];
    let v_w_distribute1 = vld1q_u8(weights_distribute1.as_ptr());
    let weights_distribute2: [u8; 16] = [8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11];
    let v_w_distribute2 = vld1q_u8(weights_distribute2.as_ptr());
    let weights_distribute3: [u8; 16] = [
        12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    let v_w_distribute3 = vld1q_u8(weights_distribute3.as_ptr());

    let initial_val = vcombine_s16(vdup_n_s16(ROUNDING), vdup_n_s16(0));

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
        let mut store = initial_val;

        while jx + 8 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..);
            let weights_set = vld1q_s16(w_ptr.as_ptr());

            let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(weights_set),
                v_w_distribute0,
            ));
            let w1 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(weights_set),
                v_w_distribute1,
            ));
            let w2 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(weights_set),
                v_w_distribute2,
            ));
            let w3 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(weights_set),
                v_w_distribute3,
            ));

            store = conv_horiz_rgba_8_u8_i16::<SCALE>(bounds_start, src, w0, w1, w2, w3, store);
            jx += 8;
        }

        while jx + 4 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..);
            let weights = vld1_s16(w_ptr.as_ptr());
            let bounds_start = bounds.start + jx;

            let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(vcombine_s16(weights, vdup_n_s16(0))),
                v_w_distribute0,
            ));
            let w1 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(vcombine_s16(weights, vdup_n_s16(0))),
                v_w_distribute1,
            ));

            store = conv_horiz_rgba_4_u8_i16::<SCALE>(bounds_start, src, w0, w1, store);
            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..);
            let bounds_start = bounds.start + jx;
            let v_weight = vreinterpret_s16_s32(vld1_dup_s32(w_ptr.as_ptr() as *const i32));
            let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(vcombine_s16(v_weight, vdup_n_s16(0))),
                v_w_distribute0,
            ));
            store = conv_horiz_rgba_2_u8_i16::<SCALE>(bounds_start, src, w0, store);
            jx += 2;
        }

        let mut store = vadd_s16(vget_low_s16(store), vget_high_s16(store));

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..);
            let weight0 = vld1_dup_s16(w_ptr.as_ptr());
            let bounds_start = bounds.start + jx;
            store = conv_horiz_rgba_1_u8_i16::<SCALE>(bounds_start, src, weight0, store);
            jx += 1;
        }

        let store_16 = vshr_n_s16::<SCALE>(store);

        let store_16_8 = vqmovun_s16(vcombine_s16(store_16, store_16));

        vst1_lane_u32::<0>(
            dst.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8),
        );
    }
}
