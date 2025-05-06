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
use crate::neon::utils::{vxmlal_high_s16, vxmlal_s16, xvld1q_s16_x2};
use crate::support::PRECISION;
use std::arch::aarch64::*;

#[must_use]
#[inline(always)]
unsafe fn accumulate_16_horiz<const D: bool>(
    store: int32x4_t,
    ptr: &[u8],
    weights: int16x8x2_t,
) -> int32x4_t {
    unsafe {
        let pixel_colors = vld1q_u8(ptr.as_ptr());
        let px_high_16 = vreinterpretq_s16_u16(vmovl_high_u8(pixel_colors));
        let px_low_16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pixel_colors)));

        let mut store = vxmlal_high_s16::<D>(store, px_high_16, weights.1);
        store = vxmlal_s16::<D>(store, vget_low_s16(px_high_16), vget_low_s16(weights.1));

        store = vxmlal_high_s16::<D>(store, px_low_16, weights.0);
        store = vxmlal_s16::<D>(store, vget_low_s16(px_low_16), vget_low_s16(weights.0));
        store
    }
}

#[must_use]
#[inline(always)]
unsafe fn accumulate_8_horiz<const D: bool>(
    store: int32x4_t,
    ptr: &[u8],
    weight: int16x8_t,
) -> int32x4_t {
    unsafe {
        let pixel_colors = vld1_u8(ptr.as_ptr());
        let px_16 = vreinterpretq_s16_u16(vmovl_u8(pixel_colors));

        let mut store = vxmlal_high_s16::<D>(store, px_16, weight);
        store = vxmlal_s16::<D>(store, vget_low_s16(px_16), vget_low_s16(weight));
        store
    }
}

#[must_use]
#[inline(always)]
unsafe fn accumulate_4_horiz<const D: bool>(
    store: int32x4_t,
    ptr: &[u8],
    weight: int16x4_t,
) -> int32x4_t {
    unsafe {
        let pixel_colors = vmovl_u8(vreinterpret_u8_u32(vld1_lane_u32::<0>(
            ptr.as_ptr() as *const u32,
            vdup_n_u32(0),
        )));
        let px_16 = vreinterpret_s16_u16(vget_low_u16(pixel_colors));
        vxmlal_s16::<D>(store, px_16, weight)
    }
}

#[must_use]
#[inline(always)]
unsafe fn accumulate_1_horiz<const D: bool>(
    store: int32x4_t,
    ptr: &[u8],
    weight: int16x4_t,
) -> int32x4_t {
    unsafe {
        let pixel_colors = vmovl_u8(vld1_lane_u8::<0>(ptr.as_ptr(), vdup_n_u8(0)));
        let px_16 = vreinterpret_s16_u16(vget_low_u16(pixel_colors));
        vxmlal_s16::<D>(store, px_16, weight)
    }
}

pub(crate) fn convolve_horizontal_plane_neon_rows_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    convolve_horizontal_plane_neon_rows_4_u8_impl::<false, PRECISION>(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
    );
}

pub(crate) fn convolve_horizontal_plane_neon_rows_4_u8_q(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    convolve_horizontal_plane_neon_rows_4_u8_impl::<true, 16>(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
    );
}

fn convolve_horizontal_plane_neon_rows_4_u8_impl<const D: bool, const PRECISION: i32>(
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

        let iter_row0 = row0_ref.iter_mut();
        let iter_row1 = row1_ref.iter_mut();
        let iter_row2 = row2_ref.iter_mut();
        let iter_row3 = row3_ref.iter_mut();

        let rnd_const = (1 << (PRECISION - 1)) - 1;

        let base_val = {
            let j = vdupq_n_s32(0);
            vsetq_lane_s32::<0>(rnd_const, j)
        };

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

            while jx + 16 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = xvld1q_s16_x2(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;

                let src_ptr = src0.get_unchecked(bounds_start..);
                store0 = accumulate_16_horiz::<D>(store0, src_ptr, weights);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                store1 = accumulate_16_horiz::<D>(store1, src_ptr1, weights);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                store2 = accumulate_16_horiz::<D>(store2, src_ptr2, weights);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                store3 = accumulate_16_horiz::<D>(store3, src_ptr3, weights);

                jx += 16;
            }

            while jx + 8 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1q_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;

                let src_ptr = src0.get_unchecked(bounds_start..);
                store0 = accumulate_8_horiz::<D>(store0, src_ptr, weights);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                store1 = accumulate_8_horiz::<D>(store1, src_ptr1, weights);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                store2 = accumulate_8_horiz::<D>(store2, src_ptr2, weights);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                store3 = accumulate_8_horiz::<D>(store3, src_ptr3, weights);

                jx += 8;
            }

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;

                let src_ptr = src0.get_unchecked(bounds_start..);
                store0 = accumulate_4_horiz::<D>(store0, src_ptr, weights);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                store1 = accumulate_4_horiz::<D>(store1, src_ptr1, weights);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                store2 = accumulate_4_horiz::<D>(store2, src_ptr2, weights);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                store3 = accumulate_4_horiz::<D>(store3, src_ptr3, weights);

                jx += 4;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight = vld1_lane_s16::<0>(w_ptr.as_ptr(), vdup_n_s16(0));
                let bounds_start = bounds.start + jx;

                let src_ptr = src0.get_unchecked(bounds_start..);
                store0 = accumulate_1_horiz::<D>(store0, src_ptr, weight);

                let src_ptr1 = src1.get_unchecked(bounds_start..);
                store1 = accumulate_1_horiz::<D>(store1, src_ptr1, weight);

                let src_ptr2 = src2.get_unchecked(bounds_start..);
                store2 = accumulate_1_horiz::<D>(store2, src_ptr2, weight);

                let src_ptr3 = src3.get_unchecked(bounds_start..);
                store3 = accumulate_1_horiz::<D>(store3, src_ptr3, weight);

                jx += 1;
            }

            let sums = vaddvq_s32(vmaxq_s32(vdupq_n_s32(0), store0));
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;
            *chunk0 = value;

            let sums = vaddvq_s32(vmaxq_s32(vdupq_n_s32(0), store1));
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;
            *chunk1 = value;

            let sums = vaddvq_s32(vmaxq_s32(vdupq_n_s32(0), store2));
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;
            *chunk2 = value;

            let sums = vaddvq_s32(vmaxq_s32(vdupq_n_s32(0), store3));
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;
            *chunk3 = value;
        }
    }
}

pub(crate) fn convolve_horizontal_plane_neon_row(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    convolve_horizontal_plane_neon_row_impl::<false, PRECISION>(src, dst, filter_weights);
}

pub(crate) fn convolve_horizontal_plane_neon_row_q(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    convolve_horizontal_plane_neon_row_impl::<true, 16>(src, dst, filter_weights);
}

fn convolve_horizontal_plane_neon_row_impl<const D: bool, const PRECISION: i32>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        let rnd_const = (1 << (PRECISION - 1)) - 1;
        let base_val = {
            let j = vdupq_n_s32(0);
            vsetq_lane_s32::<0>(rnd_const, j)
        };

        for ((dst, bounds), weights) in dst.iter_mut().zip(filter_weights.bounds.iter()).zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        ) {
            let bounds_size = bounds.size;

            let mut jx = 0usize;
            let mut store = base_val;

            while jx + 16 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = xvld1q_s16_x2(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;

                let src_ptr = src.get_unchecked(bounds_start..);
                store = accumulate_16_horiz::<D>(store, src_ptr, weights);

                jx += 16;
            }

            while jx + 8 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1q_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;

                let src_ptr = src.get_unchecked(bounds_start..);
                store = accumulate_8_horiz::<D>(store, src_ptr, weights);

                jx += 8;
            }

            while jx + 4 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;

                let src_ptr = src.get_unchecked(bounds_start..);
                store = accumulate_4_horiz::<D>(store, src_ptr, weights);

                jx += 4;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight = vld1_lane_s16::<0>(w_ptr.as_ptr(), vdup_n_s16(0));
                let bounds_start = bounds.start + jx;
                let src_ptr = src.get_unchecked(bounds_start..);
                store = accumulate_1_horiz::<D>(store, src_ptr, weight);
                jx += 1;
            }

            let sums = vaddvq_s32(vmaxq_s32(vdupq_n_s32(0), store));
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;
            *dst = value;
        }
    }
}
