/*
 * Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::neon::utils::{xvld1q_u16_x2, xvld1q_u16_x4};
use std::arch::aarch64::*;

#[must_use]
#[inline]
#[target_feature(enable = "neon")]
fn conv_horiz_rgba_1_u16_f32(
    start_x: usize,
    src: &[u16],
    w0: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let rgba_pixel = vld1_u16(src_ptr.as_ptr());
        let lo = vcvtq_f32_u32(vmovl_u16(rgba_pixel));
        vfmaq_f32(store, lo, w0)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "neon")]
fn conv_horiz_rgba_2_u16_f32(
    start_x: usize,
    src: &[u16],
    w0: float32x4_t,
    w1: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let rgb_pixel = vld1q_u16(src_ptr.as_ptr());
        let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(rgb_pixel)));
        let hi = vcvtq_f32_u32(vmovl_high_u16(rgb_pixel));
        let acc = vfmaq_f32(store, lo, w0);
        vfmaq_f32(acc, hi, w1)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "neon")]
fn conv_horiz_rgba_4_u16_f32(
    start_x: usize,
    src: &[u16],
    weights: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let rgba_pixel = xvld1q_u16_x2(src_ptr.as_ptr());

        let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(rgba_pixel.0)));
        let hi = vcvtq_f32_u32(vmovl_high_u16(rgba_pixel.0));
        let lo1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(rgba_pixel.1)));
        let hi1 = vcvtq_f32_u32(vmovl_high_u16(rgba_pixel.1));

        let acc = vfmaq_laneq_f32::<0>(store, lo, weights);
        let acc = vfmaq_laneq_f32::<1>(acc, hi, weights);
        let acc = vfmaq_laneq_f32::<2>(acc, lo1, weights);
        vfmaq_laneq_f32::<3>(acc, hi1, weights)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn conv_horiz_rgba_8_u16_f32(
    start_x: usize,
    src: &[u16],
    weights: (float32x4_t, float32x4_t),
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let rgba_pixel = xvld1q_u16_x4(src_ptr.as_ptr());

        let lo0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(rgba_pixel.0)));
        let hi0 = vcvtq_f32_u32(vmovl_high_u16(rgba_pixel.0));
        let lo1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(rgba_pixel.1)));
        let hi1 = vcvtq_f32_u32(vmovl_high_u16(rgba_pixel.1));
        let lo2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(rgba_pixel.2)));
        let hi2 = vcvtq_f32_u32(vmovl_high_u16(rgba_pixel.2));
        let lo3 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(rgba_pixel.3)));
        let hi3 = vcvtq_f32_u32(vmovl_high_u16(rgba_pixel.3));

        let acc = vfmaq_laneq_f32::<0>(store, lo0, weights.0);
        let acc = vfmaq_laneq_f32::<1>(acc, hi0, weights.0);
        let acc = vfmaq_laneq_f32::<2>(acc, lo1, weights.0);
        let acc = vfmaq_laneq_f32::<3>(acc, hi1, weights.0);
        let acc = vfmaq_laneq_f32::<0>(acc, lo2, weights.1);
        let acc = vfmaq_laneq_f32::<1>(acc, hi2, weights.1);
        let acc = vfmaq_laneq_f32::<2>(acc, lo3, weights.1);
        vfmaq_laneq_f32::<3>(acc, hi3, weights.1)
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_rows_4_f32_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgba_neon_rows_4_f32_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        )
    }
}

#[target_feature(enable = "neon")]
fn convolve_horizontal_rgba_neon_rows_4_f32_impl(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    const CN: usize = 4;
    let zeros = vdupq_n_f32(0f32);

    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
    let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
    let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
    let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

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
        unsafe {
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            let bounds_size = bounds.size;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = (
                    vld1q_f32(w_ptr.as_ptr()),
                    vld1q_f32(w_ptr.get_unchecked(4..).as_ptr()),
                );
                store_0 = conv_horiz_rgba_8_u16_f32(bounds_start, src0, weights_set, store_0);
                store_1 = conv_horiz_rgba_8_u16_f32(bounds_start, src1, weights_set, store_1);
                store_2 = conv_horiz_rgba_8_u16_f32(bounds_start, src2, weights_set, store_2);
                store_3 = conv_horiz_rgba_8_u16_f32(bounds_start, src3, weights_set, store_3);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w = vld1q_f32(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_4_u16_f32(bounds_start, src0, w, store_0);
                store_1 = conv_horiz_rgba_4_u16_f32(bounds_start, src1, w, store_1);
                store_2 = conv_horiz_rgba_4_u16_f32(bounds_start, src2, w, store_2);
                store_3 = conv_horiz_rgba_4_u16_f32(bounds_start, src3, w, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1q_dup_f32(w_ptr.as_ptr());
                let w1 = vld1q_dup_f32(w_ptr.get_unchecked(1..).as_ptr());
                store_0 = conv_horiz_rgba_2_u16_f32(bounds_start, src0, w0, w1, store_0);
                store_1 = conv_horiz_rgba_2_u16_f32(bounds_start, src1, w0, w1, store_1);
                store_2 = conv_horiz_rgba_2_u16_f32(bounds_start, src2, w0, w1, store_2);
                store_3 = conv_horiz_rgba_2_u16_f32(bounds_start, src3, w0, w1, store_3);
                jx += 2;
            }

            while jx < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1q_dup_f32(w_ptr.as_ptr());
                store_0 = conv_horiz_rgba_1_u16_f32(bounds_start, src0, w0, store_0);
                store_1 = conv_horiz_rgba_1_u16_f32(bounds_start, src1, w0, store_1);
                store_2 = conv_horiz_rgba_1_u16_f32(bounds_start, src2, w0, store_2);
                store_3 = conv_horiz_rgba_1_u16_f32(bounds_start, src3, w0, store_3);
                jx += 1;
            }

            let v_max_u16 = vdup_n_u16(((1u32 << bit_depth) - 1) as u16);

            let j0 = vqmovn_u32(vcvtaq_u32_f32(store_0));
            let j1 = vqmovn_u32(vcvtaq_u32_f32(store_1));
            let j2 = vqmovn_u32(vcvtaq_u32_f32(store_2));
            let j3 = vqmovn_u32(vcvtaq_u32_f32(store_3));

            vst1_u16(chunk0.as_mut_ptr(), vmin_u16(j0, v_max_u16));
            vst1_u16(chunk1.as_mut_ptr(), vmin_u16(j1, v_max_u16));
            vst1_u16(chunk2.as_mut_ptr(), vmin_u16(j2, v_max_u16));
            vst1_u16(chunk3.as_mut_ptr(), vmin_u16(j3, v_max_u16));
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_f32_u16_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgba_neon_f32_u16_impl(src, dst, filter_weights, bit_depth);
    }
}

#[target_feature(enable = "neon")]
fn convolve_horizontal_rgba_neon_f32_u16_impl(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    const CN: usize = 4;
    let v_max_u16 = vdup_n_u16(((1u32 << bit_depth) - 1) as u16);
    let zeros = vdupq_n_f32(0f32);

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
        unsafe {
            let bounds_size = bounds.size;
            let mut jx = 0usize;
            let mut store = zeros;

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = (
                    vld1q_f32(w_ptr.as_ptr()),
                    vld1q_f32(w_ptr.get_unchecked(4..).as_ptr()),
                );
                store = conv_horiz_rgba_8_u16_f32(bounds_start, src, weights_set, store);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w = vld1q_f32(w_ptr.as_ptr());
                store = conv_horiz_rgba_4_u16_f32(bounds_start, src, w, store);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1q_dup_f32(w_ptr.as_ptr());
                let w1 = vld1q_dup_f32(w_ptr.get_unchecked(1..).as_ptr());
                store = conv_horiz_rgba_2_u16_f32(bounds_start, src, w0, w1, store);
                jx += 2;
            }

            while jx < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1q_dup_f32(w_ptr.as_ptr());
                store = conv_horiz_rgba_1_u16_f32(bounds_start, src, w0, store);
                jx += 1;
            }

            let as_u16 = vqmovn_u32(vcvtaq_u32_f32(store));
            vst1_u16(dst.as_mut_ptr(), vmin_u16(as_u16, v_max_u16));
        }
    }
}
