/*
 * Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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
#[target_feature(enable = "neon")]
fn load_ga_1_f32(start_x: usize, src: &[u16]) -> float32x4_t {
    const CN: usize = 2;
    let src_ptr = unsafe { src.get_unchecked((start_x * CN)..) };
    let raw = unsafe { vld1_lane_u32::<0>(src_ptr.as_ptr().cast(), vdup_n_u32(0)) };
    let u16x4 = vreinterpret_u16_u32(raw); // [cb, cr, 0, 0]
    let u32x4 = vmovl_u16(u16x4);
    vcvtq_f32_u32(u32x4)
}

#[inline]
#[target_feature(enable = "neon")]
fn load_ga_2_f32(start_x: usize, src: &[u16]) -> float32x4_t {
    const CN: usize = 2;
    let src_ptr = unsafe { src.get_unchecked((start_x * CN)..) };
    let raw = unsafe { vld1_u16(src_ptr.as_ptr()) };
    let u32x4 = vmovl_u16(raw);
    vcvtq_f32_u32(u32x4)
}

#[inline]
#[target_feature(enable = "neon")]
fn load_ga_4_f32(start_x: usize, src: &[u16]) -> (float32x4_t, float32x4_t) {
    const CN: usize = 2;
    let src_ptr = unsafe { src.get_unchecked((start_x * CN)..) };
    let raw = unsafe { vld1q_u16(src_ptr.as_ptr()) };
    let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(raw)));
    let hi = vcvtq_f32_u32(vmovl_high_u16(raw));
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn vadd_f32_lo_hi(q: float32x4_t) -> float32x4_t {
    let hi = vextq_f32::<2>(q, q);
    vaddq_f32(q, hi)
}

#[must_use]
#[inline(always)]
fn cbcr_horiz_1(start_x: usize, src: &[u16], w0: float32x2_t, store: float32x4_t) -> float32x4_t {
    unsafe {
        let px = load_ga_1_f32(start_x, src);
        let w = vcombine_f32(w0, vdup_n_f32(0.0));
        vfmaq_f32(store, px, w)
    }
}

#[must_use]
#[inline(always)]
fn cbcr_horiz_2(start_x: usize, src: &[u16], w0: float32x2_t, store: float32x4_t) -> float32x4_t {
    unsafe {
        let px = load_ga_2_f32(start_x, src);
        let w = vzip1q_f32(vcombine_f32(w0, w0), vcombine_f32(w0, w0));
        vfmaq_f32(store, px, w)
    }
}

#[must_use]
#[inline(always)]
fn cbcr_horiz_4(
    start_x: usize,
    src: &[u16],
    weights: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        let (lo, hi) = load_ga_4_f32(start_x, src);
        let w_lo_dup = vget_low_f32(weights);
        let w_hi_dup = vget_high_f32(weights);
        let wlo = vzip1q_f32(
            vcombine_f32(w_lo_dup, w_lo_dup),
            vcombine_f32(w_lo_dup, w_lo_dup),
        );
        let whi = vzip1q_f32(
            vcombine_f32(w_hi_dup, w_hi_dup),
            vcombine_f32(w_hi_dup, w_hi_dup),
        );
        let acc = vfmaq_f32(store, lo, wlo);
        vfmaq_f32(acc, hi, whi)
    }
}

#[must_use]
#[inline(always)]
fn cbcr_horiz_8(
    start_x: usize,
    src: &[u16],
    weights: float32x4x2_t,
    store: float32x4_t,
) -> float32x4_t {
    let acc = cbcr_horiz_4(start_x, src, weights.0, store);
    cbcr_horiz_4(start_x + 4, src, weights.1, acc)
}

#[inline(always)]
unsafe fn load_weights_4(ptr: *const f32) -> float32x4_t {
    unsafe { vld1q_f32(ptr) }
}

#[inline(always)]
unsafe fn load_weights_8(ptr: *const f32) -> float32x4x2_t {
    unsafe { float32x4x2_t(vld1q_f32(ptr), vld1q_f32(ptr.add(4))) }
}

#[inline(always)]
unsafe fn load_weights_2(ptr: *const f32) -> float32x2_t {
    unsafe { vld1_f32(ptr) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_weight_1(ptr: *const f32) -> float32x2_t {
    unsafe { vld1_dup_f32(ptr) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_cbcr(dst: &mut [u16; 2], store: float32x4_t, v_max: float32x4_t) {
    let clamped = vmaxq_f32(vminq_f32(store, v_max), vdupq_n_f32(0.0));
    let u32x4 = vcvtaq_u32_f32(clamped);
    let u16x4 = vmovn_u32(u32x4);
    unsafe {
        vst1_lane_u32::<0>(dst.as_mut_ptr().cast(), vreinterpret_u32_u16(u16x4));
    }
}

pub(crate) fn convolve_horizontal_cbcr_neon_rows_4_f32_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        const CN: usize = 2;

        let v_max = vdupq_n_f32(((1u32 << bit_depth) - 1) as f32);

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
            let mut jx = 0usize;
            let mut store_0 = vdupq_n_f32(0.0);
            let mut store_1 = vdupq_n_f32(0.0);
            let mut store_2 = vdupq_n_f32(0.0);
            let mut store_3 = vdupq_n_f32(0.0);

            let bounds_size = bounds.size;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = load_weights_8(w_ptr.as_ptr());
                store_0 = cbcr_horiz_8(bounds_start, src0, ws, store_0);
                store_1 = cbcr_horiz_8(bounds_start, src1, ws, store_1);
                store_2 = cbcr_horiz_8(bounds_start, src2, ws, store_2);
                store_3 = cbcr_horiz_8(bounds_start, src3, ws, store_3);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = load_weights_4(w_ptr.as_ptr());
                store_0 = cbcr_horiz_4(bounds_start, src0, ws, store_0);
                store_1 = cbcr_horiz_4(bounds_start, src1, ws, store_1);
                store_2 = cbcr_horiz_4(bounds_start, src2, ws, store_2);
                store_3 = cbcr_horiz_4(bounds_start, src3, ws, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = load_weights_2(w_ptr.as_ptr());
                store_0 = cbcr_horiz_2(bounds_start, src0, ws, store_0);
                store_1 = cbcr_horiz_2(bounds_start, src1, ws, store_1);
                store_2 = cbcr_horiz_2(bounds_start, src2, ws, store_2);
                store_3 = cbcr_horiz_2(bounds_start, src3, ws, store_3);
                jx += 2;
            }

            store_0 = vadd_f32_lo_hi(store_0);
            store_1 = vadd_f32_lo_hi(store_1);
            store_2 = vadd_f32_lo_hi(store_2);
            store_3 = vadd_f32_lo_hi(store_3);

            while jx < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = load_weight_1(w_ptr.as_ptr());
                store_0 = cbcr_horiz_1(bounds_start, src0, ws, store_0);
                store_1 = cbcr_horiz_1(bounds_start, src1, ws, store_1);
                store_2 = cbcr_horiz_1(bounds_start, src2, ws, store_2);
                store_3 = cbcr_horiz_1(bounds_start, src3, ws, store_3);
                jx += 1;
            }

            store_cbcr(chunk0, store_0, v_max);
            store_cbcr(chunk1, store_1, v_max);
            store_cbcr(chunk2, store_2, v_max);
            store_cbcr(chunk3, store_3, v_max);
        }
    }
}

pub(crate) fn convolve_horizontal_cbcr_neon_f32_u16_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        const CN: usize = 2;

        let v_max = vdupq_n_f32(((1u32 << bit_depth) - 1) as f32);

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
            let mut store = vdupq_n_f32(0.0);

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = load_weights_8(w_ptr.as_ptr());
                store = cbcr_horiz_8(bounds_start, src, ws, store);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = load_weights_4(w_ptr.as_ptr());
                store = cbcr_horiz_4(bounds_start, src, ws, store);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = load_weights_2(w_ptr.as_ptr());
                store = cbcr_horiz_2(bounds_start, src, ws, store);
                jx += 2;
            }

            store = vadd_f32_lo_hi(store);

            while jx < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let ws = load_weight_1(w_ptr.as_ptr());
                store = cbcr_horiz_1(bounds_start, src, ws, store);
                jx += 1;
            }

            store_cbcr(dst, store, v_max);
        }
    }
}
