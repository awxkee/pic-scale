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
use crate::filter_weights::FilterBounds;
use std::arch::aarch64::*;

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_chunks_16_i16(
    chunks: &mut [[i16; 16]],
    bounds: &FilterBounds,
    src: &[i16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = ((1i32 << (bit_depth - 1)) - 1) as i16;
    let min_colors = -(1i32 << (bit_depth - 1)) as i16;
    let mut cx = cx;

    let bounds_size = bounds.size;

    const PRECISION: i32 = 15;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

    let initial_store = vdupq_n_s32(ROUNDING_CONST);
    let v_max_colors = vdupq_n_s16(max_colors);
    let v_min_colors = vdupq_n_s16(min_colors);

    let v_px = cx;

    for (x, dst) in chunks.iter_mut().enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;
        let mut store2 = initial_store;
        let mut store3 = initial_store;

        let v_dx = v_px + x * 16;

        let mut j = 0usize;

        // Process 4 rows at a time using lane-indexed multiply-accumulate
        while j + 4 <= bounds_size {
            let py = bounds.start + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            // Load 4 weights into a single s16x4 vector for lane addressing
            let w = unsafe { vld1_s16(weights.get_unchecked(j..).as_ptr()) };

            // Row 0 — 16 pixels = two q-vectors of s16x8
            let row0_lo = unsafe { vld1q_s16(src_ptr.as_ptr()) };
            let row0_hi = unsafe { vld1q_s16(src_ptr.get_unchecked(8..).as_ptr()) };

            store0 = vmlal_lane_s16::<0>(store0, vget_low_s16(row0_lo), w);
            store1 = vmlal_high_lane_s16::<0>(store1, row0_lo, w);
            store2 = vmlal_lane_s16::<0>(store2, vget_low_s16(row0_hi), w);
            store3 = vmlal_high_lane_s16::<0>(store3, row0_hi, w);

            // Row 1
            let row1_lo = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let row1_hi = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride + 8..).as_ptr()) };

            store0 = vmlal_lane_s16::<1>(store0, vget_low_s16(row1_lo), w);
            store1 = vmlal_high_lane_s16::<1>(store1, row1_lo, w);
            store2 = vmlal_lane_s16::<1>(store2, vget_low_s16(row1_hi), w);
            store3 = vmlal_high_lane_s16::<1>(store3, row1_hi, w);

            // Row 2
            let row2_lo = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let row2_hi =
                unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride * 2 + 8..).as_ptr()) };

            store0 = vmlal_lane_s16::<2>(store0, vget_low_s16(row2_lo), w);
            store1 = vmlal_high_lane_s16::<2>(store1, row2_lo, w);
            store2 = vmlal_lane_s16::<2>(store2, vget_low_s16(row2_hi), w);
            store3 = vmlal_high_lane_s16::<2>(store3, row2_hi, w);

            // Row 3
            let row3_lo = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };
            let row3_hi =
                unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride * 3 + 8..).as_ptr()) };

            store0 = vmlal_lane_s16::<3>(store0, vget_low_s16(row3_lo), w);
            store1 = vmlal_high_lane_s16::<3>(store1, row3_lo, w);
            store2 = vmlal_lane_s16::<3>(store2, vget_low_s16(row3_hi), w);
            store3 = vmlal_high_lane_s16::<3>(store3, row3_hi, w);

            j += 4;
        }

        // Process 2 rows at a time
        while j + 2 <= bounds_size {
            let py = bounds.start + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let w = unsafe {
                vreinterpret_s16_u32(vld1_lane_u32::<0>(
                    weights.get_unchecked(j..).as_ptr().cast(),
                    vdup_n_u32(0),
                ))
            };

            let row0_lo = unsafe { vld1q_s16(src_ptr.as_ptr()) };
            let row0_hi = unsafe { vld1q_s16(src_ptr.get_unchecked(8..).as_ptr()) };

            store0 = vmlal_lane_s16::<0>(store0, vget_low_s16(row0_lo), w);
            store1 = vmlal_high_lane_s16::<0>(store1, row0_lo, w);
            store2 = vmlal_lane_s16::<0>(store2, vget_low_s16(row0_hi), w);
            store3 = vmlal_high_lane_s16::<0>(store3, row0_hi, w);

            let row1_lo = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let row1_hi = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride + 8..).as_ptr()) };

            store0 = vmlal_lane_s16::<1>(store0, vget_low_s16(row1_lo), w);
            store1 = vmlal_high_lane_s16::<1>(store1, row1_lo, w);
            store2 = vmlal_lane_s16::<1>(store2, vget_low_s16(row1_hi), w);
            store3 = vmlal_high_lane_s16::<1>(store3, row1_hi, w);

            j += 2;
        }

        // Scalar tail — one row at a time
        for (jj, &k_weight) in weights[j..bounds_size].iter().enumerate() {
            let py = bounds.start + j + jj;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = vdupq_n_s16(k_weight);

            let row_lo = unsafe { vld1q_s16(src_ptr.as_ptr()) };
            let row_hi = unsafe { vld1q_s16(src_ptr.get_unchecked(8..).as_ptr()) };

            store0 = vmlal_s16(store0, vget_low_s16(row_lo), vget_low_s16(v_weight));
            store1 = vmlal_high_s16(store1, row_lo, v_weight);
            store2 = vmlal_s16(store2, vget_low_s16(row_hi), vget_low_s16(v_weight));
            store3 = vmlal_high_s16(store3, row_hi, v_weight);
        }

        // Shift right with signed saturation: s32 -> s16 (signed narrowing)
        let shifted0 = vqshrn_n_s32::<PRECISION>(store0);
        let shifted1 = vqshrn_n_s32::<PRECISION>(store1);
        let shifted2 = vqshrn_n_s32::<PRECISION>(store2);
        let shifted3 = vqshrn_n_s32::<PRECISION>(store3);

        // Combine pairs back to 128-bit, then clamp to [min_colors, max_colors]
        let item0 = vminq_s16(
            vmaxq_s16(vcombine_s16(shifted0, shifted1), v_min_colors),
            v_max_colors,
        );
        let item1 = vminq_s16(
            vmaxq_s16(vcombine_s16(shifted2, shifted3), v_min_colors),
            v_max_colors,
        );

        unsafe {
            vst1q_s16(dst.as_mut_ptr(), item0);
            vst1q_s16(dst.get_unchecked_mut(8..).as_mut_ptr(), item1);
        }

        cx += 16;
    }
    cx
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_chunks_8_i16(
    chunks: &mut [[i16; 8]],
    bounds: &FilterBounds,
    src: &[i16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = ((1i32 << (bit_depth - 1)) - 1) as i16;
    let min_colors = -(1i32 << (bit_depth - 1)) as i16;
    let mut cx = cx;

    const PRECISION: i32 = 15;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

    let initial_store = vdupq_n_s32(ROUNDING_CONST);
    let v_max_colors = vdupq_n_s16(max_colors);
    let v_min_colors = vdupq_n_s16(min_colors);

    let v_px = cx;

    for (x, dst) in chunks.iter_mut().enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;

        let v_dx = v_px + x * 8;

        for (j, &k_weight) in weights.iter().enumerate() {
            let py = bounds.start + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = vdupq_n_s16(k_weight);
            let item_row = unsafe { vld1q_s16(src_ptr.as_ptr()) };

            store0 = vmlal_s16(store0, vget_low_s16(item_row), vget_low_s16(v_weight));
            store1 = vmlal_high_s16(store1, item_row, v_weight);
        }

        let item = vminq_s16(
            vmaxq_s16(
                vcombine_s16(
                    vqshrn_n_s32::<PRECISION>(store0),
                    vqshrn_n_s32::<PRECISION>(store1),
                ),
                v_min_colors,
            ),
            v_max_colors,
        );

        unsafe {
            vst1q_s16(dst.as_mut_ptr(), item);
        }

        cx += 8;
    }
    cx
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_chunks_4_i16(
    chunks: &mut [[i16; 4]],
    bounds: &FilterBounds,
    src: &[i16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = ((1i32 << (bit_depth - 1)) - 1) as i16;
    let min_colors = -(1i32 << (bit_depth - 1)) as i16;
    let mut cx = cx;

    const PRECISION: i32 = 15;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

    let initial_store = vdupq_n_s32(ROUNDING_CONST);
    let v_max_colors = vdup_n_s16(max_colors);
    let v_min_colors = vdup_n_s16(min_colors);

    let v_px = cx;

    for (x, dst) in chunks.iter_mut().enumerate() {
        let mut store0 = initial_store;

        let v_dx = v_px + x * 4;

        for (j, &k_weight) in weights.iter().enumerate() {
            let py = bounds.start + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = vdup_n_s16(k_weight);
            let item_row = unsafe { vld1_s16(src_ptr.as_ptr()) };

            store0 = vmlal_s16(store0, item_row, v_weight);
        }

        let u_store0 = vmin_s16(
            vmax_s16(vqshrn_n_s32::<PRECISION>(store0), v_min_colors),
            v_max_colors,
        );

        unsafe {
            vst1_s16(dst.as_mut_ptr(), u_store0);
        }

        cx += 4;
    }
    cx
}

pub(crate) fn convolve_column_lb_i16(
    _: usize,
    bounds: &FilterBounds,
    src: &[i16],
    dst: &mut [i16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
) {
    unsafe {
        let max_colors = ((1i32 << (bit_depth - 1)) - 1) as i16;
        let min_colors = -(1i32 << (bit_depth - 1)) as i16;
        let mut cx = 0usize;

        let bounds_size = bounds.size;
        let weights = &weights[..bounds_size];

        const PRECISION: i32 = 15;
        const ROUNDING: i32 = 1 << (PRECISION - 1);

        cx = convolve_chunks_16_i16(
            dst.as_chunks_mut::<16>().0,
            bounds,
            src,
            src_stride,
            weights,
            bit_depth,
            cx,
        );
        let mut rem = dst.as_chunks_mut::<16>().1;

        cx = convolve_chunks_8_i16(
            rem.as_chunks_mut::<8>().0,
            bounds,
            src,
            src_stride,
            weights,
            bit_depth,
            cx,
        );
        rem = rem.as_chunks_mut::<8>().1;

        cx = convolve_chunks_4_i16(
            rem.as_chunks_mut::<4>().0,
            bounds,
            src,
            src_stride,
            weights,
            bit_depth,
            cx,
        );

        let tail4 = rem.as_chunks_mut::<4>().1;

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0: i32 = ROUNDING;

            let v_px = cx + x;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_val = *src.get_unchecked(offset);

                store0 = store0.wrapping_add((src_val as i32).wrapping_mul(k_weight as i32));
            }

            *dst = (store0 >> PRECISION)
                .max(min_colors as i32)
                .min(max_colors as i32) as i16;
        }
    }
}
