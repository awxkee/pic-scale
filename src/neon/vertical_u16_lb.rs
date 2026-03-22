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
use crate::filter_weights::FilterBounds;
use std::arch::aarch64::*;

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_chunks_16(
    chunks: &mut [[u16; 16]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = (1u32 << bit_depth) - 1;
    let mut cx = cx;

    let bounds_size = bounds.size;

    const PRECISION: i32 = 15;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

    let initial_store = vdupq_n_s32(ROUNDING_CONST);

    let v_max_colors = vdupq_n_u16(max_colors as u16);

    let v_px = cx;

    for (x, dst) in chunks.iter_mut().enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;
        let mut store2 = initial_store;
        let mut store3 = initial_store;

        let v_dx = v_px + x * 16;

        let mut j = 0usize;

        while j + 4 <= bounds_size {
            let py = bounds.start + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let weights = unsafe { vld1_s16(weights.get_unchecked(j..).as_ptr()) };

            let item_row0 = unsafe { vreinterpretq_s16_u16(vld1q_u16(src_ptr.as_ptr())) };
            let item_row1 =
                unsafe { vreinterpretq_s16_u16(vld1q_u16(src_ptr.get_unchecked(8..).as_ptr())) };

            store0 = vmlal_lane_s16::<0>(store0, vget_low_s16(item_row0), weights);
            store1 = vmlal_high_lane_s16::<0>(store1, item_row0, weights);
            store2 = vmlal_lane_s16::<0>(store2, vget_low_s16(item_row1), weights);
            store3 = vmlal_high_lane_s16::<0>(store3, item_row1, weights);

            let item_row0 = unsafe {
                vreinterpretq_s16_u16(vld1q_u16(src_ptr.get_unchecked(src_stride..).as_ptr()))
            };
            let item_row1 = unsafe {
                vreinterpretq_s16_u16(vld1q_u16(src_ptr.get_unchecked(src_stride + 8..).as_ptr()))
            };

            store0 = vmlal_lane_s16::<1>(store0, vget_low_s16(item_row0), weights);
            store1 = vmlal_high_lane_s16::<1>(store1, item_row0, weights);
            store2 = vmlal_lane_s16::<1>(store2, vget_low_s16(item_row1), weights);
            store3 = vmlal_high_lane_s16::<1>(store3, item_row1, weights);

            let item_row0 = unsafe {
                vreinterpretq_s16_u16(vld1q_u16(src_ptr.get_unchecked(src_stride * 2..).as_ptr()))
            };
            let item_row1 = unsafe {
                vreinterpretq_s16_u16(vld1q_u16(
                    src_ptr.get_unchecked(src_stride * 2 + 8..).as_ptr(),
                ))
            };

            store0 = vmlal_lane_s16::<2>(store0, vget_low_s16(item_row0), weights);
            store1 = vmlal_high_lane_s16::<2>(store1, item_row0, weights);
            store2 = vmlal_lane_s16::<2>(store2, vget_low_s16(item_row1), weights);
            store3 = vmlal_high_lane_s16::<2>(store3, item_row1, weights);

            let item_row0 = unsafe {
                vreinterpretq_s16_u16(vld1q_u16(src_ptr.get_unchecked(src_stride * 3..).as_ptr()))
            };
            let item_row1 = unsafe {
                vreinterpretq_s16_u16(vld1q_u16(
                    src_ptr.get_unchecked(src_stride * 3 + 8..).as_ptr(),
                ))
            };

            store0 = vmlal_lane_s16::<3>(store0, vget_low_s16(item_row0), weights);
            store1 = vmlal_high_lane_s16::<3>(store1, item_row0, weights);
            store2 = vmlal_lane_s16::<3>(store2, vget_low_s16(item_row1), weights);
            store3 = vmlal_high_lane_s16::<3>(store3, item_row1, weights);

            j += 4;
        }

        while j + 2 <= bounds_size {
            let py = bounds.start + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let weights = unsafe {
                vreinterpret_s16_u32(vld1_lane_u32::<0>(
                    weights.get_unchecked(j..).as_ptr().cast(),
                    vdup_n_u32(0),
                ))
            };

            let item_row0 = unsafe { vreinterpretq_s16_u16(vld1q_u16(src_ptr.as_ptr())) };
            let item_row1 =
                unsafe { vreinterpretq_s16_u16(vld1q_u16(src_ptr.get_unchecked(8..).as_ptr())) };

            store0 = vmlal_lane_s16::<0>(store0, vget_low_s16(item_row0), weights);
            store1 = vmlal_high_lane_s16::<0>(store1, item_row0, weights);
            store2 = vmlal_lane_s16::<0>(store2, vget_low_s16(item_row1), weights);
            store3 = vmlal_high_lane_s16::<0>(store3, item_row1, weights);

            let item_row0 = unsafe {
                vreinterpretq_s16_u16(vld1q_u16(src_ptr.get_unchecked(src_stride..).as_ptr()))
            };
            let item_row1 = unsafe {
                vreinterpretq_s16_u16(vld1q_u16(src_ptr.get_unchecked(src_stride + 8..).as_ptr()))
            };

            store0 = vmlal_lane_s16::<1>(store0, vget_low_s16(item_row0), weights);
            store1 = vmlal_high_lane_s16::<1>(store1, item_row0, weights);
            store2 = vmlal_lane_s16::<1>(store2, vget_low_s16(item_row1), weights);
            store3 = vmlal_high_lane_s16::<1>(store3, item_row1, weights);

            j += 2;
        }

        let weights = &weights[j..bounds_size];

        let base_y = bounds.start + j;

        for (j, &k_weight) in weights.iter().enumerate() {
            let py = base_y + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = vdupq_n_s16(k_weight);

            let item_row0 = unsafe { vreinterpretq_s16_u16(vld1q_u16(src_ptr.as_ptr())) };
            let item_row1 =
                unsafe { vreinterpretq_s16_u16(vld1q_u16(src_ptr.get_unchecked(8..).as_ptr())) };

            store0 = vmlal_s16(store0, vget_low_s16(item_row0), vget_low_s16(v_weight));
            store1 = vmlal_high_s16(store1, item_row0, v_weight);
            store2 = vmlal_s16(store2, vget_low_s16(item_row1), vget_low_s16(v_weight));
            store3 = vmlal_high_s16(store3, item_row1, v_weight);
        }

        let store0 = vqshrun_n_s32::<PRECISION>(store0);
        let store1 = vqshrun_n_s32::<PRECISION>(store1);
        let store2 = vqshrun_n_s32::<PRECISION>(store2);
        let store3 = vqshrun_n_s32::<PRECISION>(store3);

        let item0 = vminq_u16(vcombine_u16(store0, store1), v_max_colors);
        let item1 = vminq_u16(vcombine_u16(store2, store3), v_max_colors);

        unsafe {
            vst1q_u16(dst.as_mut_ptr(), item0);
            vst1q_u16(dst.get_unchecked_mut(8..).as_mut_ptr(), item1);
        }

        cx += 16;
    }
    cx
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_chunks_8(
    chunks: &mut [[u16; 8]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = (1u32 << bit_depth) - 1;
    let mut cx = cx;

    let bounds_size = bounds.size;

    const PRECISION: i32 = 15;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

    let initial_store = vdupq_n_s32(ROUNDING_CONST);

    let v_max_colors = vdupq_n_u16(max_colors as u16);

    let v_px = cx;

    for (x, dst) in chunks.iter_mut().enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;

        let v_dx = v_px + x * 8;

        for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
            let py = bounds.start + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = vdupq_n_s16(k_weight);

            let item_row = unsafe { vreinterpretq_s16_u16(vld1q_u16(src_ptr.as_ptr())) };

            store0 = vmlal_s16(store0, vget_low_s16(item_row), vget_low_s16(v_weight));
            store1 = vmlal_high_s16(store1, item_row, v_weight);
        }

        let item = vminq_u16(
            vcombine_u16(
                vqshrun_n_s32::<PRECISION>(store0),
                vqshrun_n_s32::<PRECISION>(store1),
            ),
            v_max_colors,
        );
        unsafe {
            vst1q_u16(dst.as_mut_ptr(), item);
        }

        cx = v_dx;
    }
    cx
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn convolve_chunks_4(
    chunks: &mut [[u16; 4]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = (1u32 << bit_depth) - 1;
    let mut cx = cx;

    let bounds_size = bounds.size;

    const PRECISION: i32 = 15;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

    let initial_store = vdupq_n_s32(ROUNDING_CONST);

    let v_max_colors = vdupq_n_u16(max_colors as u16);

    let v_px = cx;

    for (x, dst) in chunks.iter_mut().enumerate() {
        let mut store0 = initial_store;

        let v_dx = v_px + x * 4;

        for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
            let py = bounds.start + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = vdup_n_s16(k_weight);

            let item_row = unsafe { vreinterpret_s16_u16(vld1_u16(src_ptr.as_ptr())) };

            store0 = vmlal_s16(store0, item_row, v_weight);
        }

        let u_store0 = vmin_u16(
            vqshrun_n_s32::<PRECISION>(store0),
            vget_low_u16(v_max_colors),
        );
        unsafe {
            vst1_u16(dst.as_mut_ptr(), u_store0);
        }

        cx = v_dx;
    }
    cx
}

pub(crate) fn convolve_column_lb_u16(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    unsafe {
        let max_colors = (1u32 << bit_depth) - 1;
        let mut cx = 0usize;

        let bounds_size = bounds.size;

        const PRECISION: i32 = 15;
        const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

        cx = convolve_chunks_16(
            dst.as_chunks_mut::<16>().0,
            bounds,
            src,
            src_stride,
            weight,
            bit_depth,
            cx,
        );
        let mut rem = dst.as_chunks_mut::<16>().1;
        cx = convolve_chunks_8(
            rem.as_chunks_mut::<8>().0,
            bounds,
            src,
            src_stride,
            weight,
            bit_depth,
            cx,
        );

        rem = rem.as_chunks_mut::<8>().1;
        cx = convolve_chunks_4(
            rem.as_chunks_mut::<4>().0,
            bounds,
            src,
            src_stride,
            weight,
            bit_depth,
            cx,
        );

        let tail4 = rem.as_chunks_mut::<4>().1;

        let a_px = cx;

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = ROUNDING_CONST;

            let v_px = a_px + x;

            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_ptr = *src.get_unchecked(offset);

                store0 = store0.wrapping_add((src_ptr as i32).wrapping_mul(k_weight as i32));
            }

            *dst = (store0 >> PRECISION).max(0).min(max_colors as i32) as u16;
        }
    }
}
