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

pub(crate) fn convolve_column_hb_s16(
    w: usize,
    bounds: &FilterBounds,
    src: &[i16],
    dst: &mut [i16],
    src_stride: usize,
    weight: &[i32],
    bit_depth: u32,
) {
    unsafe { convolve_column_hb_s16_impl(w, bounds, src, dst, src_stride, weight, bit_depth) }
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn convolve_16_items_s16(
    chunks: &mut [[i16; 16]],
    bounds: &FilterBounds,
    src: &[i16],
    src_stride: usize,
    weights: &[i32],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = ((1i32 << (bit_depth - 1)) - 1) as i16;
    let min_colors = (-(1i32 << (bit_depth - 1))) as i16;
    let mut cx = cx;

    let initial_store = vdupq_n_s32(1i32 << 5);

    let v_max_colors = vdupq_n_s16(max_colors);
    let v_min_colors = vdupq_n_s16(min_colors);

    let v_px = cx;
    for (x, dst) in chunks.iter_mut().enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;
        let mut store2 = initial_store;
        let mut store3 = initial_store;

        let v_dx = v_px + x * 16;

        let q_weights = weights.as_chunks::<4>().0;

        for (j, k_weight) in q_weights.iter().enumerate() {
            let py = bounds.start + j * 4;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = unsafe { vld1q_s32(k_weight.as_ptr()) };

            // row 0  (offset 0)
            let item_row0 = unsafe { vld1q_s16(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_s16(src_ptr.get_unchecked(8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<0>(
                store0,
                vshll_n_s16::<6>(vget_low_s16(item_row0)),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<0>(store1, vshll_high_n_s16::<6>(item_row0), v_weight);
            store2 = vqrdmlahq_laneq_s32::<0>(
                store2,
                vshll_n_s16::<6>(vget_low_s16(item_row1)),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<0>(store3, vshll_high_n_s16::<6>(item_row1), v_weight);

            // row 1  (offset src_stride)
            let item_row0 = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let item_row1 = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride + 8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<1>(
                store0,
                vshll_n_s16::<6>(vget_low_s16(item_row0)),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<1>(store1, vshll_high_n_s16::<6>(item_row0), v_weight);
            store2 = vqrdmlahq_laneq_s32::<1>(
                store2,
                vshll_n_s16::<6>(vget_low_s16(item_row1)),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<1>(store3, vshll_high_n_s16::<6>(item_row1), v_weight);

            // row 2  (offset src_stride * 2)
            let item_row0 = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let item_row1 =
                unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride * 2 + 8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<2>(
                store0,
                vshll_n_s16::<6>(vget_low_s16(item_row0)),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<2>(store1, vshll_high_n_s16::<6>(item_row0), v_weight);
            store2 = vqrdmlahq_laneq_s32::<2>(
                store2,
                vshll_n_s16::<6>(vget_low_s16(item_row1)),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<2>(store3, vshll_high_n_s16::<6>(item_row1), v_weight);

            // row 3  (offset src_stride * 3)
            let item_row0 = unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };
            let item_row1 =
                unsafe { vld1q_s16(src_ptr.get_unchecked(src_stride * 3 + 8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<3>(
                store0,
                vshll_n_s16::<6>(vget_low_s16(item_row0)),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<3>(store1, vshll_high_n_s16::<6>(item_row0), v_weight);
            store2 = vqrdmlahq_laneq_s32::<3>(
                store2,
                vshll_n_s16::<6>(vget_low_s16(item_row1)),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<3>(store3, vshll_high_n_s16::<6>(item_row1), v_weight);
        }

        let base_y = weights.as_chunks::<4>().0.len() * 4 + bounds.start;
        let tail_w = weights.as_chunks::<4>().1;

        for (j, &k_weight) in tail_w.iter().enumerate() {
            let py = base_y + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = vdupq_n_s32(k_weight);

            let item_row0 = unsafe { vld1q_s16(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_s16(src_ptr.get_unchecked(8..).as_ptr()) };

            store0 = vqrdmlahq_s32(store0, vshll_n_s16::<6>(vget_low_s16(item_row0)), v_weight);
            store1 = vqrdmlahq_s32(store1, vshll_high_n_s16::<6>(item_row0), v_weight);
            store2 = vqrdmlahq_s32(store2, vshll_n_s16::<6>(vget_low_s16(item_row1)), v_weight);
            store3 = vqrdmlahq_s32(store3, vshll_high_n_s16::<6>(item_row1), v_weight);
        }

        let store0 = vqshrn_n_s32::<6>(store0);
        let store1 = vqshrn_n_s32::<6>(store1);
        let store2 = vqshrn_n_s32::<6>(store2);
        let store3 = vqshrn_n_s32::<6>(store3);

        let item0 = vminq_s16(
            vmaxq_s16(vcombine_s16(store0, store1), v_min_colors),
            v_max_colors,
        );
        let item1 = vminq_s16(
            vmaxq_s16(vcombine_s16(store2, store3), v_min_colors),
            v_max_colors,
        );

        unsafe {
            vst1q_s16(dst.as_mut_ptr(), item0);
            vst1q_s16(dst.get_unchecked_mut(8..).as_mut_ptr(), item1);
        }

        cx = v_dx;
    }
    cx
}

#[target_feature(enable = "rdm")]
fn convolve_column_hb_s16_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[i16],
    dst: &mut [i16],
    src_stride: usize,
    weights: &[i32],
    bit_depth: u32,
) {
    unsafe {
        let max_colors = ((1i32 << (bit_depth - 1)) - 1) as i16;
        let min_colors = (-(1i32 << (bit_depth - 1))) as i16;
        let mut cx = 0usize;

        let bounds_size = bounds.size;
        let weights = &weights[..bounds.size];

        let initial_store = vdupq_n_s32(1i32 << 5);

        let v_max_colors = vdupq_n_s16(max_colors);
        let v_min_colors = vdupq_n_s16(min_colors);

        cx = convolve_16_items_s16(
            dst.as_chunks_mut::<16>().0,
            bounds,
            src,
            src_stride,
            weights,
            bit_depth,
            cx,
        );

        let tail16 = dst.as_chunks_mut::<16>().1;
        let iter8 = tail16.chunks_exact_mut(8);
        let v_px = cx;

        for (x, dst) in iter8.enumerate() {
            let mut store0 = initial_store;
            let mut store1 = initial_store;

            let v_dx = v_px + x * 8;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = vdupq_n_s32(k_weight);

                let item_row = vld1q_s16(src_ptr.as_ptr());

                store0 = vqrdmlahq_s32(store0, vshll_n_s16::<6>(vget_low_s16(item_row)), v_weight);
                store1 = vqrdmlahq_s32(store1, vshll_high_n_s16::<6>(item_row), v_weight);
            }

            let combined = vcombine_s16(vqshrn_n_s32::<6>(store0), vqshrn_n_s32::<6>(store1));
            let item = vminq_s16(vmaxq_s16(combined, v_min_colors), v_max_colors);
            vst1q_s16(dst.as_mut_ptr(), item);

            cx += 8;
        }

        let tail8 = tail16.chunks_exact_mut(8).into_remainder();
        let iter4 = tail8.chunks_exact_mut(4);
        let v_cx = cx;

        for (x, dst) in iter4.enumerate() {
            let mut store0 = initial_store;

            let v_dx = v_cx + x * 4;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = vdupq_n_s32(k_weight);

                let item_row = vld1_s16(src_ptr.as_ptr());

                store0 = vqrdmlahq_s32(store0, vshll_n_s16::<6>(item_row), v_weight);
            }

            let narrowed = vqshrn_n_s32::<6>(store0);
            let u_store0 = vmin_s16(
                vmax_s16(narrowed, vget_low_s16(v_min_colors)),
                vget_low_s16(v_max_colors),
            );
            vst1_s16(dst.as_mut_ptr(), u_store0);

            cx += 4;
        }

        let tail4 = tail8.chunks_exact_mut(4).into_remainder();

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0: i64 = 0;

            let v_px = cx + x;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_px = src.get_unchecked(offset);

                store0 += *src_px as i64 * k_weight as i64;
            }
            const R: i64 = 1 << 30;
            *dst = ((store0 + R) >> 31)
                .max(min_colors as i64)
                .min(max_colors as i64) as i16;
        }
    }
}
