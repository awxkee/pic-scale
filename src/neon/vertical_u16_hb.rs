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

pub(crate) fn convolve_column_hb_u16(
    w: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i32],
    bit_depth: u32,
) {
    unsafe { convolve_column_hb_impl(w, bounds, src, dst, src_stride, weight, bit_depth) }
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn convolve_16_items(
    chunks: &mut [[u16; 16]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weights: &[i32],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = (1u32 << bit_depth) - 1;
    let mut cx = cx;

    let initial_store = vdupq_n_s32((1 << 5) - 1);

    let v_max_colors = vdupq_n_u16(max_colors as u16);

    let v_px = cx;
    for (x, dst) in chunks.iter_mut().enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;
        let mut store2 = initial_store;
        let mut store3 = initial_store;

        let v_dx = v_px + x * 16;

        let q_weights = weights.as_chunks::<4>().0;

        for (j, &k_weight) in q_weights.iter().enumerate() {
            let py = bounds.start + j * 4;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = unsafe { vld1q_s32(k_weight.as_ptr()) };

            let item_row0 = unsafe { vld1q_u16(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_u16(src_ptr.get_unchecked(8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<0>(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<0>(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_laneq_s32::<0>(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<0>(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );

            let item_row0 = unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let item_row1 = unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride + 8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<1>(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<1>(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_laneq_s32::<1>(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<1>(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );

            let item_row0 = unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let item_row1 =
                unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride * 2 + 8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<2>(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<2>(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_laneq_s32::<2>(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<2>(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );

            let item_row0 = unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };
            let item_row1 =
                unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride * 3 + 8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<3>(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<3>(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_laneq_s32::<3>(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<3>(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );
        }

        let base_y = weights.as_chunks::<4>().0.len() * 4 + bounds.start;
        let weights = weights.as_chunks::<4>().1;

        for (j, &k_weight) in weights.iter().enumerate() {
            let py = base_y + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = vdupq_n_s32(k_weight);

            let item_row0 = unsafe { vld1q_u16(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_u16(src_ptr.get_unchecked(8..).as_ptr()) };

            store0 = vqrdmlahq_s32(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_s32(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_s32(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_s32(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );
        }

        let store0 = vqshrun_n_s32::<6>(store0);
        let store1 = vqshrun_n_s32::<6>(store1);
        let store2 = vqshrun_n_s32::<6>(store2);
        let store3 = vqshrun_n_s32::<6>(store3);

        let item0 = vminq_u16(vcombine_u16(store0, store1), v_max_colors);
        let item1 = vminq_u16(vcombine_u16(store2, store3), v_max_colors);

        unsafe {
            vst1q_u16(dst.as_mut_ptr(), item0);
            vst1q_u16(dst.get_unchecked_mut(8..).as_mut_ptr(), item1);
        }

        cx = v_dx;
    }
    cx
}

#[target_feature(enable = "rdm")]
fn convolve_column_hb_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weights: &[i32],
    bit_depth: u32,
) {
    unsafe {
        let max_colors = (1u32 << bit_depth) - 1;
        let mut cx = 0usize;

        let bounds_size = bounds.size;
        let weights = &weights[..bounds.size];

        let initial_store = vdupq_n_s32((1 << 5) - 1);

        let v_max_colors = vdupq_n_u16(max_colors as u16);

        cx = convolve_16_items(
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

                let item_row = vld1q_u16(src_ptr.as_ptr());

                store0 = vqrdmlahq_s32(
                    store0,
                    vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row))),
                    v_weight,
                );
                store1 = vqrdmlahq_s32(
                    store1,
                    vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row)),
                    v_weight,
                );
            }

            let item = vminq_u16(
                vcombine_u16(vqshrun_n_s32::<6>(store0), vqshrun_n_s32::<6>(store1)),
                v_max_colors,
            );
            vst1q_u16(dst.as_mut_ptr(), item);

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

                let item_row = vld1_u16(src_ptr.as_ptr());

                store0 = vqrdmlahq_s32(
                    store0,
                    vreinterpretq_s32_u32(vshll_n_u16::<6>(item_row)),
                    v_weight,
                );
            }

            let u_store0 = vmin_u16(vqshrun_n_s32::<6>(store0), vget_low_u16(v_max_colors));
            vst1_u16(dst.as_mut_ptr(), u_store0);

            cx += 4;
        }

        let tail4 = tail8.chunks_exact_mut(4).into_remainder();

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = 0;

            let v_px = cx + x;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_ptr = src.get_unchecked(offset);

                store0 += *src_ptr as i64 * k_weight as i64;
            }

            const R: i64 = (1 << 30) - 1;
            *dst = ((store0 + R) >> 31).max(0).min(max_colors as i64) as u16;
        }
    }
}
