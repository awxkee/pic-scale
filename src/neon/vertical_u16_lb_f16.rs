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
use crate::neon::f16_utils::{
    xvcvt_f16_u16, xvcvta_u16_f16, xvcvtaq_u16_f16, xvcvtq_f16_u16, xvfmla_f16, xvfmlaq_f16,
    xvzerosq_f16,
};
use crate::neon::{xreinterpret_f16_u16, xreinterpretq_f16_u16, xvget_low_f16};
use std::arch::aarch64::*;

pub(crate) fn convolve_column_lb_u16_f16(
    j0: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    unsafe {
        let transmuted_u16 =
            std::slice::from_raw_parts(weight.as_ptr() as *const u16, weight.len());
        convolve_column_lb_u16_f16_impl(
            j0,
            bounds,
            src,
            dst,
            src_stride,
            transmuted_u16,
            bit_depth,
        );
    }
}

#[target_feature(enable = "fp16")]
unsafe fn convolve_column_lb_u16_f16_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[u16],
    bit_depth: u32,
) {
    let max_colors = (1 << bit_depth) - 1;
    let mut cx = 0usize;

    let bounds_size = bounds.size;

    let initial_store = xvzerosq_f16();

    let v_max_colors = vdupq_n_u16(max_colors);

    let v_px = cx;

    let iter16 = dst.chunks_exact_mut(16);

    for (x, dst) in iter16.enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;

        let v_dx = v_px + x * 16;

        for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
            let py = bounds.start + j;
            let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

            let v_weight = xreinterpretq_f16_u16(vdupq_n_u16(k_weight));

            let item_row0 = vld1q_u16(src_ptr.as_ptr());
            let item_row1 = vld1q_u16(src_ptr.as_ptr().add(8));

            store0 = xvfmlaq_f16(store0, xvcvtq_f16_u16(item_row0), v_weight);
            store1 = xvfmlaq_f16(store1, xvcvtq_f16_u16(item_row1), v_weight);
        }

        let item0 = vminq_u16(xvcvtaq_u16_f16(store0), v_max_colors);
        let item1 = vminq_u16(xvcvtaq_u16_f16(store1), v_max_colors);

        vst1q_u16(dst.as_mut_ptr(), item0);
        vst1q_u16(dst.as_mut_ptr().add(8), item1);

        cx = v_dx;
    }

    let tail16 = dst.chunks_exact_mut(16).into_remainder();
    let iter8 = tail16.chunks_exact_mut(8);

    let v_px = cx;

    for (x, dst) in iter8.enumerate() {
        let mut store0 = initial_store;

        let v_dx = v_px + x * 8;

        for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
            let py = bounds.start + j;
            let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

            let v_weight = xreinterpretq_f16_u16(vdupq_n_u16(k_weight));

            let item_row = vld1q_u16(src_ptr.as_ptr());

            store0 = xvfmlaq_f16(store0, xvcvtq_f16_u16(item_row), v_weight);
        }

        let item = vminq_u16(xvcvtaq_u16_f16(store0), v_max_colors);
        vst1q_u16(dst.as_mut_ptr(), item);

        cx = v_dx;
    }

    let tail8 = tail16.chunks_exact_mut(8).into_remainder();
    let iter4 = tail8.chunks_exact_mut(4);

    let v_cx = cx;

    for (x, dst) in iter4.enumerate() {
        let mut store0 = xvget_low_f16(initial_store);

        let v_dx = v_cx + x * 4;

        for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
            let py = bounds.start + j;
            let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

            let v_weight = xreinterpret_f16_u16(vdup_n_u16(k_weight));

            let item_row = vld1_u16(src_ptr.as_ptr());

            store0 = xvfmla_f16(store0, xvcvt_f16_u16(item_row), v_weight);
        }

        let u_store0 = vmin_u16(xvcvta_u16_f16(store0), vget_low_u16(v_max_colors));
        vst1_u16(dst.as_mut_ptr(), u_store0);

        cx = v_dx;
    }

    let tail4 = tail8.chunks_exact_mut(4).into_remainder();

    let a_px = cx;

    for (x, dst) in tail4.iter_mut().enumerate() {
        let mut store0 = xvget_low_f16(initial_store);

        let v_px = a_px + x;

        for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
            let py = bounds.start + j;
            let offset = src_stride * py + v_px;
            let src_ptr = src.get_unchecked(offset..(offset + 1));

            let v_weight = xreinterpret_f16_u16(vdup_n_u16(k_weight));
            let item_row = xvcvt_f16_u16(vld1_lane_u16::<0>(src_ptr.as_ptr(), vdup_n_u16(0)));
            store0 = xvfmla_f16(store0, item_row, v_weight);
        }

        let u_store0 = vmin_u16(xvcvta_u16_f16(store0), vget_low_u16(v_max_colors));
        vst1_lane_u16::<0>(dst, u_store0);
    }
}
