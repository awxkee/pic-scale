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
use crate::neon::utils::{xvld1q_f32_x2, xvld1q_f32_x4};
use std::arch::aarch64::*;

pub(crate) fn convolve_horizontal_rgba_neon_row_one_f32_f64(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        const CN: usize = 4;
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store0 = vdupq_n_f64(0.);
            let mut store1 = vdupq_n_f64(0.);

            while jx + 4 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let wz0 = vld1q_f64(ptr);
                let wz1 = vld1q_f64(ptr.add(2));
                let rgb_pixel = xvld1q_f32_x4(src.get_unchecked(bounds_start * CN..).as_ptr());

                store0 = vfmaq_laneq_f64::<0>(store0, vcvt_f64_f32(vget_low_f32(rgb_pixel.0)), wz0);
                store1 = vfmaq_laneq_f64::<0>(store1, vcvt_high_f64_f32(rgb_pixel.0), wz0);

                store0 = vfmaq_laneq_f64::<1>(store0, vcvt_f64_f32(vget_low_f32(rgb_pixel.1)), wz0);
                store1 = vfmaq_laneq_f64::<1>(store1, vcvt_high_f64_f32(rgb_pixel.1), wz0);

                store0 = vfmaq_laneq_f64::<0>(store0, vcvt_f64_f32(vget_low_f32(rgb_pixel.2)), wz1);
                store1 = vfmaq_laneq_f64::<0>(store1, vcvt_high_f64_f32(rgb_pixel.2), wz1);

                store0 = vfmaq_laneq_f64::<1>(store0, vcvt_f64_f32(vget_low_f32(rgb_pixel.3)), wz1);
                store1 = vfmaq_laneq_f64::<1>(store1, vcvt_high_f64_f32(rgb_pixel.3), wz1);

                jx += 4;
            }

            while jx + 2 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let wz = vld1q_f64(ptr);

                let rgb_pixel = xvld1q_f32_x2(src.get_unchecked(bounds_start * CN..).as_ptr());

                store0 = vfmaq_laneq_f64::<0>(store0, vcvt_f64_f32(vget_low_f32(rgb_pixel.0)), wz);
                store1 = vfmaq_laneq_f64::<0>(store1, vcvt_high_f64_f32(rgb_pixel.0), wz);

                store0 = vfmaq_laneq_f64::<1>(store0, vcvt_f64_f32(vget_low_f32(rgb_pixel.1)), wz);
                store1 = vfmaq_laneq_f64::<1>(store1, vcvt_high_f64_f32(rgb_pixel.1), wz);

                jx += 2;
            }

            while jx < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let w0 = vld1q_dup_f64(ptr);
                let rgb_pixel = vld1q_f32(src.get_unchecked(bounds_start * CN..).as_ptr());
                store0 = vfmaq_f64(store0, vcvt_f64_f32(vget_low_f32(rgb_pixel)), w0);
                store1 = vfmaq_f64(store1, vcvt_high_f64_f32(rgb_pixel), w0);
                jx += 1;
            }

            let px = x * CN;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            vst1q_f32(
                dest_ptr,
                vcombine_f32(vcvt_f32_f64(store0), vcvt_f32_f64(store1)),
            );

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_neon_rows_4_f32_f64(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        const CN: usize = 4;
        let mut filter_offset = 0usize;
        let zeros = vdupq_n_f64(0.);
        let weights_ptr = filter_weights.weights.as_ptr();

        let s_ptr_1 = src.get_unchecked(src_stride..);
        let s_ptr_2 = src.get_unchecked(src_stride * 2..);
        let s_ptr_3 = src.get_unchecked(src_stride * 3..);

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;
            let mut store_4 = zeros;
            let mut store_5 = zeros;
            let mut store_6 = zeros;
            let mut store_7 = zeros;

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let w0 = vld1q_f64(ptr);
                let bounds_start = bounds.start + jx;

                let rgb_pixel0 = xvld1q_f32_x2(src.get_unchecked(bounds_start * CN..).as_ptr());
                let rgb_pixel1 = xvld1q_f32_x2(s_ptr_1.get_unchecked(bounds_start * CN..).as_ptr());
                let rgb_pixel2 = xvld1q_f32_x2(s_ptr_2.get_unchecked(bounds_start * CN..).as_ptr());
                let rgb_pixel3 = xvld1q_f32_x2(s_ptr_3.get_unchecked(bounds_start * CN..).as_ptr());

                store_0 =
                    vfmaq_laneq_f64::<0>(store_0, vcvt_f64_f32(vget_low_f32(rgb_pixel0.0)), w0);
                store_1 = vfmaq_laneq_f64::<0>(store_1, vcvt_high_f64_f32(rgb_pixel0.0), w0);

                store_2 =
                    vfmaq_laneq_f64::<0>(store_2, vcvt_f64_f32(vget_low_f32(rgb_pixel1.0)), w0);
                store_3 = vfmaq_laneq_f64::<0>(store_3, vcvt_high_f64_f32(rgb_pixel1.0), w0);

                store_4 =
                    vfmaq_laneq_f64::<0>(store_4, vcvt_f64_f32(vget_low_f32(rgb_pixel2.0)), w0);
                store_5 = vfmaq_laneq_f64::<0>(store_5, vcvt_high_f64_f32(rgb_pixel2.0), w0);

                store_6 =
                    vfmaq_laneq_f64::<0>(store_6, vcvt_f64_f32(vget_low_f32(rgb_pixel3.0)), w0);
                store_7 = vfmaq_laneq_f64::<0>(store_7, vcvt_high_f64_f32(rgb_pixel3.0), w0);

                store_0 =
                    vfmaq_laneq_f64::<1>(store_0, vcvt_f64_f32(vget_low_f32(rgb_pixel0.1)), w0);
                store_1 = vfmaq_laneq_f64::<1>(store_1, vcvt_high_f64_f32(rgb_pixel0.1), w0);

                store_2 =
                    vfmaq_laneq_f64::<1>(store_2, vcvt_f64_f32(vget_low_f32(rgb_pixel1.1)), w0);
                store_3 = vfmaq_laneq_f64::<1>(store_3, vcvt_high_f64_f32(rgb_pixel1.1), w0);

                store_4 =
                    vfmaq_laneq_f64::<1>(store_4, vcvt_f64_f32(vget_low_f32(rgb_pixel2.1)), w0);
                store_5 = vfmaq_laneq_f64::<1>(store_5, vcvt_high_f64_f32(rgb_pixel2.1), w0);

                store_6 =
                    vfmaq_laneq_f64::<1>(store_6, vcvt_f64_f32(vget_low_f32(rgb_pixel3.1)), w0);
                store_7 = vfmaq_laneq_f64::<1>(store_7, vcvt_high_f64_f32(rgb_pixel3.1), w0);

                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let w0 = vld1q_dup_f64(ptr);
                let bounds_start = bounds.start + jx;

                let rgb_pixel0 = vld1q_f32(src.get_unchecked(bounds_start * CN..).as_ptr());
                let rgb_pixel1 = vld1q_f32(s_ptr_1.get_unchecked(bounds_start * CN..).as_ptr());
                let rgb_pixel2 = vld1q_f32(s_ptr_2.get_unchecked(bounds_start * CN..).as_ptr());
                let rgb_pixel3 = vld1q_f32(s_ptr_3.get_unchecked(bounds_start * CN..).as_ptr());

                store_0 = vfmaq_f64(store_0, vcvt_f64_f32(vget_low_f32(rgb_pixel0)), w0);
                store_1 = vfmaq_f64(store_1, vcvt_high_f64_f32(rgb_pixel0), w0);

                store_2 = vfmaq_f64(store_2, vcvt_f64_f32(vget_low_f32(rgb_pixel1)), w0);
                store_3 = vfmaq_f64(store_3, vcvt_high_f64_f32(rgb_pixel1), w0);

                store_4 = vfmaq_f64(store_4, vcvt_f64_f32(vget_low_f32(rgb_pixel2)), w0);
                store_5 = vfmaq_f64(store_5, vcvt_high_f64_f32(rgb_pixel2), w0);

                store_6 = vfmaq_f64(store_6, vcvt_f64_f32(vget_low_f32(rgb_pixel3)), w0);
                store_7 = vfmaq_f64(store_7, vcvt_high_f64_f32(rgb_pixel3), w0);

                jx += 1;
            }

            let px = x * CN;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            vst1q_f32(
                dest_ptr,
                vcombine_f32(vcvt_f32_f64(store_0), vcvt_f32_f64(store_1)),
            );

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
            vst1q_f32(
                dest_ptr,
                vcombine_f32(vcvt_f32_f64(store_2), vcvt_f32_f64(store_3)),
            );

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
            vst1q_f32(
                dest_ptr,
                vcombine_f32(vcvt_f32_f64(store_4), vcvt_f32_f64(store_5)),
            );

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
            vst1q_f32(
                dest_ptr,
                vcombine_f32(vcvt_f32_f64(store_6), vcvt_f32_f64(store_7)),
            );

            filter_offset += filter_weights.aligned_size;
        }
    }
}
