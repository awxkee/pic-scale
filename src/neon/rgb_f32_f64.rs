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
use std::arch::aarch64::*;

macro_rules! store_rgb {
    ($store: expr, $dest_ptr: expr) => {{
        vst1_f32($dest_ptr, vget_low_f32($store));
        vst1q_lane_f32::<2>($dest_ptr.add(2), $store);
    }};
}

#[inline]
unsafe fn ld1x3_rgb(ptr: &[f32]) -> float32x4_t {
    unsafe {
        let rgb_pixel = vcombine_u64(vld1_u64(ptr.as_ptr() as *const _), vdup_n_u64(0));
        vld1q_lane_f32::<0>(
            ptr.get_unchecked(2..).as_ptr(),
            vreinterpretq_f32_u64(rgb_pixel),
        )
    }
}

#[inline]
unsafe fn ld2x3_rgb(ptr: &[f32]) -> float32x4x2_t {
    unsafe {
        let l0 = vld1q_f32(ptr.as_ptr());
        let l1 = vreinterpretq_f32_u64(vcombine_u64(
            vld1_u64(ptr.get_unchecked(4..).as_ptr() as *const _),
            vdup_n_u64(0),
        ));
        let r0 = vextq_f32::<3>(l0, l1);
        float32x4x2_t(l0, r0)
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_rows_4_f32_f64(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        const CN: usize = 3;
        let mut filter_offset = 0usize;

        let zeros = vdupq_n_f64(0.);

        let weights_ptr = filter_weights.weights.as_ptr();

        let src1 = src.get_unchecked(src_stride..);
        let src2 = src.get_unchecked(src_stride * 3..);
        let src3 = src.get_unchecked(src_stride * 3..);

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
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let w0 = vld1q_f64(ptr);

                let px0 = ld2x3_rgb(src.get_unchecked(bounds_start * CN..));
                let px1 = ld2x3_rgb(src1.get_unchecked(bounds_start * CN..));
                let px2 = ld2x3_rgb(src2.get_unchecked(bounds_start * CN..));
                let px3 = ld2x3_rgb(src3.get_unchecked(bounds_start * CN..));

                store_0 = vfmaq_laneq_f64::<0>(store_0, vcvt_f64_f32(vget_low_f32(px0.0)), w0);
                store_1 = vfmaq_laneq_f64::<0>(store_1, vcvt_high_f64_f32(px0.0), w0);

                store_2 = vfmaq_laneq_f64::<0>(store_2, vcvt_f64_f32(vget_low_f32(px1.0)), w0);
                store_3 = vfmaq_laneq_f64::<0>(store_3, vcvt_high_f64_f32(px1.0), w0);

                store_4 = vfmaq_laneq_f64::<0>(store_4, vcvt_f64_f32(vget_low_f32(px2.0)), w0);
                store_5 = vfmaq_laneq_f64::<0>(store_5, vcvt_high_f64_f32(px2.0), w0);

                store_6 = vfmaq_laneq_f64::<0>(store_6, vcvt_f64_f32(vget_low_f32(px3.0)), w0);
                store_7 = vfmaq_laneq_f64::<0>(store_7, vcvt_high_f64_f32(px3.0), w0);

                store_0 = vfmaq_laneq_f64::<1>(store_0, vcvt_f64_f32(vget_low_f32(px0.1)), w0);
                store_1 = vfmaq_laneq_f64::<1>(store_1, vcvt_high_f64_f32(px0.1), w0);

                store_2 = vfmaq_laneq_f64::<1>(store_2, vcvt_f64_f32(vget_low_f32(px1.1)), w0);
                store_3 = vfmaq_laneq_f64::<1>(store_3, vcvt_high_f64_f32(px1.1), w0);

                store_4 = vfmaq_laneq_f64::<1>(store_4, vcvt_f64_f32(vget_low_f32(px2.1)), w0);
                store_5 = vfmaq_laneq_f64::<1>(store_5, vcvt_high_f64_f32(px2.1), w0);

                store_6 = vfmaq_laneq_f64::<1>(store_6, vcvt_f64_f32(vget_low_f32(px3.1)), w0);
                store_7 = vfmaq_laneq_f64::<1>(store_7, vcvt_high_f64_f32(px3.1), w0);

                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vld1q_dup_f64(ptr);
                let px0 = ld1x3_rgb(src.get_unchecked(bounds_start * CN..));
                let px1 = ld1x3_rgb(src1.get_unchecked(bounds_start * CN..));
                let px2 = ld1x3_rgb(src2.get_unchecked(bounds_start * CN..));
                let px3 = ld1x3_rgb(src3.get_unchecked(bounds_start * CN..));

                store_0 = vfmaq_f64(store_0, vcvt_f64_f32(vget_low_f32(px0)), weight0);
                store_1 = vfmaq_f64(store_1, vcvt_high_f64_f32(px0), weight0);

                store_2 = vfmaq_f64(store_2, vcvt_f64_f32(vget_low_f32(px1)), weight0);
                store_3 = vfmaq_f64(store_3, vcvt_high_f64_f32(px1), weight0);

                store_4 = vfmaq_f64(store_4, vcvt_f64_f32(vget_low_f32(px2)), weight0);
                store_5 = vfmaq_f64(store_5, vcvt_high_f64_f32(px2), weight0);

                store_6 = vfmaq_f64(store_6, vcvt_f64_f32(vget_low_f32(px3)), weight0);
                store_7 = vfmaq_f64(store_7, vcvt_high_f64_f32(px3), weight0);
                jx += 1;
            }

            let px = x * CN;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            store_rgb!(
                vcombine_f32(vcvt_f32_f64(store_0), vcvt_f32_f64(store_1)),
                dest_ptr
            );

            let dest_ptr_1 = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
            store_rgb!(
                vcombine_f32(vcvt_f32_f64(store_2), vcvt_f32_f64(store_3)),
                dest_ptr_1
            );

            let dest_ptr_2 = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
            store_rgb!(
                vcombine_f32(vcvt_f32_f64(store_3), vcvt_f32_f64(store_4)),
                dest_ptr_2
            );

            let dest_ptr_3 = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
            store_rgb!(
                vcombine_f32(vcvt_f32_f64(store_5), vcvt_f32_f64(store_6)),
                dest_ptr_3
            );

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_row_one_f32_f64(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        const CN: usize = 3;
        let weights_ptr = filter_weights.weights.as_ptr();
        let mut filter_offset = 0usize;

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store0 = vdupq_n_f64(0.);
            let mut store1 = vdupq_n_f64(0.);

            while jx + 2 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let w0 = vld1q_f64(ptr);
                let px = ld2x3_rgb(src.get_unchecked(bounds_start * CN..));
                store0 = vfmaq_laneq_f64::<0>(store0, vcvt_f64_f32(vget_low_f32(px.0)), w0);
                store1 = vfmaq_laneq_f64::<0>(store1, vcvt_high_f64_f32(px.0), w0);

                store0 = vfmaq_laneq_f64::<1>(store0, vcvt_f64_f32(vget_low_f32(px.1)), w0);
                store1 = vfmaq_laneq_f64::<1>(store1, vcvt_high_f64_f32(px.1), w0);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vld1q_dup_f64(ptr);
                let bounds_start = bounds.start + jx;
                let px = ld1x3_rgb(src.get_unchecked(bounds_start * CN..));
                store0 = vfmaq_f64(store0, vcvt_f64_f32(vget_low_f32(px)), weight0);
                store1 = vfmaq_f64(store1, vcvt_high_f64_f32(px), weight0);
                jx += 1;
            }

            let px = x * CN;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            store_rgb!(
                vcombine_f32(vcvt_f32_f64(store0), vcvt_f32_f64(store1)),
                dest_ptr
            );

            filter_offset += filter_weights.aligned_size;
        }
    }
}
