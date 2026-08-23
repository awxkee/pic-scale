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

pub(crate) fn convolve_horizontal_plane_neon_row_one_f32_f64(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        let dst_width = filter_weights.bounds.len();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vdupq_n_f64(0.);

            while jx + 4 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let w0 = vld1q_f64(ptr);
                let w1 = vld1q_f64(ptr.add(2));
                let rgb_pixel0 = vld1q_f32(src.get_unchecked(bounds_start..).as_ptr());
                store = vfmaq_f64(store, vcvt_f64_f32(vget_low_f32(rgb_pixel0)), w0);
                store = vfmaq_f64(store, vcvt_high_f64_f32(rgb_pixel0), w1);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights0 = vld1q_f64(ptr);
                let rgb_pixel0 = vld1_f32(src.get_unchecked(bounds_start..).as_ptr());
                store = vfmaq_f64(store, vcvt_f64_f32(rgb_pixel0), weights0);
                jx += 2;
            }

            while jx < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vld1q_dup_f64(ptr);
                let rgb_pixel0 =
                    vld1_lane_f32::<0>(src.get_unchecked(bounds_start..).as_ptr(), vdup_n_f32(0.));
                store = vfmaq_f64(store, vcvt_f64_f32(rgb_pixel0), weight0);
                jx += 1;
            }

            let px = x;
            let dest_ptr = dst.get_unchecked_mut(px);
            *dest_ptr = vpaddd_f64(store) as f32;

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_plane_neon_rows_4_f32_f64(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let zeros = vdupq_n_f64(0.);
        let weights_ptr = filter_weights.weights.as_ptr();

        let dst_width = filter_weights.bounds.len();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            let src1 = src.get_unchecked(src_stride..);
            let src2 = src.get_unchecked(src_stride * 2..);
            let src3 = src.get_unchecked(src_stride * 3..);

            while jx + 4 <= bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let w0 = vld1q_f64(ptr);
                let w1 = vld1q_f64(ptr.add(2));
                let bounds_start = bounds.start + jx;

                let rgb_pixel0 = vld1q_f32(src.get_unchecked(bounds_start..).as_ptr());
                let rgb_pixel1 = vld1q_f32(src1.get_unchecked(bounds_start..).as_ptr());
                let rgb_pixel2 = vld1q_f32(src2.get_unchecked(bounds_start..).as_ptr());
                let rgb_pixel3 = vld1q_f32(src3.get_unchecked(bounds_start..).as_ptr());

                store_0 = vfmaq_f64(store_0, vcvt_f64_f32(vget_low_f32(rgb_pixel0)), w0);
                store_0 = vfmaq_f64(store_0, vcvt_high_f64_f32(rgb_pixel0), w1);

                store_1 = vfmaq_f64(store_1, vcvt_f64_f32(vget_low_f32(rgb_pixel1)), w0);
                store_1 = vfmaq_f64(store_1, vcvt_high_f64_f32(rgb_pixel1), w1);

                store_2 = vfmaq_f64(store_2, vcvt_f64_f32(vget_low_f32(rgb_pixel2)), w0);
                store_2 = vfmaq_f64(store_2, vcvt_high_f64_f32(rgb_pixel2), w1);

                store_3 = vfmaq_f64(store_3, vcvt_f64_f32(vget_low_f32(rgb_pixel3)), w0);
                store_3 = vfmaq_f64(store_3, vcvt_high_f64_f32(rgb_pixel3), w1);

                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights0 = vld1q_f64(ptr);
                let bounds_start = bounds.start + jx;

                let rgb_pixel0 = vld1_f32(src.get_unchecked(bounds_start..).as_ptr());
                let rgb_pixel1 = vld1_f32(src1.get_unchecked(bounds_start..).as_ptr());
                let rgb_pixel2 = vld1_f32(src2.get_unchecked(bounds_start..).as_ptr());
                let rgb_pixel3 = vld1_f32(src3.get_unchecked(bounds_start..).as_ptr());

                store_0 = vfmaq_f64(store_0, vcvt_f64_f32(rgb_pixel0), weights0);
                store_1 = vfmaq_f64(store_1, vcvt_f64_f32(rgb_pixel1), weights0);
                store_2 = vfmaq_f64(store_2, vcvt_f64_f32(rgb_pixel2), weights0);
                store_3 = vfmaq_f64(store_3, vcvt_f64_f32(rgb_pixel3), weights0);

                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vld1q_dup_f64(ptr);
                let bounds_start = bounds.start + jx;
                let rgb_pixel0 =
                    vld1_lane_f32::<0>(src.get_unchecked(bounds_start..).as_ptr(), vdup_n_f32(0.));
                let rgb_pixel1 =
                    vld1_lane_f32::<0>(src1.get_unchecked(bounds_start..).as_ptr(), vdup_n_f32(0.));
                let rgb_pixel2 =
                    vld1_lane_f32::<0>(src2.get_unchecked(bounds_start..).as_ptr(), vdup_n_f32(0.));
                let rgb_pixel3 =
                    vld1_lane_f32::<0>(src3.get_unchecked(bounds_start..).as_ptr(), vdup_n_f32(0.));
                store_0 = vfmaq_f64(store_0, vcvt_f64_f32(rgb_pixel0), weight0);
                store_1 = vfmaq_f64(store_1, vcvt_f64_f32(rgb_pixel1), weight0);
                store_2 = vfmaq_f64(store_2, vcvt_f64_f32(rgb_pixel2), weight0);
                store_3 = vfmaq_f64(store_3, vcvt_f64_f32(rgb_pixel3), weight0);
                jx += 1;
            }

            let px = x;
            let dest_ptr0 = dst.get_unchecked_mut(px);
            *dest_ptr0 = vpaddd_f64(store_0) as f32;

            let dest_ptr1 = dst.get_unchecked_mut(px + dst_stride);
            *dest_ptr1 = vpaddd_f64(store_1) as f32;

            let dest_ptr2 = dst.get_unchecked_mut(px + dst_stride * 2);
            *dest_ptr2 = vpaddd_f64(store_2) as f32;

            let dest_ptr3 = dst.get_unchecked_mut(px + dst_stride * 3);
            *dest_ptr3 = vpaddd_f64(store_3) as f32;

            filter_offset += filter_weights.aligned_size;
        }
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::PicScaleError;
    use crate::ResamplingFunction;
    use crate::convolve_naive_f32::{
        convolve_horizontal_4_row_f32_f64, convolve_horizontal_native_row_f32_f64,
    };
    use crate::math::WeightsGenerator;
    use crate::test_utils::{XorShiftRng, assert_f32_slices_close};

    /// See the identical helper in `src/avx2/plane_f32_f64.rs` - `PreferQuality`
    /// weights are `f64`, which `test_utils` doesn't build a helper for.
    fn make_row_filter_weights_f64(
        resampling: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<f64>, PicScaleError> {
        <f32 as WeightsGenerator<f64>>::make_weights(resampling, in_size, out_size)
    }

    const ATOL: f32 = 1e-5;

    #[test]
    fn neon_row_matches_scalar_reference() {
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64), (256, 256), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_f64(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xC0FFEE ^ (in_size as u64) << 32 ^ out_size as u64);
                let src = rng.fill_f32_unit(in_size);

                let mut dst_scalar = vec![0f32; out_size];
                let mut dst_neon = vec![0f32; out_size];
                convolve_horizontal_native_row_f32_f64::<1>(
                    &src,
                    &mut dst_scalar,
                    &filter_weights,
                    8,
                );
                convolve_horizontal_plane_neon_row_one_f32_f64(
                    &src,
                    &mut dst_neon,
                    &filter_weights,
                    8,
                );

                assert_f32_slices_close(
                    &dst_neon,
                    &dst_scalar,
                    ATOL,
                    &format!("{resampling:?} {in_size}->{out_size}: NEON plane f32/f64 single-row"),
                );
            }
        }
    }

    #[test]
    fn neon_rows_4_matches_scalar_reference() {
        const ROWS: usize = 4;
        for resampling in [ResamplingFunction::Bilinear, ResamplingFunction::Lanczos3] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64)] {
                let filter_weights =
                    make_row_filter_weights_f64(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xBADF00D ^ (in_size as u64) << 32 ^ out_size as u64);
                let src_stride = in_size;
                let dst_stride = out_size;
                let src = rng.fill_f32_unit(src_stride * ROWS);

                let mut dst_scalar = vec![0f32; dst_stride * ROWS];
                let mut dst_neon = vec![0f32; dst_stride * ROWS];
                convolve_horizontal_4_row_f32_f64::<1>(
                    &src,
                    src_stride,
                    &mut dst_scalar,
                    dst_stride,
                    &filter_weights,
                    8,
                );
                convolve_horizontal_plane_neon_rows_4_f32_f64(
                    &src,
                    src_stride,
                    &mut dst_neon,
                    dst_stride,
                    &filter_weights,
                    8,
                );

                assert_f32_slices_close(
                    &dst_neon,
                    &dst_scalar,
                    ATOL,
                    &format!("{resampling:?} {in_size}->{out_size}: NEON plane f32/f64 4-row"),
                );
            }
        }
    }
}
