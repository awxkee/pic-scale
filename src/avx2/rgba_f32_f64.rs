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

use crate::avx2::utils::_mm256_fma_pd;
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

pub(crate) fn convolve_horizontal_rgba_avx_rows_4_f32_f64_default(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgba_avx_rows_4_f32_f64_regular(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_rows_4_f32_f64_fma(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgba_avx_rows_4_f32_f64_fma_impl(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_horizontal_rgba_avx_rows_4_f32_f64_regular(
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let unit = Row4ExecutionUnit::<false>::default();
    unit.pass(filter_weights, src, src_stride, dst, dst_stride);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_horizontal_rgba_avx_rows_4_f32_f64_fma_impl(
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let unit = Row4ExecutionUnit::<true>::default();
    unit.pass(filter_weights, src, src_stride, dst, dst_stride);
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> Row4ExecutionUnit<FMA> {
    #[inline(always)]
    fn pass(
        &self,
        filter_weights: &FilterWeights<f64>,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        dst_stride: usize,
    ) {
        unsafe {
            const CN: usize = 4;
            let mut filter_offset = 0usize;
            let weights_ptr = &filter_weights.weights;

            let dst_width = filter_weights.bounds.len();

            let src1 = src.get_unchecked(src_stride..);
            let src2 = src.get_unchecked(src_stride * 2..);
            let src3 = src.get_unchecked(src_stride * 3..);

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store_0 = _mm256_setzero_pd();
                let mut store_1 = _mm256_setzero_pd();
                let mut store_2 = _mm256_setzero_pd();
                let mut store_3 = _mm256_setzero_pd();

                while jx + 2 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let w0 = _mm256_set1_pd(*ptr.get_unchecked(0));
                    let w1 = _mm256_set1_pd(*ptr.get_unchecked(1));
                    let filter_start = jx + bounds.start;

                    let rgb_pixel0 =
                        _mm256_loadu_ps(src.get_unchecked(filter_start * CN..).as_ptr());
                    let rgb_pixel1 =
                        _mm256_loadu_ps(src1.get_unchecked(filter_start * CN..).as_ptr());
                    let rgb_pixel2 =
                        _mm256_loadu_ps(src2.get_unchecked(filter_start * CN..).as_ptr());
                    let rgb_pixel3 =
                        _mm256_loadu_ps(src3.get_unchecked(filter_start * CN..).as_ptr());

                    store_0 = _mm256_fma_pd::<FMA>(
                        store_0,
                        _mm256_cvtps_pd(_mm256_castps256_ps128(rgb_pixel0)),
                        w0,
                    );
                    store_1 = _mm256_fma_pd::<FMA>(
                        store_1,
                        _mm256_cvtps_pd(_mm256_castps256_ps128(rgb_pixel1)),
                        w0,
                    );
                    store_2 = _mm256_fma_pd::<FMA>(
                        store_2,
                        _mm256_cvtps_pd(_mm256_castps256_ps128(rgb_pixel2)),
                        w0,
                    );
                    store_3 = _mm256_fma_pd::<FMA>(
                        store_3,
                        _mm256_cvtps_pd(_mm256_castps256_ps128(rgb_pixel3)),
                        w0,
                    );

                    store_0 = _mm256_fma_pd::<FMA>(
                        store_0,
                        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(rgb_pixel0)),
                        w1,
                    );
                    store_1 = _mm256_fma_pd::<FMA>(
                        store_1,
                        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(rgb_pixel1)),
                        w1,
                    );
                    store_2 = _mm256_fma_pd::<FMA>(
                        store_2,
                        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(rgb_pixel2)),
                        w1,
                    );
                    store_3 = _mm256_fma_pd::<FMA>(
                        store_3,
                        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(rgb_pixel3)),
                        w1,
                    );

                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset);
                    let filter_start = jx + bounds.start;
                    let weight0 = _mm256_set1_pd(*ptr);

                    let rgb_pixel0 = _mm_loadu_ps(src.get_unchecked(filter_start * CN..).as_ptr());
                    let rgb_pixel1 = _mm_loadu_ps(src1.get_unchecked(filter_start * CN..).as_ptr());
                    let rgb_pixel2 = _mm_loadu_ps(src2.get_unchecked(filter_start * CN..).as_ptr());
                    let rgb_pixel3 = _mm_loadu_ps(src3.get_unchecked(filter_start * CN..).as_ptr());

                    store_0 = _mm256_fma_pd::<FMA>(store_0, _mm256_cvtps_pd(rgb_pixel0), weight0);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, _mm256_cvtps_pd(rgb_pixel1), weight0);
                    store_2 = _mm256_fma_pd::<FMA>(store_2, _mm256_cvtps_pd(rgb_pixel2), weight0);
                    store_3 = _mm256_fma_pd::<FMA>(store_3, _mm256_cvtps_pd(rgb_pixel3), weight0);
                    jx += 1;
                }

                let px = x * CN;
                let dest_ptr = dst.get_unchecked_mut(px..);
                _mm_storeu_ps(dest_ptr.as_mut_ptr(), _mm256_cvtpd_ps(store_0));

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride..);
                _mm_storeu_ps(dest_ptr.as_mut_ptr(), _mm256_cvtpd_ps(store_1));

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2..);
                _mm_storeu_ps(dest_ptr.as_mut_ptr(), _mm256_cvtpd_ps(store_2));

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3..);
                _mm_storeu_ps(dest_ptr.as_mut_ptr(), _mm256_cvtpd_ps(store_3));

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_row_one_f32_f64_default(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgba_avx_row_one_f32_f64_regular(filter_weights, src, dst);
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_row_one_f32_f64_fma(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgba_avx_row_one_f32_f64_fma_impl(filter_weights, src, dst);
    }
}

#[target_feature(enable = "avx2")]
fn convolve_horizontal_rgba_avx_row_one_f32_f64_regular(
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = OneRowExecutionUnit::<false>::default();
    unit.pass(filter_weights, src, dst);
}

#[target_feature(enable = "avx2", enable = "fma")]
fn convolve_horizontal_rgba_avx_row_one_f32_f64_fma_impl(
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = OneRowExecutionUnit::<true>::default();
    unit.pass(filter_weights, src, dst);
}

#[derive(Copy, Clone, Default)]
struct OneRowExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> OneRowExecutionUnit<FMA> {
    #[inline(always)]
    fn pass(&self, filter_weights: &FilterWeights<f64>, src: &[f32], dst: &mut [f32]) {
        unsafe {
            const CN: usize = 4;
            let mut filter_offset = 0usize;
            let weights_ptr = &filter_weights.weights;

            let dst_width = filter_weights.bounds.len();

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store = _mm256_setzero_pd();

                while jx + 4 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);

                    let w0 = _mm256_set1_pd(*ptr.get_unchecked(0));
                    let w1 = _mm256_set1_pd(*ptr.get_unchecked(1));
                    let w2 = _mm256_set1_pd(*ptr.get_unchecked(2));
                    let w3 = _mm256_set1_pd(*ptr.get_unchecked(3));

                    let filter_start = jx + bounds.start;

                    let src_ptr = src.get_unchecked(filter_start * CN..).as_ptr();

                    let rgb_pixel_0 = _mm256_loadu_ps(src_ptr);
                    let rgb_pixel_1 = _mm256_loadu_ps(src_ptr.add(8));

                    store = _mm256_fma_pd::<FMA>(
                        store,
                        _mm256_cvtps_pd(_mm256_castps256_ps128(rgb_pixel_0)),
                        w0,
                    );
                    store = _mm256_fma_pd::<FMA>(
                        store,
                        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(rgb_pixel_0)),
                        w1,
                    );

                    store = _mm256_fma_pd::<FMA>(
                        store,
                        _mm256_cvtps_pd(_mm256_castps256_ps128(rgb_pixel_1)),
                        w2,
                    );
                    store = _mm256_fma_pd::<FMA>(
                        store,
                        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(rgb_pixel_1)),
                        w3,
                    );
                    jx += 4;
                }

                while jx + 2 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm256_set1_pd(*ptr.get_unchecked(0));
                    let weight1 = _mm256_set1_pd(*ptr.get_unchecked(1));
                    let filter_start = jx + bounds.start;

                    let src_ptr = src.get_unchecked(filter_start * CN..);

                    let rgb_pixel = _mm256_loadu_ps(src_ptr.as_ptr());

                    store = _mm256_fma_pd::<FMA>(
                        store,
                        _mm256_cvtps_pd(_mm256_castps256_ps128(rgb_pixel)),
                        weight0,
                    );
                    store = _mm256_fma_pd::<FMA>(
                        store,
                        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(rgb_pixel)),
                        weight1,
                    );
                    jx += 2
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset);
                    let weight0 = _mm256_set1_pd(*ptr);
                    let filter_start = jx + bounds.start;

                    let rgb_pixel = _mm_loadu_ps(src.get_unchecked(filter_start * CN..).as_ptr());
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(rgb_pixel), weight0);
                    jx += 1;
                }

                let px = x * CN;
                let dest_ptr = dst.get_unchecked_mut(px..);
                _mm_storeu_ps(dest_ptr.as_mut_ptr(), _mm256_cvtpd_ps(store));

                filter_offset += filter_weights.aligned_size;
            }
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
    fn avx2_default_row_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        run_row_comparison(convolve_horizontal_rgba_avx_row_one_f32_f64_default, "AVX2 default");
    }

    #[test]
    fn avx2_fma_row_matches_scalar_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        run_row_comparison(convolve_horizontal_rgba_avx_row_one_f32_f64_fma, "AVX2 FMA");
    }

    fn run_row_comparison(
        simd_fn: fn(&[f32], &mut [f32], &FilterWeights<f64>, u32),
        label: &str,
    ) {
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64), (256, 256), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_f64(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xC0FFEE ^ (in_size as u64) << 32 ^ out_size as u64);
                let src = rng.fill_f32_unit(in_size * 4);

                let mut dst_scalar = vec![0f32; out_size * 4];
                let mut dst_simd = vec![0f32; out_size * 4];
                convolve_horizontal_native_row_f32_f64::<4>(
                    &src,
                    &mut dst_scalar,
                    &filter_weights,
                    8,
                );
                simd_fn(&src, &mut dst_simd, &filter_weights, 8);

                assert_f32_slices_close(
                    &dst_simd,
                    &dst_scalar,
                    ATOL,
                    &format!("{resampling:?} {in_size}->{out_size}: {label} single-row"),
                );
            }
        }
    }

    #[test]
    fn avx2_default_rows_4_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        run_rows_4_comparison(convolve_horizontal_rgba_avx_rows_4_f32_f64_default, "AVX2 default");
    }

    #[test]
    fn avx2_fma_rows_4_matches_scalar_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        run_rows_4_comparison(convolve_horizontal_rgba_avx_rows_4_f32_f64_fma, "AVX2 FMA");
    }

    fn run_rows_4_comparison(
        simd_fn: fn(&[f32], usize, &mut [f32], usize, &FilterWeights<f64>, u32),
        label: &str,
    ) {
        const ROWS: usize = 4;
        for resampling in [ResamplingFunction::Bilinear, ResamplingFunction::Lanczos3] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64)] {
                let filter_weights =
                    make_row_filter_weights_f64(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xBADF00D ^ (in_size as u64) << 32 ^ out_size as u64);
                let src_stride = in_size * 4;
                let dst_stride = out_size * 4;
                let src = rng.fill_f32_unit(src_stride * ROWS);

                let mut dst_scalar = vec![0f32; dst_stride * ROWS];
                let mut dst_simd = vec![0f32; dst_stride * ROWS];
                convolve_horizontal_4_row_f32_f64::<4>(
                    &src,
                    src_stride,
                    &mut dst_scalar,
                    dst_stride,
                    &filter_weights,
                    8,
                );
                simd_fn(&src, src_stride, &mut dst_simd, dst_stride, &filter_weights, 8);

                assert_f32_slices_close(
                    &dst_simd,
                    &dst_scalar,
                    ATOL,
                    &format!("{resampling:?} {in_size}->{out_size}: {label} 4-row"),
                );
            }
        }
    }
}
