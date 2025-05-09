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

pub(crate) fn convolve_horizontal_rgba_avx_rows_4_f32_f64<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgba_avx_rows_4_f32_f64_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        } else {
            convolve_horizontal_rgba_avx_rows_4_f32_f64_regular(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgba_avx_rows_4_f32_f64_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let unit = Row4ExecutionUnit::<false>::default();
        unit.pass(
            dst_width,
            src_width,
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgba_avx_rows_4_f32_f64_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let unit = Row4ExecutionUnit::<true>::default();
        unit.pass(
            dst_width,
            src_width,
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> Row4ExecutionUnit<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
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
            let weights_ptr = &filter_weights.weights;

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

                while jx + 2 < bounds.size {
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

                    jx += 2
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

pub(crate) fn convolve_horizontal_rgba_avx_row_one_f32_f64<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgba_avx_row_one_f32_f64_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        } else {
            convolve_horizontal_rgba_avx_row_one_f32_f64_regular(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_rgba_avx_row_one_f32_f64_regular(
    dst_width: usize,
    _: usize, // src_width
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        let unit = OneRowExecutionUnit::<false>::default();
        unit.pass(dst_width, filter_weights, src, dst);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn convolve_horizontal_rgba_avx_row_one_f32_f64_fma(
    dst_width: usize,
    _: usize, // src_width
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        let unit = OneRowExecutionUnit::<true>::default();
        unit.pass(dst_width, filter_weights, src, dst);
    }
}

#[derive(Copy, Clone, Default)]
struct OneRowExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> OneRowExecutionUnit<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        dst_width: usize,
        filter_weights: &FilterWeights<f64>,
        src: &[f32],
        dst: &mut [f32],
    ) {
        unsafe {
            const CN: usize = 4;
            let mut filter_offset = 0usize;
            let weights_ptr = &filter_weights.weights;

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store = _mm256_setzero_pd();

                while jx + 4 < bounds.size {
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

                while jx + 2 < bounds.size {
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
