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

use crate::avx2::utils::{_mm_fma_pd, _mm_hsum_pd, _mm256_fma_pd};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

pub(crate) fn convolve_hor_plane_avx_row_one_f32_f64_default(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        convolve_hor_plane_avx_row_one_regular_f32_f64(filter_weights, src, dst);
    }
}

pub(crate) fn convolve_hor_plane_avx_row_one_f32_f64_fma(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        convolve_hor_plane_avx_row_one_fma_f32_f64(filter_weights, src, dst);
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_hor_plane_avx_row_one_regular_f32_f64(
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = Row1ExecutorUnit::<false>::default();
    unit.pass(filter_weights, src, dst);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_hor_plane_avx_row_one_fma_f32_f64(
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = Row1ExecutorUnit::<true>::default();
    unit.pass(filter_weights, src, dst);
}

#[derive(Copy, Clone, Default)]
struct Row1ExecutorUnit<const FMA: bool> {}

impl<const FMA: bool> Row1ExecutorUnit<FMA> {
    #[inline(always)]
    fn pass(&self, filter_weights: &FilterWeights<f64>, src: &[f32], dst: &mut [f32]) {
        unsafe {
            let mut filter_offset = 0usize;
            let weights_ptr = &filter_weights.weights;

            let dst_width = filter_weights.bounds.len();

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store = _mm256_setzero_pd();

                while jx + 4 <= bounds.size {
                    let bounds_start = bounds.start + jx;
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let w0 = _mm256_loadu_pd(ptr.as_ptr());
                    let px0 = _mm_loadu_ps(src.get_unchecked(bounds_start..).as_ptr().cast());
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(px0), w0);
                    jx += 4;
                }

                let mut store = _mm_add_pd(
                    _mm256_castpd256_pd128(store),
                    _mm256_extractf128_pd::<1>(store),
                );

                while jx + 2 <= bounds.size {
                    let bounds_start = bounds.start + jx;
                    let w = weights_ptr.get_unchecked(jx + filter_offset..);
                    let w0 = _mm_loadu_pd(w.as_ptr().cast());
                    let px0 = _mm_castsi128_ps(_mm_loadu_si64(
                        src.get_unchecked(bounds_start..).as_ptr().cast(),
                    ));
                    store = _mm_fma_pd::<FMA>(store, _mm_cvtps_pd(px0), w0);
                    jx += 2;
                }

                while jx < bounds.size {
                    let bounds_start = bounds.start + jx;
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm_load_sd(ptr.as_ptr());
                    let px0 = _mm_load_ss(src.get_unchecked(bounds_start..).as_ptr());
                    store = _mm_fma_pd::<FMA>(store, _mm_cvtps_pd(px0), weight0);
                    jx += 1;
                }

                let px = x;
                let dest_ptr = dst.get_unchecked_mut(px);
                _mm_store_ss(dest_ptr, _mm_cvtpd_ps(_mm_hsum_pd(store)));

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}

pub(crate) fn convolve_hor_plane_avx_rows_4_f32_f64_default(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_plane_avx_rows_4_regular_f32_f64(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

pub(crate) fn convolve_hor_plane_avx_rows_4_f32_f64_fma(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f64>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_plane_avx_rows_4_fma_f32_f64(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_horizontal_plane_avx_rows_4_regular_f32_f64(
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
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_horizontal_plane_avx_rows_4_fma_f32_f64(
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
            let mut filter_offset = 0usize;
            let weights_ptr = &filter_weights.weights;

            let dst_width = filter_weights.bounds.len();

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store_0 = _mm256_setzero_pd();
                let mut store_1 = _mm256_setzero_pd();
                let mut store_2 = _mm256_setzero_pd();
                let mut store_3 = _mm256_setzero_pd();

                let src1 = src.get_unchecked(src_stride..);
                let src2 = src.get_unchecked(src_stride * 2..);
                let src3 = src.get_unchecked(src_stride * 3..);

                while jx + 4 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);

                    let bounds_start = bounds.start + jx;
                    let w0 = _mm256_loadu_pd(ptr.as_ptr());

                    let px0 = _mm_loadu_ps(src.get_unchecked(bounds_start..).as_ptr().cast());
                    let px1 = _mm_loadu_ps(src1.get_unchecked(bounds_start..).as_ptr().cast());
                    let px2 = _mm_loadu_ps(src2.get_unchecked(bounds_start..).as_ptr().cast());
                    let px3 = _mm_loadu_ps(src3.get_unchecked(bounds_start..).as_ptr().cast());

                    store_0 = _mm256_fma_pd::<FMA>(store_0, _mm256_cvtps_pd(px0), w0);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, _mm256_cvtps_pd(px1), w0);
                    store_2 = _mm256_fma_pd::<FMA>(store_2, _mm256_cvtps_pd(px2), w0);
                    store_3 = _mm256_fma_pd::<FMA>(store_3, _mm256_cvtps_pd(px3), w0);

                    jx += 4;
                }

                let mut store_0 = _mm256_add_pd(
                    _mm256_castsi256_pd(_mm256_permute2x128_si256::<0x20>(
                        _mm256_castpd_si256(store_0),
                        _mm256_castpd_si256(store_1),
                    )),
                    _mm256_castsi256_pd(_mm256_permute2x128_si256::<0x31>(
                        _mm256_castpd_si256(store_0),
                        _mm256_castpd_si256(store_1),
                    )),
                );

                let mut store_1 = _mm256_add_pd(
                    _mm256_castsi256_pd(_mm256_permute2x128_si256::<0x20>(
                        _mm256_castpd_si256(store_2),
                        _mm256_castpd_si256(store_3),
                    )),
                    _mm256_castsi256_pd(_mm256_permute2x128_si256::<0x31>(
                        _mm256_castpd_si256(store_2),
                        _mm256_castpd_si256(store_3),
                    )),
                );

                while jx + 2 <= bounds.size {
                    let w = weights_ptr.get_unchecked(jx + filter_offset..);
                    let bounds_start = bounds.start + jx;
                    let wh = _mm_loadu_pd(w.as_ptr().cast());

                    let w0 = _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(wh), wh);

                    let px0 = _mm_castsi128_ps(_mm_loadu_si64(
                        src.get_unchecked(bounds_start..).as_ptr().cast(),
                    ));
                    let px1 = _mm_castsi128_ps(_mm_loadu_si64(
                        src1.get_unchecked(bounds_start..).as_ptr().cast(),
                    ));
                    let px2 = _mm_castsi128_ps(_mm_loadu_si64(
                        src2.get_unchecked(bounds_start..).as_ptr().cast(),
                    ));
                    let px3 = _mm_castsi128_ps(_mm_loadu_si64(
                        src3.get_unchecked(bounds_start..).as_ptr().cast(),
                    ));

                    let px01 = _mm256_cvtps_pd(_mm_movelh_ps(px0, px1));
                    let px23 = _mm256_cvtps_pd(_mm_movelh_ps(px2, px3));

                    store_0 = _mm256_fma_pd::<FMA>(store_0, px01, w0);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, px23, w0);
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let wh = _mm_load_sd(ptr.as_ptr());
                    let w0 = _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(wh), wh);

                    let bounds_start = bounds.start + jx;

                    let px0 = _mm_load_ss(src.get_unchecked(bounds_start..).as_ptr());
                    let px1 = _mm_load_ss(src1.get_unchecked(bounds_start..).as_ptr());
                    let px2 = _mm_load_ss(src2.get_unchecked(bounds_start..).as_ptr());
                    let px3 = _mm_load_ss(src3.get_unchecked(bounds_start..).as_ptr());

                    let px01 = _mm256_cvtps_pd(_mm_movelh_ps(px0, px1));
                    let px23 = _mm256_cvtps_pd(_mm_movelh_ps(px2, px3));

                    store_0 = _mm256_fma_pd::<FMA>(store_0, px01, w0);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, px23, w0);

                    jx += 1;
                }

                let px = x;
                let dest_ptr = dst.get_unchecked_mut(px);
                _mm_store_ss(
                    dest_ptr,
                    _mm_cvtpd_ps(_mm_hsum_pd(_mm256_castpd256_pd128(store_0))),
                );

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride);
                _mm_store_ss(
                    dest_ptr,
                    _mm_cvtpd_ps(_mm_hsum_pd(_mm256_extractf128_pd::<1>(store_0))),
                );

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2);
                _mm_store_ss(
                    dest_ptr,
                    _mm_cvtpd_ps(_mm_hsum_pd(_mm256_castpd256_pd128(store_1))),
                );

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3);
                _mm_store_ss(
                    dest_ptr,
                    _mm_cvtpd_ps(_mm_hsum_pd(_mm256_extractf128_pd::<1>(store_1))),
                );

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

    /// `PreferQuality` weights are computed in `f64` (see
    /// `HorizontalFilterPass<f32, f64, CN>` in `src/factory/*_f32.rs`), which
    /// `test_utils::make_row_filter_weights_f32` doesn't produce - it only
    /// builds the `f32`-weighted path. Mirror it here with the `f64` generator.
    fn make_row_filter_weights_f64(
        resampling: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<f64>, PicScaleError> {
        <f32 as WeightsGenerator<f64>>::make_weights(resampling, in_size, out_size)
    }

    // f64 weights remove the weight-quantization error the f32/f32 path has, but
    // SIMD reduction-order still differs from the scalar accumulation order, so
    // this isn't bit-exact either - a slightly looser tolerance than the f32/f32
    // ATOL is warranted here.
    const ATOL: f32 = 1e-5;

    #[test]
    fn avx2_default_row_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        run_row_comparison(convolve_hor_plane_avx_row_one_f32_f64_default, "AVX2 default");
    }

    #[test]
    fn avx2_fma_row_matches_scalar_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        run_row_comparison(convolve_hor_plane_avx_row_one_f32_f64_fma, "AVX2 FMA");
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
                let src = rng.fill_f32_unit(in_size);

                let mut dst_scalar = vec![0f32; out_size];
                let mut dst_simd = vec![0f32; out_size];
                convolve_horizontal_native_row_f32_f64::<1>(
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
        run_rows_4_comparison(convolve_hor_plane_avx_rows_4_f32_f64_default, "AVX2 default");
    }

    #[test]
    fn avx2_fma_rows_4_matches_scalar_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        run_rows_4_comparison(convolve_hor_plane_avx_rows_4_f32_f64_fma, "AVX2 FMA");
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
                let src_stride = in_size;
                let dst_stride = out_size;
                let src = rng.fill_f32_unit(src_stride * ROWS);

                let mut dst_scalar = vec![0f32; dst_stride * ROWS];
                let mut dst_simd = vec![0f32; dst_stride * ROWS];
                convolve_horizontal_4_row_f32_f64::<1>(
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
