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

use crate::avx2::utils::{_mm_prefer_fma_pd, _mm256_fma_pd};
use crate::filter_weights::FilterBounds;
use std::arch::x86_64::*;

pub(crate) fn convolve_vertical_avx_row_f32_f64_default(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weights: &[f64],
    _: u32,
) {
    unsafe {
        convolve_vertical_avx_row_f32_f64_regular(width, bounds, src, dst, src_stride, weights);
    }
}

pub(crate) fn convolve_vertical_avx_row_f32_f64_fma(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weights: &[f64],
    _: u32,
) {
    unsafe {
        convolve_vertical_avx_row_f32_f64_fma_impl(width, bounds, src, dst, src_stride, weights);
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_vertical_avx_row_f32_f64_regular(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weights: &[f64],
) {
    let unit = ExecutionUnit::<false>::default();
    unit.pass(width, bounds, src, dst, src_stride, weights);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_vertical_avx_row_f32_f64_fma_impl(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weights: &[f64],
) {
    let unit = ExecutionUnit::<true>::default();
    unit.pass(width, bounds, src, dst, src_stride, weights);
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit<FMA> {
    #[inline(always)]
    fn convolve_vertical_part_avx_8_f32(
        &self,
        start_y: usize,
        start_x: usize,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        filter: &[f64],
        bounds: &FilterBounds,
    ) {
        unsafe {
            let mut store_0 = _mm256_setzero_pd();
            let mut store_1 = _mm256_setzero_pd();

            let px = start_x;

            let mut j = 0usize;

            while j + 2 <= bounds.size {
                let py = start_y + j;
                let weights = _mm_loadu_pd(filter.get_unchecked(j..).as_ptr());
                let xw0 = _mm_shuffle_pd::<0>(weights, weights);
                let xw1 = _mm_shuffle_pd::<0b11>(weights, weights);
                let w0 = _mm256_setr_m128d(xw0, xw0);
                let w1 = _mm256_setr_m128d(xw1, xw1);
                let src_ptr = src.get_unchecked(src_stride * py + px..);
                let item_row_0 = _mm256_loadu_ps(src_ptr.as_ptr());

                store_0 = _mm256_fma_pd::<FMA>(
                    store_0,
                    _mm256_cvtps_pd(_mm256_castps256_ps128(item_row_0)),
                    w0,
                );
                store_1 = _mm256_fma_pd::<FMA>(
                    store_1,
                    _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(item_row_0)),
                    w0,
                );

                let item_row_0 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride..).as_ptr());

                store_0 = _mm256_fma_pd::<FMA>(
                    store_0,
                    _mm256_cvtps_pd(_mm256_castps256_ps128(item_row_0)),
                    w1,
                );
                store_1 = _mm256_fma_pd::<FMA>(
                    store_1,
                    _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(item_row_0)),
                    w1,
                );

                j += 2;
            }

            for j in j..bounds.size {
                let py = start_y + j;
                let weight = filter.get_unchecked(j);
                let v_weight = _mm256_broadcast_sd(weight);
                let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();
                let item_row_0 = _mm256_loadu_ps(src_ptr);

                store_0 = _mm256_fma_pd::<FMA>(
                    store_0,
                    _mm256_cvtps_pd(_mm256_castps256_ps128(item_row_0)),
                    v_weight,
                );
                store_1 = _mm256_fma_pd::<FMA>(
                    store_1,
                    _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(item_row_0)),
                    v_weight,
                );
            }

            let z0 = _mm256_cvtpd_ps(store_0);
            let z1 = _mm256_cvtpd_ps(store_1);

            let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            _mm256_storeu_ps(
                dst_ptr,
                _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(z0), z1),
            );
        }
    }

    #[inline(always)]
    fn convolve_vertical_part_avx_16_f32(
        &self,
        start_y: usize,
        start_x: usize,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        filter: &[f64],
        bounds: &FilterBounds,
    ) {
        unsafe {
            let mut store_0 = _mm256_setzero_pd();
            let mut store_1 = _mm256_setzero_pd();
            let mut store_2 = _mm256_setzero_pd();
            let mut store_3 = _mm256_setzero_pd();

            let px = start_x;

            for j in 0..bounds.size {
                let py = start_y + j;
                let weight = filter.get_unchecked(j);
                let v_weight = _mm256_broadcast_sd(weight);
                let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

                let item_row_0 = _mm256_loadu_ps(src_ptr);
                let item_row_1 = _mm256_loadu_ps(src_ptr.add(8));

                store_0 = _mm256_fma_pd::<FMA>(
                    store_0,
                    _mm256_cvtps_pd(_mm256_castps256_ps128(item_row_0)),
                    v_weight,
                );
                store_1 = _mm256_fma_pd::<FMA>(
                    store_1,
                    _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(item_row_0)),
                    v_weight,
                );

                store_2 = _mm256_fma_pd::<FMA>(
                    store_2,
                    _mm256_cvtps_pd(_mm256_castps256_ps128(item_row_1)),
                    v_weight,
                );
                store_3 = _mm256_fma_pd::<FMA>(
                    store_3,
                    _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(item_row_1)),
                    v_weight,
                );
            }

            let z0 = _mm256_cvtpd_ps(store_0);
            let z1 = _mm256_cvtpd_ps(store_1);
            let z2 = _mm256_cvtpd_ps(store_2);
            let z3 = _mm256_cvtpd_ps(store_3);

            let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            _mm256_storeu_ps(
                dst_ptr,
                _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(z0), z1),
            );
            _mm256_storeu_ps(
                dst_ptr.add(8),
                _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(z2), z3),
            );
        }
    }

    #[inline(always)]
    fn convolve_vertical_part_avx_f32(
        &self,
        start_y: usize,
        start_x: usize,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        filter: &[f64],
        bounds: &FilterBounds,
    ) {
        unsafe {
            let mut store_0 = _mm_setzero_pd();

            let px = start_x;

            for j in 0..bounds.size {
                let py = start_y + j;
                let weight = filter.get_unchecked(j..);
                let v_weight = _mm_load_sd(weight.as_ptr());
                let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

                let item_row_0 = _mm_load_ss(src_ptr);

                store_0 = _mm_prefer_fma_pd::<FMA>(store_0, _mm_cvtps_pd(item_row_0), v_weight);
            }

            let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            _mm_store_ss(dst_ptr, _mm_cvtpd_ps(store_0));
        }
    }

    #[inline(always)]
    fn pass(
        &self,
        _: usize,
        bounds: &FilterBounds,
        src: &[f32],
        dst: &mut [f32],
        src_stride: usize,
        weights: &[f64],
    ) {
        let mut cx = 0usize;
        let dst_width = dst.len();

        while cx + 16 <= dst_width {
            self.convolve_vertical_part_avx_16_f32(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weights,
                bounds,
            );

            cx += 16;
        }

        while cx + 8 <= dst_width {
            self.convolve_vertical_part_avx_8_f32(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weights,
                bounds,
            );

            cx += 8;
        }

        while cx < dst_width {
            self.convolve_vertical_part_avx_f32(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weights,
                bounds,
            );
            cx += 1;
        }
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::ResamplingFunction;
    use crate::floating_point_vertical::column_handler_floating_point;
    use crate::math::WeightsGenerator;
    use crate::test_utils::{XorShiftRng, assert_f32_slices_close};

    // This is the `PreferQuality` path: weights and the accumulator are both
    // `f64`, only narrowed back to `f32` on store. AVX2 still folds several
    // lanes per register and (for the FMA variant) fuses multiply+add, which
    // reorders relative to the scalar loop's strictly sequential
    // multiply-then-add - a small tolerance is expected, same reasoning as
    // the `f32`-weighted vertical comparison in `vertical_f32.rs`.
    const ATOL: f32 = 1e-5;

    #[test]
    fn avx2_f64_default_vertical_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        run_vertical_comparison(convolve_vertical_avx_row_f32_f64_default, "AVX2 f64 default");
    }

    #[test]
    fn avx2_f64_fma_vertical_matches_scalar_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        run_vertical_comparison(convolve_vertical_avx_row_f32_f64_fma, "AVX2 f64 FMA");
    }

    #[allow(clippy::type_complexity)]
    fn run_vertical_comparison(
        simd_fn: fn(usize, &FilterBounds, &[f32], &mut [f32], usize, &[f64], u32),
        label: &str,
    ) {
        // Row widths chosen to be multiples of 1 (plane), 3 (rgb) and 4 (rgba)
        // pixels, plus a couple of odd sizes so every SIMD tail-loop width
        // (16/8/1 lanes) gets exercised at least once.
        const ROW_WIDTHS: [usize; 7] = [1, 3, 4, 39, 99, 160, 41];

        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_height, out_height) in [(64usize, 37usize), (37, 64), (5, 3)] {
                let filter_weights =
                    <f32 as WeightsGenerator<f64>>::make_weights(resampling, in_height, out_height)
                        .unwrap();

                for &row_width in ROW_WIDTHS.iter() {
                    let src_stride = row_width;
                    let mut rng = XorShiftRng::new(
                        0xBADF00D
                            ^ (in_height as u64) << 32
                            ^ (out_height as u64) << 16
                            ^ row_width as u64,
                    );
                    let src = rng.fill_f32_unit(src_stride * in_height);

                    for y in 0..out_height {
                        let bounds = filter_weights.bounds[y];
                        let filter_offset = y * filter_weights.aligned_size;
                        let weights = &filter_weights.weights[filter_offset..];

                        let mut dst_scalar = vec![0f32; row_width];
                        let mut dst_simd = vec![0f32; row_width];

                        column_handler_floating_point::<f32, f64, f64>(
                            0,
                            &bounds,
                            &src,
                            &mut dst_scalar,
                            src_stride,
                            weights,
                            8,
                        );
                        simd_fn(row_width, &bounds, &src, &mut dst_simd, src_stride, weights, 8);

                        assert_f32_slices_close(
                            &dst_simd,
                            &dst_scalar,
                            ATOL,
                            &format!(
                                "{resampling:?} {in_height}->{out_height} row {y} width {row_width}: {label}"
                            ),
                        );
                    }
                }
            }
        }
    }
}
