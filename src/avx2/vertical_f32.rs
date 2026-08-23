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

use crate::avx2::utils::{_mm_prefer_fma_ps, _mm256_fma_ps, shuffle};
use crate::filter_weights::FilterBounds;
use std::arch::x86_64::*;

pub(crate) fn convolve_vertical_avx_row_default_f32(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
    _: u32,
) {
    unsafe {
        convolve_vertical_avx_row_f32_regular(width, bounds, src, dst, src_stride, weight_ptr);
    }
}

pub(crate) fn convolve_vertical_avx_row_fma_f32(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
    _: u32,
) {
    unsafe {
        convolve_vertical_avx_row_f32_fma(width, bounds, src, dst, src_stride, weight_ptr);
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_vertical_avx_row_f32_regular(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    let unit = ExecutionUnit::<false>::default();
    unit.pass(width, bounds, src, dst, src_stride, weight_ptr);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_vertical_avx_row_f32_fma(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    let unit = ExecutionUnit::<true>::default();
    unit.pass(width, bounds, src, dst, src_stride, weight_ptr);
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit<FMA> {
    #[inline(always)]
    fn convolve_vertical_part_avx_32_f32(
        &self,
        start_y: usize,
        start_x: usize,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        filter: &[f32],
        bounds: &FilterBounds,
    ) {
        unsafe {
            let mut store_0 = _mm256_setzero_ps();
            let mut store_1 = _mm256_setzero_ps();
            let mut store_2 = _mm256_setzero_ps();
            let mut store_3 = _mm256_setzero_ps();

            let px = start_x;

            let mut j = 0usize;

            while j + 4 <= bounds.size {
                let py = start_y + j;
                let weights = _mm_loadu_ps(filter.get_unchecked(j..).as_ptr());

                let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);
                let xw2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(weights, weights);
                let xw3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(weights, weights);

                let w0 = _mm256_setr_m128(xw0, xw0);
                let w1 = _mm256_setr_m128(xw1, xw1);
                let w2 = _mm256_setr_m128(xw2, xw2);
                let w3 = _mm256_setr_m128(xw3, xw3);

                let src_ptr = src.get_unchecked(src_stride * py + px..);

                let item_row_0 = _mm256_loadu_ps(src_ptr.as_ptr());
                let item_row_1 = _mm256_loadu_ps(src_ptr.get_unchecked(8..).as_ptr());
                let item_row_2 = _mm256_loadu_ps(src_ptr.get_unchecked(16..).as_ptr());
                let item_row_3 = _mm256_loadu_ps(src_ptr.get_unchecked(24..).as_ptr());

                store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, w0);
                store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, w0);
                store_2 = _mm256_fma_ps::<FMA>(store_2, item_row_2, w0);
                store_3 = _mm256_fma_ps::<FMA>(store_3, item_row_3, w0);

                let item_row_0 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride..).as_ptr());
                let item_row_1 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride + 8..).as_ptr());
                let item_row_2 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride + 16..).as_ptr());
                let item_row_3 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride + 24..).as_ptr());

                store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, w1);
                store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, w1);
                store_2 = _mm256_fma_ps::<FMA>(store_2, item_row_2, w1);
                store_3 = _mm256_fma_ps::<FMA>(store_3, item_row_3, w1);

                let item_row_0 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride * 2..).as_ptr());
                let item_row_1 =
                    _mm256_loadu_ps(src_ptr.get_unchecked(src_stride * 2 + 8..).as_ptr());
                let item_row_2 =
                    _mm256_loadu_ps(src_ptr.get_unchecked(src_stride * 2 + 16..).as_ptr());
                let item_row_3 =
                    _mm256_loadu_ps(src_ptr.get_unchecked(src_stride * 2 + 24..).as_ptr());

                store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, w2);
                store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, w2);
                store_2 = _mm256_fma_ps::<FMA>(store_2, item_row_2, w2);
                store_3 = _mm256_fma_ps::<FMA>(store_3, item_row_3, w2);

                let item_row_0 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride * 3..).as_ptr());
                let item_row_1 =
                    _mm256_loadu_ps(src_ptr.get_unchecked(src_stride * 3 + 8..).as_ptr());
                let item_row_2 =
                    _mm256_loadu_ps(src_ptr.get_unchecked(src_stride * 3 + 16..).as_ptr());
                let item_row_3 =
                    _mm256_loadu_ps(src_ptr.get_unchecked(src_stride * 3 + 24..).as_ptr());

                store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, w3);
                store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, w3);
                store_2 = _mm256_fma_ps::<FMA>(store_2, item_row_2, w3);
                store_3 = _mm256_fma_ps::<FMA>(store_3, item_row_3, w3);

                j += 4;
            }

            while j + 2 <= bounds.size {
                let py = start_y + j;
                let weights =
                    _mm_castsi128_ps(_mm_loadu_si64(filter.get_unchecked(j..).as_ptr().cast()));

                let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);

                let w0 = _mm256_setr_m128(xw0, xw0);
                let w1 = _mm256_setr_m128(xw1, xw1);

                let src_ptr = src.get_unchecked(src_stride * py + px..);

                let item_row_0 = _mm256_loadu_ps(src_ptr.as_ptr());
                let item_row_1 = _mm256_loadu_ps(src_ptr.get_unchecked(8..).as_ptr());
                let item_row_2 = _mm256_loadu_ps(src_ptr.get_unchecked(16..).as_ptr());
                let item_row_3 = _mm256_loadu_ps(src_ptr.get_unchecked(24..).as_ptr());

                store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, w0);
                store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, w0);
                store_2 = _mm256_fma_ps::<FMA>(store_2, item_row_2, w0);
                store_3 = _mm256_fma_ps::<FMA>(store_3, item_row_3, w0);

                let item_row_0 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride..).as_ptr());
                let item_row_1 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride + 8..).as_ptr());
                let item_row_2 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride + 16..).as_ptr());
                let item_row_3 = _mm256_loadu_ps(src_ptr.get_unchecked(src_stride + 24..).as_ptr());

                store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, w1);
                store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, w1);
                store_2 = _mm256_fma_ps::<FMA>(store_2, item_row_2, w1);
                store_3 = _mm256_fma_ps::<FMA>(store_3, item_row_3, w1);

                j += 2;
            }

            for j in j..bounds.size {
                let py = start_y + j;
                let weight = filter.get_unchecked(j);
                let v_weight = _mm256_broadcast_ss(weight);
                let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();
                let item_row_0 = _mm256_loadu_ps(src_ptr);
                let item_row_1 = _mm256_loadu_ps(src_ptr.add(8));
                let item_row_2 = _mm256_loadu_ps(src_ptr.add(16));
                let item_row_3 = _mm256_loadu_ps(src_ptr.add(24));

                store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, v_weight);
                store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, v_weight);
                store_2 = _mm256_fma_ps::<FMA>(store_2, item_row_2, v_weight);
                store_3 = _mm256_fma_ps::<FMA>(store_3, item_row_3, v_weight);
            }

            let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            _mm256_storeu_ps(dst_ptr, store_0);
            _mm256_storeu_ps(dst_ptr.add(8), store_1);
            _mm256_storeu_ps(dst_ptr.add(16), store_2);
            _mm256_storeu_ps(dst_ptr.add(24), store_3);
        }
    }

    #[inline(always)]
    fn convolve_vertical_part_avx_8_f32(
        &self,
        start_y: usize,
        start_x: usize,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        filter: &[f32],
        bounds: &FilterBounds,
    ) {
        unsafe {
            let mut store_0 = _mm256_setzero_ps();

            let px = start_x;

            for j in 0..bounds.size {
                let py = start_y + j;
                let weight = filter.get_unchecked(j);
                let v_weight = _mm256_broadcast_ss(weight);
                let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();
                let item_row_0 = _mm256_loadu_ps(src_ptr);

                store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, v_weight);
            }

            let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            _mm256_storeu_ps(dst_ptr, store_0);
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
        filter: &[f32],
        bounds: &FilterBounds,
    ) {
        unsafe {
            let mut store_0 = _mm256_setzero_ps();
            let mut store_1 = _mm256_setzero_ps();

            let px = start_x;

            for j in 0..bounds.size {
                let py = start_y + j;
                let weight = filter.get_unchecked(j);
                let v_weight = _mm256_broadcast_ss(weight);
                let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

                let item_row_0 = _mm256_loadu_ps(src_ptr);
                let item_row_1 = _mm256_loadu_ps(src_ptr.add(8));

                store_0 = _mm256_fma_ps::<FMA>(store_0, item_row_0, v_weight);
                store_1 = _mm256_fma_ps::<FMA>(store_1, item_row_1, v_weight);
            }

            let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            _mm256_storeu_ps(dst_ptr, store_0);
            _mm256_storeu_ps(dst_ptr.add(8), store_1);
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
        filter: &[f32],
        bounds: &FilterBounds,
    ) {
        unsafe {
            let mut store_0 = _mm_setzero_ps();

            let px = start_x;

            for j in 0..bounds.size {
                let py = start_y + j;
                let weight = filter.get_unchecked(j..);
                let v_weight = _mm_load_ss(weight.as_ptr());
                let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

                let item_row_0 = _mm_load_ss(src_ptr);

                store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
            }

            let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            _mm_store_ss(dst_ptr, store_0);
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
        weight_ptr: &[f32],
    ) {
        let mut cx = 0usize;
        let dst_width = dst.len();

        while cx + 32 <= dst_width {
            self.convolve_vertical_part_avx_32_f32(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
                bounds,
            );

            cx += 32;
        }

        while cx + 16 <= dst_width {
            self.convolve_vertical_part_avx_16_f32(
                bounds.start,
                cx,
                src,
                src_stride,
                dst,
                weight_ptr,
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
                weight_ptr,
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
                weight_ptr,
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
    use crate::test_utils::{XorShiftRng, assert_f32_slices_close, make_row_filter_weights_f32};

    // AVX2 folds 8/16/32 lanes at a time and the FMA variant fuses multiply+add,
    // both of which reorder and re-round relative to the scalar loop's strictly
    // sequential multiply-then-add - a small tolerance (not bit-exact) is
    // expected here, mirroring the horizontal-convolution comparison tests.
    const ATOL: f32 = 1e-5;

    #[test]
    fn avx2_default_vertical_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        run_vertical_comparison(convolve_vertical_avx_row_default_f32, "AVX2 default");
    }

    #[test]
    fn avx2_fma_vertical_matches_scalar_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        run_vertical_comparison(convolve_vertical_avx_row_fma_f32, "AVX2 FMA");
    }

    #[allow(clippy::type_complexity)]
    fn run_vertical_comparison(
        simd_fn: fn(usize, &FilterBounds, &[f32], &mut [f32], usize, &[f32], u32),
        label: &str,
    ) {
        // Row widths chosen to be multiples of 1 (plane), 3 (rgb) and 4 (rgba)
        // pixels, plus a couple of odd sizes so every SIMD tail-loop width
        // (32/16/8/1 lanes) gets exercised at least once.
        const ROW_WIDTHS: [usize; 7] = [1, 3, 4, 39, 99, 160, 41];

        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_height, out_height) in [(64usize, 37usize), (37, 64), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_f32(resampling, in_height, out_height).unwrap();

                for &row_width in ROW_WIDTHS.iter() {
                    let src_stride = row_width;
                    let mut rng = XorShiftRng::new(
                        0xC0FFEE
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

                        column_handler_floating_point::<f32, f32, f32>(
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
