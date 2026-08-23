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

use crate::avx2::utils::{_mm_prefer_fma_ps, _mm256_prefer_fma_ps, shuffle};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

#[inline(always)]
fn ch_parts_4_rgb_f32_sse<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m128,
    weight1: __m128,
    weight2: __m128,
    weight3: __m128,
    store_0: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked(start_x * CN..);

        let rgb_pixel_0 = _mm_loadu_ps(src_ptr.as_ptr());
        let rgb_pixel_1 = _mm_loadu_ps(src_ptr.get_unchecked(3..).as_ptr());
        let rgb_pixel_2 = _mm_loadu_ps(src_ptr.get_unchecked(6..).as_ptr());
        let mut rgb_pixel_3 = _mm_loadu_ps(src_ptr.get_unchecked(8..).as_ptr());
        rgb_pixel_3 = _mm_shuffle_ps::<{ shuffle(0, 3, 2, 1) }>(rgb_pixel_3, rgb_pixel_3);

        let acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
        let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
        let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_2, weight2);
        _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_3, weight3)
    }
}

#[inline(always)]
fn ch_parts_4_rgb_f32_avx<const FMA: bool>(
    start_x: usize,
    src0: &[f32],
    src1: &[f32],
    weight0: __m256,
    weight1: __m256,
    weight2: __m256,
    weight3: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const CN: usize = 3;
        let src_ptr0 = src0.get_unchecked(start_x * CN..);
        let src_ptr1 = src1.get_unchecked(start_x * CN..);

        let rgb_pixel_0_0 = _mm_loadu_ps(src_ptr0.as_ptr());
        let rgb_pixel_0_1 = _mm_loadu_ps(src_ptr0.get_unchecked(3..).as_ptr());
        let rgb_pixel_0_2 = _mm_loadu_ps(src_ptr0.get_unchecked(6..).as_ptr());
        let mut rgb_pixel_0_3 = _mm_loadu_ps(src_ptr0.get_unchecked(8..).as_ptr());
        rgb_pixel_0_3 = _mm_shuffle_ps::<{ shuffle(0, 3, 2, 1) }>(rgb_pixel_0_3, rgb_pixel_0_3);

        let rgb_pixel_1_0 = _mm_loadu_ps(src_ptr1.as_ptr());
        let rgb_pixel_1_1 = _mm_loadu_ps(src_ptr1.get_unchecked(3..).as_ptr());
        let rgb_pixel_1_2 = _mm_loadu_ps(src_ptr1.get_unchecked(6..).as_ptr());
        let mut rgb_pixel_1_3 = _mm_loadu_ps(src_ptr1.get_unchecked(8..).as_ptr());
        rgb_pixel_1_3 = _mm_shuffle_ps::<{ shuffle(0, 3, 2, 1) }>(rgb_pixel_1_3, rgb_pixel_1_3);

        let rgb_pixel_0 =
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_0), rgb_pixel_1_0);
        let rgb_pixel_1 =
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_1), rgb_pixel_1_1);
        let rgb_pixel_2 =
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_2), rgb_pixel_1_2);
        let rgb_pixel_3 =
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_3), rgb_pixel_1_3);

        let acc = _mm256_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
        let acc = _mm256_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
        let acc = _mm256_prefer_fma_ps::<FMA>(acc, rgb_pixel_2, weight2);
        _mm256_prefer_fma_ps::<FMA>(acc, rgb_pixel_3, weight3)
    }
}

#[inline(always)]
fn ch_parts_2_rgb_f32_avx<const FMA: bool>(
    start_x: usize,
    src0: &[f32],
    src1: &[f32],
    weight0: __m256,
    weight1: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const CN: usize = 3;
        let src_ptr0 = src0.get_unchecked(start_x * CN..);
        let src_ptr1 = src1.get_unchecked(start_x * CN..);

        let orig0 = _mm_loadu_ps(src_ptr0.as_ptr());
        let orig1 = _mm_loadu_ps(src_ptr1.as_ptr());

        let rgb_pixel_0_0 = orig0;
        let mut rgb_pixel_0_1 = _mm_loadu_ps(src_ptr0.get_unchecked(2..).as_ptr());
        rgb_pixel_0_1 = _mm_shuffle_ps::<{ shuffle(0, 3, 2, 1) }>(rgb_pixel_0_1, rgb_pixel_0_1);

        let rgb_pixel_1_0 = orig1;
        let mut rgb_pixel_1_1 = _mm_loadu_ps(src_ptr1.get_unchecked(2..).as_ptr());
        rgb_pixel_1_1 = _mm_shuffle_ps::<{ shuffle(0, 3, 2, 1) }>(rgb_pixel_1_1, rgb_pixel_1_1);

        let rgb_pixel_0 =
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_0), rgb_pixel_1_0);
        let rgb_pixel_1 =
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_1), rgb_pixel_1_1);

        let mut acc = _mm256_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
        acc = _mm256_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
        acc
    }
}

#[inline(always)]
fn ch_parts_2_rgb_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m128,
    weight1: __m128,
    store_0: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked(start_x * CN..);

        let orig1 = _mm_loadu_ps(src_ptr.as_ptr());
        let rgb_pixel_0 = orig1;
        let mut rgb_pixel_1 = _mm_loadu_ps(src_ptr.get_unchecked(2..).as_ptr());
        rgb_pixel_1 = _mm_shuffle_ps::<{ shuffle(0, 3, 2, 1) }>(rgb_pixel_1, rgb_pixel_1);

        let mut acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
        acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
        acc
    }
}

#[inline(always)]
fn ch_parts_one_rgb_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m128,
    store_0: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked(start_x * CN..).as_ptr();
        let rgb_pixel = _mm_setr_ps(
            src_ptr.add(0).read_unaligned(),
            src_ptr.add(1).read_unaligned(),
            src_ptr.add(2).read_unaligned(),
            0f32,
        );
        _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel, weight0)
    }
}

#[inline(always)]
fn ch_parts_one_rgb_f32_avx<const FMA: bool>(
    start_x: usize,
    src0: &[f32],
    src1: &[f32],
    weight0: __m256,
    store_0: __m256,
) -> __m256 {
    unsafe {
        const CN: usize = 3;
        let src_ptr0 = src0.get_unchecked(start_x * CN..);
        let src_ptr1 = src1.get_unchecked(start_x * CN..);

        let rgb_pixel0 = _mm_setr_ps(
            *src_ptr0.get_unchecked(0),
            *src_ptr0.get_unchecked(1),
            *src_ptr0.get_unchecked(2),
            0.,
        );

        let rgb_pixel1 = _mm_setr_ps(
            *src_ptr1.get_unchecked(0),
            *src_ptr1.get_unchecked(1),
            *src_ptr1.get_unchecked(2),
            0.,
        );

        let rgb_pixel = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel0), rgb_pixel1);

        _mm256_prefer_fma_ps::<FMA>(store_0, rgb_pixel, weight0)
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_row_one_f32_default(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_avx_row_one_f32_regular(filter_weights, src, dst);
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_row_one_f32_fma(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_avx_row_one_f32_fma_impl(filter_weights, src, dst);
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_horizontal_rgb_avx_row_one_f32_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = ExecutionUnit1Row::<false>::default();
    unit.pass(filter_weights, src, dst);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_horizontal_rgb_avx_row_one_f32_fma_impl(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = ExecutionUnit1Row::<true>::default();
    unit.pass(filter_weights, src, dst);
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit1Row<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit1Row<FMA> {
    #[inline(always)]
    fn pass(&self, filter_weights: &FilterWeights<f32>, src: &[f32], dst: &mut [f32]) {
        unsafe {
            const CN: usize = 3;
            let mut filter_offset = 0usize;
            let weights = &filter_weights.weights;

            let dst_width = filter_weights.bounds.len();

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store = _mm_setzero_ps();

                while jx + 4 <= bounds.size {
                    let ptr = weights.get_unchecked(jx + filter_offset..);
                    let weights = _mm_loadu_ps(ptr.as_ptr());

                    let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                    let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);
                    let xw2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(weights, weights);
                    let xw3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(weights, weights);

                    let filter_start = jx + bounds.start;
                    store =
                        ch_parts_4_rgb_f32_sse::<FMA>(filter_start, src, xw0, xw1, xw2, xw3, store);
                    jx += 4;
                }

                while jx + 2 <= bounds.size {
                    let ptr = weights.get_unchecked(jx + filter_offset..);
                    let weights = _mm_castsi128_ps(_mm_loadu_si64(ptr.as_ptr().cast()));

                    let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                    let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);

                    let filter_start = jx + bounds.start;
                    store = ch_parts_2_rgb_f32::<FMA>(filter_start, src, xw0, xw1, store);
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = weights.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm_broadcast_ss(ptr.get_unchecked(0));
                    let filter_start = jx + bounds.start;
                    store = ch_parts_one_rgb_f32::<FMA>(filter_start, src, weight0, store);
                    jx += 1;
                }

                let px = x * CN;
                let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
                _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(store));
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(store));

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_rows_4_f32_default(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_avx_rows_4_f32_regular(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_rows_4_f32_fma(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_avx_rows_4_f32_fma_impl(
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
fn convolve_horizontal_rgb_avx_rows_4_f32_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let unit = ExecutionUnit4Row::<false>::default();
    unit.pass(filter_weights, src, src_stride, dst, dst_stride);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_horizontal_rgb_avx_rows_4_f32_fma_impl(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let unit = ExecutionUnit4Row::<true>::default();
    unit.pass(filter_weights, src, src_stride, dst, dst_stride);
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit4Row<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit4Row<FMA> {
    #[inline(always)]
    fn pass(
        &self,
        filter_weights: &FilterWeights<f32>,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        dst_stride: usize,
    ) {
        unsafe {
            const CN: usize = 3;
            let mut filter_offset = 0usize;

            let dst_width = filter_weights.bounds.len();
            let weights_ptr = &filter_weights.weights;

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store_0 = _mm256_setzero_ps();
                let mut store_1 = _mm256_setzero_ps();

                while jx + 4 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weights = _mm_loadu_ps(ptr.as_ptr());

                    let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                    let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);
                    let xw2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(weights, weights);
                    let xw3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(weights, weights);

                    let w0 = _mm256_setr_m128(xw0, xw0);
                    let w1 = _mm256_setr_m128(xw1, xw1);
                    let w2 = _mm256_setr_m128(xw2, xw2);
                    let w3 = _mm256_setr_m128(xw3, xw3);

                    let filter_start = jx + bounds.start;
                    store_0 = ch_parts_4_rgb_f32_avx::<FMA>(
                        filter_start,
                        src,
                        src.get_unchecked(src_stride..),
                        w0,
                        w1,
                        w2,
                        w3,
                        store_0,
                    );
                    store_1 = ch_parts_4_rgb_f32_avx::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 2..),
                        src.get_unchecked(src_stride * 3..),
                        w0,
                        w1,
                        w2,
                        w3,
                        store_1,
                    );
                    jx += 4;
                }

                while jx + 2 <= bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weights = _mm_castsi128_ps(_mm_loadu_si64(ptr.as_ptr().cast()));
                    let xw0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(weights, weights);
                    let xw1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(weights, weights);

                    let w0 = _mm256_setr_m128(xw0, xw0);
                    let w1 = _mm256_setr_m128(xw1, xw1);
                    let filter_start = jx + bounds.start;
                    store_0 = ch_parts_2_rgb_f32_avx::<FMA>(
                        filter_start,
                        src,
                        src.get_unchecked(src_stride..),
                        w0,
                        w1,
                        store_0,
                    );
                    store_1 = ch_parts_2_rgb_f32_avx::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 2..),
                        src.get_unchecked(src_stride * 3..),
                        w0,
                        w1,
                        store_1,
                    );
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm256_broadcast_ss(ptr.get_unchecked(0));
                    let filter_start = jx + bounds.start;
                    store_0 = ch_parts_one_rgb_f32_avx::<FMA>(
                        filter_start,
                        src,
                        src.get_unchecked(src_stride..),
                        weight0,
                        store_0,
                    );
                    store_1 = ch_parts_one_rgb_f32_avx::<FMA>(
                        filter_start,
                        src.get_unchecked(src_stride * 2..),
                        src.get_unchecked(src_stride * 3..),
                        weight0,
                        store_1,
                    );
                    jx += 1;
                }

                let px = x * CN;
                let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
                _mm_storeu_si64(
                    dest_ptr as *mut u8,
                    _mm_castps_si128(_mm256_castps256_ps128(store_0)),
                );
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(_mm256_castps256_ps128(store_0)));

                let ss1 = _mm256_extractf128_ps::<1>(store_0);

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
                _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(ss1));
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(ss1));

                let ss2 = _mm256_castps256_ps128(store_1);

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
                _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(ss2));
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(ss2));

                let ss3 = _mm256_extractf128_ps::<1>(store_1);

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
                _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(ss3));
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(ss3));

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::ResamplingFunction;
    use crate::convolve_naive_f32::{convolve_horizontal_native_row_f32, convolve_horizontal_rgba_4_row_f32};
    use crate::test_utils::{XorShiftRng, assert_f32_slices_close, make_row_filter_weights_f32};

    // Same reordering/FMA-rounding rationale as avx2/rgba_f32.rs.
    const ATOL: f32 = 1e-5;

    #[test]
    fn avx2_default_row_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        run_row_comparison(convolve_horizontal_rgb_avx_row_one_f32_default, "AVX2 default");
    }

    #[test]
    fn avx2_fma_row_matches_scalar_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        run_row_comparison(convolve_horizontal_rgb_avx_row_one_f32_fma, "AVX2 FMA");
    }

    fn run_row_comparison(
        simd_fn: fn(&[f32], &mut [f32], &FilterWeights<f32>, u32),
        label: &str,
    ) {
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64), (256, 256), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_f32(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xC0FFEE ^ (in_size as u64) << 32 ^ out_size as u64);
                let src = rng.fill_f32_unit(in_size * 3);

                let mut dst_scalar = vec![0f32; out_size * 3];
                let mut dst_simd = vec![0f32; out_size * 3];
                convolve_horizontal_native_row_f32::<3>(&src, &mut dst_scalar, &filter_weights, 8);
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
        run_rows_4_comparison(convolve_horizontal_rgb_avx_rows_4_f32_default, "AVX2 default");
    }

    #[test]
    fn avx2_fma_rows_4_matches_scalar_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        run_rows_4_comparison(convolve_horizontal_rgb_avx_rows_4_f32_fma, "AVX2 FMA");
    }

    fn run_rows_4_comparison(
        simd_fn: fn(&[f32], usize, &mut [f32], usize, &FilterWeights<f32>, u32),
        label: &str,
    ) {
        const ROWS: usize = 4;
        for resampling in [ResamplingFunction::Bilinear, ResamplingFunction::Lanczos3] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64)] {
                let filter_weights =
                    make_row_filter_weights_f32(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xBADF00D ^ (in_size as u64) << 32 ^ out_size as u64);
                let src_stride = in_size * 3;
                let dst_stride = out_size * 3;
                let src = rng.fill_f32_unit(src_stride * ROWS);

                let mut dst_scalar = vec![0f32; dst_stride * ROWS];
                let mut dst_simd = vec![0f32; dst_stride * ROWS];
                convolve_horizontal_rgba_4_row_f32::<3>(
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
