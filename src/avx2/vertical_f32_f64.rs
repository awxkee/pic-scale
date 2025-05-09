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

pub(crate) fn convolve_vertical_avx_row_f32_f64<const FMA: bool>(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weights: &[f64],
) {
    unsafe {
        if FMA {
            convolve_vertical_avx_row_f32_f64_fma(width, bounds, src, dst, src_stride, weights);
        } else {
            convolve_vertical_avx_row_f32_f64_regular(width, bounds, src, dst, src_stride, weights);
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_vertical_avx_row_f32_f64_regular(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weights: &[f64],
) {
    unsafe {
        let unit = ExecutionUnit::<false>::default();
        unit.pass(width, bounds, src, dst, src_stride, weights);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_vertical_avx_row_f32_f64_fma(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weights: &[f64],
) {
    unsafe {
        let unit = ExecutionUnit::<true>::default();
        unit.pass(width, bounds, src, dst, src_stride, weights);
    }
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit<FMA> {
    #[inline(always)]
    unsafe fn convolve_vertical_part_avx_8_f32(
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

            for j in 0..bounds.size {
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
    unsafe fn convolve_vertical_part_avx_16_f32(
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
    unsafe fn convolve_vertical_part_avx_f32(
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
    unsafe fn pass(
        &self,
        _: usize,
        bounds: &FilterBounds,
        src: &[f32],
        dst: &mut [f32],
        src_stride: usize,
        weights: &[f64],
    ) {
        unsafe {
            let mut cx = 0usize;
            let dst_width = dst.len();

            while cx + 16 < dst_width {
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

            while cx + 8 < dst_width {
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
}
