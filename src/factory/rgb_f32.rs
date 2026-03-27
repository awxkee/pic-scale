/*
 * Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::convolution::{
    ColumnFilter, ConvolutionOptions, HorizontalFilterPass, RowFilter, VerticalConvolutionPass,
};
use crate::convolve_naive_f32::*;
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::floating_point_vertical::column_handler_floating_point;
use crate::image_store::ImageStore;
#[cfg(all(target_arch = "aarch64", feature = "neon",))]
use crate::neon::*;
use crate::plan::{HorizontalFiltering, VerticalFiltering};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::*;
use crate::{ThreadingPolicy, WorkloadStrategy};
use std::sync::Arc;

pub(crate) fn convolve_vertical_rgb_native_row_f32(
    q: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight: &[f32],
    _: u32,
) {
    column_handler_floating_point::<f32, f32, f32>(q, bounds, src, dst, src_stride, weight, 8);
}

pub(crate) fn convolve_vertical_rgb_native_row_f64(
    q: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight: &[f64],
    _: u32,
) {
    column_handler_floating_point::<f32, f64, f64>(q, bounds, src, dst, src_stride, weight, 8);
}

impl HorizontalFilterPass<f32, f32, 3> for ImageStore<'_, f32, 3> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<f32, 3> + Send + Sync> {
        let mut _dispatcher_4_rows: Option<
            fn(&[f32], usize, &mut [f32], usize, &FilterWeights<f32>, u32),
        > = Some(convolve_horizontal_rgba_4_row_f32::<3>);
        let mut _dispatcher_row: fn(&[f32], &mut [f32], &FilterWeights<f32>, u32) =
            convolve_horizontal_native_row_f32::<3>;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4_f32);
            _dispatcher_row = convolve_horizontal_rgb_neon_row_one_f32;
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_sse_rows_4_f32);
                _dispatcher_row = convolve_horizontal_rgb_sse_row_one_f32;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::avx2::{
                convolve_horizontal_rgb_avx_row_one_f32_default,
                convolve_horizontal_rgb_avx_row_one_f32_fma,
                convolve_horizontal_rgb_avx_rows_4_f32_default,
                convolve_horizontal_rgb_avx_rows_4_f32_fma,
            };
            let has_fma = std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx2") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_avx_rows_4_f32_default);
                _dispatcher_row = convolve_horizontal_rgb_avx_row_one_f32_default;
                if has_fma {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgb_avx_rows_4_f32_fma);
                    _dispatcher_row = convolve_horizontal_rgb_avx_row_one_f32_fma;
                }
            }
        }
        Arc::new(HorizontalFiltering {
            filter_weights,
            filter_4_rows: _dispatcher_4_rows,
            filter_row: _dispatcher_row,
            threading_policy,
        })
    }
}

impl HorizontalFilterPass<f32, f64, 3> for ImageStore<'_, f32, 3> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f64>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<f32, 3> + Send + Sync> {
        let mut _dispatcher_4_rows: Option<
            fn(&[f32], usize, &mut [f32], usize, &FilterWeights<f64>, u32),
        > = Some(convolve_horizontal_4_row_f32_f64::<3>);
        let mut _dispatcher_row: fn(&[f32], &mut [f32], &FilterWeights<f64>, u32) =
            convolve_horizontal_native_row_f32_f64::<3>;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::{
                convolve_horizontal_rgb_neon_row_one_f32_f64,
                convolve_horizontal_rgb_neon_rows_4_f32_f64,
            };
            _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4_f32_f64);
            _dispatcher_row = convolve_horizontal_rgb_neon_row_one_f32_f64;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::avx2::{
                convolve_horizontal_rgb_avx_row_one_f32_f64_default,
                convolve_horizontal_rgb_avx_row_one_f32_f64_fma,
                convolve_horizontal_rgb_avx_rows_4_f32_f64_default,
                convolve_horizontal_rgb_avx_rows_4_f32_f64_fma,
            };
            let has_fma = std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx2") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_avx_rows_4_f32_f64_default);
                _dispatcher_row = convolve_horizontal_rgb_avx_row_one_f32_f64_default;
                if has_fma {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgb_avx_rows_4_f32_f64_fma);
                    _dispatcher_row = convolve_horizontal_rgb_avx_row_one_f32_f64_fma;
                }
            }
        }
        Arc::new(HorizontalFiltering {
            filter_weights,
            filter_4_rows: _dispatcher_4_rows,
            filter_row: _dispatcher_row,
            threading_policy,
        })
    }
}

impl VerticalConvolutionPass<f32, f32, 3> for ImageStore<'_, f32, 3> {
    fn vertical_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<f32, 3> + Send + Sync> {
        #[allow(clippy::type_complexity)]
        let mut _dispatcher: fn(
            usize,
            &FilterBounds,
            &[f32],
            &mut [f32],
            usize,
            &[f32],
            u32,
        ) = convolve_vertical_rgb_native_row_f32;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f32;
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_rgb_sse_row_f32;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_fma = std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::convolve_vertical_avx_row_default_f32;
                _dispatcher = convolve_vertical_avx_row_default_f32;
                if has_fma {
                    use crate::avx2::convolve_vertical_avx_row_fma_f32;
                    _dispatcher = convolve_vertical_avx_row_fma_f32;
                }
            }
        }
        Arc::new(VerticalFiltering {
            filter_weights,
            filter_row: _dispatcher,
            threading_policy,
        })
    }
}

impl VerticalConvolutionPass<f32, f64, 3> for ImageStore<'_, f32, 3> {
    fn vertical_plan(
        filter_weights: FilterWeights<f64>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<f32, 3> + Send + Sync> {
        #[allow(clippy::type_complexity)]
        let mut _dispatcher: fn(
            usize,
            &FilterBounds,
            &[f32],
            &mut [f32],
            usize,
            &[f64],
            u32,
        ) = convolve_vertical_rgb_native_row_f64;
        if options.workload_strategy == WorkloadStrategy::PreferQuality {
            use crate::factory::rgb_f32::convolve_vertical_rgb_native_row_f64;
            _dispatcher = convolve_vertical_rgb_native_row_f64;
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::convolve_vertical_neon_row_f32_f64;
            _dispatcher = convolve_vertical_neon_row_f32_f64;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                if std::arch::is_x86_feature_detected!("fma") {
                    use crate::avx2::convolve_vertical_avx_row_f32_f64_fma;
                    _dispatcher = convolve_vertical_avx_row_f32_f64_fma;
                } else {
                    use crate::avx2::convolve_vertical_avx_row_f32_f64_default;
                    _dispatcher = convolve_vertical_avx_row_f32_f64_default;
                }
            }
        }
        Arc::new(VerticalFiltering {
            filter_weights,
            filter_row: _dispatcher,
            threading_policy,
        })
    }
}
