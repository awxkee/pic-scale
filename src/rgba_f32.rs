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
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::avx2::{
    convolve_horizontal_rgba_avx_row_one_f32, convolve_horizontal_rgba_avx_rows_4_f32,
    convolve_vertical_avx_row_f32,
};
use crate::convolution::{ConvolutionOptions, RowFilter, HorizontalFilterPass, VerticalConvolutionPass, ColumnFilter};
use crate::convolve_naive_f32::{
    convolve_horizontal_4_row_f32_f64, convolve_horizontal_native_row_f32,
    convolve_horizontal_native_row_f32_f64, convolve_horizontal_rgba_4_row_f32,
};
use crate::filter_weights::*;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use crate::neon::*;
use crate::plan::{HorizontalFiltering, VerticalFiltering};
use crate::rgb_f32::{convolve_vertical_rgb_native_row_f32, convolve_vertical_rgb_native_row_f64};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::*;
use crate::{ImageStore, ThreadingPolicy};
use std::sync::Arc;

impl HorizontalFilterPass<f32, f32, 4> for ImageStore<'_, f32, 4> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<f32, 4> + Send + Sync> {
        let mut _dispatcher_4_rows: Option<
            fn(&[f32], usize, &mut [f32], usize, &FilterWeights<f32>, u32),
        > = Some(convolve_horizontal_rgba_4_row_f32::<4>);
        let mut _dispatcher_row: fn(&[f32], &mut [f32], &FilterWeights<f32>, u32) =
            convolve_horizontal_native_row_f32::<4>;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgba_neon_rows_4);
            _dispatcher_row = convolve_horizontal_rgba_neon_row_one;
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_sse_rows_4_f32::<false>);
                _dispatcher_row = convolve_horizontal_rgba_sse_row_one_f32::<false>;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_avx_rows_4_f32::<false>);
                _dispatcher_row = convolve_horizontal_rgba_avx_row_one_f32::<false>;
                if std::arch::is_x86_feature_detected!("fma") {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgba_avx_rows_4_f32::<true>);
                    _dispatcher_row = convolve_horizontal_rgba_avx_row_one_f32::<true>;
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

impl HorizontalFilterPass<f32, f64, 4> for ImageStore<'_, f32, 4> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f64>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<f32, 4> + Send + Sync> {
        #[allow(clippy::type_complexity)]
        let mut _dispatcher_4_rows: Option<
            fn(&[f32], usize, &mut [f32], usize, &FilterWeights<f64>, u32),
        > = Some(convolve_horizontal_4_row_f32_f64::<4>);
        #[allow(clippy::type_complexity)]
        let mut _dispatcher_row: fn(&[f32], &mut [f32], &FilterWeights<f64>, u32) =
            convolve_horizontal_native_row_f32_f64::<4>;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::{
                convolve_horizontal_rgba_neon_row_one_f32_f64,
                convolve_horizontal_rgba_neon_rows_4_f32_f64,
            };
            _dispatcher_4_rows = Some(convolve_horizontal_rgba_neon_rows_4_f32_f64);
            _dispatcher_row = convolve_horizontal_rgba_neon_row_one_f32_f64;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::avx2::{
                convolve_horizontal_rgba_avx_row_one_f32_f64,
                convolve_horizontal_rgba_avx_rows_4_f32_f64,
            };
            if std::arch::is_x86_feature_detected!("avx2") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_avx_rows_4_f32_f64::<false>);
                _dispatcher_row = convolve_horizontal_rgba_avx_row_one_f32_f64::<false>;
                if std::arch::is_x86_feature_detected!("fma") {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgba_avx_rows_4_f32_f64::<true>);
                    _dispatcher_row = convolve_horizontal_rgba_avx_row_one_f32_f64::<true>;
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

impl VerticalConvolutionPass<f32, f32, 4> for ImageStore<'_, f32, 4> {
    fn vertical_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<f32, 4> + Send + Sync> {
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
                _dispatcher = convolve_vertical_rgb_sse_row_f32::<false>;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_fma = std::arch::is_x86_feature_detected!("fma");
            if std::is_x86_feature_detected!("avx2") {
                _dispatcher = convolve_vertical_avx_row_f32::<false>;
                if has_fma {
                    _dispatcher = convolve_vertical_avx_row_f32::<true>;
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

impl VerticalConvolutionPass<f32, f64, 4> for ImageStore<'_, f32, 4> {
    fn vertical_plan(
        filter_weights: FilterWeights<f64>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<f32, 4> + Send + Sync> {
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
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::convolve_vertical_neon_row_f32_f64;
            _dispatcher = convolve_vertical_neon_row_f32_f64;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::convolve_vertical_avx_row_f32_f64;
                if std::arch::is_x86_feature_detected!("fma") {
                    _dispatcher = convolve_vertical_avx_row_f32_f64::<true>;
                } else {
                    _dispatcher = convolve_vertical_avx_row_f32_f64::<false>;
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
