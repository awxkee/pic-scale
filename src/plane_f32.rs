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
use crate::avx2::convolve_vertical_avx_row_f32;
use crate::convolution::{ConvolutionOptions, HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_f32::{
    convolve_horizontal_4_row_f32_f64, convolve_horizontal_native_row_f32,
    convolve_horizontal_native_row_f32_f64, convolve_horizontal_rgba_4_row_f32,
};
use crate::dispatch_group_f32::{convolve_horizontal_dispatch_f32, convolve_vertical_dispatch_f32};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_store::ImageStoreMut;
#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::neon::{
    convolve_horizontal_plane_neon_row_one, convolve_horizontal_plane_neon_rows_4,
    convolve_vertical_rgb_neon_row_f32,
};
use crate::rgb_f32::{convolve_vertical_rgb_native_row_f32, convolve_vertical_rgb_native_row_f64};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::{
    convolve_horizontal_plane_sse_row_one, convolve_horizontal_plane_sse_rows_4,
    convolve_vertical_rgb_sse_row_f32,
};
use crate::{ImageStore, WorkloadStrategy};
use rayon::ThreadPool;

impl HorizontalConvolutionPass<f32, 1> for ImageStore<'_, f32, 1> {
    #[allow(clippy::type_complexity)]
    #[inline(always)]
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f32, 1>,
        pool: &Option<ThreadPool>,
        options: ConvolutionOptions,
    ) {
        match options.workload_strategy {
            WorkloadStrategy::PreferQuality => {
                let mut _dispatcher_4_rows: Option<
                    fn(usize, usize, &FilterWeights<f64>, &[f32], usize, &mut [f32], usize),
                > = Some(convolve_horizontal_4_row_f32_f64::<1>);
                let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f64>, &[f32], &mut [f32]) =
                    convolve_horizontal_native_row_f32_f64::<1>;
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    use crate::neon::{
                        convolve_horizontal_plane_neon_row_one_f32_f64,
                        convolve_horizontal_plane_neon_rows_4_f32_f64,
                    };
                    _dispatcher_4_rows = Some(convolve_horizontal_plane_neon_rows_4_f32_f64);
                    _dispatcher_row = convolve_horizontal_plane_neon_row_one_f32_f64;
                }
                #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                {
                    if std::arch::is_x86_feature_detected!("avx2") {
                        use crate::avx2::{
                            convolve_hor_plane_avx_row_one_f32_f64,
                            convolve_hor_plane_avx_rows_4_f32_f64,
                        };
                        _dispatcher_4_rows = Some(convolve_hor_plane_avx_rows_4_f32_f64::<false>);
                        _dispatcher_row = convolve_hor_plane_avx_row_one_f32_f64::<false>;
                        if std::arch::is_x86_feature_detected!("fma") {
                            _dispatcher_4_rows =
                                Some(convolve_hor_plane_avx_rows_4_f32_f64::<true>);
                            _dispatcher_row = convolve_hor_plane_avx_row_one_f32_f64::<true>;
                        }
                    }
                }
                let weights = filter_weights.cast::<f64>();
                convolve_horizontal_dispatch_f32(
                    self,
                    weights,
                    destination,
                    pool,
                    _dispatcher_4_rows,
                    _dispatcher_row,
                );
            }
            WorkloadStrategy::PreferSpeed => {
                let mut _dispatcher_4_rows: Option<
                    fn(usize, usize, &FilterWeights<f32>, &[f32], usize, &mut [f32], usize),
                > = Some(convolve_horizontal_rgba_4_row_f32::<1>);
                let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, &[f32], &mut [f32]) =
                    convolve_horizontal_native_row_f32::<1>;
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    _dispatcher_4_rows = Some(convolve_horizontal_plane_neon_rows_4);
                    _dispatcher_row = convolve_horizontal_plane_neon_row_one;
                }
                #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
                {
                    if std::arch::is_x86_feature_detected!("sse4.1") {
                        _dispatcher_4_rows = Some(convolve_horizontal_plane_sse_rows_4::<false>);
                        _dispatcher_row = convolve_horizontal_plane_sse_row_one::<false>;
                        if std::arch::is_x86_feature_detected!("fma") {
                            _dispatcher_4_rows = Some(convolve_horizontal_plane_sse_rows_4::<true>);
                            _dispatcher_row = convolve_horizontal_plane_sse_row_one::<true>;
                        }
                    }
                }
                #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                {
                    use crate::avx2::{
                        convolve_horizontal_plane_avx_row_one_f32,
                        convolve_horizontal_plane_avx_rows_4_f32,
                    };
                    let has_fma = std::arch::is_x86_feature_detected!("fma");
                    if std::arch::is_x86_feature_detected!("avx2") {
                        _dispatcher_4_rows =
                            Some(convolve_horizontal_plane_avx_rows_4_f32::<false>);
                        _dispatcher_row = convolve_horizontal_plane_avx_row_one_f32::<false>;
                        if has_fma {
                            _dispatcher_4_rows =
                                Some(convolve_horizontal_plane_avx_rows_4_f32::<true>);
                            _dispatcher_row = convolve_horizontal_plane_avx_row_one_f32::<true>;
                        }
                    }
                }
                convolve_horizontal_dispatch_f32(
                    self,
                    filter_weights,
                    destination,
                    pool,
                    _dispatcher_4_rows,
                    _dispatcher_row,
                );
            }
        }
    }
}

impl VerticalConvolutionPass<f32, 1> for ImageStore<'_, f32, 1> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f32, 1>,
        pool: &Option<ThreadPool>,
        options: ConvolutionOptions,
    ) {
        match options.workload_strategy {
            WorkloadStrategy::PreferQuality => {
                #[allow(clippy::type_complexity)]
                let mut _dispatcher: fn(
                    usize,
                    &FilterBounds,
                    &[f32],
                    &mut [f32],
                    usize,
                    &[f64],
                ) = convolve_vertical_rgb_native_row_f64;
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    use crate::neon::convolve_vertical_neon_row_f32_f64;
                    _dispatcher = convolve_vertical_neon_row_f32_f64;
                }
                let weights = filter_weights.cast::<f64>();
                convolve_vertical_dispatch_f32(self, weights, destination, pool, _dispatcher);
            }
            WorkloadStrategy::PreferSpeed => {
                #[allow(clippy::type_complexity)]
                let mut _dispatcher: fn(
                    usize,
                    &FilterBounds,
                    &[f32],
                    &mut [f32],
                    usize,
                    &[f32],
                ) = convolve_vertical_rgb_native_row_f32;
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
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
                    if std::arch::is_x86_feature_detected!("avx2") {
                        _dispatcher = convolve_vertical_avx_row_f32::<false>;
                        if has_fma {
                            _dispatcher = convolve_vertical_avx_row_f32::<true>;
                        }
                    }
                }
                convolve_vertical_dispatch_f32(
                    self,
                    filter_weights,
                    destination,
                    pool,
                    _dispatcher,
                );
            }
        }
    }
}
