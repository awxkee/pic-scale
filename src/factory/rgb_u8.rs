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
use crate::ThreadingPolicy;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::avx2::convolve_vertical_avx_row;
use crate::convolution::{
    ColumnFilter, ConvolutionOptions, HorizontalFilterPass, RowFilter, VerticalConvolutionPass,
};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::handler_provider::{
    handle_fixed_column_u8, handle_fixed_row_u8, handle_fixed_rows_4_u8,
};
use crate::image_store::ImageStore;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use crate::neon::*;
use crate::plan::{HorizontalFiltering, VerticalFiltering};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::{
    convolve_horizontal_rgb_sse_row_one, convolve_horizontal_rgb_sse_rows_4,
    convolve_vertical_sse_row,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::wasm_vertical_neon_row;
use std::sync::Arc;

impl HorizontalFilterPass<u8, f32, 3> for ImageStore<'_, u8, 3> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        _options: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<u8, 3> + Send + Sync> {
        let _scale_factor = _options.src_size.width as f32 / _options.dst_size.width as f32;
        let mut _dispatcher_4_rows: Option<
            fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i16>, u32),
        > = Some(handle_fixed_rows_4_u8::<3>);
        let mut _dispatcher_1_row: fn(&[u8], &mut [u8], &FilterWeights<i16>, u32) =
            handle_fixed_row_u8::<3>;

        #[cfg(all(target_arch = "aarch64", feature = "sve"))]
        if _scale_factor < 10.
            && std::arch::is_aarch64_feature_detected!("sve2")
            && std::arch::is_aarch64_feature_detected!("i8mm")
        {
            use crate::sve2::{
                sve_convolve_horizontal_rgb_neon_row_one_dot,
                sve_convolve_horizontal_rgb_neon_rows_4_dot,
            };

            let i_weights = filter_weights.numerical_approximation_q0_7(0);
            return Arc::new(HorizontalFiltering {
                filter_weights: i_weights,
                filter_4_rows: Some(sve_convolve_horizontal_rgb_neon_rows_4_dot),
                filter_row: sve_convolve_horizontal_rgb_neon_row_one_dot,
                threading_policy,
            });
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            match _options.workload_strategy {
                crate::WorkloadStrategy::PreferQuality => {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4);
                    _dispatcher_1_row = convolve_horizontal_rgb_neon_row_one;
                }
                crate::WorkloadStrategy::PreferSpeed => {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4);
                    _dispatcher_1_row = convolve_horizontal_rgb_neon_row_one;
                    #[cfg(feature = "rdm")]
                    if _scale_factor < 8.0 && std::arch::is_aarch64_feature_detected!("rdm") {
                        use crate::neon::{
                            convolve_horizontal_rgb_neon_rdm_row_one,
                            convolve_horizontal_rgb_neon_rdm_rows_4,
                        };
                        _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rdm_rows_4);
                        _dispatcher_1_row = convolve_horizontal_rgb_neon_rdm_row_one;
                    }
                    #[cfg(feature = "nightly_i8mm")]
                    if _scale_factor < 5.5 && std::arch::is_aarch64_feature_detected!("i8mm") {
                        use crate::neon::{
                            convolve_horizontal_rgb_neon_row_one_dot,
                            convolve_horizontal_rgb_neon_rows_4_dot,
                        };
                        let i_weights = filter_weights.numerical_approximation_q0_7(0);
                        return Arc::new(HorizontalFiltering {
                            filter_weights: i_weights,
                            filter_4_rows: Some(convolve_horizontal_rgb_neon_rows_4_dot),
                            filter_row: convolve_horizontal_rgb_neon_row_one_dot,
                            threading_policy,
                        });
                    }
                }
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_sse_rows_4);
                _dispatcher_1_row = convolve_horizontal_rgb_sse_row_one;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::{
                    convolve_horizontal_rgb_avx_row_one, convolve_horizontal_rgb_avx_rows_4,
                };
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_avx_rows_4);
                _dispatcher_1_row = convolve_horizontal_rgb_avx_row_one;
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            use crate::wasm32::{
                convolve_horizontal_rgb_wasm_row_one, convolve_horizontal_rgb_wasm_rows_4,
            };
            _dispatcher_4_rows = Some(convolve_horizontal_rgb_wasm_rows_4);
            _dispatcher_1_row = convolve_horizontal_rgb_wasm_row_one;
        }
        use crate::support::PRECISION;
        let i_weights = filter_weights.numerical_approximation::<i16, PRECISION>(0);
        Arc::new(HorizontalFiltering {
            filter_weights: i_weights,
            filter_4_rows: _dispatcher_4_rows,
            filter_row: _dispatcher_1_row,
            threading_policy,
        })
    }
}

impl VerticalConvolutionPass<u8, f32, 3> for ImageStore<'_, u8, 3> {
    fn vertical_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<u8, 3> + Send + Sync> {
        vertical_strategy_u8(filter_weights, threading_policy, options)
    }
}

pub(crate) fn vertical_strategy_u8<const N: usize>(
    filter_weights: FilterWeights<f32>,
    threading_policy: ThreadingPolicy,
    _options: ConvolutionOptions,
) -> Arc<dyn ColumnFilter<u8, N> + Send + Sync> {
    let _scale_factor = _options.src_size.height as f32 / _options.dst_size.height as f32;
    #[allow(clippy::type_complexity)]
    let mut _dispatcher: fn(usize, &FilterBounds, &[u8], &mut [u8], usize, &[i16], u32) =
        handle_fixed_column_u8;
    // For more downscaling better to use more precise version
    #[cfg(all(target_arch = "aarch64", feature = "sve"))]
    match _options.workload_strategy {
        crate::WorkloadStrategy::PreferQuality => {}
        crate::WorkloadStrategy::PreferSpeed => {
            if std::arch::is_aarch64_feature_detected!("sve2")
                && std::arch::is_aarch64_feature_detected!("i8mm")
            {
                use crate::sve2::convolve_vertical_sve2_i8_dot;
                let i_weights = filter_weights.numerical_approximation_q0_7(0);
                return Arc::new(VerticalFiltering {
                    filter_weights: i_weights,
                    filter_row: convolve_vertical_sve2_i8_dot,
                    threading_policy,
                });
            }
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        match _options.workload_strategy {
            crate::WorkloadStrategy::PreferQuality => {
                use crate::neon::convolve_vertical_neon_i32_precision;
                _dispatcher = convolve_vertical_neon_i32_precision;
            }
            crate::WorkloadStrategy::PreferSpeed => {
                #[cfg(feature = "rdm")]
                if _scale_factor < 8. && std::arch::is_aarch64_feature_detected!("rdm") {
                    use crate::neon::convolve_vertical_neon_i16_precision;
                    _dispatcher = convolve_vertical_neon_i16_precision;
                } else {
                    use crate::neon::convolve_vertical_neon_i32_precision;
                    _dispatcher = convolve_vertical_neon_i32_precision;
                }
                #[cfg(feature = "nightly_i8mm")]
                if _scale_factor < 10. && std::arch::is_aarch64_feature_detected!("i8mm") {
                    use crate::neon::convolve_vertical_neon_i8_dot;
                    let i_weights = filter_weights.numerical_approximation_q0_7(0);
                    return Arc::new(VerticalFiltering {
                        filter_weights: i_weights,
                        filter_row: convolve_vertical_neon_i8_dot,
                        threading_policy,
                    });
                }
                #[cfg(not(feature = "rdm"))]
                {
                    use crate::neon::convolve_vertical_neon_i32_precision;
                    _dispatcher = convolve_vertical_neon_i32_precision;
                }
            }
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _dispatcher = convolve_vertical_sse_row;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            _dispatcher = convolve_vertical_avx_row;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        _dispatcher = wasm_vertical_neon_row;
    }
    use crate::support::PRECISION;
    let i_weights = filter_weights.numerical_approximation::<i16, PRECISION>(0);
    Arc::new(VerticalFiltering {
        filter_weights: i_weights,
        filter_row: _dispatcher,
        threading_policy,
    })
}
