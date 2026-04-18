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
use crate::convolution::{
    ColumnFilter, ConvolutionOptions, HorizontalFilterPass, RowFilter, VerticalConvolutionPass,
};
use crate::factory::rgb_u8::vertical_strategy_u8;
use crate::filter_weights::FilterWeights;
use crate::handler_provider::{handle_fixed_row_u8, handle_fixed_rows_4_u8};
use crate::plan::HorizontalFiltering;
use crate::{ImageStore, ThreadingPolicy};
use std::sync::Arc;

impl HorizontalFilterPass<u8, f32, 2> for ImageStore<'_, u8, 2> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        _options: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<u8, 2> + Send + Sync> {
        let _scale_factor = _options.src_size.width as f32 / _options.dst_size.width as f32;
        let mut _dispatcher_4_rows: Option<
            fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i16>, u32),
        > = Some(handle_fixed_rows_4_u8::<2>);
        let mut _dispatcher_1_row: fn(&[u8], &mut [u8], &FilterWeights<i16>, u32) =
            handle_fixed_row_u8::<2>;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::{
                convolve_horizontal_cbcr_neon_row, convolve_horizontal_cbcr_neon_rows_4_u8,
            };
            _dispatcher_4_rows = Some(convolve_horizontal_cbcr_neon_rows_4_u8);
            _dispatcher_1_row = convolve_horizontal_cbcr_neon_row;
            #[cfg(feature = "rdm")]
            if _scale_factor < 8.
                && std::arch::is_aarch64_feature_detected!("rdm")
                && _options.workload_strategy == crate::WorkloadStrategy::PreferSpeed
            {
                use crate::neon::{
                    convolve_horizontal_cbcr_neon_rdm_row,
                    convolve_horizontal_cbcr_neon_rows_rdm_4_u8,
                };
                _dispatcher_4_rows = Some(convolve_horizontal_cbcr_neon_rows_rdm_4_u8);
                _dispatcher_1_row = convolve_horizontal_cbcr_neon_rdm_row;
            }
            #[cfg(feature = "nightly_i8mm")]
            if _scale_factor < 10.
                && std::arch::is_aarch64_feature_detected!("i8mm")
                && _options.workload_strategy == crate::WorkloadStrategy::PreferSpeed
            {
                use crate::neon::{
                    convolve_horizontal_cbcr_neon_dot_row,
                    convolve_horizontal_cbcr_neon_rows_dot_4_u8,
                };
                let dispatcher_4_rows: Option<
                    fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i8>, u32),
                > = Some(convolve_horizontal_cbcr_neon_rows_dot_4_u8);
                let dispatcher_1_row = convolve_horizontal_cbcr_neon_dot_row;
                let i_weights = filter_weights.numerical_approximation_q0_7(0);
                return Arc::new(HorizontalFiltering {
                    filter_weights: i_weights,
                    filter_4_rows: dispatcher_4_rows,
                    filter_row: dispatcher_1_row,
                    threading_policy,
                });
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::sse::{
                    convolve_horizontal_cbcr_sse_row_one, convolve_horizontal_cbcr_sse_rows_4,
                };
                _dispatcher_4_rows = Some(convolve_horizontal_cbcr_sse_rows_4);
                _dispatcher_1_row = convolve_horizontal_cbcr_sse_row_one;
            }
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

impl VerticalConvolutionPass<u8, f32, 2> for ImageStore<'_, u8, 2> {
    fn vertical_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<u8, 2> + Send + Sync> {
        vertical_strategy_u8(filter_weights, threading_policy, options)
    }
}
