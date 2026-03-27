/*
 * Copyright (c) Radzivon Bartoshyk 01/2025. All rights reserved.
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
use crate::factory::plane_u16::default_u16_column_plan;
use crate::filter_weights::FilterWeights;
use crate::plan::HorizontalFiltering;
use crate::{ImageStore, ThreadingPolicy};
use std::sync::Arc;

impl HorizontalFilterPass<u16, f32, 2> for ImageStore<'_, u16, 2> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<u16, 2> + Send + Sync> {
        if options.bit_depth <= 12 {
            let approx =
                filter_weights.numerical_approximation_i16::<{ crate::support::PRECISION }>(0);
            use crate::fixed_point_horizontal::{
                convolve_row_handler_fixed_point, convolve_row_handler_fixed_point_4,
            };
            return Arc::new(HorizontalFiltering {
                filter_weights: approx,
                filter_4_rows: Some(convolve_row_handler_fixed_point_4::<u16, i32, 2>),
                filter_row: convolve_row_handler_fixed_point::<u16, i32, 2>,
                threading_policy,
            });
        }

        use crate::floating_point_horizontal::{
            convolve_row_handler_floating_point, convolve_row_handler_floating_point_4,
        };
        Arc::new(HorizontalFiltering {
            filter_weights,
            filter_4_rows: Some(convolve_row_handler_floating_point_4::<u16, f32, f32, 2>),
            filter_row: convolve_row_handler_floating_point::<u16, f32, f32, 2>,
            threading_policy,
        })
    }
}

impl VerticalConvolutionPass<u16, f32, 2> for ImageStore<'_, u16, 2> {
    fn vertical_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<u16, 2> + Send + Sync> {
        default_u16_column_plan::<2>(filter_weights, threading_policy, options)
    }
}
