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
#![allow(clippy::type_complexity)]

use crate::convolution::{ConvolutionOptions, Filtering};
use crate::filter_weights::{
    DefaultWeightsConverter, FilterWeights, WeightsConverter,
};
use crate::handler_provider::{
    ColumnHandlerFixedPoint, ColumnHandlerFloatingPoint, RowHandlerFixedPoint,
    RowHandlerFloatingPoint,
};
use crate::plan::{HorizontalFiltering, VerticalFiltering};
use crate::support::PRECISION;
use crate::ThreadingPolicy;
use std::sync::Arc;

pub(crate) trait RowFactoryProducer {
    fn make_plan<const CN: usize>(
        weights: &FilterWeights<f32>,
        bit_depth: usize,
        threading_policy: ThreadingPolicy,
    ) -> Arc<dyn Filtering<u16, CN> + Send + Sync>;
}

impl RowFactoryProducer for u16 {
    fn make_plan<const CN: usize>(
        weights: &FilterWeights<f32>,
        bit_depth: usize,
        threading_policy: ThreadingPolicy,
    ) -> Arc<dyn Filtering<u16, CN> + Send + Sync> {
        if bit_depth < 12 {
            let approx = weights.numerical_approximation_i16::<PRECISION>(0);
            return Arc::new(HorizontalFiltering {
                filter_weights: approx,
                filter_4_rows: Some(u16::handle_fixed_row_4::<i32, CN>),
                filter_row: u16::handle_fixed_row::<i32, CN>,
                threading_policy,
            });
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "rdm"))]
        {
            let has_rdm = std::arch::is_aarch64_feature_detected!("rdm");
            if has_rdm && CN == 4 {
                use crate::neon::{
                    convolve_horizontal_rgba_neon_rows_4_hb_u16,
                    convolve_horizontal_rgba_neon_u16_hb_row,
                };
                let approx_num = weights.numerical_approximation::<i32, 31>(0);
                return Arc::new(HorizontalFiltering {
                    filter_weights: approx_num,
                    filter_4_rows: Some(convolve_horizontal_rgba_neon_rows_4_hb_u16),
                    filter_row: convolve_horizontal_rgba_neon_u16_hb_row,
                    threading_policy,
                });
            } else if has_rdm && CN == 3 {
                use crate::neon::{
                    convolve_horizontal_rgb_neon_rows_4_hb_u16,
                    convolve_horizontal_rgb_neon_u16_hb_row,
                };
                let approx_num = weights.numerical_approximation::<i32, 31>(0);
                return Arc::new(HorizontalFiltering {
                    filter_weights: approx_num,
                    filter_4_rows: Some(convolve_horizontal_rgb_neon_rows_4_hb_u16),
                    filter_row: convolve_horizontal_rgb_neon_u16_hb_row,
                    threading_policy,
                });
            } else if has_rdm && CN == 1 {
                use crate::neon::{
                    convolve_horizontal_plane_neon_rows_4_hb_u16,
                    convolve_horizontal_plane_neon_u16_hb_row,
                };
                let approx_num = weights.numerical_approximation::<i32, 31>(0);

                return Arc::new(HorizontalFiltering {
                    filter_weights: approx_num,
                    filter_4_rows: Some(convolve_horizontal_plane_neon_rows_4_hb_u16),
                    filter_row: convolve_horizontal_plane_neon_u16_hb_row,
                    threading_policy,
                });
            }
        }
        Arc::new(HorizontalFiltering {
            filter_weights: weights.clone(),
            filter_4_rows: Some(u16::handle_row_4::<CN>),
            filter_row: u16::handle_row::<CN>,
            threading_policy,
        })
    }
}

pub(crate) fn vertical_plan_u16<const CN: usize>(
    filter_weights: FilterWeights<f32>,
    threading_policy: ThreadingPolicy,
    _options: ConvolutionOptions,
) -> Arc<dyn Filtering<u16, CN> + Send + Sync> {
    if _options.bit_depth > 12 {
        #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "rdm"))]
        {
            if DefaultHighBitDepthHighHandlerNeon::is_available()
                && _options.workload_strategy == crate::WorkloadStrategy::PreferSpeed
            {
                let filter_weights =
                    WeightsConverterQ0_31::default().prepare_weights(&filter_weights);
                use crate::neon::convolve_column_hb_u16;
                return Arc::new(VerticalFiltering {
                    filter_weights,
                    filter_row: convolve_column_hb_u16,
                    threading_policy,
                });
            }
        }
        Arc::new(VerticalFiltering {
            filter_weights,
            filter_row: u16::handle_floating_column,
            threading_policy,
        })
    } else {
        let filter_weights = DefaultWeightsConverter::default().prepare_weights(&filter_weights);
        Arc::new(VerticalFiltering {
            filter_weights,
            threading_policy,
            filter_row: u16::handle_fixed_column::<i32, CN>,
        })
    }
}


#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "rdm"))]
#[derive(Default)]
pub(crate) struct WeightsConverterQ0_31 {}

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "rdm"))]
impl WeightsConverter<i32> for WeightsConverterQ0_31 {
    fn prepare_weights(&self, weights: &FilterWeights<f32>) -> FilterWeights<i32> {
        weights.numerical_approximation::<i32, 31>(0)
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "rdm"))]
#[derive(Default)]
struct DefaultHighBitDepthHighHandlerNeon {}

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "rdm"))]
impl DefaultHighBitDepthHighHandlerNeon {
    fn is_available() -> bool {
        std::arch::is_aarch64_feature_detected!("rdm")
    }
}