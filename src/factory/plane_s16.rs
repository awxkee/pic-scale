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
use crate::filter_weights::{DefaultWeightsConverter, FilterWeights, WeightsConverter};
use crate::floating_point_horizontal::{
    convolve_row_handler_floating_point, convolve_row_handler_floating_point_4,
};
use crate::floating_point_vertical::column_handler_floating_point;
use crate::plan::{HorizontalFiltering, VerticalFiltering};
use crate::{ImageStore, ThreadingPolicy};
use std::sync::Arc;

impl HorizontalFilterPass<i16, f32, 1> for ImageStore<'_, i16, 1> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<i16, 1> + Send + Sync> {
        if options.bit_depth > 12 {
            return Arc::new(HorizontalFiltering {
                filter_weights,
                filter_4_rows: Some(convolve_row_handler_floating_point_4::<i16, f32, f32, 1>),
                filter_row: convolve_row_handler_floating_point::<i16, f32, f32, 1>,
                threading_policy,
            });
        }
        let weights_q0_15 = DefaultWeightsConverter::default().prepare_weights(&filter_weights);
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::{
                convolve_horizontal_plane_neon_i16_lb_row,
                convolve_horizontal_plane_neon_rows_4_lb_i16,
            };
            Arc::new(HorizontalFiltering {
                filter_weights: weights_q0_15,
                filter_4_rows: Some(convolve_horizontal_plane_neon_rows_4_lb_i16),
                filter_row: convolve_horizontal_plane_neon_i16_lb_row,
                threading_policy,
            })
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");
            if has_avx {
                use crate::avx2::{
                    convolve_horizontal_plane_avx_i16lp_row,
                    convolve_horizontal_plane_avx_rows_4_i16,
                };
                return Arc::new(HorizontalFiltering {
                    filter_weights: weights_q0_15,
                    filter_4_rows: Some(convolve_horizontal_plane_avx_rows_4_i16),
                    filter_row: convolve_horizontal_plane_avx_i16lp_row,
                    threading_policy,
                });
            }
        }

        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::fixed_point_horizontal::{
                convolve_row_handler_fixed_point, convolve_row_handler_fixed_point_4,
            };
            Arc::new(HorizontalFiltering {
                filter_weights: weights_q0_15,
                filter_4_rows: Some(convolve_row_handler_fixed_point_4::<i16, i32, 1>),
                filter_row: convolve_row_handler_fixed_point::<i16, i32, 1>,
                threading_policy,
            })
        }
    }
}

impl VerticalConvolutionPass<i16, f32, 1> for ImageStore<'_, i16, 1> {
    fn vertical_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<i16, 1> + Send + Sync> {
        if options.bit_depth > 12 {
            return Arc::new(VerticalFiltering {
                filter_weights,
                threading_policy,
                filter_row: column_handler_floating_point::<i16, f32, f32>,
            });
        }
        let filter_weights = DefaultWeightsConverter::default().prepare_weights(&filter_weights);
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::convolve_column_lb_i16;
            Arc::new(VerticalFiltering {
                filter_weights,
                threading_policy,
                filter_row: convolve_column_lb_i16,
            })
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");
            if has_avx {
                use crate::avx2::convolve_column_lb_avx2_s16;
                return Arc::new(VerticalFiltering {
                    filter_weights,
                    threading_policy,
                    filter_row: convolve_column_lb_avx2_s16,
                });
            }
        }

        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::fixed_point_vertical::column_handler_fixed_point;
            Arc::new(VerticalFiltering {
                filter_weights,
                threading_policy,
                filter_row: column_handler_fixed_point::<i16, i32>,
            })
        }
    }
}
