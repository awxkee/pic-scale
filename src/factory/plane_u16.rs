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
use crate::filter_weights::{DefaultWeightsConverter, FilterWeights, WeightsConverter};
use crate::plan::{HorizontalFiltering, VerticalFiltering};
use crate::{ImageStore, ThreadingPolicy};
use std::sync::Arc;

impl HorizontalFilterPass<u16, f32, 1> for ImageStore<'_, u16, 1> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<u16, 1> + Send + Sync> {
        if options.bit_depth <= 12 {
            let approx =
                filter_weights.numerical_approximation_i16::<{ crate::support::PRECISION }>(0);
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2") {
                #[cfg(feature = "avx512")]
                {
                    if std::arch::is_x86_feature_detected!("avxvnni") {
                        use crate::avx2::{
                            convolve_horizontal_plane_avx_rows_4_u16_vnni,
                            convolve_horizontal_plane_avx_u16lp_row_vnni,
                        };
                        return Arc::new(HorizontalFiltering {
                            filter_weights: approx,
                            filter_4_rows: Some(convolve_horizontal_plane_avx_rows_4_u16_vnni),
                            filter_row: convolve_horizontal_plane_avx_u16lp_row_vnni,
                            threading_policy,
                        });
                    }
                }
                use crate::avx2::{
                    convolve_horizontal_plane_avx_rows_4_u16,
                    convolve_horizontal_plane_avx_u16lp_row,
                };
                return Arc::new(HorizontalFiltering {
                    filter_weights: approx,
                    filter_4_rows: Some(convolve_horizontal_plane_avx_rows_4_u16),
                    filter_row: convolve_horizontal_plane_avx_u16lp_row,
                    threading_policy,
                });
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::{
                    convolve_horizontal_plane_neon_rows_4_lb_u16,
                    convolve_horizontal_plane_neon_u16_lb_row,
                };
                return Arc::new(HorizontalFiltering {
                    filter_weights: approx,
                    filter_4_rows: Some(convolve_horizontal_plane_neon_rows_4_lb_u16),
                    filter_row: convolve_horizontal_plane_neon_u16_lb_row,
                    threading_policy,
                });
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::fixed_point_horizontal::{
                    convolve_row_handler_fixed_point, convolve_row_handler_fixed_point_4,
                };
                return Arc::new(HorizontalFiltering {
                    filter_weights: approx,
                    filter_4_rows: Some(convolve_row_handler_fixed_point_4::<u16, i32, 1>),
                    filter_row: convolve_row_handler_fixed_point::<u16, i32, 1>,
                    threading_policy,
                });
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let has_avx = std::arch::is_x86_feature_detected!("avx2");
            let has_fma = std::arch::is_x86_feature_detected!("fma");
            if has_avx {
                if has_fma {
                    use crate::avx2::{
                        convolve_horizontal_plane_avx_rows_4_u16_fma,
                        convolve_horizontal_plane_avx_u16_row_fma,
                    };
                    return Arc::new(HorizontalFiltering {
                        filter_weights,
                        filter_4_rows: Some(convolve_horizontal_plane_avx_rows_4_u16_fma),
                        filter_row: convolve_horizontal_plane_avx_u16_row_fma,
                        threading_policy,
                    });
                }
                use crate::avx2::{
                    convolve_horizontal_plane_avx_rows_4_u16_default,
                    convolve_horizontal_plane_avx_u16_row_default,
                };
                return Arc::new(HorizontalFiltering {
                    filter_weights,
                    filter_4_rows: Some(convolve_horizontal_plane_avx_rows_4_u16_default),
                    filter_row: convolve_horizontal_plane_avx_u16_row_default,
                    threading_policy,
                });
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "rdm")]
            {
                let has_rdm = std::arch::is_aarch64_feature_detected!("rdm");
                if has_rdm {
                    use crate::neon::{
                        convolve_horizontal_plane_neon_rows_4_hb_u16,
                        convolve_horizontal_plane_neon_u16_hb_row,
                    };
                    let approx_num = filter_weights.numerical_approximation::<i32, 31>(0);

                    return Arc::new(HorizontalFiltering {
                        filter_weights: approx_num,
                        filter_4_rows: Some(convolve_horizontal_plane_neon_rows_4_hb_u16),
                        filter_row: convolve_horizontal_plane_neon_u16_hb_row,
                        threading_policy,
                    });
                }
            }
            use crate::neon::{
                convolve_horizontal_plane_neon_f32_u16_row,
                convolve_horizontal_plane_neon_rows_4_f32_u16,
            };

            Arc::new(HorizontalFiltering {
                filter_weights,
                filter_4_rows: Some(convolve_horizontal_plane_neon_rows_4_f32_u16),
                filter_row: convolve_horizontal_plane_neon_f32_u16_row,
                threading_policy,
            })
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::floating_point_horizontal::{
                convolve_row_handler_floating_point, convolve_row_handler_floating_point_4,
            };
            Arc::new(HorizontalFiltering {
                filter_weights,
                filter_4_rows: Some(convolve_row_handler_floating_point_4::<u16, f32, f32, 1>),
                filter_row: convolve_row_handler_floating_point::<u16, f32, f32, 1>,
                threading_policy,
            })
        }
    }
}

impl VerticalConvolutionPass<u16, f32, 1> for ImageStore<'_, u16, 1> {
    fn vertical_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<u16, 1> + Send + Sync> {
        default_u16_column_plan::<1>(filter_weights, threading_policy, options)
    }
}

pub(crate) fn default_u16_column_plan<const CN: usize>(
    filter_weights: FilterWeights<f32>,
    threading_policy: ThreadingPolicy,
    options: ConvolutionOptions,
) -> Arc<dyn ColumnFilter<u16, CN> + Send + Sync> {
    if options.bit_depth > 12 {
        #[cfg(all(target_arch = "aarch64", feature = "sve"))]
        {
            if options.bit_depth <= 15 {
                use crate::sve2::convolve_vertical_sve2_u16_dot;
                let filter_weights =
                    DefaultWeightsConverter::default().prepare_weights(&filter_weights);
                if std::arch::is_aarch64_feature_detected!("sve2") {
                    return Arc::new(VerticalFiltering {
                        filter_weights,
                        threading_policy,
                        filter_row: convolve_vertical_sve2_u16_dot,
                    });
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "rdm"))]
        {
            if std::arch::is_aarch64_feature_detected!("rdm")
                && options.workload_strategy == crate::WorkloadStrategy::PreferSpeed
            {
                let filter_weights = filter_weights.numerical_approximation::<i32, 31>(0);
                use crate::neon::convolve_column_hb_u16;
                return Arc::new(VerticalFiltering {
                    filter_weights,
                    filter_row: convolve_column_hb_u16,
                    threading_policy,
                });
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::convolve_column_u16;
            Arc::new(VerticalFiltering {
                filter_weights,
                filter_row: convolve_column_u16,
                threading_policy,
            })
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    if std::arch::is_x86_feature_detected!("fma") {
                        use crate::avx2::convolve_column_avx_u16_fma;
                        return Arc::new(VerticalFiltering {
                            filter_weights,
                            filter_row: convolve_column_avx_u16_fma,
                            threading_policy,
                        });
                    }
                    use crate::avx2::convolve_column_avx_u16_default;
                    return Arc::new(VerticalFiltering {
                        filter_weights,
                        filter_row: convolve_column_avx_u16_default,
                        threading_policy,
                    });
                }
            }
            #[cfg(feature = "sse")]
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::sse::convolve_column_sse_u16;
                return Arc::new(VerticalFiltering {
                    filter_weights,
                    filter_row: convolve_column_sse_u16,
                    threading_policy,
                });
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::floating_point_vertical::column_handler_floating_point;
            Arc::new(VerticalFiltering {
                filter_weights,
                filter_row: column_handler_floating_point::<u16, f32, f32>,
                threading_policy,
            })
        }
    } else {
        let filter_weights = DefaultWeightsConverter::default().prepare_weights(&filter_weights);
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::convolve_column_lb_u16;
            Arc::new(VerticalFiltering {
                filter_weights,
                threading_policy,
                filter_row: convolve_column_lb_u16,
            })
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                #[cfg(feature = "avx512")]
                if std::arch::is_x86_feature_detected!("avxvnni") {
                    use crate::avx2::convolve_column_lb_avx2_u16_vnni;
                    return Arc::new(VerticalFiltering {
                        filter_weights,
                        threading_policy,
                        filter_row: convolve_column_lb_avx2_u16_vnni,
                    });
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    use crate::avx2::convolve_column_lb_avx2_u16;
                    return Arc::new(VerticalFiltering {
                        filter_weights,
                        threading_policy,
                        filter_row: convolve_column_lb_avx2_u16,
                    });
                }
            }
            #[cfg(feature = "sse")]
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::sse::convolve_column_lb_sse_u16;
                return Arc::new(VerticalFiltering {
                    filter_weights,
                    threading_policy,
                    filter_row: convolve_column_lb_sse_u16,
                });
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::fixed_point_vertical::column_handler_fixed_point;
            Arc::new(VerticalFiltering {
                filter_weights,
                threading_policy,
                filter_row: column_handler_fixed_point::<u16, i32>,
            })
        }
    }
}
