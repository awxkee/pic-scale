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
use crate::convolve_naive_f32::{
    convolve_horizontal_4_row_f32_f64, convolve_horizontal_native_row_f32,
    convolve_horizontal_native_row_f32_f64, convolve_horizontal_rgba_4_row_f32,
};
use crate::factory::rgb_f32::{
    convolve_vertical_rgb_native_row_f32, convolve_vertical_rgb_native_row_f64,
};
use crate::filter_weights::{FilterBounds, FilterWeights};
#[cfg(all(target_arch = "aarch64", feature = "neon",))]
use crate::neon::{
    convolve_horizontal_plane_neon_row_one, convolve_horizontal_plane_neon_rows_4,
    convolve_vertical_rgb_neon_row_f32,
};
use crate::plan::{HorizontalFiltering, VerticalFiltering};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::{
    convolve_horizontal_plane_sse_row_one, convolve_horizontal_plane_sse_rows_4,
    convolve_vertical_rgb_sse_row_f32,
};
use crate::{ImageStore, ThreadingPolicy};
use std::sync::Arc;

impl HorizontalFilterPass<f32, f32, 1> for ImageStore<'_, f32, 1> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<f32, 1> + Send + Sync> {
        let mut _dispatcher_4_rows: Option<
            fn(&[f32], usize, &mut [f32], usize, &FilterWeights<f32>, u32),
        > = Some(convolve_horizontal_rgba_4_row_f32::<1>);
        let mut _dispatcher_row: fn(&[f32], &mut [f32], &FilterWeights<f32>, u32) =
            convolve_horizontal_native_row_f32::<1>;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_plane_neon_rows_4);
            _dispatcher_row = convolve_horizontal_plane_neon_row_one;
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_plane_sse_rows_4);
                _dispatcher_row = convolve_horizontal_plane_sse_row_one;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::avx2::{
                convolve_horizontal_plane_avx_row_one_f32_default,
                convolve_horizontal_plane_avx_row_one_f32_fma,
                convolve_horizontal_plane_avx_rows_4_f32_default,
                convolve_horizontal_plane_avx_rows_4_f32_fma,
            };
            let has_fma = std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx2") {
                _dispatcher_4_rows = Some(convolve_horizontal_plane_avx_rows_4_f32_default);
                _dispatcher_row = convolve_horizontal_plane_avx_row_one_f32_default;
                if has_fma {
                    _dispatcher_4_rows = Some(convolve_horizontal_plane_avx_rows_4_f32_fma);
                    _dispatcher_row = convolve_horizontal_plane_avx_row_one_f32_fma;
                }
            }
        }
        Arc::new(HorizontalFiltering {
            filter_weights,
            filter_4_rows: _dispatcher_4_rows,
            filter_row: _dispatcher_row,
            _threading_policy: threading_policy,
        })
    }
}

impl HorizontalFilterPass<f32, f64, 1> for ImageStore<'_, f32, 1> {
    fn horizontal_plan(
        filter_weights: FilterWeights<f64>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<f32, 1> + Send + Sync> {
        let mut _dispatcher_4_rows: Option<
            fn(&[f32], usize, &mut [f32], usize, &FilterWeights<f64>, u32),
        > = Some(convolve_horizontal_4_row_f32_f64::<1>);
        let mut _dispatcher_row: fn(&[f32], &mut [f32], &FilterWeights<f64>, u32) =
            convolve_horizontal_native_row_f32_f64::<1>;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
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
                    convolve_hor_plane_avx_row_one_f32_f64_default,
                    convolve_hor_plane_avx_row_one_f32_f64_fma,
                    convolve_hor_plane_avx_rows_4_f32_f64_default,
                    convolve_hor_plane_avx_rows_4_f32_f64_fma,
                };
                _dispatcher_4_rows = Some(convolve_hor_plane_avx_rows_4_f32_f64_default);
                _dispatcher_row = convolve_hor_plane_avx_row_one_f32_f64_default;
                if std::arch::is_x86_feature_detected!("fma") {
                    _dispatcher_4_rows = Some(convolve_hor_plane_avx_rows_4_f32_f64_fma);
                    _dispatcher_row = convolve_hor_plane_avx_row_one_f32_f64_fma;
                }
            }
        }
        Arc::new(HorizontalFiltering {
            filter_weights,
            filter_4_rows: _dispatcher_4_rows,
            filter_row: _dispatcher_row,
            _threading_policy: threading_policy,
        })
    }
}

impl VerticalConvolutionPass<f32, f32, 1> for ImageStore<'_, f32, 1> {
    fn vertical_plan(
        filter_weights: FilterWeights<f32>,
        threading_policy: ThreadingPolicy,
        _: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<f32, 1> + Send + Sync> {
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
        let filter = VerticalFiltering {
            filter_weights,
            filter_row: _dispatcher,
            threading_policy,
        };
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        return crate::neon::paired_vertical_f32(filter);
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        Arc::new(filter)
    }
}

impl VerticalConvolutionPass<f32, f64, 1> for ImageStore<'_, f32, 1> {
    fn vertical_plan(
        filter_weights: FilterWeights<f64>,
        threading_policy: ThreadingPolicy,
        _options: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<f32, 1> + Send + Sync> {
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
        if _options.src_size.width >= 8 && std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::{
                convolve_vertical_avx_row_f32_f64_default, convolve_vertical_avx_row_f32_f64_fma,
            };
            // Match the scalar path's compile-time choice of fused arithmetic.
            _dispatcher = if cfg!(target_feature = "fma") {
                convolve_vertical_avx_row_f32_f64_fma
            } else {
                convolve_vertical_avx_row_f32_f64_default
            };
        }
        let filter = VerticalFiltering {
            filter_weights,
            filter_row: _dispatcher,
            threading_policy,
        };
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        return crate::neon::paired_vertical_f64(filter);
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        Arc::new(filter)
    }
}

#[cfg(all(test, target_arch = "x86_64", feature = "avx"))]
mod tests {
    use super::*;
    use crate::{ImageSize, WorkloadStrategy};

    fn planar_quality_vertical_plan(
        width: usize,
        weights: [f64; 2],
    ) -> Arc<dyn ColumnFilter<f32, 1> + Send + Sync> {
        let filter =
            FilterWeights::new(weights.to_vec(), 2, 2, 1, 2, vec![FilterBounds::new(0, 2)]);
        <ImageStore<'_, f32, 1> as VerticalConvolutionPass<f32, f64, 1>>::vertical_plan(
            filter,
            ThreadingPolicy::Single,
            ConvolutionOptions {
                workload_strategy: WorkloadStrategy::PreferQuality,
                bit_depth: 0,
                src_size: ImageSize::new(width, 2),
                dst_size: ImageSize::new(width, 1),
            },
        )
    }

    #[test]
    fn planar_quality_vertical_preserves_scalar_cancellation() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        // This width exercises both vector block sizes and the single-element tail.
        let width = 25;
        let stride = width + 3;
        let mut source = vec![1.0; stride + width];
        source[stride..].fill(3.0);
        let weights = [-(3.0 + 4.0 * f64::EPSILON), 1.0 + f64::EPSILON];
        let plan = planar_quality_vertical_plan(width, weights);
        let mut output = vec![f32::NAN; width];
        plan.run_on_row(&source, &mut output, width, stride, 0, 0);
        let expected = if cfg!(target_feature = "fma") {
            -f64::EPSILON as f32
        } else {
            0.0
        };
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == expected.to_bits())
        );
    }

    #[test]
    fn planar_quality_vertical_propagates_nans() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let width = 25;
        let stride = width + 3;
        let weights = [0.25, 0.75];
        let plan = planar_quality_vertical_plan(width, weights);
        // Test both vector block sizes and the scalar tail.
        for nan_column in [0, 16, 24] {
            let mut source = vec![1.0; stride + width];
            source[nan_column] = f32::from_bits(0xffc0_5678);
            source[stride + nan_column] = f32::from_bits(0x7fc0_9a20);
            let mut expected = vec![0.0; width];
            convolve_vertical_rgb_native_row_f64(
                width,
                &FilterBounds::new(0, 2),
                &source,
                &mut expected,
                stride,
                &weights,
                0,
            );
            assert!(expected[nan_column].is_nan());

            let mut output = vec![0.0; width];
            plan.run_on_row(&source, &mut output, width, stride, 0, 0);
            assert!(
                output
                    .iter()
                    .zip(&expected)
                    .all(|(actual, expected)| if expected.is_nan() {
                        actual.is_nan()
                    } else {
                        actual.to_bits() == expected.to_bits()
                    })
            );
        }
    }
}
