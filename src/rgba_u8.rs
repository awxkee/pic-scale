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
#![forbid(unsafe_code)]

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx2::{
    convolve_horizontal_rgba_avx_rows_4_lb, convolve_horizontal_rgba_avx_rows_one_lb,
    convolve_vertical_avx_row, convolve_vertical_avx_row_lp,
};
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::dispatch_group_u8::{convolve_horizontal_dispatch_u8, convolve_vertical_dispatch_u8};
use crate::filter_weights::{DefaultWeightsConverter, FilterBounds, FilterWeights};
use crate::handler_provider::{
    handle_fixed_column_u8, handle_fixed_row_u8, handle_fixed_rows_4_u8,
};
use crate::image_store::ImageStoreMut;
#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::neon::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{
    convolve_horizontal_rgba_sse_rows_4, convolve_horizontal_rgba_sse_rows_4_lb,
    convolve_horizontal_rgba_sse_rows_one, convolve_horizontal_rgba_sse_rows_one_lb,
    convolve_vertical_sse_row, convolve_vertical_sse_row_lp,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::wasm_vertical_neon_row;
use crate::ImageStore;
use rayon::ThreadPool;

impl HorizontalConvolutionPass<u8, 4> for ImageStore<'_, u8, 4> {
    #[allow(clippy::type_complexity)]
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<u8, 4>,
        _pool: &Option<ThreadPool>,
    ) {
        let _scale_factor = self.width as f32 / destination.width as f32;
        #[cfg(all(
            feature = "nightly_i8mm",
            target_arch = "aarch64",
            target_feature = "neon"
        ))]
        {
            if _scale_factor < 6. && std::arch::is_aarch64_feature_detected!("i8mm") {
                use crate::filter_weights::WeightsConverterQ7;
                use crate::neon::{
                    convolve_horizontal_rgba_neon_row_dot, convolve_horizontal_rgba_neon_rows_4_dot,
                };
                let _dispatcher_4_rows: Option<
                    fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i8>),
                > = Some(convolve_horizontal_rgba_neon_rows_4_dot);
                let _dispatcher_1_row: fn(&[u8], &mut [u8], &FilterWeights<i8>) =
                    convolve_horizontal_rgba_neon_row_dot;
                convolve_horizontal_dispatch_u8(
                    self,
                    filter_weights,
                    destination,
                    _pool,
                    _dispatcher_4_rows,
                    _dispatcher_1_row,
                    WeightsConverterQ7::default(),
                );
                return;
            }
        }
        #[cfg(all(
            feature = "nightly_avx512",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        {
            let has_avxvnni = std::arch::is_x86_feature_detected!("avxvnni");
            if _scale_factor < 6. && has_avxvnni {
                use crate::avx512::{
                    convolve_horizontal_rgba_vnni_row_dot, convolve_horizontal_rgba_vnni_rows_4_dot,
                };
                use crate::filter_weights::WeightsConverterQ7;
                let _dispatcher_4_rows: Option<
                    fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i8>),
                > = Some(convolve_horizontal_rgba_vnni_rows_4_dot);
                let _dispatcher_1_row: fn(&[u8], &mut [u8], &FilterWeights<i8>) =
                    convolve_horizontal_rgba_vnni_row_dot;
                convolve_horizontal_dispatch_u8(
                    self,
                    filter_weights,
                    destination,
                    _pool,
                    _dispatcher_4_rows,
                    _dispatcher_1_row,
                    WeightsConverterQ7::default(),
                );
                return;
            }
        }
        let mut _dispatcher_4_rows: Option<
            fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i16>),
        > = Some(handle_fixed_rows_4_u8::<4>);
        let mut _dispatcher_1_row: fn(&[u8], &mut [u8], &FilterWeights<i16>) =
            handle_fixed_row_u8::<4>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if _scale_factor < 8. && crate::cpu_features::is_aarch_rdm_supported() {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_neon_rows_4_u8_i16);
                _dispatcher_1_row = convolve_horizontal_rgba_neon_row_i16;
            } else {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_neon_rows_4_u8);
                _dispatcher_1_row = convolve_horizontal_rgba_neon_row;
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                if _scale_factor < 8. {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgba_sse_rows_4_lb);
                    _dispatcher_1_row = convolve_horizontal_rgba_sse_rows_one_lb;
                } else {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgba_sse_rows_4);
                    _dispatcher_1_row = convolve_horizontal_rgba_sse_rows_one;
                }
            }
            if std::arch::is_x86_feature_detected!("avx2") || _scale_factor < 8. {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_avx_rows_4_lb);
                _dispatcher_1_row = convolve_horizontal_rgba_avx_rows_one_lb;
            }
        }
        convolve_horizontal_dispatch_u8(
            self,
            filter_weights,
            destination,
            _pool,
            _dispatcher_4_rows,
            _dispatcher_1_row,
            DefaultWeightsConverter::default(),
        );
    }
}

impl VerticalConvolutionPass<u8, 4> for ImageStore<'_, u8, 4> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<u8, 4>,
        pool: &Option<ThreadPool>,
    ) {
        let _scale_factor = self.height as f32 / destination.height as f32;
        #[allow(clippy::type_complexity)]
        let mut _dispatcher: fn(usize, &FilterBounds, &[u8], &mut [u8], usize, &[i16]) =
            handle_fixed_column_u8;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            // For more downscaling better to use more precise version
            if _scale_factor < 8. && crate::cpu_features::is_aarch_rdm_supported() {
                _dispatcher = convolve_vertical_neon_i16_precision;
            } else {
                _dispatcher = convolve_vertical_neon_i32_precision;
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                if _scale_factor < 8. {
                    _dispatcher = convolve_vertical_sse_row_lp;
                } else {
                    _dispatcher = convolve_vertical_sse_row;
                }
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                if _scale_factor < 8. {
                    _dispatcher = convolve_vertical_avx_row_lp;
                } else {
                    _dispatcher = convolve_vertical_avx_row;
                }
            }
            #[cfg(feature = "nightly_avx512")]
            if std::arch::is_x86_feature_detected!("avx512bw") {
                if _scale_factor < 8. {
                    use crate::avx512::convolve_vertical_avx512_row_lp;
                    _dispatcher = convolve_vertical_avx512_row_lp;
                }
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            _dispatcher = wasm_vertical_neon_row;
        }
        convolve_vertical_dispatch_u8(self, filter_weights, destination, pool, _dispatcher);
    }
}
