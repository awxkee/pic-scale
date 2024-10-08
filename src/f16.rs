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

// RGBA

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx2::{
    convolve_horizontal_rgba_avx_row_one_f16, convolve_horizontal_rgba_avx_rows_4_f16,
    convolve_vertical_avx_row_f16,
};
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_f32::{
    convolve_horizontal_rgb_native_row, convolve_horizontal_rgba_4_row_f32,
};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::cpu_features::{is_aarch_f16_supported, is_aarch_f16c_supported};
use crate::dispatch_group_f16::{convolve_horizontal_dispatch_f16, convolve_vertical_dispatch_f16};
use crate::filter_weights::{FilterBounds, FilterWeights};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    convolve_horizontal_rgb_neon_row_one_f16, convolve_horizontal_rgb_neon_rows_4_f16,
    convolve_horizontal_rgba_neon_row_one_f16, convolve_horizontal_rgba_neon_rows_4_f16,
    convolve_vertical_rgb_neon_row_f16,
};
#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::neon::{
    xconvolve_horizontal_rgb_neon_row_one_f16, xconvolve_horizontal_rgb_neon_rows_4_f16,
    xconvolve_horizontal_rgba_neon_row_one_f16, xconvolve_horizontal_rgba_neon_rows_4_f16,
    xconvolve_vertical_rgb_neon_row_f16,
};
use crate::rgb_f32::convolve_vertical_rgb_native_row_f32;
#[cfg(all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    feature = "riscv"
))]
use crate::risc::{
    convolve_horizontal_rgba_risc_row_one_f16, convolve_horizontal_rgba_risc_rows_4_f16,
    convolve_vertical_risc_row_f16, risc_is_features_supported,
};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{
    convolve_horizontal_rgb_sse_row_one_f16, convolve_horizontal_rgb_sse_rows_4_f16,
    convolve_horizontal_rgba_sse_row_one_f16, convolve_horizontal_rgba_sse_rows_4_f16,
    convolve_vertical_sse_row_f16,
};
use crate::ImageStore;
use half::f16;
use rayon::ThreadPool;

impl<'a> HorizontalConvolutionPass<f16, 4> for ImageStore<'a, f16, 4> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f16, 4>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, *const f16, usize, *mut f16, usize),
        > = Some(convolve_horizontal_rgba_4_row_f32::<f16, f32, 4>);
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f16, *mut f16) =
            convolve_horizontal_rgb_native_row::<f16, f32, 4>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if is_aarch_f16c_supported() {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_neon_rows_4_f16);
                _dispatcher_row = convolve_horizontal_rgba_neon_row_one_f16;
                if is_aarch_f16_supported() {
                    _dispatcher_4_rows = Some(xconvolve_horizontal_rgba_neon_rows_4_f16);
                    _dispatcher_row = xconvolve_horizontal_rgba_neon_row_one_f16;
                }
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let is_f16c_available = is_x86_feature_detected!("f16c");
            let fma_available = is_x86_feature_detected!("fma");
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_sse_rows_4_f16::<false, false>);
                _dispatcher_row = convolve_horizontal_rgba_sse_row_one_f16::<false, false>;
                if is_f16c_available {
                    _dispatcher_4_rows =
                        Some(convolve_horizontal_rgba_sse_rows_4_f16::<true, false>);
                    _dispatcher_row = convolve_horizontal_rgba_sse_row_one_f16::<true, false>;
                    if fma_available {
                        _dispatcher_4_rows =
                            Some(convolve_horizontal_rgba_sse_rows_4_f16::<true, true>);
                        _dispatcher_row = convolve_horizontal_rgba_sse_row_one_f16::<true, true>;
                    }
                }
            }
            if is_x86_feature_detected!("avx2") && is_f16c_available {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_avx_rows_4_f16::<false>);
                _dispatcher_row = convolve_horizontal_rgba_avx_row_one_f16::<false>;
                if fma_available {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgba_avx_rows_4_f16::<true>);
                    _dispatcher_row = convolve_horizontal_rgba_avx_row_one_f16::<true>;
                }
            }
        }
        #[cfg(all(
            any(target_arch = "riscv64", target_arch = "riscv32"),
            feature = "riscv"
        ))]
        {
            let features: [String; 2] = ["zvfh".parse().unwrap(), "zfh".parse().unwrap()];
            if std::arch::is_riscv_feature_detected!("v") && risc_is_features_supported(&features) {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_risc_rows_4_f16);
                _dispatcher_row = convolve_horizontal_rgba_risc_row_one_f16;
            }
        }
        convolve_horizontal_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher_4_rows,
            _dispatcher_row,
        );
    }
}

impl<'a> VerticalConvolutionPass<f16, 4> for ImageStore<'a, f16, 4> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f16, 4>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(usize, &FilterBounds, *const f16, *mut f16, usize, &[f32]) =
            convolve_vertical_rgb_native_row_f32::<f16, 4>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if is_aarch_f16c_supported() {
                _dispatcher = convolve_vertical_rgb_neon_row_f16::<4>;
                if is_aarch_f16_supported() {
                    _dispatcher = xconvolve_vertical_rgb_neon_row_f16::<4>;
                }
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let is_f16c_available = is_x86_feature_detected!("f16c");
            let is_fma_available = is_x86_feature_detected!("fma");
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_sse_row_f16::<4, false, false>;
                if is_f16c_available {
                    if is_fma_available {
                        _dispatcher = convolve_vertical_sse_row_f16::<4, true, true>;
                    } else {
                        _dispatcher = convolve_vertical_sse_row_f16::<4, true, false>;
                    }
                }
            }

            if is_x86_feature_detected!("avx2") && is_f16c_available {
                _dispatcher = convolve_vertical_avx_row_f16::<4, false>;
                if is_fma_available {
                    _dispatcher = convolve_vertical_avx_row_f16::<4, true>;
                }
            }
        }
        #[cfg(all(
            any(target_arch = "riscv64", target_arch = "riscv32"),
            feature = "riscv"
        ))]
        {
            let features: [String; 2] = ["zvfh".parse().unwrap(), "zfh".parse().unwrap()];
            if std::arch::is_riscv_feature_detected!("v") && risc_is_features_supported(&features) {
                _dispatcher = convolve_vertical_risc_row_f16::<4>;
            }
        }
        convolve_vertical_dispatch_f16(self, filter_weights, destination, pool, _dispatcher);
    }
}

impl<'a> HorizontalConvolutionPass<f16, 3> for ImageStore<'a, f16, 3> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f16, 3>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, *const f16, usize, *mut f16, usize),
        > = Some(convolve_horizontal_rgba_4_row_f32::<f16, f32, 3>);
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f16, *mut f16) =
            convolve_horizontal_rgb_native_row::<f16, f32, 3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if is_aarch_f16c_supported() {
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4_f16);
                _dispatcher_row = convolve_horizontal_rgb_neon_row_one_f16;
                if is_aarch_f16_supported() {
                    _dispatcher_4_rows = Some(xconvolve_horizontal_rgb_neon_rows_4_f16);
                    _dispatcher_row = xconvolve_horizontal_rgb_neon_row_one_f16;
                }
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_sse_rows_4_f16::<false, false>);
                _dispatcher_row = convolve_horizontal_rgb_sse_row_one_f16::<false, false>;
                if is_x86_feature_detected!("f16c") {
                    if is_x86_feature_detected!("fma") {
                        _dispatcher_4_rows =
                            Some(convolve_horizontal_rgb_sse_rows_4_f16::<true, true>);
                        _dispatcher_row = convolve_horizontal_rgb_sse_row_one_f16::<true, true>;
                    } else {
                        _dispatcher_4_rows =
                            Some(convolve_horizontal_rgb_sse_rows_4_f16::<true, false>);
                        _dispatcher_row = convolve_horizontal_rgb_sse_row_one_f16::<true, false>;
                    }
                }
            }
        }
        convolve_horizontal_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher_4_rows,
            _dispatcher_row,
        );
    }
}

impl<'a> VerticalConvolutionPass<f16, 3> for ImageStore<'a, f16, 3> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f16, 3>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(usize, &FilterBounds, *const f16, *mut f16, usize, &[f32]) =
            convolve_vertical_rgb_native_row_f32::<f16, 3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if is_aarch_f16c_supported() {
                _dispatcher = convolve_vertical_rgb_neon_row_f16::<3>;
                if is_aarch_f16_supported() {
                    _dispatcher = xconvolve_vertical_rgb_neon_row_f16::<3>;
                }
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let is_f16c_available = is_x86_feature_detected!("f16c");
            let is_fma_available = is_x86_feature_detected!("fma");
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_sse_row_f16::<3, false, false>;
                if is_f16c_available {
                    if is_fma_available {
                        _dispatcher = convolve_vertical_sse_row_f16::<3, true, true>;
                    } else {
                        _dispatcher = convolve_vertical_sse_row_f16::<3, true, false>;
                    }
                }
            }

            if is_x86_feature_detected!("avx2") && is_f16c_available {
                _dispatcher = convolve_vertical_avx_row_f16::<3, false>;
                if is_fma_available {
                    _dispatcher = convolve_vertical_avx_row_f16::<3, true>;
                }
            }
        }
        #[cfg(all(
            any(target_arch = "riscv64", target_arch = "riscv32"),
            feature = "riscv"
        ))]
        {
            let features: [String; 2] = ["zvfh".parse().unwrap(), "zfh".parse().unwrap()];
            if std::arch::is_riscv_feature_detected!("v") && risc_is_features_supported(&features) {
                _dispatcher = convolve_vertical_risc_row_f16::<3>;
            }
        }
        convolve_vertical_dispatch_f16(self, filter_weights, destination, pool, _dispatcher);
    }
}

impl<'a> HorizontalConvolutionPass<f16, 1> for ImageStore<'a, f16, 1> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f16, 1>,
        pool: &Option<ThreadPool>,
    ) {
        let _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, *const f16, usize, *mut f16, usize),
        > = Some(convolve_horizontal_rgba_4_row_f32::<f16, f32, 1>);
        let _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f16, *mut f16) =
            convolve_horizontal_rgb_native_row::<f16, f32, 1>;
        convolve_horizontal_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher_4_rows,
            _dispatcher_row,
        );
    }
}

impl<'a> VerticalConvolutionPass<f16, 1> for ImageStore<'a, f16, 1> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f16, 1>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(usize, &FilterBounds, *const f16, *mut f16, usize, &[f32]) =
            convolve_vertical_rgb_native_row_f32::<f16, 1>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if is_aarch_f16c_supported() {
                _dispatcher = convolve_vertical_rgb_neon_row_f16::<1>;
                if is_aarch_f16_supported() {
                    _dispatcher = xconvolve_vertical_rgb_neon_row_f16::<1>;
                }
            }
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let is_f16c_available = is_x86_feature_detected!("f16c");
            let is_fma_available = is_x86_feature_detected!("fma");
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_sse_row_f16::<1, false, false>;
                if is_f16c_available {
                    if is_fma_available {
                        _dispatcher = convolve_vertical_sse_row_f16::<1, true, true>;
                    } else {
                        _dispatcher = convolve_vertical_sse_row_f16::<1, true, false>;
                    }
                }
            }
            if is_x86_feature_detected!("avx2") && is_f16c_available {
                _dispatcher = convolve_vertical_avx_row_f16::<1, false>;
                if is_fma_available {
                    _dispatcher = convolve_vertical_avx_row_f16::<1, true>;
                }
            }
        }
        #[cfg(all(
            any(target_arch = "riscv64", target_arch = "riscv32"),
            feature = "riscv"
        ))]
        {
            let features: [String; 2] = ["zvfh".parse().unwrap(), "zfh".parse().unwrap()];
            if std::arch::is_riscv_feature_detected!("v") && risc_is_features_supported(&features) {
                _dispatcher = convolve_vertical_risc_row_f16::<1>;
            }
        }
        convolve_vertical_dispatch_f16(self, filter_weights, destination, pool, _dispatcher);
    }
}
