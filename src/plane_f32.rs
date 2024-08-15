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
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx2::convolve_vertical_avx_row_f32;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_f32::{
    convolve_horizontal_rgb_native_row, convolve_horizontal_rgba_4_row_f32,
};
use crate::dispatch_group_f32::{convolve_horizontal_dispatch_f32, convolve_vertical_dispatch_f32};
use crate::filter_weights::{FilterBounds, FilterWeights};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    convolve_horizontal_plane_neon_row_one, convolve_horizontal_plane_neon_rows_4,
    convolve_vertical_rgb_neon_row_f32,
};
use crate::rgb_f32::convolve_vertical_rgb_native_row_f32;
#[cfg(all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    feature = "riscv"
))]
use crate::risc::convolve_vertical_rgb_risc_row_f32;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{
    convolve_horizontal_plane_sse_row_one, convolve_horizontal_plane_sse_rows_4,
    convolve_vertical_rgb_sse_row_f32,
};
use crate::ImageStore;
use rayon::ThreadPool;
#[cfg(all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    feature = "riscv"
))]
use std::arch::is_riscv_feature_detected;

impl<'a> HorizontalConvolutionPass<f32, 1> for ImageStore<'a, f32, 1> {
    #[inline(always)]
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 1>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, *const f32, usize, *mut f32, usize),
        > = Some(convolve_horizontal_rgba_4_row_f32::<f32, 1>);
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f32, *mut f32) =
            convolve_horizontal_rgb_native_row::<f32, 1>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_plane_neon_rows_4);
            _dispatcher_row = convolve_horizontal_plane_neon_row_one;
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_plane_sse_rows_4::<false>);
                _dispatcher_row = convolve_horizontal_plane_sse_row_one::<false>;
                if is_x86_feature_detected!("fma") {
                    _dispatcher_4_rows = Some(convolve_horizontal_plane_sse_rows_4::<true>);
                    _dispatcher_row = convolve_horizontal_plane_sse_row_one::<true>;
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

impl<'a> VerticalConvolutionPass<f32, 1> for ImageStore<'a, f32, 1> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 1>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(usize, &FilterBounds, *const f32, *mut f32, usize, *const f32) =
            convolve_vertical_rgb_native_row_f32::<f32, 1>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f32::<1>;
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let has_fma = is_x86_feature_detected!("fma");
            if is_x86_feature_detected!("sse4.1") {
                if has_fma {
                    _dispatcher = convolve_vertical_rgb_sse_row_f32::<1, true>;
                } else {
                    _dispatcher = convolve_vertical_rgb_sse_row_f32::<1, false>;
                }
            }
            if is_x86_feature_detected!("avx2") {
                _dispatcher = convolve_vertical_avx_row_f32::<1, false>;
                if has_fma {
                    _dispatcher = convolve_vertical_avx_row_f32::<1, true>;
                }
            }
        }
        #[cfg(all(
            any(target_arch = "riscv64", target_arch = "riscv32"),
            feature = "riscv"
        ))]
        {
            if is_riscv_feature_detected!("v") {
                _dispatcher = convolve_vertical_rgb_risc_row_f32::<1>;
            }
        }
        convolve_vertical_dispatch_f32(self, filter_weights, destination, pool, _dispatcher);
    }
}
