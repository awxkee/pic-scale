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
use crate::avx2::convolve_vertical_avx_row;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_u8::convolve_horizontal_rgba_native_row;
use crate::dispatch_group_u8::{convolve_horizontal_dispatch_u8, convolve_vertical_dispatch_u8};
use crate::filter_weights::{FilterBounds, FilterWeights};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::convolve_vertical_neon_row;
#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::neon::{convolve_horizontal_plane_neon_row, convolve_horizontal_plane_neon_rows_4_u8};
use crate::rgb_u8::convolve_vertical_rgb_native_row_u8;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{
    convolve_horizontal_plane_sse_row, convolve_horizontal_plane_sse_rows_4_u8,
    convolve_vertical_sse_row,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::wasm_vertical_neon_row;
use crate::ImageStore;
use rayon::ThreadPool;

impl HorizontalConvolutionPass<u8, 1> for ImageStore<'_, u8, 1> {
    #[allow(clippy::type_complexity)]
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 1>,
        _pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<i16>, *const u8, usize, *mut u8, usize),
        > = None;
        let mut _dispatcher_1_row: fn(usize, usize, &FilterWeights<i16>, *const u8, *mut u8) =
            convolve_horizontal_rgba_native_row::<u8, i32, 1>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_plane_neon_rows_4_u8);
            _dispatcher_1_row = convolve_horizontal_plane_neon_row;
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_plane_sse_rows_4_u8);
                _dispatcher_1_row = convolve_horizontal_plane_sse_row;
            }
        }
        convolve_horizontal_dispatch_u8(
            self,
            filter_weights,
            destination,
            _pool,
            _dispatcher_4_rows,
            _dispatcher_1_row,
        );
    }
}

impl VerticalConvolutionPass<u8, 1> for ImageStore<'_, u8, 1> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 1>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(
            dst_width: usize,
            bounds: &FilterBounds,
            unsafe_source_ptr_0: *const u8,
            unsafe_destination_ptr_0: *mut u8,
            src_stride: usize,
            weight_ptr: &[i16],
        ) = convolve_vertical_rgb_native_row_u8::<u8, i32, 1>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_neon_row::<1>;
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_sse_row::<1>;
            }
            if is_x86_feature_detected!("avx2") {
                _dispatcher = convolve_vertical_avx_row::<1>;
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            _dispatcher = wasm_vertical_neon_row::<1>;
        }
        convolve_vertical_dispatch_u8(self, filter_weights, destination, pool, _dispatcher);
    }
}
