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

use half::f16;
use rayon::ThreadPool;
use crate::avx2::convolve_vertical_avx_row_f16;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_f32::{
    convolve_horizontal_rgb_native_row, convolve_horizontal_rgba_4_row_f32,
};
use crate::dispatch_group_f16::{convolve_horizontal_dispatch_f16, convolve_vertical_dispatch_f16};
use crate::filter_weights::{FilterBounds, FilterWeights};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    convolve_horizontal_rgb_neon_row_one_f16, convolve_horizontal_rgb_neon_rows_4_f16,
    convolve_horizontal_rgba_neon_row_one_f16, convolve_horizontal_rgba_neon_rows_4_f16,
    convolve_vertical_rgb_neon_row_f16,
};
use crate::rgb_f32::convolve_vertical_rgb_native_row_f32;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    all(target_feature = "sse4.1", target_feature = "f16c")
))]
use crate::sse::convolve_vertical_rgb_sse_row_f16;
use crate::ImageStore;

impl<'a> HorizontalConvolutionPass<f16, 4> for ImageStore<'a, f16, 4> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f16, 4>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, *const f16, usize, *mut f16, usize),
        > = Some(convolve_horizontal_rgba_4_row_f32::<f16, 4>);
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f16, *mut f16) =
            convolve_horizontal_rgb_native_row::<f16, 4>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgba_neon_rows_4_f16);
            _dispatcher_row = convolve_horizontal_rgba_neon_row_one_f16;
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
        let mut _dispatcher: fn(usize, &FilterBounds, *const f16, *mut f16, usize, *const f32) =
            convolve_vertical_rgb_native_row_f32::<f16, 4>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f16::<4>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            all(target_feature = "sse4.1", target_feature = "f16c")
        ))]
        {
            _dispatcher = convolve_vertical_rgb_sse_row_f16::<4>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            all(target_feature = "avx2", target_feature = "f16c")
        ))]
        {
            _dispatcher = convolve_vertical_avx_row_f16::<4>;
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
        > = Some(convolve_horizontal_rgba_4_row_f32::<f16, 3>);
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f16, *mut f16) =
            convolve_horizontal_rgb_native_row::<f16, 3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4_f16);
            _dispatcher_row = convolve_horizontal_rgb_neon_row_one_f16;
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
        let mut _dispatcher: fn(usize, &FilterBounds, *const f16, *mut f16, usize, *const f32) =
            convolve_vertical_rgb_native_row_f32::<f16, 3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f16::<3>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            all(target_feature = "sse4.1", target_feature = "f16c")
        ))]
        {
            _dispatcher = convolve_vertical_rgb_sse_row_f16::<3>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            all(target_feature = "avx2", target_feature = "f16c")
        ))]
        {
            _dispatcher = convolve_vertical_avx_row_f16::<3>;
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
        > = Some(convolve_horizontal_rgba_4_row_f32::<f16, 1>);
        let _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f16, *mut f16) =
            convolve_horizontal_rgb_native_row::<f16, 1>;
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
        let mut _dispatcher: fn(usize, &FilterBounds, *const f16, *mut f16, usize, *const f32) =
            convolve_vertical_rgb_native_row_f32::<f16, 1>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f16::<1>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            all(target_feature = "sse4.1", target_feature = "f16c")
        ))]
        {
            _dispatcher = convolve_vertical_rgb_sse_row_f16::<1>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            all(target_feature = "avx2", target_feature = "f16c")
        ))]
        {
            _dispatcher = convolve_vertical_avx_row_f16::<1>;
        }
        convolve_vertical_dispatch_f16(self, filter_weights, destination, pool, _dispatcher);
    }
}
