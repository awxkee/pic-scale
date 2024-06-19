/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use rayon::ThreadPool;

use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_f32::{
    convolve_horizontal_rgb_native_row, convolve_horizontal_rgba_4_row_f32,
};
use crate::dispatch_group_f32::{convolve_horizontal_dispatch_f32, convolve_vertical_dispatch_f32};
use crate::filter_weights::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::*;
use crate::rgb_f32::convolve_vertical_rgb_native_row_f32;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::*;
use crate::ImageStore;

impl<'a> HorizontalConvolutionPass<f32, 4> for ImageStore<'a, f32, 4> {
    #[inline(always)]
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 4>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, *const f32, usize, *mut f32, usize),
        > = Some(convolve_horizontal_rgba_4_row_f32::<4>);
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f32, *mut f32) =
            convolve_horizontal_rgb_native_row::<4>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgba_neon_rows_4);
            _dispatcher_row = convolve_horizontal_rgba_neon_row_one;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_sse_rows_4_f32);
                _dispatcher_row = convolve_horizontal_rgba_sse_row_one_f32;
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

impl<'a> VerticalConvolutionPass<f32, 4> for ImageStore<'a, f32, 4> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 4>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(usize, &FilterBounds, *const f32, *mut f32, usize, *const f32) =
            convolve_vertical_rgb_native_row_f32::<4>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f32::<4>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_rgb_sse_row_f32::<4>;
            }
        }
        convolve_vertical_dispatch_f32(self, filter_weights, destination, pool, _dispatcher);
    }
}
