/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use rayon::ThreadPool;

use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_f32::*;
use crate::dispatch_group_f32::{convolve_horizontal_dispatch_f32, convolve_vertical_dispatch_f32};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_store::ImageStore;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_convolve_f32::convolve_vertical_rgb_sse_row_f32;

#[inline(always)]
pub(crate) fn convolve_vertical_rgb_native_row_f32<const COMPONENTS: usize>(
    dst_width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    src_stride: usize,
    weight_ptr: *const f32,
) {
    let mut cx = 0usize;
    while cx + 12 < dst_width {
        unsafe {
            convolve_vertical_part_f32::<12, COMPONENTS>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 12;
    }

    while cx + 8 < dst_width {
        unsafe {
            convolve_vertical_part_f32::<8, COMPONENTS>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 8;
    }

    while cx < dst_width {
        unsafe {
            convolve_vertical_part_f32::<1, COMPONENTS>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 1;
    }
}

impl<'a> HorizontalConvolutionPass<f32, 3> for ImageStore<'a, f32, 3> {
    #[inline(always)]
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 3>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, *const f32, usize, *mut f32, usize),
        > = Some(convolve_horizontal_rgba_4_row_f32::<3>);
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f32, *mut f32) =
            convolve_horizontal_rgb_native_row::<3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4_f32);
            _dispatcher_row = convolve_horizontal_rgb_neon_row_one_f32;
        }
        // #[cfg(all(
        //     any(target_arch = "x86_64", target_arch = "x86"),
        //     target_feature = "sse4.1"
        // ))]
        // {
        //     if is_x86_feature_detected!("sse4.1") {
        //         _dispatcher_4_rows = Some(convolve_horizontal_rgba_sse_rows_4_f32);
        //         _dispatcher_row = convolve_horizontal_rgb_sse_row_f32;
        //     }
        // }
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

impl<'a> VerticalConvolutionPass<f32, 3> for ImageStore<'a, f32, 3> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 3>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(usize, &FilterBounds, *const f32, *mut f32, usize, *const f32) =
            convolve_vertical_rgb_native_row_f32::<3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f32::<3>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_rgb_sse_row_f32;
            }
        }
        convolve_vertical_dispatch_f32(self, filter_weights, destination, pool, _dispatcher);
    }
}
