/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_u8::*;
use crate::dispatch_group_u8::{convolve_horizontal_dispatch_u8, convolve_vertical_dispatch_u8};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_store::ImageStore;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::sse_rgb::{
    convolve_horizontal_rgb_sse_row_one, convolve_horizontal_rgb_sse_rows_4,
    convolve_vertical_rgb_sse_row,
};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::*;
use rayon::ThreadPool;

#[inline(always)]
pub(crate) fn convolve_vertical_rgb_native_row_u8<const COMPONENTS: usize>(
    dst_width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
    src_stride: usize,
    weight_ptr: *const i16,
) {
    let mut cx = 0usize;
    while cx + 12 < dst_width {
        unsafe {
            convolve_vertical_part::<12, COMPONENTS>(
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
            convolve_vertical_part::<8, COMPONENTS>(
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
            convolve_vertical_part::<1, COMPONENTS>(
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

impl<'a> HorizontalConvolutionPass<u8, 3> for ImageStore<'a, u8, 3> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 3>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<i16>, *const u8, usize, *mut u8, usize),
        > = None;
        let mut _dispatcher_1_row: fn(usize, usize, &FilterWeights<i16>, *const u8, *mut u8) =
            convolve_horizontal_rgba_native_row::<3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4);
            _dispatcher_1_row = convolve_horizontal_rgb_neon_row_one;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgba_native_4_row::<3>);
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_sse_rows_4);
                _dispatcher_1_row = convolve_horizontal_rgb_sse_row_one;
            }
        }
        convolve_horizontal_dispatch_u8(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher_4_rows,
            _dispatcher_1_row,
        );
    }
}

impl<'a> VerticalConvolutionPass<u8, 3> for ImageStore<'a, u8, 3> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 3>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(
            dst_width: usize,
            bounds: &FilterBounds,
            unsafe_source_ptr_0: *const u8,
            unsafe_destination_ptr_0: *mut u8,
            src_stride: usize,
            weight_ptr: *const i16,
        ) = convolve_vertical_rgb_native_row_u8::<3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row::<3>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_rgb_sse_row;
            }
        }
        convolve_vertical_dispatch_u8(self, filter_weights, destination, pool, _dispatcher);
    }
}
