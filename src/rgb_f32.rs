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
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2"
))]
use crate::avx2::convolve_vertical_avx_row_f32;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_f32::*;
use crate::dispatch_group_f32::{convolve_horizontal_dispatch_f32, convolve_vertical_dispatch_f32};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_store::ImageStore;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::*;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::*;
use num_traits::AsPrimitive;
use rayon::ThreadPool;

pub(crate) fn convolve_vertical_rgb_native_row_f32<
    T: Copy + 'static + AsPrimitive<f32>,
    const COMPONENTS: usize,
>(
    dst_width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const T,
    unsafe_destination_ptr_0: *mut T,
    src_stride: usize,
    weight_ptr: *const f32,
) where
    f32: AsPrimitive<T>,
{
    let mut cx = 0usize;

    let total_width = dst_width * COMPONENTS;

    while cx + 64 < total_width {
        unsafe {
            convolve_vertical_part_f32::<T, 64>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 64;
    }

    while cx + 32 < total_width {
        unsafe {
            convolve_vertical_part_f32::<T, 32>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 32;
    }

    while cx + 24 < total_width {
        unsafe {
            convolve_vertical_part_f32::<T, 24>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 24;
    }

    while cx + 12 < total_width {
        unsafe {
            convolve_vertical_part_f32::<T, 12>(
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

    while cx + 8 < total_width {
        unsafe {
            convolve_vertical_part_f32::<T, 8>(
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

    while cx < total_width {
        unsafe {
            convolve_vertical_part_f32::<T, 1>(
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
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 3>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, *const f32, usize, *mut f32, usize),
        > = Some(convolve_horizontal_rgba_4_row_f32::<f32, 3>);
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, *const f32, *mut f32) =
            convolve_horizontal_rgb_native_row::<f32, 3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4_f32);
            _dispatcher_row = convolve_horizontal_rgb_neon_row_one_f32;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_sse_rows_4_f32);
                _dispatcher_row = convolve_horizontal_rgb_sse_row_one_f32;
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

impl<'a> VerticalConvolutionPass<f32, 3> for ImageStore<'a, f32, 3> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 3>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(usize, &FilterBounds, *const f32, *mut f32, usize, *const f32) =
            convolve_vertical_rgb_native_row_f32::<f32, 3>;
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
                _dispatcher = convolve_vertical_rgb_sse_row_f32::<3>;
            }
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "avx2"
        ))]
        {
            if is_x86_feature_detected!("avx2") {
                _dispatcher = convolve_vertical_avx_row_f32::<3>;
            }
        }
        convolve_vertical_dispatch_f32(self, filter_weights, destination, pool, _dispatcher);
    }
}
