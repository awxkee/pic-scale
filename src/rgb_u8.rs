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
use crate::avx2::convolve_vertical_avx_row;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_u8::*;
use crate::dispatch_group_u8::{convolve_horizontal_dispatch_u8, convolve_vertical_dispatch_u8};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_store::ImageStore;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::*;
use crate::saturate_narrow::SaturateNarrow;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::convolve_vertical_sse_row;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::{convolve_horizontal_rgb_sse_row_one, convolve_horizontal_rgb_sse_rows_4};
use num_traits::AsPrimitive;
use rayon::ThreadPool;
use std::ops::{AddAssign, Mul};

/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn convolve_vertical_rgb_native_row_u8<
    T: Copy + 'static + AsPrimitive<J>,
    J: Copy + 'static + AsPrimitive<T> + Mul<Output = J> + AddAssign + SaturateNarrow<T>,
    const COMPONENTS: usize,
>(
    dst_width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const T,
    unsafe_destination_ptr_0: *mut T,
    src_stride: usize,
    weight_ptr: *const i16,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
{
    let mut cx = 0usize;

    let total_width = COMPONENTS * dst_width;

    while cx + 64 < total_width {
        unsafe {
            convolve_vertical_part::<T, J, 64>(
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

    while cx + 48 < total_width {
        unsafe {
            convolve_vertical_part::<T, J, 48>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 48;
    }

    while cx + 36 < total_width {
        unsafe {
            convolve_vertical_part::<T, J, 36>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 36;
    }

    while cx + 24 < total_width {
        unsafe {
            convolve_vertical_part::<T, J, 24>(
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
            convolve_vertical_part::<T, J, 12>(
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
            convolve_vertical_part::<T, J, 8>(
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
            convolve_vertical_part::<T, J, 1>(
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
        > = Some(convolve_horizontal_rgba_native_4_row::<u8, i32, 3>);
        let mut _dispatcher_1_row: fn(usize, usize, &FilterWeights<i16>, *const u8, *mut u8) =
            convolve_horizontal_rgba_native_row::<u8, i32, 3>;
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
        ) = convolve_vertical_rgb_native_row_u8::<u8, i32, 3>;
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
                _dispatcher = convolve_vertical_sse_row::<3>;
            }
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "avx2"
        ))]
        {
            if is_x86_feature_detected!("avx2") {
                _dispatcher = convolve_vertical_avx_row::<3>;
            }
        }
        convolve_vertical_dispatch_u8(self, filter_weights, destination, pool, _dispatcher);
    }
}
