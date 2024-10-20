/*
 * Copyright (c) Radzivon Bartoshyk, 10/2024. All rights reserved.
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
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::floating_point_horizontal::{
    convolve_row_handler_floating_point, convolve_row_handler_floating_point_4,
};
use crate::floating_point_vertical::column_handler_floating_point;
use crate::mixed_storage::MixedStorage;
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(feature = "disable_simd")
))]
use crate::neon::convolve_column_u16;
use num_traits::{AsPrimitive, Float, MulAdd};

pub trait ColumnHandlerFloatingPoint<T, J, F>
where
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy + 'static + AsPrimitive<T> + MulAdd<J, Output = J> + Default + MixedStorage<T>,
    F: Copy + 'static + AsPrimitive<J>,
    i32: AsPrimitive<J>,
    f32: AsPrimitive<J>,
{
    fn handle_floating_column<const COMPONENTS: usize>(
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[T],
        dst: &mut [T],
        src_stride: usize,
        weight: &[F],
        bit_depth: u32,
    );
}

macro_rules! default_floating_column_handler {
    ($column_type:ty) => {
        impl<F> ColumnHandlerFloatingPoint<$column_type, f32, F> for $column_type
        where
            F: Copy + 'static + Float + AsPrimitive<f32>,
            $column_type: AsPrimitive<f32>,
        {
            fn handle_floating_column<const COMPONENTS: usize>(
                dst_width: usize,
                bounds: &FilterBounds,
                src: &[$column_type],
                dst: &mut [$column_type],
                src_stride: usize,
                weight: &[F],
                bit_depth: u32,
            ) {
                column_handler_floating_point::<$column_type, f32, F, COMPONENTS>(
                    dst_width, bounds, src, dst, src_stride, weight, bit_depth,
                )
            }
        }
    };
}

#[cfg(any(
    feature = "disable_simd",
    not(all(target_arch = "aarch64", target_feature = "neon"))
))]
impl ColumnHandlerFloatingPoint<u16, f32, f32> for u16 {
    fn handle_floating_column<const COMPONENTS: usize>(
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[f32],
        bit_depth: u32,
    ) {
        column_handler_floating_point::<u16, f32, f32, COMPONENTS>(
            dst_width, bounds, src, dst, src_stride, weight, bit_depth,
        )
    }
}

#[cfg(all(
    not(feature = "disable_simd"),
    all(target_arch = "aarch64", target_feature = "neon")
))]
impl ColumnHandlerFloatingPoint<u16, f32, f32> for u16 {
    fn handle_floating_column<const COMPONENTS: usize>(
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[f32],
        bit_depth: u32,
    ) {
        convolve_column_u16::<COMPONENTS>(
            dst_width, bounds, src, dst, src_stride, weight, bit_depth,
        )
    }
}

default_floating_column_handler!(u8);
default_floating_column_handler!(f32);

pub trait RowHandlerFloatingPoint<T, J, F>
where
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy + 'static + AsPrimitive<T> + MulAdd<J, Output = J> + Default + MixedStorage<T>,
    F: Copy + 'static + AsPrimitive<J>,
    i32: AsPrimitive<J>,
    f32: AsPrimitive<J>,
{
    fn handle_row_4<const COMPONENTS: usize>(
        src: &[T],
        src_stride: usize,
        dst: &mut [T],
        dst_stride: usize,
        filter_weights: &FilterWeights<F>,
        bit_depth: u32,
    );

    fn handle_row<const COMPONENTS: usize>(
        src: &[T],
        dst: &mut [T],
        filter_weights: &FilterWeights<F>,
        bit_depth: u32,
    );
}

impl RowHandlerFloatingPoint<u16, f32, f32> for u16 {
    fn handle_row<const COMPONENTS: usize>(
        src: &[u16],
        dst: &mut [u16],
        filter_weights: &FilterWeights<f32>,
        bit_depth: u32,
    ) {
        convolve_row_handler_floating_point::<u16, f32, f32, COMPONENTS>(
            src,
            dst,
            filter_weights,
            bit_depth,
        )
    }

    fn handle_row_4<const COMPONENTS: usize>(
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        filter_weights: &FilterWeights<f32>,
        bit_depth: u32,
    ) {
        convolve_row_handler_floating_point_4::<u16, f32, f32, COMPONENTS>(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        )
    }
}
