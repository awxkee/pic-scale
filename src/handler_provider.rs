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
use crate::fixed_point_horizontal::{
    convolve_row_handler_fixed_point, convolve_row_handler_fixed_point_4,
};
use crate::fixed_point_vertical::column_handler_fixed_point;
use crate::floating_point_horizontal::{
    convolve_row_handler_floating_point, convolve_row_handler_floating_point_4,
};
use crate::floating_point_vertical::column_handler_floating_point;
use crate::mixed_storage::MixedStorage;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    convolve_column_lb_u16, convolve_column_u16, convolve_horizontal_rgba_neon_rows_4_lb_u16,
    convolve_horizontal_rgba_neon_u16_lb_row,
};
use crate::saturate_narrow::SaturateNarrow;
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::{
    convolve_column_lb_sse_u16, convolve_column_sse_u16, convolve_horizontal_rgba_sse_rows_4_lb_u8,
    convolve_horizontal_rgba_sse_rows_4_u16, convolve_horizontal_rgba_sse_u16_lb_row,
    convolve_horizontal_rgba_sse_u16_row,
};
use num_traits::{AsPrimitive, Float, MulAdd};
use std::ops::{Add, AddAssign, Mul};

pub(crate) trait ColumnHandlerFloatingPoint<T, J, F>
where
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy + 'static + AsPrimitive<T> + MulAdd<J, Output = J> + Default + MixedStorage<T>,
    F: Copy + 'static + AsPrimitive<J>,
    i32: AsPrimitive<J>,
    f32: AsPrimitive<J>,
{
    fn handle_floating_column(
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
            fn handle_floating_column(
                _: usize,
                bounds: &FilterBounds,
                src: &[$column_type],
                dst: &mut [$column_type],
                src_stride: usize,
                weight: &[F],
                bit_depth: u32,
            ) {
                column_handler_floating_point::<$column_type, f32, F>(
                    bounds, src, dst, src_stride, weight, bit_depth,
                )
            }
        }
    };
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    any(target_arch = "x86_64", target_arch = "x86")
)))]
impl ColumnHandlerFloatingPoint<u16, f32, f32> for u16 {
    fn handle_floating_column(
        _: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[f32],
        bit_depth: u32,
    ) {
        column_handler_floating_point::<u16, f32, f32>(
            bounds, src, dst, src_stride, weight, bit_depth,
        )
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
impl ColumnHandlerFloatingPoint<u16, f32, f32> for u16 {
    fn handle_floating_column(
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[f32],
        bit_depth: u32,
    ) {
        convolve_column_u16(dst_width, bounds, src, dst, src_stride, weight, bit_depth)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
impl ColumnHandlerFloatingPoint<u16, f32, f32> for u16 {
    fn handle_floating_column(
        _dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[f32],
        bit_depth: u32,
    ) {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::convolve_column_avx_u16;
                return convolve_column_avx_u16(
                    _dst_width, bounds, src, dst, src_stride, weight, bit_depth,
                );
            }
        }
        #[cfg(feature = "sse")]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return convolve_column_sse_u16(
                _dst_width, bounds, src, dst, src_stride, weight, bit_depth,
            );
        }
        column_handler_floating_point::<u16, f32, f32>(
            bounds, src, dst, src_stride, weight, bit_depth,
        );
    }
}

default_floating_column_handler!(u8);
default_floating_column_handler!(f32);

pub(crate) trait RowHandlerFloatingPoint<T, J, F>
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
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
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

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn handle_row<const COMPONENTS: usize>(
        src: &[u16],
        dst: &mut [u16],
        filter_weights: &FilterWeights<f32>,
        bit_depth: u32,
    ) {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if COMPONENTS == 4 && std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::convolve_horizontal_rgba_avx_u16_row_f;
                return convolve_horizontal_rgba_avx_u16_row_f(src, dst, filter_weights, bit_depth);
            }
        }
        #[cfg(feature = "sse")]
        if COMPONENTS == 4 && std::arch::is_x86_feature_detected!("sse4.1") {
            return convolve_horizontal_rgba_sse_u16_row(src, dst, filter_weights, bit_depth);
        }
        convolve_row_handler_floating_point::<u16, f32, f32, COMPONENTS>(
            src,
            dst,
            filter_weights,
            bit_depth,
        )
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
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

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn handle_row_4<const COMPONENTS: usize>(
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        filter_weights: &FilterWeights<f32>,
        bit_depth: u32,
    ) {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if COMPONENTS == 4 && std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::convolve_horizontal_rgba_avx_rows_4_u16_f;
                return convolve_horizontal_rgba_avx_rows_4_u16_f(
                    src,
                    src_stride,
                    dst,
                    dst_stride,
                    filter_weights,
                    bit_depth,
                );
            }
        }
        #[cfg(feature = "sse")]
        if COMPONENTS == 4 && std::arch::is_x86_feature_detected!("sse4.1") {
            return convolve_horizontal_rgba_sse_rows_4_u16(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        }
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

pub(crate) trait ColumnHandlerFixedPoint<T> {
    fn handle_fixed_column<J, const COMPONENTS: usize>(
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[T],
        dst: &mut [T],
        src_stride: usize,
        weight: &[i16],
        bit_depth: u32,
    ) where
        T: Copy + 'static + AsPrimitive<J> + Default,
        J: Copy
            + 'static
            + AsPrimitive<T>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<T>
            + Default,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>;
}

pub(crate) trait RowHandlerFixedPoint<T> {
    fn handle_fixed_row_4<J, const COMPONENTS: usize>(
        src: &[T],
        src_stride: usize,
        dst: &mut [T],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) where
        T: Copy + 'static + AsPrimitive<J> + Default,
        J: Copy
            + 'static
            + AsPrimitive<T>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<T>
            + Default
            + Add<J, Output = J>,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>;

    fn handle_fixed_row<J, const COMPONENTS: usize>(
        src: &[T],
        dst: &mut [T],
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) where
        T: Copy + 'static + AsPrimitive<J> + Default,
        J: Copy
            + 'static
            + AsPrimitive<T>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<T>
            + Default
            + Add<J, Output = J>,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>;
}

impl RowHandlerFixedPoint<u16> for u16 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn handle_fixed_row_4<J, const COMPONENTS: usize>(
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) where
        J: Copy
            + 'static
            + AsPrimitive<u16>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<u16>
            + Default
            + Add<J, Output = J>,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>,
        u16: AsPrimitive<J>,
    {
        convolve_row_handler_fixed_point_4::<u16, J, COMPONENTS>(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        )
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn handle_fixed_row_4<J, const COMPONENTS: usize>(
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) where
        J: Copy
            + 'static
            + AsPrimitive<u16>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<u16>
            + Default
            + Add<J, Output = J>,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>,
        u16: AsPrimitive<J>,
    {
        if COMPONENTS == 4 {
            convolve_horizontal_rgba_neon_rows_4_lb_u16(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            )
        } else if COMPONENTS == 3 {
            use crate::neon::convolve_horizontal_rgb_neon_rows_4_lb_u16;
            return convolve_horizontal_rgb_neon_rows_4_lb_u16(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        } else if COMPONENTS == 1 {
            use crate::neon::convolve_horizontal_plane_neon_rows_4_lb_u16;
            return convolve_horizontal_plane_neon_rows_4_lb_u16(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        } else {
            convolve_row_handler_fixed_point_4::<u16, J, COMPONENTS>(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            )
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn handle_fixed_row_4<J, const COMPONENTS: usize>(
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) where
        J: Copy
            + 'static
            + AsPrimitive<u16>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<u16>
            + Default
            + Add<J, Output = J>,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>,
        u16: AsPrimitive<J>,
    {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if COMPONENTS == 4 && std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::convolve_horizontal_rgba_avx_rows_4_u16;
            return convolve_horizontal_rgba_avx_rows_4_u16(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if COMPONENTS == 1 && std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::convolve_horizontal_plane_avx_rows_4_u16;
            return convolve_horizontal_plane_avx_rows_4_u16(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        }
        #[cfg(feature = "sse")]
        if COMPONENTS == 4 && std::arch::is_x86_feature_detected!("sse4.1") {
            return convolve_horizontal_rgba_sse_rows_4_lb_u8(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        }
        convolve_row_handler_fixed_point_4::<u16, J, COMPONENTS>(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        )
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn handle_fixed_row<J, const COMPONENTS: usize>(
        src: &[u16],
        dst: &mut [u16],
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) where
        J: Copy
            + 'static
            + AsPrimitive<u16>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<u16>
            + Default
            + Add<J, Output = J>,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>,
        u16: AsPrimitive<J>,
    {
        convolve_row_handler_fixed_point::<u16, J, COMPONENTS>(src, dst, filter_weights, bit_depth)
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn handle_fixed_row<J, const COMPONENTS: usize>(
        src: &[u16],
        dst: &mut [u16],
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) where
        J: Copy
            + 'static
            + AsPrimitive<u16>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<u16>
            + Default
            + Add<J, Output = J>,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>,
        u16: AsPrimitive<J>,
    {
        if COMPONENTS == 4 {
            convolve_horizontal_rgba_neon_u16_lb_row(src, dst, filter_weights, bit_depth)
        } else if COMPONENTS == 3 {
            use crate::neon::convolve_horizontal_rgb_neon_u16_lb_row;
            convolve_horizontal_rgb_neon_u16_lb_row(src, dst, filter_weights, bit_depth)
        } else if COMPONENTS == 1 {
            use crate::neon::convolve_horizontal_plane_neon_u16_lb_row;
            convolve_horizontal_plane_neon_u16_lb_row(src, dst, filter_weights, bit_depth)
        } else {
            convolve_row_handler_fixed_point::<u16, J, COMPONENTS>(
                src,
                dst,
                filter_weights,
                bit_depth,
            )
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn handle_fixed_row<J, const COMPONENTS: usize>(
        src: &[u16],
        dst: &mut [u16],
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) where
        J: Copy
            + 'static
            + AsPrimitive<u16>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<u16>
            + Default
            + Add<J, Output = J>,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>,
        u16: AsPrimitive<J>,
    {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if COMPONENTS == 4 && std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::convolve_horizontal_rgba_avx_u16lp_row;
            return convolve_horizontal_rgba_avx_u16lp_row(src, dst, filter_weights, bit_depth);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if COMPONENTS == 1 && std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::convolve_horizontal_plane_avx_u16lp_row;
            return convolve_horizontal_plane_avx_u16lp_row(src, dst, filter_weights, bit_depth);
        }
        #[cfg(feature = "sse")]
        if COMPONENTS == 4 && std::arch::is_x86_feature_detected!("sse4.1") {
            return convolve_horizontal_rgba_sse_u16_lb_row(src, dst, filter_weights, bit_depth);
        }
        convolve_row_handler_fixed_point::<u16, J, COMPONENTS>(src, dst, filter_weights, bit_depth)
    }
}

impl ColumnHandlerFixedPoint<u16> for u16 {
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    fn handle_fixed_column<J, const COMPONENTS: usize>(
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[i16],
        bit_depth: u32,
    ) where
        u16: Copy + 'static + AsPrimitive<J> + Default,
        J: Copy
            + 'static
            + AsPrimitive<u16>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<u16>
            + Default,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>,
    {
        column_handler_fixed_point::<u16, J>(
            dst_width, bounds, src, dst, src_stride, weight, bit_depth,
        );
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn handle_fixed_column<J, const COMPONENTS: usize>(
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[i16],
        bit_depth: u32,
    ) where
        u16: Copy + 'static + AsPrimitive<J> + Default,
        J: Copy
            + 'static
            + AsPrimitive<u16>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<u16>
            + Default,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>,
    {
        convolve_column_lb_u16(dst_width, bounds, src, dst, src_stride, weight, bit_depth);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn handle_fixed_column<J, const COMPONENTS: usize>(
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[i16],
        bit_depth: u32,
    ) where
        u16: Copy + 'static + AsPrimitive<J> + Default,
        J: Copy
            + 'static
            + AsPrimitive<u16>
            + Mul<Output = J>
            + AddAssign
            + SaturateNarrow<u16>
            + Default,
        i32: AsPrimitive<J>,
        i16: AsPrimitive<J>,
    {
        #[cfg(all(target_arch = "x86_64", feature = "nightly_avx512"))]
        if std::arch::is_x86_feature_detected!("avx512bw") {
            use crate::avx512::convolve_column_lb_avx512_u16;
            return convolve_column_lb_avx512_u16(
                dst_width, bounds, src, dst, src_stride, weight, bit_depth,
            );
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::avx2::convolve_column_lb_avx2_u16;
            if std::arch::is_x86_feature_detected!("avx2") {
                return convolve_column_lb_avx2_u16(
                    dst_width, bounds, src, dst, src_stride, weight, bit_depth,
                );
            }
        }
        #[cfg(feature = "sse")]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return convolve_column_lb_sse_u16(
                dst_width, bounds, src, dst, src_stride, weight, bit_depth,
            );
        }
        column_handler_fixed_point::<u16, J>(
            dst_width, bounds, src, dst, src_stride, weight, bit_depth,
        )
    }
}

pub(crate) fn handle_fixed_column_u8(
    dst_width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    column_handler_fixed_point::<u8, i32>(dst_width, bounds, src, dst, src_stride, weight, 8);
}

pub(crate) fn handle_fixed_row_u8<const CHANNELS: usize>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    convolve_row_handler_fixed_point::<u8, i32, CHANNELS>(src, dst, filter_weights, 8);
}

pub(crate) fn handle_fixed_rows_4_u8<const CHANNELS: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    convolve_row_handler_fixed_point_4::<u8, i32, CHANNELS>(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
        8,
    );
}
