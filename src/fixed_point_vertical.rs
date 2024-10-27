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
use crate::filter_weights::FilterBounds;
use crate::saturate_narrow::SaturateNarrow;
use num_traits::AsPrimitive;
use std::ops::{AddAssign, Mul};

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn convolve_column_handler_fixed_point_direct_buffer<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy + 'static + AsPrimitive<T> + Mul<Output = J> + AddAssign + SaturateNarrow<T> + Default,
    const BUFFER_SIZE: usize,
>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    filter: &[i16],
    bounds: &FilterBounds,
    bit_depth: u32,
    x: usize,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
{
    if filter.is_empty() {
        return;
    }
    let mut direct_store: [J; BUFFER_SIZE] = [J::default(); BUFFER_SIZE];

    let v_start_px = x;

    let py = bounds.start;
    let weight = filter[0].as_();
    let offset = src_stride * py + v_start_px;
    let src_ptr = &src[offset..(offset + BUFFER_SIZE)];

    for (dst, src) in direct_store.iter_mut().zip(src_ptr) {
        *dst += src.as_() * weight;
    }

    for (j, &k_weight) in filter.iter().take(bounds.size).skip(1).enumerate() {
        // Adding 1 is necessary because skip do not incrementing value on values that skipped
        let py = bounds.start + j + 1;
        let weight = k_weight.as_();
        let offset = src_stride * py + v_start_px;
        let src_ptr = &src[offset..(offset + BUFFER_SIZE)];

        for (dst, src) in direct_store.iter_mut().zip(src_ptr.iter()) {
            *dst += src.as_() * weight;
        }
    }

    let v_dst = &mut dst[v_start_px..(v_start_px + BUFFER_SIZE)];
    for (dst, src) in v_dst.iter_mut().zip(direct_store) {
        *dst = src.saturate_narrow(bit_depth);
    }
}

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn convolve_column_handler_fixed_point_direct_buffer_double<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy + 'static + AsPrimitive<T> + Mul<Output = J> + AddAssign + SaturateNarrow<T> + Default,
    const BUFFER_SIZE: usize,
>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    filter: &[i16],
    bounds: &FilterBounds,
    bit_depth: u32,
    x: usize,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
{
    if filter.is_empty() {
        return;
    }
    let mut direct_store0: [J; BUFFER_SIZE] = [J::default(); BUFFER_SIZE];
    let mut direct_store1: [J; BUFFER_SIZE] = [J::default(); BUFFER_SIZE];

    let v_start_px = x;

    let py = bounds.start;
    let weight = filter[0].as_();
    let offset = src_stride * py + v_start_px;
    let src_ptr0 = &src[offset..(offset + BUFFER_SIZE)];
    let src_ptr1 = &src[(offset + BUFFER_SIZE)..(offset + BUFFER_SIZE * 2)];

    for (dst, src) in direct_store0.iter_mut().zip(src_ptr0) {
        *dst += src.as_() * weight;
    }

    for (dst, src) in direct_store1.iter_mut().zip(src_ptr1) {
        *dst += src.as_() * weight;
    }

    for (j, &k_weight) in filter.iter().take(bounds.size).skip(1).enumerate() {
        // Adding 1 is necessary because skip do not incrementing value on values that skipped
        let py = bounds.start + j + 1;
        let weight = k_weight.as_();
        let offset = src_stride * py + v_start_px;
        let src_ptr0 = &src[offset..(offset + BUFFER_SIZE)];
        let src_ptr1 = &src[(offset + BUFFER_SIZE)..(offset + BUFFER_SIZE * 2)];

        for (dst, src) in direct_store0.iter_mut().zip(src_ptr0.iter()) {
            *dst += src.as_() * weight;
        }
        for (dst, src) in direct_store1.iter_mut().zip(src_ptr1.iter()) {
            *dst += src.as_() * weight;
        }
    }

    let v_dst0 = &mut dst[v_start_px..(v_start_px + BUFFER_SIZE)];
    for (dst, src) in v_dst0.iter_mut().zip(direct_store0) {
        *dst = src.saturate_narrow(bit_depth);
    }

    let v_dst1 = &mut dst[(v_start_px + BUFFER_SIZE)..(v_start_px + BUFFER_SIZE * 2)];
    for (dst, src) in v_dst1.iter_mut().zip(direct_store1) {
        *dst = src.saturate_narrow(bit_depth);
    }
}

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn convolve_column_handler_fixed_point_direct_buffer_four<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy + 'static + AsPrimitive<T> + Mul<Output = J> + AddAssign + SaturateNarrow<T> + Default,
    const BUFFER_SIZE: usize,
>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    filter: &[i16],
    bounds: &FilterBounds,
    bit_depth: u32,
    x: usize,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
{
    if filter.is_empty() {
        return;
    }
    let mut direct_store0: [J; BUFFER_SIZE] = [J::default(); BUFFER_SIZE];
    let mut direct_store1: [J; BUFFER_SIZE] = [J::default(); BUFFER_SIZE];
    let mut direct_store2: [J; BUFFER_SIZE] = [J::default(); BUFFER_SIZE];
    let mut direct_store3: [J; BUFFER_SIZE] = [J::default(); BUFFER_SIZE];

    let v_start_px = x;

    let py = bounds.start;
    let weight = filter[0].as_();
    let offset = src_stride * py + v_start_px;
    let src_ptr0 = &src[offset..(offset + BUFFER_SIZE)];
    let src_ptr1 = &src[(offset + BUFFER_SIZE)..(offset + BUFFER_SIZE * 2)];
    let src_ptr2 = &src[(offset + BUFFER_SIZE * 2)..(offset + BUFFER_SIZE * 3)];
    let src_ptr3 = &src[(offset + BUFFER_SIZE * 3)..(offset + BUFFER_SIZE * 4)];

    for (dst, src) in direct_store0.iter_mut().zip(src_ptr0) {
        *dst += src.as_() * weight;
    }

    for (dst, src) in direct_store1.iter_mut().zip(src_ptr1) {
        *dst += src.as_() * weight;
    }

    for (dst, src) in direct_store2.iter_mut().zip(src_ptr2) {
        *dst += src.as_() * weight;
    }

    for (dst, src) in direct_store3.iter_mut().zip(src_ptr3) {
        *dst += src.as_() * weight;
    }

    for (j, &k_weight) in filter.iter().take(bounds.size).skip(1).enumerate() {
        // Adding 1 is necessary because skip do not incrementing value on values that skipped
        let py = bounds.start + j + 1;
        let weight = k_weight.as_();
        let offset = src_stride * py + v_start_px;
        let src_ptr0 = &src[offset..(offset + BUFFER_SIZE)];
        let src_ptr1 = &src[(offset + BUFFER_SIZE)..(offset + BUFFER_SIZE * 2)];
        let src_ptr2 = &src[(offset + BUFFER_SIZE * 2)..(offset + BUFFER_SIZE * 3)];
        let src_ptr3 = &src[(offset + BUFFER_SIZE * 3)..(offset + BUFFER_SIZE * 4)];

        for (dst, src) in direct_store0.iter_mut().zip(src_ptr0.iter()) {
            *dst += src.as_() * weight;
        }
        for (dst, src) in direct_store1.iter_mut().zip(src_ptr1.iter()) {
            *dst += src.as_() * weight;
        }
        for (dst, src) in direct_store2.iter_mut().zip(src_ptr2.iter()) {
            *dst += src.as_() * weight;
        }
        for (dst, src) in direct_store3.iter_mut().zip(src_ptr3.iter()) {
            *dst += src.as_() * weight;
        }
    }

    let v_dst0 = &mut dst[v_start_px..(v_start_px + BUFFER_SIZE)];
    for (dst, src) in v_dst0.iter_mut().zip(direct_store0) {
        *dst = src.saturate_narrow(bit_depth);
    }

    let v_dst1 = &mut dst[(v_start_px + BUFFER_SIZE)..(v_start_px + BUFFER_SIZE * 2)];
    for (dst, src) in v_dst1.iter_mut().zip(direct_store1) {
        *dst = src.saturate_narrow(bit_depth);
    }

    let v_dst2 = &mut dst[(v_start_px + BUFFER_SIZE * 2)..(v_start_px + BUFFER_SIZE * 3)];
    for (dst, src) in v_dst2.iter_mut().zip(direct_store2) {
        *dst = src.saturate_narrow(bit_depth);
    }

    let v_dst3 = &mut dst[(v_start_px + BUFFER_SIZE * 3)..(v_start_px + BUFFER_SIZE * 4)];
    for (dst, src) in v_dst3.iter_mut().zip(direct_store3) {
        *dst = src.saturate_narrow(bit_depth);
    }
}

/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn column_handler_fixed_point<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy + 'static + AsPrimitive<T> + Mul<Output = J> + AddAssign + SaturateNarrow<T> + Default,
    const COMPONENTS: usize,
>(
    dst_width: usize,
    bounds: &FilterBounds,
    src: &[T],
    dst: &mut [T],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
{
    let mut cx = 0usize;

    let total_width = COMPONENTS * dst_width;

    while cx + 64 < total_width {
        convolve_column_handler_fixed_point_direct_buffer_four::<T, J, 16>(
            src, src_stride, dst, weight, bounds, bit_depth, cx,
        );

        cx += 64;
    }

    while cx + 32 < total_width {
        convolve_column_handler_fixed_point_direct_buffer_double::<T, J, 16>(
            src, src_stride, dst, weight, bounds, bit_depth, cx,
        );

        cx += 32;
    }

    while cx + 16 < total_width {
        convolve_column_handler_fixed_point_direct_buffer::<T, J, 16>(
            src, src_stride, dst, weight, bounds, bit_depth, cx,
        );

        cx += 16;
    }

    while cx + 8 < total_width {
        convolve_column_handler_fixed_point_direct_buffer::<T, J, 8>(
            src, src_stride, dst, weight, bounds, bit_depth, cx,
        );

        cx += 8;
    }

    while cx < total_width {
        convolve_column_handler_fixed_point_direct_buffer::<T, J, 1>(
            src, src_stride, dst, weight, bounds, bit_depth, cx,
        );

        cx += 1;
    }
}
