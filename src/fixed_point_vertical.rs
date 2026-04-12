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
#![forbid(unsafe_code)]
use crate::filter_weights::FilterBounds;
use crate::saturate_narrow::SaturateNarrow;
use crate::support::ROUNDING_CONST;
use num_traits::{AsPrimitive, WrappingAdd, WrappingMul};
use std::ops::{AddAssign, Mul};

pub(crate) trait RoundableAccumulator<T> {
    const ROUNDING: T;
}

impl RoundableAccumulator<i32> for i32 {
    const ROUNDING: i32 = ROUNDING_CONST;
}

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn convolve_column_handler_fixed_point_direct_buffer<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy
        + 'static
        + AsPrimitive<T>
        + Mul<Output = J>
        + AddAssign
        + SaturateNarrow<T>
        + RoundableAccumulator<J>
        + WrappingMul<Output = J>
        + WrappingAdd<Output = J>,
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

    let rc: J = J::ROUNDING;
    let mut store0: [J; 4] = [rc; 4];
    let mut store1: [J; 4] = [rc; 4];
    let mut store2: [J; 4] = [rc; 4];
    let mut store3: [J; 4] = [rc; 4];

    let base = src_stride * bounds.start + x;
    let quarter = BUFFER_SIZE / 4;

    for (j, filter_weight) in filter[..bounds.size].iter().enumerate() {
        let w: J = filter_weight.as_();
        let off = base + src_stride * j;

        let (chunk0, rest) = src[off..off + BUFFER_SIZE].split_at(quarter);
        let (chunk1, rest) = rest.split_at(quarter);
        let (chunk2, chunk3) = rest.split_at(quarter);

        for (acc, &s) in store0.iter_mut().zip(chunk0) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
        for (acc, &s) in store1.iter_mut().zip(chunk1) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
        for (acc, &s) in store2.iter_mut().zip(chunk2) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
        for (acc, &s) in store3.iter_mut().zip(chunk3) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
    }

    let v_dst = &mut dst[x..x + BUFFER_SIZE];
    let (out0, rest) = v_dst.split_at_mut(quarter);
    let (out1, rest) = rest.split_at_mut(quarter);
    let (out2, out3) = rest.split_at_mut(quarter);

    for (d, s) in out0.iter_mut().zip(store0) {
        *d = s.saturate_narrow(bit_depth);
    }
    for (d, s) in out1.iter_mut().zip(store1) {
        *d = s.saturate_narrow(bit_depth);
    }
    for (d, s) in out2.iter_mut().zip(store2) {
        *d = s.saturate_narrow(bit_depth);
    }
    for (d, s) in out3.iter_mut().zip(store3) {
        *d = s.saturate_narrow(bit_depth);
    }
}

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn convolve_column_handler_fixed_point_direct_buffer_double<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy
        + 'static
        + AsPrimitive<T>
        + Mul<Output = J>
        + AddAssign
        + SaturateNarrow<T>
        + RoundableAccumulator<J>
        + WrappingMul<Output = J>
        + WrappingAdd<Output = J>,
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

    let rc: J = J::ROUNDING;
    let quarter = BUFFER_SIZE / 4;

    let mut s00: [J; 4] = [rc; 4];
    let mut s01: [J; 4] = [rc; 4];
    let mut s02: [J; 4] = [rc; 4];
    let mut s03: [J; 4] = [rc; 4];

    let mut s10: [J; 4] = [rc; 4];
    let mut s11: [J; 4] = [rc; 4];
    let mut s12: [J; 4] = [rc; 4];
    let mut s13: [J; 4] = [rc; 4];

    let base = src_stride * bounds.start + x;

    for (j, filter_weight) in filter[..bounds.size].iter().enumerate() {
        let w: J = filter_weight.as_();
        let off = base + src_stride * j;

        let (c00, rest) = src[off..off + BUFFER_SIZE].split_at(quarter);
        let (c01, rest) = rest.split_at(quarter);
        let (c02, c03) = rest.split_at(quarter);

        let off1 = off + BUFFER_SIZE;
        let (c10, rest) = src[off1..off1 + BUFFER_SIZE].split_at(quarter);
        let (c11, rest) = rest.split_at(quarter);
        let (c12, c13) = rest.split_at(quarter);

        for (acc, &s) in s00.iter_mut().zip(c00) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
        for (acc, &s) in s01.iter_mut().zip(c01) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
        for (acc, &s) in s02.iter_mut().zip(c02) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
        for (acc, &s) in s03.iter_mut().zip(c03) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }

        for (acc, &s) in s10.iter_mut().zip(c10) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
        for (acc, &s) in s11.iter_mut().zip(c11) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
        for (acc, &s) in s12.iter_mut().zip(c12) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
        for (acc, &s) in s13.iter_mut().zip(c13) {
            *acc = acc.wrapping_add(&(s.as_() * w));
        }
    }

    let v_dst0 = &mut dst[x..x + BUFFER_SIZE];
    let (o00, rest) = v_dst0.split_at_mut(quarter);
    let (o01, rest) = rest.split_at_mut(quarter);
    let (o02, o03) = rest.split_at_mut(quarter);

    for (d, s) in o00.iter_mut().zip(s00) {
        *d = s.saturate_narrow(bit_depth);
    }
    for (d, s) in o01.iter_mut().zip(s01) {
        *d = s.saturate_narrow(bit_depth);
    }
    for (d, s) in o02.iter_mut().zip(s02) {
        *d = s.saturate_narrow(bit_depth);
    }
    for (d, s) in o03.iter_mut().zip(s03) {
        *d = s.saturate_narrow(bit_depth);
    }

    let v_dst1 = &mut dst[x + BUFFER_SIZE..x + BUFFER_SIZE * 2];
    let (o10, rest) = v_dst1.split_at_mut(quarter);
    let (o11, rest) = rest.split_at_mut(quarter);
    let (o12, o13) = rest.split_at_mut(quarter);

    for (d, s) in o10.iter_mut().zip(s10) {
        *d = s.saturate_narrow(bit_depth);
    }
    for (d, s) in o11.iter_mut().zip(s11) {
        *d = s.saturate_narrow(bit_depth);
    }
    for (d, s) in o12.iter_mut().zip(s12) {
        *d = s.saturate_narrow(bit_depth);
    }
    for (d, s) in o13.iter_mut().zip(s13) {
        *d = s.saturate_narrow(bit_depth);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn convolve_column_handler_fixed_point_direct_buffer_four<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy
        + 'static
        + AsPrimitive<T>
        + Mul<Output = J>
        + AddAssign
        + SaturateNarrow<T>
        + RoundableAccumulator<J>
        + WrappingAdd<Output = J>
        + WrappingMul<Output = J>,
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
    let mut direct_store0: [J; BUFFER_SIZE] = [J::ROUNDING; BUFFER_SIZE];
    let mut direct_store1: [J; BUFFER_SIZE] = [J::ROUNDING; BUFFER_SIZE];
    let mut direct_store2: [J; BUFFER_SIZE] = [J::ROUNDING; BUFFER_SIZE];
    let mut direct_store3: [J; BUFFER_SIZE] = [J::ROUNDING; BUFFER_SIZE];

    let v_start_px = x;

    let py = bounds.start;
    let weight = filter[0].as_();
    let offset = src_stride * py + v_start_px;
    let src_ptr0 = &src[offset..(offset + BUFFER_SIZE)];
    let src_ptr1 = &src[(offset + BUFFER_SIZE)..(offset + BUFFER_SIZE * 2)];
    let src_ptr2 = &src[(offset + BUFFER_SIZE * 2)..(offset + BUFFER_SIZE * 3)];
    let src_ptr3 = &src[(offset + BUFFER_SIZE * 3)..(offset + BUFFER_SIZE * 4)];

    for (dst, src) in direct_store0.iter_mut().zip(src_ptr0) {
        *dst = dst.wrapping_add(&src.as_().wrapping_mul(&weight));
    }

    for (dst, src) in direct_store1.iter_mut().zip(src_ptr1) {
        *dst = dst.wrapping_add(&src.as_().wrapping_mul(&weight));
    }

    for (dst, src) in direct_store2.iter_mut().zip(src_ptr2) {
        *dst = dst.wrapping_add(&src.as_().wrapping_mul(&weight));
    }

    for (dst, src) in direct_store3.iter_mut().zip(src_ptr3) {
        *dst = dst.wrapping_add(&src.as_().wrapping_mul(&weight));
    }

    for (j, &k_weight) in filter[1..bounds.size].iter().enumerate() {
        // Adding 1 is necessary because skip do not incrementing value on values that skipped
        let py = bounds.start + j + 1;
        let weight = k_weight.as_();
        let offset = src_stride * py + v_start_px;
        let src_ptr0 = &src[offset..(offset + BUFFER_SIZE)];
        let src_ptr1 = &src[(offset + BUFFER_SIZE)..(offset + BUFFER_SIZE * 2)];
        let src_ptr2 = &src[(offset + BUFFER_SIZE * 2)..(offset + BUFFER_SIZE * 3)];
        let src_ptr3 = &src[(offset + BUFFER_SIZE * 3)..(offset + BUFFER_SIZE * 4)];

        for (dst, src) in direct_store0.iter_mut().zip(src_ptr0.iter()) {
            *dst = dst.wrapping_add(&src.as_().wrapping_mul(&weight));
        }
        for (dst, src) in direct_store1.iter_mut().zip(src_ptr1.iter()) {
            *dst = dst.wrapping_add(&src.as_().wrapping_mul(&weight));
        }
        for (dst, src) in direct_store2.iter_mut().zip(src_ptr2.iter()) {
            *dst = dst.wrapping_add(&src.as_().wrapping_mul(&weight));
        }
        for (dst, src) in direct_store3.iter_mut().zip(src_ptr3.iter()) {
            *dst = dst.wrapping_add(&src.as_().wrapping_mul(&weight));
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

#[inline(never)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn column_handler_fixed_point<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy
        + 'static
        + AsPrimitive<T>
        + Mul<Output = J>
        + AddAssign
        + SaturateNarrow<T>
        + RoundableAccumulator<J>
        + WrappingMul<Output = J>
        + WrappingAdd<Output = J>,
>(
    _: usize,
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

    let total_width = dst.len();

    #[cfg(target_arch = "aarch64")]
    while cx + 64 <= total_width {
        convolve_column_handler_fixed_point_direct_buffer_four::<T, J, 16>(
            src, src_stride, dst, weight, bounds, bit_depth, cx,
        );

        cx += 64;
    }

    while cx + 32 <= total_width {
        convolve_column_handler_fixed_point_direct_buffer_double::<T, J, 16>(
            src, src_stride, dst, weight, bounds, bit_depth, cx,
        );

        cx += 32;
    }

    while cx + 16 <= total_width {
        convolve_column_handler_fixed_point_direct_buffer::<T, J, 16>(
            src, src_stride, dst, weight, bounds, bit_depth, cx,
        );

        cx += 16;
    }

    while cx + 8 <= total_width {
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
