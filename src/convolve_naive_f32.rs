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
use crate::color_group::ColorGroup;
use crate::filter_weights::{FilterBounds, FilterWeights};
use num_traits::{AsPrimitive, MulAdd};
use std::ops::{Add, Mul};

pub(crate) unsafe fn convolve_vertical_part_f32<
    T: Copy + 'static + AsPrimitive<I>,
    I: Copy
        + 'static
        + AsPrimitive<T>
        + Default
        + MulAdd<I, Output = I>
        + Mul<I, Output = I>
        + Add<I, Output = I>,
    const CHANNELS: usize,
>(
    start_y: usize,
    start_x: usize,
    src: *const T,
    src_stride: usize,
    dst: *mut T,
    filter: &[f32],
    bounds: &FilterBounds,
) where
    f32: AsPrimitive<I>,
{
    let mut sums0 = ColorGroup::<CHANNELS, I>::dup(I::default());

    let v_start_px = start_x * CHANNELS;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight: I = filter.get_unchecked(j).as_();
        let src_ptr = src.add(src_stride * py);

        let new_px0 = ColorGroup::<CHANNELS, I>::from_ptr(src_ptr, v_start_px);

        sums0 = sums0.mul_add(new_px0, weight);
    }

    sums0.as_ptr(dst, v_start_px);
}

pub(crate) unsafe fn convolve_vertical_part_4_f32<
    T: Copy + 'static + AsPrimitive<I>,
    I: Copy
        + 'static
        + AsPrimitive<T>
        + Default
        + MulAdd<I, Output = I>
        + Mul<I, Output = I>
        + Add<I, Output = I>,
    const CHANNELS: usize,
>(
    start_y: usize,
    start_x: usize,
    src: *const T,
    src_stride: usize,
    dst: *mut T,
    filter: &[f32],
    bounds: &FilterBounds,
) where
    f32: AsPrimitive<I>,
{
    let mut sums0 = ColorGroup::<CHANNELS, I>::dup(I::default());
    let mut sums1 = ColorGroup::<CHANNELS, I>::dup(I::default());
    let mut sums2 = ColorGroup::<CHANNELS, I>::dup(I::default());
    let mut sums3 = ColorGroup::<CHANNELS, I>::dup(I::default());

    let v_start_px = start_x * CHANNELS;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight: I = filter.get_unchecked(j).as_();
        let src_ptr = src.add(src_stride * py);

        let new_px0 = ColorGroup::<CHANNELS, I>::from_ptr(src_ptr, v_start_px);
        let new_px1 = ColorGroup::<CHANNELS, I>::from_ptr(src_ptr, v_start_px + CHANNELS);
        let new_px2 = ColorGroup::<CHANNELS, I>::from_ptr(src_ptr, v_start_px + CHANNELS * 2);
        let new_px3 = ColorGroup::<CHANNELS, I>::from_ptr(src_ptr, v_start_px + CHANNELS * 3);

        sums0 = sums0.mul_add(new_px0, weight);
        sums1 = sums1.mul_add(new_px1, weight);
        sums2 = sums2.mul_add(new_px2, weight);
        sums3 = sums3.mul_add(new_px3, weight);
    }

    sums0.as_ptr(dst, v_start_px);
    sums0.as_ptr(dst, v_start_px + CHANNELS);
    sums0.as_ptr(dst, v_start_px + CHANNELS * 2);
    sums0.as_ptr(dst, v_start_px + CHANNELS * 3);
}

#[inline]
pub(crate) fn convolve_horizontal_rgb_native_row<
    T: Copy + 'static + AsPrimitive<I>,
    I: Copy
        + 'static
        + Default
        + MulAdd<I, Output = I>
        + AsPrimitive<T>
        + Mul<I, Output = I>
        + Add<I, Output = I>,
    const CHANNELS: usize,
>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const T,
    unsafe_destination_ptr_0: *mut T,
) where
    f32: AsPrimitive<T> + AsPrimitive<I>,
{
    unsafe {
        let weights_ptr = &filter_weights.weights;
        let mut filter_offset = 0usize;

        for x in 0..dst_width {
            let mut sums = ColorGroup::<CHANNELS, I>::dup(0f32.as_());

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight = *weights_ptr.get_unchecked(j + filter_offset);

                let new_px = ColorGroup::<CHANNELS, I>::from_ptr(unsafe_source_ptr_0, px);

                sums = sums.mul_add(new_px, weight.as_());
            }

            let px = x * CHANNELS;

            sums.as_ptr(unsafe_destination_ptr_0, px);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_4_row_f32<
    T: Copy + 'static + AsPrimitive<I>,
    I: Copy
        + 'static
        + Default
        + MulAdd<I, Output = I>
        + AsPrimitive<T>
        + Mul<I, Output = I>
        + Add<I, Output = I>,
    const CHANNELS: usize,
>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const T,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut T,
    dst_stride: usize,
) where
    f32: AsPrimitive<T> + AsPrimitive<I>,
{
    unsafe {
        let mut filter_offset = 0usize;
        let weights = &filter_weights.weights;

        let src_row0 = unsafe_source_ptr_0;
        let src_row1 = unsafe_source_ptr_0.add(src_stride);
        let src_row2 = unsafe_source_ptr_0.add(src_stride * 2);
        let src_row3 = unsafe_source_ptr_0.add(src_stride * 3);

        let dst_row0 = unsafe_destination_ptr_0;
        let dst_row1 = unsafe_destination_ptr_0.add(dst_stride);
        let dst_row2 = unsafe_destination_ptr_0.add(dst_stride * 2);
        let dst_row3 = unsafe_destination_ptr_0.add(dst_stride * 3);

        for x in 0..dst_width {
            let mut sums0 = ColorGroup::<CHANNELS, I>::dup(0f32.as_());
            let mut sums1 = ColorGroup::<CHANNELS, I>::dup(0f32.as_());
            let mut sums2 = ColorGroup::<CHANNELS, I>::dup(0f32.as_());
            let mut sums3 = ColorGroup::<CHANNELS, I>::dup(0f32.as_());

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight = *weights.get_unchecked(j + filter_offset);

                let new_px0 = ColorGroup::<CHANNELS, I>::from_ptr(src_row0, px);
                sums0 = sums0.mul_add(new_px0, weight.as_());

                let new_px1 = ColorGroup::<CHANNELS, I>::from_ptr(src_row1, px);
                sums1 = sums1.mul_add(new_px1, weight.as_());

                let new_px2 = ColorGroup::<CHANNELS, I>::from_ptr(src_row2, px);
                sums2 = sums2.mul_add(new_px2, weight.as_());

                let new_px3 = ColorGroup::<CHANNELS, I>::from_ptr(src_row3, px);
                sums3 = sums3.mul_add(new_px3, weight.as_());
            }

            let px = x * CHANNELS;

            sums0.as_ptr(dst_row0, px);
            sums1.as_ptr(dst_row1, px);
            sums2.as_ptr(dst_row2, px);
            sums3.as_ptr(dst_row3, px);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
