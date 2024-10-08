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
use crate::saturate_narrow::SaturateNarrow;
use crate::support::ROUNDING_CONST;
use num_traits::AsPrimitive;
use std::ops::{AddAssign, Mul};

#[inline]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) unsafe fn convolve_vertical_part<
    T: Copy + 'static + AsPrimitive<J>,
    J: Copy + 'static + AsPrimitive<T> + Mul<Output = J> + AddAssign + SaturateNarrow<T>,
    const BUFFER_SIZE: usize,
>(
    start_y: usize,
    start_x: usize,
    src: *const T,
    src_stride: usize,
    dst: *mut T,
    filter: &[i16],
    bounds: &FilterBounds,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
{
    let mut store: [J; BUFFER_SIZE] = [ROUNDING_CONST.as_(); BUFFER_SIZE];

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { *filter.get_unchecked(j) }.as_();
        let src_ptr = src.add(src_stride * py);
        for x in 0..BUFFER_SIZE {
            let px = start_x + x;
            let s_ptr = src_ptr.add(px);

            let store_p = store.get_unchecked_mut(x);
            *store_p += unsafe { s_ptr.read_unaligned() }.as_() * weight;
        }
    }

    for x in 0..BUFFER_SIZE {
        let px = start_x + x;
        let dst_ptr = dst.add(px);
        let vl = *store.get_unchecked_mut(x);
        dst_ptr.write_unaligned(vl.saturate_narrow());
    }
}

pub(crate) fn convolve_horizontal_rgba_native_row<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy + 'static + AsPrimitive<T> + Mul<Output = J> + AddAssign + SaturateNarrow<T> + Default,
    const CHANNELS: usize,
>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const T,
    unsafe_destination_ptr_0: *mut T,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
{
    unsafe {
        let mut filter_offset = 0usize;
        let weights = &filter_weights.weights;

        for x in 0..dst_width {
            let mut sums = ColorGroup::<CHANNELS, J>::dup(ROUNDING_CONST.as_());

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight: J = weights.get_unchecked(j + filter_offset).as_();
                let new_px = ColorGroup::<CHANNELS, J>::from_ptr(unsafe_source_ptr_0, px);
                sums += new_px * weight;
            }

            let px = x * CHANNELS;

            let narrowed = sums.saturate_narrow();
            narrowed.to_ptr(unsafe_destination_ptr_0, px);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn convolve_horizontal_rgba_native_4_row<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy + 'static + AsPrimitive<T> + Mul<Output = J> + AddAssign + SaturateNarrow<T> + Default,
    const CHANNELS: usize,
>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const T,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut T,
    dst_stride: usize,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
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
            let mut sums0 = ColorGroup::<CHANNELS, J>::dup(ROUNDING_CONST.as_());
            let mut sums1 = ColorGroup::<CHANNELS, J>::dup(ROUNDING_CONST.as_());
            let mut sums2 = ColorGroup::<CHANNELS, J>::dup(ROUNDING_CONST.as_());
            let mut sums3 = ColorGroup::<CHANNELS, J>::dup(ROUNDING_CONST.as_());

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight = weights.get_unchecked(j + filter_offset).as_();

                let new_px0 = ColorGroup::<CHANNELS, J>::from_ptr(src_row0, px);
                sums0 += new_px0 * weight;

                let new_px1 = ColorGroup::<CHANNELS, J>::from_ptr(src_row1, px);
                sums1 += new_px1 * weight;

                let new_px2 = ColorGroup::<CHANNELS, J>::from_ptr(src_row2, px);
                sums2 += new_px2 * weight;

                let new_px3 = ColorGroup::<CHANNELS, J>::from_ptr(src_row3, px);
                sums3 += new_px3 * weight;
            }

            let px = x * CHANNELS;

            let narrow0 = sums0.saturate_narrow();
            let narrow1 = sums1.saturate_narrow();
            let narrow2 = sums2.saturate_narrow();
            let narrow3 = sums3.saturate_narrow();

            narrow0.to_ptr(dst_row0, px);
            narrow1.to_ptr(dst_row1, px);
            narrow2.to_ptr(dst_row2, px);
            narrow3.to_ptr(dst_row3, px);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
