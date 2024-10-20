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
use crate::color_group::ColorGroup;
use crate::filter_weights::FilterWeights;
use crate::mixed_storage::MixedStorage;
use crate::{fast_load_color_group, fast_mixed_store_color_group};
use num_traits::{AsPrimitive, Float, MulAdd};
use std::ops::{Add, Mul};

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
/// `F` - filter floating type
pub(crate) fn convolve_row_handler_floating_point<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy
        + 'static
        + AsPrimitive<T>
        + MulAdd<J, Output = J>
        + Mul<J, Output = J>
        + Add<J, Output = J>
        + Default
        + MixedStorage<T>,
    F: Copy + 'static + Float + AsPrimitive<J>,
    const CHANNELS: usize,
>(
    src: &[T],
    dst: &mut [T],
    filter_weights: &FilterWeights<F>,
    bit_depth: u32,
) where
    i32: AsPrimitive<J>,
{
    unsafe {
        for ((chunk, &bounds), weights) in dst
            .chunks_exact_mut(CHANNELS)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut sums = ColorGroup::<CHANNELS, J>::dup(0.as_());

            let start_x = bounds.start;

            let px = start_x * CHANNELS;

            let src_ptr0 = src.get_unchecked(px..(px + bounds.size * CHANNELS));

            for (&k_weight, src) in weights
                .iter()
                .zip(src_ptr0.chunks_exact(CHANNELS))
                .take(bounds.size)
            {
                let weight: J = k_weight.as_();
                let new_px = fast_load_color_group!(src, CHANNELS);
                sums = sums.mul_add(new_px, weight);
            }

            fast_mixed_store_color_group!(sums, chunk, CHANNELS, bit_depth);
        }
    }
}

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
/// `F` - filter floating type
pub(crate) fn convolve_row_handler_floating_point_4<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy
        + 'static
        + AsPrimitive<T>
        + MulAdd<J, Output = J>
        + Mul<J, Output = J>
        + Add<J, Output = J>
        + Default
        + MixedStorage<T>,
    F: Copy + 'static + Float + AsPrimitive<J>,
    const CHANNELS: usize,
>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    dst_stride: usize,
    filter_weights: &FilterWeights<F>,
    bit_depth: u32,
) where
    i32: AsPrimitive<J>,
{
    unsafe {
        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
        let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
        let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
        let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

        for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
            .zip(iter_row1)
            .zip(iter_row2)
            .zip(iter_row3)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut sums0 = ColorGroup::<CHANNELS, J>::dup(0.as_());
            let mut sums1 = ColorGroup::<CHANNELS, J>::dup(0.as_());
            let mut sums2 = ColorGroup::<CHANNELS, J>::dup(0.as_());
            let mut sums3 = ColorGroup::<CHANNELS, J>::dup(0.as_());

            let start_x = bounds.start;
            let px = start_x * CHANNELS;
            let src_ptr0 = src.get_unchecked(px..(px + bounds.size * CHANNELS));
            let src_ptr1 =
                src.get_unchecked((px + src_stride)..(px + src_stride + bounds.size * CHANNELS));
            let src_ptr2 = src.get_unchecked(
                (px + src_stride * 2)..(px + src_stride * 2 + bounds.size * CHANNELS),
            );
            let src_ptr3 = src.get_unchecked(
                (px + src_stride * 3)..(px + src_stride * 3 + bounds.size * CHANNELS),
            );

            for ((((&k_weight, src0), src1), src2), src3) in weights
                .iter()
                .zip(src_ptr0.chunks_exact(CHANNELS))
                .zip(src_ptr1.chunks_exact(CHANNELS))
                .zip(src_ptr2.chunks_exact(CHANNELS))
                .zip(src_ptr3.chunks_exact(CHANNELS))
                .take(bounds.size)
            {
                let weight: J = k_weight.as_();

                let new_px0 = fast_load_color_group!(src0, CHANNELS);
                let new_px1 = fast_load_color_group!(src1, CHANNELS);
                let new_px2 = fast_load_color_group!(src2, CHANNELS);
                let new_px3 = fast_load_color_group!(src3, CHANNELS);

                sums0 = sums0.mul_add(new_px0, weight);
                sums1 = sums1.mul_add(new_px1, weight);
                sums2 = sums2.mul_add(new_px2, weight);
                sums3 = sums3.mul_add(new_px3, weight);
            }

            fast_mixed_store_color_group!(sums0, chunk0, CHANNELS, bit_depth);
            fast_mixed_store_color_group!(sums1, chunk1, CHANNELS, bit_depth);
            fast_mixed_store_color_group!(sums2, chunk2, CHANNELS, bit_depth);
            fast_mixed_store_color_group!(sums3, chunk3, CHANNELS, bit_depth);
        }
    }
}
