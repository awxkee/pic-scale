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
use crate::color_group::{ColorGroup, ld_g, st_g};
use crate::filter_weights::FilterWeights;
use crate::saturate_narrow::SaturateNarrow;
use crate::support::ROUNDING_CONST;
use num_traits::{AsPrimitive, WrappingAdd, WrappingMul};
use std::ops::{Add, AddAssign, Mul};

#[inline(always)]
pub(crate) fn convolve_row_handler_fixed_point<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy
        + 'static
        + AsPrimitive<T>
        + WrappingMul<Output = J>
        + AddAssign
        + SaturateNarrow<T>
        + Default
        + Add<J, Output = J>
        + WrappingAdd<Output = J>,
    const CN: usize,
>(
    src: &[T],
    dst: &mut [T],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
{
    for ((chunk, &bounds), weights) in dst
        .as_chunks_mut::<CN>()
        .0
        .iter_mut()
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut sums = ColorGroup::<CN, J>::dup(ROUNDING_CONST.as_());

        let start_x = bounds.start;
        let bounds_size = bounds.size;

        let px = start_x * CN;

        let src_ptr0 = &src[px..(px + bounds_size * CN)];
        for (&k_weight, src) in weights
            .iter()
            .zip(src_ptr0.as_chunks::<CN>().0.iter())
            .take(bounds.size)
        {
            let weight: J = k_weight.as_();
            let new_px = ld_g!(src, CN, J);
            sums = sums.trunc_add(&new_px.trunc_mul(weight));
        }

        let narrowed = sums.saturate_narrow(bit_depth);
        st_g!(narrowed, chunk, CN);
    }
}

#[inline(always)]
pub(crate) fn convolve_row_handler_fixed_point_4<
    T: Copy + 'static + AsPrimitive<J> + Default,
    J: Copy
        + 'static
        + AsPrimitive<T>
        + Mul<Output = J>
        + AddAssign
        + SaturateNarrow<T>
        + Default
        + Add<J, Output = J>
        + WrappingMul<Output = J>
        + WrappingAdd<Output = J>,
    const CN: usize,
>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) where
    i32: AsPrimitive<J>,
    i16: AsPrimitive<J>,
{
    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
    let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
    let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
    let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

    for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
        .iter_mut()
        .zip(iter_row1.iter_mut())
        .zip(iter_row2.iter_mut())
        .zip(iter_row3.iter_mut())
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut sums0 = ColorGroup::<CN, J>::dup(ROUNDING_CONST.as_());
        let mut sums1 = ColorGroup::<CN, J>::dup(ROUNDING_CONST.as_());
        let mut sums2 = ColorGroup::<CN, J>::dup(ROUNDING_CONST.as_());
        let mut sums3 = ColorGroup::<CN, J>::dup(ROUNDING_CONST.as_());

        let start_x = bounds.start;

        let px = start_x * CN;
        let bounds_size = bounds.size;

        let src_ptr0 = &src[px..(px + bounds_size * CN)];
        let src_ptr1 = &src[(px + src_stride)..(px + src_stride + bounds_size * CN)];
        let src_ptr2 = &src[(px + src_stride * 2)..(px + src_stride * 2 + bounds_size * CN)];
        let src_ptr3 = &src[(px + src_stride * 3)..(px + src_stride * 3 + bounds_size * CN)];

        for ((((&k_weight, src0), src1), src2), src3) in weights
            .iter()
            .zip(src_ptr0.as_chunks::<CN>().0.iter())
            .zip(src_ptr1.as_chunks::<CN>().0.iter())
            .zip(src_ptr2.as_chunks::<CN>().0.iter())
            .zip(src_ptr3.as_chunks::<CN>().0.iter())
            .take(bounds.size)
        {
            let weight: J = k_weight.as_();

            let new_px0 = ld_g!(src0, CN, J);
            let new_px1 = ld_g!(src1, CN, J);
            let new_px2 = ld_g!(src2, CN, J);
            let new_px3 = ld_g!(src3, CN, J);

            sums0 = sums0.trunc_add(&new_px0.trunc_mul(weight));
            sums1 = sums1.trunc_add(&new_px1.trunc_mul(weight));
            sums2 = sums2.trunc_add(&new_px2.trunc_mul(weight));
            sums3 = sums3.trunc_add(&new_px3.trunc_mul(weight));
        }

        let narrowed0 = sums0.saturate_narrow(bit_depth);
        let narrowed1 = sums1.saturate_narrow(bit_depth);
        let narrowed2 = sums2.saturate_narrow(bit_depth);
        let narrowed3 = sums3.saturate_narrow(bit_depth);

        st_g!(narrowed0, chunk0, CN);
        st_g!(narrowed1, chunk1, CN);
        st_g!(narrowed2, chunk2, CN);
        st_g!(narrowed3, chunk3, CN);
    }
}
