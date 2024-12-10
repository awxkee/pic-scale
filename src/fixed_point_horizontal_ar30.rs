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
use crate::color_group::{load_ar30, load_ar30_p, load_ar30_with_offset, ColorGroup};
use crate::filter_weights::FilterWeights;
use crate::support::ROUNDING_CONST;
use num_traits::AsPrimitive;

#[inline(always)]
pub(crate) fn convolve_row_handler_fixed_point_ar30<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
>(
    src: &[u32],
    dst: &mut [u32],
    filter_weights: &FilterWeights<i16>,
) {
    for ((chunk, &bounds), weights) in dst.iter_mut().zip(filter_weights.bounds.iter()).zip(
        filter_weights
            .weights
            .chunks_exact(filter_weights.aligned_size),
    ) {
        let mut sums = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());

        let start_x = bounds.start;
        let bounds_size = bounds.size;

        let px = start_x;

        if bounds_size == 2 {
            let src_ptr0 = &src[px..(px + 2)];
            let sliced_weights = &weights[0..2];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            sums += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1;
        } else if bounds_size == 3 {
            let src_ptr0 = &src[px..(px + 3)];
            let sliced_weights = &weights[0..3];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            let weight2 = sliced_weights[2] as i32;
            sums += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 2) * weight2;
        } else if bounds_size == 4 {
            let src_ptr0 = &src[px..(px + 4)];
            let sliced_weights = &weights[0..4];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            let weight2 = sliced_weights[2] as i32;
            let weight3 = sliced_weights[3] as i32;
            sums += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 3) * weight3;
        } else if bounds_size == 6 {
            let src_ptr0 = &src[px..(px + 6)];

            let sliced_weights = &weights[0..6];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            let weight2 = sliced_weights[2] as i32;
            let weight3 = sliced_weights[3] as i32;
            let weight4 = sliced_weights[4] as i32;
            let weight5 = sliced_weights[5] as i32;
            sums += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 3) * weight3
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 4) * weight4
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 5) * weight5;
        } else {
            let src_ptr0 = &src[px..(px + bounds_size)];
            for (&k_weight, src) in weights.iter().zip(src_ptr0.iter()).take(bounds.size) {
                let weight: i32 = k_weight as i32;
                let new_px = load_ar30_p!(src, AR30_TYPE, AR30_ORDER);
                sums += new_px * weight;
            }
        }

        let narrowed = sums.saturate_ar30();
        *chunk = narrowed.to_ar30::<AR30_TYPE, AR30_ORDER>();
    }
}

#[inline(always)]
pub(crate) fn convolve_row_handler_fixed_point_4_ar30<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
>(
    src: &[u32],
    src_stride: usize,
    dst: &mut [u32],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.iter_mut();
    let iter_row1 = row1_ref.iter_mut();
    let iter_row2 = row2_ref.iter_mut();
    let iter_row3 = row3_ref.iter_mut();

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
        let mut sums0 = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());
        let mut sums1 = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());
        let mut sums2 = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());
        let mut sums3 = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());

        let start_x = bounds.start;

        let px = start_x;
        let bounds_size = bounds.size;

        if bounds_size == 2 {
            let src_ptr0 = &src[px..(px + 2)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + 2)];
            let src_ptr2 = &src[(px + src_stride * 2)..(px + src_stride * 2 + 2)];
            let src_ptr3 = &src[(px + src_stride * 3)..(px + src_stride * 3 + 2)];

            let sliced_weights = &weights[0..2];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            sums0 += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1;
            sums1 += load_ar30!(src_ptr1, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 1) * weight1;
            sums2 += load_ar30!(src_ptr2, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 1) * weight1;
            sums3 += load_ar30!(src_ptr3, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 1) * weight1;
        } else if bounds_size == 3 {
            let src_ptr0 = &src[px..(px + 3)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + 3)];
            let src_ptr2 = &src[(px + src_stride * 2)..(px + src_stride * 2 + 3)];
            let src_ptr3 = &src[(px + src_stride * 3)..(px + src_stride * 3 + 3)];

            let sliced_weights = &weights[0..3];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            let weight2 = sliced_weights[2] as i32;
            sums0 += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 2) * weight2;
            sums1 += load_ar30!(src_ptr1, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 2) * weight2;
            sums2 += load_ar30!(src_ptr2, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 2) * weight2;
            sums3 += load_ar30!(src_ptr3, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 2) * weight2;
        } else if bounds_size == 4 {
            let src_ptr0 = &src[px..(px + 4)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + 4)];
            let src_ptr2 = &src[(px + src_stride * 2)..(px + src_stride * 2 + 4)];
            let src_ptr3 = &src[(px + src_stride * 3)..(px + src_stride * 3 + 4)];

            let sliced_weights = &weights[0..4];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            let weight2 = sliced_weights[2] as i32;
            let weight3 = sliced_weights[3] as i32;
            sums0 += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 3) * weight3;
            sums1 += load_ar30!(src_ptr1, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 3) * weight3;
            sums2 += load_ar30!(src_ptr2, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 3) * weight3;
            sums3 += load_ar30!(src_ptr3, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 3) * weight3;
        } else if bounds_size == 6 {
            let src_ptr0 = &src[px..(px + 6)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + 6)];
            let src_ptr2 = &src[(px + src_stride * 2)..(px + src_stride * 2 + 6)];
            let src_ptr3 = &src[(px + src_stride * 3)..(px + src_stride * 3 + 6)];

            let sliced_weights = &weights[0..6];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            let weight2 = sliced_weights[2] as i32;
            let weight3 = sliced_weights[3] as i32;
            let weight4 = sliced_weights[4] as i32;
            let weight5 = sliced_weights[5] as i32;
            sums0 += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 3) * weight3
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 4) * weight4
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 5) * weight5;
            sums1 += load_ar30!(src_ptr1, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 3) * weight3
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 4) * weight4
                + load_ar30_with_offset!(src_ptr1, AR30_TYPE, AR30_ORDER, 5) * weight5;
            sums2 += load_ar30!(src_ptr2, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 3) * weight3
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 4) * weight4
                + load_ar30_with_offset!(src_ptr2, AR30_TYPE, AR30_ORDER, 5) * weight5;
            sums3 += load_ar30!(src_ptr3, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 2) * weight2
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 3) * weight3
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 4) * weight4
                + load_ar30_with_offset!(src_ptr3, AR30_TYPE, AR30_ORDER, 5) * weight5;
        } else {
            let src_ptr0 = &src[px..(px + bounds_size)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + bounds_size)];
            let src_ptr2 = &src[(px + src_stride * 2)..(px + src_stride * 2 + bounds_size)];
            let src_ptr3 = &src[(px + src_stride * 3)..(px + src_stride * 3 + bounds_size)];

            for ((((&k_weight, src0), src1), src2), src3) in weights
                .iter()
                .zip(src_ptr0.iter())
                .zip(src_ptr1.iter())
                .zip(src_ptr2.iter())
                .zip(src_ptr3.iter())
                .take(bounds.size)
            {
                let weight: i32 = k_weight as i32;

                let new_px0 = load_ar30_p!(src0, AR30_TYPE, AR30_ORDER);
                let new_px1 = load_ar30_p!(src1, AR30_TYPE, AR30_ORDER);
                let new_px2 = load_ar30_p!(src2, AR30_TYPE, AR30_ORDER);
                let new_px3 = load_ar30_p!(src3, AR30_TYPE, AR30_ORDER);

                sums0 += new_px0 * weight;
                sums1 += new_px1 * weight;
                sums2 += new_px2 * weight;
                sums3 += new_px3 * weight;
            }
        }

        let narrowed0 = sums0.saturate_ar30();
        let narrowed1 = sums1.saturate_ar30();
        let narrowed2 = sums2.saturate_ar30();
        let narrowed3 = sums3.saturate_ar30();

        *chunk0 = narrowed0.to_ar30::<AR30_TYPE, AR30_ORDER>();
        *chunk1 = narrowed1.to_ar30::<AR30_TYPE, AR30_ORDER>();
        *chunk2 = narrowed2.to_ar30::<AR30_TYPE, AR30_ORDER>();
        *chunk3 = narrowed3.to_ar30::<AR30_TYPE, AR30_ORDER>();
    }
}
