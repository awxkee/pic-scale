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
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    for ((chunk, &bounds), weights) in dst
        .chunks_exact_mut(4)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut sums = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());

        let start_x = bounds.start;
        let bounds_size = bounds.size;

        const CN: usize = 4;
        let px = start_x * CN;

        if bounds_size == 2 {
            let src_ptr0 = &src[px..(px + 2 * CN)];
            let sliced_weights = &weights[0..2];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            sums += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1;
        } else if bounds_size == 3 {
            let src_ptr0 = &src[px..(px + 3 * CN)];
            let sliced_weights = &weights[0..3];
            let weight0 = sliced_weights[0] as i32;
            let weight1 = sliced_weights[1] as i32;
            let weight2 = sliced_weights[2] as i32;
            sums += load_ar30!(src_ptr0, AR30_TYPE, AR30_ORDER) * weight0
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 1) * weight1
                + load_ar30_with_offset!(src_ptr0, AR30_TYPE, AR30_ORDER, 2) * weight2;
        } else if bounds_size == 4 {
            let src_ptr0 = &src[px..(px + 4 * CN)];
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
            let src_ptr0 = &src[px..(px + 6 * CN)];

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
            for (&k_weight, src) in weights
                .iter()
                .zip(src_ptr0.chunks_exact(4))
                .take(bounds.size)
            {
                let weight: i32 = k_weight as i32;
                let new_px = load_ar30_p!(src, AR30_TYPE, AR30_ORDER);
                sums += new_px * weight;
            }
        }

        let narrowed = sums.saturate_ar30();
        let bytes0 = narrowed.to_ar30::<AR30_TYPE, AR30_ORDER>().to_ne_bytes();
        chunk[0] = bytes0[0];
        chunk[1] = bytes0[1];
        chunk[2] = bytes0[2];
        chunk[3] = bytes0[3];
    }
}

#[inline(always)]
pub(crate) fn convolve_row_handler_fixed_point_4_ar30<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    const CN: usize = 4;

    let iter_row0 = row0_ref.chunks_exact_mut(CN);
    let iter_row1 = row1_ref.chunks_exact_mut(CN);
    let iter_row2 = row2_ref.chunks_exact_mut(CN);
    let iter_row3 = row3_ref.chunks_exact_mut(CN);

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

        let px = start_x * CN;
        let bounds_size = bounds.size;

        if bounds_size == 2 {
            let src_ptr0 = &src[px..(px + 2 * CN)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + 2 * 4)];
            let src_ptr2 = &src[(px + src_stride * 2 * 4)..(px + src_stride * 2 + 2 * 4)];
            let src_ptr3 = &src[(px + src_stride * 3 * 4)..(px + src_stride * 3 + 2 * 4)];

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
            let src_ptr0 = &src[px..(px + 3 * CN)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + 3 * 4)];
            let src_ptr2 = &src[(px + src_stride * 2 * 4)..(px + src_stride * 2 + 3 * 4)];
            let src_ptr3 = &src[(px + src_stride * 3 * 4)..(px + src_stride * 3 + 3 * 4)];

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
            let src_ptr0 = &src[px..(px + 4 * CN)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + 4 * 4)];
            let src_ptr2 = &src[(px + src_stride * 2 * 4)..(px + src_stride * 2 + 4 * 4)];
            let src_ptr3 = &src[(px + src_stride * 3 * 4)..(px + src_stride * 3 + 4 * 4)];

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
            let src_ptr0 = &src[px..(px + 6 * CN)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + 6 * 4)];
            let src_ptr2 = &src[(px + src_stride * 2 * 4)..(px + src_stride * 2 + 6 * 4)];
            let src_ptr3 = &src[(px + src_stride * 3 * 4)..(px + src_stride * 3 + 6 * 4)];

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
            let src_ptr0 = &src[px..(px + bounds_size * CN)];
            let src_ptr1 = &src[(px + src_stride)..(px + src_stride + bounds_size * CN)];
            let src_ptr2 = &src[(px + src_stride * 2 * CN)..(px + src_stride * 2 + bounds_size * CN)];
            let src_ptr3 = &src[(px + src_stride * 3 * CN)..(px + src_stride * 3 + bounds_size * CN)];

            for ((((&k_weight, src0), src1), src2), src3) in weights
                .iter()
                .zip(src_ptr0.chunks_exact(4))
                .zip(src_ptr1.chunks_exact(4))
                .zip(src_ptr2.chunks_exact(4))
                .zip(src_ptr3.chunks_exact(4))
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

        let bytes0 = narrowed0.to_ar30::<AR30_TYPE, AR30_ORDER>().to_ne_bytes();
        chunk0[0] = bytes0[0];
        chunk0[1] = bytes0[1];
        chunk0[2] = bytes0[2];
        chunk0[3] = bytes0[3];

        let bytes1 = narrowed1.to_ar30::<AR30_TYPE, AR30_ORDER>().to_ne_bytes();
        chunk1[0] = bytes1[0];
        chunk1[1] = bytes1[1];
        chunk1[2] = bytes1[2];
        chunk1[3] = bytes1[3];

        let bytes2 = narrowed2.to_ar30::<AR30_TYPE, AR30_ORDER>().to_ne_bytes();
        chunk2[0] = bytes2[0];
        chunk2[1] = bytes2[1];
        chunk2[2] = bytes2[2];
        chunk2[3] = bytes2[3];

        let bytes3 = narrowed3.to_ar30::<AR30_TYPE, AR30_ORDER>().to_ne_bytes();
        chunk3[0] = bytes3[0];
        chunk3[1] = bytes3[1];
        chunk3[2] = bytes3[2];
        chunk3[3] = bytes3[3];
    }
}
