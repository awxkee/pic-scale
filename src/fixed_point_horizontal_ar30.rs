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
use crate::color_group::{ColorGroup, load_ar30_p};
use crate::filter_weights::FilterWeights;
use crate::support::ROUNDING_CONST;
use num_traits::AsPrimitive;

pub(crate) fn convolve_row_handler_fixed_point_ar30<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
    _: u32,
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

        let src_ptr0 = &src[px..(px + bounds_size)];
        for (&k_weight, src) in weights
            .iter()
            .zip(src_ptr0.chunks_exact(4))
            .take(bounds.size)
        {
            let weight: i32 = k_weight as i32;
            let new_px = load_ar30_p!(src, AR30_TYPE, AR30_ORDER);
            sums = sums.trunc_add(&new_px.trunc_mul(weight));
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
    _: u32,
) {
    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    const CN: usize = 4;

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
        let mut sums0 = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());
        let mut sums1 = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());
        let mut sums2 = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());
        let mut sums3 = ColorGroup::<4, i32>::dup(ROUNDING_CONST.as_());

        let start_x = bounds.start;

        let px = start_x * CN;
        let bounds_size = bounds.size;

        let src_ptr0 = &src[px..(px + bounds_size * CN)];
        let src_ptr1 = &src[(px + src_stride)..(px + src_stride + bounds_size * CN)];
        let src_ptr2 = &src[(px + src_stride * 2)..(px + src_stride * 2 + bounds_size * CN)];
        let src_ptr3 = &src[(px + src_stride * 3)..(px + src_stride * 3 + bounds_size * CN)];

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

            sums0 = sums0.trunc_add(&new_px0.trunc_mul(weight));
            sums1 = sums1.trunc_add(&new_px1.trunc_mul(weight));
            sums2 = sums2.trunc_add(&new_px2.trunc_mul(weight));
            sums3 = sums3.trunc_add(&new_px3.trunc_mul(weight));
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
