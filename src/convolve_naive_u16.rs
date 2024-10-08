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
use crate::support::{PRECISION, ROUNDING_CONST};
use num_traits::AsPrimitive;

macro_rules! compress_u16 {
    ($accumulator: expr, $max_colors: expr) => {{
        ($accumulator >> PRECISION).max(0).min($max_colors) as u16
    }};
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn convolve_vertical_part_u16<const BUFFER_SIZE: usize>(
    start_y: usize,
    start_x: usize,
    src: *const u16,
    src_stride: usize,
    dst: *mut u16,
    filter: &[i16],
    bounds: &FilterBounds,
    bit_depth: usize,
) {
    unsafe {
        let max_colors = (1 << bit_depth) - 1;
        let mut store: [i64; BUFFER_SIZE] = [ROUNDING_CONST.as_(); BUFFER_SIZE];

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = *filter.get_unchecked(j) as i64;
            let src_ptr = src.add(src_stride * py);
            for x in 0..BUFFER_SIZE {
                let px = start_x + x;
                let s_ptr = src_ptr.add(px);

                let store_p = store.get_unchecked_mut(x);
                *store_p += s_ptr.read_unaligned() as i64 * weight;
            }
        }

        for x in 0..BUFFER_SIZE {
            let px = start_x + x;
            let dst_ptr = dst.add(px);
            let vl = *store.get_unchecked_mut(x);
            dst_ptr.write_unaligned(compress_u16!(vl, max_colors));
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_native_row_u16<const CHANNELS: usize>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u16,
    unsafe_destination_ptr_0: *mut u16,
    bit_depth: usize,
) {
    let max_colors = (1 << bit_depth) - 1;
    unsafe {
        let mut filter_offset = 0usize;
        let weights = &filter_weights.weights;

        for x in 0..dst_width {
            let mut sums = ColorGroup::<CHANNELS, i64>::dup(ROUNDING_CONST as i64);

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight: i64 = (*weights.get_unchecked(j + filter_offset)) as i64;
                let new_px0 = ColorGroup::<CHANNELS, i64>::from_ptr(unsafe_source_ptr_0, px);
                sums += new_px0 * weight;
            }

            let px = x * CHANNELS;

            sums >>= PRECISION.as_();
            sums = sums.max_scalar(0).min_scalar(max_colors);
            sums.as_ptr(unsafe_destination_ptr_0, px);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn convolve_horizontal_rgba_native_4_row_u16<const CHANNELS: usize>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u16,
    dst_stride: usize,
    bit_depth: usize,
) {
    let max_colors = (1 << bit_depth) - 1;
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
            let mut sums0 = ColorGroup::<CHANNELS, i64>::dup(ROUNDING_CONST as i64);
            let mut sums1 = ColorGroup::<CHANNELS, i64>::dup(ROUNDING_CONST as i64);
            let mut sums2 = ColorGroup::<CHANNELS, i64>::dup(ROUNDING_CONST as i64);
            let mut sums3 = ColorGroup::<CHANNELS, i64>::dup(ROUNDING_CONST as i64);

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight = (*weights.get_unchecked(j + filter_offset)) as i64;

                let new_px0 = ColorGroup::<CHANNELS, i64>::from_ptr(src_row0, px);
                sums0 += new_px0 * weight;

                let new_px1 = ColorGroup::<CHANNELS, i64>::from_ptr(src_row1, px);
                sums1 += new_px1 * weight;

                let new_px2 = ColorGroup::<CHANNELS, i64>::from_ptr(src_row2, px);
                sums2 += new_px2 * weight;

                let new_px3 = ColorGroup::<CHANNELS, i64>::from_ptr(src_row3, px);
                sums3 += new_px3 * weight;
            }

            let px = x * CHANNELS;

            sums0 >>= PRECISION.as_();
            sums1 >>= PRECISION.as_();
            sums2 >>= PRECISION.as_();
            sums3 >>= PRECISION.as_();

            sums0 = sums0.max_scalar(0).min_scalar(max_colors);
            sums1 = sums1.max_scalar(0).min_scalar(max_colors);
            sums2 = sums2.max_scalar(0).min_scalar(max_colors);
            sums3 = sums3.max_scalar(0).min_scalar(max_colors);

            sums0.as_ptr(dst_row0, px);
            sums1.as_ptr(dst_row1, px);
            sums2.as_ptr(dst_row2, px);
            sums3.as_ptr(dst_row3, px);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_vertical_rgb_native_row_u16<const COMPONENTS: usize>(
    dst_width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const u16,
    unsafe_destination_ptr_0: *mut u16,
    src_stride: usize,
    weight_ptr: &[i16],
    bit_depth: usize,
) {
    let mut cx = 0usize;

    let total_width = COMPONENTS * dst_width;

    while cx + 36 < total_width {
        convolve_vertical_part_u16::<36>(
            bounds.start,
            cx,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            weight_ptr,
            bounds,
            bit_depth,
        );

        cx += 36;
    }

    while cx + 24 < total_width {
        convolve_vertical_part_u16::<24>(
            bounds.start,
            cx,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            weight_ptr,
            bounds,
            bit_depth,
        );

        cx += 24;
    }

    while cx + 12 < total_width {
        convolve_vertical_part_u16::<12>(
            bounds.start,
            cx,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            weight_ptr,
            bounds,
            bit_depth,
        );
        cx += 12;
    }

    while cx + 8 < total_width {
        convolve_vertical_part_u16::<8>(
            bounds.start,
            cx,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            weight_ptr,
            bounds,
            bit_depth,
        );

        cx += 8;
    }

    while cx < total_width {
        convolve_vertical_part_u16::<1>(
            bounds.start,
            cx,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            weight_ptr,
            bounds,
            bit_depth,
        );

        cx += 1;
    }
}
