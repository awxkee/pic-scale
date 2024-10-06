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
use num_traits::AsPrimitive;

use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::support::{PRECISION, ROUNDING_APPROX};

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
    filter: *const i16,
    bounds: &FilterBounds,
    bit_depth: usize,
) {
    unsafe {
        let max_colors = 2i64.pow(bit_depth as u32) - 1;
        let mut store: [i64; BUFFER_SIZE] = [ROUNDING_APPROX.as_(); BUFFER_SIZE];

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.add(j).read_unaligned() as i64;
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

macro_rules! make_naive_sum {
    ($sum_r:expr, $sum_g:expr, $sum_b:expr, $sum_a:expr, $weight: expr,
        $src:expr, $channels:expr) => {{
        $sum_r += $src.read_unaligned() as i64 * $weight;
        if $channels > 1 {
            $sum_g += $src.add(1).read_unaligned() as i64 * $weight;
        }
        if $channels > 2 {
            $sum_b += $src.add(2).read_unaligned() as i64 * $weight;
        }
        if $channels == 4 {
            $sum_a += $src.add(3).read_unaligned() as i64 * $weight;
        }
    }};
}

macro_rules! write_out_pixels {
    ($sum_r:expr, $sum_g:expr, $sum_b:expr, $sum_a:expr, $dst:expr, $channels:expr, $max_colors: expr) => {{
        $dst.write_unaligned(compress_u16!($sum_r, $max_colors));
        if $channels > 1 {
            $dst.add(1)
                .write_unaligned(compress_u16!($sum_g, $max_colors));
        }
        if $channels > 2 {
            $dst.add(2)
                .write_unaligned(compress_u16!($sum_b, $max_colors));
        }
        if $channels == 4 {
            $dst.add(3)
                .write_unaligned(compress_u16!($sum_a, $max_colors));
        }
    }};
}
pub(crate) fn convolve_horizontal_rgba_native_row_u16<const CHANNELS: usize>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u16,
    unsafe_destination_ptr_0: *mut u16,
    bit_depth: usize,
) {
    let max_colors = 2i64.pow(bit_depth as u32) - 1i64;
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let mut sum_r: i64 = ROUNDING_APPROX as i64;
            let mut sum_g: i64 = ROUNDING_APPROX as i64;
            let mut sum_b: i64 = ROUNDING_APPROX as i64;
            let mut sum_a: i64 = ROUNDING_APPROX as i64;

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight: i64 = weights_ptr.add(j + filter_offset).read_unaligned() as i64;
                let src = unsafe_source_ptr_0.add(px);
                make_naive_sum!(sum_r, sum_g, sum_b, sum_a, weight, src, CHANNELS);
            }

            let px = x * CHANNELS;

            let dest_ptr = unsafe_destination_ptr_0.add(px);

            write_out_pixels!(sum_r, sum_g, sum_b, sum_a, dest_ptr, CHANNELS, max_colors);

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
    let max_colors = 2i64.pow(bit_depth as u32) - 1i64;
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        let src_row0 = unsafe_source_ptr_0;
        let src_row1 = unsafe_source_ptr_0.add(src_stride);
        let src_row2 = unsafe_source_ptr_0.add(src_stride * 2);
        let src_row3 = unsafe_source_ptr_0.add(src_stride * 3);

        let dst_row0 = unsafe_destination_ptr_0;
        let dst_row1 = unsafe_destination_ptr_0.add(dst_stride);
        let dst_row2 = unsafe_destination_ptr_0.add(dst_stride * 2);
        let dst_row3 = unsafe_destination_ptr_0.add(dst_stride * 3);

        for x in 0..dst_width {
            let mut sum_r_0: i64 = ROUNDING_APPROX as i64;
            let mut sum_g_0: i64 = ROUNDING_APPROX as i64;
            let mut sum_b_0: i64 = ROUNDING_APPROX as i64;
            let mut sum_a_0: i64 = ROUNDING_APPROX as i64;
            let mut sum_r_1: i64 = ROUNDING_APPROX as i64;
            let mut sum_g_1: i64 = ROUNDING_APPROX as i64;
            let mut sum_b_1: i64 = ROUNDING_APPROX as i64;
            let mut sum_a_1: i64 = ROUNDING_APPROX as i64;
            let mut sum_r_2: i64 = ROUNDING_APPROX as i64;
            let mut sum_g_2: i64 = ROUNDING_APPROX as i64;
            let mut sum_b_2: i64 = ROUNDING_APPROX as i64;
            let mut sum_a_2: i64 = ROUNDING_APPROX as i64;
            let mut sum_r_3: i64 = ROUNDING_APPROX as i64;
            let mut sum_g_3: i64 = ROUNDING_APPROX as i64;
            let mut sum_b_3: i64 = ROUNDING_APPROX as i64;
            let mut sum_a_3: i64 = ROUNDING_APPROX as i64;

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight = weights_ptr.add(j + filter_offset).read_unaligned() as i64;

                let src0 = src_row0.add(px);
                make_naive_sum!(sum_r_0, sum_g_0, sum_b_0, sum_a_0, weight, src0, CHANNELS);

                let src1 = src_row1.add(px);
                make_naive_sum!(sum_r_1, sum_g_1, sum_b_1, sum_a_1, weight, src1, CHANNELS);

                let src2 = src_row2.add(px);
                make_naive_sum!(sum_r_2, sum_g_2, sum_b_2, sum_a_2, weight, src2, CHANNELS);

                let src3 = src_row3.add(px);
                make_naive_sum!(sum_r_3, sum_g_3, sum_b_3, sum_a_3, weight, src3, CHANNELS);
            }

            let px = x * CHANNELS;

            let dest_ptr_0 = dst_row0.add(px);
            let dest_ptr_1 = dst_row1.add(px);
            let dest_ptr_2 = dst_row2.add(px);
            let dest_ptr_3 = dst_row3.add(px);

            write_out_pixels!(sum_r_0, sum_g_0, sum_b_0, sum_a_0, dest_ptr_0, CHANNELS, max_colors);
            write_out_pixels!(sum_r_1, sum_g_1, sum_b_1, sum_a_1, dest_ptr_1, CHANNELS, max_colors);
            write_out_pixels!(sum_r_2, sum_g_2, sum_b_2, sum_a_2, dest_ptr_2, CHANNELS, max_colors);
            write_out_pixels!(sum_r_3, sum_g_3, sum_b_3, sum_a_3, dest_ptr_3, CHANNELS, max_colors);

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
    weight_ptr: *const i16,
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
