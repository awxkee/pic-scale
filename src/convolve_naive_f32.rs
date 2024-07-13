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

use crate::filter_weights::{FilterBounds, FilterWeights};

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_f32<const PART: usize, const CHANNELS: usize>(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: *const f32,
    bounds: &FilterBounds,
) {
    let mut store: [[f32; CHANNELS]; PART] = [[0f32; CHANNELS]; PART];

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let src_ptr = src.add(src_stride * py);
        for x in 0..PART {
            let px = (start_x + x) * CHANNELS;
            let s_ptr = src_ptr.add(px);
            for c in 0..CHANNELS {
                let store_p = store.get_unchecked_mut(x);
                let store_v = store_p.get_unchecked_mut(c);
                *store_v += unsafe { s_ptr.add(c).read_unaligned() } * weight;
            }
        }
    }

    for x in 0..PART {
        let px = (start_x + x) * CHANNELS;
        let dst_ptr = dst.add(px);
        for c in 0..CHANNELS {
            let vl = *(*store.get_unchecked_mut(x)).get_unchecked_mut(c);
            dst_ptr.add(c).write_unaligned(vl);
        }
    }
}

macro_rules! make_naive_sum {
    ($sum_r:expr, $sum_g:expr, $sum_b:expr, $sum_a:expr, $weight: expr,
        $src:expr, $channels:expr) => {{
        $sum_r += $src.read_unaligned() * $weight;
        if $channels > 1 {
            $sum_g += $src.add(1).read_unaligned() * $weight;
        }
        if $channels > 2 {
            $sum_b += $src.add(2).read_unaligned() * $weight;
        }
        if $channels == 4 {
            $sum_a += $src.add(3).read_unaligned() * $weight;
        }
    }};
}

macro_rules! write_out_pixels {
    ($sum_r:expr, $sum_g:expr, $sum_b:expr, $sum_a:expr, $dst:expr, $channels:expr) => {{
        $dst.write_unaligned($sum_r);
        if $channels > 1 {
            $dst.add(1).write_unaligned($sum_g);
        }
        if $channels > 2 {
            $dst.add(2).write_unaligned($sum_b);
        }
        if $channels == 4 {
            $dst.add(3).write_unaligned($sum_a);
        }
    }};
}

#[inline(always)]
pub(crate) fn convolve_horizontal_rgb_native_row<const CHANNELS: usize>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    unsafe {
        let weights_ptr = filter_weights.weights.as_ptr();
        let mut filter_offset = 0usize;

        for x in 0..dst_width {
            let mut _sum_r = 0f32;
            let mut _sum_g = 0f32;
            let mut _sum_b = 0f32;
            let mut _sum_a = 0f32;

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight = weights_ptr.add(j + filter_offset).read_unaligned();
                let src = unsafe_source_ptr_0.add(px);
                make_naive_sum!(_sum_r, _sum_g, _sum_b, _sum_a, weight, src, CHANNELS);
            }

            let px = x * CHANNELS;

            let dest_ptr = unsafe_destination_ptr_0.add(px);
            write_out_pixels!(_sum_r, _sum_g, _sum_b, _sum_a, dest_ptr, CHANNELS);
            filter_offset += filter_weights.aligned_size;
        }
    }
}

#[allow(unused)]
pub(crate) fn convolve_horizontal_rgba_4_row_f32<const CHANNELS: usize>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f32,
    dst_stride: usize,
) {
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
            let mut sum_r_0 = 0f32;
            let mut sum_g_0 = 0f32;
            let mut sum_b_0 = 0f32;
            let mut sum_a_0 = 0f32;
            let mut sum_r_1 = 0f32;
            let mut sum_g_1 = 0f32;
            let mut sum_b_1 = 0f32;
            let mut sum_a_1 = 0f32;
            let mut sum_r_2 = 0f32;
            let mut sum_g_2 = 0f32;
            let mut sum_b_2 = 0f32;
            let mut sum_a_2 = 0f32;
            let mut sum_r_3 = 0f32;
            let mut sum_g_3 = 0f32;
            let mut sum_b_3 = 0f32;
            let mut sum_a_3 = 0f32;

            let bounds = filter_weights.bounds.get_unchecked(x);
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * CHANNELS;
                let weight = weights_ptr.add(j + filter_offset).read_unaligned();

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

            write_out_pixels!(sum_r_0, sum_g_0, sum_b_0, sum_a_0, dest_ptr_0, CHANNELS);
            write_out_pixels!(sum_r_1, sum_g_1, sum_b_1, sum_a_1, dest_ptr_1, CHANNELS);
            write_out_pixels!(sum_r_2, sum_g_2, sum_b_2, sum_a_2, dest_ptr_2, CHANNELS);
            write_out_pixels!(sum_r_3, sum_g_3, sum_b_3, sum_a_3, dest_ptr_3, CHANNELS);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
