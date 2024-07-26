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
use crate::filter_weights::FilterWeights;
use crate::support::{PRECISION, ROUNDING_APPROX};
use std::arch::aarch64::*;

macro_rules! vfullq_sum_s32 {
    ($reg: expr) => {{
        let acc = vadd_s32(vget_low_s32($reg), vget_high_s32($reg));
        vget_lane_s32::<0>(vpadd_s32(acc, acc))
    }};
}

macro_rules! accumulate_16_horiz {
    ($store: expr, $ptr: expr, $weights: expr) => {{
        let pixel_colors = vld1q_u8($ptr);
        let px_high_16 = vreinterpretq_s16_u16(vmovl_high_u8(pixel_colors));
        let px_low_16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pixel_colors)));

        $store = vmlal_high_s16($store, px_high_16, $weights.1);
        $store = vmlal_s16($store, vget_low_s16(px_high_16), vget_low_s16($weights.1));

        $store = vmlal_high_s16($store, px_low_16, $weights.0);
        $store = vmlal_s16($store, vget_low_s16(px_low_16), vget_low_s16($weights.0));
    }};
}

macro_rules! accumulate_8_horiz {
    ($store: expr, $ptr: expr, $weights: expr) => {{
        let pixel_colors = vld1_u8($ptr);
        let px_16 = vreinterpretq_s16_u16(vmovl_u8(pixel_colors));

        $store = vmlal_high_s16($store, px_16, $weights);
        $store = vmlal_s16($store, vget_low_s16(px_16), vget_low_s16($weights));
    }};
}

macro_rules! accumulate_4_horiz {
    ($store: expr, $ptr: expr, $weights: expr) => {{
        let pixel_colors = vld1_u16(
            [
                $ptr.read_unaligned() as u16,
                $ptr.add(1).read_unaligned() as u16,
                $ptr.add(2).read_unaligned() as u16,
                $ptr.add(3).read_unaligned() as u16,
            ]
            .as_ptr(),
        );
        let px_16 = vreinterpret_s16_u16(pixel_colors);

        $store = vmlal_s16($store, px_16, $weights);
    }};
}

macro_rules! accumulate_1_horiz {
    ($store: expr, $ptr: expr, $weight: expr) => {{
        let pixel_colors = vld1_u16([$ptr.read_unaligned() as u16, 0u16, 0u16, 0u16].as_ptr());
        let px_16 = vreinterpret_s16_u16(pixel_colors);
        $store = vmlal_s16($store, px_16, $weight);
    }};
}

pub fn convolve_horizontal_plane_neon_rows_4_u8(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    unsafe {
        let mut filter_offset = 0usize;

        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store0 = vdupq_n_s32(0i32);
            store0 = vsetq_lane_s32::<0>(ROUNDING_APPROX, store0);
            let mut store1 = vdupq_n_s32(0i32);
            store1 = vsetq_lane_s32::<0>(ROUNDING_APPROX, store1);
            let mut store2 = vdupq_n_s32(0i32);
            store2 = vsetq_lane_s32::<0>(ROUNDING_APPROX, store2);
            let mut store3 = vdupq_n_s32(0i32);
            store3 = vsetq_lane_s32::<0>(ROUNDING_APPROX, store3);

            let row1 = unsafe_source_ptr_0.add(src_stride);
            let row2 = unsafe_source_ptr_0.add(src_stride * 2);
            let row3 = unsafe_source_ptr_0.add(src_stride * 3);

            while jx + 16 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1q_s16_x2(ptr);
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                accumulate_16_horiz!(store0, src_ptr, weights);

                let src_ptr1 = row1.add(bounds_start);
                accumulate_16_horiz!(store1, src_ptr1, weights);

                let src_ptr2 = row2.add(bounds_start);
                accumulate_16_horiz!(store2, src_ptr2, weights);

                let src_ptr3 = row3.add(bounds_start);
                accumulate_16_horiz!(store3, src_ptr3, weights);

                jx += 16;
            }

            while jx + 8 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1q_s16(ptr);
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                accumulate_8_horiz!(store0, src_ptr, weights);

                let src_ptr1 = row1.add(bounds_start);
                accumulate_8_horiz!(store1, src_ptr1, weights);

                let src_ptr2 = row2.add(bounds_start);
                accumulate_8_horiz!(store2, src_ptr2, weights);

                let src_ptr3 = row3.add(bounds_start);
                accumulate_8_horiz!(store3, src_ptr3, weights);

                jx += 8;
            }

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1_s16(ptr);
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                accumulate_4_horiz!(store0, src_ptr, weights);

                let src_ptr1 = row1.add(bounds_start);
                accumulate_4_horiz!(store1, src_ptr1, weights);

                let src_ptr2 = row2.add(bounds_start);
                accumulate_4_horiz!(store2, src_ptr2, weights);

                let src_ptr3 = row3.add(bounds_start);
                accumulate_4_horiz!(store3, src_ptr3, weights);

                jx += 4;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight = vld1_lane_s16::<0>(ptr, vdup_n_s16(0));
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                accumulate_1_horiz!(store0, src_ptr, weight);

                let src_ptr1 = row1.add(bounds_start);
                accumulate_1_horiz!(store1, src_ptr1, weight);

                let src_ptr2 = row2.add(bounds_start);
                accumulate_1_horiz!(store2, src_ptr2, weight);

                let src_ptr3 = row3.add(bounds_start);
                accumulate_1_horiz!(store3, src_ptr3, weight);

                jx += 1;
            }

            let sums = vfullq_sum_s32!(store0).max(0);
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;

            let px = x;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            dest_ptr.write_unaligned(value);

            let sums = vfullq_sum_s32!(store1).max(0);
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;

            let px = x;
            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            dest_ptr.write_unaligned(value);

            let sums = vfullq_sum_s32!(store2).max(0);
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;

            let px = x;
            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            dest_ptr.write_unaligned(value);

            let sums = vfullq_sum_s32!(store3).max(0);
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;

            let px = x;
            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            dest_ptr.write_unaligned(value);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_plane_neon_row(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    unsafe {
        let mut filter_offset = 0usize;

        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vdupq_n_s32(0i32);
            store = vsetq_lane_s32::<0>(ROUNDING_APPROX, store);

            while jx + 16 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1q_s16_x2(ptr);
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                accumulate_16_horiz!(store, src_ptr, weights);

                jx += 16;
            }

            while jx + 8 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1q_s16(ptr);
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                accumulate_8_horiz!(store, src_ptr, weights);

                jx += 8;
            }

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights = vld1_s16(ptr);
                let bounds_start = bounds.start + jx;

                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                accumulate_4_horiz!(store, src_ptr, weights);

                jx += 4;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight = vld1_lane_s16::<0>(ptr, vdup_n_s16(0));
                let bounds_start = bounds.start + jx;
                let src_ptr = unsafe_source_ptr_0.add(bounds_start);
                accumulate_1_horiz!(store, src_ptr, weight);
                jx += 1;
            }

            let sums = vfullq_sum_s32!(store).max(0);
            let shifted = sums >> PRECISION;
            let value = shifted.min(255) as u8;

            let px = x;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            dest_ptr.write_unaligned(value);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
