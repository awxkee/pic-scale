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
use crate::support::PRECISION;
use crate::support::ROUNDING_APPROX;
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn consume_u16_4(
    start_x: usize,
    src: *const u16,
    weight0: int32x4_t,
    weight1: int32x4_t,
    weight2: int32x4_t,
    weight3: int32x4_t,
    store_0: int64x2_t,
    store_1: int64x2_t,
) -> (int64x2_t, int64x2_t) {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);
    let pixel_0 = vld1q_u16(src_ptr);
    let pixel_1 = vld1q_u16(src_ptr.add(8));

    let pixel_low_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(pixel_0)));
    let pixel_high_0 = vreinterpretq_s32_u32(vmovl_high_u16(pixel_0));

    let pixel_low_1 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(pixel_1)));
    let pixel_high_1 = vreinterpretq_s32_u32(vmovl_high_u16(pixel_1));

    let mut acc0 = vmlal_s32(store_0, vget_low_s32(pixel_low_0), vget_low_s32(weight0));
    let mut acc1 = vmlal_high_s32(store_1, pixel_low_0, weight0);
    acc0 = vmlal_s32(acc0, vget_low_s32(pixel_high_0), vget_low_s32(weight1));
    acc1 = vmlal_high_s32(acc1, pixel_high_0, weight1);

    acc0 = vmlal_s32(acc0, vget_low_s32(pixel_low_1), vget_low_s32(weight2));
    acc1 = vmlal_high_s32(acc1, pixel_low_1, weight2);

    acc0 = vmlal_s32(acc0, vget_low_s32(pixel_high_1), vget_low_s32(weight3));
    acc1 = vmlal_high_s32(acc1, pixel_high_1, weight3);

    (acc0, acc1)
}

#[inline(always)]
pub unsafe fn consume_u16_2(
    start_x: usize,
    src: *const u16,
    weight0: int32x4_t,
    weight1: int32x4_t,
    store_0: int64x2_t,
    store_1: int64x2_t,
) -> (int64x2_t, int64x2_t) {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);
    let pixel = vld1q_u16(src_ptr);

    let pixel_low = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(pixel)));
    let pixel_high = vreinterpretq_s32_u32(vmovl_high_u16(pixel));

    let mut acc0 = vmlal_s32(store_0, vget_low_s32(pixel_low), vget_low_s32(weight0));
    let mut acc1 = vmlal_high_s32(store_1, pixel_low, weight0);
    acc0 = vmlal_s32(acc0, vget_low_s32(pixel_high), vget_low_s32(weight1));
    acc1 = vmlal_high_s32(acc1, pixel_high, weight1);
    (acc0, acc1)
}

#[inline(always)]
pub unsafe fn consume_u16_1(
    start_x: usize,
    src: *const u16,
    weight: int32x4_t,
    store_0: int64x2_t,
    store_1: int64x2_t,
) -> (int64x2_t, int64x2_t) {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);
    let pixel = vreinterpretq_s32_u32(vmovl_u16(vld1_u16(src_ptr)));

    let acc0 = vmlal_s32(store_0, vget_low_s32(pixel), vget_low_s32(weight));
    let acc1 = vmlal_high_s32(store_1, pixel, weight);
    (acc0, acc1)
}

pub fn convolve_horizontal_rgba_neon_rows_4_u16(
    dst_width: usize,
    _: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u16,
    dst_stride: usize,
    bit_depth: usize,
) {
    let max_colors = 2i32.pow(bit_depth as u32) - 1i32;
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = approx_weights.weights.as_ptr();
        const CHANNELS: usize = 4;
        let zeros = vdupq_n_s32(0i32);
        let v_max_colors = vdupq_n_s32(max_colors);
        let init = vdupq_n_s64(ROUNDING_APPROX as i64);
        for x in 0..dst_width {
            let bounds = approx_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = init;
            let mut store_1 = init;
            let mut store_2 = init;
            let mut store_3 = init;
            let mut store_4 = init;
            let mut store_5 = init;
            let mut store_6 = init;
            let mut store_7 = init;

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vdupq_n_s32(ptr.read_unaligned() as i32);
                let weight1 = vdupq_n_s32(ptr.add(1).read_unaligned() as i32);
                let weight2 = vdupq_n_s32(ptr.add(2).read_unaligned() as i32);
                let weight3 = vdupq_n_s32(ptr.add(3).read_unaligned() as i32);
                let ptr_0 = unsafe_source_ptr_0;
                (store_0, store_1) = consume_u16_4(
                    bounds_start,
                    ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_0,
                    store_1,
                );
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                (store_2, store_3) = consume_u16_4(
                    bounds_start,
                    ptr_1,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_2,
                    store_3,
                );
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                (store_4, store_5) = consume_u16_4(
                    bounds_start,
                    ptr_2,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_4,
                    store_5,
                );
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                (store_6, store_7) = consume_u16_4(
                    bounds_start,
                    ptr_3,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_6,
                    store_7,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vdupq_n_s32(ptr.read_unaligned() as i32);
                let weight1 = vdupq_n_s32(ptr.add(1).read_unaligned() as i32);
                let ptr_0 = unsafe_source_ptr_0;
                (store_0, store_1) =
                    consume_u16_2(bounds_start, ptr_0, weight0, weight1, store_0, store_1);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                (store_2, store_3) =
                    consume_u16_2(bounds_start, ptr_1, weight0, weight1, store_2, store_3);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                (store_4, store_5) =
                    consume_u16_2(bounds_start, ptr_2, weight0, weight1, store_4, store_5);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                (store_6, store_7) =
                    consume_u16_2(bounds_start, ptr_3, weight0, weight1, store_6, store_7);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vdupq_n_s32(ptr.read_unaligned() as i32);
                let ptr_0 = unsafe_source_ptr_0;
                (store_0, store_1) = consume_u16_1(bounds_start, ptr_0, weight0, store_0, store_1);
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                (store_2, store_3) = consume_u16_1(bounds_start, ptr_1, weight0, store_2, store_3);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                (store_4, store_5) = consume_u16_1(bounds_start, ptr_2, weight0, store_4, store_5);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                (store_6, store_7) = consume_u16_1(bounds_start, ptr_3, weight0, store_6, store_7);
                jx += 1;
            }

            let px = x * CHANNELS;

            let new_store_0 = vqshrn_n_s64::<PRECISION>(store_0);
            let new_store_1 = vqshrn_n_s64::<PRECISION>(store_1);
            let new_store_2 = vqshrn_n_s64::<PRECISION>(store_2);
            let new_store_3 = vqshrn_n_s64::<PRECISION>(store_3);
            let new_store_4 = vqshrn_n_s64::<PRECISION>(store_4);
            let new_store_5 = vqshrn_n_s64::<PRECISION>(store_5);
            let new_store_6 = vqshrn_n_s64::<PRECISION>(store_6);
            let new_store_7 = vqshrn_n_s64::<PRECISION>(store_7);

            let store_u32 = vreinterpretq_u32_s32(vminq_s32(
                vmaxq_s32(vcombine_s32(new_store_0, new_store_1), zeros),
                v_max_colors,
            ));
            let store_16 = vmovn_u32(store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px);
            let dest_ptr_32 = dest_ptr;
            vst1_u16(dest_ptr_32, store_16);

            let store_u32 = vreinterpretq_u32_s32(vminq_s32(
                vmaxq_s32(vcombine_s32(new_store_2, new_store_3), zeros),
                v_max_colors,
            ));
            let store_16 = vmovn_u32(store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            let dest_ptr_32 = dest_ptr;
            vst1_u16(dest_ptr_32, store_16);

            let store_u32 = vreinterpretq_u32_s32(vminq_s32(
                vmaxq_s32(vcombine_s32(new_store_4, new_store_5), zeros),
                v_max_colors,
            ));
            let store_16 = vmovn_u32(store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            let dest_ptr_32 = dest_ptr;
            vst1_u16(dest_ptr_32, store_16);

            let store_u32 = vreinterpretq_u32_s32(vminq_s32(
                vmaxq_s32(vcombine_s32(new_store_6, new_store_7), zeros),
                v_max_colors,
            ));
            let store_16 = vmovn_u32(store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            let dest_ptr_32 = dest_ptr;
            vst1_u16(dest_ptr_32, store_16);

            filter_offset += approx_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgba_neon_row_u16(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u16,
    unsafe_destination_ptr_0: *mut u16,
    bit_depth: usize,
) {
    let max_colors = 2i32.pow(bit_depth as u32) - 1i32;
    unsafe {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;

        let weights_ptr = filter_weights.weights.as_ptr();

        let v_max_colors = vdupq_n_s32(max_colors);
        let zeros = vdupq_n_s32(0);

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store0 = vdupq_n_s64(ROUNDING_APPROX as i64);
            let mut store1 = vdupq_n_s64(ROUNDING_APPROX as i64);

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vdupq_n_s32(ptr.read_unaligned() as i32);
                let weight1 = vdupq_n_s32(ptr.add(1).read_unaligned() as i32);
                let weight2 = vdupq_n_s32(ptr.add(2).read_unaligned() as i32);
                let weight3 = vdupq_n_s32(ptr.add(3).read_unaligned() as i32);
                let bounds_start = bounds.start + jx;
                (store0, store1) = consume_u16_4(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store0,
                    store1,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vdupq_n_s32(ptr.read_unaligned() as i32);
                let weight1 = vdupq_n_s32(ptr.add(1).read_unaligned() as i32);
                let bounds_start = bounds.start + jx;
                (store0, store1) = consume_u16_2(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    store0,
                    store1,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vdupq_n_s32(ptr.read_unaligned() as i32);
                let bounds_start = bounds.start + jx;
                (store0, store1) =
                    consume_u16_1(bounds_start, unsafe_source_ptr_0, weight0, store0, store1);
                jx += 1;
            }

            let px = x * CHANNELS;

            let new_store_0 = vqshrn_n_s64::<PRECISION>(store0);
            let new_store_1 = vqshrn_n_s64::<PRECISION>(store1);

            let store_u32 = vreinterpretq_u32_s32(vminq_s32(
                vmaxq_s32(vcombine_s32(new_store_0, new_store_1), zeros),
                v_max_colors,
            ));
            let store_16 = vmovn_u32(store_u32);

            let dest_ptr = unsafe_destination_ptr_0.add(px);
            let dest_ptr_32 = dest_ptr;
            vst1_u16(dest_ptr_32, store_16);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
