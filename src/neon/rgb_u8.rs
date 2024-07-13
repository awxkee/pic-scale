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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon_rgb {
    use crate::filter_weights::{FilterBounds, FilterWeights};
    use crate::support::{PRECISION, ROUNDING_APPROX};
    use std::arch::aarch64::*;
    use crate::neon::utils::{convolve_horizontal_parts_2_rgb, convolve_horizontal_parts_4_rgb, convolve_horizontal_parts_one_rgb, convolve_vertical_part_neon_16, convolve_vertical_part_neon_32, convolve_vertical_part_neon_8};

    pub fn convolve_horizontal_rgb_neon_rows_4(
        dst_width: usize,
        src_width: usize,
        approx_weights: &FilterWeights<i16>,
        unsafe_source_ptr_0: *const u8,
        src_stride: usize,
        unsafe_destination_ptr_0: *mut u8,
        dst_stride: usize,
    ) {
        unsafe {
            let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
            let shuffle_1 = vld1_u8(shuf_table_1.as_ptr());
            let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
            let shuffle_2 = vld1_u8(shuf_table_2.as_ptr());
            let shuffle = vcombine_u8(shuffle_1, shuffle_2);

            let mut filter_offset = 0usize;
            let weights_ptr = approx_weights.weights.as_ptr();
            const CHANNELS: usize = 3;
            let zeros = vdupq_n_s32(0i32);
            let init = vdupq_n_s32(ROUNDING_APPROX);
            for x in 0..dst_width {
                let bounds = approx_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store_0 = init;
                let mut store_1 = init;
                let mut store_2 = init;
                let mut store_3 = init;

                while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                    let bounds_start = bounds.start + jx;
                    let ptr = weights_ptr.add(jx + filter_offset);
                    let weight0 = vdup_n_s16(ptr.read_unaligned());
                    let weight1 = vdupq_n_s16(ptr.add(1).read_unaligned());
                    let weight2 = vdup_n_s16(ptr.add(2).read_unaligned());
                    let weight3 = vdupq_n_s16(ptr.add(3).read_unaligned());
                    store_0 = convolve_horizontal_parts_4_rgb(
                        bounds_start,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_0,
                        shuffle,
                    );
                    store_1 = convolve_horizontal_parts_4_rgb(
                        bounds_start,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_1,
                        shuffle,
                    );
                    store_2 = convolve_horizontal_parts_4_rgb(
                        bounds_start,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_2,
                        shuffle,
                    );
                    store_3 = convolve_horizontal_parts_4_rgb(
                        bounds_start,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_3,
                        shuffle,
                    );
                    jx += 4;
                }

                while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
                    let ptr = weights_ptr.add(jx + filter_offset);
                    let bounds_start = bounds.start + jx;
                    let weight0 = vdup_n_s16(ptr.read_unaligned());
                    let weight1 = vdupq_n_s16(ptr.add(1).read_unaligned());
                    store_0 = convolve_horizontal_parts_2_rgb(
                        bounds_start,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        store_0,
                        shuffle_1,
                    );
                    store_1 = convolve_horizontal_parts_2_rgb(
                        bounds_start,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        weight1,
                        store_1,
                        shuffle_1,
                    );
                    store_2 = convolve_horizontal_parts_2_rgb(
                        bounds_start,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        weight1,
                        store_2,
                        shuffle_1,
                    );
                    store_3 = convolve_horizontal_parts_2_rgb(
                        bounds_start,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        weight1,
                        store_3,
                        shuffle_1,
                    );
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.add(jx + filter_offset);
                    let bounds_start = bounds.start + jx;
                    let weight0 = vdup_n_s16(ptr.read_unaligned());
                    store_0 = convolve_horizontal_parts_one_rgb(
                        bounds_start,
                        unsafe_source_ptr_0,
                        weight0,
                        store_0,
                    );
                    store_1 = convolve_horizontal_parts_one_rgb(
                        bounds_start,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        store_1,
                    );
                    store_2 = convolve_horizontal_parts_one_rgb(
                        bounds_start,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        store_2,
                    );
                    store_3 = convolve_horizontal_parts_one_rgb(
                        bounds_start,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        store_3,
                    );
                    jx += 1;
                }

                let store_16 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_0, zeros));
                let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

                let px = x * CHANNELS;
                let dest_ptr = unsafe_destination_ptr_0.add(px);
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                dest_ptr.write_unaligned(bytes[0]);
                dest_ptr.add(1).write_unaligned(bytes[1]);
                dest_ptr.add(2).write_unaligned(bytes[2]);

                let store_16 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_1, zeros));
                let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

                let px = x * CHANNELS;
                let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);

                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                dest_ptr.write_unaligned(bytes[0]);
                dest_ptr.add(1).write_unaligned(bytes[1]);
                dest_ptr.add(2).write_unaligned(bytes[2]);

                let store_16 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_2, zeros));
                let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

                let px = x * CHANNELS;
                let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                dest_ptr.write_unaligned(bytes[0]);
                dest_ptr.add(1).write_unaligned(bytes[1]);
                dest_ptr.add(2).write_unaligned(bytes[2]);

                let store_16 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store_3, zeros));
                let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

                let px = x * CHANNELS;
                let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                dest_ptr.write_unaligned(bytes[0]);
                dest_ptr.add(1).write_unaligned(bytes[1]);
                dest_ptr.add(2).write_unaligned(bytes[2]);

                filter_offset += approx_weights.aligned_size;
            }
        }
    }

    pub fn convolve_horizontal_rgb_neon_row_one(
        dst_width: usize,
        src_width: usize,
        approx_weights: &FilterWeights<i16>,
        unsafe_source_ptr_0: *const u8,
        unsafe_destination_ptr_0: *mut u8,
    ) {
        unsafe {
            const CHANNELS: usize = 3;
            let mut filter_offset = 0usize;
            let zeros = vdupq_n_s32(0i32);
            let weights_ptr = approx_weights.weights.as_ptr();

            let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
            let shuffle_1 = vld1_u8(shuf_table_1.as_ptr());
            let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
            let shuffle_2 = vld1_u8(shuf_table_2.as_ptr());
            let shuffle = vcombine_u8(shuffle_1, shuffle_2);

            for x in 0..dst_width {
                let bounds = approx_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store = vdupq_n_s32(ROUNDING_APPROX);

                while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                    let bounds_start = bounds.start + jx;
                    let ptr = weights_ptr.add(jx + filter_offset);
                    let weight0 = vdup_n_s16(ptr.read_unaligned());
                    let weight1 = vdupq_n_s16(ptr.add(1).read_unaligned());
                    let weight2 = vdup_n_s16(ptr.add(2).read_unaligned());
                    let weight3 = vdupq_n_s16(ptr.add(3).read_unaligned());
                    store = convolve_horizontal_parts_4_rgb(
                        bounds_start,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store,
                        shuffle,
                    );
                    jx += 4;
                }

                while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
                    let ptr = weights_ptr.add(jx + filter_offset);
                    let bounds_start = bounds.start + jx;
                    let weight0 = vdup_n_s16(ptr.read_unaligned());
                    let weight1 = vdupq_n_s16(ptr.add(1).read_unaligned());
                    store = convolve_horizontal_parts_2_rgb(
                        bounds_start,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        store,
                        shuffle_1,
                    );
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.add(jx + filter_offset);
                    let weight0 = vdup_n_s16(ptr.read_unaligned());
                    store = convolve_horizontal_parts_one_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                    );
                    jx += 1;
                }

                let store_16 = vqshrun_n_s32::<PRECISION>(vmaxq_s32(store, zeros));
                let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

                let px = x * CHANNELS;
                let dest_ptr = unsafe_destination_ptr_0.add(px);
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                dest_ptr.write_unaligned(bytes[0]);
                dest_ptr.add(1).write_unaligned(bytes[1]);
                dest_ptr.add(2).write_unaligned(bytes[2]);

                filter_offset += approx_weights.aligned_size;
            }
        }
    }

    #[inline(always)]
    pub fn convolve_vertical_rgb_neon_row<const CHANNELS: usize>(
        width: usize,
        bounds: &FilterBounds,
        unsafe_source_ptr_0: *const u8,
        unsafe_destination_ptr_0: *mut u8,
        src_stride: usize,
        weight_ptr: *const i16,
    ) {
        let mut cx = 0usize;
        let dst_width = width * CHANNELS;
        while cx + 32 < dst_width {
            unsafe {
                convolve_vertical_part_neon_32(
                    bounds.start,
                    cx,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    weight_ptr,
                    bounds,
                );
            }

            cx += 32;
        }

        while cx + 16 < dst_width {
            unsafe {
                convolve_vertical_part_neon_16(
                    bounds.start,
                    cx,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    weight_ptr,
                    bounds,
                );
            }

            cx += 16;
        }

        while cx + 8 < dst_width {
            unsafe {
                convolve_vertical_part_neon_8::<false>(
                    bounds.start,
                    cx,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    weight_ptr,
                    bounds,
                    8,
                );
            }

            cx += 8;
        }

        let left = dst_width - cx;
        if left > 0 {
            unsafe {
                convolve_vertical_part_neon_8::<true>(
                    bounds.start,
                    cx,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    weight_ptr,
                    bounds,
                    left,
                );
            }
        }
    }
}
