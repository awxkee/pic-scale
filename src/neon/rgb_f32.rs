/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon_convolve_floats {
    use crate::filter_weights::{FilterBounds, FilterWeights};
    use crate::neon::*;
    use std::arch::aarch64::*;

    pub fn convolve_horizontal_rgba_neon_row_one(
        dst_width: usize,
        _: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        unsafe_destination_ptr_0: *mut f32,
    ) {
        unsafe {
            const CHANNELS: usize = 4;
            let mut filter_offset = 0usize;
            let weights_ptr = filter_weights.weights.as_ptr();

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store = vdupq_n_f32(0f32);

                while jx + 4 < bounds.size {
                    let ptr = weights_ptr.add(jx + filter_offset);
                    let weight0 = ptr.read_unaligned();
                    let weight1 = ptr.add(1).read_unaligned();
                    let weight2 = ptr.add(2).read_unaligned();
                    let weight3 = ptr.add(3).read_unaligned();
                    store = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store,
                    );
                    jx += 4;
                }
                while jx < bounds.size {
                    let ptr = weights_ptr.add(jx + filter_offset);
                    let weight0 = ptr.read_unaligned();
                    store = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                    );
                    jx += 1;
                }

                let px = x * CHANNELS;
                let dest_ptr = unsafe_destination_ptr_0.add(px);
                vst1q_f32(dest_ptr, store);

                filter_offset += filter_weights.aligned_size;
            }
        }
    }

    pub fn convolve_horizontal_rgba_neon_rows_4(
        dst_width: usize,
        _: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        src_stride: usize,
        unsafe_destination_ptr_0: *mut f32,
        dst_stride: usize,
    ) {
        unsafe {
            const CHANNELS: usize = 4;
            let mut filter_offset = 0usize;
            let zeros = vdupq_n_f32(0f32);
            let weights_ptr = filter_weights.weights.as_ptr();

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store_0 = zeros;
                let mut store_1 = zeros;
                let mut store_2 = zeros;
                let mut store_3 = zeros;

                while jx + 4 < bounds.size {
                    let ptr =  weights_ptr.add(jx + filter_offset);
                    let weight0 = ptr.read_unaligned();
                    let weight1 = ptr.add(1).read_unaligned();
                    let weight2 = ptr.add(2).read_unaligned();
                    let weight3 = ptr.add(3).read_unaligned();
                    store_0 = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_0,
                    );
                    store_1 = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_1,
                    );
                    store_2 = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_2,
                    );
                    store_3 = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_3,
                    );
                    jx += 4;
                }
                while jx < bounds.size {
                    let ptr = weights_ptr.add(jx + filter_offset);
                    let weight0 = ptr.read_unaligned();
                    store_0 = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0,
                        weight0,
                        store_0,
                    );
                    store_1 = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        store_1,
                    );
                    store_2 = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        store_2,
                    );
                    store_3 = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        store_3,
                    );
                    jx += 1;
                }

                let px = x * CHANNELS;
                let dest_ptr =  unsafe_destination_ptr_0.add(px);
                vst1q_f32(dest_ptr, store_0);

                let dest_ptr =  unsafe_destination_ptr_0.add(px + dst_stride);
                vst1q_f32(dest_ptr, store_1);

                let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
                vst1q_f32(dest_ptr, store_2);

                let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
                vst1q_f32(dest_ptr, store_3);

                filter_offset += filter_weights.aligned_size;
            }
        }
    }

    pub fn convolve_horizontal_rgb_neon_rows_4_f32(
        dst_width: usize,
        src_width: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        src_stride: usize,
        unsafe_destination_ptr_0: *mut f32,
        dst_stride: usize,
    ) {
       unsafe {
           const CHANNELS: usize = 3;
           let mut filter_offset = 0usize;

           let zeros =  vdupq_n_f32(0f32);

           let weights_ptr = filter_weights.weights.as_ptr();

           for x in 0..dst_width {
               let bounds = filter_weights.bounds.get_unchecked(x);
               let mut jx = 0usize;
               let mut store_0 = zeros;
               let mut store_1 = zeros;
               let mut store_2 = zeros;
               let mut store_3 = zeros;

               while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                   let bounds_start = bounds.start + jx;
                   let ptr = weights_ptr.add(jx + filter_offset);
                   let weight0 = vdupq_n_f32(ptr.read_unaligned());
                   let weight1 = vdupq_n_f32(ptr.add(1).read_unaligned());
                   let weight2 = vdupq_n_f32(ptr.add(2).read_unaligned());
                   let weight3 = vdupq_n_f32(ptr.add(3).read_unaligned());
                   store_0 = convolve_horizontal_parts_4_rgb_f32(
                       bounds_start,
                       unsafe_source_ptr_0,
                       weight0,
                       weight1,
                       weight2,
                       weight3,
                       store_0,
                   );
                   store_1 = convolve_horizontal_parts_4_rgb_f32(
                       bounds_start,
                       unsafe_source_ptr_0.add(src_stride),
                       weight0,
                       weight1,
                       weight2,
                       weight3,
                       store_1,
                   );
                   store_2 = convolve_horizontal_parts_4_rgb_f32(
                       bounds_start,
                       unsafe_source_ptr_0.add(src_stride * 2),
                       weight0,
                       weight1,
                       weight2,
                       weight3,
                       store_2,
                   );
                   store_3 = convolve_horizontal_parts_4_rgb_f32(
                       bounds_start,
                       unsafe_source_ptr_0.add(src_stride * 3),
                       weight0,
                       weight1,
                       weight2,
                       weight3,
                       store_3,
                   );
                   jx += 4;
               }

               while jx < bounds.size {
                   let ptr = weights_ptr.add(jx + filter_offset);
                   let bounds_start = bounds.start + jx;
                   let weight0 = vdupq_n_f32(ptr.read_unaligned());
                   store_0 = convolve_horizontal_parts_one_rgb_f32(
                       bounds_start,
                       unsafe_source_ptr_0,
                       weight0,
                       store_0,
                   );
                   store_1 = convolve_horizontal_parts_one_rgb_f32(
                       bounds_start,
                       unsafe_source_ptr_0.add(src_stride),
                       weight0,
                       store_1,
                   );
                   store_2 = convolve_horizontal_parts_one_rgb_f32(
                       bounds_start,
                       unsafe_source_ptr_0.add(src_stride * 2),
                       weight0,
                       store_2,
                   );
                   store_3 = convolve_horizontal_parts_one_rgb_f32(
                       bounds_start,
                       unsafe_source_ptr_0.add(src_stride * 3),
                       weight0,
                       store_3,
                   );
                   jx += 1;
               }

               let px = x * CHANNELS;
               let dest_ptr = unsafe_destination_ptr_0.add(px);
               let l1 = vgetq_lane_f32::<0>(store_0);
               let l2 = vgetq_lane_f32::<1>(store_0);
               let l3 = vgetq_lane_f32::<2>(store_0);
               dest_ptr.write_unaligned(l1);
               dest_ptr.add(1).write_unaligned(l2);
               dest_ptr.add(2).write_unaligned(l3);

               let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
               let l1 = vgetq_lane_f32::<0>(store_1);
               let l2 = vgetq_lane_f32::<1>(store_1);
               let l3 = vgetq_lane_f32::<2>(store_1);
               dest_ptr.write_unaligned(l1);
               dest_ptr.add(1).write_unaligned(l2);
               dest_ptr.add(2).write_unaligned(l3);

               let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
               let l1 = vgetq_lane_f32::<0>(store_2);
               let l2 = vgetq_lane_f32::<1>(store_2);
               let l3 = vgetq_lane_f32::<2>(store_2);
               dest_ptr.write_unaligned(l1);
               dest_ptr.add(1).write_unaligned(l2);
               dest_ptr.add(2).write_unaligned(l3);

               let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
               let l1 = vgetq_lane_f32::<0>(store_3);
               let l2 = vgetq_lane_f32::<1>(store_3);
               let l3 = vgetq_lane_f32::<2>(store_3);
               dest_ptr.write_unaligned(l1);
               dest_ptr.add(1).write_unaligned(l2);
               dest_ptr.add(2).write_unaligned(l3);

               filter_offset += filter_weights.aligned_size;
           }
       }
    }

    pub fn convolve_horizontal_rgb_neon_row_one_f32(
        dst_width: usize,
        src_width: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        unsafe_destination_ptr_0: *mut f32,
    ) {
       unsafe {
           const CHANNELS: usize = 3;
           let weights_ptr = filter_weights.weights.as_ptr();
           let mut filter_offset = 0usize;

           for x in 0..dst_width {
               let bounds = filter_weights.bounds.get_unchecked(x);
               let mut jx = 0usize;
               let mut store =  vdupq_n_f32(0f32);

               while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                   let bounds_start = bounds.start + jx;
                   let ptr = weights_ptr.add(jx + filter_offset);
                   let weight0 = vdupq_n_f32(ptr.read_unaligned());
                   let weight1 = vdupq_n_f32(ptr.add(1).read_unaligned());
                   let weight2 = vdupq_n_f32(ptr.add(2).read_unaligned());
                   let weight3 = vdupq_n_f32(ptr.add(3).read_unaligned());
                   store = convolve_horizontal_parts_4_rgb_f32(
                       bounds_start,
                       unsafe_source_ptr_0,
                       weight0,
                       weight1,
                       weight2,
                       weight3,
                       store,
                   );
                   jx += 4;
               }

               while jx < bounds.size {
                   let ptr = weights_ptr.add(jx + filter_offset);
                   let weight0 = vdupq_n_f32(ptr.read_unaligned());
                   store = convolve_horizontal_parts_one_rgb_f32(
                       bounds.start + jx,
                       unsafe_source_ptr_0,
                       weight0,
                       store,
                   );
                   jx += 1;
               }

               let px = x * CHANNELS;
               let dest_ptr = unsafe_destination_ptr_0.add(px);
               let l1 = vgetq_lane_f32::<0>(store);
               let l2 = vgetq_lane_f32::<1>(store);
               let l3 = vgetq_lane_f32::<2>(store);
               dest_ptr.write_unaligned(l1);
               dest_ptr.add(1).write_unaligned(l2);
               dest_ptr.add(2).write_unaligned(l3);

               filter_offset += filter_weights.aligned_size;
           }
       }
    }

    #[inline(always)]
    pub(crate) fn convolve_vertical_rgb_neon_row_f32<const CHANNELS: usize>(
        width: usize,
        bounds: &FilterBounds,
        unsafe_source_ptr_0: *const f32,
        unsafe_destination_ptr_0: *mut f32,
        src_stride: usize,
        weight_ptr: *const f32,
    ) {
        let mut cx = 0usize;
        let dst_width = width * CHANNELS;

        while cx + 16 < dst_width {
            unsafe {
                convolve_vertical_part_neon_16_f32(
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
                convolve_vertical_part_neon_8_f32::<false>(
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
                convolve_vertical_part_neon_8_f32::<true>(
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
