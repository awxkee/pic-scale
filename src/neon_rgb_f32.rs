/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon_convolve_floats {
    use crate::convolve_f32::{
        convolve_horizontal_parts_4_rgb_f32, convolve_horizontal_parts_4_rgba_f32,
        convolve_horizontal_parts_one_rgb_f32, convolve_horizontal_parts_one_rgba_f32,
    };
    use crate::filter_weights::FilterWeights;
    use std::arch::aarch64::*;

    pub unsafe fn convolve_horizontal_rgba_neon_row_one(
        dst_width: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        unsafe_destination_ptr_0: *mut f32,
    ) {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store = unsafe { vdupq_n_f32(0f32) };

            while jx + 4 < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                let weight2 = unsafe { ptr.add(2).read_unaligned() };
                let weight3 = unsafe { ptr.add(3).read_unaligned() };
                unsafe {
                    store = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store,
                    );
                }
                jx += 4;
            }
            while jx < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                unsafe {
                    store = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                    );
                }
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                vst1q_f32(dest_ptr, store);
            }

            filter_offset += filter_weights.aligned_size;
        }
    }

    pub unsafe fn convolve_horizontal_rgba_neon_rows_4(
        dst_width: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        src_stride: usize,
        unsafe_destination_ptr_0: *mut f32,
        dst_stride: usize,
    ) {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let zeros = unsafe { vdupq_n_f32(0f32) };
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 4 < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                let weight2 = unsafe { ptr.add(2).read_unaligned() };
                let weight3 = unsafe { ptr.add(3).read_unaligned() };
                unsafe {
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
                }
                jx += 4;
            }
            while jx < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                unsafe {
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
                }
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                vst1q_f32(dest_ptr, store_0);
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride) };
            unsafe {
                vst1q_f32(dest_ptr, store_1);
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 2) };
            unsafe {
                vst1q_f32(dest_ptr, store_2);
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 3) };
            unsafe {
                vst1q_f32(dest_ptr, store_3);
            }

            filter_offset += filter_weights.aligned_size;
        }
    }

    pub unsafe fn convolve_horizontal_rgb_neon_rows_4(
        dst_width: usize,
        src_width: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        src_stride: usize,
        unsafe_destination_ptr_0: *mut f32,
        dst_stride: usize,
    ) {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;

        let zeros = unsafe { vdupq_n_f32(0f32) };

        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
                let bounds_start = bounds.start + jx;
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                unsafe {
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
                }
                jx += 4;
            }

            while jx < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                unsafe {
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
                }
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                let l1 = vgetq_lane_f32::<0>(store_0);
                let l2 = vgetq_lane_f32::<1>(store_0);
                let l3 = vgetq_lane_f32::<2>(store_0);
                *dest_ptr = l1;
                *dest_ptr.add(1) = l2;
                *dest_ptr.add(2) = l3;
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride) };
            unsafe {
                let l1 = vgetq_lane_f32::<0>(store_1);
                let l2 = vgetq_lane_f32::<1>(store_1);
                let l3 = vgetq_lane_f32::<2>(store_1);
                *dest_ptr = l1;
                *dest_ptr.add(1) = l2;
                *dest_ptr.add(2) = l3;
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 2) };
            unsafe {
                let l1 = vgetq_lane_f32::<0>(store_2);
                let l2 = vgetq_lane_f32::<1>(store_2);
                let l3 = vgetq_lane_f32::<2>(store_2);
                *dest_ptr = l1;
                *dest_ptr.add(1) = l2;
                *dest_ptr.add(2) = l3;
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 3) };
            unsafe {
                let l1 = vgetq_lane_f32::<0>(store_3);
                let l2 = vgetq_lane_f32::<1>(store_3);
                let l3 = vgetq_lane_f32::<2>(store_3);
                *dest_ptr = l1;
                *dest_ptr.add(1) = l2;
                *dest_ptr.add(2) = l3;
            }

            filter_offset += filter_weights.aligned_size;
        }
    }

    pub unsafe fn convolve_horizontal_rgb_neon_row_one(
        dst_width: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        unsafe_destination_ptr_0: *mut f32,
    ) {
        const CHANNELS: usize = 3;
        let weights_ptr = filter_weights.weights.as_ptr();
        let mut filter_offset = 0usize;

        for x in 0..dst_width {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store = unsafe { vdupq_n_f32(0f32) };

            while jx + 4 < bounds.size && bounds.start + jx + 6 < dst_width {
                let bounds_start = bounds.start + jx;
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                unsafe {
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
                }
                jx += 4;
            }

            while jx < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                unsafe {
                    let weight0 = vdupq_n_f32(ptr.read_unaligned());
                    store = convolve_horizontal_parts_one_rgb_f32(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                    );
                }
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                let l1 = vgetq_lane_f32::<0>(store);
                let l2 = vgetq_lane_f32::<1>(store);
                let l3 = vgetq_lane_f32::<2>(store);
                *dest_ptr = l1;
                *dest_ptr.add(1) = l2;
                *dest_ptr.add(2) = l3;
            }

            filter_offset += filter_weights.aligned_size;
        }
    }
}
