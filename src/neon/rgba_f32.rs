/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::filter_weights::FilterWeights;
use crate::neon::{convolve_horizontal_parts_4_rgba_f32, convolve_horizontal_parts_one_rgba_f32};
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
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = ptr.read_unaligned();
                let weight1 = ptr.add(1).read_unaligned();
                let weight2 = ptr.add(2).read_unaligned();
                let weight3 = ptr.add(3).read_unaligned();
                store = convolve_horizontal_parts_4_rgba_f32(
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
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = ptr.read_unaligned();
                store = convolve_horizontal_parts_one_rgba_f32(
                    bounds_start,
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
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = ptr.read_unaligned();
                let weight1 = ptr.add(1).read_unaligned();
                let weight2 = ptr.add(2).read_unaligned();
                let weight3 = ptr.add(3).read_unaligned();
                let bounds_start = bounds.start + jx;
                store_0 = convolve_horizontal_parts_4_rgba_f32(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_4_rgba_f32(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_4_rgba_f32(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_4_rgba_f32(
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
                let weight0 = ptr.read_unaligned();
                let bounds_start = bounds.start + jx;
                store_0 = convolve_horizontal_parts_one_rgba_f32(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_one_rgba_f32(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_one_rgba_f32(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_one_rgba_f32(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 3),
                    weight0,
                    store_3,
                );
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            vst1q_f32(dest_ptr, store_0);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            vst1q_f32(dest_ptr, store_1);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            vst1q_f32(dest_ptr, store_2);

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            vst1q_f32(dest_ptr, store_3);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
