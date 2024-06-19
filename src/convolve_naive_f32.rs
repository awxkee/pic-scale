/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
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
                _sum_r += src.read_unaligned() * weight;
                if CHANNELS > 1 {
                    _sum_g += src.add(1).read_unaligned() * weight;
                }
                if CHANNELS > 2 {
                    _sum_b += src.add(2).read_unaligned() * weight;
                }
                if CHANNELS == 4 {
                    _sum_a += src.add(3).read_unaligned() * weight;
                }
            }

            let px = x * CHANNELS;

            let dest_ptr = unsafe_destination_ptr_0.add(px);
            dest_ptr.write_unaligned(_sum_r);
            if CHANNELS > 1 {
                dest_ptr.add(1).write_unaligned(_sum_g);
            }
            if CHANNELS > 2 {
                dest_ptr.add(2).write_unaligned(_sum_b);
            }
            if CHANNELS == 4 {
                dest_ptr.add(3).write_unaligned(_sum_a);
            }

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
                sum_r_0 += src0.read_unaligned() * weight;
                if CHANNELS > 1 {
                    sum_g_0 += src0.add(1).read_unaligned() * weight;
                }
                if CHANNELS > 2 {
                    sum_b_0 += src0.add(2).read_unaligned() * weight;
                }
                if CHANNELS == 4 {
                    sum_a_0 += src0.add(3).read_unaligned() * weight;
                }

                let src1 = src_row1.add(px);
                sum_r_1 += src1.read_unaligned() * weight;
                if CHANNELS > 1 {
                    sum_g_1 += src1.add(1).read_unaligned() * weight;
                }
                if CHANNELS > 2 {
                    sum_b_1 += src1.add(2).read_unaligned() * weight;
                }
                if CHANNELS == 4 {
                    sum_a_1 += src1.add(3).read_unaligned() * weight;
                }

                let src2 = src_row2.add(px);
                sum_r_2 += src2.read_unaligned() * weight;
                if CHANNELS > 1 {
                    sum_g_2 += src2.add(1).read_unaligned() * weight;
                }
                if CHANNELS > 2 {
                    sum_b_2 += src2.add(2).read_unaligned() * weight;
                }
                if CHANNELS == 4 {
                    sum_a_2 += src2.add(3).read_unaligned() * weight;
                }

                let src3 = src_row3.add(px);
                sum_r_3 += src3.read_unaligned() * weight;
                if CHANNELS > 1 {
                    sum_g_3 += src3.add(1).read_unaligned() * weight;
                }
                if CHANNELS > 2 {
                    sum_b_3 += src3.add(2).read_unaligned() * weight;
                }
                if CHANNELS == 4 {
                    sum_a_3 += src3.add(3).read_unaligned() * weight;
                }
            }

            let px = x * CHANNELS;

            let dest_ptr_0 = dst_row0.add(px);
            let dest_ptr_1 = dst_row1.add(px);
            let dest_ptr_2 = dst_row2.add(px);
            let dest_ptr_3 = dst_row3.add(px);

            dest_ptr_0.write_unaligned(sum_r_0);
            if CHANNELS > 1 {
                dest_ptr_0.add(1).write_unaligned(sum_g_0);
            }
            if CHANNELS > 2 {
                dest_ptr_0.add(2).write_unaligned(sum_b_0);
            }
            if CHANNELS == 4 {
                dest_ptr_0.add(3).write_unaligned(sum_a_0);
            }

            dest_ptr_1.write_unaligned(sum_r_1);
            if CHANNELS > 1 {
                dest_ptr_1.add(1).write_unaligned(sum_g_1);
            }
            if CHANNELS > 2 {
                dest_ptr_1.add(2).write_unaligned(sum_b_1);
            }
            if CHANNELS == 4 {
                dest_ptr_1.add(3).write_unaligned(sum_a_1);
            }

            dest_ptr_2.write_unaligned(sum_r_2);
            if CHANNELS > 1 {
                dest_ptr_2.add(1).write_unaligned(sum_g_2);
            }
            if CHANNELS > 2 {
                dest_ptr_2.add(2).write_unaligned(sum_b_2);
            }
            if CHANNELS == 4 {
                dest_ptr_2.add(3).write_unaligned(sum_a_2);
            }

            dest_ptr_3.write_unaligned(sum_r_3);
            if CHANNELS > 1 {
                dest_ptr_3.add(1).write_unaligned(sum_g_3);
            }
            if CHANNELS > 2 {
                dest_ptr_3.add(2).write_unaligned(sum_b_3);
            }
            if CHANNELS == 4 {
                dest_ptr_3.add(3).write_unaligned(sum_a_3);
            }

            filter_offset += filter_weights.aligned_size;
        }
    }
}
