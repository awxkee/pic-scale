/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::filter_weights::FilterBounds;
use crate::neon::utils::prefer_vfmaq_f32;
use std::arch::aarch64::*;

#[allow(unused)]
pub(crate) unsafe fn convolve_vertical_part_neon_16_f32(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: *const f32,
    bounds: &FilterBounds,
) {
    let mut store_0 = vdupq_n_f32(0f32);
    let mut store_1 = vdupq_n_f32(0f32);
    let mut store_2 = vdupq_n_f32(0f32);
    let mut store_3 = vdupq_n_f32(0f32);

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let v_weight = vdupq_n_f32(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = vld1q_f32_x4(s_ptr);

        store_0 = prefer_vfmaq_f32(store_0, item_row.0, v_weight);
        store_1 = prefer_vfmaq_f32(store_1, item_row.1, v_weight);
        store_2 = prefer_vfmaq_f32(store_2, item_row.2, v_weight);
        store_3 = prefer_vfmaq_f32(store_3, item_row.3, v_weight);
    }

    let dst_ptr = dst.add(px);
    let f_set = float32x4x4_t(store_0, store_1, store_2, store_3);
    vst1q_f32_x4(dst_ptr, f_set);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_neon_8_f32<const USE_BLENDING: bool>(
    start_y: usize,
    start_x: usize,
    src: *const f32,
    src_stride: usize,
    dst: *mut f32,
    filter: *const f32,
    bounds: &FilterBounds,
    blend_length: usize,
) {
    let mut store_0 = vdupq_n_f32(0f32);
    let mut store_1 = vdupq_n_f32(0f32);

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let v_weight = vdupq_n_f32(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = if USE_BLENDING {
            let mut transient: [f32; 8] = [0f32; 8];
            std::ptr::copy_nonoverlapping(s_ptr, transient.as_mut_ptr(), blend_length);
            vld1q_f32_x2(transient.as_ptr())
        } else {
            vld1q_f32_x2(s_ptr)
        };

        store_0 = prefer_vfmaq_f32(store_0, item_row.0, v_weight);
        store_1 = prefer_vfmaq_f32(store_1, item_row.1, v_weight);
    }

    let item = float32x4x2_t(store_0, store_1);

    let dst_ptr = dst.add(px);
    if USE_BLENDING {
        let mut transient: [f32; 8] = [0f32; 8];
        vst1q_f32_x2(transient.as_mut_ptr(), item);
        std::ptr::copy_nonoverlapping(transient.as_ptr(), dst_ptr, blend_length);
    } else {
        vst1q_f32_x2(dst_ptr, item);
    }
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_4_rgb_f32(
    start_x: usize,
    src: *const f32,
    weight0: float32x4_t,
    weight1: float32x4_t,
    weight2: float32x4_t,
    weight3: float32x4_t,
    store_0: float32x4_t,
) -> float32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let mut rgb_pixel_0 = vld1q_f32(src_ptr);
    rgb_pixel_0 = vsetq_lane_f32::<3>(0f32, rgb_pixel_0);
    let mut rgb_pixel_1 = vld1q_f32(src_ptr.add(3));
    rgb_pixel_1 = vsetq_lane_f32::<3>(0f32, rgb_pixel_1);
    let mut rgb_pixel_2 = vld1q_f32(src_ptr.add(6));
    rgb_pixel_2 = vsetq_lane_f32::<3>(0f32, rgb_pixel_2);
    let rgb_pixel_3 = vld1q_f32(
        [
            src_ptr.add(9).read_unaligned(),
            src_ptr.add(10).read_unaligned(),
            src_ptr.add(11).read_unaligned(),
            0f32,
        ]
        .as_ptr(),
    );

    let acc = prefer_vfmaq_f32(store_0, rgb_pixel_0, weight0);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel_1, weight1);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel_2, weight2);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel_3, weight3);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_2_rgb_f32(
    start_x: usize,
    src: *const f32,
    weight0: float32x4_t,
    weight1: float32x4_t,
    store_0: float32x4_t,
) -> float32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let mut rgb_pixel_0 = vld1q_f32(src_ptr);
    rgb_pixel_0 = vsetq_lane_f32::<3>(0f32, rgb_pixel_0);
    let rgb_pixel_1 = vld1q_f32(
        [
            src_ptr.add(3).read_unaligned(),
            src_ptr.add(4).read_unaligned(),
            src_ptr.add(5).read_unaligned(),
            0f32,
        ]
        .as_ptr(),
    );

    let acc = prefer_vfmaq_f32(store_0, rgb_pixel_0, weight0);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel_1, weight1);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_one_rgb_f32(
    start_x: usize,
    src: *const f32,
    weight0: float32x4_t,
    store_0: float32x4_t,
) -> float32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let transient: [f32; 4] = [
        src_ptr.read_unaligned(),
        src_ptr.add(1).read_unaligned(),
        src_ptr.add(2).read_unaligned(),
        0f32,
    ];
    let rgb_pixel = vld1q_f32(transient.as_ptr());

    let acc = prefer_vfmaq_f32(store_0, rgb_pixel, weight0);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_4_rgba_f32(
    start_x: usize,
    src: *const f32,
    weight0: float32x4_t,
    weight1: float32x4_t,
    weight2: float32x4_t,
    weight3: float32x4_t,
    store_0: float32x4_t,
) -> float32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);

    let rgb_pixel = vld1q_f32_x4(src_ptr);

    let acc = prefer_vfmaq_f32(store_0, rgb_pixel.0, weight0);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.1, weight1);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.2, weight2);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.3, weight3);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_2_rgba_f32(
    start_x: usize,
    src: *const f32,
    weight0: float32x4_t,
    weight1: float32x4_t,
    store_0: float32x4_t,
) -> float32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);

    let rgb_pixel = vld1q_f32_x2(src_ptr);

    let acc = prefer_vfmaq_f32(store_0, rgb_pixel.0, weight0);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.1, weight1);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_one_rgba_f32(
    start_x: usize,
    src: *const f32,
    weight0: float32x4_t,
    store_0: float32x4_t,
) -> float32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);
    let rgb_pixel = vld1q_f32(src_ptr);
    let acc = prefer_vfmaq_f32(store_0, rgb_pixel, weight0);
    acc
}
