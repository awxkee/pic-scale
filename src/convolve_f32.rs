/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::filter_weights::FilterBounds;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn prefer_vfmaq_f32(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    #[cfg(target_arch = "aarch64")]
    {
        return vfmaq_f32(a, b, c);
    }
    #[cfg(target_arch = "arm")]
    {
        return vmlaq_f32(a, b, c);
    }
}

#[inline(always)]
#[allow(unused)]
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
        let weight = *unsafe { filter.add(j) };
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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(unused)]
#[inline(always)]
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
        let weight = *unsafe { filter.add(j) };
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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub unsafe fn vtransposeq_f32(matrix: float32x4x4_t) -> float32x4x4_t {
    let row0 = matrix.0;
    let row1 = matrix.1;
    let row2 = matrix.2;
    let row3 = matrix.3;

    let row01 = vtrnq_f32(row0, row1);
    let row23 = vtrnq_f32(row2, row3);

    let r = float32x4x4_t(
        vcombine_f32(vget_low_f32(row01.0), vget_low_f32(row23.0)),
        vcombine_f32(vget_low_f32(row01.1), vget_low_f32(row23.1)),
        vcombine_f32(vget_high_f32(row01.0), vget_high_f32(row23.0)),
        vcombine_f32(vget_high_f32(row01.1), vget_high_f32(row23.1)),
    );
    return r;
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
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

    let rgb_pixel_3 = vld3q_f32(src_ptr);

    let rgb_pixel = float32x4x4_t(
        rgb_pixel_3.0,
        rgb_pixel_3.1,
        rgb_pixel_3.2,
        vdupq_n_f32(0f32),
    );
    let rgb_pixel = vtransposeq_f32(rgb_pixel);

    let acc = prefer_vfmaq_f32(store_0, rgb_pixel.0, weight0);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.1, weight1);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.2, weight2);
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.3, weight3);
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_4_rgba_f32(
    start_x: usize,
    src: *const f32,
    weight0: f32,
    weight1: f32,
    weight2: f32,
    weight3: f32,
    store_0: float32x4_t,
) -> float32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);

    let rgb_pixel = vld1q_f32_x4(src_ptr);

    let acc = prefer_vfmaq_f32(store_0, rgb_pixel.0, vdupq_n_f32(weight0));
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.1, vdupq_n_f32(weight1));
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.2, vdupq_n_f32(weight2));
    let acc = prefer_vfmaq_f32(acc, rgb_pixel.3, vdupq_n_f32(weight3));
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_one_rgba_f32(
    start_x: usize,
    src: *const f32,
    weight0: f32,
    store_0: float32x4_t,
) -> float32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);
    let rgb_pixel = vld1q_f32(src_ptr);
    let acc = prefer_vfmaq_f32(store_0, rgb_pixel, vdupq_n_f32(weight0));
    acc
}
