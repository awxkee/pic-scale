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

use std::arch::aarch64::*;

use crate::filter_weights::FilterBounds;
use crate::support::{PRECISION, ROUNDING_APPROX};

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_one_rgba(
    start_x: usize,
    src: *const u8,
    weight0: int16x4_t,
    store_0: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);
    let vl = u64::from_le_bytes([
        src_ptr.read_unaligned(),
        0,
        src_ptr.add(1).read_unaligned(),
        0,
        src_ptr.add(2).read_unaligned(),
        0,
        src_ptr.add(3).read_unaligned(),
        0,
    ]);
    let rgba_pixel = vcreate_u16(vl);
    let lo = vreinterpret_s16_u16(rgba_pixel);

    let acc = vmlal_s16(store_0, lo, weight0);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_one_rgb(
    start_x: usize,
    src: *const u8,
    weight0: int16x4_t,
    store_0: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);
    let vl = u64::from_le_bytes([
        src_ptr.read_unaligned(),
        0,
        src_ptr.add(1).read_unaligned(),
        0,
        src_ptr.add(2).read_unaligned(),
        0,
        0,
        0,
    ]);
    let rgb_pixel = vcreate_u16(vl);
    let lo = vreinterpret_s16_u16(rgb_pixel);
    let acc = vmlal_s16(store_0, lo, weight0);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_4_rgb(
    start_x: usize,
    src: *const u8,
    weight0: int16x4_t,
    weight1: int16x8_t,
    weight2: int16x4_t,
    weight3: int16x8_t,
    store_0: int32x4_t,
    shuffle: uint8x16_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let mut rgb_pixel = vld1q_u8(src_ptr);
    rgb_pixel = vqtbl1q_u8(rgb_pixel, shuffle);
    let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb_pixel));
    let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb_pixel)));

    let acc = vmlal_high_s16(store_0, hi, weight3);
    let acc = vmlal_s16(acc, vget_low_s16(hi), weight2);
    let acc = vmlal_high_s16(acc, lo, weight1);
    let acc = vmlal_s16(acc, vget_low_s16(lo), weight0);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_2_rgb(
    start_x: usize,
    src: *const u8,
    weight0: int16x4_t,
    weight1: int16x8_t,
    store_0: int32x4_t,
    shuffle: uint8x8_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let mut rgb_pixel = vld1_u8(src_ptr);
    rgb_pixel = vtbl1_u8(rgb_pixel, shuffle);
    let wide = vreinterpretq_s16_u16(vmovl_u8(rgb_pixel));

    let acc = vmlal_high_s16(store_0, wide, weight1);
    let acc = vmlal_s16(acc, vget_low_s16(wide), weight0);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_2_rgba(
    start_x: usize,
    src: *const u8,
    weight0: int16x4_t,
    weight1: int16x8_t,
    store_0: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);

    let rgb_pixel = vld1_u8(src_ptr);
    let wide = vreinterpretq_s16_u16(vmovl_u8(rgb_pixel));

    let acc = vmlal_high_s16(store_0, wide, weight1);
    let acc = vmlal_s16(acc, vget_low_s16(wide), weight0);
    acc
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_neon_32(
    start_y: usize,
    start_x: usize,
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    filter: *const i16,
    bounds: &FilterBounds,
) {
    let vld = vdupq_n_s32(ROUNDING_APPROX);
    let mut store_0 = vld;
    let mut store_1 = vld;
    let mut store_2 = vld;
    let mut store_3 = vld;
    let mut store_4 = vld;
    let mut store_5 = vld;
    let mut store_6 = vld;
    let mut store_7 = vld;

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let v_weight = vdupq_n_s16(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let items = vld1q_u8_x2(s_ptr);
        let item_row_0 = items.0;
        let item_row_1 = items.1;

        let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(item_row_0)));
        let high = vreinterpretq_s16_u16(vmovl_high_u8(item_row_0));

        store_0 = vmlal_s16(store_0, vget_low_s16(low), vget_low_s16(v_weight));
        store_1 = vmlal_high_s16(store_1, low, v_weight);
        store_2 = vmlal_s16(store_2, vget_low_s16(high), vget_low_s16(v_weight));
        store_3 = vmlal_high_s16(store_3, high, v_weight);

        let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(item_row_1)));
        let high = vreinterpretq_s16_u16(vmovl_high_u8(item_row_1));

        store_4 = vmlal_s16(store_4, vget_low_s16(low), vget_low_s16(v_weight));
        store_5 = vmlal_high_s16(store_5, low, v_weight);
        store_6 = vmlal_s16(store_6, vget_low_s16(high), vget_low_s16(v_weight));
        store_7 = vmlal_high_s16(store_7, high, v_weight);
    }

    let zeros = vdupq_n_s32(0);

    store_0 = vmaxq_s32(store_0, zeros);
    store_1 = vmaxq_s32(store_1, zeros);
    store_2 = vmaxq_s32(store_2, zeros);
    store_3 = vmaxq_s32(store_3, zeros);
    store_4 = vmaxq_s32(store_4, zeros);
    store_5 = vmaxq_s32(store_5, zeros);
    store_6 = vmaxq_s32(store_6, zeros);
    store_7 = vmaxq_s32(store_7, zeros);

    let low_16 = vcombine_u16(
        vqshrun_n_s32::<PRECISION>(store_0),
        vqshrun_n_s32::<PRECISION>(store_1),
    );
    let high_16 = vcombine_u16(
        vqshrun_n_s32::<PRECISION>(store_2),
        vqshrun_n_s32::<PRECISION>(store_3),
    );

    let item_0 = vcombine_u8(vqmovn_u16(low_16), vqmovn_u16(high_16));

    let low_16 = vcombine_u16(
        vqshrun_n_s32::<PRECISION>(store_4),
        vqshrun_n_s32::<PRECISION>(store_5),
    );
    let high_16 = vcombine_u16(
        vqshrun_n_s32::<PRECISION>(store_6),
        vqshrun_n_s32::<PRECISION>(store_7),
    );

    let item_1 = vcombine_u8(vqmovn_u16(low_16), vqmovn_u16(high_16));

    let dst_ptr = dst.add(px);

    let dst_items = uint8x16x2_t(item_0, item_1);
    vst1q_u8_x2(dst_ptr, dst_items);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_neon_16(
    start_y: usize,
    start_x: usize,
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    filter: *const i16,
    bounds: &FilterBounds,
) {
    let vld = vdupq_n_s32(ROUNDING_APPROX);
    let mut store_0 = vld;
    let mut store_1 = vld;
    let mut store_2 = vld;
    let mut store_3 = vld;

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let v_weight = vdupq_n_s16(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = vld1q_u8(s_ptr);

        let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(item_row)));
        let high = vreinterpretq_s16_u16(vmovl_high_u8(item_row));

        store_0 = vmlal_s16(store_0, vget_low_s16(low), vget_low_s16(v_weight));
        store_1 = vmlal_high_s16(store_1, low, v_weight);
        store_2 = vmlal_s16(store_2, vget_low_s16(high), vget_low_s16(v_weight));
        store_3 = vmlal_high_s16(store_3, high, v_weight);
    }

    let zeros = vdupq_n_s32(0);

    store_0 = vmaxq_s32(store_0, zeros);
    store_1 = vmaxq_s32(store_1, zeros);
    store_2 = vmaxq_s32(store_2, zeros);
    store_3 = vmaxq_s32(store_3, zeros);

    let low_16 = vcombine_u16(
        vqshrun_n_s32::<PRECISION>(store_0),
        vqshrun_n_s32::<PRECISION>(store_1),
    );
    let high_16 = vcombine_u16(
        vqshrun_n_s32::<PRECISION>(store_2),
        vqshrun_n_s32::<PRECISION>(store_3),
    );

    let item = vcombine_u8(vqmovn_u16(low_16), vqmovn_u16(high_16));

    let dst_ptr = dst.add(px);
    vst1q_u8(dst_ptr, item);
}

#[inline(always)]
pub(crate) unsafe fn convolve_vertical_part_neon_8<const USE_BLENDING: bool>(
    start_y: usize,
    start_x: usize,
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    filter: *const i16,
    bounds: &FilterBounds,
    blend_length: usize,
) {
    let vld = vdupq_n_s32(ROUNDING_APPROX);
    let mut store_0 = vld;
    let mut store_1 = vld;

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = unsafe { filter.add(j).read_unaligned() };
        let v_weight = vdupq_n_s16(weight);
        let src_ptr = src.add(src_stride * py);

        let s_ptr = src_ptr.add(px);
        let item_row = if USE_BLENDING {
            let mut transient: [u8; 8] = [0; 8];
            std::ptr::copy_nonoverlapping(s_ptr, transient.as_mut_ptr(), blend_length);
            vld1_u8(transient.as_ptr())
        } else {
            vld1_u8(s_ptr)
        };

        let low = vreinterpretq_s16_u16(vmovl_u8(item_row));
        store_0 = vmlal_s16(store_0, vget_low_s16(low), vget_low_s16(v_weight));
        store_1 = vmlal_high_s16(store_1, low, v_weight);
    }

    let zeros = vdupq_n_s32(0);

    store_0 = vmaxq_s32(store_0, zeros);
    store_1 = vmaxq_s32(store_1, zeros);

    let low_16 = vcombine_u16(
        vqshrun_n_s32::<PRECISION>(store_0),
        vqshrun_n_s32::<PRECISION>(store_1),
    );

    let item = vqmovn_u16(low_16);

    let dst_ptr = dst.add(px);
    if USE_BLENDING {
        let mut transient: [u8; 8] = [0; 8];
        vst1_u8(transient.as_mut_ptr(), item);
        std::ptr::copy_nonoverlapping(transient.as_ptr(), dst_ptr, blend_length);
    } else {
        vst1_u8(dst_ptr, item);
    }
}

#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_4_rgba(
    start_x: usize,
    src: *const u8,
    weight0: int16x4_t,
    weight1: int16x8_t,
    weight2: int16x4_t,
    weight3: int16x8_t,
    store_0: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);

    let rgba_pixel = vld1q_u8(src_ptr);

    let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel));
    let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel)));

    let acc = vmlal_high_s16(store_0, hi, weight3);
    let acc = vmlal_s16(acc, vget_low_s16(hi), weight2);
    let acc = vmlal_high_s16(acc, lo, weight1);
    let acc = vmlal_s16(acc, vget_low_s16(lo), weight0);
    acc
}

#[inline(always)]
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
pub(crate) unsafe fn vsplit_rgb_5(px: float32x4x4_t) -> Float32x5T {
    let first_pixel = vsetq_lane_f32::<3>(0f32, px.0);
    let second_pixel = vsetq_lane_f32::<3>(0f32, vextq_f32::<3>(px.0, px.1));
    let third_pixel = vsetq_lane_f32::<3>(0f32, vextq_f32::<2>(px.1, px.2));
    let four_pixel = vsetq_lane_f32::<3>(0f32, vextq_f32::<1>(px.2, px.3));
    Float32x5T(
        first_pixel,
        second_pixel,
        third_pixel,
        four_pixel,
        vsetq_lane_f32::<3>(0f32, px.3),
    )
}

pub(crate) struct Float32x5T(
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
);
