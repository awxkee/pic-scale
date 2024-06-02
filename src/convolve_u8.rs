use crate::filter_weights::FilterBounds;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

#[inline(always)]
#[allow(unused)]
pub(crate) unsafe fn convolve_vertical_part<const PART: usize, const CHANNELS: usize>(
    start_y: usize,
    start_x: usize,
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    filter: *const i16,
    bounds: &FilterBounds,
) {
    let mut store: [[i32; CHANNELS]; PART] = [[0; CHANNELS]; PART];

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = *unsafe { filter.add(j) } as i32;
        let src_ptr = src.add(src_stride * py);
        for x in 0..PART {
            let px = (start_x + x) * CHANNELS;
            let s_ptr = src_ptr.add(px);
            for c in 0..CHANNELS {
                let store_p = store.get_unchecked_mut(x);
                let store_v = store_p.get_unchecked_mut(c);
                *store_v += unsafe { *s_ptr.add(c) } as i32 * weight;
            }
        }
    }

    for x in 0..PART {
        let px = (start_x + x) * CHANNELS;
        let dst_ptr = dst.add(px);
        for c in 0..CHANNELS {
            let vl = *(*store.get_unchecked_mut(x)).get_unchecked_mut(c);
            let ck = vl >> 12;
            *dst_ptr.add(c) = ck.max(0).min(255) as u8;
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_4_rgba(
    start_x: usize,
    src: *const u8,
    weight0: i16,
    weight1: i16,
    weight2: i16,
    weight3: i16,
    store_0: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);

    let rgba_pixel = vld1q_u8(src_ptr);

    let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgba_pixel));
    let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgba_pixel)));

    let acc = vmlal_high_s16(store_0, hi, vdupq_n_s16(weight3));
    let acc = vmlal_s16(acc, vget_low_s16(hi), vdup_n_s16(weight2));
    let acc = vmlal_high_s16(acc, lo, vdupq_n_s16(weight1));
    let acc = vmlal_s16(acc, vget_low_s16(lo), vdup_n_s16(weight0));
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_one_rgba(
    start_x: usize,
    src: *const u8,
    weight0: i16,
    store_0: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 4;
    let src_ptr = src.add(start_x * COMPONENTS);

    let mut transient: [u8; 8] = [0; 8];
    std::ptr::copy_nonoverlapping(src_ptr, transient.as_mut_ptr(), 4);

    let rgba_pixel = vld1_u8(transient.as_ptr());
    let lo = vreinterpretq_s16_u16(vmovl_u8(rgba_pixel));

    let acc = vmlal_s16(store_0, vget_low_s16(lo), vdup_n_s16(weight0));
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_4_rgb(
    start_x: usize,
    src: *const u8,
    weight0: i16,
    weight1: i16,
    weight2: i16,
    weight3: i16,
    store_0: int32x4_t,
    shuffle: uint8x16_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let mut rgb_pixel = vld1q_u8(src_ptr);
    rgb_pixel = vqtbl1q_u8(rgb_pixel, shuffle);
    let hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb_pixel));
    let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb_pixel)));

    let acc = vmlal_high_s16(store_0, hi, vdupq_n_s16(weight3));
    let acc = vmlal_s16(acc, vget_low_s16(hi), vdup_n_s16(weight2));
    let acc = vmlal_high_s16(acc, lo, vdupq_n_s16(weight1));
    let acc = vmlal_s16(acc, vget_low_s16(lo), vdup_n_s16(weight0));
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub(crate) unsafe fn convolve_horizontal_parts_one_rgb(
    start_x: usize,
    src: *const u8,
    weight0: i16,
    store_0: int32x4_t,
) -> int32x4_t {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let mut transient: [u8; 8] = [0; 8];
    std::ptr::copy_nonoverlapping(src_ptr, transient.as_mut_ptr(), 3);

    let rgb_pixel = vld1_u8(transient.as_ptr());
    let lo = vreinterpretq_s16_u16(vmovl_u8(rgb_pixel));

    let acc = vmlal_s16(store_0, vget_low_s16(lo), vdup_n_s16(weight0));
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
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
    let mut store_0 = vdupq_n_s32(0i32);
    let mut store_1 = vdupq_n_s32(0i32);
    let mut store_2 = vdupq_n_s32(0i32);
    let mut store_3 = vdupq_n_s32(0i32);

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = *unsafe { filter.add(j) };
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

    let low_16 = vcombine_u16(vqshrun_n_s32::<12>(store_0), vqshrun_n_s32::<12>(store_1));
    let high_16 = vcombine_u16(vqshrun_n_s32::<12>(store_2), vqshrun_n_s32::<12>(store_3));

    let item = vcombine_u8(vqmovn_u16(low_16), vqmovn_u16(high_16));

    let dst_ptr = dst.add(px);
    vst1q_u8(dst_ptr, item);
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
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
    let mut store_0 = vdupq_n_s32(0i32);
    let mut store_1 = vdupq_n_s32(0i32);

    let px = start_x;

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = *unsafe { filter.add(j) };
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

    let low_16 = vcombine_u16(vqshrun_n_s32::<12>(store_0), vqshrun_n_s32::<12>(store_1));

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
