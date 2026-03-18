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
use crate::filter_weights::FilterBounds;
use crate::neon::utils::{expand8_to_14, xvld1q_u8_x2, xvst1q_u8_x2};
use std::arch::aarch64::*;

/// Checking NEON `rdm` availability is required before a call.
///
/// RDM feature has slightly lower precision and won't work really well on huge kernel which
/// edges fades out fast. Therefore, it would be reasonable to avoid using feature for huge downscaling.
///
/// # Safety
/// - Check `rdm` availability before the call.
pub(crate) fn convolve_vertical_neon_i16_precision(
    width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
    _: u32,
) {
    unsafe {
        convolve_vertical_neon_row_upper(width, bounds, src, dst, src_stride, weight);
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "rdm")]
fn vdot<const SCALE: i32>(
    store0: int16x8_t,
    store1: int16x8_t,
    row: uint8x16_t,
    weight: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let lo0 = vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_u8(vzip1q_u8(row, row))));
    let store0 = vqrdmlahq_s16(store0, lo0, weight);
    let hi0 = vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_u8(vzip2q_u8(row, row))));
    let store1 = vqrdmlahq_s16(store1, hi0, weight);
    (store0, store1)
}

#[must_use]
#[inline]
#[target_feature(enable = "rdm")]
fn vdot_lane<const SCALE: i32, const LANE: i32>(
    store0: int16x8_t,
    store1: int16x8_t,
    row: uint8x16_t,
    weight: int16x4_t,
) -> (int16x8_t, int16x8_t) {
    let lo0 = vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_u8(vzip1q_u8(row, row))));
    let store0 = vqrdmlahq_lane_s16::<LANE>(store0, lo0, weight);
    let hi0 = vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_u8(vzip2q_u8(row, row))));
    let store1 = vqrdmlahq_lane_s16::<LANE>(store1, hi0, weight);
    (store0, store1)
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn convolve_32_items_unrolled<const BOUNDS: usize>(
    chunks: &mut [[u8; 32]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weight: &[i16],
    cx: usize,
) -> usize {
    let mut cx = cx;

    const SCALE: i32 = 6;
    const R_SHR_SCALE: i32 = SCALE;
    const ROUNDING: i16 = 1 << (SCALE - 1);

    let weights: [int16x8_t; BOUNDS] = std::array::from_fn(|x| vdupq_n_s16(weight[x]));

    for dst in chunks {
        let vld = vdupq_n_s16(ROUNDING);
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;

        let px = cx;

        #[allow(clippy::needless_range_loop)]
        for j in 0..BOUNDS {
            let py = bounds.start + j;
            let v_weight = weights[j];
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let items = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };

            (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items.0, v_weight);
            (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items.1, v_weight);
        }

        let item00 = vqshrun_n_s16::<R_SHR_SCALE>(store_0);
        let item01 = vqshrun_n_s16::<R_SHR_SCALE>(store_1);
        let item10 = vqshrun_n_s16::<R_SHR_SCALE>(store_2);
        let item11 = vqshrun_n_s16::<R_SHR_SCALE>(store_3);
        let item0 = vcombine_u8(item00, item01);
        let item1 = vcombine_u8(item10, item11);

        let dst_items = uint8x16x2_t(item0, item1);
        unsafe {
            xvst1q_u8_x2(dst.as_mut_ptr(), dst_items);
        }

        cx += 32;
    }

    cx
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn convolve_32_items(
    chunks: &mut [[u8; 32]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weights: &[i16],
    cx: usize,
) -> usize {
    let mut cx = cx;

    const SCALE: i32 = 6;
    const R_SHR_SCALE: i32 = SCALE;
    const ROUNDING: i16 = 1 << (SCALE - 1);

    for dst in chunks {
        let vld = vdupq_n_s16(ROUNDING);
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;

        let px = cx;

        let mut j = 0usize;

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { vld1_s16(weights.get_unchecked(j..).as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };

            let items0 = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };
            let items1 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let items2 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let items3 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };

            (store_0, store_1) = vdot_lane::<SCALE, 0>(store_0, store_1, items0.0, weight);
            (store_2, store_3) = vdot_lane::<SCALE, 0>(store_2, store_3, items0.1, weight);

            (store_0, store_1) = vdot_lane::<SCALE, 1>(store_0, store_1, items1.0, weight);
            (store_2, store_3) = vdot_lane::<SCALE, 1>(store_2, store_3, items1.1, weight);

            (store_0, store_1) = vdot_lane::<SCALE, 2>(store_0, store_1, items2.0, weight);
            (store_2, store_3) = vdot_lane::<SCALE, 2>(store_2, store_3, items2.1, weight);

            (store_0, store_1) = vdot_lane::<SCALE, 3>(store_0, store_1, items3.0, weight);
            (store_2, store_3) = vdot_lane::<SCALE, 3>(store_2, store_3, items3.1, weight);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let weight = unsafe {
                vreinterpret_s16_u32(vld1_lane_u32::<0>(
                    weights.get_unchecked(j..).as_ptr().cast(),
                    vdup_n_u32(0),
                ))
            };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };

            let items0 = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };
            let items1 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride..).as_ptr()) };

            (store_0, store_1) = vdot_lane::<SCALE, 0>(store_0, store_1, items0.0, weight);
            (store_2, store_3) = vdot_lane::<SCALE, 0>(store_2, store_3, items0.1, weight);

            (store_0, store_1) = vdot_lane::<SCALE, 1>(store_0, store_1, items1.0, weight);
            (store_2, store_3) = vdot_lane::<SCALE, 1>(store_2, store_3, items1.1, weight);

            j += 2;
        }

        for j in j..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weights.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s16(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let items = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };

            (store_0, store_1) = vdot::<SCALE>(store_0, store_1, items.0, v_weight);
            (store_2, store_3) = vdot::<SCALE>(store_2, store_3, items.1, v_weight);
        }

        let item00 = vqshrun_n_s16::<R_SHR_SCALE>(store_0);
        let item01 = vqshrun_n_s16::<R_SHR_SCALE>(store_1);
        let item10 = vqshrun_n_s16::<R_SHR_SCALE>(store_2);
        let item11 = vqshrun_n_s16::<R_SHR_SCALE>(store_3);
        let item0 = vcombine_u8(item00, item01);
        let item1 = vcombine_u8(item10, item11);

        let dst_items = uint8x16x2_t(item0, item1);
        unsafe {
            xvst1q_u8_x2(dst.as_mut_ptr(), dst_items);
        }

        cx += 32;
    }

    cx
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn convolve_16_items(
    chunks: &mut [[u8; 16]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weights: &[i16],
    cx: usize,
) -> usize {
    let mut cx = cx;

    const SCALE: i32 = 6;
    const R_SHR_SCALE: i32 = SCALE;
    const ROUNDING: i16 = 1 << (SCALE - 1);

    for dst in chunks {
        let vld = vdupq_n_s16(ROUNDING);
        let mut store_0 = vld;
        let mut store_1 = vld;

        let px = cx;

        let mut j = 0usize;
        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { vld1_s16(weights.get_unchecked(j..).as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1q_u8(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_u8(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let item_row2 = unsafe { vld1q_u8(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let item_row3 = unsafe { vld1q_u8(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };

            (store_0, store_1) = vdot_lane::<SCALE, 0>(store_0, store_1, item_row, weight);
            (store_0, store_1) = vdot_lane::<SCALE, 1>(store_0, store_1, item_row1, weight);
            (store_0, store_1) = vdot_lane::<SCALE, 2>(store_0, store_1, item_row2, weight);
            (store_0, store_1) = vdot_lane::<SCALE, 3>(store_0, store_1, item_row3, weight);
            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let weight = unsafe {
                vreinterpret_s16_u32(vld1_lane_u32::<0>(
                    weights.get_unchecked(j..).as_ptr().cast(),
                    vdup_n_u32(0),
                ))
            };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1q_u8(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_u8(src_ptr.get_unchecked(src_stride..).as_ptr()) };

            (store_0, store_1) = vdot_lane::<SCALE, 0>(store_0, store_1, item_row, weight);
            (store_0, store_1) = vdot_lane::<SCALE, 1>(store_0, store_1, item_row1, weight);
            j += 2;
        }

        for j in j..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weights.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s16(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1q_u8(src_ptr.as_ptr()) };

            (store_0, store_1) = vdot::<SCALE>(store_0, store_1, item_row, v_weight);
        }

        let item0 = vqshrun_n_s16::<R_SHR_SCALE>(store_0);
        let item1 = vqshrun_n_s16::<R_SHR_SCALE>(store_1);

        unsafe {
            vst1q_u8(dst.as_mut_ptr(), vcombine_u8(item0, item1));
        }

        cx += 16;
    }

    cx
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn convolve_8_items(
    chunks: &mut [[u8; 8]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weight: &[i16],
    cx: usize,
) -> usize {
    let mut cx = cx;

    const SCALE: i32 = 6;
    const R_SHR_SCALE: i32 = SCALE;
    const ROUNDING: i16 = 1 << (SCALE - 1);

    for dst in chunks {
        let vld = vdupq_n_s16(ROUNDING);
        let mut store_0 = vld;

        let px = cx;

        for j in 0..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weight.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s16(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1_u8(src_ptr.as_ptr()) };

            let low = expand8_to_14(item_row);
            store_0 = vqrdmlahq_s16(store_0, low, v_weight);
        }

        let item = vqshrun_n_s16::<R_SHR_SCALE>(store_0);
        unsafe {
            vst1_u8(dst.as_mut_ptr(), item);
        }

        cx += 8;
    }

    cx
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn convolve_items(
    chunks: &mut [u8],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weight: &[i16],
    cx: usize,
) {
    let mut cx = cx;

    const SCALE: i32 = 6;
    const R_SHR_SCALE: i32 = SCALE;
    const ROUNDING: i16 = 1 << (SCALE - 1);

    #[allow(clippy::explicit_counter_loop)]
    for dst in chunks {
        let vld = vdupq_n_s16(ROUNDING);
        let mut store = vld;

        let px = cx;

        for j in 0..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weight.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s16(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1_dup_u8(src_ptr.as_ptr()) };

            let low = expand8_to_14(item_row);
            store = vqrdmlahq_s16(store, low, v_weight);
        }

        let shrank_store = vqshrun_n_s16::<R_SHR_SCALE>(store);
        let value = vget_lane_u8::<0>(shrank_store);
        *dst = value;
        cx += 1;
    }
}

#[target_feature(enable = "rdm")]
fn convolve_vertical_neon_row_upper(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    let mut cx = 0usize;

    let bounds_size = bounds.size;

    if bounds_size == 8 {
        cx = convolve_32_items_unrolled::<8>(
            dst.as_chunks_mut::<32>().0,
            bounds,
            src,
            src_stride,
            weight,
            cx,
        );
    } else if bounds_size == 6 {
        cx = convolve_32_items_unrolled::<6>(
            dst.as_chunks_mut::<32>().0,
            bounds,
            src,
            src_stride,
            weight,
            cx,
        );
    } else if bounds_size == 4 {
        cx = convolve_32_items_unrolled::<4>(
            dst.as_chunks_mut::<32>().0,
            bounds,
            src,
            src_stride,
            weight,
            cx,
        );
    } else if bounds_size == 2 {
        cx = convolve_32_items_unrolled::<2>(
            dst.as_chunks_mut::<32>().0,
            bounds,
            src,
            src_stride,
            weight,
            cx,
        );
    } else {
        cx = convolve_32_items(
            dst.as_chunks_mut::<32>().0,
            bounds,
            src,
            src_stride,
            weight,
            cx,
        );
    }

    let mut rem = dst.as_chunks_mut::<32>().1;
    cx = convolve_16_items(
        rem.as_chunks_mut::<16>().0,
        bounds,
        src,
        src_stride,
        weight,
        cx,
    );

    rem = rem.as_chunks_mut::<16>().1;
    cx = convolve_8_items(
        rem.as_chunks_mut::<8>().0,
        bounds,
        src,
        src_stride,
        weight,
        cx,
    );

    rem = rem.as_chunks_mut::<8>().1;
    convolve_items(rem, bounds, src, src_stride, weight, cx);
}
