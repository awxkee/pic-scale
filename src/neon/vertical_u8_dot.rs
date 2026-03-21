/*
 * Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::neon::utils::{xvld1q_u8_x2, xvst1q_u8_x2};
use std::arch::aarch64::*;

static WEIGHTS_SHUFFLE_TABLE: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];

/// Checking NEON `i8mm` availability is required before a call.
///
/// RDM feature has slightly lower precision and won't work really well on huge kernel which
/// edges fades out fast. Therefore, it would be reasonable to avoid using feature for huge downscaling.
///
/// # Safety
/// - Check `i8mm` availability before the call.
pub(crate) fn convolve_vertical_neon_i8_dot(
    width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i8],
    _: u32,
) {
    unsafe {
        convolve_vertical_neon_row_upper(width, bounds, src, dst, src_stride, weight);
    }
}

#[inline(never)]
#[target_feature(enable = "neon,i8mm")]
fn convolve_32_items(
    chunks: &mut [[u8; 32]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weights: &[i8],
    cx: usize,
) -> usize {
    let mut cx = cx;

    const SCALE: i32 = 7;
    const ROUNDING: i32 = 1 << (SCALE - 1);

    let weights_shuffle = unsafe { vld1q_u8(WEIGHTS_SHUFFLE_TABLE.as_ptr()) };

    for dst in chunks {
        let mut store_0 = vdupq_n_s32(ROUNDING);
        let mut store_1 = vdupq_n_s32(ROUNDING);
        let mut store_2 = vdupq_n_s32(ROUNDING);
        let mut store_3 = vdupq_n_s32(ROUNDING);
        let mut store_4 = vdupq_n_s32(ROUNDING);
        let mut store_5 = vdupq_n_s32(ROUNDING);
        let mut store_6 = vdupq_n_s32(ROUNDING);
        let mut store_7 = vdupq_n_s32(ROUNDING);

        let px = cx;

        let mut j = 0usize;

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let weights = unsafe { weights.get_unchecked(j..) };
            let mut v_weight = vreinterpretq_s8_s32(unsafe {
                vld1q_lane_s32::<0>(weights.as_ptr().cast(), vdupq_n_s32(0))
            });
            v_weight = vqtbl1q_s8(v_weight, weights_shuffle);
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row0 = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };
            let item_row1 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let item_row2 =
                unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let item_row3 =
                unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };

            let packed0 = packq_4_rows(item_row0.0, item_row1.0, item_row2.0, item_row3.0);
            let packed1 = packq_4_rows(item_row0.1, item_row1.1, item_row2.1, item_row3.1);

            store_0 = vusdotq_s32(store_0, packed0.0, v_weight);
            store_1 = vusdotq_s32(store_1, packed0.1, v_weight);
            store_2 = vusdotq_s32(store_2, packed0.2, v_weight);
            store_3 = vusdotq_s32(store_3, packed0.3, v_weight);
            store_4 = vusdotq_s32(store_4, packed1.0, v_weight);
            store_5 = vusdotq_s32(store_5, packed1.1, v_weight);
            store_6 = vusdotq_s32(store_6, packed1.2, v_weight);
            store_7 = vusdotq_s32(store_7, packed1.3, v_weight);
            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let weights = unsafe { weights.get_unchecked(j..) };
            let mut v_weight = vreinterpretq_s8_s16(unsafe {
                vld1q_lane_s16::<0>(weights.as_ptr().cast(), vdupq_n_s16(0))
            });
            v_weight = vqtbl1q_s8(v_weight, weights_shuffle);
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row0 = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };
            let item_row1 = unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride..).as_ptr()) };

            let packed0 = packq_4_rows(item_row0.0, item_row1.0, vdupq_n_u8(0), vdupq_n_u8(0));
            let packed1 = packq_4_rows(item_row0.1, item_row1.1, vdupq_n_u8(0), vdupq_n_u8(0));

            store_0 = vusdotq_s32(store_0, packed0.0, v_weight);
            store_1 = vusdotq_s32(store_1, packed0.1, v_weight);
            store_2 = vusdotq_s32(store_2, packed0.2, v_weight);
            store_3 = vusdotq_s32(store_3, packed0.3, v_weight);
            store_4 = vusdotq_s32(store_4, packed1.0, v_weight);
            store_5 = vusdotq_s32(store_5, packed1.1, v_weight);
            store_6 = vusdotq_s32(store_6, packed1.2, v_weight);
            store_7 = vusdotq_s32(store_7, packed1.3, v_weight);
            j += 2;
        }

        for j in j..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weights.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s8(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };

            let packed0 = packq_4_rows(item_row.0, vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0));
            let packed1 = packq_4_rows(item_row.1, vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0));

            store_0 = vusdotq_s32(store_0, packed0.0, v_weight);
            store_1 = vusdotq_s32(store_1, packed0.1, v_weight);
            store_2 = vusdotq_s32(store_2, packed0.2, v_weight);
            store_3 = vusdotq_s32(store_3, packed0.3, v_weight);
            store_4 = vusdotq_s32(store_4, packed1.0, v_weight);
            store_5 = vusdotq_s32(store_5, packed1.1, v_weight);
            store_6 = vusdotq_s32(store_6, packed1.2, v_weight);
            store_7 = vusdotq_s32(store_7, packed1.3, v_weight);
        }

        let item0 = vqshrun_n_s32::<SCALE>(store_0);
        let item1 = vqshrun_n_s32::<SCALE>(store_1);
        let item2 = vqshrun_n_s32::<SCALE>(store_2);
        let item3 = vqshrun_n_s32::<SCALE>(store_3);
        let item4 = vqshrun_n_s32::<SCALE>(store_4);
        let item5 = vqshrun_n_s32::<SCALE>(store_5);
        let item6 = vqshrun_n_s32::<SCALE>(store_6);
        let item7 = vqshrun_n_s32::<SCALE>(store_7);

        let row0 = vqmovn_u16(vcombine_u16(item0, item1));
        let row1 = vqmovn_u16(vcombine_u16(item2, item3));
        let row2 = vqmovn_u16(vcombine_u16(item4, item5));
        let row3 = vqmovn_u16(vcombine_u16(item6, item7));

        unsafe {
            xvst1q_u8_x2(
                dst.as_mut_ptr(),
                uint8x16x2_t(vcombine_u8(row0, row1), vcombine_u8(row2, row3)),
            );
        }

        cx += 32;
    }

    cx
}

#[inline(never)]
#[target_feature(enable = "neon,i8mm")]
fn convolve_32_items_unrolled<const SIZE: usize, const WEIGHTS: usize>(
    chunks: &mut [[u8; 32]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weights: &[i8],
    cx: usize,
) -> usize {
    let mut cx = cx;

    const SCALE: i32 = 7;
    const ROUNDING: i32 = 1 << (SCALE - 1);

    let weights_shuffle = unsafe { vld1q_u8(WEIGHTS_SHUFFLE_TABLE.as_ptr()) };

    let cap = SIZE / 4;
    let rem = SIZE % 4;

    let weights: [int8x16_t; WEIGHTS] = std::array::from_fn(|x| {
        if x < cap {
            let v_weight = vreinterpretq_s8_s32(unsafe {
                vld1q_lane_s32::<0>(weights[x * 4..].as_ptr().cast(), vdupq_n_s32(0))
            });
            vqtbl1q_s8(v_weight, weights_shuffle)
        } else {
            let v_weight = vreinterpretq_s8_s16(unsafe {
                vld1q_lane_s16::<0>(weights[x * 4..].as_ptr().cast(), vdupq_n_s16(0))
            });
            vqtbl1q_s8(v_weight, weights_shuffle)
        }
    });

    if rem == 2 {
        for dst in chunks {
            let mut store_0 = vdupq_n_s32(ROUNDING);
            let mut store_1 = vdupq_n_s32(ROUNDING);
            let mut store_2 = vdupq_n_s32(ROUNDING);
            let mut store_3 = vdupq_n_s32(ROUNDING);
            let mut store_4 = vdupq_n_s32(ROUNDING);
            let mut store_5 = vdupq_n_s32(ROUNDING);
            let mut store_6 = vdupq_n_s32(ROUNDING);
            let mut store_7 = vdupq_n_s32(ROUNDING);

            let px = cx;

            for j in 0..WEIGHTS - 1 {
                let py = bounds.start + j * 4;
                let v_weight = weights[j];
                let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
                let item_row0 = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };
                let item_row1 =
                    unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride..).as_ptr()) };
                let item_row2 =
                    unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
                let item_row3 =
                    unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };

                let packed0 = packq_4_rows(item_row0.0, item_row1.0, item_row2.0, item_row3.0);
                let packed1 = packq_4_rows(item_row0.1, item_row1.1, item_row2.1, item_row3.1);

                store_0 = vusdotq_s32(store_0, packed0.0, v_weight);
                store_1 = vusdotq_s32(store_1, packed0.1, v_weight);
                store_2 = vusdotq_s32(store_2, packed0.2, v_weight);
                store_3 = vusdotq_s32(store_3, packed0.3, v_weight);
                store_4 = vusdotq_s32(store_4, packed1.0, v_weight);
                store_5 = vusdotq_s32(store_5, packed1.1, v_weight);
                store_6 = vusdotq_s32(store_6, packed1.2, v_weight);
                store_7 = vusdotq_s32(store_7, packed1.3, v_weight);
            }

            {
                let py = bounds.start + (WEIGHTS - 1) * 4;
                let v_weight = weights[WEIGHTS - 1];
                let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
                let item_row0 = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };
                let item_row1 =
                    unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride..).as_ptr()) };

                let packed0 = packq_4_rows(item_row0.0, item_row1.0, vdupq_n_u8(0), vdupq_n_u8(0));
                let packed1 = packq_4_rows(item_row0.1, item_row1.1, vdupq_n_u8(0), vdupq_n_u8(0));

                store_0 = vusdotq_s32(store_0, packed0.0, v_weight);
                store_1 = vusdotq_s32(store_1, packed0.1, v_weight);
                store_2 = vusdotq_s32(store_2, packed0.2, v_weight);
                store_3 = vusdotq_s32(store_3, packed0.3, v_weight);
                store_4 = vusdotq_s32(store_4, packed1.0, v_weight);
                store_5 = vusdotq_s32(store_5, packed1.1, v_weight);
                store_6 = vusdotq_s32(store_6, packed1.2, v_weight);
                store_7 = vusdotq_s32(store_7, packed1.3, v_weight);
            }

            let item0 = vqshrun_n_s32::<SCALE>(store_0);
            let item1 = vqshrun_n_s32::<SCALE>(store_1);
            let item2 = vqshrun_n_s32::<SCALE>(store_2);
            let item3 = vqshrun_n_s32::<SCALE>(store_3);
            let item4 = vqshrun_n_s32::<SCALE>(store_4);
            let item5 = vqshrun_n_s32::<SCALE>(store_5);
            let item6 = vqshrun_n_s32::<SCALE>(store_6);
            let item7 = vqshrun_n_s32::<SCALE>(store_7);

            let row0 = vqmovn_u16(vcombine_u16(item0, item1));
            let row1 = vqmovn_u16(vcombine_u16(item2, item3));
            let row2 = vqmovn_u16(vcombine_u16(item4, item5));
            let row3 = vqmovn_u16(vcombine_u16(item6, item7));

            unsafe {
                xvst1q_u8_x2(
                    dst.as_mut_ptr(),
                    uint8x16x2_t(vcombine_u8(row0, row1), vcombine_u8(row2, row3)),
                );
            }

            cx += 32;
        }
    } else {
        for dst in chunks {
            let mut store_0 = vdupq_n_s32(ROUNDING);
            let mut store_1 = vdupq_n_s32(ROUNDING);
            let mut store_2 = vdupq_n_s32(ROUNDING);
            let mut store_3 = vdupq_n_s32(ROUNDING);
            let mut store_4 = vdupq_n_s32(ROUNDING);
            let mut store_5 = vdupq_n_s32(ROUNDING);
            let mut store_6 = vdupq_n_s32(ROUNDING);
            let mut store_7 = vdupq_n_s32(ROUNDING);

            let px = cx;

            for j in 0..WEIGHTS {
                let py = bounds.start + j * 4;
                let v_weight = weights[j];
                let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
                let item_row0 = unsafe { xvld1q_u8_x2(src_ptr.as_ptr()) };
                let item_row1 =
                    unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride..).as_ptr()) };
                let item_row2 =
                    unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
                let item_row3 =
                    unsafe { xvld1q_u8_x2(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };

                let packed0 = packq_4_rows(item_row0.0, item_row1.0, item_row2.0, item_row3.0);
                let packed1 = packq_4_rows(item_row0.1, item_row1.1, item_row2.1, item_row3.1);

                store_0 = vusdotq_s32(store_0, packed0.0, v_weight);
                store_1 = vusdotq_s32(store_1, packed0.1, v_weight);
                store_2 = vusdotq_s32(store_2, packed0.2, v_weight);
                store_3 = vusdotq_s32(store_3, packed0.3, v_weight);
                store_4 = vusdotq_s32(store_4, packed1.0, v_weight);
                store_5 = vusdotq_s32(store_5, packed1.1, v_weight);
                store_6 = vusdotq_s32(store_6, packed1.2, v_weight);
                store_7 = vusdotq_s32(store_7, packed1.3, v_weight);
            }

            let item0 = vqshrun_n_s32::<SCALE>(store_0);
            let item1 = vqshrun_n_s32::<SCALE>(store_1);
            let item2 = vqshrun_n_s32::<SCALE>(store_2);
            let item3 = vqshrun_n_s32::<SCALE>(store_3);
            let item4 = vqshrun_n_s32::<SCALE>(store_4);
            let item5 = vqshrun_n_s32::<SCALE>(store_5);
            let item6 = vqshrun_n_s32::<SCALE>(store_6);
            let item7 = vqshrun_n_s32::<SCALE>(store_7);

            let row0 = vqmovn_u16(vcombine_u16(item0, item1));
            let row1 = vqmovn_u16(vcombine_u16(item2, item3));
            let row2 = vqmovn_u16(vcombine_u16(item4, item5));
            let row3 = vqmovn_u16(vcombine_u16(item6, item7));

            unsafe {
                xvst1q_u8_x2(
                    dst.as_mut_ptr(),
                    uint8x16x2_t(vcombine_u8(row0, row1), vcombine_u8(row2, row3)),
                );
            }

            cx += 32;
        }
    }

    cx
}

#[inline(never)]
#[target_feature(enable = "neon,i8mm")]
fn convolve_16_items(
    chunks: &mut [[u8; 16]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weights: &[i8],
    cx: usize,
) -> usize {
    let mut cx = cx;

    const SCALE: i32 = 7;
    const ROUNDING: i32 = 1 << (SCALE - 1);

    let weights_shuffle = unsafe { vld1q_u8(WEIGHTS_SHUFFLE_TABLE.as_ptr()) };

    for dst in chunks {
        let mut store_0 = vdupq_n_s32(ROUNDING);
        let mut store_1 = vdupq_n_s32(ROUNDING);
        let mut store_2 = vdupq_n_s32(ROUNDING);
        let mut store_3 = vdupq_n_s32(ROUNDING);

        let px = cx;

        let mut j = 0usize;

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let weights = unsafe { weights.get_unchecked(j..) };
            let mut v_weight = vreinterpretq_s8_s32(unsafe {
                vld1q_lane_s32::<0>(weights.as_ptr().cast(), vdupq_n_s32(0))
            });
            v_weight = vqtbl1q_s8(v_weight, weights_shuffle);
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row0 = unsafe { vld1q_u8(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_u8(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let item_row2 = unsafe { vld1q_u8(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let item_row3 = unsafe { vld1q_u8(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };

            let packed = packq_4_rows(item_row0, item_row1, item_row2, item_row3);

            store_0 = vusdotq_s32(store_0, packed.0, v_weight);
            store_1 = vusdotq_s32(store_1, packed.1, v_weight);
            store_2 = vusdotq_s32(store_2, packed.2, v_weight);
            store_3 = vusdotq_s32(store_3, packed.3, v_weight);
            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let weights = unsafe { weights.get_unchecked(j..) };
            let mut v_weight = vreinterpretq_s8_s16(unsafe {
                vld1q_lane_s16::<0>(weights.as_ptr().cast(), vdupq_n_s16(0))
            });
            v_weight = vqtbl1q_s8(v_weight, weights_shuffle);
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row0 = unsafe { vld1q_u8(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_u8(src_ptr.get_unchecked(src_stride..).as_ptr()) };

            let packed = packq_4_rows(item_row0, item_row1, vdupq_n_u8(0), vdupq_n_u8(0));

            store_0 = vusdotq_s32(store_0, packed.0, v_weight);
            store_1 = vusdotq_s32(store_1, packed.1, v_weight);
            store_2 = vusdotq_s32(store_2, packed.2, v_weight);
            store_3 = vusdotq_s32(store_3, packed.3, v_weight);
            j += 2;
        }

        for j in j..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weights.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s8(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1q_u8(src_ptr.as_ptr()) };

            let packed = packq_4_rows(item_row, vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0));

            store_0 = vusdotq_s32(store_0, packed.0, v_weight);
            store_1 = vusdotq_s32(store_1, packed.1, v_weight);
            store_2 = vusdotq_s32(store_2, packed.2, v_weight);
            store_3 = vusdotq_s32(store_3, packed.3, v_weight);
        }

        let item0 = vqshrun_n_s32::<SCALE>(store_0);
        let item1 = vqshrun_n_s32::<SCALE>(store_1);
        let item2 = vqshrun_n_s32::<SCALE>(store_2);
        let item3 = vqshrun_n_s32::<SCALE>(store_3);

        let row0 = vqmovn_u16(vcombine_u16(item0, item1));
        let row1 = vqmovn_u16(vcombine_u16(item2, item3));
        unsafe {
            vst1q_u8(dst.as_mut_ptr(), vcombine_u8(row0, row1));
        }

        cx += 16;
    }

    cx
}

#[inline]
#[target_feature(enable = "neon")]
fn packq_u8_as_u16(a: uint8x16_t, b: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    let r0 = vreinterpretq_u16_u8(a);
    let r1 = vreinterpretq_u16_u8(b);

    let q0 = vzip1q_u16(r0, r1);
    let q1 = vzip2q_u16(r0, r1);

    (vreinterpretq_u8_u16(q0), vreinterpretq_u8_u16(q1))
}

#[inline]
#[target_feature(enable = "neon")]
fn pack_u8_as_u16(a: uint8x8_t, b: uint8x8_t) -> (uint8x8_t, uint8x8_t) {
    let r0 = vreinterpret_u16_u8(a);
    let r1 = vreinterpret_u16_u8(b);

    let q0 = vzip1_u16(r0, r1);
    let q1 = vzip2_u16(r0, r1);

    (vreinterpret_u8_u16(q0), vreinterpret_u8_u16(q1))
}

#[inline]
#[target_feature(enable = "neon")]
fn pack_4_rows(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t, d: uint8x8_t) -> (uint8x16_t, uint8x16_t) {
    let ab0 = vzip1_u8(a, b);
    let ab1 = vzip1_u8(c, d);

    let ab2 = vzip2_u8(a, b);
    let ab3 = vzip2_u8(c, d);

    let packed0 = pack_u8_as_u16(ab0, ab1);
    let packed1 = pack_u8_as_u16(ab2, ab3);
    let q0 = vcombine_u8(packed0.0, packed0.1);
    let q1 = vcombine_u8(packed1.0, packed1.1);
    (q0, q1)
}

#[inline]
#[target_feature(enable = "neon")]
fn packq_4_rows(
    a: uint8x16_t,
    b: uint8x16_t,
    c: uint8x16_t,
    d: uint8x16_t,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    let ab0 = vzip1q_u8(a, b);
    let ab1 = vzip1q_u8(c, d);

    let ab2 = vzip2q_u8(a, b);
    let ab3 = vzip2q_u8(c, d);

    let packed0 = packq_u8_as_u16(ab0, ab1);
    let packed1 = packq_u8_as_u16(ab2, ab3);
    (packed0.0, packed0.1, packed1.0, packed1.1)
}

#[inline(never)]
#[target_feature(enable = "neon,i8mm")]
fn convolve_8_items(
    chunks: &mut [[u8; 8]],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weight: &[i8],
    cx: usize,
) -> usize {
    let mut cx = cx;

    const SCALE: i32 = 7;
    const ROUNDING: i32 = 1 << (SCALE - 1);

    let weights_shuffle = unsafe { vld1q_u8(WEIGHTS_SHUFFLE_TABLE.as_ptr()) };

    for dst in chunks {
        let mut store_0 = vdupq_n_s32(ROUNDING);
        let mut store_1 = vdupq_n_s32(ROUNDING);

        let px = cx;

        let mut j = 0usize;

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weight.get_unchecked(j..) };
            let mut v_weight = vreinterpretq_s8_s32(unsafe {
                vld1q_lane_s32::<0>(weight.as_ptr().cast(), vdupq_n_s32(0))
            });
            v_weight = vqtbl1q_s8(v_weight, weights_shuffle);
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row0 = unsafe { vld1_u8(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1_u8(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let item_row2 = unsafe { vld1_u8(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let item_row3 = unsafe { vld1_u8(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };

            let packed = pack_4_rows(item_row0, item_row1, item_row2, item_row3);

            store_0 = vusdotq_s32(store_0, packed.0, v_weight);
            store_1 = vusdotq_s32(store_1, packed.1, v_weight);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weight.get_unchecked(j..) };
            let mut v_weight = vreinterpretq_s8_s16(unsafe {
                vld1q_lane_s16::<0>(weight.as_ptr().cast(), vdupq_n_s16(0))
            });
            v_weight = vqtbl1q_s8(v_weight, weights_shuffle);
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row0 = unsafe { vld1_u8(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1_u8(src_ptr.get_unchecked(src_stride..).as_ptr()) };

            let packed = pack_4_rows(item_row0, item_row1, vdup_n_u8(0), vdup_n_u8(0));

            store_0 = vusdotq_s32(store_0, packed.0, v_weight);
            store_1 = vusdotq_s32(store_1, packed.1, v_weight);

            j += 2;
        }

        for j in j..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { weight.get_unchecked(j..) };
            let v_weight = unsafe { vld1q_dup_s8(weight.as_ptr()) };
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + px)..) };
            let item_row = unsafe { vld1_u8(src_ptr.as_ptr()) };

            let packed = pack_4_rows(item_row, vdup_n_u8(0), vdup_n_u8(0), vdup_n_u8(0));

            store_0 = vusdotq_s32(store_0, packed.0, v_weight);
            store_1 = vusdotq_s32(store_1, packed.1, v_weight);
        }

        let item0 = vqshrun_n_s32::<SCALE>(store_0);
        let item1 = vqshrun_n_s32::<SCALE>(store_1);
        let row = vqmovn_u16(vcombine_u16(item0, item1));
        unsafe {
            vst1_u8(dst.as_mut_ptr(), row);
        }

        cx += 8;
    }

    cx
}

#[inline(never)]
#[target_feature(enable = "i8mm")]
fn convolve_items(
    chunks: &mut [u8],
    bounds: &FilterBounds,
    src: &[u8],
    src_stride: usize,
    weight: &[i8],
    cx: usize,
) {
    let mut cx = cx;

    const SCALE: i32 = 7;
    const ROUNDING: i32 = 1 << (SCALE - 1);

    #[allow(clippy::explicit_counter_loop)]
    for dst in chunks {
        let vld = ROUNDING;
        let mut store = vld;

        let px = cx;

        for j in 0..bounds.size {
            let py = bounds.start + j;
            let weight = unsafe { *weight.get_unchecked(j) };
            let src = unsafe { *src.get_unchecked(src_stride * py + px) };
            store += src as i32 * weight as i32;
        }

        *dst = (store >> SCALE).max(0).min(255) as u8;
        cx += 1;
    }
}

#[target_feature(enable = "neon,i8mm")]
fn convolve_vertical_neon_row_upper(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i8],
) {
    let mut cx = 0usize;

    let size = bounds.size;
    // if size == 10 {
    //     cx = convolve_32_items_unrolled::<10, 3>(
    //         dst.as_chunks_mut::<32>().0,
    //         bounds,
    //         src,
    //         src_stride,
    //         weight,
    //         cx,
    //     );
    // } else if size == 8 {
    //     cx = convolve_32_items_unrolled::<8, 2>(
    //         dst.as_chunks_mut::<32>().0,
    //         bounds,
    //         src,
    //         src_stride,
    //         weight,
    //         cx,
    //     );
    // } else if size == 6 {
    //     cx = convolve_32_items_unrolled::<6, 2>(
    //         dst.as_chunks_mut::<32>().0,
    //         bounds,
    //         src,
    //         src_stride,
    //         weight,
    //         cx,
    //     );
    // } else if size == 4 {
    //     cx = convolve_32_items_unrolled::<4, 1>(
    //         dst.as_chunks_mut::<32>().0,
    //         bounds,
    //         src,
    //         src_stride,
    //         weight,
    //         cx,
    //     );
    // } else if size == 2 {
    //     cx = convolve_32_items_unrolled::<2, 1>(
    //         dst.as_chunks_mut::<32>().0,
    //         bounds,
    //         src,
    //         src_stride,
    //         weight,
    //         cx,
    //     );
    // } else {
        cx = convolve_32_items(
            dst.as_chunks_mut::<32>().0,
            bounds,
            src,
            src_stride,
            weight,
            cx,
        );
    // }

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
