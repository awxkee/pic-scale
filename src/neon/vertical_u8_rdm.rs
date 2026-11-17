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
use crate::convolution::{ColumnFilter, ColumnSource};
use crate::filter_weights::FilterBounds;
use crate::neon::utils::{expand8_high_to_14, expand8_to_14, xvld1q_u8_x2, xvst1q_u8_x2};
use crate::plan::VerticalFiltering;
use crate::{ImageStore, ImageStoreMut};
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

#[inline(always)]
fn expand_paired_input(row: uint8x16_t) -> (int16x8_t, int16x8_t) {
    unsafe {
        (
            vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_u8(vzip1q_u8(row, row)))),
            expand8_high_to_14(row),
        )
    }
}

/// # Safety
/// RDM must be available. `STEP` must not exceed `TAPS`, and each weight slice
/// must contain at least `TAPS` entries. The output rows must have the same
/// length, divisible by 16. Starting at a valid `start * src_stride` offset,
/// `src` must contain that many bytes in each of the `TAPS + STEP` rows.
#[target_feature(enable = "rdm")]
unsafe fn convolve_row_pair<const TAPS: usize, const STEP: usize>(
    src: &[u8],
    start: usize,
    src_stride: usize,
    first: &mut [u8],
    second: &mut [u8],
    first_weights: &[i16],
    second_weights: &[i16],
) {
    let base = unsafe { src.as_ptr().add(start * src_stride) };
    let first_weights: [_; TAPS] = std::array::from_fn(|i| vdupq_n_s16(first_weights[i]));
    let second_weights: [_; TAPS] = std::array::from_fn(|i| vdupq_n_s16(second_weights[i]));
    let rounding = vdupq_n_s16(1 << 5);

    for x in (0..first.len()).step_by(16) {
        let mut first_low = rounding;
        let mut first_high = rounding;
        let mut second_low = rounding;
        let mut second_high = rounding;
        unsafe {
            let source = base.add(x);
            for (j, &weight) in first_weights[..STEP].iter().enumerate() {
                let (lo, hi) = expand_paired_input(vld1q_u8(source.add(j * src_stride)));
                first_low = vqrdmlahq_s16(first_low, lo, weight);
                first_high = vqrdmlahq_s16(first_high, hi, weight);
            }
            // The shared rows feed both accumulators in their original tap order.
            for j in STEP..TAPS {
                let (lo, hi) = expand_paired_input(vld1q_u8(source.add(j * src_stride)));
                first_low = vqrdmlahq_s16(first_low, lo, first_weights[j]);
                first_high = vqrdmlahq_s16(first_high, hi, first_weights[j]);
                second_low = vqrdmlahq_s16(second_low, lo, second_weights[j - STEP]);
                second_high = vqrdmlahq_s16(second_high, hi, second_weights[j - STEP]);
            }
            for j in TAPS..TAPS + STEP {
                let (lo, hi) = expand_paired_input(vld1q_u8(source.add(j * src_stride)));
                second_low = vqrdmlahq_s16(second_low, lo, second_weights[j - STEP]);
                second_high = vqrdmlahq_s16(second_high, hi, second_weights[j - STEP]);
            }

            vst1q_u8(
                first.as_mut_ptr().add(x),
                vqshrun_high_n_s16::<6>(vqshrun_n_s16::<6>(first_low), first_high),
            );
            vst1q_u8(
                second.as_mut_ptr().add(x),
                vqshrun_high_n_s16::<6>(vqshrun_n_s16::<6>(second_low), second_high),
            );
        }
    }
}

pub(crate) struct PairedRdmVertical<const N: usize>(pub(crate) VerticalFiltering<u8, i16, N>);

type RdmRowPair = unsafe fn(&[u8], usize, usize, &mut [u8], &mut [u8], &[i16], &[i16]);

impl<const N: usize> ColumnFilter<u8, N> for PairedRdmVertical<N> {
    fn filter(&self, source: &ImageStore<'_, u8, N>, destination: &mut ImageStoreMut<'_, u8, N>) {
        self.0.filter(source, destination);
    }

    fn source_bounds(&self, row: usize) -> FilterBounds {
        self.0.source_bounds(row)
    }

    fn run_on_row(
        &self,
        src: &[u8],
        dst: &mut [u8],
        dst_width: usize,
        src_stride: usize,
        row: usize,
        bit_depth: u32,
    ) {
        self.0
            .run_on_row(src, dst, dst_width, src_stride, row, bit_depth);
    }

    fn run_on_4_rows(
        &self,
        src: &[u8],
        dst: &mut [u8],
        dst_width: usize,
        src_stride: usize,
        row: usize,
        bit_depth: u32,
    ) {
        let source = ColumnSource {
            pixels: src,
            width: dst_width,
            stride: src_stride,
            first_row: 0,
        };
        self.run_on_band_rows(&source, dst, row, bit_depth);
    }

    fn run_on_band_row(
        &self,
        source: &ColumnSource<'_, u8>,
        dst: &mut [u8],
        row: usize,
        bit_depth: u32,
    ) {
        self.0.run_on_band_row(source, dst, row, bit_depth);
    }

    fn run_on_band_rows(
        &self,
        source: &ColumnSource<'_, u8>,
        dst: &mut [u8],
        row: usize,
        bit_depth: u32,
    ) {
        let row_len = source.width * N;
        let filter = &self.0.filter_weights;
        for (index, pair) in dst.chunks_exact_mut(2 * row_len).enumerate() {
            let y = row + index * 2;
            let (first, second) = pair.split_at_mut(row_len);
            let a = filter.bounds[y];
            let b = filter.bounds[y + 1];
            let displacement = b.start.checked_sub(a.start);
            let kernel: Option<RdmRowPair> = if a.size == b.size && row_len.is_multiple_of(16) {
                match (a.size, displacement) {
                    (2, Some(0)) => Some(convolve_row_pair::<2, 0>),
                    (2, Some(1)) => Some(convolve_row_pair::<2, 1>),
                    (4, Some(0)) => Some(convolve_row_pair::<4, 0>),
                    (4, Some(1)) => Some(convolve_row_pair::<4, 1>),
                    (4, Some(2)) => Some(convolve_row_pair::<4, 2>),
                    (6, Some(0)) => Some(convolve_row_pair::<6, 0>),
                    (6, Some(1)) => Some(convolve_row_pair::<6, 1>),
                    (8, Some(2)) => Some(convolve_row_pair::<8, 2>),
                    (12, Some(2)) => Some(convolve_row_pair::<12, 2>),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(kernel) = kernel {
                // SAFETY: The factory checks RDM. The selected tap count and
                // displacement fit the weights and the validated source or band.
                // Both output rows are `row_len` bytes, a multiple of 16.
                unsafe {
                    kernel(
                        source.pixels,
                        a.start - source.first_row,
                        source.stride,
                        first,
                        second,
                        &filter.weights[y * filter.aligned_size..],
                        &filter.weights[(y + 1) * filter.aligned_size..],
                    );
                }
            } else {
                self.0.run_on_band_row(source, first, y, bit_depth);
                self.0.run_on_band_row(source, second, y + 1, bit_depth);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_weights::FilterWeights;

    #[test]
    fn paired_rdm_rows_match_individual_rows() {
        if !std::arch::is_aarch64_feature_detected!("rdm") {
            return;
        }
        for width in [8, 9, 20] {
            let row_len = width * 4;
            let src_stride = row_len + 5;
            for (taps, step) in [
                (2, 0),
                (2, 1),
                (4, 0),
                (4, 1),
                (4, 2),
                (6, 0),
                (6, 1),
                (8, 2),
                (12, 2),
                (5, 1),
                (6, 3),
            ] {
                let bounds = [1, 1 + step, 4, 4 + step].map(|start| FilterBounds::new(start, taps));
                let height = bounds[3].start + taps;
                let src: Vec<u8> = (0..(height - 1) * src_stride + row_len)
                    .map(|i| (i * 37 + i / src_stride * 53) as u8)
                    .collect();
                let weights = (0..taps * 4).map(|i| (3659 * i + 17123) as i16).collect();
                let filter = PairedRdmVertical(VerticalFiltering::<_, _, 4> {
                    filter_weights: FilterWeights::new(
                        weights,
                        taps,
                        taps,
                        4,
                        taps as i32,
                        bounds.to_vec(),
                    ),
                    filter_row: convolve_vertical_neon_i16_precision,
                    threading_policy: crate::ThreadingPolicy::Single,
                });
                let source = ColumnSource {
                    pixels: &src[src_stride..],
                    width,
                    stride: src_stride,
                    first_row: 1,
                };
                let mut expected = vec![0; row_len * 4];
                filter.0.run_on_band_rows(&source, &mut expected, 0, 8);
                let mut actual = vec![0; row_len * 4];
                filter.run_on_band_rows(&source, &mut actual, 0, 8);
                assert_eq!(actual, expected, "width={width}, taps={taps}, step={step}");
            }
        }
    }
}
