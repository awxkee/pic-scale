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
use std::arch::aarch64::*;

pub(crate) fn convolve_column_hb_u16(
    w: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i32],
    bit_depth: u32,
) {
    unsafe { convolve_column_hb_impl(w, bounds, src, dst, src_stride, weight, bit_depth) }
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn convolve_16_items(
    chunks: &mut [[u16; 16]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weights: &[i32],
    bit_depth: u32,
    cx: usize,
) -> usize {
    let max_colors = (1u32 << bit_depth) - 1;
    let mut cx = cx;

    let initial_store = vdupq_n_s32(1 << 5);

    let v_max_colors = vdupq_n_u16(max_colors as u16);

    let v_px = cx;
    for (x, dst) in chunks.iter_mut().enumerate() {
        let mut store0 = initial_store;
        let mut store1 = initial_store;
        let mut store2 = initial_store;
        let mut store3 = initial_store;

        let v_dx = v_px + x * 16;

        let q_weights = weights.as_chunks::<4>().0;

        for (j, &k_weight) in q_weights.iter().enumerate() {
            let py = bounds.start + j * 4;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = unsafe { vld1q_s32(k_weight.as_ptr()) };

            let item_row0 = unsafe { vld1q_u16(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_u16(src_ptr.get_unchecked(8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<0>(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<0>(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_laneq_s32::<0>(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<0>(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );

            let item_row0 = unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride..).as_ptr()) };
            let item_row1 = unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride + 8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<1>(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<1>(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_laneq_s32::<1>(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<1>(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );

            let item_row0 = unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride * 2..).as_ptr()) };
            let item_row1 =
                unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride * 2 + 8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<2>(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<2>(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_laneq_s32::<2>(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<2>(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );

            let item_row0 = unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride * 3..).as_ptr()) };
            let item_row1 =
                unsafe { vld1q_u16(src_ptr.get_unchecked(src_stride * 3 + 8..).as_ptr()) };

            store0 = vqrdmlahq_laneq_s32::<3>(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_laneq_s32::<3>(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_laneq_s32::<3>(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_laneq_s32::<3>(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );
        }

        let base_y = weights.as_chunks::<4>().0.len() * 4 + bounds.start;
        let weights = weights.as_chunks::<4>().1;

        for (j, &k_weight) in weights.iter().enumerate() {
            let py = base_y + j;
            let src_ptr = unsafe { src.get_unchecked((src_stride * py + v_dx)..) };

            let v_weight = vdupq_n_s32(k_weight);

            let item_row0 = unsafe { vld1q_u16(src_ptr.as_ptr()) };
            let item_row1 = unsafe { vld1q_u16(src_ptr.get_unchecked(8..).as_ptr()) };

            store0 = vqrdmlahq_s32(
                store0,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row0))),
                v_weight,
            );
            store1 = vqrdmlahq_s32(
                store1,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row0)),
                v_weight,
            );
            store2 = vqrdmlahq_s32(
                store2,
                vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row1))),
                v_weight,
            );
            store3 = vqrdmlahq_s32(
                store3,
                vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row1)),
                v_weight,
            );
        }

        let store0 = vqshrun_n_s32::<6>(store0);
        let store1 = vqshrun_n_s32::<6>(store1);
        let store2 = vqshrun_n_s32::<6>(store2);
        let store3 = vqshrun_n_s32::<6>(store3);

        let item0 = vminq_u16(vcombine_u16(store0, store1), v_max_colors);
        let item1 = vminq_u16(vcombine_u16(store2, store3), v_max_colors);

        unsafe {
            vst1q_u16(dst.as_mut_ptr(), item0);
            vst1q_u16(dst.get_unchecked_mut(8..).as_mut_ptr(), item1);
        }
        cx = v_dx + 16;
    }
    cx
}

#[target_feature(enable = "rdm")]
fn convolve_column_hb_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weights: &[i32],
    bit_depth: u32,
) {
    unsafe {
        let max_colors = (1u32 << bit_depth) - 1;
        let mut cx = 0usize;

        let bounds_size = bounds.size;
        let weights = &weights[..bounds.size];

        let initial_store = vdupq_n_s32(1 << 5);

        let v_max_colors = vdupq_n_u16(max_colors as u16);

        cx = convolve_16_items(
            dst.as_chunks_mut::<16>().0,
            bounds,
            src,
            src_stride,
            weights,
            bit_depth,
            cx,
        );

        let tail16 = dst.as_chunks_mut::<16>().1;
        let iter8 = tail16.as_chunks_mut::<8>();

        let v_px = cx;

        for (x, dst) in iter8.0.iter_mut().enumerate() {
            let mut store0 = initial_store;
            let mut store1 = initial_store;

            let v_dx = v_px + x * 8;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = vdupq_n_s32(k_weight);

                let item_row = vld1q_u16(src_ptr.as_ptr());

                store0 = vqrdmlahq_s32(
                    store0,
                    vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(item_row))),
                    v_weight,
                );
                store1 = vqrdmlahq_s32(
                    store1,
                    vreinterpretq_s32_u32(vshll_high_n_u16::<6>(item_row)),
                    v_weight,
                );
            }

            let item = vminq_u16(
                vcombine_u16(vqshrun_n_s32::<6>(store0), vqshrun_n_s32::<6>(store1)),
                v_max_colors,
            );
            vst1q_u16(dst.as_mut_ptr(), item);

            cx += 8;
        }

        let tail8 = tail16.as_chunks_mut::<8>().1;
        let iter4 = tail8.as_chunks_mut::<4>();

        let v_cx = cx;

        for (x, dst) in iter4.0.iter_mut().enumerate() {
            let mut store0 = initial_store;

            let v_dx = v_cx + x * 4;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = vdupq_n_s32(k_weight);

                let item_row = vld1_u16(src_ptr.as_ptr());

                store0 = vqrdmlahq_s32(
                    store0,
                    vreinterpretq_s32_u32(vshll_n_u16::<6>(item_row)),
                    v_weight,
                );
            }

            let u_store0 = vmin_u16(vqshrun_n_s32::<6>(store0), vget_low_u16(v_max_colors));
            vst1_u16(dst.as_mut_ptr(), u_store0);

            cx += 4;
        }

        let tail4 = tail8.as_chunks_mut::<4>().1;
        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = 1 << 5;

            let v_px = cx + x;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_ptr = src.get_unchecked(offset);

                // Match the vector blocks: round each Q31 product into Q6.
                store0 = vqrdmlahs_s32(store0, i32::from(*src_ptr) << 6, k_weight);
            }

            *dst = (store0 >> 6).max(0).min(max_colors as i32) as u16;
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn accumulate_pair_output(acc: &mut [int32x4_t; 4], input: [int32x4_t; 4], weight: i32) {
    let weight = vdupq_n_s32(weight);
    acc[0] = vqrdmlahq_s32(acc[0], input[0], weight);
    acc[1] = vqrdmlahq_s32(acc[1], input[1], weight);
    acc[2] = vqrdmlahq_s32(acc[2], input[2], weight);
    acc[3] = vqrdmlahq_s32(acc[3], input[3], weight);
}

#[inline(always)]
unsafe fn load_pair_input(src: *const u16) -> [int32x4_t; 4] {
    unsafe {
        let low = vld1q_u16(src);
        let high = vld1q_u16(src.add(8));
        [
            vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(low))),
            vreinterpretq_s32_u32(vshll_high_n_u16::<6>(low)),
            vreinterpretq_s32_u32(vshll_n_u16::<6>(vget_low_u16(high))),
            vreinterpretq_s32_u32(vshll_high_n_u16::<6>(high)),
        ]
    }
}

#[inline(always)]
unsafe fn store_pair_output(dst: *mut u16, acc: [int32x4_t; 4], max: uint16x8_t) {
    unsafe {
        let low = vqshrun_high_n_s32::<6>(vqshrun_n_s32::<6>(acc[0]), acc[1]);
        let high = vqshrun_high_n_s32::<6>(vqshrun_n_s32::<6>(acc[2]), acc[3]);
        vst1q_u16(dst, vminq_u16(low, max));
        vst1q_u16(dst.add(8), vminq_u16(high, max));
    }
}

#[target_feature(enable = "rdm")]
pub(crate) fn convolve_column_hb_u16_pair(
    src: &[u16],
    stride: usize,
    first: &mut [u16],
    second: &mut [u16],
    first_weights: &[i32; super::vertical_pair::U16_PAIR_TAPS],
    second_weights: &[i32; super::vertical_pair::U16_PAIR_TAPS],
    bit_depth: u32,
) {
    use super::vertical_pair::{U16_PAIR_STEP as PAIRED_ROW_STEP, U16_PAIR_TAPS as PAIRED_TAPS};

    let max = vdupq_n_u16(((1u32 << bit_depth) - 1) as u16);
    let round = vdupq_n_s32(1 << 5);
    for x in (0..first.len()).step_by(16) {
        let mut a = [round; 4];
        let mut b = [round; 4];
        unsafe {
            let input = src.as_ptr().add(x);
            for (j, &weight) in first_weights[..PAIRED_ROW_STEP].iter().enumerate() {
                accumulate_pair_output(&mut a, load_pair_input(input.add(j * stride)), weight);
            }
            // Each shared input row feeds both sums in tap order.
            for j in PAIRED_ROW_STEP..PAIRED_TAPS {
                let pixels = load_pair_input(input.add(j * stride));
                accumulate_pair_output(&mut a, pixels, first_weights[j]);
                accumulate_pair_output(&mut b, pixels, second_weights[j - PAIRED_ROW_STEP]);
            }
            for j in PAIRED_TAPS..PAIRED_TAPS + PAIRED_ROW_STEP {
                accumulate_pair_output(
                    &mut b,
                    load_pair_input(input.add(j * stride)),
                    second_weights[j - PAIRED_ROW_STEP],
                );
            }
            store_pair_output(first.as_mut_ptr().add(x), a, max);
            store_pair_output(second.as_mut_ptr().add(x), b, max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_q6_rdm(samples: impl IntoIterator<Item = (u16, i32)>, depth: u32) -> u16 {
        let mut sum = 32i32;
        for (sample, weight) in samples {
            // Independent fused SQRDMLAH pseudocode, with a wide exact intermediate.
            let product = (i128::from(sample) << 6) * i128::from(weight);
            let fused = (i128::from(sum) << 32) + product * 2;
            let rounded = (fused + (1i128 << 31)) >> 32;
            sum = rounded.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32;
        }
        (sum >> 6).clamp(0, ((1u32 << depth) - 1) as i32) as u16
    }

    fn weights(taps: usize, mode: usize) -> Vec<i32> {
        (0..taps)
            .map(|i| match mode {
                0 => {
                    if i % 2 == 0 {
                        1073741760
                    } else {
                        1073741888
                    }
                }
                1 => {
                    if i % 2 == 0 {
                        i32::MIN
                    } else {
                        i32::MAX
                    }
                }
                2 => i32::MAX,
                3 => i32::MIN,
                _ => {
                    if i < taps.div_ceil(2) {
                        i32::MAX
                    } else {
                        i32::MIN
                    }
                }
            })
            .collect()
    }

    #[test]
    fn hb_vertical_half_boundary_matches_vector_and_scalar_tail() {
        if !std::arch::is_aarch64_feature_detected!("rdm") {
            return;
        }
        for depth in [13, 14, 15, 16] {
            for width in 1..=40 {
                let mut src = vec![1u16; width];
                src.extend(std::iter::repeat_n(0, width));
                let mut dst = vec![0u16; width];
                convolve_column_hb_u16(
                    width,
                    &FilterBounds { start: 0, size: 2 },
                    &src,
                    &mut dst,
                    width,
                    &[1073741760, 1073741888],
                    depth,
                );
                assert!(
                    dst.iter().all(|&v| v == 1),
                    "depth={depth}, width={width}, {dst:?}"
                );
            }
        }
    }

    #[test]
    fn hb_vertical_matches_independent_scalar_q6_all_blocks_and_tails() {
        if !std::arch::is_aarch64_feature_detected!("rdm") {
            return;
        }
        let mut widths: Vec<usize> = (1..=40).collect();
        widths.extend([63, 65, 127]);
        for depth in [13, 14, 15, 16] {
            let max = ((1u32 << depth) - 1) as u16;
            for width in widths.iter().copied() {
                for padding in [0, 5] {
                    let stride = width + padding;
                    for taps in [
                        0usize, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 17, 24, 31, 32, 48, 63, 64,
                        127, 513,
                    ] {
                        let start = 2;
                        let rows = start + taps;
                        // Final input row ends exactly at the last active sample.
                        let mut src = vec![0xbeef; stride * (rows - 1) + width];
                        for y in 0..rows {
                            for x in 0..width {
                                src[y * stride + x] = match (x + y) % 5 {
                                    0 => 0,
                                    1 => max,
                                    2 => 1,
                                    _ => ((x * 997 + y * 53 + x * y * 19) as u16) & max,
                                };
                            }
                        }
                        for mode in 0..5 {
                            let w = weights(taps, mode);
                            let expected: Vec<_> = (0..width)
                                .map(|x| {
                                    scalar_q6_rdm(
                                        w.iter()
                                            .enumerate()
                                            .map(|(i, &c)| (src[(start + i) * stride + x], c)),
                                        depth,
                                    )
                                })
                                .collect();
                            let mut dst = vec![0xa55a; width + 8];
                            convolve_column_hb_u16(
                                width,
                                &FilterBounds { start, size: taps },
                                &src,
                                &mut dst[1..width + 1],
                                stride,
                                &w,
                                depth,
                            );
                            assert_eq!(
                                &dst[1..width + 1],
                                expected,
                                "depth={depth}, width={width}, padding={padding}, taps={taps}, mode={mode}"
                            );
                            assert_eq!(dst[0], 0xa55a);
                            assert!(dst[width + 1..].iter().all(|&v| v == 0xa55a));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn hb_vertical_q6_saturates_before_negative_taps() {
        if !std::arch::is_aarch64_feature_detected!("rdm") {
            return;
        }
        for depth in [13, 14, 15, 16] {
            let max = ((1u32 << depth) - 1) as u16;
            let negative_taps = i32::MAX as usize / (usize::from(max) << 6);
            let mut weights = vec![i32::MAX; negative_taps + 2];
            weights.extend(std::iter::repeat_n(i32::MIN, negative_taps));
            let src = vec![max; weights.len() * 5];
            let mut dst = [0; 5];
            let expected = scalar_q6_rdm(weights.iter().map(|&w| (max, w)), depth);
            convolve_column_hb_u16(
                5,
                &FilterBounds {
                    start: 0,
                    size: weights.len(),
                },
                &src,
                &mut dst,
                5,
                &weights,
                depth,
            );
            assert_eq!(dst, [expected; 5], "depth={depth}");
            assert!(
                expected < max,
                "must return from saturation into the visible range"
            );
        }
    }

    #[test]
    fn hb_vertical_paired_twelve_matches_single_and_independent_q6() {
        if !std::arch::is_aarch64_feature_detected!("rdm") {
            return;
        }
        for depth in [13, 14, 15, 16] {
            let max = ((1u32 << depth) - 1) as u16;
            for width in [16usize, 32, 48, 64, 80, 128] {
                for padding in [0, 5] {
                    let stride = width + padding;
                    let mut src = vec![0xbeef; stride * 13 + width];
                    for y in 0..14 {
                        for x in 0..width {
                            src[y * stride + x] = match (x + y) % 3 {
                                0 => 0,
                                1 => max,
                                _ => ((x * 97 + y * 953) as u16) & max,
                            };
                        }
                    }
                    for mode in 0..5 {
                        let wa: [i32; 12] = weights(12, mode).try_into().unwrap();
                        let wb: [i32; 12] = weights(12, (mode + 1) % 5).try_into().unwrap();
                        let mut a = vec![0xa55a; width + 2];
                        let mut b = a.clone();
                        unsafe {
                            convolve_column_hb_u16_pair(
                                &src,
                                stride,
                                &mut a[1..width + 1],
                                &mut b[1..width + 1],
                                &wa,
                                &wb,
                                depth,
                            );
                        }
                        for (output, start, w) in [(&a, 0, &wa), (&b, 2, &wb)] {
                            let expected: Vec<_> = (0..width)
                                .map(|x| {
                                    scalar_q6_rdm(
                                        w.iter()
                                            .enumerate()
                                            .map(|(i, &c)| (src[(start + i) * stride + x], c)),
                                        depth,
                                    )
                                })
                                .collect();
                            assert_eq!(
                                &output[1..width + 1],
                                expected,
                                "paired depth={depth}, width={width}, padding={padding}, mode={mode}"
                            );
                            let mut single = vec![0; width];
                            convolve_column_hb_u16(
                                width,
                                &FilterBounds { start, size: 12 },
                                &src,
                                &mut single,
                                stride,
                                w,
                                depth,
                            );
                            assert_eq!(single, expected);
                            assert_eq!(output[0], 0xa55a);
                            assert_eq!(output[width + 1], 0xa55a);
                        }
                    }
                }
            }
        }
    }
}
