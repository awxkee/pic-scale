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

use crate::filter_weights::FilterWeights;
use crate::neon::utils::load_3b_as_u8x16;
use std::arch::aarch64::*;

#[must_use]
#[inline]
#[target_feature(enable = "rdm")]
fn conv_horiz_rgb_4(
    start_x: usize,
    src: &[u8],
    w0: int16x8_t,
    w1: int16x8_t,
    store: int16x8_t,
    shuffle_lo: uint8x16_t,
    shuffle_hi: uint8x16_t,
) -> int16x8_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);

        let px_lo = vld1_u8(src_ptr.as_ptr());
        let px_hi_part =
            vld1_lane_u32::<0>(src_ptr.get_unchecked(8..).as_ptr().cast(), vdup_n_u32(0));

        let rgb_pixel = vcombine_u8(px_lo, vreinterpret_u8_u32(px_hi_part));

        let hi = vshrq_n_u16::<2>(vreinterpretq_u16_u8(vqtbl1q_u8(rgb_pixel, shuffle_hi)));
        let lo = vshrq_n_u16::<2>(vreinterpretq_u16_u8(vqtbl1q_u8(rgb_pixel, shuffle_lo)));

        let p = vqrdmlahq_s16(store, vreinterpretq_s16_u16(lo), w0);
        vqrdmlahq_s16(p, vreinterpretq_s16_u16(hi), w1)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "rdm")]
fn conv_horiz_rgb_2(
    start_x: usize,
    src: &[u8],
    weights: int16x8_t,
    store: int16x8_t,
    shuffle: uint8x16_t,
) -> int16x8_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let mut rgb_pixel = vld1q_lane_u32::<0>(src_ptr.as_ptr().cast(), vdupq_n_u32(0));
        rgb_pixel = vreinterpretq_u32_u16(vld1q_lane_u16::<2>(
            src_ptr.get_unchecked(4..).as_ptr() as *const u16,
            vreinterpretq_u16_u32(rgb_pixel),
        ));
        let pixel_14 = vshrq_n_u16::<2>(vreinterpretq_u16_u8(vqtbl1q_u8(
            vreinterpretq_u8_u32(rgb_pixel),
            shuffle,
        )));
        vqrdmlahq_s16(store, vreinterpretq_s16_u16(pixel_14), weights)
    }
}

#[must_use]
#[inline]
#[target_feature(enable = "rdm")]
fn conv_hor_rgb_1(
    start_x: usize,
    src: &[u8],
    w0: int16x8_t,
    store: int16x8_t,
    shuffle: uint8x16_t,
) -> int16x8_t {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let rgb_pixel = vshrq_n_u16::<2>(vreinterpretq_u16_u8(vqtbl1q_u8(
            load_3b_as_u8x16(src_ptr.as_ptr()),
            shuffle,
        )));
        vqrdmlahq_s16(store, vreinterpretq_s16_u16(rgb_pixel), w0)
    }
}

#[inline(always)]
fn write_accumulator_u8(store: int16x8_t, dst: &mut [u8]) {
    unsafe {
        let store_16 = vqshrun_n_s16::<6>(vcombine_s16(
            vadd_s16(vget_low_s16(store), vget_high_s16(store)),
            vdup_n_s16(0),
        ));
        vst1_lane_u16::<0>(dst.as_mut_ptr() as *mut u16, vreinterpret_u16_u8(store_16));
        vst1_lane_u8::<2>(dst.as_mut_ptr().add(2), store_16);
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_rdm_rows_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_neon_rdm_rows_4_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[target_feature(enable = "rdm")]
fn convolve_horizontal_rgb_neon_rdm_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CN: usize = 3;

        let shuf_table_lo: [u8; 16] = [0, 0, 1, 1, 2, 2, 255, 255, 3, 3, 4, 4, 5, 5, 255, 255];
        let shuffle_lo = vld1q_u8(shuf_table_lo.as_ptr());
        let shuf_table_hi: [u8; 16] = [6, 6, 7, 7, 8, 8, 255, 255, 9, 9, 10, 10, 11, 11, 255, 255];
        let shuffle_hi = vld1q_u8(shuf_table_hi.as_ptr());

        let weights01_shuf: [u8; 16] = [0, 1, 0, 1, 0, 1, 255, 255, 2, 3, 2, 3, 2, 3, 255, 255];
        let w01 = vld1q_u8(weights01_shuf.as_ptr());
        let weights23_shuf: [u8; 16] = [4, 5, 4, 5, 4, 5, 255, 255, 6, 7, 6, 7, 6, 7, 255, 255];
        let w23 = vld1q_u8(weights23_shuf.as_ptr());

        const ROUNDING_CONST: i16 = 1 << 5;
        let base_values: [i16; 8] = [
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            0,
            0,
            0,
            0,
            0,
        ];
        let v_base = vld1q_s16(base_values.as_ptr());

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
        let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
        let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
        let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

        for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
            .iter_mut()
            .zip(iter_row1.iter_mut())
            .zip(iter_row2.iter_mut())
            .zip(iter_row3.iter_mut())
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut jx = 0usize;
            let mut store_0 = v_base;
            let mut store_1 = v_base;
            let mut store_2 = v_base;
            let mut store_3 = v_base;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 4 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vcombine_s16(vld1_s16(w_ptr.as_ptr()), vdup_n_s16(0));
                let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(weights), w01));
                let w1 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(weights), w23));
                store_0 =
                    conv_horiz_rgb_4(bounds_start, src0, w0, w1, store_0, shuffle_lo, shuffle_hi);
                store_1 =
                    conv_horiz_rgb_4(bounds_start, src1, w0, w1, store_1, shuffle_lo, shuffle_hi);
                store_2 =
                    conv_horiz_rgb_4(bounds_start, src2, w0, w1, store_2, shuffle_lo, shuffle_hi);
                store_3 =
                    conv_horiz_rgb_4(bounds_start, src3, w0, w1, store_3, shuffle_lo, shuffle_hi);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bnds = bounds.start + jx;
                let ld_w = vld1q_dup_s32(w_ptr.as_ptr().cast());
                let v_weight = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s32(ld_w), w01));
                store_0 = conv_horiz_rgb_2(bnds, src0, v_weight, store_0, shuffle_lo);
                store_1 = conv_horiz_rgb_2(bnds, src1, v_weight, store_1, shuffle_lo);
                store_2 = conv_horiz_rgb_2(bnds, src2, v_weight, store_2, shuffle_lo);
                store_3 = conv_horiz_rgb_2(bnds, src3, v_weight, store_3, shuffle_lo);
                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bnds = bounds.start + jx;
                let weight0 = vcombine_s16(vld1_dup_s16(w_ptr.as_ptr()), vdup_n_s16(0));
                store_0 = conv_hor_rgb_1(bnds, src0, weight0, store_0, shuffle_lo);
                store_1 = conv_hor_rgb_1(bnds, src1, weight0, store_1, shuffle_lo);
                store_2 = conv_hor_rgb_1(bnds, src2, weight0, store_2, shuffle_lo);
                store_3 = conv_hor_rgb_1(bnds, src3, weight0, store_3, shuffle_lo);
                jx += 1;
            }

            write_accumulator_u8(store_0, chunk0);
            write_accumulator_u8(store_1, chunk1);
            write_accumulator_u8(store_2, chunk2);
            write_accumulator_u8(store_3, chunk3);
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_neon_rdm_row_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_neon_row_rdm_one_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "rdm")]
fn convolve_horizontal_rgb_neon_row_rdm_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CN: usize = 3;

        let shuf_table_lo: [u8; 16] = [0, 0, 1, 1, 2, 2, 255, 255, 3, 3, 4, 4, 5, 5, 255, 255];
        let shuffle_lo = vld1q_u8(shuf_table_lo.as_ptr());
        let shuf_table_hi: [u8; 16] = [6, 6, 7, 7, 8, 8, 255, 255, 9, 9, 10, 10, 11, 11, 255, 255];
        let shuffle_hi = vld1q_u8(shuf_table_hi.as_ptr());

        let weights01_shuf: [u8; 16] = [0, 1, 0, 1, 0, 1, 255, 255, 2, 3, 2, 3, 2, 3, 255, 255];
        let w01 = vld1q_u8(weights01_shuf.as_ptr());
        let weights23_shuf: [u8; 16] = [4, 5, 4, 5, 4, 5, 255, 255, 6, 7, 6, 7, 6, 7, 255, 255];
        let w23 = vld1q_u8(weights23_shuf.as_ptr());

        const ROUNDING_CONST: i16 = 1 << 5;

        let base_values: [i16; 8] = [
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            0,
            0,
            0,
            0,
            0,
        ];
        let v_base = vld1q_s16(base_values.as_ptr());

        for ((dst, bounds), weights) in dst
            .as_chunks_mut::<CN>()
            .0
            .iter_mut()
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let bounds_size = bounds.size;

            let mut jx = 0usize;
            let mut store = v_base;

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vcombine_s16(vld1_s16(w_ptr.as_ptr()), vdup_n_s16(0));
                let w0 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(weights), w01));
                let w1 = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(weights), w23));
                store = conv_horiz_rgb_4(bounds_start, src, w0, w1, store, shuffle_lo, shuffle_hi);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let ld_w = vld1q_dup_s32(w_ptr.as_ptr().cast());
                let v_weight = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s32(ld_w), w01));
                store = conv_horiz_rgb_2(bounds_start, src, v_weight, store, shuffle_lo);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = vcombine_s16(vld1_dup_s16(w_ptr.as_ptr()), vdup_n_s16(0));
                let bnds = bounds.start + jx;
                store = conv_hor_rgb_1(bnds, src, weight0, store, shuffle_lo);
                jx += 1;
            }

            write_accumulator_u8(store, dst);
        }
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::ResamplingFunction;
    use crate::test_utils::{XorShiftRng, make_row_filter_weights_u8};

    // The `rdm` path keeps two independent i16 accumulators per channel
    // inside one 8-lane register (lanes 0-2 for even-indexed taps, lanes 4-6
    // for odd-indexed taps, folded together only in `write_accumulator_u8` at
    // the very end), each updated with `vqrdmlahq_s16` - a *saturating*
    // rounding-doubling-multiply-accumulate, not a plain widening MAC. That
    // makes `handler_provider`'s i32-accumulator scalar reference unusable
    // bit-exact here, so this reimplements the exact hardware arithmetic:
    // pixels are widened to a 14-bit fixed representation the same way
    // `expand8_to_14` does (`(byte * 257) >> 2`), each tap is folded in with
    // the same rounding-doubling-multiply-high-add formula RDM uses, and the
    // two per-channel accumulators are added (wrapping, like `vadd_s16`) only
    // after every tap has been processed.
    fn expand8_to_14(x: u8) -> i16 {
        (((x as u32) * 257) >> 2) as i16
    }

    fn sqrdmlah_i16(acc: i16, a: i16, b: i16) -> i16 {
        let product2 = 2i64 * (a as i64) * (b as i64);
        let rounded = (product2 + (1i64 << 15)) >> 16;
        (acc as i64 + rounded).clamp(i16::MIN as i64, i16::MAX as i64) as i16
    }

    fn scalar_reference_rgb_rdm_row(
        src: &[u8],
        dst: &mut [u8],
        filter_weights: &FilterWeights<i16>,
    ) {
        const CN: usize = 3;
        const ROUNDING: i16 = 1 << 5;
        for (dst_px, (&bounds, weights)) in dst.as_chunks_mut::<CN>().0.iter_mut().zip(
            filter_weights
                .bounds
                .iter()
                .zip(filter_weights.weights.chunks_exact(filter_weights.aligned_size)),
        ) {
            let mut low = [ROUNDING; CN];
            let mut high = [0i16; CN];
            for k in 0..bounds.size {
                let w = weights[k];
                let src_px = &src[(bounds.start + k) * CN..];
                let target = if k % 2 == 0 { &mut low } else { &mut high };
                for c in 0..CN {
                    target[c] = sqrdmlah_i16(target[c], expand8_to_14(src_px[c]), w);
                }
            }
            for c in 0..CN {
                let folded = low[c].wrapping_add(high[c]);
                dst_px[c] = ((folded >> 6) as i32).clamp(0, 255) as u8;
            }
        }
    }

    #[test]
    fn neon_rdm_row_matches_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("rdm") {
            return;
        }
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64), (256, 256), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_u8(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xC0FFEE ^ (in_size as u64) << 32 ^ out_size as u64);
                let src = rng.fill_u8(in_size * 3);

                let mut dst_scalar = vec![0u8; out_size * 3];
                let mut dst_rdm = vec![0u8; out_size * 3];
                scalar_reference_rgb_rdm_row(&src, &mut dst_scalar, &filter_weights);
                convolve_horizontal_rgb_neon_rdm_row_one(&src, &mut dst_rdm, &filter_weights, 8);

                assert_eq!(
                    dst_scalar, dst_rdm,
                    "{resampling:?} {in_size}->{out_size}: NEON rdm rgb single-row output diverges from the scalar reference"
                );
            }
        }
    }

    #[test]
    fn neon_rdm_rows_4_matches_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("rdm") {
            return;
        }
        const ROWS: usize = 4;
        for resampling in [ResamplingFunction::Bilinear, ResamplingFunction::Lanczos3] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64)] {
                let filter_weights =
                    make_row_filter_weights_u8(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xBADF00D ^ (in_size as u64) << 32 ^ out_size as u64);
                let src_stride = in_size * 3;
                let dst_stride = out_size * 3;
                let src = rng.fill_u8(src_stride * ROWS);

                let mut dst_scalar = vec![0u8; dst_stride * ROWS];
                let mut dst_rdm = vec![0u8; dst_stride * ROWS];
                for row in 0..ROWS {
                    scalar_reference_rgb_rdm_row(
                        &src[row * src_stride..],
                        &mut dst_scalar[row * dst_stride..(row + 1) * dst_stride],
                        &filter_weights,
                    );
                }
                convolve_horizontal_rgb_neon_rdm_rows_4(
                    &src,
                    src_stride,
                    &mut dst_rdm,
                    dst_stride,
                    &filter_weights,
                    8,
                );

                assert_eq!(
                    dst_scalar, dst_rdm,
                    "{resampling:?} {in_size}->{out_size}: NEON rdm rgb 4-row output diverges from the scalar reference"
                );
            }
        }
    }
}
