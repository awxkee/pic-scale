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
use crate::filter_weights::FilterWeights;
use std::arch::aarch64::*;

#[must_use]
#[inline(always)]
fn conv_horiz_1_i16(start_x: usize, src: &[i16], w0: int16x4_t, store: int32x4_t) -> int32x4_t {
    unsafe {
        const CN: usize = 1;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let px = vld1_lane_s16::<0>(src_ptr.as_ptr(), vdup_n_s16(0));
        vmlal_s16(store, px, w0)
    }
}

#[must_use]
#[inline(always)]
fn conv_horiz_2_i16(start_x: usize, src: &[i16], w0: int16x4_t, store: int32x4_t) -> int32x4_t {
    unsafe {
        const CN: usize = 1;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        // Load 2 x i16 as a single u32 lane — reinterpret keeps the bits intact
        let px = vreinterpret_s16_u32(vld1_lane_u32::<0>(src_ptr.as_ptr().cast(), vdup_n_u32(0)));
        vmlal_s16(store, px, w0)
    }
}

#[must_use]
#[inline(always)]
fn conv_horiz_4_i16(
    start_x: usize,
    src: &[i16],
    weights: int16x4_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const CN: usize = 1;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let px = vld1_s16(src_ptr.as_ptr());
        vmlal_s16(store, px, weights)
    }
}

#[must_use]
#[inline(always)]
fn conv_horiz_8_i16(
    start_x: usize,
    src: &[i16],
    weights: int16x8_t,
    store: int32x4_t,
) -> int32x4_t {
    unsafe {
        const CN: usize = 1;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let px = vld1q_s16(src_ptr.as_ptr());
        let acc = vmlal_s16(store, vget_low_s16(px), vget_low_s16(weights));
        vmlal_high_s16(acc, px, weights)
    }
}

pub(crate) fn convolve_horizontal_plane_neon_rows_4_lb_i16(
    src: &[i16],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        const PRECISION: i32 = 15;
        const RND: i32 = 1 << (PRECISION - 1);
        let init = vld1q_s32([RND, 0, 0, 0].as_ptr());

        let v_max_colors = vdup_n_s16(((1i32 << (bit_depth - 1)) - 1) as i16);
        let v_min_colors = vdup_n_s16((-(1i32 << (bit_depth - 1))) as i16);

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.iter_mut();
        let iter_row1 = row1_ref.iter_mut();
        let iter_row2 = row2_ref.iter_mut();
        let iter_row3 = row3_ref.iter_mut();

        for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
            .zip(iter_row1)
            .zip(iter_row2)
            .zip(iter_row3)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut jx = 0usize;
            let mut store_0 = init;
            let mut store_1 = init;
            let mut store_2 = init;
            let mut store_3 = init;

            let bounds_size = bounds.size;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = vld1q_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_8_i16(bounds_start, src0, weights_set, store_0);
                store_1 = conv_horiz_8_i16(bounds_start, src1, weights_set, store_1);
                store_2 = conv_horiz_8_i16(bounds_start, src2, weights_set, store_2);
                store_3 = conv_horiz_8_i16(bounds_start, src3, weights_set, store_3);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_4_i16(bounds_start, src0, weights, store_0);
                store_1 = conv_horiz_4_i16(bounds_start, src1, weights, store_1);
                store_2 = conv_horiz_4_i16(bounds_start, src2, weights, store_2);
                store_3 = conv_horiz_4_i16(bounds_start, src3, weights, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 =
                    vreinterpret_s16_s32(vld1_lane_s32::<0>(w_ptr.as_ptr().cast(), vdup_n_s32(0)));
                store_0 = conv_horiz_2_i16(bounds_start, src0, w0, store_0);
                store_1 = conv_horiz_2_i16(bounds_start, src1, w0, store_1);
                store_2 = conv_horiz_2_i16(bounds_start, src2, w0, store_2);
                store_3 = conv_horiz_2_i16(bounds_start, src3, w0, store_3);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let bounds_start = bounds.start + jx;
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                store_0 = conv_horiz_1_i16(bounds_start, src0, weight0, store_0);
                store_1 = conv_horiz_1_i16(bounds_start, src1, weight0, store_1);
                store_2 = conv_horiz_1_i16(bounds_start, src2, weight0, store_2);
                store_3 = conv_horiz_1_i16(bounds_start, src3, weight0, store_3);
                jx += 1;
            }

            let packed = vpaddq_s32(vpaddq_s32(store_0, store_1), vpaddq_s32(store_2, store_3));
            let saturated = vqshrn_n_s32::<PRECISION>(packed);
            // Clamp to [-(2^(bit_depth-1)), (2^(bit_depth-1))-1]
            let clamped = vmin_s16(vmax_s16(saturated, v_min_colors), v_max_colors);

            vst1_lane_s16::<0>(chunk0, clamped);
            vst1_lane_s16::<1>(chunk1, clamped);
            vst1_lane_s16::<2>(chunk2, clamped);
            vst1_lane_s16::<3>(chunk3, clamped);
        }
    }
}

pub(crate) fn convolve_horizontal_plane_neon_i16_lb_row(
    src: &[i16],
    dst: &mut [i16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        let max_colors = (1i32 << (bit_depth - 1)) - 1;
        let min_colors = -(1i32 << (bit_depth - 1));

        const PRECISION: i32 = 15;
        const RND: i32 = 1 << (PRECISION - 1);

        let init = vld1q_s32([RND, 0, 0, 0].as_ptr());

        for ((dst, bounds), weights) in dst.iter_mut().zip(filter_weights.bounds.iter()).zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        ) {
            let bounds_size = bounds.size;
            let mut jx = 0usize;
            let mut store = init;

            while jx + 8 <= bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let weights_set = vld1q_s16(w_ptr.as_ptr());
                store = conv_horiz_8_i16(bounds_start, src, weights_set, store);
                jx += 8;
            }

            while jx + 4 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weights = vld1_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_4_i16(bounds_start, src, weights, store);
                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 =
                    vreinterpret_s16_s32(vld1_lane_s32::<0>(w_ptr.as_ptr().cast(), vdup_n_s32(0)));
                store = conv_horiz_2_i16(bounds_start, src, w0, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = vld1_dup_s16(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_1_i16(bounds_start, src, weight0, store);
                jx += 1;
            }

            let sum = vaddvq_s32(store);
            let result = (sum >> PRECISION).max(min_colors).min(max_colors) as i16;
            *dst = result;
        }
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::PicScaleError;
    use crate::ResamplingFunction;
    use crate::filter_weights::FilterWeights;
    use crate::fixed_point_horizontal::{
        convolve_row_handler_fixed_point, convolve_row_handler_fixed_point_4,
    };
    use crate::math::WeightsGenerator;
    use crate::support::PRECISION;
    use crate::test_utils::XorShiftRng;

    const BIT_DEPTH: u32 = 10;

    fn make_row_filter_weights_i16(
        resampling: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<i16>, PicScaleError> {
        let weights_f32 = <i16 as WeightsGenerator<f32>>::make_weights(resampling, in_size, out_size)?;
        Ok(weights_f32.numerical_approximation::<i16, PRECISION>(0))
    }

    /// Signed source pixels centered near zero within `bit_depth`'s signed
    /// range, matching how `i16` plane data is actually represented.
    fn fill_i16_signed(rng: &mut XorShiftRng, len: usize, bit_depth: u32) -> Vec<i16> {
        let max_unsigned = (1u16 << bit_depth) - 1;
        rng.fill_u16(len, max_unsigned)
            .into_iter()
            .map(|v| (v as i32 - (1i32 << (bit_depth - 1))) as i16)
            .collect()
    }

    #[test]
    fn neon_row_matches_scalar_reference() {
        const CN: usize = 1;
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64), (256, 256), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_i16(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xC0FFEE ^ (in_size as u64) << 32 ^ out_size as u64);
                let src = fill_i16_signed(&mut rng, in_size * CN, BIT_DEPTH);

                let mut dst_scalar = vec![0i16; out_size * CN];
                let mut dst_neon = vec![0i16; out_size * CN];
                convolve_row_handler_fixed_point::<i16, i32, CN>(
                    &src,
                    &mut dst_scalar,
                    &filter_weights,
                    BIT_DEPTH,
                );
                convolve_horizontal_plane_neon_i16_lb_row(
                    &src,
                    &mut dst_neon,
                    &filter_weights,
                    BIT_DEPTH,
                );

                assert_eq!(
                    dst_scalar, dst_neon,
                    "{resampling:?} {in_size}->{out_size}: NEON single-row output diverges from the scalar reference"
                );
            }
        }
    }

    #[test]
    fn neon_rows_4_matches_scalar_reference() {
        const CN: usize = 1;
        const ROWS: usize = 4;
        for resampling in [ResamplingFunction::Bilinear, ResamplingFunction::Lanczos3] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64)] {
                let filter_weights =
                    make_row_filter_weights_i16(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xBADF00D ^ (in_size as u64) << 32 ^ out_size as u64);
                let src_stride = in_size * CN;
                let dst_stride = out_size * CN;
                let src = fill_i16_signed(&mut rng, src_stride * ROWS, BIT_DEPTH);

                let mut dst_scalar = vec![0i16; dst_stride * ROWS];
                let mut dst_neon = vec![0i16; dst_stride * ROWS];
                convolve_row_handler_fixed_point_4::<i16, i32, CN>(
                    &src,
                    src_stride,
                    &mut dst_scalar,
                    dst_stride,
                    &filter_weights,
                    BIT_DEPTH,
                );
                convolve_horizontal_plane_neon_rows_4_lb_i16(
                    &src,
                    src_stride,
                    &mut dst_neon,
                    dst_stride,
                    &filter_weights,
                    BIT_DEPTH,
                );

                assert_eq!(
                    dst_scalar, dst_neon,
                    "{resampling:?} {in_size}->{out_size}: NEON 4-row output diverges from the scalar reference"
                );
            }
        }
    }
}
