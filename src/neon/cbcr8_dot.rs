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
use std::arch::aarch64::*;

#[inline]
unsafe fn ld1(src: &[u8]) -> uint8x16_t {
    unsafe {
        vreinterpretq_u8_u16(vld1q_lane_u16::<0>(
            src.as_ptr() as *const u16,
            vdupq_n_u16(0),
        ))
    }
}

#[inline]
unsafe fn ld2(src: &[u8]) -> uint8x16_t {
    unsafe {
        vreinterpretq_u8_u32(vld1q_lane_u32::<0>(
            src.as_ptr() as *const u32,
            vdupq_n_u32(0),
        ))
    }
}

#[inline]
unsafe fn ld4(src: &[u8]) -> uint8x16_t {
    unsafe { vcombine_u8(vld1_u8(src.as_ptr()), vdup_n_u8(0)) }
}

#[inline(always)]
fn store_cbcr(ptr: &mut [u8], store: int32x2_t) {
    unsafe {
        let m0 = vqshrun_n_s32::<7>(vcombine_s32(store, vdup_n_s32(0)));
        let v0 = vreinterpret_u16_u8(vqmovn_u16(vcombine_u16(m0, m0)));
        vst1_lane_u16::<0>(ptr.as_mut_ptr() as *mut u16, v0);
    }
}

pub(crate) fn convolve_horizontal_cbcr_neon_rows_dot_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_cbcr_neon_rows_4_u8_impl_dot(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[target_feature(enable = "i8mm")]
fn convolve_horizontal_cbcr_neon_rows_4_u8_impl_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        const CN: usize = 2;

        let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
        let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
        let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
        let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

        static ST: [i32; 2] = [1 << 6, 1 << 6];
        let base_val = vld1_s32(ST.as_ptr());

        let tbl: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let v_tbl = vld1_u8(tbl.as_ptr());
        let weights_tbl: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let v_weights = vld1q_u8(weights_tbl.as_ptr());

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
            let mut store0 = base_val;
            let mut store1 = base_val;
            let mut store2 = base_val;
            let mut store3 = base_val;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 4 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s32(vld1q_lane_s32::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s32(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let bounds_start = bounds.start + jx;

                let src_ptr0 = src0.get_unchecked(bounds_start * CN..);
                let src_ptr1 = src1.get_unchecked(bounds_start * CN..);
                let src_ptr2 = src2.get_unchecked(bounds_start * CN..);
                let src_ptr3 = src3.get_unchecked(bounds_start * CN..);

                store0 = vusdot_s32(
                    store0,
                    vqtbl1_u8(ld4(src_ptr0), v_tbl),
                    vget_low_s8(v_weight),
                );
                store1 = vusdot_s32(
                    store1,
                    vqtbl1_u8(ld4(src_ptr1), v_tbl),
                    vget_low_s8(v_weight),
                );
                store2 = vusdot_s32(
                    store2,
                    vqtbl1_u8(ld4(src_ptr2), v_tbl),
                    vget_low_s8(v_weight),
                );
                store3 = vusdot_s32(
                    store3,
                    vqtbl1_u8(ld4(src_ptr3), v_tbl),
                    vget_low_s8(v_weight),
                );

                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s16(vld1q_lane_s16::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s16(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let bounds_start = bounds.start + jx;

                let src_ptr0 = src0.get_unchecked(bounds_start * CN..);
                let src_ptr1 = src1.get_unchecked(bounds_start * CN..);
                let src_ptr2 = src2.get_unchecked(bounds_start * CN..);
                let src_ptr3 = src3.get_unchecked(bounds_start * CN..);

                store0 = vusdot_s32(
                    store0,
                    vqtbl1_u8(ld2(src_ptr0), v_tbl),
                    vget_low_s8(v_weight),
                );
                store1 = vusdot_s32(
                    store1,
                    vqtbl1_u8(ld2(src_ptr1), v_tbl),
                    vget_low_s8(v_weight),
                );
                store2 = vusdot_s32(
                    store2,
                    vqtbl1_u8(ld2(src_ptr2), v_tbl),
                    vget_low_s8(v_weight),
                );
                store3 = vusdot_s32(
                    store3,
                    vqtbl1_u8(ld2(src_ptr3), v_tbl),
                    vget_low_s8(v_weight),
                );

                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1_dup_s8(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;

                let src_ptr0 = src0.get_unchecked(bounds_start * CN..);
                let src_ptr1 = src1.get_unchecked(bounds_start * CN..);
                let src_ptr2 = src2.get_unchecked(bounds_start * CN..);
                let src_ptr3 = src3.get_unchecked(bounds_start * CN..);

                store0 = vusdot_s32(store0, vqtbl1_u8(ld1(src_ptr0), v_tbl), w0);
                store1 = vusdot_s32(store1, vqtbl1_u8(ld1(src_ptr1), v_tbl), w0);
                store2 = vusdot_s32(store2, vqtbl1_u8(ld1(src_ptr2), v_tbl), w0);
                store3 = vusdot_s32(store3, vqtbl1_u8(ld1(src_ptr3), v_tbl), w0);

                jx += 1;
            }

            store_cbcr(chunk0, store0);
            store_cbcr(chunk1, store1);
            store_cbcr(chunk2, store2);
            store_cbcr(chunk3, store3);
        }
    }
}

pub fn convolve_horizontal_cbcr_neon_dot_row(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_cbcr_neon_dot_row_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "i8mm")]
fn convolve_horizontal_cbcr_neon_dot_row_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        const CN: usize = 2;

        static ST: [i32; 2] = [1 << 6, 1 << 6];
        let base_val = vld1_s32(ST.as_ptr());

        let tbl: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let v_tbl = vld1_u8(tbl.as_ptr());
        let weights_tbl: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let v_weights = vld1q_u8(weights_tbl.as_ptr());

        for ((dst, bounds), weights) in dst
            .as_chunks_mut::<2>()
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
            let mut store = base_val;

            while jx + 4 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s32(vld1q_lane_s32::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s32(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let bounds_start = bounds.start + jx;

                let src_ptr = src.get_unchecked(bounds_start * CN..);
                store = vusdot_s32(store, vqtbl1_u8(ld4(src_ptr), v_tbl), vget_low_s8(v_weight));

                jx += 4;
            }

            while jx + 2 <= bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let mut v_weight = vreinterpretq_s8_s16(vld1q_lane_s16::<0>(
                    w_ptr.as_ptr() as *const _,
                    vdupq_n_s16(0),
                ));
                v_weight = vqtbl1q_s8(v_weight, v_weights);
                let bounds_start = bounds.start + jx;

                let src_ptr = src.get_unchecked(bounds_start * CN..);
                store = vusdot_s32(store, vqtbl1_u8(ld2(src_ptr), v_tbl), vget_low_s8(v_weight));

                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = vld1_dup_s8(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                let src_ptr = src.get_unchecked(bounds_start * CN..);
                store = vusdot_s32(store, vqtbl1_u8(ld1(src_ptr), v_tbl), w0);
                jx += 1;
            }

            store_cbcr(dst, store);
        }
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::ResamplingFunction;
    use crate::math::WeightsGenerator;
    use crate::test_utils::XorShiftRng;

    /// The i8mm dot-product path quantizes weights to Q0.7 (via
    /// `FilterWeights::numerical_approximation_q0_7`) instead of the Q15 `i16`
    /// weights every other u8 backend in this crate uses, so `handler_provider`'s
    /// scalar reference (which expects Q15 weights) can't be reused bit-exact
    /// here. Reimplement the same Q0.7 arithmetic `store_cbcr` performs: an i32
    /// accumulator with a rounding bias of `1 << 6`, then `>> 7`, clamped to
    /// `u8` range (mirrors the two-stage saturating narrow `vqshrun_n_s32::<7>`
    /// + `vqmovn_u16` perform in hardware).
    fn scalar_reference_cbcr_dot_row(
        src: &[u8],
        dst: &mut [u8],
        filter_weights: &FilterWeights<i8>,
    ) {
        const CN: usize = 2;
        const ROUNDING: i32 = 1 << 6;
        for (dst_px, (&bounds, weights)) in dst.as_chunks_mut::<CN>().0.iter_mut().zip(
            filter_weights
                .bounds
                .iter()
                .zip(filter_weights.weights.chunks_exact(filter_weights.aligned_size)),
        ) {
            let mut acc = [ROUNDING; CN];
            for k in 0..bounds.size {
                let w = weights[k] as i32;
                let src_px = &src[(bounds.start + k) * CN..];
                for (c, acc_c) in acc.iter_mut().enumerate() {
                    *acc_c += w * src_px[c] as i32;
                }
            }
            for (c, dst_c) in dst_px.iter_mut().enumerate() {
                *dst_c = (acc[c] >> 7).clamp(0, 255) as u8;
            }
        }
    }

    fn make_row_filter_weights_q0_7(
        resampling: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> FilterWeights<i8> {
        let weights_f32 =
            <u8 as WeightsGenerator<f32>>::make_weights(resampling, in_size, out_size).unwrap();
        weights_f32.numerical_approximation_q0_7(0)
    }

    #[test]
    fn neon_dot_row_matches_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64), (256, 256), (5, 3)] {
                let filter_weights = make_row_filter_weights_q0_7(resampling, in_size, out_size);
                let mut rng = XorShiftRng::new(0xC0FFEE ^ (in_size as u64) << 32 ^ out_size as u64);
                let src = rng.fill_u8(in_size * 2);

                let mut dst_scalar = vec![0u8; out_size * 2];
                let mut dst_dot = vec![0u8; out_size * 2];
                scalar_reference_cbcr_dot_row(&src, &mut dst_scalar, &filter_weights);
                convolve_horizontal_cbcr_neon_dot_row(&src, &mut dst_dot, &filter_weights, 8);

                assert_eq!(
                    dst_scalar, dst_dot,
                    "{resampling:?} {in_size}->{out_size}: NEON i8mm dot cbcr single-row output diverges from the scalar reference"
                );
            }
        }
    }

    #[test]
    fn neon_dot_rows_4_matches_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }
        const ROWS: usize = 4;
        for resampling in [ResamplingFunction::Bilinear, ResamplingFunction::Lanczos3] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64)] {
                let filter_weights = make_row_filter_weights_q0_7(resampling, in_size, out_size);
                let mut rng = XorShiftRng::new(0xBADF00D ^ (in_size as u64) << 32 ^ out_size as u64);
                let src_stride = in_size * 2;
                let dst_stride = out_size * 2;
                let src = rng.fill_u8(src_stride * ROWS);

                let mut dst_scalar = vec![0u8; dst_stride * ROWS];
                let mut dst_dot = vec![0u8; dst_stride * ROWS];
                for row in 0..ROWS {
                    scalar_reference_cbcr_dot_row(
                        &src[row * src_stride..],
                        &mut dst_scalar[row * dst_stride..(row + 1) * dst_stride],
                        &filter_weights,
                    );
                }
                convolve_horizontal_cbcr_neon_rows_dot_4_u8(
                    &src,
                    src_stride,
                    &mut dst_dot,
                    dst_stride,
                    &filter_weights,
                    8,
                );

                assert_eq!(
                    dst_scalar, dst_dot,
                    "{resampling:?} {in_size}->{out_size}: NEON i8mm dot cbcr 4-row output diverges from the scalar reference"
                );
            }
        }
    }
}
