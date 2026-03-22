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

#[inline]
#[target_feature(enable = "neon")]
fn u8x8_to_i16x8(v: uint8x8_t) -> int16x8_t {
    vreinterpretq_s16_u16(vmovl_u8(v))
}

#[inline]
#[target_feature(enable = "neon")]
fn u8x8_hi_to_i16x8(v: uint8x16_t) -> int16x8_t {
    vreinterpretq_s16_u16(vmovl_high_u8(v))
}

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_8_horiz(store: int32x4_t, ptr: &[u8], w0: int16x8_t, w1: int16x8_t) -> int32x4_t {
    let pixels = unsafe { vld1q_u8(ptr.as_ptr()) };
    let lo = u8x8_to_i16x8(vget_low_u8(pixels));
    let hi = u8x8_hi_to_i16x8(pixels);

    let mut s0 = vmlal_s16(store, vget_low_s16(lo), vget_low_s16(w0));
    s0 = vmlal_high_s16(s0, lo, w0);

    s0 = vmlal_s16(s0, vget_low_s16(hi), vget_low_s16(w1));
    vmlal_high_s16(s0, hi, w1)
}

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_4_horiz(store: int32x4_t, ptr: &[u8], w: int16x8_t) -> int32x4_t {
    let pixels = unsafe { vld1_u8(ptr.as_ptr()) };
    let lo = u8x8_to_i16x8(pixels);

    let s0 = vmlal_s16(store, vget_low_s16(lo), vget_low_s16(w));
    vmlal_high_s16(s0, lo, w)
}

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_1_horiz(store: int32x4_t, ptr: &[u8], w: int16x8_t) -> int32x4_t {
    unsafe {
        let raw = vld1_lane_u16::<0>(ptr.as_ptr().cast(), vdup_n_u16(0));
        let pixels = u8x8_to_i16x8(vreinterpret_u8_u16(raw));
        vmlal_s16(store, vget_low_s16(pixels), vget_low_s16(w))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_cbcr<const PRECISION: i32>(ptr: &mut [u8; 2], lo: int32x4_t) {
    unsafe {
        let uz = vextq_s32::<2>(lo, lo);
        let cb_sum = vaddq_s32(uz, lo);
        let shifted = vshrq_n_s32::<PRECISION>(cb_sum);
        let narrow16 = vqmovn_s32(shifted);
        let narrow8 = vqmovun_s16(vcombine_s16(narrow16, narrow16));
        vst1_lane_u16::<0>(ptr.as_mut_ptr().cast(), vreinterpret_u16_u8(narrow8));
    }
}

pub(crate) fn convolve_horizontal_cbcr_neon_rows_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_cbcr_neon_rows_4_u8_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        )
    }
}

#[target_feature(enable = "neon")]
fn convolve_horizontal_cbcr_neon_rows_4_u8_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    const CN: usize = 2;

    const PRECISION: i32 = 15;
    const ROUNDING: i32 = 1 << (PRECISION - 1);

    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let base = unsafe { vld1q_s32([ROUNDING, ROUNDING, 0, 0].as_ptr()) };

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
        unsafe {
            let src0 = src;
            let src1 = src.get_unchecked(src_stride..);
            let src2 = src.get_unchecked(src_stride * 2..);
            let src3 = src.get_unchecked(src_stride * 3..);

            let mut st0 = base;
            let mut st1 = base;
            let mut st2 = base;
            let mut st3 = base;

            let mut jx = 0usize;

            while jx + 8 <= bounds.size {
                let wptr = weights.get_unchecked(jx..);
                let raw_w = vld1q_s16(wptr.as_ptr());
                let zipped = vzipq_s16(
                    vcombine_s16(vget_low_s16(raw_w), vget_low_s16(raw_w)),
                    vcombine_s16(vget_low_s16(raw_w), vget_low_s16(raw_w)),
                );
                let w0 = zipped.0;
                let raw_w_hi = vcombine_s16(vget_high_s16(raw_w), vget_high_s16(raw_w));
                let zipped1 = vzipq_s16(raw_w_hi, raw_w_hi);
                let w1 = zipped1.0;

                let bstart = (bounds.start + jx) * CN;

                st0 = accumulate_8_horiz(st0, src0.get_unchecked(bstart..), w0, w1);
                st1 = accumulate_8_horiz(st1, src1.get_unchecked(bstart..), w0, w1);
                st2 = accumulate_8_horiz(st2, src2.get_unchecked(bstart..), w0, w1);
                st3 = accumulate_8_horiz(st3, src3.get_unchecked(bstart..), w0, w1);
                jx += 8;
            }

            while jx + 4 <= bounds.size {
                let wptr = weights.get_unchecked(jx..);
                let raw_w = vld1_s16(wptr.as_ptr());
                let zipped = vzip_s16(raw_w, raw_w);
                let w = vcombine_s16(zipped.0, zipped.1);

                let bstart = (bounds.start + jx) * CN;
                st0 = accumulate_4_horiz(st0, src0.get_unchecked(bstart..), w);
                st1 = accumulate_4_horiz(st1, src1.get_unchecked(bstart..), w);
                st2 = accumulate_4_horiz(st2, src2.get_unchecked(bstart..), w);
                st3 = accumulate_4_horiz(st3, src3.get_unchecked(bstart..), w);
                jx += 4;
            }

            while jx < bounds.size {
                let w = vld1q_dup_s16(weights.get_unchecked(jx));
                let bstart = (bounds.start + jx) * CN;
                st0 = accumulate_1_horiz(st0, src0.get_unchecked(bstart..), w);
                st1 = accumulate_1_horiz(st1, src1.get_unchecked(bstart..), w);
                st2 = accumulate_1_horiz(st2, src2.get_unchecked(bstart..), w);
                st3 = accumulate_1_horiz(st3, src3.get_unchecked(bstart..), w);
                jx += 1;
            }

            store_cbcr::<PRECISION>(chunk0, st0);
            store_cbcr::<PRECISION>(chunk1, st1);
            store_cbcr::<PRECISION>(chunk2, st2);
            store_cbcr::<PRECISION>(chunk3, st3);
        }
    }
}

pub(crate) fn convolve_horizontal_cbcr_neon_row(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe { convolve_horizontal_cbcr_neon_row_impl(src, dst, filter_weights) }
}

#[target_feature(enable = "neon")]
fn convolve_horizontal_cbcr_neon_row_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    const CN: usize = 2;
    const PRECISION: i32 = 15;
    const ROUNDING: i32 = 1 << (PRECISION - 1);
    let base = unsafe { vld1q_s32([ROUNDING, ROUNDING, 0, 0].as_ptr()) };

    for (chunk, (bounds, weights)) in dst.as_chunks_mut::<2>().0.iter_mut().zip(
        filter_weights.bounds.iter().zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        ),
    ) {
        unsafe {
            let mut st = base;
            let mut jx = 0usize;

            while jx + 8 <= bounds.size {
                let wptr = weights.get_unchecked(jx..);
                let raw_w = vld1q_s16(wptr.as_ptr());
                let zipped = vzipq_s16(
                    vcombine_s16(vget_low_s16(raw_w), vget_low_s16(raw_w)),
                    vcombine_s16(vget_low_s16(raw_w), vget_low_s16(raw_w)),
                );
                let w0 = zipped.0;
                let raw_w_hi = vcombine_s16(vget_high_s16(raw_w), vget_high_s16(raw_w));
                let w1 = vzipq_s16(raw_w_hi, raw_w_hi).0;

                let bstart = (bounds.start + jx) * CN;
                st = accumulate_8_horiz(st, src.get_unchecked(bstart..), w0, w1);
                jx += 8;
            }

            while jx + 4 <= bounds.size {
                let wptr = weights.get_unchecked(jx..);
                let raw_w = vld1_s16(wptr.as_ptr());
                let zipped = vzip_s16(raw_w, raw_w);
                let w = vcombine_s16(zipped.0, zipped.1);

                let bstart = (bounds.start + jx) * CN;
                st = accumulate_4_horiz(st, src.get_unchecked(bstart..), w);
                jx += 4;
            }

            while jx < bounds.size {
                let w = vld1q_dup_s16(weights.get_unchecked(jx));
                let bstart = (bounds.start + jx) * CN;
                st = accumulate_1_horiz(st, src.get_unchecked(bstart..), w);
                jx += 1;
            }

            store_cbcr::<PRECISION>(chunk, st);
        }
    }
}
