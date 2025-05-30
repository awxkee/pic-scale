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
use crate::support::PRECISION;
use crate::wasm32::utils::i32x4_saturate_to_u8;
use std::arch::wasm32::*;

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgb_4_u8(
    start_x: usize,
    src: &[u8],
    w0: v128,
    w1: v128,
    w2: v128,
    w3: v128,
    store: v128,
    shuffle: v128,
) -> v128 {
    unsafe {
        const COMPONENTS: usize = 3;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let px_lo = v128_load64_lane::<0>(i32x4_splat(0), src_ptr.as_ptr() as *const _);
        let mut rgb_pixel =
            v128_load32_lane::<2>(px_lo, src_ptr.get_unchecked(8..).as_ptr() as *const u32);

        rgb_pixel = u8x16_swizzle(rgb_pixel, shuffle);
        let hi = u16x8_extend_high_u8x16(rgb_pixel);
        let lo = u16x8_extend_low_u8x16(rgb_pixel);

        let acc = i32x4_add(store, i32x4_extmul_high_i16x8(hi, w3));
        let acc = i32x4_add(acc, i32x4_extmul_low_i16x8(hi, w2));
        let acc = i32x4_add(acc, i32x4_extmul_high_i16x8(lo, w1));
        i32x4_add(acc, i32x4_extmul_low_i16x8(lo, w0))
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_2_u8(
    start_x: usize,
    src: &[u8],
    w0: v128,
    w1: v128,
    store: v128,
    shuffle: v128,
) -> v128 {
    unsafe {
        const COMPONENTS: usize = 3;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
        let mut rgb_pixel = v128_load32_lane::<0>(i32x4_splat(0), src_ptr.as_ptr() as *const u32);
        rgb_pixel =
            v128_load16_lane::<2>(rgb_pixel, src_ptr.get_unchecked(4..).as_ptr() as *const u16);
        rgb_pixel = u8x16_swizzle(rgb_pixel, shuffle);

        let wide = u16x8_extend_low_u8x16(rgb_pixel);

        let acc = i32x4_add(store, i32x4_extmul_high_i16x8(wide, w1));
        i32x4_add(acc, i32x4_extmul_low_i16x8(wide, w0))
    }
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_1_u8(start_x: usize, src: &[u8], w0: v128, store: v128) -> v128 {
    unsafe {
        const COMPONENTS: usize = 3;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
        let mut rgb_pixel = v128_load16_lane::<0>(i32x4_splat(0), src_ptr.as_ptr() as *const _);
        rgb_pixel = v128_load8_lane::<2>(rgb_pixel, src_ptr.get_unchecked(2..).as_ptr());
        let lo = u16x8_extend_low_u8x16(rgb_pixel);
        i32x4_add(store, i32x4_extmul_low_i16x8(lo, w0))
    }
}

#[inline(always)]
unsafe fn write_accumulator_u8<const PRECISION: i32>(store: v128, dst: &mut [u8]) {
    unsafe {
        let mut store_16 = i32x4_shr(store, PRECISION as u32);
        store_16 = i32x4_max(store_16, i32x4_splat(0));
        store_16 = i32x4_saturate_to_u8(store_16);
        v128_store16_lane::<0>(store_16, dst.as_mut_ptr() as *mut u16);
        v128_store8_lane::<2>(store_16, dst.get_unchecked_mut(2..).as_mut_ptr());
    }
}

pub(crate) fn convolve_horizontal_rgb_wasm_rows_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgb_neon_rows_4_impl::<PRECISION>(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[target_feature(enable = "simd128")]
unsafe fn convolve_horizontal_rgb_neon_rows_4_impl<const PRECISION: i32>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        let shuf_table_1: [u8; 16] = [0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255];
        let shuffle = v128_load(shuf_table_1.as_ptr() as *const _);

        // (r0 g0 b0 r1) (g2 b2 r3 g3) (b3 r4 g4 b4) (r5 g5 b5 r6)

        let rnd_const: i32 = 1 << (PRECISION - 1);

        const CHANNELS: usize = 3;
        let init = i32x4_splat(rnd_const);
        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
        let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
        let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
        let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

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

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 4 < bounds.size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let w1 = v128_load16_splat(w_ptr.get_unchecked(1..).as_ptr() as *const _);
                let w2 = v128_load16_splat(w_ptr.get_unchecked(2..).as_ptr() as *const _);
                let w3 = v128_load16_splat(w_ptr.get_unchecked(3..).as_ptr() as *const _);
                store_0 = conv_horiz_rgb_4_u8(bounds_start, src0, w0, w1, w2, w3, store_0, shuffle);
                store_1 = conv_horiz_rgb_4_u8(bounds_start, src1, w0, w1, w2, w3, store_1, shuffle);
                store_2 = conv_horiz_rgb_4_u8(bounds_start, src2, w0, w1, w2, w3, store_2, shuffle);
                store_3 = conv_horiz_rgb_4_u8(bounds_start, src3, w0, w1, w2, w3, store_3, shuffle);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bnds = bounds.start + jx;
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let w1 = v128_load16_splat(w_ptr.get_unchecked(1..).as_ptr() as *const _);
                store_0 = conv_horiz_rgba_2_u8(bnds, src0, w0, w1, store_0, shuffle);
                store_1 = conv_horiz_rgba_2_u8(bnds, src1, w0, w1, store_1, shuffle);
                store_2 = conv_horiz_rgba_2_u8(bnds, src2, w0, w1, store_2, shuffle);
                store_3 = conv_horiz_rgba_2_u8(bnds, src3, w0, w1, store_3, shuffle);
                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bnds = bounds.start + jx;
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                store_0 = conv_horiz_rgba_1_u8(bnds, src0, w0, store_0);
                store_1 = conv_horiz_rgba_1_u8(bnds, src1, w0, store_1);
                store_2 = conv_horiz_rgba_1_u8(bnds, src2, w0, store_2);
                store_3 = conv_horiz_rgba_1_u8(bnds, src3, w0, store_3);
                jx += 1;
            }

            write_accumulator_u8::<PRECISION>(store_0, chunk0);
            write_accumulator_u8::<PRECISION>(store_1, chunk1);
            write_accumulator_u8::<PRECISION>(store_2, chunk2);
            write_accumulator_u8::<PRECISION>(store_3, chunk3);
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_wasm_row_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgb_neon_row_one_impl::<PRECISION>(src, dst, filter_weights);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn convolve_horizontal_rgb_neon_row_one_impl<const PRECISION: i32>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CHANNELS: usize = 3;

        let shuf_table_1: [u8; 16] = [0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255];
        let shuffle = v128_load(shuf_table_1.as_ptr() as *const _);

        let rnd_const: i32 = 1 << (PRECISION - 1);

        for ((dst, bounds), weights) in dst
            .chunks_exact_mut(CHANNELS)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let bounds_size = bounds.size;

            let mut jx = 0usize;
            let mut store = i32x4_splat(rnd_const);

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let w1 = v128_load16_splat(w_ptr.get_unchecked(1..).as_ptr() as *const _);
                let w2 = v128_load16_splat(w_ptr.get_unchecked(2..).as_ptr() as *const _);
                let w3 = v128_load16_splat(w_ptr.get_unchecked(3..).as_ptr() as *const _);
                store = conv_horiz_rgb_4_u8(bounds_start, src, w0, w1, w2, w3, store, shuffle);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let w1 = v128_load16_splat(w_ptr.get_unchecked(1..).as_ptr() as *const _);
                store = conv_horiz_rgba_2_u8(bounds_start, src, w0, w1, store, shuffle);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let bnds = bounds.start + jx;
                store = conv_horiz_rgba_1_u8(bnds, src, w0, store);
                jx += 1;
            }

            write_accumulator_u8::<PRECISION>(store, dst);
        }
    }
}
