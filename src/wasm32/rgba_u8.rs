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
use crate::wasm32::utils::{i32x4_saturate2_to_u8, i32x4_saturate_to_u8};
use std::arch::wasm32::*;

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_2_u8(
    start_x: usize,
    src: &[u8],
    w0: v128,
    w1: v128,
    store: v128,
) -> v128 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgb_pixel = v128_load64_lane::<0>(i32x4_splat(0), src_ptr.as_ptr() as *const _);
    let wide = u16x8_extend_low_u8x16(rgb_pixel);

    let acc = i32x4_add(store, i32x4_extmul_high_i16x8(wide, w1));
    i32x4_add(acc, i32x4_extmul_low_i16x8(wide, w0))
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_4_u8(
    start_x: usize,
    src: &[u8],
    w0: v128,
    w1: v128,
    w2: v128,
    w3: v128,
    store: v128,
) -> v128 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = v128_load(src_ptr.as_ptr() as *const _);

    let hi = u16x8_extend_high_u8x16(rgba_pixel);
    let lo = u16x8_extend_low_u8x16(rgba_pixel);

    let acc = i32x4_add(store, i32x4_extmul_high_i16x8(hi, w3));
    let acc = i32x4_add(acc, i32x4_extmul_low_i16x8(hi, w2));
    let acc = i32x4_add(acc, i32x4_extmul_high_i16x8(lo, w1));
    i32x4_add(acc, i32x4_extmul_low_i16x8(lo, w0))
}

#[must_use]
#[inline(always)]
unsafe fn conv_horiz_rgba_1_u8(start_x: usize, src: &[u8], w0: v128, store: v128) -> v128 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel = v128_load32_lane::<0>(i32x4_splat(0), src_ptr.as_ptr() as *const _);
    let lo = u16x8_extend_low_u8x16(rgba_pixel);
    i32x4_add(store, i32x4_extmul_low_i16x8(lo, w0))
}

pub(crate) fn convolve_horizontal_rgba_wasm_rows_4_u8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_wasm_rows_4_u8_impl::<15>(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[target_feature(enable = "simd128")]
unsafe fn convolve_horizontal_rgba_wasm_rows_4_u8_impl<const PRECISION: i32>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let rnd_const: i32 = (1 << (PRECISION - 1)) - 1;
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

            let bounds_size = bounds.size;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let w1 = v128_load16_splat(w_ptr.get_unchecked(1..).as_ptr() as *const _);
                let w2 = v128_load16_splat(w_ptr.get_unchecked(2..).as_ptr() as *const _);
                let w3 = v128_load16_splat(w_ptr.get_unchecked(3..).as_ptr() as *const _);
                store_0 = conv_horiz_rgba_4_u8(bounds_start, src0, w0, w1, w2, w3, store_0);
                store_1 = conv_horiz_rgba_4_u8(bounds_start, src1, w0, w1, w2, w3, store_1);
                store_2 = conv_horiz_rgba_4_u8(bounds_start, src2, w0, w1, w2, w3, store_2);
                store_3 = conv_horiz_rgba_4_u8(bounds_start, src3, w0, w1, w2, w3, store_3);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let w1 = v128_load16_splat(w_ptr.get_unchecked(1..).as_ptr() as *const _);
                store_0 = conv_horiz_rgba_2_u8(bounds_start, src0, w0, w1, store_0);
                store_1 = conv_horiz_rgba_2_u8(bounds_start, src1, w0, w1, store_1);
                store_2 = conv_horiz_rgba_2_u8(bounds_start, src2, w0, w1, store_2);
                store_3 = conv_horiz_rgba_2_u8(bounds_start, src3, w0, w1, store_3);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                store_0 = conv_horiz_rgba_1_u8(bounds_start, src0, w0, store_0);
                store_1 = conv_horiz_rgba_1_u8(bounds_start, src1, w0, store_1);
                store_2 = conv_horiz_rgba_1_u8(bounds_start, src2, w0, store_2);
                store_3 = conv_horiz_rgba_1_u8(bounds_start, src3, w0, store_3);
                jx += 1;
            }

            let mut store_16_0 = i32x4_shr(store_0, PRECISION as u32);
            let mut store_16_1 = i32x4_shr(store_1, PRECISION as u32);
            let mut store_16_2 = i32x4_shr(store_2, PRECISION as u32);
            let mut store_16_3 = i32x4_shr(store_3, PRECISION as u32);

            store_16_0 = i32x4_max(store_16_0, i32x4_splat(0));
            store_16_1 = i32x4_max(store_16_1, i32x4_splat(0));
            store_16_2 = i32x4_max(store_16_2, i32x4_splat(0));
            store_16_3 = i32x4_max(store_16_3, i32x4_splat(0));

            let zs0 = i32x4_saturate2_to_u8(store_16_0, store_16_1);
            let zs1 = i32x4_saturate2_to_u8(store_16_2, store_16_3);

            v128_store32_lane::<0>(zs0, chunk0.as_mut_ptr() as *mut _);
            v128_store32_lane::<1>(zs0, chunk1.as_mut_ptr() as *mut _);
            v128_store32_lane::<0>(zs1, chunk2.as_mut_ptr() as *mut _);
            v128_store32_lane::<1>(zs1, chunk3.as_mut_ptr() as *mut _);
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_wasm_row(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_wasm_row_impl::<15>(src, dst, filter_weights);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn convolve_horizontal_rgba_wasm_row_impl<const PRECISION: i32>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let rnd_const: i32 = (1 << (PRECISION - 1)) - 1;

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
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let w1 = v128_load16_splat(w_ptr.get_unchecked(1..).as_ptr() as *const _);
                let w2 = v128_load16_splat(w_ptr.get_unchecked(2..).as_ptr() as *const _);
                let w3 = v128_load16_splat(w_ptr.get_unchecked(3..).as_ptr() as *const _);
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_4_u8(bounds_start, src, w0, w1, w2, w3, store);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let w1 = v128_load16_splat(w_ptr.get_unchecked(1..).as_ptr() as *const _);
                store = conv_horiz_rgba_2_u8(bounds_start, src, w0, w1, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = v128_load16_splat(w_ptr.as_ptr() as *const _);
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_1_u8(bounds_start, src, w0, store);
                jx += 1;
            }

            let mut store_16 = i32x4_shr(store, PRECISION as u32);
            store_16 = i32x4_max(store_16, i32x4_splat(0));
            store_16 = i32x4_saturate_to_u8(store_16);

            v128_store32_lane::<0>(store_16, dst.as_mut_ptr() as *mut _);
        }
    }
}
