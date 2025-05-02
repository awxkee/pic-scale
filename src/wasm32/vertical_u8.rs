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
use crate::support::{PRECISION, ROUNDING_CONST};
use crate::wasm32::utils::{u16x8_pack_sat_u8x16, u32x4_pack_trunc_u16x8, w_zeros};
use std::arch::wasm32::*;

#[inline]
unsafe fn consume_u8_32(
    start_y: usize,
    start_x: usize,
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    filter: &[i16],
    bounds: &FilterBounds,
) {
    unsafe {
        let vld = i32x4_splat(ROUNDING_CONST);
        let mut store0 = vld;
        let mut store1 = vld;
        let mut store2 = vld;
        let mut store3 = vld;
        let mut store4 = vld;
        let mut store5 = vld;
        let mut store6 = vld;
        let mut store7 = vld;

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = v128_load16_splat(weight.as_ptr() as *const _);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row_n_0 = v128_load(s_ptr as *const v128);
            let item_row_n_1 = v128_load(s_ptr.add(16) as *const v128);

            let low_0 = u16x8_extend_low_u8x16(item_row_n_0);
            let hi_0 = u16x8_extend_high_u8x16(item_row_n_0);
            let low_1 = u16x8_extend_low_u8x16(item_row_n_1);
            let hi_1 = u16x8_extend_high_u8x16(item_row_n_1);

            store0 = i32x4_add(store0, i32x4_extmul_low_i16x8(low_0, v_weight));
            store1 = i32x4_add(store1, i32x4_extmul_high_i16x8(low_0, v_weight));

            store2 = i32x4_add(store2, i32x4_extmul_low_i16x8(hi_0, v_weight));
            store3 = i32x4_add(store3, i32x4_extmul_high_i16x8(hi_0, v_weight));

            store4 = i32x4_add(store4, i32x4_extmul_low_i16x8(low_1, v_weight));
            store5 = i32x4_add(store5, i32x4_extmul_high_i16x8(low_1, v_weight));

            store6 = i32x4_add(store6, i32x4_extmul_low_i16x8(hi_1, v_weight));
            store7 = i32x4_add(store7, i32x4_extmul_high_i16x8(hi_1, v_weight));
        }

        let zeros = w_zeros();

        store0 = i32x4_max(store0, zeros);
        store1 = i32x4_max(store1, zeros);
        store2 = i32x4_max(store2, zeros);
        store3 = i32x4_max(store3, zeros);
        store4 = i32x4_max(store4, zeros);
        store5 = i32x4_max(store5, zeros);
        store6 = i32x4_max(store6, zeros);
        store7 = i32x4_max(store7, zeros);

        store0 = i32x4_shr(store0, PRECISION as u32);
        store1 = i32x4_shr(store1, PRECISION as u32);
        store2 = i32x4_shr(store2, PRECISION as u32);
        store3 = i32x4_shr(store3, PRECISION as u32);
        store4 = i32x4_shr(store4, PRECISION as u32);
        store5 = i32x4_shr(store5, PRECISION as u32);
        store6 = i32x4_shr(store6, PRECISION as u32);
        store7 = i32x4_shr(store7, PRECISION as u32);

        let packed_16_lo_0 = u32x4_pack_trunc_u16x8(store0, store1);
        let packed_16_hi_0 = u32x4_pack_trunc_u16x8(store2, store3);

        let packed_16_lo_1 = u32x4_pack_trunc_u16x8(store4, store5);
        let packed_16_hi_1 = u32x4_pack_trunc_u16x8(store6, store7);

        let packed_8_0 = u16x8_pack_sat_u8x16(packed_16_lo_0, packed_16_hi_0);
        let packed_8_1 = u16x8_pack_sat_u8x16(packed_16_lo_1, packed_16_hi_1);

        let dst_ptr = dst.add(px);
        v128_store(dst_ptr as *mut v128, packed_8_0);
        v128_store(dst_ptr.add(16) as *mut v128, packed_8_1);
    }
}

#[inline]
unsafe fn consume_u8_16(
    start_y: usize,
    start_x: usize,
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    filter: &[i16],
    bounds: &FilterBounds,
) {
    unsafe {
        let vld = i32x4_splat(ROUNDING_CONST);
        let mut store0 = vld;
        let mut store1 = vld;
        let mut store2 = vld;
        let mut store3 = vld;

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = v128_load16_splat(weight.as_ptr() as *const _);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row_n = v128_load(s_ptr as *const v128);

            let low = u16x8_extend_low_u8x16(item_row_n);
            let hi = u16x8_extend_high_u8x16(item_row_n);

            store0 = i32x4_add(store0, i32x4_extmul_low_i16x8(low, v_weight));
            store1 = i32x4_add(store1, i32x4_extmul_high_i16x8(low, v_weight));

            store2 = i32x4_add(store2, i32x4_extmul_low_i16x8(hi, v_weight));
            store3 = i32x4_add(store3, i32x4_extmul_high_i16x8(hi, v_weight));
        }

        let zeros = w_zeros();

        store0 = i32x4_max(store0, zeros);
        store1 = i32x4_max(store1, zeros);
        store2 = i32x4_max(store2, zeros);
        store3 = i32x4_max(store3, zeros);
        store0 = i32x4_shr(store0, PRECISION as u32);
        store1 = i32x4_shr(store1, PRECISION as u32);
        store2 = i32x4_shr(store2, PRECISION as u32);
        store3 = i32x4_shr(store3, PRECISION as u32);

        let packed_16_lo = u32x4_pack_trunc_u16x8(store0, store1);
        let packed_16_hi = u32x4_pack_trunc_u16x8(store2, store3);

        let packed_8 = u16x8_pack_sat_u8x16(packed_16_lo, packed_16_hi);

        let dst_ptr = dst.add(px);
        v128_store(dst_ptr as *mut v128, packed_8);
    }
}

#[inline]
unsafe fn consume_u8_8(
    start_y: usize,
    start_x: usize,
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    filter: &[i16],
    bounds: &FilterBounds,
) {
    unsafe {
        let vld = i32x4_splat(ROUNDING_CONST);
        let mut store0 = vld;
        let mut store1 = vld;

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = v128_load16_splat(weight.as_ptr() as *const _);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let mut item_row_n = v128_load64_lane::<0>(w_zeros(), s_ptr as *const u64);
            item_row_n = u16x8_extend_low_u8x16(item_row_n);

            store0 = i32x4_add(store0, i32x4_extmul_low_i16x8(item_row_n, v_weight));
            store1 = i32x4_add(store1, i32x4_extmul_high_i16x8(item_row_n, v_weight));
        }

        let zeros = w_zeros();

        store0 = i32x4_max(store0, zeros);
        store1 = i32x4_max(store1, zeros);
        store0 = i32x4_shr(store0, PRECISION as u32);
        store1 = i32x4_shr(store1, PRECISION as u32);

        let packed_16 = u32x4_pack_trunc_u16x8(store0, store1);
        let packed_8 = u16x8_pack_sat_u8x16(packed_16, packed_16);

        let dst_ptr = dst.add(px);
        v128_store64_lane::<0>(packed_8, dst_ptr as *mut _);
    }
}

#[inline]
unsafe fn consume_u8_1(
    start_y: usize,
    start_x: usize,
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    filter: &[i16],
    bounds: &FilterBounds,
) {
    unsafe {
        let vld = i32x4_splat(ROUNDING_CONST);
        let mut store = vld;

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = v128_load16_splat(weight.as_ptr() as *const _);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row = v128_load8_splat(s_ptr);

            let low = u16x8_extend_low_u8x16(item_row);
            store = i32x4_add(store, i32x4_extmul_low_i16x8(low, v_weight));
        }

        let zeros = w_zeros();

        store = i32x4_max(store, zeros);
        store = i32x4_shr(store, PRECISION as u32);

        let packed_16 = u32x4_pack_trunc_u16x8(store, store);
        let packed_8 = u16x8_pack_sat_u8x16(packed_16, packed_16);

        let dst_ptr = dst.add(px);
        v128_store8_lane::<0>(packed_8, dst_ptr);
    }
}

#[inline]
pub fn wasm_vertical_neon_row(
    dst_width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    unsafe {
        convolve_vertical_neon_row_impl(dst_width, bounds, src, dst, src_stride, weight);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn convolve_vertical_neon_row_impl(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    unsafe {
        let mut cx = 0usize;
        let dst_width = dst.len();

        while cx + 32 < dst_width {
            consume_u8_32(
                bounds.start,
                cx,
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                weight,
                bounds,
            );

            cx += 32;
        }

        while cx + 16 < dst_width {
            consume_u8_16(
                bounds.start,
                cx,
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                weight,
                bounds,
            );

            cx += 16;
        }

        while cx + 8 < dst_width {
            consume_u8_8(
                bounds.start,
                cx,
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                weight,
                bounds,
            );

            cx += 8;
        }

        while cx < dst_width {
            consume_u8_1(
                bounds.start,
                cx,
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                weight,
                bounds,
            );
            cx += 1;
        }
    }
}
