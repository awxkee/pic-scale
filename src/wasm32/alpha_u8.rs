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
use crate::alpha_handle_u8::premultiply_alpha_rgba_row_impl;
use crate::wasm32::transpose::{wasm_load_deinterleave_u8x4, wasm_store_interleave_u8x4};
use crate::wasm32::utils::*;
use crate::WorkloadStrategy;
use rayon::ThreadPool;
use std::arch::wasm32::*;

pub fn wasm_unpremultiply_alpha_rgba(
    in_place: &mut [u8],
    width: usize,
    _: usize,
    stride: usize,
    _: &Option<ThreadPool>,
    _: WorkloadStrategy,
) {
    unsafe {
        wasm_unpremultiply_alpha_rgba_impl(in_place, width, stride);
    }
}

#[inline]
unsafe fn unpremultiply_vec(pixel: v128, alpha: v128) -> v128 {
    let scale_back = u8x16_splat(255);

    let low_part = u16x8_extmul_low_u8x16(pixel, scale_back);
    let high_part = u16x8_extmul_high_u8x16(pixel, scale_back);

    let low_alpha_part = u16x8_extend_low_u8x16(alpha);
    let high_alpha_part = u16x8_extend_high_u8x16(alpha);

    let lo_lo = f32x4_convert_u32x4(u32x4_extend_low_u16x8(low_part));
    let lo_hi = f32x4_convert_u32x4(u32x4_extend_high_u16x8(low_part));
    let hi_lo = f32x4_convert_u32x4(u32x4_extend_low_u16x8(high_part));
    let hi_hi = f32x4_convert_u32x4(u32x4_extend_high_u16x8(high_part));

    // f32x4_convert_u32x4 properly handles NaN so we can ignore 0 masking
    let lo_lo_alpha = f32x4_convert_u32x4(u32x4_extend_low_u16x8(low_alpha_part));
    let lo_hi_alpha = f32x4_convert_u32x4(u32x4_extend_high_u16x8(low_alpha_part));
    let hi_lo_alpha = f32x4_convert_u32x4(u32x4_extend_low_u16x8(high_alpha_part));
    let hi_hi_alpha = f32x4_convert_u32x4(u32x4_extend_high_u16x8(high_alpha_part));

    let lo_lo_0 = u32x4_trunc_sat_f32x4(f32x4_div(lo_lo, lo_lo_alpha));
    let lo_hi_0 = u32x4_trunc_sat_f32x4(f32x4_div(lo_hi, lo_hi_alpha));
    let hi_lo_0 = u32x4_trunc_sat_f32x4(f32x4_div(hi_lo, hi_lo_alpha));
    let hi_hi_0 = u32x4_trunc_sat_f32x4(f32x4_div(hi_hi, hi_hi_alpha));

    let packed_lo_16 = u32x4_pack_trunc_u16x8(lo_lo_0, lo_hi_0);
    let packed_hi_16 = u32x4_pack_trunc_u16x8(hi_lo_0, hi_hi_0);
    u16x8_pack_sat_u8x16(packed_lo_16, packed_hi_16)
}

#[inline]
pub(crate) unsafe fn wasm_u16x8_div_by_255(v: v128) -> v128 {
    let addition = u16x8_splat(127);
    u16x8_shr(u16x8_add(u16x8_add(v, addition), u16x8_shr(v, 8)), 8)
}

#[inline]
unsafe fn premultiply_vec(pixel: v128, alpha: v128) -> v128 {
    let lo_product = u16x8_extmul_low_u8x16(pixel, alpha);
    let hi_product = u16x8_extmul_high_u8x16(pixel, alpha);

    let lo_packed = wasm_u16x8_div_by_255(lo_product);
    let hi_packed = wasm_u16x8_div_by_255(hi_product);
    u16x8_pack_sat_u8x16(lo_packed, hi_packed)
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_unpremultiply_alpha_rgba_impl(in_place: &mut [u8], width: usize, stride: usize) {
    in_place.chunks_exact_mut(stride).for_each(|row| unsafe {
        let mut rem = &mut row[..width * 4];

        for dst in rem.chunks_exact_mut(16 * 4) {
            let src_ptr = dst.as_ptr();
            let mut pixel = wasm_load_deinterleave_u8x4(src_ptr);

            pixel.0 = unpremultiply_vec(pixel.0, pixel.3);
            pixel.1 = unpremultiply_vec(pixel.1, pixel.3);
            pixel.2 = unpremultiply_vec(pixel.2, pixel.3);
            let dst_ptr = dst.as_mut_ptr();
            wasm_store_interleave_u8x4(dst_ptr, pixel);
        }

        rem = rem.chunks_exact_mut(16 * 4).into_remainder();

        for dst in rem.chunks_exact_mut(4) {
            let a = dst[3];
            if a != 0 {
                let a_recip = 1. / a as f32;
                dst[0] = ((dst[0] as f32 * 255.) * a_recip) as u8;
                dst[1] = ((dst[1] as f32 * 255.) * a_recip) as u8;
                dst[2] = ((dst[2] as f32 * 255.) * a_recip) as u8;
                dst[3] = ((a as f32 * 255.) * a_recip) as u8;
            }
        }
    });
}

pub fn wasm_premultiply_alpha_rgba(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    width: usize,
    _: usize,
    stride: usize,
    _: &Option<ThreadPool>,
) {
    unsafe {
        wasm_premultiply_alpha_rgba_impl(dst, dst_stride, src, stride, width);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn wasm_premultiply_alpha_rgba_impl(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    width: usize,
) {
    dst.chunks_exact_mut(dst_stride)
        .zip(src.chunks_exact(src_stride))
        .for_each(|(dst, src)| unsafe {
            let mut rem = &mut dst[..width * 4];
            let mut src_rem = src;

            for (dst, src) in rem
                .chunks_exact_mut(16 * 4)
                .zip(src_rem.chunks_exact(16 * 4))
            {
                let mut pixel = wasm_load_deinterleave_u8x4(src.as_ptr());
                pixel.0 = premultiply_vec(pixel.0, pixel.3);
                pixel.1 = premultiply_vec(pixel.1, pixel.3);
                pixel.2 = premultiply_vec(pixel.2, pixel.3);
                wasm_store_interleave_u8x4(dst.as_mut_ptr(), pixel);
            }

            rem = rem.chunks_exact_mut(16 * 4).into_remainder();
            src_rem = src_rem.chunks_exact(16 * 4).remainder();

            premultiply_alpha_rgba_row_impl(rem, src_rem);
        });
}
