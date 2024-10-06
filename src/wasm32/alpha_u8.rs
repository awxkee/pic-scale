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
use crate::wasm32::transpose::{wasm_load_deinterleave_u8x4, wasm_store_interleave_u8x4};
use crate::wasm32::utils::*;
use crate::{premultiply_pixel, unpremultiply_pixel, ThreadingPolicy};
use std::arch::wasm32::*;

pub fn wasm_unpremultiply_alpha_rgba(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    _: ThreadingPolicy,
) {
    unsafe {
        wasm_unpremultiply_alpha_rgba_impl(dst, src, width, height);
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
pub unsafe fn wasm_u16x8_div_by_255(v: v128) -> v128 {
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
unsafe fn wasm_unpremultiply_alpha_rgba_impl(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;

    let mut offset = 0usize;
    offset += _cy * width * 4;

    for _ in _cy..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 16 < width {
                let px = _cx * 4;
                let pixel_offset = offset + px;
                let src_ptr = src.as_ptr().add(pixel_offset);
                let mut pixel = wasm_load_deinterleave_u8x4(src_ptr);

                pixel.0 = unpremultiply_vec(pixel.0, pixel.3);
                pixel.1 = unpremultiply_vec(pixel.1, pixel.3);
                pixel.2 = unpremultiply_vec(pixel.2, pixel.3);
                let dst_ptr = dst.as_mut_ptr().add(pixel_offset);
                wasm_store_interleave_u8x4(dst_ptr, pixel);
                _cx += 16;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel!(dst, src, pixel_offset);
        }

        offset += 4 * width;
    }
}

pub fn wasm_premultiply_alpha_rgba(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    _: ThreadingPolicy,
) {
    unsafe {
        wasm_premultiply_alpha_rgba_impl(dst, src, width, height);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn wasm_premultiply_alpha_rgba_impl(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;
    let src_stride = 4 * width;

    let mut offset = _cy * src_stride;

    for _ in _cy..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 16 < width {
                let px = _cx * 4;
                let src_ptr = src.as_ptr().add(offset + px);
                let mut pixel = wasm_load_deinterleave_u8x4(src_ptr);
                pixel.0 = premultiply_vec(pixel.0, pixel.3);
                pixel.1 = premultiply_vec(pixel.1, pixel.3);
                pixel.2 = premultiply_vec(pixel.2, pixel.3);
                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                wasm_store_interleave_u8x4(dst_ptr, pixel);
                _cx += 16;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            premultiply_pixel!(dst, src, offset + px);
        }

        offset += 4 * width;
    }
}
