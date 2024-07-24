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

use crate::{premultiply_pixel, unpremultiply_pixel};
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_div_by_255(v: uint16x8_t) -> uint16x8_t {
    let addition = vdupq_n_u16(127);
    vshrq_n_u16::<8>(vaddq_u16(vaddq_u16(v, addition), vshrq_n_u16::<8>(v)))
}

macro_rules! premultiply_vec {
    ($v: expr, $a_values: expr) => {{
        let acc_hi = vmull_high_u8($v, $a_values);
        let acc_lo = vmull_u8(vget_low_u8($v), vget_low_u8($a_values));
        let hi = vqmovn_u16(neon_div_by_255(acc_hi));
        let lo = vqmovn_u16(neon_div_by_255(acc_lo));
        vcombine_u8(lo, hi)
    }};
}

macro_rules! unpremultiply_vec {
    ($v: expr, $a_values: expr) => {{
        let scale = vdupq_n_u8(255);
        let hi = vmull_high_u8($v, scale);
        let lo = vmull_u8(vget_low_u8($v), vget_low_u8(scale));
        let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
        let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo));
        let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
        let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi));
        let zero_mask = vmvnq_u8(vceqzq_u8($a_values));
        let a_hi = vmovl_high_u8($a_values);
        let a_lo = vmovl_u8(vget_low_u8($a_values));
        let a_lo_lo = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_lo))));
        let a_lo_hi = vrecpeq_f32(vcvtq_f32_u32(vmovl_high_u16(a_lo)));
        let a_hi_lo = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_hi))));
        let a_hi_ho = vrecpeq_f32(vcvtq_f32_u32(vmovl_high_u16(a_hi)));

        let lo_lo = vcvtaq_u32_f32(vmulq_f32(lo_lo, a_lo_lo));
        let lo_hi = vcvtaq_u32_f32(vmulq_f32(lo_hi, a_lo_hi));
        let hi_lo = vcvtaq_u32_f32(vmulq_f32(hi_lo, a_hi_lo));
        let hi_hi = vcvtaq_u32_f32(vmulq_f32(hi_hi, a_hi_ho));
        let lo = vcombine_u16(vmovn_u32(lo_lo), vmovn_u32(lo_hi));
        let hi = vcombine_u16(vmovn_u32(hi_lo), vmovn_u32(hi_hi));
        vandq_u8(vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi)), zero_mask)
    }};
}

pub fn neon_premultiply_alpha_rgba(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
    let mut _cy = 0usize;
    let src_stride = 4 * width;

    let mut offset = _cy * src_stride;

    for _ in _cy..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 64 < width {
                let px = _cx * 4;
                let src_ptr = src.as_ptr().add(offset + px);
                let mut pixel0 = vld4q_u8(src_ptr);
                let mut pixel1 = vld4q_u8(src_ptr.add(16 * 4));
                let mut pixel2 = vld4q_u8(src_ptr.add(16 * 4 * 2));
                let mut pixel3 = vld4q_u8(src_ptr.add(16 * 4 * 3));
                pixel0.0 = premultiply_vec!(pixel0.0, pixel0.3);
                pixel0.1 = premultiply_vec!(pixel0.1, pixel0.3);
                pixel0.2 = premultiply_vec!(pixel0.2, pixel0.3);

                pixel1.0 = premultiply_vec!(pixel1.0, pixel1.3);
                pixel1.1 = premultiply_vec!(pixel1.1, pixel1.3);
                pixel1.2 = premultiply_vec!(pixel1.2, pixel1.3);

                pixel2.0 = premultiply_vec!(pixel2.0, pixel2.3);
                pixel2.1 = premultiply_vec!(pixel2.1, pixel2.3);
                pixel2.2 = premultiply_vec!(pixel2.2, pixel2.3);

                pixel3.0 = premultiply_vec!(pixel3.0, pixel3.3);
                pixel3.1 = premultiply_vec!(pixel3.1, pixel3.3);
                pixel3.2 = premultiply_vec!(pixel3.2, pixel3.3);
                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                vst4q_u8(dst_ptr, pixel0);
                vst4q_u8(dst_ptr.add(16 * 4), pixel1);
                vst4q_u8(dst_ptr.add(16 * 4 * 2), pixel2);
                vst4q_u8(dst_ptr.add(16 * 4 * 3), pixel3);
                _cx += 64;
            }

            while _cx + 16 < width {
                let px = _cx * 4;
                let src_ptr = src.as_ptr().add(offset + px);
                let mut pixel = vld4q_u8(src_ptr);
                pixel.0 = premultiply_vec!(pixel.0, pixel.3);
                pixel.1 = premultiply_vec!(pixel.1, pixel.3);
                pixel.2 = premultiply_vec!(pixel.2, pixel.3);
                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                vst4q_u8(dst_ptr, pixel);
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

pub fn neon_unpremultiply_alpha_rgba(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
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
                let mut pixel = vld4q_u8(src_ptr);
                pixel.0 = unpremultiply_vec!(pixel.0, pixel.3);
                pixel.1 = unpremultiply_vec!(pixel.1, pixel.3);
                pixel.2 = unpremultiply_vec!(pixel.2, pixel.3);
                let dst_ptr = dst.as_mut_ptr().add(pixel_offset);
                vst4q_u8(dst_ptr, pixel);
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
