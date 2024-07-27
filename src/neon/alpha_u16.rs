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
use crate::{premultiply_pixel_u16, unpremultiply_pixel_u16};
use std::arch::aarch64::*;

pub fn neon_premultiply_alpha_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
) {
    let mut offset = 0usize;

    let max_colors = 2i64.pow(bit_depth as u32) - 1;

    let v_max_colors_scale = unsafe { vdupq_n_f32((1. / max_colors as f64) as f32) };

    for _ in 0..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 8 < width {
                let px = _cx * 4;
                let pixel = vld4q_u16(src.as_ptr().add(offset + px));

                let low_a = vmovl_u16(vget_low_u16(pixel.3));
                let high_a = vmovl_high_u16(pixel.3);

                let low_a = vmulq_f32(vcvtq_f32_u32(low_a), v_max_colors_scale);
                let hi_a = vmulq_f32(vcvtq_f32_u32(high_a), v_max_colors_scale);

                let new_r = v_scale_by_alpha(pixel.0, low_a, hi_a);

                let new_g = v_scale_by_alpha(pixel.1, low_a, hi_a);

                let new_b = v_scale_by_alpha(pixel.2, low_a, hi_a);

                let new_px = uint16x8x4_t(new_r, new_g, new_b, pixel.3);

                vst4q_u16(dst.as_mut_ptr().add(offset + px), new_px);

                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            premultiply_pixel_u16!(dst, src, offset + px, max_colors);
        }

        offset += 4 * width;
    }
}

#[inline]
unsafe fn v_scale_by_alpha(
    px: uint16x8_t,
    low_low_a: float32x4_t,
    low_high_a: float32x4_t,
) -> uint16x8_t {
    let low_px_u = vmovl_u16(vget_low_u16(px));
    let high_px_u = vmovl_high_u16(px);

    let low_px = vcvtq_f32_u32(low_px_u);
    let high_px = vcvtq_f32_u32(high_px_u);

    let new_ll = vcvtaq_u32_f32(vmulq_f32(low_px, low_low_a));
    let new_lh = vcvtaq_u32_f32(vmulq_f32(high_px, low_high_a));

    vcombine_u16(vmovn_u32(new_ll), vmovn_u32(new_lh))
}

pub fn neon_unpremultiply_alpha_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
) {
    let mut offset = 0usize;

    let max_colors = 2i64.pow(bit_depth as u32) - 1;

    for _ in 0..height {
        let mut _cx = 0usize;

        unsafe {
            let v_max_colors_f = vdupq_n_f32(max_colors as f32);
            let ones = vdupq_n_f32(1.);
            while _cx + 8 < width {
                let px = _cx * 4;
                let pixel = vld4q_u16(src.as_ptr().add(offset + px));

                let is_alpha_zero_mask = vceqzq_u16(pixel.3);

                let low_a = vmovl_u16(vget_low_u16(pixel.3));
                let high_a = vmovl_high_u16(pixel.3);

                let low_a = vmulq_f32(vdivq_f32(ones, vcvtq_f32_u32(low_a)), v_max_colors_f);
                let hi_a = vmulq_f32(vdivq_f32(ones, vcvtq_f32_u32(high_a)), v_max_colors_f);

                let new_r = vbslq_u16(
                    is_alpha_zero_mask,
                    pixel.0,
                    v_scale_by_alpha(pixel.0, low_a, hi_a),
                );

                let new_g = vbslq_u16(
                    is_alpha_zero_mask,
                    pixel.1,
                    v_scale_by_alpha(pixel.1, low_a, hi_a),
                );

                let new_b = vbslq_u16(
                    is_alpha_zero_mask,
                    pixel.2,
                    v_scale_by_alpha(pixel.2, low_a, hi_a),
                );

                let new_px = uint16x8x4_t(new_r, new_g, new_b, pixel.3);

                vst4q_u16(dst.as_mut_ptr().add(offset + px), new_px);

                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel_u16!(dst, src, pixel_offset, max_colors);
        }

        offset += 4 * width;
    }
}
