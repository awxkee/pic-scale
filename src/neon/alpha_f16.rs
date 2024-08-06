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

use std::arch::aarch64::*;

use crate::neon::f16_utils::*;
use crate::{premultiply_pixel_f16, unpremultiply_pixel_f16};

#[cfg(target_feature = "fp16")]
pub fn neon_premultiply_alpha_rgba_f16_full(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;
    let src_stride = 4 * width;

    let mut offset = _cy * src_stride;

    for _ in _cy..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 8 < width {
                let px = _cx * 4;
                let src_ptr = src.as_ptr().add(offset + px);
                let pixel = vld4q_u16(src_ptr as *const u16);

                let low_alpha = xreinterpretq_f16_u16(pixel.3);
                let r_values = xvmulq_f16(xreinterpretq_f16_u16(pixel.0), low_alpha);
                let g_values = xvmulq_f16(xreinterpretq_f16_u16(pixel.1), low_alpha);
                let b_values = xvmulq_f16(xreinterpretq_f16_u16(pixel.2), low_alpha);

                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                let store_pixel = uint16x8x4_t(
                    xreinterpretq_u16_f16(r_values),
                    xreinterpretq_u16_f16(g_values),
                    xreinterpretq_u16_f16(b_values),
                    pixel.3,
                );
                vst4q_u16(dst_ptr as *mut u16, store_pixel);
                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            premultiply_pixel_f16!(dst, src, offset + px);
        }

        offset += 4 * width;
    }
}

pub fn neon_premultiply_alpha_rgba_f16(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;
    let src_stride = 4 * width;

    let mut offset = _cy * src_stride;

    for _ in _cy..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 8 < width {
                let px = _cx * 4;
                let src_ptr = src.as_ptr().add(offset + px);
                let pixel = vld4q_u16(src_ptr as *const u16);

                let low_alpha = xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.3)));
                let low_r = vmulq_f32(
                    xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.0))),
                    low_alpha,
                );
                let low_g = vmulq_f32(
                    xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.1))),
                    low_alpha,
                );
                let low_b = vmulq_f32(
                    xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.2))),
                    low_alpha,
                );

                let high_alpha = xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.3)));
                let high_r = vmulq_f32(
                    xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.0))),
                    high_alpha,
                );
                let high_g = vmulq_f32(
                    xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.1))),
                    high_alpha,
                );
                let high_b = vmulq_f32(
                    xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.2))),
                    high_alpha,
                );
                let r_values = xreinterpretq_u16_f16(xcombine_f16(
                    xvcvt_f16_f32(low_r),
                    xvcvt_f16_f32(high_r),
                ));
                let g_values = xreinterpretq_u16_f16(xcombine_f16(
                    xvcvt_f16_f32(low_g),
                    xvcvt_f16_f32(high_g),
                ));
                let b_values = xreinterpretq_u16_f16(xcombine_f16(
                    xvcvt_f16_f32(low_b),
                    xvcvt_f16_f32(high_b),
                ));
                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                let store_pixel = uint16x8x4_t(r_values, g_values, b_values, pixel.3);
                vst4q_u16(dst_ptr as *mut u16, store_pixel);
                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            premultiply_pixel_f16!(dst, src, offset + px);
        }

        offset += 4 * width;
    }
}

pub fn neon_unpremultiply_alpha_rgba_f16(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;

    let mut offset = 0usize;
    offset += _cy * width * 4;

    for _ in _cy..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 8 < width {
                let px = _cx * 4;
                let pixel_offset = offset + px;
                let src_ptr = src.as_ptr().add(pixel_offset);
                let pixel = vld4q_u16(src_ptr as *const u16);

                let low_alpha = xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.3)));
                let low_zero_mask = vceqzq_f32(low_alpha);
                let zeros = vdupq_n_f32(0.);

                let low_r = vbslq_f32(
                    low_zero_mask,
                    zeros,
                    vdivq_f32(
                        xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.0))),
                        low_alpha,
                    ),
                );
                let low_g = vbslq_f32(
                    low_zero_mask,
                    zeros,
                    vdivq_f32(
                        xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.1))),
                        low_alpha,
                    ),
                );
                let low_b = vbslq_f32(
                    low_zero_mask,
                    zeros,
                    vdivq_f32(
                        xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.2))),
                        low_alpha,
                    ),
                );

                let high_alpha = xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.3)));
                let high_zero_mask = vceqzq_f32(high_alpha);

                let high_r = vbslq_f32(
                    high_zero_mask,
                    zeros,
                    vdivq_f32(
                        xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.0))),
                        high_alpha,
                    ),
                );
                let high_g = vbslq_f32(
                    high_zero_mask,
                    zeros,
                    vdivq_f32(
                        xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.1))),
                        high_alpha,
                    ),
                );
                let high_b = vbslq_f32(
                    high_zero_mask,
                    zeros,
                    vdivq_f32(
                        xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.2))),
                        high_alpha,
                    ),
                );

                let r_values = xreinterpretq_u16_f16(xcombine_f16(
                    xvcvt_f16_f32(low_r),
                    xvcvt_f16_f32(high_r),
                ));
                let g_values = xreinterpretq_u16_f16(xcombine_f16(
                    xvcvt_f16_f32(low_g),
                    xvcvt_f16_f32(high_g),
                ));
                let b_values = xreinterpretq_u16_f16(xcombine_f16(
                    xvcvt_f16_f32(low_b),
                    xvcvt_f16_f32(high_b),
                ));

                let dst_ptr = dst.as_mut_ptr().add(pixel_offset);
                let store_pixel = uint16x8x4_t(r_values, g_values, b_values, pixel.3);
                vst4q_u16(dst_ptr as *mut u16, store_pixel);
                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel_f16!(dst, src, pixel_offset);
        }

        offset += 4 * width;
    }
}

#[cfg(target_feature = "fp16")]
pub fn neon_unpremultiply_alpha_rgba_f16_full(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;

    let mut offset = 0usize;
    offset += _cy * width * 4;

    for _ in _cy..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 8 < width {
                let px = _cx * 4;
                let pixel_offset = offset + px;
                let src_ptr = src.as_ptr().add(pixel_offset);
                let pixel = vld4q_u16(src_ptr as *const u16);

                let alphas = xreinterpretq_f16_u16(pixel.3);
                let zero_mask = vceqzq_f16(alphas);

                let r_values = xvbslq_f16(
                    zero_mask,
                    xreinterpretq_f16_u16(pixel.0),
                    xvdivq_f16(xreinterpretq_f16_u16(pixel.0), alphas),
                );
                let g_values = xvbslq_f16(
                    zero_mask,
                    xreinterpretq_f16_u16(pixel.1),
                    xvdivq_f16(xreinterpretq_f16_u16(pixel.1), alphas),
                );
                let b_values = xvbslq_f16(
                    zero_mask,
                    xreinterpretq_f16_u16(pixel.2),
                    xvdivq_f16(xreinterpretq_f16_u16(pixel.2), alphas),
                );

                let dst_ptr = dst.as_mut_ptr().add(pixel_offset);
                let store_pixel = uint16x8x4_t(
                    xreinterpretq_u16_f16(r_values),
                    xreinterpretq_u16_f16(g_values),
                    xreinterpretq_u16_f16(b_values),
                    pixel.3,
                );
                vst4q_u16(dst_ptr as *mut u16, store_pixel);
                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel_f16!(dst, src, pixel_offset);
        }

        offset += 4 * width;
    }
}
