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

use crate::alpha_handle_f16::{premultiply_pixel_f16_row, unpremultiply_pixel_f16_row};
use core::f16;
use novtb::{ParallelZonedIterator, TbSliceMut};
use std::arch::aarch64::*;

unsafe fn neon_premultiply_alpha_rgba_row_f16(dst: &mut [f16], src: &[f16]) {
    unsafe {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
            let src_ptr = src.as_ptr();
            let pixel = vld4q_u16(src_ptr as *const u16);

            let low_alpha = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.3)));
            let low_r = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.0))),
                low_alpha,
            );
            let low_g = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.1))),
                low_alpha,
            );
            let low_b = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.2))),
                low_alpha,
            );

            let high_alpha = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.3)));
            let high_r = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.0))),
                high_alpha,
            );
            let high_g = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.1))),
                high_alpha,
            );
            let high_b = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.2))),
                high_alpha,
            );
            let r_values = vcombine_u16(
                vreinterpret_u16_f16(vcvt_f16_f32(low_r)),
                vreinterpret_u16_f16(vcvt_f16_f32(high_r)),
            );
            let g_values = vcombine_u16(
                vreinterpret_u16_f16(vcvt_f16_f32(low_g)),
                vreinterpret_u16_f16(vcvt_f16_f32(high_g)),
            );
            let b_values = vcombine_u16(
                vreinterpret_u16_f16(vcvt_f16_f32(low_b)),
                vreinterpret_u16_f16(vcvt_f16_f32(high_b)),
            );

            let dst_ptr = dst.as_mut_ptr();
            let store_pixel = uint16x8x4_t(r_values, g_values, b_values, pixel.3);
            vst4q_u16(dst_ptr as *mut u16, store_pixel);
        }

        rem = rem.chunks_exact_mut(8 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(8 * 4).remainder();

        premultiply_pixel_f16_row(rem, src_rem);
    }
}

pub(crate) fn neon_premultiply_alpha_rgba_f16(
    dst: &mut [f16],
    dst_stride: usize,
    src: &[f16],
    src_stride: usize,
    width: usize,
    _: usize,
    pool: &novtb::ThreadPool,
) {
    dst.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| unsafe {
            let src = &src[y * src_stride..(y + 1) * src_stride];
            neon_premultiply_alpha_rgba_row_f16(&mut dst[..width * 4], &src[..width * 4]);
        });
}

unsafe fn neon_unpremultiply_alpha_rgba_row_f16(in_place: &mut [f16]) {
    unsafe {
        let mut rem = in_place;

        for dst in rem.chunks_exact_mut(8 * 4) {
            let src_ptr = dst.as_ptr();
            let pixel = vld4q_u16(src_ptr as *const u16);

            let zero_mask = vceqzq_u16(pixel.3);

            let low_alpha = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.3)));

            let low_r = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.0))),
                low_alpha,
            );
            let low_g = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.1))),
                low_alpha,
            );
            let low_b = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.2))),
                low_alpha,
            );

            let high_alpha = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.3)));

            let high_r = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.0))),
                high_alpha,
            );
            let high_g = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.1))),
                high_alpha,
            );
            let high_b = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.2))),
                high_alpha,
            );

            let u_zeros = vdupq_n_u16(0);

            let r_values = vbslq_u16(
                zero_mask,
                u_zeros,
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(low_r)),
                    vreinterpret_u16_f16(vcvt_f16_f32(high_r)),
                ),
            );
            let g_values = vbslq_u16(
                zero_mask,
                u_zeros,
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(low_g)),
                    vreinterpret_u16_f16(vcvt_f16_f32(high_g)),
                ),
            );
            let b_values = vbslq_u16(
                zero_mask,
                u_zeros,
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(low_b)),
                    vreinterpret_u16_f16(vcvt_f16_f32(high_b)),
                ),
            );

            let dst_ptr = dst.as_mut_ptr();
            let store_pixel = uint16x8x4_t(r_values, g_values, b_values, pixel.3);
            vst4q_u16(dst_ptr as *mut u16, store_pixel);
        }

        rem = rem.chunks_exact_mut(8 * 4).into_remainder();

        unpremultiply_pixel_f16_row(rem);
    }
}

pub(crate) fn neon_unpremultiply_alpha_rgba_f16(
    in_place: &mut [f16],
    stride: usize,
    width: usize,
    _: usize,
    pool: &novtb::ThreadPool,
) {
    in_place
        .tb_par_chunks_exact_mut(stride)
        .for_each(pool, |row| unsafe {
            neon_unpremultiply_alpha_rgba_row_f16(&mut row[..width * 4]);
        });
}
