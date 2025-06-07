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

use crate::alpha_handle_f16::premultiply_pixel_f16_row;
use core::f16;
use novtb::{ParallelZonedIterator, TbSliceMut};
use std::arch::aarch64::*;

#[target_feature(enable = "fp16")]
unsafe fn neon_premultiply_alpha_rgba_row_f16_full(dst: &mut [f16], src: &[f16]) {
    unsafe {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
            let src_ptr = src.as_ptr();
            let pixel = vld4q_u16(src_ptr as *const u16);

            let low_alpha = vreinterpretq_f16_u16(pixel.3);
            let r_values = vmulq_f16(vreinterpretq_f16_u16(pixel.0), low_alpha);
            let g_values = vmulq_f16(vreinterpretq_f16_u16(pixel.1), low_alpha);
            let b_values = vmulq_f16(vreinterpretq_f16_u16(pixel.2), low_alpha);

            let dst_ptr = dst.as_mut_ptr();
            let store_pixel = uint16x8x4_t(
                vreinterpretq_u16_f16(r_values),
                vreinterpretq_u16_f16(g_values),
                vreinterpretq_u16_f16(b_values),
                pixel.3,
            );
            vst4q_u16(dst_ptr as *mut u16, store_pixel);
        }

        rem = rem.chunks_exact_mut(8 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(8 * 4).remainder();

        premultiply_pixel_f16_row(rem, src_rem);

        if !rem.is_empty() {
            let mut transient: [f16; 4 * 8] = [0.; 4 * 8];
            assert_eq!(rem.len(), src_rem.len());
            assert!(rem.len() <= 4 * 8);

            std::ptr::copy_nonoverlapping(src_rem.as_ptr(), transient.as_mut_ptr(), src_rem.len());

            let pixel = vld4q_u16(transient.as_ptr() as *const u16);

            let low_alpha = vreinterpretq_f16_u16(pixel.3);
            let r_values = vmulq_f16(vreinterpretq_f16_u16(pixel.0), low_alpha);
            let g_values = vmulq_f16(vreinterpretq_f16_u16(pixel.1), low_alpha);
            let b_values = vmulq_f16(vreinterpretq_f16_u16(pixel.2), low_alpha);

            let store_pixel = uint16x8x4_t(
                vreinterpretq_u16_f16(r_values),
                vreinterpretq_u16_f16(g_values),
                vreinterpretq_u16_f16(b_values),
                pixel.3,
            );
            vst4q_u16(transient.as_mut_ptr() as *mut u16, store_pixel);

            std::ptr::copy_nonoverlapping(transient.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

pub(crate) fn neon_premultiply_alpha_rgba_f16_full(
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
            neon_premultiply_alpha_rgba_row_f16_full(&mut dst[..width * 4], &src[..width * 4]);
        });
}

#[target_feature(enable = "fp16")]
unsafe fn neon_unpremultiply_alpha_rgba_f16_row_full(in_place: &mut [f16]) {
    unsafe {
        let mut rem = in_place;

        for dst in rem.chunks_exact_mut(8 * 4) {
            let src_ptr = dst.as_ptr();
            let pixel = vld4q_u16(src_ptr as *const u16);

            let alphas = vreinterpretq_f16_u16(pixel.3);
            let zero_mask = vceqzq_f16(alphas);

            let r_values = vbslq_f16(
                zero_mask,
                vreinterpretq_f16_u16(pixel.0),
                vdivq_f16(vreinterpretq_f16_u16(pixel.0), alphas),
            );
            let g_values = vbslq_f16(
                zero_mask,
                vreinterpretq_f16_u16(pixel.1),
                vdivq_f16(vreinterpretq_f16_u16(pixel.1), alphas),
            );
            let b_values = vbslq_f16(
                zero_mask,
                vreinterpretq_f16_u16(pixel.2),
                vdivq_f16(vreinterpretq_f16_u16(pixel.2), alphas),
            );

            let dst_ptr = dst.as_mut_ptr();
            let store_pixel = uint16x8x4_t(
                vreinterpretq_u16_f16(r_values),
                vreinterpretq_u16_f16(g_values),
                vreinterpretq_u16_f16(b_values),
                pixel.3,
            );
            vst4q_u16(dst_ptr as *mut u16, store_pixel);
        }

        rem = rem.chunks_exact_mut(8 * 4).into_remainder();
        if !rem.is_empty() {
            let mut transient: [f16; 4 * 8] = [0.; 4 * 8];
            assert!(rem.len() <= 4 * 8);
            std::ptr::copy_nonoverlapping(rem.as_ptr(), transient.as_mut_ptr(), rem.len());

            let pixel = vld4q_u16(transient.as_ptr() as *const u16);

            let alphas = vreinterpretq_f16_u16(pixel.3);
            let zero_mask = vceqzq_f16(alphas);

            let r_values = vbslq_f16(
                zero_mask,
                vreinterpretq_f16_u16(pixel.0),
                vdivq_f16(vreinterpretq_f16_u16(pixel.0), alphas),
            );
            let g_values = vbslq_f16(
                zero_mask,
                vreinterpretq_f16_u16(pixel.1),
                vdivq_f16(vreinterpretq_f16_u16(pixel.1), alphas),
            );
            let b_values = vbslq_f16(
                zero_mask,
                vreinterpretq_f16_u16(pixel.2),
                vdivq_f16(vreinterpretq_f16_u16(pixel.2), alphas),
            );

            let store_pixel = uint16x8x4_t(
                vreinterpretq_u16_f16(r_values),
                vreinterpretq_u16_f16(g_values),
                vreinterpretq_u16_f16(b_values),
                pixel.3,
            );
            vst4q_u16(transient.as_mut_ptr() as *mut u16, store_pixel);

            std::ptr::copy_nonoverlapping(transient.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

pub(crate) fn neon_unpremultiply_alpha_rgba_f16_full(
    in_place: &mut [f16],
    stride: usize,
    width: usize,
    _: usize,
    pool: &novtb::ThreadPool,
) {
    in_place
        .tb_par_chunks_exact_mut(stride)
        .for_each(pool, |row| unsafe {
            neon_unpremultiply_alpha_rgba_f16_row_full(&mut row[..width * 4]);
        });
}
