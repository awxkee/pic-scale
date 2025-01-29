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

use crate::alpha_handle_f16::premultiply_pixel_f16_row;
use crate::neon::f16_utils::*;
use core::f16;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;

#[target_feature(enable = "fp16")]
unsafe fn neon_premultiply_alpha_rgba_row_f16_full(dst: &mut [f16], src: &[f16]) {
    let mut rem = dst;
    let mut src_rem = src;

    for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
        let src_ptr = src.as_ptr();
        let pixel = vld4q_u16(src_ptr as *const u16);

        let low_alpha = xreinterpretq_f16_u16(pixel.3);
        let r_values = xvmulq_f16(xreinterpretq_f16_u16(pixel.0), low_alpha);
        let g_values = xvmulq_f16(xreinterpretq_f16_u16(pixel.1), low_alpha);
        let b_values = xvmulq_f16(xreinterpretq_f16_u16(pixel.2), low_alpha);

        let dst_ptr = dst.as_mut_ptr();
        let store_pixel = uint16x8x4_t(
            xreinterpretq_u16_f16(r_values),
            xreinterpretq_u16_f16(g_values),
            xreinterpretq_u16_f16(b_values),
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

        let low_alpha = xreinterpretq_f16_u16(pixel.3);
        let r_values = xvmulq_f16(xreinterpretq_f16_u16(pixel.0), low_alpha);
        let g_values = xvmulq_f16(xreinterpretq_f16_u16(pixel.1), low_alpha);
        let b_values = xvmulq_f16(xreinterpretq_f16_u16(pixel.2), low_alpha);

        let store_pixel = uint16x8x4_t(
            xreinterpretq_u16_f16(r_values),
            xreinterpretq_u16_f16(g_values),
            xreinterpretq_u16_f16(b_values),
            pixel.3,
        );
        vst4q_u16(transient.as_mut_ptr() as *mut u16, store_pixel);

        std::ptr::copy_nonoverlapping(transient.as_ptr(), rem.as_mut_ptr(), rem.len());
    }
}

pub(crate) fn neon_premultiply_alpha_rgba_f16_full(
    dst: &mut [f16],
    dst_stride: usize,
    src: &[f16],
    src_stride: usize,
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(dst_stride)
                .zip(src.par_chunks_exact(src_stride))
                .for_each(|(dst, src)| unsafe {
                    neon_premultiply_alpha_rgba_row_f16_full(
                        &mut dst[..width * 4],
                        &src[..width * 4],
                    );
                });
        });
    } else {
        dst.chunks_exact_mut(dst_stride)
            .zip(src.chunks_exact(src_stride))
            .for_each(|(dst, src)| unsafe {
                neon_premultiply_alpha_rgba_row_f16_full(&mut dst[..width * 4], &src[..width * 4]);
            });
    }
}

#[target_feature(enable = "fp16")]
unsafe fn neon_unpremultiply_alpha_rgba_f16_row_full(in_place: &mut [f16]) {
    let mut rem = in_place;

    for dst in rem.chunks_exact_mut(8 * 4) {
        let src_ptr = dst.as_ptr();
        let pixel = vld4q_u16(src_ptr as *const u16);

        let alphas = xreinterpretq_f16_u16(pixel.3);
        let zero_mask = xvceqzq_f16(alphas);

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

        let dst_ptr = dst.as_mut_ptr();
        let store_pixel = uint16x8x4_t(
            xreinterpretq_u16_f16(r_values),
            xreinterpretq_u16_f16(g_values),
            xreinterpretq_u16_f16(b_values),
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

        let alphas = xreinterpretq_f16_u16(pixel.3);
        let zero_mask = xvceqzq_f16(alphas);

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

        let store_pixel = uint16x8x4_t(
            xreinterpretq_u16_f16(r_values),
            xreinterpretq_u16_f16(g_values),
            xreinterpretq_u16_f16(b_values),
            pixel.3,
        );
        vst4q_u16(transient.as_mut_ptr() as *mut u16, store_pixel);

        std::ptr::copy_nonoverlapping(transient.as_ptr(), rem.as_mut_ptr(), rem.len());
    }
}

pub(crate) fn neon_unpremultiply_alpha_rgba_f16_full(
    in_place: &mut [f16],
    stride: usize,
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            in_place
                .par_chunks_exact_mut(stride)
                .for_each(|row| unsafe {
                    neon_unpremultiply_alpha_rgba_f16_row_full(&mut row[..width * 4]);
                });
        });
    } else {
        in_place.chunks_exact_mut(stride).for_each(|row| unsafe {
            neon_unpremultiply_alpha_rgba_f16_row_full(&mut row[..width * 4]);
        });
    }
}
