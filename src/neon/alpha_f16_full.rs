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
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;
use rayon::ThreadPool;

#[target_feature(enable = "fp16")]
unsafe fn neon_premultiply_alpha_rgba_row_f16_full(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    offset: usize,
) {
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
}

pub fn neon_premultiply_alpha_rgba_f16_full(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| unsafe {
                    neon_premultiply_alpha_rgba_row_f16_full(dst, src, width, 0);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(4 * width)
            .zip(src.chunks_exact(4 * width))
        {
            unsafe {
                neon_premultiply_alpha_rgba_row_f16_full(dst_row, src_row, width, 0);
            }
        }
    }
}

#[target_feature(enable = "fp16")]
unsafe fn neon_unpremultiply_alpha_rgba_f16_row_full(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    offset: usize,
) {
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
}

pub fn neon_unpremultiply_alpha_rgba_f16_full(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| unsafe {
                    neon_unpremultiply_alpha_rgba_f16_row_full(dst, src, width, 0);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(4 * width)
            .zip(src.chunks_exact(4 * width))
        {
            unsafe {
                neon_unpremultiply_alpha_rgba_f16_row_full(dst_row, src_row, width, 0);
            }
        }
    }
}
