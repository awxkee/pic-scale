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

use crate::{premultiply_pixel_f32, unpremultiply_pixel_f32};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
use std::arch::aarch64::*;

macro_rules! unpremultiply_vec_f32 {
    ($v: expr, $a_values: expr) => {{
        let is_zero_mask = vceqzq_f32($a_values);
        let rs = vdivq_f32($v, $a_values);
        vbslq_f32(is_zero_mask, vdupq_n_f32(0.), rs)
    }};
}

unsafe fn neon_premultiply_alpha_rgba_row_f32(
    dst: &mut [f32],
    src: &[f32],
    width: usize,
    offset: usize,
) {
    let mut _cx = 0usize;

    unsafe {
        while _cx + 4 < width {
            let px = _cx * 4;
            let src_ptr = src.as_ptr().add(offset + px);
            let mut pixel = vld4q_f32(src_ptr);
            pixel.0 = vmulq_f32(pixel.0, pixel.3);
            pixel.1 = vmulq_f32(pixel.1, pixel.3);
            pixel.2 = vmulq_f32(pixel.2, pixel.3);
            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            vst4q_f32(dst_ptr, pixel);
            _cx += 4;
        }
    }

    for x in _cx..width {
        let px = x * 4;
        premultiply_pixel_f32!(dst, src, offset + px);
    }
}

pub fn neon_premultiply_alpha_rgba_f32(
    dst: &mut [f32],
    src: &[f32],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| unsafe {
                    neon_premultiply_alpha_rgba_row_f32(dst, src, width, 0);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(4 * width)
            .zip(src.chunks_exact(4 * width))
        {
            unsafe {
                neon_premultiply_alpha_rgba_row_f32(dst_row, src_row, width, 0);
            }
        }
    }
}

unsafe fn neon_unpremultiply_alpha_rgba_f32_row(
    dst: &mut [f32],
    src: &[f32],
    width: usize,
    offset: usize,
) {
    let mut _cx = 0usize;

    unsafe {
        while _cx + 4 < width {
            let px = _cx * 4;
            let pixel_offset = offset + px;
            let src_ptr = src.as_ptr().add(pixel_offset);
            let mut pixel = vld4q_f32(src_ptr);
            pixel.0 = unpremultiply_vec_f32!(pixel.0, pixel.3);
            pixel.1 = unpremultiply_vec_f32!(pixel.1, pixel.3);
            pixel.2 = unpremultiply_vec_f32!(pixel.2, pixel.3);
            let dst_ptr = dst.as_mut_ptr().add(pixel_offset);
            vst4q_f32(dst_ptr, pixel);
            _cx += 4;
        }
    }

    for x in _cx..width {
        let px = x * 4;
        let pixel_offset = offset + px;
        unpremultiply_pixel_f32!(dst, src, pixel_offset);
    }
}

pub fn neon_unpremultiply_alpha_rgba_f32(
    dst: &mut [f32],
    src: &[f32],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| unsafe {
                    neon_unpremultiply_alpha_rgba_f32_row(dst, src, width, 0);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(4 * width)
            .zip(src.chunks_exact(4 * width))
        {
            unsafe {
                neon_unpremultiply_alpha_rgba_f32_row(dst_row, src_row, width, 0);
            }
        }
    }
}
