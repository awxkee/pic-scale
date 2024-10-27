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
 *
 */
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx2::{avx_premultiply_alpha_rgba_u16, avx_unpremultiply_alpha_rgba_u16};
#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::neon::{neon_premultiply_alpha_rgba_u16, neon_unpremultiply_alpha_rgba_u16};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{premultiply_alpha_sse_rgba_u16, unpremultiply_alpha_sse_rgba_u16};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;

#[macro_export]
macro_rules! unpremultiply_pixel_u16 {
    ($dst: expr, $src: expr, $pixel_offset: expr, $max_colors: expr) => {{
        let mut r = *unsafe { $src.get_unchecked($pixel_offset) } as u32;
        let mut g = *unsafe { $src.get_unchecked($pixel_offset + 1) } as u32;
        let mut b = *unsafe { $src.get_unchecked($pixel_offset + 2) } as u32;
        let a = *unsafe { $src.get_unchecked($pixel_offset + 3) } as u32;
        if a != 0 {
            r = ((r * $max_colors) / a);
            g = ((g * $max_colors) / a);
            b = ((b * $max_colors) / a);
        } else {
            r = 0;
            g = 0;
            b = 0;
        }
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = r as u16;
            *$dst.get_unchecked_mut($pixel_offset + 1) = g as u16;
            *$dst.get_unchecked_mut($pixel_offset + 2) = b as u16;
            *$dst.get_unchecked_mut($pixel_offset + 3) = a as u16;
        }
    }};
}

#[macro_export]
macro_rules! premultiply_pixel_u16 {
    ($dst: expr, $src: expr, $pixel_offset: expr, $max_colors: expr) => {{
        let mut r = *unsafe { $src.get_unchecked($pixel_offset) } as u32;
        let mut g = *unsafe { $src.get_unchecked($pixel_offset + 1) } as u32;
        let mut b = *unsafe { $src.get_unchecked($pixel_offset + 2) } as u32;
        let a = *unsafe { $src.get_unchecked($pixel_offset + 3) } as u32;
        r *= a;
        g *= a;
        b *= a;
        r /= $max_colors;
        g /= $max_colors;
        b /= $max_colors;
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = r as u16;
            *$dst.get_unchecked_mut($pixel_offset + 1) = g as u16;
            *$dst.get_unchecked_mut($pixel_offset + 2) = b as u16;
            *$dst.get_unchecked_mut($pixel_offset + 3) = a as u16;
        }
    }};
}

fn premultiply_alpha_rgba_row(dst: &mut [u16], src: &[u16], max_colors: u32) {
    let recip_max_colors = 1. / max_colors as f32;
    for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        let a = src[3] as u32;
        dst[0] = (((src[0] as u32 * a) as f32 * recip_max_colors) as u32).min(max_colors) as u16;
        dst[1] = (((src[1] as u32 * a) as f32 * recip_max_colors) as u32).min(max_colors) as u16;
        dst[2] = (((src[2] as u32 * a) as f32 * recip_max_colors) as u32).min(max_colors) as u16;
        dst[3] = (((a * a) as f32 * recip_max_colors) as u32).min(max_colors) as u16;
    }
}

fn unpremultiply_alpha_rgba_row(dst: &mut [u16], src: &[u16], max_colors: u32) {
    for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        let a = src[3] as u32;
        if a != 0 {
            let a_recip = 1. / a as f32;
            dst[0] = ((src[0] as u32 * max_colors) as f32 * a_recip) as u16;
            dst[1] = ((src[1] as u32 * max_colors) as f32 * a_recip) as u16;
            dst[2] = ((src[2] as u32 * max_colors) as f32 * a_recip) as u16;
            dst[3] = ((a * max_colors) as f32 * a_recip) as u16;
        }
    }
}

fn premultiply_alpha_rgba_impl(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    _: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    let max_colors = (1 << bit_depth) - 1;
    if let Some(pool) = pool {
        pool.install(|| {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| {
                    premultiply_alpha_rgba_row(dst, src, max_colors);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(width * 4)
            .zip(src.chunks_exact(4 * width))
        {
            premultiply_alpha_rgba_row(dst_row, src_row, max_colors);
        }
    }
}

fn unpremultiply_alpha_rgba_impl(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    _: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    let max_colors = (1 << bit_depth) - 1;
    if let Some(pool) = pool {
        pool.install(|| {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| {
                    unpremultiply_alpha_rgba_row(dst, src, max_colors);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(width * 4)
            .zip(src.chunks_exact(4 * width))
        {
            unpremultiply_alpha_rgba_row(dst_row, src_row, max_colors);
        }
    }
}

pub fn premultiply_alpha_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    #[allow(clippy::type_complexity)]
    let mut _dispatcher: fn(&mut [u16], &[u16], usize, usize, usize, &Option<ThreadPool>) =
        premultiply_alpha_rgba_impl;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            _dispatcher = premultiply_alpha_sse_rgba_u16;
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            _dispatcher = avx_premultiply_alpha_rgba_u16;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_premultiply_alpha_rgba_u16;
    }
    _dispatcher(dst, src, width, height, bit_depth, pool);
}

pub fn unpremultiply_alpha_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    #[allow(clippy::type_complexity)]
    let mut _dispatcher: fn(&mut [u16], &[u16], usize, usize, usize, &Option<ThreadPool>) =
        unpremultiply_alpha_rgba_impl;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            _dispatcher = unpremultiply_alpha_sse_rgba_u16;
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            _dispatcher = avx_unpremultiply_alpha_rgba_u16;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_unpremultiply_alpha_rgba_u16;
    }
    _dispatcher(dst, src, width, height, bit_depth, pool);
}
