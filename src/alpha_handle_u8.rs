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
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx2::{avx_premultiply_alpha_rgba, avx_unpremultiply_alpha_rgba};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{neon_premultiply_alpha_rgba, neon_unpremultiply_alpha_rgba};
#[cfg(all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    feature = "riscv"
))]
use crate::risc::{risc_premultiply_alpha_rgba_u8, risc_unpremultiply_alpha_rgba_u8};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::*;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::{wasm_premultiply_alpha_rgba, wasm_unpremultiply_alpha_rgba};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;

#[macro_export]
macro_rules! unpremultiply_pixel {
    ($dst: expr, $src: expr, $pixel_offset: expr) => {{
        let mut r = *unsafe { $src.get_unchecked($pixel_offset) } as i32;
        let mut g = *unsafe { $src.get_unchecked($pixel_offset + 1) } as i32;
        let mut b = *unsafe { $src.get_unchecked($pixel_offset + 2) } as i32;
        let a = *unsafe { $src.get_unchecked($pixel_offset + 3) } as i32;
        if a != 0 {
            r = (r * 255) / a;
            g = (g * 255) / a;
            b = (b * 255) / a;
        }
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = r as u8;
            *$dst.get_unchecked_mut($pixel_offset + 1) = g as u8;
            *$dst.get_unchecked_mut($pixel_offset + 2) = b as u8;
            *$dst.get_unchecked_mut($pixel_offset + 3) = a as u8;
        }
    }};
}

#[macro_export]
macro_rules! premultiply_pixel {
    ($dst: expr, $src: expr, $pixel_offset: expr) => {{
        let mut r = *unsafe { $src.get_unchecked($pixel_offset) } as i32;
        let mut g = *unsafe { $src.get_unchecked($pixel_offset + 1) } as i32;
        let mut b = *unsafe { $src.get_unchecked($pixel_offset + 2) } as i32;
        let a = *unsafe { $src.get_unchecked($pixel_offset + 3) } as i32;
        r *= a;
        g *= a;
        b *= a;
        r /= 255;
        g /= 255;
        b /= 255;
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = r as u8;
            *$dst.get_unchecked_mut($pixel_offset + 1) = g as u8;
            *$dst.get_unchecked_mut($pixel_offset + 2) = b as u8;
            *$dst.get_unchecked_mut($pixel_offset + 3) = a as u8;
        }
    }};
}

fn premultiply_alpha_rgba_impl(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| {
                    for x in 0..width {
                        let px = x * 4;
                        premultiply_pixel!(dst, src, px);
                    }
                });
        });
    } else {
        let mut offset = 0usize;

        for _ in 0..height {
            for x in 0..width {
                let px = x * 4;
                premultiply_pixel!(dst, src, offset + px);
            }

            offset += 4 * width;
        }
    }
}

fn unpremultiply_alpha_rgba_impl(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(move || {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| {
                    for x in 0..width {
                        let px = x * 4;
                        unpremultiply_pixel!(dst, src, px);
                    }
                });
        });
    } else {
        let mut offset = 0usize;
        for _ in 0..height {
            for x in 0..width {
                let px = x * 4;
                let pixel_offset = offset + px;
                unpremultiply_pixel!(dst, src, pixel_offset);
            }

            offset += 4 * width;
        }
    }
}

pub fn premultiply_alpha_rgba(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    let mut _dispatcher: fn(&mut [u8], &[u8], usize, usize, &Option<ThreadPool>) =
        premultiply_alpha_rgba_impl;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_premultiply_alpha_rgba;
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            _dispatcher = sse_premultiply_alpha_rgba;
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            _dispatcher = avx_premultiply_alpha_rgba;
        }
    }
    #[cfg(all(
        any(target_arch = "riscv64", target_arch = "riscv32"),
        feature = "riscv"
    ))]
    {
        if std::arch::is_riscv_feature_detected!("v") {
            _dispatcher = risc_premultiply_alpha_rgba_u8;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        _dispatcher = wasm_premultiply_alpha_rgba;
    }
    _dispatcher(dst, src, width, height, pool);
}

pub fn unpremultiply_alpha_rgba(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    let mut _dispatcher: fn(&mut [u8], &[u8], usize, usize, &Option<ThreadPool>) =
        unpremultiply_alpha_rgba_impl;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_unpremultiply_alpha_rgba;
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            _dispatcher = sse_unpremultiply_alpha_rgba;
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            _dispatcher = avx_unpremultiply_alpha_rgba;
        }
    }
    #[cfg(all(
        any(target_arch = "riscv64", target_arch = "riscv32"),
        feature = "riscv"
    ))]
    {
        if std::arch::is_riscv_feature_detected!("v") {
            _dispatcher = risc_unpremultiply_alpha_rgba_u8;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        _dispatcher = wasm_unpremultiply_alpha_rgba;
    }
    _dispatcher(dst, src, width, height, pool);
}
