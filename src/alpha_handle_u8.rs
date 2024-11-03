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
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::*;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128",))]
use crate::wasm32::{wasm_premultiply_alpha_rgba, wasm_unpremultiply_alpha_rgba};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;
use rayon::slice::ParallelSliceMut;
use rayon::ThreadPool;

#[inline]
pub fn div_by_255(v: u16) -> u8 {
    ((((v + 0x80) >> 8) + v + 0x80) >> 8).min(255) as u8
}

#[inline]
pub(crate) fn premultiply_alpha_rgba_row_impl(dst: &mut [u8], src: &[u8]) {
    for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        let a = src[3] as u16;
        dst[0] = div_by_255(src[0] as u16 * a);
        dst[1] = div_by_255(src[1] as u16 * a);
        dst[2] = div_by_255(src[2] as u16 * a);
        dst[3] = div_by_255(a * a);
    }
}

fn premultiply_alpha_rgba_impl(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(width * 4)
                .zip(src.par_chunks_exact(width * 4))
                .for_each(|(dst, src)| {
                    premultiply_alpha_rgba_row_impl(dst, src);
                });
        });
    } else {
        dst.chunks_exact_mut(width * 4)
            .zip(src.chunks_exact(width * 4))
            .for_each(|(dst, src)| {
                premultiply_alpha_rgba_row_impl(dst, src);
            });
    }
}

#[inline]
pub(crate) fn unpremultiply_alpha_rgba_row_impl(in_place: &mut [u8]) {
    for dst in in_place.chunks_exact_mut(4) {
        let a = dst[3];
        if a != 0 {
            let a_recip = 1. / a as f32;
            dst[0] = ((dst[0] as f32 * 255.) * a_recip) as u8;
            dst[1] = ((dst[1] as f32 * 255.) * a_recip) as u8;
            dst[2] = ((dst[2] as f32 * 255.) * a_recip) as u8;
            dst[3] = ((a as f32 * 255.) * a_recip) as u8;
        }
    }
}

fn unpremultiply_alpha_rgba_impl(
    in_place: &mut [u8],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            in_place.par_chunks_exact_mut(width * 4).for_each(|row| {
                unpremultiply_alpha_rgba_row_impl(row);
            });
        });
    } else {
        in_place.chunks_exact_mut(width * 4).for_each(|row| {
            unpremultiply_alpha_rgba_row_impl(row);
        });
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
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        _dispatcher = wasm_premultiply_alpha_rgba;
    }
    _dispatcher(dst, src, width, height, pool);
}

pub fn unpremultiply_alpha_rgba(
    in_place: &mut [u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    let mut _dispatcher: fn(&mut [u8], usize, usize, &Option<ThreadPool>) =
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
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        _dispatcher = wasm_unpremultiply_alpha_rgba;
    }
    _dispatcher(in_place, width, height, pool);
}
