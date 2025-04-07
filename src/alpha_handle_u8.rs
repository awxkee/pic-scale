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
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::avx2::{avx_premultiply_alpha_rgba, avx_unpremultiply_alpha_rgba};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{neon_premultiply_alpha_rgba, neon_unpremultiply_alpha_rgba};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::*;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128",))]
use crate::wasm32::{wasm_premultiply_alpha_rgba, wasm_unpremultiply_alpha_rgba};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;
use rayon::slice::ParallelSliceMut;
use rayon::ThreadPool;

#[inline]
/// Divides value by 255 with rounding to nearest
pub(crate) fn div_by_255(v: u16) -> u8 {
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
    dst_stride: usize,
    src: &[u8],
    width: usize,
    _: usize,
    stride: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(dst_stride)
                .zip(src.par_chunks_exact(stride))
                .for_each(|(dst, src)| {
                    premultiply_alpha_rgba_row_impl(&mut dst[..width * 4], &src[..width * 4]);
                });
        });
    } else {
        dst.chunks_exact_mut(dst_stride)
            .zip(src.chunks_exact(stride))
            .for_each(|(dst, src)| {
                premultiply_alpha_rgba_row_impl(&mut dst[..width * 4], &src[..width * 4]);
            });
    }
}

const fn make_unpremultiplication_table() -> [u8; 65536] {
    let mut alpha = 0usize;
    let mut buf = [0u8; 65536];
    while alpha < 256 {
        let mut pixel = 0usize;
        while pixel < 256 {
            if alpha == 0 {
                buf[alpha * 255 + pixel] = 0;
            } else {
                let value = (pixel * 255 + alpha / 2) / alpha;
                buf[alpha * 255 + pixel] = value as u8;
            }
            pixel += 1;
        }
        alpha += 1;
    }
    buf
}

static UNPREMULTIPLICATION_TABLE: [u8; 65536] = make_unpremultiplication_table();

#[inline]
pub(crate) fn unpremultiply_alpha_rgba_row_impl(in_place: &mut [u8]) {
    for dst in in_place.chunks_exact_mut(4) {
        let a = dst[3];
        let z = a as u16 * 255;
        dst[0] = UNPREMULTIPLICATION_TABLE[(z + dst[0] as u16) as usize];
        dst[1] = UNPREMULTIPLICATION_TABLE[(z + dst[1] as u16) as usize];
        dst[2] = UNPREMULTIPLICATION_TABLE[(z + dst[2] as u16) as usize];
    }
}

fn unpremultiply_alpha_rgba_impl(
    in_place: &mut [u8],
    width: usize,
    _: usize,
    stride: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            in_place.par_chunks_exact_mut(stride).for_each(|row| {
                unpremultiply_alpha_rgba_row_impl(&mut row[..width * 4]);
            });
        });
    } else {
        in_place.chunks_exact_mut(stride).for_each(|row| {
            unpremultiply_alpha_rgba_row_impl(&mut row[..width * 4]);
        });
    }
}

pub(crate) fn premultiply_alpha_rgba(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    width: usize,
    height: usize,
    src_stride: usize,
    pool: &Option<ThreadPool>,
) {
    #[allow(clippy::type_complexity)]
    let mut _dispatcher: fn(&mut [u8], usize, &[u8], usize, usize, usize, &Option<ThreadPool>) =
        premultiply_alpha_rgba_impl;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_premultiply_alpha_rgba;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _dispatcher = sse_premultiply_alpha_rgba;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") {
            _dispatcher = avx_premultiply_alpha_rgba;
        }
    }
    #[cfg(all(feature = "nightly_avx512", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
    {
        use crate::avx512::avx512_premultiply_alpha_rgba;
        _dispatcher = avx512_premultiply_alpha_rgba;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        _dispatcher = wasm_premultiply_alpha_rgba;
    }
    _dispatcher(dst, dst_stride, src, width, height, src_stride, pool);
}

pub(crate) fn unpremultiply_alpha_rgba(
    in_place: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    pool: &Option<ThreadPool>,
) {
    let mut _dispatcher: fn(&mut [u8], usize, usize, usize, &Option<ThreadPool>) =
        unpremultiply_alpha_rgba_impl;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_unpremultiply_alpha_rgba;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _dispatcher = sse_unpremultiply_alpha_rgba;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            _dispatcher = avx_unpremultiply_alpha_rgba;
        }
    }
    #[cfg(all(feature = "nightly_avx512", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
    {
        use crate::avx512::avx512_unpremultiply_alpha_rgba;
        _dispatcher = avx512_unpremultiply_alpha_rgba;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        _dispatcher = wasm_unpremultiply_alpha_rgba;
    }
    _dispatcher(in_place, width, height, stride, pool);
}
