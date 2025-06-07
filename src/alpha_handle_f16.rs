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
#![forbid(unsafe_code)]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::avx2::{avx_premultiply_alpha_rgba_f16, avx_unpremultiply_alpha_rgba_f16};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{neon_premultiply_alpha_rgba_f16, neon_unpremultiply_alpha_rgba_f16};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{neon_premultiply_alpha_rgba_f16_full, neon_unpremultiply_alpha_rgba_f16_full};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::{sse_premultiply_alpha_rgba_f16, sse_unpremultiply_alpha_rgba_f16};
use core::f16;
use novtb::{ParallelZonedIterator, TbSliceMut};

#[inline]
pub(crate) fn unpremultiply_pixel_f16_row(in_place: &mut [f16]) {
    for dst in in_place.chunks_exact_mut(4) {
        let mut r = dst[0] as f32;
        let mut g = dst[1] as f32;
        let mut b = dst[2] as f32;
        let a = dst[3] as f32;
        if a != 0. {
            let scale_alpha = 1. / a;
            r *= scale_alpha;
            g *= scale_alpha;
            b *= scale_alpha;
        } else {
            r = 0.;
            g = 0.;
            b = 0.;
        }
        dst[0] = r as f16;
        dst[1] = g as f16;
        dst[2] = b as f16;
    }
}

#[inline]
pub(crate) fn premultiply_pixel_f16_row(dst: &mut [f16], src: &[f16]) {
    for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        let mut r = src[0] as f32;
        let mut g = src[1] as f32;
        let mut b = src[2] as f32;
        let a = src[3] as f32;
        r *= a;
        g *= a;
        b *= a;
        dst[0] = r as f16;
        dst[1] = g as f16;
        dst[2] = b as f16;
        dst[3] = a as f16;
    }
}

fn premultiply_alpha_rgba_impl_f16(
    dst: &mut [f16],
    dst_stride: usize,
    src: &[f16],
    src_stride: usize,
    width: usize,
    _: usize,
    pool: &novtb::ThreadPool,
) {
    dst.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| {
            let src = &src[y * src_stride..(y + 1) * src_stride];
            premultiply_pixel_f16_row(&mut dst[..width * 4], &src[..width * 4]);
        });
}

fn unpremultiply_alpha_rgba_impl_f16(
    dst: &mut [f16],
    stride: usize,
    width: usize,
    _: usize,
    pool: &novtb::ThreadPool,
) {
    dst.tb_par_chunks_exact_mut(stride).for_each(pool, |row| {
        unpremultiply_pixel_f16_row(&mut row[..width * 4]);
    });
}

pub(crate) fn premultiply_alpha_rgba_f16(
    dst: &mut [f16],
    dst_stride: usize,
    src: &[f16],
    src_stride: usize,
    width: usize,
    height: usize,
    pool: &novtb::ThreadPool,
) {
    #[allow(clippy::type_complexity)]
    let mut _dispatcher: fn(
        &mut [f16],
        usize,
        &[f16],
        usize,
        usize,
        usize,
        &novtb::ThreadPool,
    ) = premultiply_alpha_rgba_impl_f16;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_premultiply_alpha_rgba_f16;
        if std::arch::is_aarch64_feature_detected!("fp16") {
            _dispatcher = neon_premultiply_alpha_rgba_f16_full;
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _dispatcher = sse_premultiply_alpha_rgba_f16;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            _dispatcher = avx_premultiply_alpha_rgba_f16;
        }
    }
    _dispatcher(dst, dst_stride, src, src_stride, width, height, pool);
}

pub(crate) fn unpremultiply_alpha_rgba_f16(
    in_place: &mut [f16],
    stride: usize,
    width: usize,
    height: usize,
    pool: &novtb::ThreadPool,
) {
    let mut _dispatcher: fn(&mut [f16], usize, usize, usize, &novtb::ThreadPool) =
        unpremultiply_alpha_rgba_impl_f16;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_unpremultiply_alpha_rgba_f16;
        if std::arch::is_aarch64_feature_detected!("fp16") {
            _dispatcher = neon_unpremultiply_alpha_rgba_f16_full;
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _dispatcher = sse_unpremultiply_alpha_rgba_f16;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            _dispatcher = avx_unpremultiply_alpha_rgba_f16;
        }
    }
    _dispatcher(in_place, stride, width, height, pool);
}
