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
#![forbid(unsafe_code)]

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::avx2::{avx_premultiply_alpha_rgba_u16, avx_unpremultiply_alpha_rgba_u16};
#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::neon::{neon_premultiply_alpha_rgba_u16, neon_unpremultiply_alpha_rgba_u16};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::{premultiply_alpha_sse_rgba_u16, unpremultiply_alpha_sse_rgba_u16};
use novtb::{ParallelZonedIterator, TbSliceMut};

#[inline]
/// Divides value by 1023 with rounding to nearest
pub(crate) fn div_by_1023(v: u32) -> u16 {
    let round = 1 << 9;
    let v = v + round;
    (((v >> 10) + v) >> 10) as u16
}

#[inline]
/// Divides value by 4095 with rounding to nearest
pub(crate) fn div_by_4095(v: u32) -> u16 {
    let round = 1 << 11;
    let v = v + round;
    (((v >> 12) + v) >> 12) as u16
}

#[inline]
/// Divides value by 655353 with rounding to nearest
pub(crate) fn div_by_65535(v: u32) -> u16 {
    let round = 1 << 15;
    let v_expand = v;
    let v = v_expand + round;
    (((v >> 16) + v) >> 16) as u16
}

pub(crate) fn premultiply_alpha_rgba_row(dst: &mut [u16], src: &[u16], max_colors: u32) {
    if max_colors == 1023 {
        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            let a = src[3] as u32;
            dst[0] = div_by_1023((src[0] as u32).wrapping_mul(a));
            dst[1] = div_by_1023((src[1] as u32).wrapping_mul(a));
            dst[2] = div_by_1023((src[2] as u32).wrapping_mul(a));
            dst[3] = div_by_1023((src[3] as u32).wrapping_mul(a));
        }
    } else if max_colors == 4096 {
        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            let a = src[3] as u32;
            dst[0] = div_by_4095((src[0] as u32).wrapping_mul(a));
            dst[1] = div_by_4095((src[1] as u32).wrapping_mul(a));
            dst[2] = div_by_4095((src[2] as u32).wrapping_mul(a));
            dst[3] = div_by_4095((src[3] as u32).wrapping_mul(a));
        }
    } else if max_colors == 65535 {
        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            let a = src[3] as u32;
            dst[0] = div_by_65535((src[0] as u32).wrapping_mul(a));
            dst[1] = div_by_65535((src[1] as u32).wrapping_mul(a));
            dst[2] = div_by_65535((src[2] as u32).wrapping_mul(a));
            dst[3] = div_by_65535((src[3] as u32).wrapping_mul(a));
        }
    } else {
        let recip_max_colors = 1. / max_colors as f32;
        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            let a = src[3] as u32;
            dst[0] = (((src[0] as u32).wrapping_mul(a) as f32 * recip_max_colors).round() as u32)
                .min(max_colors) as u16;
            dst[1] = (((src[1] as u32).wrapping_mul(a) as f32 * recip_max_colors).round() as u32)
                .min(max_colors) as u16;
            dst[2] = (((src[2] as u32).wrapping_mul(a) as f32 * recip_max_colors).round() as u32)
                .min(max_colors) as u16;
            dst[3] = ((a.wrapping_mul(a) as f32 * recip_max_colors).round() as u32).min(max_colors)
                as u16;
        }
    }
}

pub(crate) fn premultiply_alpha_gray_alpha_row(dst: &mut [u16], src: &[u16], max_colors: u32) {
    if max_colors == 1023 {
        for (dst, src) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let a = src[1] as u32;
            dst[0] = div_by_1023((src[0] as u32).wrapping_mul(a));
            dst[1] = div_by_1023(a.wrapping_mul(a));
        }
    } else if max_colors == 4096 {
        for (dst, src) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let a = src[1] as u32;
            dst[0] = div_by_4095((src[0] as u32).wrapping_mul(a));
            dst[1] = div_by_4095(a.wrapping_mul(a));
        }
    } else if max_colors == 65535 {
        for (dst, src) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let a = src[1] as u32;
            dst[0] = div_by_65535((src[0] as u32).wrapping_mul(a));
            dst[1] = div_by_65535(a.wrapping_mul(a));
        }
    } else {
        let recip_max_colors = 1. / max_colors as f32;
        for (dst, src) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let a = src[1] as u32;
            dst[0] = (((src[0] as u32).wrapping_mul(a) as f32 * recip_max_colors).round() as u32)
                .min(max_colors) as u16;
            dst[1] = ((a.wrapping_mul(a) as f32 * recip_max_colors).round() as u32).min(max_colors)
                as u16;
        }
    }
}

pub(crate) fn unpremultiply_alpha_rgba_row(in_place: &mut [u16], max_colors: u32) {
    for dst in in_place.chunks_exact_mut(4) {
        let a = dst[3] as u32;
        if a != 0 {
            let a_recip = 1. / a as f32;
            dst[0] = ((dst[0] as u32 * max_colors) as f32 * a_recip)
                .round()
                .min(max_colors as f32) as u16;
            dst[1] = ((dst[1] as u32 * max_colors) as f32 * a_recip)
                .round()
                .min(max_colors as f32) as u16;
            dst[2] = ((dst[2] as u32 * max_colors) as f32 * a_recip)
                .round()
                .min(max_colors as f32) as u16;
            dst[3] = ((a * max_colors) as f32 * a_recip)
                .round()
                .min(max_colors as f32) as u16;
        }
    }
}

pub(crate) fn unpremultiply_alpha_gray_alpha_row(in_place: &mut [u16], max_colors: u32) {
    for dst in in_place.chunks_exact_mut(2) {
        let a = dst[1] as u32;
        if a != 0 {
            let a_recip = 1. / a as f32;
            dst[0] = ((dst[0] as u32 * max_colors) as f32 * a_recip)
                .round()
                .min(max_colors as f32) as u16;
            dst[1] = ((a * max_colors) as f32 * a_recip)
                .round()
                .min(max_colors as f32) as u16;
        }
    }
}

fn premultiply_alpha_rgba_impl(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    width: usize,
    _: usize,
    src_stride: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    let max_colors = (1 << bit_depth) - 1;
    dst.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| {
            let src = &src[y * src_stride..(y + 1) * src_stride];
            premultiply_alpha_rgba_row(&mut dst[..width * 4], &src[..width * 4], max_colors);
        });
}

fn premultiply_alpha_gray_alpha_impl(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    width: usize,
    _: usize,
    src_stride: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    let max_colors = (1 << bit_depth) - 1;
    dst.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| {
            let src = &src[y * src_stride..(y + 1) * src_stride];
            premultiply_alpha_gray_alpha_row(&mut dst[..width * 2], &src[..width * 2], max_colors);
        });
}

fn unpremultiply_alpha_rgba_impl(
    in_place: &mut [u16],
    src_stride: usize,
    width: usize,
    _: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    let max_colors = (1 << bit_depth) - 1;
    in_place
        .tb_par_chunks_exact_mut(src_stride)
        .for_each(pool, |row| {
            unpremultiply_alpha_rgba_row(&mut row[..width * 4], max_colors);
        });
}

fn unpremultiply_alpha_gray_alpha_impl(
    in_place: &mut [u16],
    src_stride: usize,
    width: usize,
    _: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    let max_colors = (1 << bit_depth) - 1;
    in_place
        .tb_par_chunks_exact_mut(src_stride)
        .for_each(pool, |row| {
            unpremultiply_alpha_gray_alpha_row(&mut row[..width * 2], max_colors);
        });
}

pub(crate) fn premultiply_alpha_rgba_u16(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    width: usize,
    height: usize,
    src_stride: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    #[allow(clippy::type_complexity)]
    let mut _dispatcher: fn(
        &mut [u16],
        usize,
        &[u16],
        usize,
        usize,
        usize,
        usize,
        &novtb::ThreadPool,
    ) = premultiply_alpha_rgba_impl;
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _dispatcher = premultiply_alpha_sse_rgba_u16;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            _dispatcher = avx_premultiply_alpha_rgba_u16;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_premultiply_alpha_rgba_u16;
    }
    _dispatcher(
        dst, dst_stride, src, width, height, src_stride, bit_depth, pool,
    );
}

pub(crate) fn premultiply_alpha_gray_alpha_u16(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    width: usize,
    height: usize,
    src_stride: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    #[allow(clippy::type_complexity)]
    let mut _dispatcher: fn(
        &mut [u16],
        usize,
        &[u16],
        usize,
        usize,
        usize,
        usize,
        &novtb::ThreadPool,
    ) = premultiply_alpha_gray_alpha_impl;
    _dispatcher(
        dst, dst_stride, src, width, height, src_stride, bit_depth, pool,
    );
}

pub(crate) fn unpremultiply_alpha_rgba_u16(
    in_place: &mut [u16],
    src_stride: usize,
    width: usize,
    height: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    #[allow(clippy::type_complexity)]
    let mut _dispatcher: fn(&mut [u16], usize, usize, usize, usize, &novtb::ThreadPool) =
        unpremultiply_alpha_rgba_impl;
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _dispatcher = unpremultiply_alpha_sse_rgba_u16;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            _dispatcher = avx_unpremultiply_alpha_rgba_u16;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_unpremultiply_alpha_rgba_u16;
    }
    _dispatcher(in_place, src_stride, width, height, bit_depth, pool);
}

pub(crate) fn unpremultiply_alpha_gray_alpha_u16(
    in_place: &mut [u16],
    src_stride: usize,
    width: usize,
    height: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    #[allow(clippy::type_complexity)]
    let mut _dispatcher: fn(&mut [u16], usize, usize, usize, usize, &novtb::ThreadPool) =
        unpremultiply_alpha_gray_alpha_impl;
    _dispatcher(in_place, src_stride, width, height, bit_depth, pool);
}
