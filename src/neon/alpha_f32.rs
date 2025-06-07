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

use crate::alpha_handle_f32::{premultiply_rgba_f32_row, unpremultiply_rgba_f32_row};
use novtb::{ParallelZonedIterator, TbSliceMut};
use std::arch::aarch64::*;

macro_rules! unpremultiply_vec_f32 {
    ($v: expr, $a_values: expr) => {{
        let is_zero_mask = vceqzq_f32($a_values);
        let rs = vdivq_f32($v, $a_values);
        vbslq_f32(is_zero_mask, vdupq_n_f32(0.), rs)
    }};
}

unsafe fn neon_premultiply_alpha_rgba_row_f32(dst: &mut [f32], src: &[f32]) {
    unsafe {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem.chunks_exact_mut(4 * 4).zip(src_rem.chunks_exact(4 * 4)) {
            let src_ptr = src.as_ptr();
            let mut pixel = vld4q_f32(src_ptr);
            pixel.0 = vmulq_f32(pixel.0, pixel.3);
            pixel.1 = vmulq_f32(pixel.1, pixel.3);
            pixel.2 = vmulq_f32(pixel.2, pixel.3);
            let dst_ptr = dst.as_mut_ptr();
            vst4q_f32(dst_ptr, pixel);
        }

        rem = rem.chunks_exact_mut(4 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(4 * 4).remainder();

        premultiply_rgba_f32_row(rem, src_rem);
    }
}

pub(crate) fn neon_premultiply_alpha_rgba_f32(
    dst: &mut [f32],
    dst_stride: usize,
    src: &[f32],
    src_stride: usize,
    width: usize,
    _: usize,
    pool: &novtb::ThreadPool,
) {
    dst.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| unsafe {
            let src = &src[y * src_stride..(y + 1) * src_stride];
            neon_premultiply_alpha_rgba_row_f32(&mut dst[..width * 4], &src[..width * 4]);
        });
}

unsafe fn neon_unpremultiply_alpha_rgba_f32_row(in_place: &mut [f32]) {
    unsafe {
        let mut rem = in_place;

        for dst in rem.chunks_exact_mut(4 * 4) {
            let src_ptr = dst.as_ptr();
            let mut pixel = vld4q_f32(src_ptr);
            pixel.0 = unpremultiply_vec_f32!(pixel.0, pixel.3);
            pixel.1 = unpremultiply_vec_f32!(pixel.1, pixel.3);
            pixel.2 = unpremultiply_vec_f32!(pixel.2, pixel.3);
            let dst_ptr = dst.as_mut_ptr();
            vst4q_f32(dst_ptr, pixel);
        }

        rem = rem.chunks_exact_mut(4 * 4).into_remainder();

        unpremultiply_rgba_f32_row(rem);
    }
}

pub(crate) fn neon_unpremultiply_alpha_rgba_f32(
    in_place: &mut [f32],
    stride: usize,
    width: usize,
    _: usize,
    pool: &novtb::ThreadPool,
) {
    in_place
        .tb_par_chunks_exact_mut(stride)
        .for_each(pool, |row| unsafe {
            neon_unpremultiply_alpha_rgba_f32_row(&mut row[..width * 4]);
        });
}
