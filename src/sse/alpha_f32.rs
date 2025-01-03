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
use crate::alpha_handle_f32::{premultiply_pixel_f32_row, unpremultiply_pixel_f32_row};
use crate::sse::{sse_deinterleave_rgba_ps, sse_interleave_rgba_ps};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;
use rayon::slice::ParallelSliceMut;
use rayon::ThreadPool;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_unpremultiply_row_f32(x: __m128, a: __m128) -> __m128 {
    let is_zero_mask = _mm_cmpeq_ps(a, _mm_setzero_ps());
    let rs = _mm_div_ps(x, a);
    _mm_blendv_ps(rs, _mm_setzero_ps(), is_zero_mask)
}

pub(crate) fn sse_unpremultiply_alpha_rgba_f32(
    in_place: &mut [f32],
    stride: usize,
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        sse_unpremultiply_alpha_rgba_f32_impl(in_place, stride, width, height, pool);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_unpremultiply_alpha_rgba_f32_row_impl(in_place: &mut [f32]) {
    for dst in in_place.chunks_exact_mut(4) {
        let src_ptr = dst.as_ptr();
        let rgba0 = _mm_loadu_ps(src_ptr);
        let rgba1 = _mm_loadu_ps(src_ptr.add(4));
        let rgba2 = _mm_loadu_ps(src_ptr.add(8));
        let rgba3 = _mm_loadu_ps(src_ptr.add(12));

        let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba_ps(rgba0, rgba1, rgba2, rgba3);

        let rrr = sse_unpremultiply_row_f32(rrr, aaa);
        let ggg = sse_unpremultiply_row_f32(ggg, aaa);
        let bbb = sse_unpremultiply_row_f32(bbb, aaa);

        let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba_ps(rrr, ggg, bbb, aaa);

        let dst_ptr = dst.as_mut_ptr();
        _mm_storeu_ps(dst_ptr, rgba0);
        _mm_storeu_ps(dst_ptr.add(4), rgba1);
        _mm_storeu_ps(dst_ptr.add(8), rgba2);
        _mm_storeu_ps(dst_ptr.add(12), rgba3);
    }

    let rem = in_place.chunks_exact_mut(4).into_remainder();

    unpremultiply_pixel_f32_row(rem);
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_unpremultiply_alpha_rgba_f32_impl(
    in_place: &mut [f32],
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
                    sse_unpremultiply_alpha_rgba_f32_row_impl(&mut row[..width * 4]);
                });
        });
    } else {
        in_place.chunks_exact_mut(stride).for_each(|row| unsafe {
            sse_unpremultiply_alpha_rgba_f32_row_impl(&mut row[..width * 4]);
        });
    }
}

pub(crate) fn sse_premultiply_alpha_rgba_f32(
    dst: &mut [f32],
    src: &[f32],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        sse_premultiply_alpha_rgba_f32_impl(dst, src, width, height, pool);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_premultiply_alpha_rgba_f32_row_impl(dst: &mut [f32], src: &[f32]) {
    let mut rem = dst;
    let mut src_rem = src;

    for (dst, src) in rem.chunks_exact_mut(4 * 4).zip(src_rem.chunks_exact(4 * 4)) {
        let src_ptr = src.as_ptr();
        let rgba0 = _mm_loadu_ps(src_ptr);
        let rgba1 = _mm_loadu_ps(src_ptr.add(4));
        let rgba2 = _mm_loadu_ps(src_ptr.add(8));
        let rgba3 = _mm_loadu_ps(src_ptr.add(12));
        let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba_ps(rgba0, rgba1, rgba2, rgba3);

        let rrr = _mm_mul_ps(rrr, aaa);
        let ggg = _mm_mul_ps(ggg, aaa);
        let bbb = _mm_mul_ps(bbb, aaa);

        let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba_ps(rrr, ggg, bbb, aaa);

        let dst_ptr = dst.as_mut_ptr();
        _mm_storeu_ps(dst_ptr, rgba0);
        _mm_storeu_ps(dst_ptr.add(4), rgba1);
        _mm_storeu_ps(dst_ptr.add(8), rgba2);
        _mm_storeu_ps(dst_ptr.add(12), rgba3);
    }

    rem = rem.chunks_exact_mut(4 * 4).into_remainder();
    src_rem = src_rem.chunks_exact(4 * 4).remainder();

    premultiply_pixel_f32_row(rem, src_rem);
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_premultiply_alpha_rgba_f32_impl(
    dst: &mut [f32],
    src: &[f32],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(width * 4)
                .zip(src.par_chunks_exact(width * 4))
                .for_each(|(dst, src)| unsafe {
                    sse_premultiply_alpha_rgba_f32_row_impl(dst, src);
                });
        });
    } else {
        dst.chunks_exact_mut(width * 4)
            .zip(src.chunks_exact(width * 4))
            .for_each(|(dst, src)| unsafe {
                sse_premultiply_alpha_rgba_f32_row_impl(dst, src);
            });
    }
}
