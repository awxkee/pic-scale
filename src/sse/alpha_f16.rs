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

use crate::alpha_handle_f16::{premultiply_pixel_f16_row, unpremultiply_pixel_f16_row};
use crate::sse::f16_utils::{_mm_cvtph_psx, _mm_cvtps_phx};
use crate::sse::{sse_deinterleave_rgba_epi16, sse_interleave_rgba_epi16};
use core::f16;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_premultiply_alpha_rgba_f16(
    dst: &mut [f16],
    dst_stride: usize,
    src: &[f16],
    src_stride: usize,
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        if std::arch::is_x86_feature_detected!("f16c") {
            sse_premultiply_alpha_rgba_f16c(dst, dst_stride, src, src_stride, width, height, pool);
        } else {
            sse_premultiply_alpha_rgba_f16_regular(
                dst, dst_stride, src, src_stride, width, height, pool,
            );
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_premultiply_alpha_rgba_f16_regular(
    dst: &mut [f16],
    dst_stride: usize,
    src: &[f16],
    src_stride: usize,
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    sse_premultiply_alpha_rgba_f16_impl::<false>(
        dst, dst_stride, src, src_stride, width, height, pool,
    );
}

#[target_feature(enable = "sse4.1", enable = "f16c")]
unsafe fn sse_premultiply_alpha_rgba_f16c(
    dst: &mut [f16],
    dst_stride: usize,
    src: &[f16],
    src_stride: usize,
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    sse_premultiply_alpha_rgba_f16_impl::<true>(
        dst, dst_stride, src, src_stride, width, height, pool,
    );
}

#[inline(always)]
unsafe fn sse_premultiply_alpha_rgba_row_f16_impl<const F16C: bool>(dst: &mut [f16], src: &[f16]) {
    let mut rem = dst;
    let mut src_rem = src;

    for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
        let src_ptr = src.as_ptr();
        let lane0 = _mm_loadu_si128(src_ptr as *const __m128i);
        let lane1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
        let lane2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
        let lane3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
        let pixel = sse_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

        let low_alpha = _mm_cvtph_psx::<F16C>(pixel.3);
        let low_r = _mm_mul_ps(_mm_cvtph_psx::<F16C>(pixel.0), low_alpha);
        let low_g = _mm_mul_ps(_mm_cvtph_psx::<F16C>(pixel.1), low_alpha);
        let low_b = _mm_mul_ps(_mm_cvtph_psx::<F16C>(pixel.2), low_alpha);

        let high_alpha = _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.3));
        let high_r = _mm_mul_ps(
            _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.0)),
            high_alpha,
        );
        let high_g = _mm_mul_ps(
            _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.1)),
            high_alpha,
        );
        let high_b = _mm_mul_ps(
            _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.2)),
            high_alpha,
        );
        let r_values =
            _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_r), _mm_cvtps_phx::<F16C>(high_r));
        let g_values =
            _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_g), _mm_cvtps_phx::<F16C>(high_g));
        let b_values =
            _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_b), _mm_cvtps_phx::<F16C>(high_b));
        let dst_ptr = dst.as_mut_ptr();
        let (d_lane0, d_lane1, d_lane2, d_lane3) =
            sse_interleave_rgba_epi16(r_values, g_values, b_values, pixel.3);
        _mm_storeu_si128(dst_ptr as *mut __m128i, d_lane0);
        _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, d_lane1);
        _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, d_lane2);
        _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, d_lane3);
    }

    rem = rem.chunks_exact_mut(8 * 4).into_remainder();
    src_rem = src_rem.chunks_exact(8 * 4).remainder();

    premultiply_pixel_f16_row(rem, src_rem);
}

#[inline(always)]
unsafe fn sse_premultiply_alpha_rgba_f16_impl<const F16C: bool>(
    dst: &mut [f16],
    dst_stride: usize,
    src: &[f16],
    src_stride: usize,
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(dst_stride)
                .zip(src.par_chunks_exact(src_stride))
                .for_each(|(dst, src)| unsafe {
                    sse_premultiply_alpha_rgba_row_f16_impl::<F16C>(
                        &mut dst[..width * 4],
                        &src[..width * 4],
                    );
                });
        });
    } else {
        dst.chunks_exact_mut(dst_stride)
            .zip(src.chunks_exact(src_stride))
            .for_each(|(dst, src)| unsafe {
                sse_premultiply_alpha_rgba_row_f16_impl::<F16C>(
                    &mut dst[..width * 4],
                    &src[..width * 4],
                );
            });
    }
}

pub(crate) fn sse_unpremultiply_alpha_rgba_f16(
    in_place: &mut [f16],
    stride: usize,
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        if is_x86_feature_detected!("f16c") {
            sse_unpremultiply_alpha_rgba_f16c(in_place, stride, width, height, pool);
        } else {
            sse_unpremultiply_alpha_rgba_f16_regular(in_place, stride, width, height, pool);
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_unpremultiply_alpha_rgba_f16_regular(
    in_place: &mut [f16],
    stride: usize,
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    sse_unpremultiply_alpha_rgba_f16_impl::<false>(in_place, stride, width, height, pool);
}

#[target_feature(enable = "sse4.1", enable = "f16c")]
unsafe fn sse_unpremultiply_alpha_rgba_f16c(
    in_place: &mut [f16],
    stride: usize,
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    sse_unpremultiply_alpha_rgba_f16_impl::<true>(in_place, stride, width, height, pool);
}

#[inline(always)]
unsafe fn sse_unpremultiply_alpha_rgba_f16_row_impl<const F16C: bool>(in_place: &mut [f16]) {
    let mut rem = in_place;

    for dst in rem.chunks_exact_mut(8 * 4) {
        let src_ptr = dst.as_ptr();
        let lane0 = _mm_loadu_si128(src_ptr as *const __m128i);
        let lane1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
        let lane2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
        let lane3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
        let pixel = sse_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

        let low_alpha = _mm_cvtph_psx::<F16C>(pixel.3);
        let zeros = _mm_setzero_ps();
        let low_alpha_zero_mask = _mm_cmpeq_ps(low_alpha, zeros);
        let low_r = _mm_blendv_ps(
            _mm_mul_ps(_mm_cvtph_psx::<F16C>(pixel.0), low_alpha),
            zeros,
            low_alpha_zero_mask,
        );
        let low_g = _mm_blendv_ps(
            _mm_mul_ps(_mm_cvtph_psx::<F16C>(pixel.1), low_alpha),
            zeros,
            low_alpha_zero_mask,
        );
        let low_b = _mm_blendv_ps(
            _mm_mul_ps(_mm_cvtph_psx::<F16C>(pixel.2), low_alpha),
            zeros,
            low_alpha_zero_mask,
        );

        let high_alpha = _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.3));
        let high_alpha_zero_mask = _mm_cmpeq_ps(high_alpha, zeros);
        let high_r = _mm_blendv_ps(
            _mm_mul_ps(
                _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.0)),
                high_alpha,
            ),
            zeros,
            high_alpha_zero_mask,
        );
        let high_g = _mm_blendv_ps(
            _mm_mul_ps(
                _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.1)),
                high_alpha,
            ),
            zeros,
            high_alpha_zero_mask,
        );
        let high_b = _mm_blendv_ps(
            _mm_mul_ps(
                _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.2)),
                high_alpha,
            ),
            zeros,
            high_alpha_zero_mask,
        );
        let r_values =
            _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_r), _mm_cvtps_phx::<F16C>(high_r));
        let g_values =
            _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_g), _mm_cvtps_phx::<F16C>(high_g));
        let b_values =
            _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_b), _mm_cvtps_phx::<F16C>(high_b));
        let dst_ptr = dst.as_mut_ptr();
        let (d_lane0, d_lane1, d_lane2, d_lane3) =
            sse_interleave_rgba_epi16(r_values, g_values, b_values, pixel.3);
        _mm_storeu_si128(dst_ptr as *mut __m128i, d_lane0);
        _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, d_lane1);
        _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, d_lane2);
        _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, d_lane3);
    }

    rem = rem.chunks_exact_mut(8 * 4).into_remainder();

    unpremultiply_pixel_f16_row(rem);
}

#[inline(always)]
unsafe fn sse_unpremultiply_alpha_rgba_f16_impl<const F16C: bool>(
    in_place: &mut [f16],
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
                    sse_unpremultiply_alpha_rgba_f16_row_impl::<F16C>(&mut row[..width * 4]);
                });
        });
    } else {
        in_place.chunks_exact_mut(stride).for_each(|row| unsafe {
            sse_unpremultiply_alpha_rgba_f16_row_impl::<F16C>(&mut row[..width * 4]);
        });
    }
}
