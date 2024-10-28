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

use crate::sse::{sse_deinterleave_rgba, sse_interleave_rgba};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;
use rayon::ThreadPool;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn _mm_select_si128(mask: __m128i, true_vals: __m128i, false_vals: __m128i) -> __m128i {
    _mm_or_si128(
        _mm_and_si128(mask, true_vals),
        _mm_andnot_si128(mask, false_vals),
    )
}

#[inline(always)]
pub unsafe fn _mm_div_by_255_epi16(v: __m128i) -> __m128i {
    let addition = _mm_set1_epi16(127);
    _mm_srli_epi16::<8>(_mm_add_epi16(
        _mm_add_epi16(v, addition),
        _mm_srli_epi16::<8>(v),
    ))
}

#[inline(always)]
pub unsafe fn sse_unpremultiply_row(x: __m128i, a: __m128i) -> __m128i {
    let zeros = _mm_setzero_si128();
    let lo = _mm_cvtepu8_epi16(x);
    let hi = _mm_unpackhi_epi8(x, zeros);

    let scale = _mm_set1_epi16(255);

    let is_zero_mask = _mm_cmpeq_epi8(a, zeros);
    let a = _mm_select_si128(is_zero_mask, scale, a);

    let scale_ps = _mm_set1_ps(255f32);

    let lo_lo = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(lo)), scale_ps);
    let lo_hi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(lo, zeros)), scale_ps);
    let hi_lo = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(hi)), scale_ps);
    let hi_hi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(hi, zeros)), scale_ps);
    let a_lo = _mm_cvtepu8_epi16(a);
    let a_hi = _mm_unpackhi_epi8(a, zeros);
    let a_lo_lo = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_lo)));
    let a_lo_hi = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a_lo, zeros)));
    let a_hi_lo = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_hi)));
    let a_hi_hi = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a_hi, zeros)));

    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    let lo_lo = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(lo_lo, a_lo_lo)));
    let lo_hi = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(lo_hi, a_lo_hi)));
    let hi_lo = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(hi_lo, a_hi_lo)));
    let hi_hi = _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(hi_hi, a_hi_hi)));

    let lo = _mm_packs_epi32(lo_lo, lo_hi);
    let hi = _mm_packs_epi32(hi_lo, hi_hi);
    _mm_select_si128(is_zero_mask, _mm_setzero_si128(), _mm_packus_epi16(lo, hi))
}

pub fn sse_premultiply_alpha_rgba(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        sse_premultiply_alpha_rgba_impl(dst, src, width, height, pool);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_premultiply_alpha_rgba_impl_row(dst: &mut [u8], src: &[u8]) {
    let mut rem = dst;
    let mut src_rem = src;

    unsafe {
        let zeros = _mm_setzero_si128();
        for (dst, src) in rem
            .chunks_exact_mut(16 * 4)
            .zip(src_rem.chunks_exact(16 * 4))
        {
            let src_ptr = src.as_ptr();
            let rgba0 = _mm_loadu_si128(src_ptr as *const __m128i);
            let rgba1 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
            let rgba2 = _mm_loadu_si128(src_ptr.add(32) as *const __m128i);
            let rgba3 = _mm_loadu_si128(src_ptr.add(48) as *const __m128i);
            let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

            let mut rrr_low = _mm_cvtepu8_epi16(rrr);
            let mut rrr_high = _mm_unpackhi_epi8(rrr, zeros);

            let mut ggg_low = _mm_cvtepu8_epi16(ggg);
            let mut ggg_high = _mm_unpackhi_epi8(ggg, zeros);

            let mut bbb_low = _mm_cvtepu8_epi16(bbb);
            let mut bbb_high = _mm_unpackhi_epi8(bbb, zeros);

            let aaa_low = _mm_cvtepu8_epi16(aaa);
            let aaa_high = _mm_unpackhi_epi8(aaa, zeros);

            rrr_low = _mm_div_by_255_epi16(_mm_mullo_epi16(rrr_low, aaa_low));
            rrr_high = _mm_div_by_255_epi16(_mm_mullo_epi16(rrr_high, aaa_high));
            ggg_low = _mm_div_by_255_epi16(_mm_mullo_epi16(ggg_low, aaa_low));
            ggg_high = _mm_div_by_255_epi16(_mm_mullo_epi16(ggg_high, aaa_high));
            bbb_low = _mm_div_by_255_epi16(_mm_mullo_epi16(bbb_low, aaa_low));
            bbb_high = _mm_div_by_255_epi16(_mm_mullo_epi16(bbb_high, aaa_high));

            let rrr = _mm_packus_epi16(rrr_low, rrr_high);
            let ggg = _mm_packus_epi16(ggg_low, ggg_high);
            let bbb = _mm_packus_epi16(bbb_low, bbb_high);

            let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba(rrr, ggg, bbb, aaa);

            let dst_ptr = dst.as_mut_ptr();
            _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
            _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
            _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);
        }

        rem = rem.chunks_exact_mut(16 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(16 * 4).remainder();
    }

    for (dst, src) in rem.chunks_exact_mut(4).zip(src_rem.chunks_exact(4)) {
        let a = src[3];
        if a != 0 {
            let a_recip = 1. / a as f32;
            dst[0] = ((src[0] as f32 * 255.) * a_recip) as u8;
            dst[1] = ((src[1] as f32 * 255.) * a_recip) as u8;
            dst[2] = ((src[2] as f32 * 255.) * a_recip) as u8;
            dst[3] = ((a as f32 * 255.) * a_recip) as u8;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_premultiply_alpha_rgba_impl(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| unsafe {
                    sse_premultiply_alpha_rgba_impl_row(dst, src);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(4 * width)
            .zip(src.chunks_exact(4 * width))
        {
            unsafe {
                sse_premultiply_alpha_rgba_impl_row(dst_row, src_row);
            }
        }
    }
}

pub fn sse_unpremultiply_alpha_rgba(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        sse_unpremultiply_alpha_rgba_impl(dst, src, width, height, pool);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_unpremultiply_alpha_rgba_impl_row(dst: &mut [u8], src: &[u8]) {
    let mut rem = dst;
    let mut src_rem = src;
    unsafe {
        for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
            let src_ptr = src.as_ptr();
            let rgba0 = _mm_loadu_si128(src_ptr as *const __m128i);
            let rgba1 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
            let rgba2 = _mm_loadu_si128(src_ptr.add(32) as *const __m128i);
            let rgba3 = _mm_loadu_si128(src_ptr.add(48) as *const __m128i);
            let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

            let rrr = sse_unpremultiply_row(rrr, aaa);
            let ggg = sse_unpremultiply_row(ggg, aaa);
            let bbb = sse_unpremultiply_row(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba(rrr, ggg, bbb, aaa);

            let dst_ptr = dst.as_mut_ptr();
            _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
            _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
            _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);
        }

        rem = rem.chunks_exact_mut(8 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(8 * 4).remainder();
    }

    for (dst, src) in rem.chunks_exact_mut(4).zip(src_rem.chunks_exact(4)) {
        let a = src[3];
        if a != 0 {
            let a_recip = 1. / a as f32;
            dst[0] = ((src[0] as f32 * 255.) * a_recip) as u8;
            dst[1] = ((src[1] as f32 * 255.) * a_recip) as u8;
            dst[2] = ((src[2] as f32 * 255.) * a_recip) as u8;
            dst[3] = ((a as f32 * 255.) * a_recip) as u8;
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_unpremultiply_alpha_rgba_impl(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| unsafe {
                    sse_unpremultiply_alpha_rgba_impl_row(dst, src);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(4 * width)
            .zip(src.chunks_exact(4 * width))
        {
            unsafe {
                sse_unpremultiply_alpha_rgba_impl_row(dst_row, src_row);
            }
        }
    }
}
