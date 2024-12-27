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

use crate::alpha_handle_u8::{premultiply_alpha_rgba_row_impl, unpremultiply_alpha_rgba_row_impl};
use crate::avx2::utils::{
    _mm256_select_si256, avx2_deinterleave_rgba, avx2_div_by255, avx2_interleave_rgba,
};
use crate::sse::{
    _mm_div_by_255_epi16, sse_deinterleave_rgba, sse_interleave_rgba, sse_unpremultiply_row,
};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn avx2_unpremultiply_row(x: __m256i, a: __m256i) -> __m256i {
    let zeros = _mm256_setzero_si256();
    let lo = _mm256_unpacklo_epi8(x, zeros);
    let hi = _mm256_unpackhi_epi8(x, zeros);

    let scale = _mm256_set1_epi16(255);

    let is_zero_mask = _mm256_cmpeq_epi8(a, zeros);
    let a = _mm256_select_si256(is_zero_mask, scale, a);

    let scale_ps = _mm256_set1_ps(255f32);

    let lo_lo = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(lo, zeros)),
        scale_ps,
    );
    let lo_hi = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(lo, zeros)),
        scale_ps,
    );
    let hi_lo = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(hi, zeros)),
        scale_ps,
    );
    let hi_hi = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(hi, zeros)),
        scale_ps,
    );
    let a_lo = _mm256_unpacklo_epi8(a, zeros);
    let a_hi = _mm256_unpackhi_epi8(x, zeros);
    let a_lo_lo = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_unpacklo_epi16(a_lo, zeros)));
    let a_lo_hi = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_unpackhi_epi16(a_lo, zeros)));
    let a_hi_lo = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_unpacklo_epi16(a_hi, zeros)));
    let a_hi_hi = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_unpackhi_epi16(a_hi, zeros)));

    let lo_lo = _mm256_cvtps_epi32(_mm256_mul_ps(lo_lo, a_lo_lo));
    let lo_hi = _mm256_cvtps_epi32(_mm256_mul_ps(lo_hi, a_lo_hi));
    let hi_lo = _mm256_cvtps_epi32(_mm256_mul_ps(hi_lo, a_hi_lo));
    let hi_hi = _mm256_cvtps_epi32(_mm256_mul_ps(hi_hi, a_hi_hi));

    _mm256_select_si256(
        is_zero_mask,
        zeros,
        _mm256_packus_epi16(
            _mm256_packus_epi32(lo_lo, lo_hi),
            _mm256_packus_epi32(hi_lo, hi_hi),
        ),
    )
}

pub(crate) fn avx_premultiply_alpha_rgba(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        avx_premultiply_alpha_rgba_impl(dst, src, width, height, pool);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_premultiply_alpha_rgba_impl_row(dst: &mut [u8], src: &[u8]) {
    let mut rem = dst;
    let mut src_rem = src;

    unsafe {
        for (dst, src) in rem
            .chunks_exact_mut(32 * 4)
            .zip(src_rem.chunks_exact(32 * 4))
        {
            let src_ptr = src.as_ptr();
            let rgba0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let rgba1 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let rgba2 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
            let rgba3 = _mm256_loadu_si256(src_ptr.add(96) as *const __m256i);
            let (rrr, ggg, bbb, aaa) = avx2_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

            let zeros = _mm256_setzero_si256();

            let mut rrr_low = _mm256_unpacklo_epi8(rrr, zeros);
            let mut rrr_high = _mm256_unpackhi_epi8(rrr, zeros);

            let mut ggg_low = _mm256_unpacklo_epi8(ggg, zeros);
            let mut ggg_high = _mm256_unpackhi_epi8(ggg, zeros);

            let mut bbb_low = _mm256_unpacklo_epi8(bbb, zeros);
            let mut bbb_high = _mm256_unpackhi_epi8(bbb, zeros);

            let aaa_low = _mm256_unpacklo_epi8(aaa, zeros);
            let aaa_high = _mm256_unpackhi_epi8(aaa, zeros);

            rrr_low = avx2_div_by255(_mm256_mullo_epi16(rrr_low, aaa_low));
            rrr_high = avx2_div_by255(_mm256_mullo_epi16(rrr_high, aaa_high));
            ggg_low = avx2_div_by255(_mm256_mullo_epi16(ggg_low, aaa_low));
            ggg_high = avx2_div_by255(_mm256_mullo_epi16(ggg_high, aaa_high));
            bbb_low = avx2_div_by255(_mm256_mullo_epi16(bbb_low, aaa_low));
            bbb_high = avx2_div_by255(_mm256_mullo_epi16(bbb_high, aaa_high));

            let rrr = _mm256_packus_epi16(rrr_low, rrr_high);
            let ggg = _mm256_packus_epi16(ggg_low, ggg_high);
            let bbb = _mm256_packus_epi16(bbb_low, bbb_high);

            let (rgba0, rgba1, rgba2, rgba3) = avx2_interleave_rgba(rrr, ggg, bbb, aaa);
            let dst_ptr = dst.as_mut_ptr();
            _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba0);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, rgba1);
            _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, rgba2);
            _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, rgba3);
        }

        rem = rem.chunks_exact_mut(32 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(32 * 4).remainder();

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

            let mut rrr_low = _mm_unpacklo_epi8(rrr, zeros);
            let mut rrr_high = _mm_unpackhi_epi8(rrr, zeros);

            let mut ggg_low = _mm_unpacklo_epi8(ggg, zeros);
            let mut ggg_high = _mm_unpackhi_epi8(ggg, zeros);

            let mut bbb_low = _mm_unpacklo_epi8(bbb, zeros);
            let mut bbb_high = _mm_unpackhi_epi8(bbb, zeros);

            let aaa_low = _mm_unpacklo_epi8(aaa, zeros);
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

    premultiply_alpha_rgba_row_impl(rem, src_rem);
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx_premultiply_alpha_rgba_impl(
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
                .for_each(|(dst, src)| unsafe {
                    avx_premultiply_alpha_rgba_impl_row(dst, src);
                });
        });
    } else {
        dst.chunks_exact_mut(width * 4)
            .zip(src.chunks_exact(width * 4))
            .for_each(|(dst, src)| unsafe {
                avx_premultiply_alpha_rgba_impl_row(dst, src);
            });
    }
}

pub(crate) fn avx_unpremultiply_alpha_rgba(
    in_place: &mut [u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        avx_unpremultiply_alpha_rgba_impl(in_place, width, height, pool);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_impl_row(in_place: &mut [u8]) {
    let mut rem = in_place;

    unsafe {
        for dst in rem.chunks_exact_mut(32 * 4) {
            let src_ptr = dst.as_ptr();
            let rgba0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let rgba1 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let rgba2 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
            let rgba3 = _mm256_loadu_si256(src_ptr.add(96) as *const __m256i);
            let (rrr, ggg, bbb, aaa) = avx2_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

            let rrr = avx2_unpremultiply_row(rrr, aaa);
            let ggg = avx2_unpremultiply_row(ggg, aaa);
            let bbb = avx2_unpremultiply_row(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) = avx2_interleave_rgba(rrr, ggg, bbb, aaa);

            let dst_ptr = dst.as_mut_ptr();
            _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba0);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, rgba1);
            _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, rgba2);
            _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, rgba3);
        }

        rem = rem.chunks_exact_mut(32 * 4).into_remainder();

        for dst in rem.chunks_exact_mut(16 * 4) {
            let src_ptr = dst.as_ptr();
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

        rem = rem.chunks_exact_mut(16 * 4).into_remainder();
    }

    unpremultiply_alpha_rgba_row_impl(rem);
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_impl(
    in_place: &mut [u8],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            in_place
                .par_chunks_exact_mut(width * 4)
                .for_each(|row| unsafe {
                    avx_unpremultiply_alpha_rgba_impl_row(row);
                });
        });
    } else {
        in_place.chunks_exact_mut(width * 4).for_each(|row| unsafe {
            avx_unpremultiply_alpha_rgba_impl_row(row);
        });
    }
}
