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

use crate::avx2::utils::{
    _mm256_packus_four_epi32, _mm256_select_si256, avx2_deinterleave_rgba, avx2_div_by255,
    avx2_interleave_rgba, avx2_pack_u16,
};
use crate::sse::{
    _mm_div_by_255_epi16, sse_deinterleave_rgba, sse_interleave_rgba, sse_unpremultiply_row,
};
use crate::{premultiply_pixel, unpremultiply_pixel};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;
use rayon::ThreadPool;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn avx2_unpremultiply_row(x: __m256i, a: __m256i) -> __m256i {
    let zeros = _mm256_setzero_si256();
    let lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(x));
    let hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(x));

    let scale = _mm256_set1_epi16(255);

    let is_zero_mask = _mm256_cmpeq_epi8(a, zeros);
    let a = _mm256_select_si256(is_zero_mask, scale, a);

    let scale_ps = _mm256_set1_ps(255f32);

    let lo_lo = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo))),
        scale_ps,
    );
    let lo_hi = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo))),
        scale_ps,
    );
    let hi_lo = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi))),
        scale_ps,
    );
    let hi_hi = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi))),
        scale_ps,
    );
    let a_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a));
    let a_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a));
    let a_lo_lo = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
        _mm256_castsi256_si128(a_lo),
    )));
    let a_lo_hi = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
        _mm256_extracti128_si256::<1>(a_lo),
    )));
    let a_hi_lo = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
        _mm256_castsi256_si128(a_hi),
    )));
    let a_hi_hi = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
        _mm256_extracti128_si256::<1>(a_hi),
    )));

    let lo_lo = _mm256_cvtps_epi32(_mm256_mul_ps(lo_lo, a_lo_lo));
    let lo_hi = _mm256_cvtps_epi32(_mm256_mul_ps(lo_hi, a_lo_hi));
    let hi_lo = _mm256_cvtps_epi32(_mm256_mul_ps(hi_lo, a_hi_lo));
    let hi_hi = _mm256_cvtps_epi32(_mm256_mul_ps(hi_hi, a_hi_hi));

    _mm256_select_si256(
        is_zero_mask,
        zeros,
        _mm256_packus_four_epi32(lo_lo, lo_hi, hi_lo, hi_hi),
    )
}

pub fn avx_premultiply_alpha_rgba(
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
unsafe fn avx_premultiply_alpha_rgba_impl_row(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    offset: usize,
) {
    let mut _cx = 0usize;

    unsafe {
        while _cx + 32 < width {
            let px = _cx * 4;
            let src_ptr = src.as_ptr().add(offset + px);
            let rgba0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let rgba1 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let rgba2 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
            let rgba3 = _mm256_loadu_si256(src_ptr.add(96) as *const __m256i);
            let (rrr, ggg, bbb, aaa) = avx2_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

            let mut rrr_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rrr));
            let mut rrr_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(rrr));

            let mut ggg_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ggg));
            let mut ggg_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(ggg));

            let mut bbb_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(bbb));
            let mut bbb_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(bbb));

            let aaa_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(aaa));
            let aaa_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(aaa));

            rrr_low = avx2_div_by255(_mm256_mullo_epi16(rrr_low, aaa_low));
            rrr_high = avx2_div_by255(_mm256_mullo_epi16(rrr_high, aaa_high));
            ggg_low = avx2_div_by255(_mm256_mullo_epi16(ggg_low, aaa_low));
            ggg_high = avx2_div_by255(_mm256_mullo_epi16(ggg_high, aaa_high));
            bbb_low = avx2_div_by255(_mm256_mullo_epi16(bbb_low, aaa_low));
            bbb_high = avx2_div_by255(_mm256_mullo_epi16(bbb_high, aaa_high));

            let rrr = avx2_pack_u16(rrr_low, rrr_high);
            let ggg = avx2_pack_u16(ggg_low, ggg_high);
            let bbb = avx2_pack_u16(bbb_low, bbb_high);

            let (rgba0, rgba1, rgba2, rgba3) = avx2_interleave_rgba(rrr, ggg, bbb, aaa);
            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba0);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, rgba1);
            _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, rgba2);
            _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, rgba3);

            _cx += 32;
        }

        let zeros = _mm_setzero_si128();
        while _cx + 16 < width {
            let px = _cx * 4;
            let src_ptr = src.as_ptr().add(offset + px);
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

            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
            _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
            _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);

            _cx += 16;
        }
    }

    for x in _cx..width {
        let px = x * 4;
        premultiply_pixel!(dst, src, offset + px);
    }
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
            src.par_chunks_exact(width * 4)
                .zip(dst.par_chunks_exact_mut(width * 4))
                .for_each(|(src, dst)| unsafe {
                    avx_premultiply_alpha_rgba_impl_row(dst, src, width, 0);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(4 * width)
            .zip(src.chunks_exact(4 * width))
        {
            unsafe {
                avx_premultiply_alpha_rgba_impl_row(dst_row, src_row, width, 0);
            }
        }
    }
}

pub fn avx_unpremultiply_alpha_rgba(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        avx_unpremultiply_alpha_rgba_impl(dst, src, width, height, pool);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_impl_row(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    offset: usize,
) {
    let mut _cx = 0usize;

    unsafe {
        while _cx + 32 < width {
            let px = _cx * 4;
            let pixel_offset = offset + px;
            let src_ptr = src.as_ptr().add(pixel_offset);
            let rgba0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let rgba1 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let rgba2 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
            let rgba3 = _mm256_loadu_si256(src_ptr.add(96) as *const __m256i);
            let (rrr, ggg, bbb, aaa) = avx2_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

            let rrr = avx2_unpremultiply_row(rrr, aaa);
            let ggg = avx2_unpremultiply_row(ggg, aaa);
            let bbb = avx2_unpremultiply_row(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) = avx2_interleave_rgba(rrr, ggg, bbb, aaa);

            let dst_ptr = dst.as_mut_ptr().add(pixel_offset);
            _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba0);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, rgba1);
            _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, rgba2);
            _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, rgba3);

            _cx += 32;
        }

        while _cx + 16 < width {
            let px = _cx * 4;
            let pixel_offset = offset + px;
            let src_ptr = src.as_ptr().add(pixel_offset);
            let rgba0 = _mm_loadu_si128(src_ptr as *const __m128i);
            let rgba1 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
            let rgba2 = _mm_loadu_si128(src_ptr.add(32) as *const __m128i);
            let rgba3 = _mm_loadu_si128(src_ptr.add(48) as *const __m128i);
            let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

            let rrr = sse_unpremultiply_row(rrr, aaa);
            let ggg = sse_unpremultiply_row(ggg, aaa);
            let bbb = sse_unpremultiply_row(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba(rrr, ggg, bbb, aaa);

            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
            _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
            _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);

            _cx += 16;
        }
    }

    for x in _cx..width {
        let px = x * 4;
        let pixel_offset = offset + px;
        unpremultiply_pixel!(dst, src, pixel_offset);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_impl(
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
                    avx_unpremultiply_alpha_rgba_impl_row(dst, src, width, 0);
                });
        });
    } else {
        for (dst_row, src_row) in dst
            .chunks_exact_mut(4 * width)
            .zip(src.chunks_exact(4 * width))
        {
            unsafe {
                avx_unpremultiply_alpha_rgba_impl_row(dst_row, src_row, width, 0);
            }
        }
    }
}
