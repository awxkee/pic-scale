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
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm_select_si128(
    mask: __m128i,
    true_vals: __m128i,
    false_vals: __m128i,
) -> __m128i {
    _mm_blendv_epi8(false_vals, true_vals, mask)
}

/// Exact division by 255 with rounding to nearest
#[inline(always)]
pub(crate) unsafe fn _mm_div_by_255_epi16(v: __m128i) -> __m128i {
    let addition = _mm_set1_epi16(127);
    let j0 = _mm_add_epi16(v, addition);
    let j1 = _mm_srli_epi16::<8>(v);
    _mm_srli_epi16::<8>(_mm_add_epi16(j0, j1))
}

#[inline(always)]
pub(crate) unsafe fn sse_unpremultiply_row(x: __m128i, a: __m128i) -> __m128i {
    let zeros = _mm_setzero_si128();
    let lo = _mm_unpacklo_epi8(x, zeros);
    let hi = _mm_unpackhi_epi8(x, zeros);

    let is_zero_mask = _mm_cmpeq_epi8(a, zeros);

    let scale_ps = _mm_set1_ps(255f32);

    let llw = _mm_unpacklo_epi16(lo, zeros);
    let lhw = _mm_unpackhi_epi16(lo, zeros);
    let hlw = _mm_unpacklo_epi16(hi, zeros);
    let hhw = _mm_unpackhi_epi16(hi, zeros);

    let llwc = _mm_cvtepi32_ps(llw);
    let lhwc = _mm_cvtepi32_ps(lhw);
    let hlwc = _mm_cvtepi32_ps(hlw);
    let hhwc = _mm_cvtepi32_ps(hhw);

    let lo_lo = _mm_mul_ps(llwc, scale_ps);
    let lo_hi = _mm_mul_ps(lhwc, scale_ps);
    let hi_lo = _mm_mul_ps(hlwc, scale_ps);
    let hi_hi = _mm_mul_ps(hhwc, scale_ps);

    let a_lo = _mm_unpacklo_epi8(a, zeros);
    let a_hi = _mm_unpackhi_epi8(a, zeros);

    let allw = _mm_unpacklo_epi16(a_lo, zeros);
    let alhw = _mm_unpackhi_epi16(a_lo, zeros);
    let ahlw = _mm_unpacklo_epi16(a_hi, zeros);
    let ahhw = _mm_unpackhi_epi16(a_hi, zeros);

    let allf = _mm_cvtepi32_ps(allw);
    let alhf = _mm_cvtepi32_ps(alhw);
    let ahlf = _mm_cvtepi32_ps(ahlw);
    let ahhf = _mm_cvtepi32_ps(ahhw);

    let a_lo_lo = _mm_rcp_ps(allf);
    let a_lo_hi = _mm_rcp_ps(alhf);
    let a_hi_lo = _mm_rcp_ps(ahlf);
    let a_hi_hi = _mm_rcp_ps(ahhf);

    let mut fllw = _mm_mul_ps(lo_lo, a_lo_lo);
    let mut flhw = _mm_mul_ps(lo_hi, a_lo_hi);
    let mut fhlw = _mm_mul_ps(hi_lo, a_hi_lo);
    let mut fhhw = _mm_mul_ps(hi_hi, a_hi_hi);

    fllw = _mm_add_ps(_mm_set1_ps(0.5f32), fllw);
    flhw = _mm_add_ps(_mm_set1_ps(0.5f32), flhw);
    fhlw = _mm_add_ps(_mm_set1_ps(0.5f32), fhlw);
    fhhw = _mm_add_ps(_mm_set1_ps(0.5f32), fhhw);

    let lo_lo = _mm_cvtps_epi32(fllw);
    let lo_hi = _mm_cvtps_epi32(flhw);
    let hi_lo = _mm_cvtps_epi32(fhlw);
    let hi_hi = _mm_cvtps_epi32(fhhw);

    let lo = _mm_packs_epi32(lo_lo, lo_hi);
    let hi = _mm_packs_epi32(hi_lo, hi_hi);
    _mm_select_si128(is_zero_mask, _mm_setzero_si128(), _mm_packus_epi16(lo, hi))
}

pub(crate) fn sse_premultiply_alpha_rgba(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    width: usize,
    height: usize,
    src_stride: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        sse_premultiply_alpha_rgba_impl(dst, dst_stride, src, width, height, src_stride, pool);
    }
}

trait Sse41PremultiplyExecutorRgba8 {
    unsafe fn premultiply(&self, dst: &mut [u8], src: &[u8]);
}

#[derive(Default)]
struct Sse41PremultiplyExecutor8Default {}

impl Sse41PremultiplyExecutor8Default {
    #[inline(always)]
    unsafe fn premultiply_chunk(&self, dst: &mut [u8], src: &[u8]) {
        let zeros = _mm_setzero_si128();
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

        rrr_low = _mm_mullo_epi16(rrr_low, aaa_low);
        rrr_high = _mm_mullo_epi16(rrr_high, aaa_high);
        ggg_low = _mm_mullo_epi16(ggg_low, aaa_low);
        ggg_high = _mm_mullo_epi16(ggg_high, aaa_high);
        bbb_low = _mm_mullo_epi16(bbb_low, aaa_low);
        bbb_high = _mm_mullo_epi16(bbb_high, aaa_high);

        rrr_low = _mm_div_by_255_epi16(rrr_low);
        rrr_high = _mm_div_by_255_epi16(rrr_high);
        ggg_low = _mm_div_by_255_epi16(ggg_low);
        ggg_high = _mm_div_by_255_epi16(ggg_high);
        bbb_low = _mm_div_by_255_epi16(bbb_low);
        bbb_high = _mm_div_by_255_epi16(bbb_high);

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
}

impl Sse41PremultiplyExecutorRgba8 for Sse41PremultiplyExecutor8Default {
    #[target_feature(enable = "sse4.1")]
    unsafe fn premultiply(&self, dst: &mut [u8], src: &[u8]) {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .chunks_exact_mut(16 * 4)
            .zip(src_rem.chunks_exact(16 * 4))
        {
            self.premultiply_chunk(dst, src);
        }

        rem = rem.chunks_exact_mut(16 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(16 * 4).remainder();

        if !rem.is_empty() {
            const PART_SIZE: usize = 16 * 4;
            assert!(src_rem.len() < PART_SIZE);
            assert!(rem.len() < PART_SIZE);
            assert_eq!(src_rem.len(), rem.len());

            let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];
            let mut dst_buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];

            std::ptr::copy_nonoverlapping(src_rem.as_ptr(), buffer.as_mut_ptr(), src_rem.len());

            self.premultiply_chunk(&mut dst_buffer, &buffer);

            std::ptr::copy_nonoverlapping(dst_buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_premultiply_alpha_rgba_impl_row(
    dst: &mut [u8],
    src: &[u8],
    executor: impl Sse41PremultiplyExecutorRgba8,
) {
    executor.premultiply(dst, src);
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_premultiply_alpha_rgba_impl(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    width: usize,
    _: usize,
    src_stride: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(dst_stride)
                .zip(src.par_chunks_exact(src_stride))
                .for_each(|(dst, src)| unsafe {
                    sse_premultiply_alpha_rgba_impl_row(
                        &mut dst[..width * 4],
                        &src[..width * 4],
                        Sse41PremultiplyExecutor8Default::default(),
                    );
                });
        });
    } else {
        dst.chunks_exact_mut(dst_stride)
            .zip(src.chunks_exact(src_stride))
            .for_each(|(dst, src)| unsafe {
                sse_premultiply_alpha_rgba_impl_row(
                    &mut dst[..width * 4],
                    &src[..width * 4],
                    Sse41PremultiplyExecutor8Default::default(),
                );
            });
    }
}

pub(crate) fn sse_unpremultiply_alpha_rgba(
    in_place: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    pool: &Option<ThreadPool>,
) {
    unsafe {
        sse_unpremultiply_alpha_rgba_impl(in_place, width, height, stride, pool);
    }
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u8]);
}

#[derive(Default)]
struct DisassociateAlphaDefault {}

impl DisassociateAlphaDefault {
    #[inline(always)]
    unsafe fn disassociate_chunk(&self, in_place: &mut [u8]) {
        let src_ptr = in_place.as_ptr();
        let rgba0 = _mm_loadu_si128(src_ptr as *const __m128i);
        let rgba1 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
        let rgba2 = _mm_loadu_si128(src_ptr.add(32) as *const __m128i);
        let rgba3 = _mm_loadu_si128(src_ptr.add(48) as *const __m128i);
        let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

        let rrr = sse_unpremultiply_row(rrr, aaa);
        let ggg = sse_unpremultiply_row(ggg, aaa);
        let bbb = sse_unpremultiply_row(bbb, aaa);

        let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba(rrr, ggg, bbb, aaa);

        let dst_ptr = in_place.as_mut_ptr();
        _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
        _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
        _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
        _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);
    }
}

impl DisassociateAlpha for DisassociateAlphaDefault {
    #[target_feature(enable = "sse4.1")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        let mut rem = in_place;

        for dst in rem.chunks_exact_mut(16 * 4) {
            self.disassociate_chunk(dst);
        }

        rem = rem.chunks_exact_mut(16 * 4).into_remainder();

        if !rem.is_empty() {
            const PART_SIZE: usize = 16 * 4;
            assert!(rem.len() < PART_SIZE);

            let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];

            std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

            self.disassociate_chunk(&mut buffer);

            std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_unpremultiply_alpha_rgba_impl_row(
    in_place: &mut [u8],
    executor: impl DisassociateAlpha,
) {
    executor.disassociate(in_place);
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_unpremultiply_alpha_rgba_impl(
    in_place: &mut [u8],
    width: usize,
    _: usize,
    stride: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            in_place
                .par_chunks_exact_mut(stride)
                .for_each(|row| unsafe {
                    sse_unpremultiply_alpha_rgba_impl_row(
                        &mut row[..width * 4],
                        DisassociateAlphaDefault::default(),
                    );
                });
        });
    } else {
        in_place.chunks_exact_mut(stride).for_each(|row| unsafe {
            sse_unpremultiply_alpha_rgba_impl_row(
                &mut row[..width * 4],
                DisassociateAlphaDefault::default(),
            );
        });
    }
}
