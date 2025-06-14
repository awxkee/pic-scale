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

use crate::WorkloadStrategy;
use crate::avx2::utils::{
    _mm256_select_si256, avx2_deinterleave_rgba, avx2_div_by255, avx2_interleave_rgba,
};
use novtb::{ParallelZonedIterator, TbSliceMut};
use std::arch::x86_64::*;

pub(crate) fn avx_premultiply_alpha_rgba(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    width: usize,
    height: usize,
    src_stride: usize,
    pool: &novtb::ThreadPool,
) {
    unsafe {
        avx_premultiply_alpha_rgba_impl(dst, dst_stride, src, width, height, src_stride, pool);
    }
}

trait AssociateAlpha {
    unsafe fn associate(&self, dst: &mut [u8], src: &[u8]);
}

#[derive(Default)]
struct AssociateAlphaDefault {}

impl AssociateAlphaDefault {
    #[inline(always)]
    unsafe fn associate_chunk(&self, dst: &mut [u8], src: &[u8]) {
        unsafe {
            let shuffle = _mm256_setr_epi8(
                3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15, 3, 3, 3, 3, 7, 7, 7, 7, 11,
                11, 11, 11, 15, 15, 15, 15,
            );
            let blend_mask = _mm256_setr_epi8(
                0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                0, -1, 0, 0, 0, -1,
            );
            let src_ptr = src.as_ptr();
            let rgba0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let multiplicand = _mm256_shuffle_epi8(rgba0, shuffle);

            let zeros = _mm256_setzero_si256();

            let mut v_ll = _mm256_unpacklo_epi8(rgba0, zeros);
            let mut v_hi = _mm256_unpackhi_epi8(rgba0, zeros);

            let a_lo = _mm256_unpacklo_epi8(multiplicand, zeros);
            let a_hi = _mm256_unpackhi_epi8(multiplicand, zeros);

            let la_lo = _mm256_mullo_epi16(v_ll, a_lo);
            let la_hi = _mm256_mullo_epi16(v_hi, a_hi);

            v_ll = avx2_div_by255(la_lo);
            v_hi = avx2_div_by255(la_hi);

            let values = _mm256_blendv_epi8(_mm256_packus_epi16(v_ll, v_hi), rgba0, blend_mask);

            let dst_ptr = dst.as_mut_ptr();
            _mm256_storeu_si256(dst_ptr as *mut __m256i, values);
        }
    }
}

impl AssociateAlpha for AssociateAlphaDefault {
    #[target_feature(enable = "avx2")]
    unsafe fn associate(&self, dst: &mut [u8], src: &[u8]) {
        unsafe {
            let mut rem = dst;
            let mut src_rem = src;

            for (dst, src) in rem.chunks_exact_mut(32).zip(src_rem.chunks_exact(32)) {
                self.associate_chunk(dst, src);
            }

            rem = rem.chunks_exact_mut(32).into_remainder();
            src_rem = src_rem.chunks_exact(32).remainder();

            if !rem.is_empty() {
                const PART_SIZE: usize = 32;
                assert!(src_rem.len() < PART_SIZE);
                assert!(rem.len() < PART_SIZE);
                assert_eq!(src_rem.len(), rem.len());

                let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];
                let mut dst_buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];
                std::ptr::copy_nonoverlapping(src_rem.as_ptr(), buffer.as_mut_ptr(), src_rem.len());

                self.associate_chunk(&mut dst_buffer, &buffer);

                std::ptr::copy_nonoverlapping(dst_buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_premultiply_alpha_rgba_impl_row(
    dst: &mut [u8],
    src: &[u8],
    executor: impl AssociateAlpha,
) {
    unsafe {
        executor.associate(dst, src);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_premultiply_alpha_rgba_impl(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    width: usize,
    _: usize,
    src_stride: usize,
    pool: &novtb::ThreadPool,
) {
    dst.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| unsafe {
            let src = &src[y * src_stride..(y + 1) * src_stride];
            avx_premultiply_alpha_rgba_impl_row(
                &mut dst[..width * 4],
                &src[..width * 4],
                AssociateAlphaDefault::default(),
            );
        });
}

pub(crate) fn avx_unpremultiply_alpha_rgba(
    in_place: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    pool: &novtb::ThreadPool,
    _: WorkloadStrategy,
) {
    unsafe {
        avx_unpremultiply_alpha_rgba_impl(in_place, width, height, stride, pool);
    }
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u8]);
}

#[derive(Default)]
struct Avx2DisassociateAlpha {}

impl Avx2DisassociateAlpha {
    #[inline(always)]
    unsafe fn avx2_unpremultiply_row(&self, x: __m256i, a: __m256i) -> __m256i {
        unsafe {
            let zeros = _mm256_setzero_si256();
            let lo = _mm256_unpacklo_epi8(x, zeros);
            let hi = _mm256_unpackhi_epi8(x, zeros);

            let is_zero_mask = _mm256_cmpeq_epi8(a, zeros);

            let scale_ps = _mm256_set1_ps(255f32);

            let llw = _mm256_unpacklo_epi16(lo, zeros);
            let lhw = _mm256_unpackhi_epi16(lo, zeros);
            let hlw = _mm256_unpacklo_epi16(hi, zeros);
            let hhw = _mm256_unpackhi_epi16(hi, zeros);

            let llwc = _mm256_cvtepi32_ps(llw);
            let lhwc = _mm256_cvtepi32_ps(lhw);
            let hlwc = _mm256_cvtepi32_ps(hlw);
            let hhwc = _mm256_cvtepi32_ps(hhw);

            let lo_lo = _mm256_mul_ps(llwc, scale_ps);
            let lo_hi = _mm256_mul_ps(lhwc, scale_ps);
            let hi_lo = _mm256_mul_ps(hlwc, scale_ps);
            let hi_hi = _mm256_mul_ps(hhwc, scale_ps);

            let a_lo = _mm256_unpacklo_epi8(a, zeros);
            let a_hi = _mm256_unpackhi_epi8(a, zeros);

            let allw = _mm256_unpacklo_epi16(a_lo, zeros);
            let alhw = _mm256_unpackhi_epi16(a_lo, zeros);
            let ahlw = _mm256_unpacklo_epi16(a_hi, zeros);
            let ahhw = _mm256_unpackhi_epi16(a_hi, zeros);

            let allf = _mm256_cvtepi32_ps(allw);
            let alhf = _mm256_cvtepi32_ps(alhw);
            let ahlf = _mm256_cvtepi32_ps(ahlw);
            let ahhf = _mm256_cvtepi32_ps(ahhw);

            let a_lo_lo = _mm256_rcp_ps(allf);
            let a_lo_hi = _mm256_rcp_ps(alhf);
            let a_hi_lo = _mm256_rcp_ps(ahlf);
            let a_hi_hi = _mm256_rcp_ps(ahhf);

            let fllw = _mm256_mul_ps(lo_lo, a_lo_lo);
            let flhw = _mm256_mul_ps(lo_hi, a_lo_hi);
            let fhlw = _mm256_mul_ps(hi_lo, a_hi_lo);
            let fhhw = _mm256_mul_ps(hi_hi, a_hi_hi);

            let lo_lo = _mm256_cvtps_epi32(fllw);
            let lo_hi = _mm256_cvtps_epi32(flhw);
            let hi_lo = _mm256_cvtps_epi32(fhlw);
            let hi_hi = _mm256_cvtps_epi32(fhhw);

            let packed0 = _mm256_packus_epi32(lo_lo, lo_hi);
            let packed1 = _mm256_packus_epi32(hi_lo, hi_hi);

            _mm256_select_si256(is_zero_mask, zeros, _mm256_packus_epi16(packed0, packed1))
        }
    }

    #[inline(always)]
    unsafe fn disassociate_chunk<const FMA: bool>(&self, in_place: &mut [u8]) {
        unsafe {
            let src_ptr = in_place.as_ptr();
            let rgba0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let rgba1 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let rgba2 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
            let rgba3 = _mm256_loadu_si256(src_ptr.add(96) as *const __m256i);
            let (rrr, ggg, bbb, aaa) = avx2_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

            let rrr = self.avx2_unpremultiply_row(rrr, aaa);
            let ggg = self.avx2_unpremultiply_row(ggg, aaa);
            let bbb = self.avx2_unpremultiply_row(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) = avx2_interleave_rgba(rrr, ggg, bbb, aaa);

            let dst_ptr = in_place.as_mut_ptr();
            _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba0);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, rgba1);
            _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, rgba2);
            _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, rgba3);
        }
    }

    #[inline(always)]
    unsafe fn disassociate_work<const FMA: bool>(&self, in_place: &mut [u8]) {
        unsafe {
            let mut rem = in_place;

            for dst in rem.chunks_exact_mut(32 * 4) {
                self.disassociate_chunk::<FMA>(dst);
            }

            rem = rem.chunks_exact_mut(32 * 4).into_remainder();

            if !rem.is_empty() {
                const PART_SIZE: usize = 32 * 4;
                assert!(rem.len() < PART_SIZE);

                let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];

                std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

                self.disassociate_chunk::<FMA>(&mut buffer);

                std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn disassociate_avx2(&self, in_place: &mut [u8]) {
        unsafe {
            self.disassociate_work::<false>(in_place);
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn disassociate_fma(&self, in_place: &mut [u8]) {
        unsafe {
            self.disassociate_work::<true>(in_place);
        }
    }
}

impl DisassociateAlpha for Avx2DisassociateAlpha {
    #[target_feature(enable = "avx2")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        unsafe {
            if std::arch::is_x86_feature_detected!("fma") {
                self.disassociate_fma(in_place);
            } else {
                self.disassociate_avx2(in_place);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_impl_row(
    in_place: &mut [u8],
    executor: impl DisassociateAlpha,
) {
    unsafe {
        executor.disassociate(in_place);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_impl(
    in_place: &mut [u8],
    width: usize,
    _: usize,
    stride: usize,
    pool: &novtb::ThreadPool,
) {
    in_place
        .tb_par_chunks_exact_mut(stride)
        .for_each(pool, |row| unsafe {
            avx_unpremultiply_alpha_rgba_impl_row(
                &mut row[..width * 4],
                Avx2DisassociateAlpha::default(),
            );
        });
}
