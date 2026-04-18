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
    _mm256_select_si256, avx2_deinterleave_rgba, avx2_div_by255, avx2_interleave_rgba, shuffle,
};
use std::arch::x86_64::*;
use std::sync::OnceLock;

pub(crate) fn avx_premultiply_alpha_rgba(dst: &mut [u8], src: &[u8]) {
    unsafe {
        avx_premultiply_alpha_rgba_impl(dst, src);
    }
}

trait AssociateAlpha {
    unsafe fn associate(&self, dst: &mut [u8], src: &[u8]);
}

#[derive(Default)]
struct AssociateAlphaDefault {}

impl AssociateAlphaDefault {
    #[inline(always)]
    fn associate_chunk(&self, dst: &mut [u8], src: &[u8]) {
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
fn avx_premultiply_alpha_rgba_impl_row(dst: &mut [u8], src: &[u8], executor: impl AssociateAlpha) {
    unsafe {
        executor.associate(dst, src);
    }
}

#[target_feature(enable = "avx2")]
fn avx_premultiply_alpha_rgba_impl(dst: &mut [u8], src: &[u8]) {
    avx_premultiply_alpha_rgba_impl_row(dst, src, AssociateAlphaDefault::default());
}

static RECIP_TABLE_U32: OnceLock<[u32; 256]> = OnceLock::new();

fn get_recip_table() -> &'static [u32; 256] {
    RECIP_TABLE_U32.get_or_init(|| {
        let mut table = [0u32; 256];
        for i in 1..256 {
            table[i] = ((65536u32 + i as u32 / 2) / i as u32).min(65535);
        }
        table
    })
}

pub(crate) fn avx_unpremultiply_alpha_rgba(in_place: &mut [u8], _: WorkloadStrategy) {
    unsafe {
        avx_unpremultiply_alpha_rgba_impl(in_place);
    }
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u8]);
}

#[derive(Default)]
struct Avx2DisassociateAlpha {}

impl Avx2DisassociateAlpha {
    #[inline(always)]
    fn avx2_unpremultiply_row(&self, x: __m256i, is_zero_mask: __m256i, a: [__m256; 4]) -> __m256i {
        unsafe {
            let zeros = _mm256_setzero_si256();
            let lo = _mm256_unpacklo_epi8(x, zeros);
            let hi = _mm256_unpackhi_epi8(x, zeros);

            let scale_ps = _mm256_set1_ps(255.);

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

            let fllw = _mm256_mul_ps(lo_lo, a[0]);
            let flhw = _mm256_mul_ps(lo_hi, a[1]);
            let fhlw = _mm256_mul_ps(hi_lo, a[2]);
            let fhhw = _mm256_mul_ps(hi_hi, a[3]);

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
    fn disassociate_chunk<const FMA: bool>(&self, in_place: &mut [u8], table: &[u32; 256]) {
        unsafe {
            static ALPHA_IDX_SHUF: [u8; 32] = [
                3, 255, 255, 255, 7, 255, 255, 255, 11, 255, 255, 255, 15, 255, 255, 255, 3, 255,
                255, 255, 7, 255, 255, 255, 11, 255, 255, 255, 15, 255, 255, 255,
            ];

            static RECIP_BCAST_LO: [u8; 32] = [
                0, 1, 0, 1, 0, 1, 0, 1, 4, 5, 4, 5, 4, 5, 4, 5, 8, 9, 8, 9, 8, 9, 8, 9, 12, 13, 12,
                13, 12, 13, 12, 13,
            ];

            let bcast_lo = _mm256_loadu_si256(RECIP_BCAST_LO.as_ptr() as *const __m256i);

            let shuf = _mm256_loadu_si256(ALPHA_IDX_SHUF.as_ptr() as *const __m256i);

            let rgba = _mm256_loadu_si256(in_place.as_ptr() as *const __m256i);

            let idx = _mm256_shuffle_epi8(rgba, shuf);

            let recip = _mm256_i32gather_epi32::<4>(table.as_ptr().cast(), idx);

            const HI_HI: i32 = 0b0011_0001;
            const LO_LO: i32 = 0b0010_0000;

            let recip_lo =
                _mm256_shuffle_epi8(_mm256_permute2x128_si256::<LO_LO>(recip, recip), bcast_lo);
            let recip_hi =
                _mm256_shuffle_epi8(_mm256_permute2x128_si256::<HI_HI>(recip, recip), bcast_lo);

            let lo16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rgba));
            let hi16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(rgba));

            let lo_mul =
                _mm256_mulhi_epu16(_mm256_mullo_epi16(lo16, _mm256_set1_epi16(255)), recip_lo);
            let hi_mul =
                _mm256_mulhi_epu16(_mm256_mullo_epi16(hi16, _mm256_set1_epi16(255)), recip_hi);

            let mut packed = _mm256_packus_epi16(lo_mul, hi_mul);

            packed = _mm256_permute4x64_epi64::<{ shuffle(3, 1, 2, 0) }>(packed);

            let alpha_mask = _mm256_set1_epi32(0xFF000000u32 as i32);
            packed = _mm256_blendv_epi8(packed, rgba, alpha_mask);

            let is_zero = _mm256_cmpeq_epi32(idx, _mm256_setzero_si256());
            let result = _mm256_blendv_epi8(packed, rgba, is_zero);

            _mm256_storeu_si256(in_place.as_mut_ptr().cast(), result);
        }
    }

    #[inline(always)]
    fn disassociate_work<const FMA: bool>(&self, in_place: &mut [u8]) {
        unsafe {
            let mut rem = in_place;

            let table = get_recip_table();

            for dst in rem.chunks_exact_mut(32) {
                self.disassociate_chunk::<FMA>(dst, table);
            }

            rem = rem.chunks_exact_mut(32).into_remainder();

            if !rem.is_empty() {
                let mut buffer: [u8; 32] = [0u8; 32];

                buffer[..rem.len()].copy_from_slice(rem);

                self.disassociate_chunk::<FMA>(&mut buffer, table);

                rem.copy_from_slice(&buffer[..rem.len()]);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    fn disassociate_avx2(&self, in_place: &mut [u8]) {
        self.disassociate_work::<false>(in_place);
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn disassociate_fma(&self, in_place: &mut [u8]) {
        self.disassociate_work::<true>(in_place);
    }
}

impl DisassociateAlpha for Avx2DisassociateAlpha {
    #[target_feature(enable = "avx2")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        if std::arch::is_x86_feature_detected!("fma") {
            unsafe {
                self.disassociate_fma(in_place);
            }
        } else {
            self.disassociate_avx2(in_place);
        }
    }
}

#[target_feature(enable = "avx2")]
fn avx_unpremultiply_alpha_rgba_impl_row(in_place: &mut [u8], executor: impl DisassociateAlpha) {
    unsafe {
        executor.disassociate(in_place);
    }
}

#[target_feature(enable = "avx2")]
fn avx_unpremultiply_alpha_rgba_impl(in_place: &mut [u8]) {
    avx_unpremultiply_alpha_rgba_impl_row(in_place, Avx2DisassociateAlpha::default());
}
