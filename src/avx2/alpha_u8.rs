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
use crate::avx2::utils::avx2_div_by255;
use std::arch::x86_64::*;

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
    fn associate_chunk(&self, dst: &mut [u8; 32], src: &[u8; 32]) {
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
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(src_rem.as_chunks::<32>().0.iter())
        {
            self.associate_chunk(dst, src);
        }

        rem = rem.as_chunks_mut::<32>().1;
        src_rem = src_rem.as_chunks::<32>().1;

        if !rem.is_empty() {
            const PART_SIZE: usize = 32;
            assert!(src_rem.len() < PART_SIZE);
            assert!(rem.len() < PART_SIZE);
            assert_eq!(src_rem.len(), rem.len());

            let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];
            let mut dst_buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];
            buffer[..rem.len()].copy_from_slice(rem);

            self.associate_chunk(&mut dst_buffer, &buffer);

            rem.copy_from_slice(&dst_buffer[..rem.len()]);
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

pub(crate) fn avx_unpremultiply_alpha_rgba(in_place: &mut [u8], strategy: WorkloadStrategy) {
    unsafe {
        match strategy {
            WorkloadStrategy::PreferQuality => {
                avx_unpremultiply_alpha_rgba_impl_row(in_place, Avx2DisassociateAlpha::default())
            }
            WorkloadStrategy::PreferSpeed => avx_unpremultiply_alpha_rgba_impl_row(
                in_place,
                Avx2DisassociateAlphaFast::default(),
            ),
        }
    }
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u8]);
}

#[derive(Default)]
struct Avx2DisassociateAlpha {}

impl Avx2DisassociateAlpha {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn disassociate_chunk(&self, in_place: &mut [u8; 32]) {
        let alpha_mask = _mm256_setr_epi8(
            3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1, 3, -1, -1, -1, 7, -1, -1,
            -1, 11, -1, -1, -1, 15, -1, -1, -1,
        );
        let rgba = unsafe { _mm256_loadu_si256(in_place.as_ptr().cast()) };
        let alpha_u32 = _mm256_shuffle_epi8(rgba, alpha_mask);
        let alpha_f32 = _mm256_mul_ps(
            _mm256_rcp_ps(_mm256_cvtepi32_ps(alpha_u32)),
            _mm256_set1_ps(255.),
        );
        let is_zero_mask = _mm256_cmpeq_epi32(alpha_u32, _mm256_setzero_si256());

        let a_lo = _mm256_unpacklo_epi8(rgba, _mm256_setzero_si256());
        let a_hi = _mm256_unpackhi_epi8(rgba, _mm256_setzero_si256());

        let p0 = _mm256_unpacklo_epi16(a_lo, _mm256_setzero_si256());
        let p1 = _mm256_unpackhi_epi16(a_lo, _mm256_setzero_si256());
        let p2 = _mm256_unpacklo_epi16(a_hi, _mm256_setzero_si256());
        let p3 = _mm256_unpackhi_epi16(a_hi, _mm256_setzero_si256());

        let mut v0 = _mm256_cvtepi32_ps(p0);
        let mut v1 = _mm256_cvtepi32_ps(p1);
        let mut v2 = _mm256_cvtepi32_ps(p2);
        let mut v3 = _mm256_cvtepi32_ps(p3);

        let a0 = _mm256_permutevar8x32_ps(alpha_f32, _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4));
        let a1 = _mm256_permutevar8x32_ps(alpha_f32, _mm256_setr_epi32(1, 1, 1, 1, 5, 5, 5, 5));
        let a2 = _mm256_permutevar8x32_ps(alpha_f32, _mm256_setr_epi32(2, 2, 2, 2, 6, 6, 6, 6));
        let a3 = _mm256_permutevar8x32_ps(alpha_f32, _mm256_setr_epi32(3, 3, 3, 3, 7, 7, 7, 7));

        v0 = _mm256_mul_ps(v0, a0);
        v1 = _mm256_mul_ps(v1, a1);
        v2 = _mm256_mul_ps(v2, a2);
        v3 = _mm256_mul_ps(v3, a3);

        let s0 = _mm256_cvtps_epi32(v0);
        let s1 = _mm256_cvtps_epi32(v1);
        let s2 = _mm256_cvtps_epi32(v2);
        let s3 = _mm256_cvtps_epi32(v3);

        let packed16_0 = _mm256_packus_epi32(s0, s1);
        let packed16_1 = _mm256_packus_epi32(s2, s3);

        let mut packed = _mm256_packus_epi16(packed16_0, packed16_1);
        packed = _mm256_blendv_epi8(packed, _mm256_setzero_si256(), is_zero_mask);
        packed = _mm256_blendv_epi8(
            packed,
            rgba,
            _mm256_set1_epi32(i32::from_ne_bytes([0, 0, 0, 255])),
        );
        unsafe {
            _mm256_storeu_si256(in_place.as_mut_ptr().cast(), packed);
        }
    }
}

impl DisassociateAlpha for Avx2DisassociateAlpha {
    #[target_feature(enable = "avx2")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        let mut rem = in_place;

        for dst in rem.as_chunks_mut::<32>().0.iter_mut() {
            self.disassociate_chunk(dst);
        }

        rem = rem.as_chunks_mut::<32>().1;

        if !rem.is_empty() {
            const PART_SIZE: usize = 32;
            assert!(rem.len() < PART_SIZE);

            let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];

            buffer[..rem.len()].copy_from_slice(rem);

            self.disassociate_chunk(&mut buffer);

            rem.copy_from_slice(&buffer[..rem.len()]);
        }
    }
}

#[derive(Default)]
struct Avx2DisassociateAlphaFast {}

impl Avx2DisassociateAlphaFast {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn disassociate_chunk(&self, in_place: &mut [u8; 32]) {
        let alpha_mask = _mm256_setr_epi8(
            3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1, 3, -1, -1, -1, 7, -1, -1,
            -1, 11, -1, -1, -1, 15, -1, -1, -1,
        );

        let rgba = unsafe { _mm256_loadu_si256(in_place.as_ptr().cast()) };

        let alpha_u32 = _mm256_shuffle_epi8(rgba, alpha_mask);
        let is_zero_mask = _mm256_cmpeq_epi32(alpha_u32, _mm256_setzero_si256());
        let alpha_f32 = _mm256_cvtepi32_ps(alpha_u32);

        let recip_f32 = _mm256_mul_ps(_mm256_set1_ps(65536.0), _mm256_rcp_ps(alpha_f32));
        let recip_i32 = _mm256_cvtps_epi32(recip_f32);

        let lo = _mm256_mullo_epi16(
            _mm256_unpacklo_epi8(rgba, _mm256_setzero_si256()),
            _mm256_set1_epi16(255),
        );
        let hi = _mm256_mullo_epi16(
            _mm256_unpackhi_epi8(rgba, _mm256_setzero_si256()),
            _mm256_set1_epi16(255),
        );

        let recip_packed = _mm256_packus_epi32(recip_i32, recip_i32);
        let shuf_lo = _mm256_setr_epi8(
            0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2,
            3, 2, 3,
        );
        let shuf_hi = _mm256_setr_epi8(
            4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6,
            7, 6, 7,
        );

        let recip_lo_full = _mm256_shuffle_epi8(recip_packed, shuf_lo);
        let recip_hi_full = _mm256_shuffle_epi8(recip_packed, shuf_hi);

        let res_lo = _mm256_mulhi_epu16(lo, recip_lo_full);
        let res_hi = _mm256_mulhi_epu16(hi, recip_hi_full);

        let mut packed = _mm256_packus_epi16(res_lo, res_hi);

        packed = _mm256_blendv_epi8(packed, _mm256_setzero_si256(), is_zero_mask);
        packed = _mm256_blendv_epi8(
            packed,
            rgba,
            _mm256_set1_epi32(i32::from_ne_bytes([0, 0, 0, 255])),
        );

        unsafe {
            _mm256_storeu_si256(in_place.as_mut_ptr().cast(), packed);
        }
    }
}

impl DisassociateAlpha for Avx2DisassociateAlphaFast {
    #[target_feature(enable = "avx2")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        let mut rem = in_place;

        for dst in rem.as_chunks_mut::<32>().0.iter_mut() {
            self.disassociate_chunk(dst);
        }

        rem = rem.as_chunks_mut::<32>().1;

        if !rem.is_empty() {
            const PART_SIZE: usize = 32;
            assert!(rem.len() < PART_SIZE);

            let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];

            buffer[..rem.len()].copy_from_slice(rem);

            self.disassociate_chunk(&mut buffer);

            rem.copy_from_slice(&buffer[..rem.len()]);
        }
    }
}

#[target_feature(enable = "avx2")]
fn avx_unpremultiply_alpha_rgba_impl_row(in_place: &mut [u8], executor: impl DisassociateAlpha) {
    unsafe {
        executor.disassociate(in_place);
    }
}
