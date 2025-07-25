/*
 * Copyright (c) Radzivon Bartoshyk 01/2025. All rights reserved.
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
use crate::avx512::utils::{avx512_deinterleave_rgba, avx512_div_by255, avx512_interleave_rgba};
use novtb::{ParallelZonedIterator, TbSliceMut};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

trait AssociateAlpha {
    unsafe fn associate(&self, dst: &mut [u8], src: &[u8]);
}

#[derive(Default)]
struct AssociateAlphaDefault {}

impl AssociateAlphaDefault {
    #[inline(always)]
    unsafe fn associate_chunk(&self, dst: &mut [u8], src: &[u8]) {
        unsafe {
            let working_mask: __mmask64 = if dst.len() == 64 {
                0xffff_ffff_ffff_ffff
            } else {
                0xffff_ffff_ffff_ffff >> (64 - dst.len())
            };

            let shuffle = _mm512_set_epi8(
                63, 63, 63, 63, 59, 59, 59, 59, 55, 55, 55, 55, 51, 51, 51, 51, 47, 47, 47, 47, 43,
                43, 43, 43, 39, 39, 39, 39, 35, 35, 35, 35, 31, 31, 31, 31, 27, 27, 27, 27, 23, 23,
                23, 23, 19, 19, 19, 19, 15, 15, 15, 15, 11, 11, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3,
            );

            let mask: __mmask64 =
                0b0001_0001_0001_0001_0001_0001_0001_0001_0001_0001_0001_0001_0001_0001_0001_0001;

            let src_ptr = src.as_ptr();
            let rgba0 = _mm512_maskz_loadu_epi8(working_mask, src_ptr as *const _);
            let multiplicand = _mm512_shuffle_epi8(rgba0, shuffle);

            let zeros = _mm512_setzero_si512();

            let mut v_ll = _mm512_unpacklo_epi8(rgba0, zeros);
            let mut v_hi = _mm512_unpackhi_epi8(rgba0, zeros);

            let a_lo = _mm512_unpacklo_epi8(multiplicand, zeros);
            let a_hi = _mm512_unpackhi_epi8(multiplicand, zeros);

            v_ll = avx512_div_by255(_mm512_mullo_epi16(v_ll, a_lo));
            v_hi = avx512_div_by255(_mm512_mullo_epi16(v_hi, a_hi));

            let values = _mm512_mask_blend_epi8(mask, _mm512_packus_epi16(v_ll, v_hi), rgba0);

            let dst_ptr = dst.as_mut_ptr();
            _mm512_mask_storeu_epi8(dst_ptr as *mut _, working_mask, values);
        }
    }
}

impl AssociateAlpha for AssociateAlphaDefault {
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn associate(&self, dst: &mut [u8], src: &[u8]) {
        unsafe {
            let mut rem = dst;
            let mut src_rem = src;

            for (dst, src) in rem.chunks_exact_mut(64).zip(src_rem.chunks_exact(64)) {
                self.associate_chunk(dst, src);
            }

            rem = rem.chunks_exact_mut(64).into_remainder();
            src_rem = src_rem.chunks_exact(64).remainder();

            if !rem.is_empty() {
                assert!(rem.len() <= 64);
                assert!(src_rem.len() <= 64);
                self.associate_chunk(rem, src_rem);
            }
        }
    }
}

fn avx_premultiply_alpha_rgba_impl_row(dst: &mut [u8], src: &[u8], executor: impl AssociateAlpha) {
    unsafe {
        executor.associate(dst, src);
    }
}

pub(crate) fn avx512_premultiply_alpha_rgba(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    width: usize,
    _: usize,
    src_stride: usize,
    pool: &novtb::ThreadPool,
) {
    let executor: fn(&mut [u8], &[u8]) = |dst: &mut [u8], src: &[u8]| {
        avx_premultiply_alpha_rgba_impl_row(dst, src, AssociateAlphaDefault::default());
    };

    dst.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| {
            let src = &src[y * src_stride..(y + 1) * src_stride];
            executor(&mut dst[..width * 4], &src[..width * 4]);
        });
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u8]);
}

#[derive(Default)]
struct Avx512DisassociateAlpha<const HAS_VBMI: bool> {}

impl<const HAS_VBMI: bool> Avx512DisassociateAlpha<HAS_VBMI> {
    #[inline(always)]
    unsafe fn avx512_unpremultiply_row(&self, x: __m512i, a: __m512i) -> __m512i {
        unsafe {
            let zeros = _mm512_setzero_si512();
            let lo = _mm512_unpacklo_epi8(x, zeros);
            let hi = _mm512_unpackhi_epi8(x, zeros);

            let is_zero_mask = _mm512_cmp_epi8_mask::<0>(a, zeros);

            let scale_ps = _mm512_set1_ps(255f32);

            let lo_lo = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(lo, zeros)),
                scale_ps,
            );
            let lo_hi = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(lo, zeros)),
                scale_ps,
            );
            let hi_lo = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(hi, zeros)),
                scale_ps,
            );
            let hi_hi = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(hi, zeros)),
                scale_ps,
            );
            let a_lo = _mm512_unpacklo_epi8(a, zeros);
            let a_hi = _mm512_unpackhi_epi8(a, zeros);

            let a_lo_lo = _mm512_rcp14_ps(_mm512_cvtepi32_ps(_mm512_unpacklo_epi16(a_lo, zeros)));
            let a_lo_hi = _mm512_rcp14_ps(_mm512_cvtepi32_ps(_mm512_unpackhi_epi16(a_lo, zeros)));
            let a_hi_lo = _mm512_rcp14_ps(_mm512_cvtepi32_ps(_mm512_unpacklo_epi16(a_hi, zeros)));
            let a_hi_hi = _mm512_rcp14_ps(_mm512_cvtepi32_ps(_mm512_unpackhi_epi16(a_hi, zeros)));

            let lo_lo = _mm512_cvtps_epi32(_mm512_mul_ps(lo_lo, a_lo_lo));
            let lo_hi = _mm512_cvtps_epi32(_mm512_mul_ps(lo_hi, a_lo_hi));
            let hi_lo = _mm512_cvtps_epi32(_mm512_mul_ps(hi_lo, a_hi_lo));
            let hi_hi = _mm512_cvtps_epi32(_mm512_mul_ps(hi_hi, a_hi_hi));

            let packed = _mm512_packus_epi16(
                _mm512_packus_epi32(lo_lo, lo_hi),
                _mm512_packus_epi32(hi_lo, hi_hi),
            );
            _mm512_mask_blend_epi8(is_zero_mask, packed, _mm512_setzero_si512())
        }
    }

    #[inline(always)]
    unsafe fn disassociate_chunk(&self, in_place: &mut [u8]) {
        unsafe {
            let src_ptr = in_place.as_ptr();
            let rgba0 = _mm512_loadu_si512(src_ptr as *const _);
            let rgba1 = _mm512_loadu_si512(src_ptr.add(64) as *const _);
            let rgba2 = _mm512_loadu_si512(src_ptr.add(128) as *const _);
            let rgba3 = _mm512_loadu_si512(src_ptr.add(64 + 128) as *const _);

            let (rrr, ggg, bbb, aaa) =
                avx512_deinterleave_rgba::<HAS_VBMI>(rgba0, rgba1, rgba2, rgba3);

            let rrr = self.avx512_unpremultiply_row(rrr, aaa);
            let ggg = self.avx512_unpremultiply_row(ggg, aaa);
            let bbb = self.avx512_unpremultiply_row(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) =
                avx512_interleave_rgba::<HAS_VBMI>(rrr, ggg, bbb, aaa);

            let dst_ptr = in_place.as_mut_ptr();
            _mm512_storeu_si512(dst_ptr as *mut _, rgba0);
            _mm512_storeu_si512(dst_ptr.add(64) as *mut _, rgba1);
            _mm512_storeu_si512(dst_ptr.add(128) as *mut _, rgba2);
            _mm512_storeu_si512(dst_ptr.add(128 + 64) as *mut _, rgba3);
        }
    }
}

impl DisassociateAlpha for Avx512DisassociateAlpha<false> {
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        unsafe {
            let mut rem = in_place;

            for dst in rem.chunks_exact_mut(64 * 4) {
                self.disassociate_chunk(dst);
            }

            rem = rem.chunks_exact_mut(64 * 4).into_remainder();

            if !rem.is_empty() {
                const PART_SIZE: usize = 64 * 4;
                assert!(rem.len() < PART_SIZE);

                let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];

                std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

                self.disassociate_chunk(&mut buffer);

                std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }
}

impl DisassociateAlpha for Avx512DisassociateAlpha<true> {
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        unsafe {
            let mut rem = in_place;

            for dst in rem.chunks_exact_mut(64 * 4) {
                self.disassociate_chunk(dst);
            }

            rem = rem.chunks_exact_mut(64 * 4).into_remainder();

            if !rem.is_empty() {
                const PART_SIZE: usize = 64 * 4;
                assert!(rem.len() < PART_SIZE);

                let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];

                std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

                self.disassociate_chunk(&mut buffer);

                std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }
}

/// Uses f16 instead of f32 to unpremultiply alpha
#[derive(Default)]
#[cfg(feature = "nightly_avx512fp16")]
struct Avx512DisassociateAlphaFloat16<const HAS_VBMI: bool> {}

#[cfg(feature = "nightly_avx512fp16")]
impl<const HAS_VBMI: bool> Avx512DisassociateAlphaFloat16<HAS_VBMI> {
    #[inline(always)]
    unsafe fn avx512_unpremultiply_row(&self, x: __m512i, a: __m512i) -> __m512i {
        unsafe {
            let zeros = _mm512_setzero_si512();
            let lo = _mm512_unpacklo_epi8(x, zeros);
            let hi = _mm512_unpackhi_epi8(x, zeros);

            let is_zero_mask = _mm512_cmp_epi8_mask::<0>(a, zeros);

            let scale_ps = _mm512_castsi512_ph(_mm512_set1_epi16(23544));

            let lo = _mm512_mul_ph(_mm512_cvtepu16_ph(lo), scale_ps);
            let hi = _mm512_mul_ph(_mm512_cvtepu16_ph(hi), scale_ps);

            let a_lo = _mm512_unpacklo_epi8(a, zeros);
            let a_hi = _mm512_unpackhi_epi8(a, zeros);

            let a_lo = _mm512_rcp_ph(_mm512_cvtepu16_ph(a_lo));
            let a_hi = _mm512_rcp_ph(_mm512_cvtepu16_ph(a_hi));
            const FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
            let lo = _mm512_cvtph_epi16(_mm512_mul_round_ph::<FLAGS>(lo, a_lo));
            let hi = _mm512_cvtph_epi16(_mm512_mul_round_ph::<FLAGS>(hi, a_hi));

            let packed = _mm512_packus_epi16(lo, hi);
            _mm512_mask_blend_epi8(is_zero_mask, packed, _mm512_setzero_si512())
        }
    }

    #[inline(always)]
    unsafe fn disassociate_chunk(&self, in_place: &mut [u8]) {
        unsafe {
            let src_ptr = in_place.as_ptr();
            let rgba0 = _mm512_loadu_si512(src_ptr as *const _);
            let rgba1 = _mm512_loadu_si512(src_ptr.add(64) as *const _);
            let rgba2 = _mm512_loadu_si512(src_ptr.add(128) as *const _);
            let rgba3 = _mm512_loadu_si512(src_ptr.add(64 + 128) as *const _);

            let (rrr, ggg, bbb, aaa) =
                avx512_deinterleave_rgba::<HAS_VBMI>(rgba0, rgba1, rgba2, rgba3);

            let rrr = self.avx512_unpremultiply_row(rrr, aaa);
            let ggg = self.avx512_unpremultiply_row(ggg, aaa);
            let bbb = self.avx512_unpremultiply_row(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) =
                avx512_interleave_rgba::<HAS_VBMI>(rrr, ggg, bbb, aaa);

            let dst_ptr = in_place.as_mut_ptr();
            _mm512_storeu_si512(dst_ptr as *mut _, rgba0);
            _mm512_storeu_si512(dst_ptr.add(64) as *mut _, rgba1);
            _mm512_storeu_si512(dst_ptr.add(128) as *mut _, rgba2);
            _mm512_storeu_si512(dst_ptr.add(128 + 64) as *mut _, rgba3);
        }
    }
}

#[cfg(feature = "nightly_avx512fp16")]
impl DisassociateAlpha for Avx512DisassociateAlphaFloat16<false> {
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512fp16")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        unsafe {
            let mut rem = in_place;

            for dst in rem.chunks_exact_mut(64 * 4) {
                self.disassociate_chunk(dst);
            }

            rem = rem.chunks_exact_mut(64 * 4).into_remainder();

            if !rem.is_empty() {
                const PART_SIZE: usize = 64 * 4;
                assert!(rem.len() < PART_SIZE);

                let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];

                std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

                self.disassociate_chunk(&mut buffer);

                std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }
}

#[cfg(feature = "nightly_avx512fp16")]
impl DisassociateAlpha for Avx512DisassociateAlphaFloat16<true> {
    #[target_feature(
        enable = "avx512f",
        enable = "avx512bw",
        enable = "avx512vbmi",
        enable = "avx512fp16"
    )]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        unsafe {
            let mut rem = in_place;

            for dst in rem.chunks_exact_mut(64 * 4) {
                self.disassociate_chunk(dst);
            }

            rem = rem.chunks_exact_mut(64 * 4).into_remainder();

            if !rem.is_empty() {
                const PART_SIZE: usize = 64 * 4;
                assert!(rem.len() < PART_SIZE);

                let mut buffer: [u8; PART_SIZE] = [0u8; PART_SIZE];

                std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

                self.disassociate_chunk(&mut buffer);

                std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn avx512_unp_row(in_place: &mut [u8], executor: impl DisassociateAlpha) {
    unsafe {
        executor.disassociate(in_place);
    }
}

pub(crate) fn avx512_unpremultiply_alpha_rgba(
    in_place: &mut [u8],
    width: usize,
    _: usize,
    stride: usize,
    pool: &novtb::ThreadPool,
    _: WorkloadStrategy,
) {
    let has_vbmi = std::arch::is_x86_feature_detected!("avx512vbmi");
    let mut executor: fn(&mut [u8]) = |row: &mut [u8]| unsafe {
        avx512_unp_row(row, Avx512DisassociateAlpha::<false>::default());
    };
    if has_vbmi {
        executor = |row: &mut [u8]| unsafe {
            avx512_unp_row(row, Avx512DisassociateAlpha::<true>::default());
        };
    }
    #[cfg(feature = "nightly_avx512fp16")]
    {
        let has_fp16 = std::arch::is_x86_feature_detected!("avx512fp16");
        if has_fp16 && has_vbmi {
            executor = |row: &mut [u8]| unsafe {
                avx512_unp_row(row, Avx512DisassociateAlphaFloat16::<true>::default());
            };
        } else if has_fp16 {
            executor = |row: &mut [u8]| unsafe {
                avx512_unp_row(row, Avx512DisassociateAlphaFloat16::<false>::default());
            };
        }
    }

    in_place
        .tb_par_chunks_exact_mut(stride)
        .for_each(pool, |row| {
            executor(&mut row[..width * 4]);
        });
}
