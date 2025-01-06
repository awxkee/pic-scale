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
use crate::avx512::utils::{avx512_deinterleave_rgba, avx512_div_by255, avx512_interleave_rgba};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

trait AssociateAlpha {
    unsafe fn associate(&self, dst: &mut [u8], src: &[u8]);
}

#[derive(Default)]
struct AssociateAlphaDefault<const HAS_VBMI: bool> {}

impl<const HAS_VBMI: bool> AssociateAlphaDefault<HAS_VBMI> {
    #[inline(always)]
    unsafe fn associate_chunk(&self, dst: &mut [u8], src: &[u8]) {
        let src_ptr = src.as_ptr();
        let rgba0 = _mm512_loadu_si512(src_ptr as *const _);
        let rgba1 = _mm512_loadu_si512(src_ptr.add(64) as *const _);
        let rgba2 = _mm512_loadu_si512(src_ptr.add(128) as *const _);
        let rgba3 = _mm512_loadu_si512(src_ptr.add(128 + 64) as *const _);
        let (rrr, ggg, bbb, aaa) = avx512_deinterleave_rgba::<HAS_VBMI>(rgba0, rgba1, rgba2, rgba3);

        let zeros = _mm512_setzero_si512();

        let mut rrr_low = _mm512_unpacklo_epi8(rrr, zeros);
        let mut rrr_high = _mm512_unpackhi_epi8(rrr, zeros);

        let mut ggg_low = _mm512_unpacklo_epi8(ggg, zeros);
        let mut ggg_high = _mm512_unpackhi_epi8(ggg, zeros);

        let mut bbb_low = _mm512_unpacklo_epi8(bbb, zeros);
        let mut bbb_high = _mm512_unpackhi_epi8(bbb, zeros);

        let aaa_low = _mm512_unpacklo_epi8(aaa, zeros);
        let aaa_high = _mm512_unpackhi_epi8(aaa, zeros);

        rrr_low = avx512_div_by255(_mm512_mullo_epi16(rrr_low, aaa_low));
        rrr_high = avx512_div_by255(_mm512_mullo_epi16(rrr_high, aaa_high));
        ggg_low = avx512_div_by255(_mm512_mullo_epi16(ggg_low, aaa_low));
        ggg_high = avx512_div_by255(_mm512_mullo_epi16(ggg_high, aaa_high));
        bbb_low = avx512_div_by255(_mm512_mullo_epi16(bbb_low, aaa_low));
        bbb_high = avx512_div_by255(_mm512_mullo_epi16(bbb_high, aaa_high));

        let rrr = _mm512_packus_epi16(rrr_low, rrr_high);
        let ggg = _mm512_packus_epi16(ggg_low, ggg_high);
        let bbb = _mm512_packus_epi16(bbb_low, bbb_high);

        let (rgba0, rgba1, rgba2, rgba3) = avx512_interleave_rgba::<HAS_VBMI>(rrr, ggg, bbb, aaa);
        let dst_ptr = dst.as_mut_ptr();
        _mm512_storeu_si512(dst_ptr as *mut _, rgba0);
        _mm512_storeu_si512(dst_ptr.add(64) as *mut _, rgba1);
        _mm512_storeu_si512(dst_ptr.add(128) as *mut _, rgba2);
        _mm512_storeu_si512(dst_ptr.add(128 + 64) as *mut _, rgba3);
    }
}

impl AssociateAlpha for AssociateAlphaDefault<false> {
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn associate(&self, dst: &mut [u8], src: &[u8]) {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .chunks_exact_mut(64 * 4)
            .zip(src_rem.chunks_exact(64 * 4))
        {
            self.associate_chunk(dst, src);
        }

        rem = rem.chunks_exact_mut(64 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(64 * 4).remainder();

        if !rem.is_empty() {
            const PART_SIZE: usize = 64 * 4;
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

impl AssociateAlpha for AssociateAlphaDefault<true> {
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
    unsafe fn associate(&self, dst: &mut [u8], src: &[u8]) {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .chunks_exact_mut(64 * 4)
            .zip(src_rem.chunks_exact(64 * 4))
        {
            self.associate_chunk(dst, src);
        }

        rem = rem.chunks_exact_mut(64 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(64 * 4).remainder();

        if !rem.is_empty() {
            const PART_SIZE: usize = 64 * 4;
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
    pool: &Option<ThreadPool>,
) {
    let has_vbmi = std::arch::is_x86_feature_detected!("avx512vbmi");

    let mut executor: fn(&mut [u8], &[u8]) = |dst: &mut [u8], src: &[u8]| {
        avx_premultiply_alpha_rgba_impl_row(dst, src, AssociateAlphaDefault::<false>::default());
    };
    if has_vbmi {
        executor = |dst: &mut [u8], src: &[u8]| {
            avx_premultiply_alpha_rgba_impl_row(dst, src, AssociateAlphaDefault::<true>::default());
        };
    }

    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(dst_stride)
                .zip(src.par_chunks_exact(src_stride))
                .for_each(|(dst, src)| {
                    executor(&mut dst[..width * 4], &src[..width * 4]);
                });
        });
    } else {
        dst.chunks_exact_mut(dst_stride)
            .zip(src.chunks_exact(src_stride))
            .for_each(|(dst, src)| {
                executor(&mut dst[..width * 4], &src[..width * 4]);
            });
    }
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u8]);
}

#[derive(Default)]
struct Avx512DisassociateAlpha<const HAS_VBMI: bool> {}

impl<const HAS_VBMI: bool> Avx512DisassociateAlpha<HAS_VBMI> {
    #[inline(always)]
    unsafe fn avx512_unpremultiply_row(&self, x: __m512i, a: __m512i) -> __m512i {
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

        const FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
        let lo_lo = _mm512_cvtps_epi32(_mm512_mul_round_ps::<FLAGS>(lo_lo, a_lo_lo));
        let lo_hi = _mm512_cvtps_epi32(_mm512_mul_round_ps::<FLAGS>(lo_hi, a_lo_hi));
        let hi_lo = _mm512_cvtps_epi32(_mm512_mul_round_ps::<FLAGS>(hi_lo, a_hi_lo));
        let hi_hi = _mm512_cvtps_epi32(_mm512_mul_round_ps::<FLAGS>(hi_hi, a_hi_hi));

        let packed = _mm512_packus_epi16(
            _mm512_packus_epi32(lo_lo, lo_hi),
            _mm512_packus_epi32(hi_lo, hi_hi),
        );
        _mm512_mask_blend_epi8(is_zero_mask, packed, _mm512_setzero_si512())
    }

    #[inline(always)]
    unsafe fn disassociate_chunk(&self, in_place: &mut [u8]) {
        let src_ptr = in_place.as_ptr();
        let rgba0 = _mm512_loadu_si512(src_ptr as *const _);
        let rgba1 = _mm512_loadu_si512(src_ptr.add(64) as *const _);
        let rgba2 = _mm512_loadu_si512(src_ptr.add(128) as *const _);
        let rgba3 = _mm512_loadu_si512(src_ptr.add(64 + 128) as *const _);

        let (rrr, ggg, bbb, aaa) = avx512_deinterleave_rgba::<HAS_VBMI>(rgba0, rgba1, rgba2, rgba3);

        let rrr = self.avx512_unpremultiply_row(rrr, aaa);
        let ggg = self.avx512_unpremultiply_row(ggg, aaa);
        let bbb = self.avx512_unpremultiply_row(bbb, aaa);

        let (rgba0, rgba1, rgba2, rgba3) = avx512_interleave_rgba::<HAS_VBMI>(rrr, ggg, bbb, aaa);

        let dst_ptr = in_place.as_mut_ptr();
        _mm512_storeu_si512(dst_ptr as *mut _, rgba0);
        _mm512_storeu_si512(dst_ptr.add(64) as *mut _, rgba1);
        _mm512_storeu_si512(dst_ptr.add(128) as *mut _, rgba2);
        _mm512_storeu_si512(dst_ptr.add(128 + 64) as *mut _, rgba3);
    }
}

impl DisassociateAlpha for Avx512DisassociateAlpha<false> {
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
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

impl DisassociateAlpha for Avx512DisassociateAlpha<true> {
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
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

/// Uses f16 instead of f32 to unpremultiply alpha
#[derive(Default)]
#[cfg(feature = "nightly_avx512fp16")]
struct Avx512DisassociateAlphaFloat16<const HAS_VBMI: bool> {}

#[cfg(feature = "nightly_avx512fp16")]
impl<const HAS_VBMI: bool> Avx512DisassociateAlphaFloat16<HAS_VBMI> {
    #[inline(always)]
    unsafe fn avx512_unpremultiply_row(&self, x: __m512i, a: __m512i) -> __m512i {
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

    #[inline(always)]
    unsafe fn disassociate_chunk(&self, in_place: &mut [u8]) {
        let src_ptr = in_place.as_ptr();
        let rgba0 = _mm512_loadu_si512(src_ptr as *const _);
        let rgba1 = _mm512_loadu_si512(src_ptr.add(64) as *const _);
        let rgba2 = _mm512_loadu_si512(src_ptr.add(128) as *const _);
        let rgba3 = _mm512_loadu_si512(src_ptr.add(64 + 128) as *const _);

        let (rrr, ggg, bbb, aaa) = avx512_deinterleave_rgba::<HAS_VBMI>(rgba0, rgba1, rgba2, rgba3);

        let rrr = self.avx512_unpremultiply_row(rrr, aaa);
        let ggg = self.avx512_unpremultiply_row(ggg, aaa);
        let bbb = self.avx512_unpremultiply_row(bbb, aaa);

        let (rgba0, rgba1, rgba2, rgba3) = avx512_interleave_rgba::<HAS_VBMI>(rrr, ggg, bbb, aaa);

        let dst_ptr = in_place.as_mut_ptr();
        _mm512_storeu_si512(dst_ptr as *mut _, rgba0);
        _mm512_storeu_si512(dst_ptr.add(64) as *mut _, rgba1);
        _mm512_storeu_si512(dst_ptr.add(128) as *mut _, rgba2);
        _mm512_storeu_si512(dst_ptr.add(128 + 64) as *mut _, rgba3);
    }
}

#[cfg(feature = "nightly_avx512fp16")]
impl DisassociateAlpha for Avx512DisassociateAlphaFloat16<false> {
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512fp16")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
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

#[cfg(feature = "nightly_avx512fp16")]
impl DisassociateAlpha for Avx512DisassociateAlphaFloat16<true> {
    #[target_feature(
        enable = "avx512f",
        enable = "avx512bw",
        enable = "avx512vbmi",
        enable = "avx512fp16"
    )]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
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

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn avx512_unp_row(in_place: &mut [u8], executor: impl DisassociateAlpha) {
    executor.disassociate(in_place);
}

pub(crate) fn avx512_unpremultiply_alpha_rgba(
    in_place: &mut [u8],
    width: usize,
    _: usize,
    stride: usize,
    pool: &Option<ThreadPool>,
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

    if let Some(pool) = pool {
        pool.install(|| {
            in_place
                .par_chunks_exact_mut(stride)
                .for_each(|row| unsafe {
                    executor(&mut row[..width * 4]);
                });
        });
    } else {
        in_place.chunks_exact_mut(stride).for_each(|row| unsafe {
            executor(&mut row[..width * 4]);
        });
    }
}
