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

use crate::ar30::Rgb30;
use crate::support::PRECISION;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) fn compress_i32(x: __m128i) -> __m128i {
    unsafe {
        let store_32 = _mm_srai_epi32::<PRECISION>(x);
        _mm_packus_epi32(store_32, store_32)
    }
}

#[inline]
pub(crate) unsafe fn convolve_horizontal_parts_one_sse_rgb(
    start_x: usize,
    src: &[u8],
    weight0: __m128i,
    store_0: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..).as_ptr();
    let base_pixel = _mm_loadu_si16(src_ptr);
    let m_vl = _mm_insert_epi8::<2>(base_pixel, src_ptr.add(2).read_unaligned() as i32);
    let lo = _mm_unpacklo_epi8(m_vl, _mm_setzero_si128());
    _mm_add_epi32(
        store_0,
        _mm_madd_epi16(_mm_unpacklo_epi16(lo, _mm_setzero_si128()), weight0),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_rev128_epi32(v: __m128i) -> __m128i {
    let sh = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    _mm_shuffle_epi8(v, sh)
}

#[inline(always)]
pub(crate) unsafe fn _mm_unzip_3_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: (__m128i, __m128i),
) -> (__m128i, __m128i, __m128i) {
    let mask = _mm_set1_epi32(0x3ff);
    let ar_type: Rgb30 = AR30_TYPE.into();

    let v = if AR30_ORDER == 0 {
        v
    } else {
        (_mm_rev128_epi32(v.0), _mm_rev128_epi32(v.1))
    };

    match ar_type {
        Rgb30::Ar30 => {
            let r0 = _mm_and_si128(v.0, mask);
            let r1 = _mm_and_si128(v.1, mask);
            let g0 = _mm_srli_epi32::<10>(v.0);
            let g1 = _mm_srli_epi32::<10>(v.1);
            let b0 = _mm_srli_epi32::<20>(v.0);
            let b1 = _mm_srli_epi32::<20>(v.1);
            let r = _mm_packus_epi32(r0, r1);
            let g = _mm_packus_epi32(_mm_and_si128(g0, mask), _mm_and_si128(g1, mask));
            let b = _mm_packus_epi32(_mm_and_si128(b0, mask), _mm_and_si128(b1, mask));
            (r, g, b)
        }
        Rgb30::Ra30 => {
            let r0 = _mm_srli_epi32::<22>(v.0);
            let r1 = _mm_srli_epi32::<22>(v.1);
            let g0 = _mm_srli_epi32::<12>(v.0);
            let g1 = _mm_srli_epi32::<12>(v.1);
            let b0 = _mm_srli_epi32::<2>(v.0);
            let b1 = _mm_srli_epi32::<2>(v.1);
            let r = _mm_packus_epi32(_mm_and_si128(r0, mask), _mm_and_si128(r1, mask));
            let g = _mm_packus_epi32(_mm_and_si128(g0, mask), _mm_and_si128(g1, mask));
            let b = _mm_packus_epi32(_mm_and_si128(b0, mask), _mm_and_si128(b1, mask));
            (r, g, b)
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_zip_4_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: (__m128i, __m128i, __m128i, __m128i),
) -> (__m128i, __m128i) {
    let ar_type: Rgb30 = AR30_TYPE.into();
    match ar_type {
        Rgb30::Ar30 => {
            let mut a0 = _mm_set1_epi32(3 << 30);
            let mut a1 = _mm_set1_epi32(3 << 30);

            let r0 = _mm_slli_epi32::<20>(_mm_unpacklo_epi16(v.2, _mm_setzero_si128()));
            let r1 = _mm_slli_epi32::<20>(_mm_unpackhi_epi16(v.2, _mm_setzero_si128()));

            a0 = _mm_or_si128(a0, r0);
            a1 = _mm_or_si128(a1, r1);

            let g0 = _mm_slli_epi32::<10>(_mm_unpacklo_epi16(v.1, _mm_setzero_si128()));
            let g1 = _mm_slli_epi32::<10>(_mm_unpackhi_epi16(v.1, _mm_setzero_si128()));

            a0 = _mm_or_si128(a0, g0);
            a1 = _mm_or_si128(a1, g1);

            a0 = _mm_or_si128(a0, _mm_unpacklo_epi16(v.0, _mm_setzero_si128()));
            a1 = _mm_or_si128(a1, _mm_unpackhi_epi16(v.0, _mm_setzero_si128()));

            if AR30_ORDER == 0 {
                (a0, a1)
            } else {
                (_mm_rev128_epi32(a0), _mm_rev128_epi32(a1))
            }
        }
        Rgb30::Ra30 => {
            let mut a0 = _mm_set1_epi32(3);
            let mut a1 = _mm_set1_epi32(3);

            let r0 = _mm_slli_epi32::<22>(_mm_unpacklo_epi16(v.0, _mm_setzero_si128()));
            let r1 = _mm_slli_epi32::<22>(_mm_unpackhi_epi16(v.0, _mm_setzero_si128()));

            a0 = _mm_or_si128(a0, r0);
            a1 = _mm_or_si128(a1, r1);

            let g0 = _mm_slli_epi32::<12>(_mm_unpacklo_epi16(v.1, _mm_setzero_si128()));
            let g1 = _mm_slli_epi32::<12>(_mm_unpackhi_epi16(v.1, _mm_setzero_si128()));

            a0 = _mm_or_si128(a0, g0);
            a1 = _mm_or_si128(a1, g1);

            a0 = _mm_or_si128(
                a0,
                _mm_slli_epi32::<2>(_mm_unpacklo_epi16(v.2, _mm_setzero_si128())),
            );
            a1 = _mm_or_si128(
                a1,
                _mm_slli_epi32::<2>(_mm_unpackhi_epi16(v.2, _mm_setzero_si128())),
            );

            if AR30_ORDER == 0 {
                (a0, a1)
            } else {
                (_mm_rev128_epi32(a0), _mm_rev128_epi32(a1))
            }
        }
    }
}
