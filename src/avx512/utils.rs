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
use crate::avx512::avx512_setr::{_v512_set_epu32, _v512_set_epu8};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn avx512_deinterleave_rgba<const HAS_VBMI: bool>(
    bgra0: __m512i,
    bgra1: __m512i,
    bgra2: __m512i,
    bgra3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    if HAS_VBMI {
        let mask0 = _v512_set_epu8(
            126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92,
            90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48,
            46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2,
            0,
        );
        let mask1 = _v512_set_epu8(
            127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 97, 95, 93,
            91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 63, 61, 59, 57, 55, 53, 51, 49,
            47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3,
            1,
        );

        let br01 = _mm512_permutex2var_epi8(bgra0, mask0, bgra1);
        let ga01 = _mm512_permutex2var_epi8(bgra0, mask1, bgra1);
        let br23 = _mm512_permutex2var_epi8(bgra2, mask0, bgra3);
        let ga23 = _mm512_permutex2var_epi8(bgra2, mask1, bgra3);

        let a = _mm512_permutex2var_epi8(br01, mask0, br23);
        let c = _mm512_permutex2var_epi8(br01, mask1, br23);
        let b = _mm512_permutex2var_epi8(ga01, mask0, ga23);
        let d = _mm512_permutex2var_epi8(ga01, mask1, ga23);
        (a, b, c, d)
    } else {
        let mask = _mm512_set4_epi32(0x0f0b0703, 0x0e0a0602, 0x0d090501, 0x0c080400);
        let b0g0r0a0 = _mm512_shuffle_epi8(bgra0, mask);
        let b1g1r1a1 = _mm512_shuffle_epi8(bgra1, mask);
        let b2g2r2a2 = _mm512_shuffle_epi8(bgra2, mask);
        let b3g3r3a3 = _mm512_shuffle_epi8(bgra3, mask);

        let mask0 = _v512_set_epu32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
        let mask1 = _v512_set_epu32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

        let br01 = _mm512_permutex2var_epi32(b0g0r0a0, mask0, b1g1r1a1);
        let ga01 = _mm512_permutex2var_epi32(b0g0r0a0, mask1, b1g1r1a1);
        let br23 = _mm512_permutex2var_epi32(b2g2r2a2, mask0, b3g3r3a3);
        let ga23 = _mm512_permutex2var_epi32(b2g2r2a2, mask1, b3g3r3a3);

        let a = _mm512_permutex2var_epi32(br01, mask0, br23);
        let c = _mm512_permutex2var_epi32(br01, mask1, br23);
        let b = _mm512_permutex2var_epi32(ga01, mask0, ga23);
        let d = _mm512_permutex2var_epi32(ga01, mask1, ga23);
        (a, b, c, d)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_zip_epi8<const HAS_VBMI: bool>(
    a: __m512i,
    b: __m512i,
) -> (__m512i, __m512i) {
    if HAS_VBMI {
        let mask0 = _v512_set_epu8(
            95, 31, 94, 30, 93, 29, 92, 28, 91, 27, 90, 26, 89, 25, 88, 24, 87, 23, 86, 22, 85, 21,
            84, 20, 83, 19, 82, 18, 81, 17, 80, 16, 79, 15, 78, 14, 77, 13, 76, 12, 75, 11, 74, 10,
            73, 9, 72, 8, 71, 7, 70, 6, 69, 5, 68, 4, 67, 3, 66, 2, 65, 1, 64, 0,
        );
        let ab0 = _mm512_permutex2var_epi8(a, mask0, b);
        let mask1 = _v512_set_epu8(
            127, 63, 126, 62, 125, 61, 124, 60, 123, 59, 122, 58, 121, 57, 120, 56, 119, 55, 118,
            54, 117, 53, 116, 52, 115, 51, 114, 50, 113, 49, 112, 48, 111, 47, 110, 46, 109, 45,
            108, 44, 107, 43, 106, 42, 105, 41, 104, 40, 103, 39, 102, 38, 101, 37, 100, 36, 99,
            35, 98, 34, 97, 33, 96, 32,
        );
        let ab1 = _mm512_permutex2var_epi8(a, mask1, b);
        (ab0, ab1)
    } else {
        let low = _mm512_unpacklo_epi8(a, b);
        let high = _mm512_unpackhi_epi8(a, b);
        let ab0 = _mm512_permutex2var_epi64(low, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), high);
        let ab1 =
            _mm512_permutex2var_epi64(low, _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), high);
        (ab0, ab1)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_interleave_rgba<const HAS_VBMI: bool>(
    a: __m512i,
    b: __m512i,
    c: __m512i,
    d: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let (br01, br23) = avx512_zip_epi8::<HAS_VBMI>(a, c);
    let (ga01, ga23) = avx512_zip_epi8::<HAS_VBMI>(b, d);
    let (bgra0, bgra1) = avx512_zip_epi8::<HAS_VBMI>(br01, ga01);
    let (bgra2, bgra3) = avx512_zip_epi8::<HAS_VBMI>(br23, ga23);
    (bgra0, bgra1, bgra2, bgra3)
}

/// Exact division by 255 with rounding to nearest
#[inline(always)]
pub(crate) unsafe fn avx512_div_by255(v: __m512i) -> __m512i {
    let addition = _mm512_set1_epi16(127);
    _mm512_srli_epi16::<8>(_mm512_add_epi16(
        _mm512_add_epi16(v, addition),
        _mm512_srli_epi16::<8>(v),
    ))
}
