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
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm_rev128_epi32(v: __m128i) -> __m128i {
    unsafe {
        let sh = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        _mm_shuffle_epi8(v, sh)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_unzip_3_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: (__m128i, __m128i),
) -> (__m128i, __m128i, __m128i) {
    unsafe {
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
}

#[inline(always)]
pub(crate) unsafe fn _mm_zip_4_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: (__m128i, __m128i, __m128i, __m128i),
) -> (__m128i, __m128i) {
    unsafe {
        let ar_type: Rgb30 = AR30_TYPE.into();
        match ar_type {
            Rgb30::Ar30 => {
                let mut a0 = _mm_set1_epi32(3 << 30);
                let mut a1 = _mm_set1_epi32(3 << 30);

                let r0 = _mm_slli_epi32::<20>(_mm_unpacklo_epi16(v.2, _mm_setzero_si128()));
                let r1 = _mm_slli_epi32::<20>(_mm_unpackhi_epi16(v.2, _mm_setzero_si128()));

                a0 = _mm_or_si128(a0, r0);
                a1 = _mm_or_si128(a1, r1);

                let j0 = _mm_unpacklo_epi16(v.0, _mm_setzero_si128());
                let j1 = _mm_unpackhi_epi16(v.0, _mm_setzero_si128());

                let g0 = _mm_slli_epi32::<10>(_mm_unpacklo_epi16(v.1, _mm_setzero_si128()));
                let g1 = _mm_slli_epi32::<10>(_mm_unpackhi_epi16(v.1, _mm_setzero_si128()));

                a0 = _mm_or_si128(a0, g0);
                a1 = _mm_or_si128(a1, g1);

                a0 = _mm_or_si128(a0, j0);
                a1 = _mm_or_si128(a1, j1);

                if AR30_ORDER == 0 {
                    (a0, a1)
                } else {
                    (_mm_rev128_epi32(a0), _mm_rev128_epi32(a1))
                }
            }
            Rgb30::Ra30 => {
                let mut a0 = _mm_set1_epi32(3);
                let mut a1 = _mm_set1_epi32(3);

                let j0 = _mm_unpacklo_epi16(v.0, _mm_setzero_si128());
                let j1 = _mm_unpackhi_epi16(v.0, _mm_setzero_si128());
                let j2 = _mm_unpacklo_epi16(v.1, _mm_setzero_si128());
                let j3 = _mm_unpackhi_epi16(v.1, _mm_setzero_si128());

                let r0 = _mm_slli_epi32::<22>(j0);
                let r1 = _mm_slli_epi32::<22>(j1);

                a0 = _mm_or_si128(a0, r0);
                a1 = _mm_or_si128(a1, r1);

                let j4 = _mm_unpacklo_epi16(v.2, _mm_setzero_si128());
                let j5 = _mm_unpackhi_epi16(v.2, _mm_setzero_si128());

                let g0 = _mm_slli_epi32::<12>(j2);
                let g1 = _mm_slli_epi32::<12>(j3);

                a0 = _mm_or_si128(a0, g0);
                a1 = _mm_or_si128(a1, g1);

                a0 = _mm_or_si128(a0, _mm_slli_epi32::<2>(j4));
                a1 = _mm_or_si128(a1, _mm_slli_epi32::<2>(j5));

                if AR30_ORDER == 0 {
                    (a0, a1)
                } else {
                    (_mm_rev128_epi32(a0), _mm_rev128_epi32(a1))
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_extract_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: __m128i,
) -> __m128i {
    unsafe {
        let v_mask = _mm_set1_epi64x(0x3ff);
        let k1 = _mm_srli_epi64::<16>(v);
        let k2 = _mm_srli_epi64::<32>(v);
        let r = _mm_and_si128(v, v_mask);
        let g = _mm_and_si128(k1, v_mask);
        let b = _mm_and_si128(k2, v_mask);

        let ar_type: Rgb30 = AR30_TYPE.into();

        let mut a;

        match ar_type {
            Rgb30::Ar30 => {
                a = _mm_set1_epi64x(3 << 30);
                let j0 = _mm_slli_epi64::<20>(b);
                let j1 = _mm_slli_epi64::<10>(g);
                a = _mm_or_si128(a, j0);
                a = _mm_or_si128(a, j1);
                a = _mm_or_si128(a, r);
            }
            Rgb30::Ra30 => {
                a = _mm_set1_epi64x(3);
                let j0 = _mm_slli_epi64::<2>(b);
                let j1 = _mm_slli_epi64::<12>(g);
                let j2 = _mm_slli_epi64::<22>(r);
                a = _mm_or_si128(a, j0);
                a = _mm_or_si128(a, j1);
                a = _mm_or_si128(a, j2);
            }
        }

        if AR30_ORDER == 1 {
            a = _mm_rev128_epi32(a);
        }
        a
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_ld1_ar30_s16<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    arr: &[u8],
) -> __m128i {
    unsafe {
        let item = u32::from_ne_bytes([
            *arr.get_unchecked(0),
            *arr.get_unchecked(1),
            *arr.get_unchecked(2),
            *arr.get_unchecked(3),
        ]);
        let ar_type: Rgb30 = AR30_TYPE.into();
        let vl = ar_type.unpack::<AR30_ORDER>(item);
        let temp = [vl.0 as i16, vl.1 as i16, vl.2 as i16, 1023];
        _mm_loadu_si64(temp.as_ptr() as *const _)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_unzips_3_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    unsafe {
        let mask = _mm_set1_epi32(0x3ff);
        let ar_type: Rgb30 = AR30_TYPE.into();

        let v = if AR30_ORDER == 0 {
            v
        } else {
            _mm_rev128_epi32(v)
        };

        match ar_type {
            Rgb30::Ar30 => {
                let j0 = _mm_and_si128(v, mask);
                let j1 = _mm_srli_epi32::<10>(v);
                let j2 = _mm_srli_epi32::<20>(v);
                let r = j0;
                let g = _mm_and_si128(j1, mask);
                let b = _mm_and_si128(j2, mask);
                (
                    _mm_packus_epi32(r, r),
                    _mm_packus_epi32(g, g),
                    _mm_packus_epi32(b, b),
                    _mm_set1_epi16(3),
                )
            }
            Rgb30::Ra30 => {
                let j0 = _mm_srli_epi32::<22>(v);
                let j1 = _mm_srli_epi32::<12>(v);
                let j2 = _mm_srli_epi32::<2>(v);
                let r = _mm_and_si128(j0, mask);
                let g = _mm_and_si128(j1, mask);
                let b = _mm_and_si128(j2, mask);
                (
                    _mm_packus_epi32(r, r),
                    _mm_packus_epi32(g, g),
                    _mm_packus_epi32(b, b),
                    _mm_set1_epi16(3),
                )
            }
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_unzips_4_ar30_separate<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: __m128i,
) -> (__m128i, __m128i) {
    unsafe {
        let values = _mm_unzips_3_ar30::<AR30_TYPE, AR30_ORDER>(v);
        let a0 = (
            _mm_unpacklo_epi16(values.0, values.1),
            _mm_unpackhi_epi16(values.0, values.1),
        );
        let a1 = (
            _mm_unpacklo_epi16(values.2, values.3),
            _mm_unpackhi_epi16(values.2, values.3),
        );
        let v1 = (
            _mm_unpacklo_epi32(a0.0, a1.0),
            _mm_unpackhi_epi32(a0.0, a1.0),
        );
        let v2 = (
            _mm_unpacklo_epi32(a0.1, a1.1),
            _mm_unpackhi_epi32(a0.1, a1.1),
        );
        let k0 = v1.0;
        let k1 = v2.0;
        let k2 = v1.1;
        let k3 = v2.1;
        (_mm_unpacklo_epi64(k0, k1), _mm_unpacklo_epi64(k2, k3))
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_unzip_4_ar30_separate<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: (__m128i, __m128i),
) -> (__m128i, __m128i, __m128i, __m128i) {
    unsafe {
        let values = _mm_unzip_3_ar30::<AR30_TYPE, AR30_ORDER>(v);
        let a0 = (
            _mm_unpacklo_epi16(values.0, values.1),
            _mm_unpackhi_epi16(values.0, values.1),
        );
        let a1 = (
            _mm_unpacklo_epi16(values.2, _mm_set1_epi16(3)),
            _mm_unpackhi_epi16(values.2, _mm_set1_epi16(3)),
        );
        let v1 = (
            _mm_unpacklo_epi32(a0.0, a1.0),
            _mm_unpackhi_epi32(a0.0, a1.0),
        );
        let v2 = (
            _mm_unpacklo_epi32(a0.1, a1.1),
            _mm_unpackhi_epi32(a0.1, a1.1),
        );
        let k0 = v1.0;
        let k1 = v2.0;
        let k2 = v1.1;
        let k3 = v2.1;
        (k0, k1, k2, k3)
    }
}
