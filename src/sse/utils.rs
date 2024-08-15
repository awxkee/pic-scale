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

use crate::sse::shuffle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub unsafe fn _mm_prefer_fma_ps<const FMA: bool>(a: __m128, b: __m128, c: __m128) -> __m128 {
    if FMA {
        _mm_fma_psx(a, b, c)
    } else {
        _mm_add_ps(_mm_mul_ps(b, c), a)
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn _mm_fma_psx(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_fmadd_ps(b, c, a)
}

#[inline(always)]
pub unsafe fn sse_deinterleave_rgba_ps(
    v0: __m128,
    v1: __m128,
    v2: __m128,
    v3: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let t02lo = _mm_unpacklo_ps(v0, v2);
    let t13lo = _mm_unpacklo_ps(v1, v3);
    let t02hi = _mm_unpackhi_ps(v0, v2);
    let t13hi = _mm_unpackhi_ps(v1, v3);
    let a = _mm_unpacklo_ps(t02lo, t13lo);
    let b = _mm_unpackhi_ps(t02lo, t13lo);
    let c = _mm_unpacklo_ps(t02hi, t13hi);
    let d = _mm_unpackhi_ps(t02hi, t13hi);
    (a, b, c, d)
}

#[inline(always)]
pub unsafe fn sse_interleave_rgba_ps(
    v0: __m128,
    v1: __m128,
    v2: __m128,
    v3: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let u0 = _mm_unpacklo_ps(v0, v2);
    let u1 = _mm_unpacklo_ps(v1, v3);
    let u2 = _mm_unpackhi_ps(v0, v2);
    let u3 = _mm_unpackhi_ps(v1, v3);
    let j0 = _mm_unpacklo_ps(u0, u1);
    let j2 = _mm_unpacklo_ps(u2, u3);
    let j1 = _mm_unpackhi_ps(u0, u1);
    let j3 = _mm_unpackhi_ps(u2, u3);

    (j0, j1, j2, j3)
}

#[inline(always)]
pub unsafe fn sse_deinterleave_rgba(
    rgba0: __m128i,
    rgba1: __m128i,
    rgba2: __m128i,
    rgba3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_unpacklo_epi8(rgba0, rgba1); // r1 R1 g1 G1 b1 B1 a1 A1 r2 R2 g2 G2 b2 B2 a2 A2
    let t1 = _mm_unpackhi_epi8(rgba0, rgba1);
    let t2 = _mm_unpacklo_epi8(rgba2, rgba3); // r4 R4 g4 G4 b4 B4 a4 A4 r5 R5 g5 G5 b5 B5 a5 A5
    let t3 = _mm_unpackhi_epi8(rgba2, rgba3);

    let t4 = _mm_unpacklo_epi16(t0, t2); // r1 R1 r4 R4 g1 G1 G4 g4 G4 b1 B1 b4 B4 a1 A1 a4 A4
    let t5 = _mm_unpackhi_epi16(t0, t2);
    let t6 = _mm_unpacklo_epi16(t1, t3);
    let t7 = _mm_unpackhi_epi16(t1, t3);

    let l1 = _mm_unpacklo_epi32(t4, t6); // r1 R1 r4 R4 g1 G1 G4 g4 G4 b1 B1 b4 B4 a1 A1 a4 A4
    let l2 = _mm_unpackhi_epi32(t4, t6);
    let l3 = _mm_unpacklo_epi32(t5, t7);
    let l4 = _mm_unpackhi_epi32(t5, t7);

    #[rustfmt::skip]
    let shuffle = _mm_setr_epi8(0, 4, 8, 12,
                                    1, 5, 9, 13,
                                    2, 6, 10, 14,
                                    3, 7, 11, 15,
    );

    let r1 = _mm_shuffle_epi8(_mm_unpacklo_epi32(l1, l3), shuffle);
    let r2 = _mm_shuffle_epi8(_mm_unpackhi_epi32(l1, l3), shuffle);
    let r3 = _mm_shuffle_epi8(_mm_unpacklo_epi32(l2, l4), shuffle);
    let r4 = _mm_shuffle_epi8(_mm_unpackhi_epi32(l2, l4), shuffle);

    (r1, r2, r3, r4)
}

#[inline(always)]
pub unsafe fn sse_interleave_rgba(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    a: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let rg_lo = _mm_unpacklo_epi8(r, g);
    let rg_hi = _mm_unpackhi_epi8(r, g);
    let ba_lo = _mm_unpacklo_epi8(b, a);
    let ba_hi = _mm_unpackhi_epi8(b, a);

    let rgba_0_lo = _mm_unpacklo_epi16(rg_lo, ba_lo);
    let rgba_0_hi = _mm_unpackhi_epi16(rg_lo, ba_lo);
    let rgba_1_lo = _mm_unpacklo_epi16(rg_hi, ba_hi);
    let rgba_1_hi = _mm_unpackhi_epi16(rg_hi, ba_hi);
    (rgba_0_lo, rgba_0_hi, rgba_1_lo, rgba_1_hi)
}

/// Sums all lanes in float32
#[inline(always)]
pub unsafe fn _mm_hsum_ps(v: __m128) -> f32 {
    let mut shuf = _mm_movehdup_ps(v);
    let mut sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_deinterleave_rgba_epi16(
    rgba0: __m128i,
    rgba1: __m128i,
    rgba2: __m128i,
    rgba3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let v0 = _mm_unpacklo_epi16(rgba0, rgba2); // a0 a4 b0 b4 ...
    let v1 = _mm_unpackhi_epi16(rgba0, rgba2); // a1 a5 b1 b5 ...
    let v2 = _mm_unpacklo_epi16(rgba1, rgba3); // a2 a6 b2 b6 ...
    let v3 = _mm_unpackhi_epi16(rgba1, rgba3); // a3 a7 b3 b7 ...

    let u0 = _mm_unpacklo_epi16(v0, v2); // a0 a2 a4 a6 ...
    let u1 = _mm_unpacklo_epi16(v1, v3); // a1 a3 a5 a7 ...
    let u2 = _mm_unpackhi_epi16(v0, v2); // c0 c2 c4 c6 ...
    let u3 = _mm_unpackhi_epi16(v1, v3); // c1 c3 c5 c7 ...

    let a = _mm_unpacklo_epi16(u0, u1);
    let b = _mm_unpackhi_epi16(u0, u1);
    let c = _mm_unpacklo_epi16(u2, u3);
    let d = _mm_unpackhi_epi16(u2, u3);
    (a, b, c, d)
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_interleave_rgba_epi16(
    a: __m128i,
    b: __m128i,
    c: __m128i,
    d: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    let u0 = _mm_unpacklo_epi16(a, c); // a0 c0 a1 c1 ...
    let u1 = _mm_unpackhi_epi16(a, c); // a4 c4 a5 c5 ...
    let u2 = _mm_unpacklo_epi16(b, d); // b0 d0 b1 d1 ...
    let u3 = _mm_unpackhi_epi16(b, d); // b4 d4 b5 d5 ...

    let v0 = _mm_unpacklo_epi16(u0, u2); // a0 b0 c0 d0 ...
    let v1 = _mm_unpackhi_epi16(u0, u2); // a2 b2 c2 d2 ...
    let v2 = _mm_unpacklo_epi16(u1, u3); // a4 b4 c4 d4 ...
    let v3 = _mm_unpackhi_epi16(u1, u3); // a6 b6 c6 d6 ...
    (v0, v1, v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn _mm_hsum_epi32(x: __m128i) -> i32 {
    const FIRST_MASK: i32 = shuffle(1, 0, 3, 2);
    let hi64 = _mm_shuffle_epi32::<FIRST_MASK>(x);
    let sum64 = _mm_add_epi32(hi64, x);
    const SM: i32 = shuffle(1, 0, 3, 2);
    let hi32 = _mm_shufflelo_epi16::<SM>(sum64);
    let sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

#[inline(always)]
pub(crate) unsafe fn _mm_muladd_epi32(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    _mm_add_epi32(a, _mm_mullo_epi32(b, c))
}

#[inline]
/// Arithmetic shift for i64, shifting with sign bits
pub unsafe fn _mm_srai_epi64x<const IMM8: i32>(a: __m128i) -> __m128i {
    let m = _mm_set1_epi64x(1 << (64 - 1));
    let x = _mm_srli_epi64::<IMM8>(a);
    let result = _mm_sub_epi64(_mm_xor_si128(x, m), m); //result = x^m - m
    return result;
}

#[inline]
/// Packs i64 into i32 using truncation
pub(crate) unsafe fn _mm_packus_epi64(a: __m128i, b: __m128i) -> __m128i {
    const SHUFFLE_MASK: i32 = shuffle(3, 1, 2, 0);
    let a = _mm_shuffle_epi32::<SHUFFLE_MASK>(a);
    let b1 = _mm_shuffle_epi32::<SHUFFLE_MASK>(b);
    let moved = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b1)));
    moved
}

#[inline(always)]
/// Extracts i64 value
pub unsafe fn _mm_extract_epi64x<const IMM: i32>(d: __m128i) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        return if IMM == 0 {
            _mm_cvtsi128_si64(d)
        } else {
            _mm_extract_epi64::<IMM>(d)
        };
    }
    #[cfg(target_arch = "x86")]
    {
        let (low, high);
        if IMM == 0 {
            low = _mm_cvtsi128_si32(d);
            high = _mm_cvtsi128_si32(_mm_srli_si128::<4>(d));
        } else {
            low = _mm_cvtsi128_si32(_mm_srli_si128::<8>(d));
            high = _mm_cvtsi128_si32(_mm_srli_si128::<12>(d));
        }
        return ((high as i64) << 32) | low as i64;
    }
}

#[inline]
pub unsafe fn _mm_store3_u16(ptr: *mut u16, a: __m128i) {
    let low_pixel = _mm_extract_epi32::<0>(a);
    (ptr as *mut i32).write_unaligned(low_pixel);
    (ptr as *mut i16)
        .add(2)
        .write_unaligned(_mm_extract_epi16::<2>(a) as i16);
}
