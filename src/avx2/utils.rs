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

use std::arch::x86_64::*;

#[inline]
pub(crate) unsafe fn _mm256_fma_ps<const FMA: bool>(a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe {
        if FMA {
            _mm256_fma_psx(a, b, c)
        } else {
            _mm256_add_ps(_mm256_mul_ps(b, c), a)
        }
    }
}

#[inline(always)]
unsafe fn _mm256_fma_psx(a: __m256, b: __m256, c: __m256) -> __m256 {
    unsafe { _mm256_fmadd_ps(b, c, a) }
}

#[inline(always)]
pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub(crate) unsafe fn _mm256_select_si256(
    mask: __m256i,
    true_vals: __m256i,
    false_vals: __m256i,
) -> __m256i {
    unsafe { _mm256_blendv_epi8(false_vals, true_vals, mask) }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_selecti_ps(
    mask: __m256i,
    true_vals: __m256,
    false_vals: __m256,
) -> __m256 {
    unsafe { _mm256_blendv_ps(false_vals, true_vals, _mm256_castsi256_ps(mask)) }
}

/// Exact division by 255 with rounding to nearest
#[inline(always)]
pub(crate) unsafe fn avx2_div_by255(v: __m256i) -> __m256i {
    unsafe {
        let addition = _mm256_set1_epi16(127);
        let j0 = _mm256_add_epi16(v, addition);
        let j1 = _mm256_srli_epi16::<8>(v);
        _mm256_srli_epi16::<8>(_mm256_add_epi16(j0, j1))
    }
}

#[inline(always)]
pub(crate) unsafe fn avx2_deinterleave_rgba(
    rgba0: __m256i,
    rgba1: __m256i,
    rgba2: __m256i,
    rgba3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    unsafe {
        #[rustfmt::skip]
    let sh = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5,
        9, 13, 2, 6, 10, 14,
        3, 7, 11, 15, 0, 4,
        8, 12, 1, 5, 9, 13,
        2, 6, 10, 14, 3, 7,
        11, 15,
    );

        let p0 = _mm256_shuffle_epi8(rgba0, sh);
        let p1 = _mm256_shuffle_epi8(rgba1, sh);
        let p2 = _mm256_shuffle_epi8(rgba2, sh);
        let p3 = _mm256_shuffle_epi8(rgba3, sh);

        let p01l = _mm256_unpacklo_epi32(p0, p1);
        let p01h = _mm256_unpackhi_epi32(p0, p1);
        let p23l = _mm256_unpacklo_epi32(p2, p3);
        let p23h = _mm256_unpackhi_epi32(p2, p3);

        let pll = _mm256_permute2x128_si256::<32>(p01l, p23l);
        let plh = _mm256_permute2x128_si256::<49>(p01l, p23l);
        let phl = _mm256_permute2x128_si256::<32>(p01h, p23h);
        let phh = _mm256_permute2x128_si256::<49>(p01h, p23h);

        let b0 = _mm256_unpacklo_epi32(pll, plh);
        let g0 = _mm256_unpackhi_epi32(pll, plh);
        let r0 = _mm256_unpacklo_epi32(phl, phh);
        let a0 = _mm256_unpackhi_epi32(phl, phh);

        (b0, g0, r0, a0)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx_deinterleave_rgba_epi32(
    p0: __m256i,
    p1: __m256i,
    p2: __m256i,
    p3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    unsafe {
        let p01l = _mm256_unpacklo_epi32(p0, p1);
        let p01h = _mm256_unpackhi_epi32(p0, p1);
        let p23l = _mm256_unpacklo_epi32(p2, p3);
        let p23h = _mm256_unpackhi_epi32(p2, p3);

        let pll = _mm256_permute2x128_si256::<32>(p01l, p23l);
        let plh = _mm256_permute2x128_si256::<49>(p01l, p23l);
        let phl = _mm256_permute2x128_si256::<32>(p01h, p23h);
        let phh = _mm256_permute2x128_si256::<49>(p01h, p23h);

        let b0 = _mm256_unpacklo_epi32(pll, plh);
        let g0 = _mm256_unpackhi_epi32(pll, plh);
        let r0 = _mm256_unpacklo_epi32(phl, phh);
        let a0 = _mm256_unpackhi_epi32(phl, phh);
        (b0, g0, r0, a0)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx_interleave_rgba_epi32(
    p0: __m256i,
    p1: __m256i,
    p2: __m256i,
    p3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    unsafe {
        let bg0 = _mm256_unpacklo_epi32(p0, p1);
        let bg1 = _mm256_unpackhi_epi32(p0, p1);
        let ra0 = _mm256_unpacklo_epi32(p2, p3);
        let ra1 = _mm256_unpackhi_epi32(p2, p3);

        let bgra0_ = _mm256_unpacklo_epi64(bg0, ra0);
        let bgra1_ = _mm256_unpackhi_epi64(bg0, ra0);
        let bgra2_ = _mm256_unpacklo_epi64(bg1, ra1);
        let bgra3_ = _mm256_unpackhi_epi64(bg1, ra1);

        let bgra0 = _mm256_permute2x128_si256::<32>(bgra0_, bgra1_);
        let bgra2 = _mm256_permute2x128_si256::<49>(bgra0_, bgra1_);
        let bgra1 = _mm256_permute2x128_si256::<32>(bgra2_, bgra3_);
        let bgra3 = _mm256_permute2x128_si256::<49>(bgra2_, bgra3_);

        (bgra0, bgra1, bgra2, bgra3)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx_interleave_rgba_epi16(
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    unsafe {
        let bg0 = _mm256_unpacklo_epi16(a, b);
        let bg1 = _mm256_unpackhi_epi16(a, b);
        let ra0 = _mm256_unpacklo_epi16(c, d);
        let ra1 = _mm256_unpackhi_epi16(c, d);

        let bgra0_ = _mm256_unpacklo_epi32(bg0, ra0);
        let bgra1_ = _mm256_unpackhi_epi32(bg0, ra0);
        let bgra2_ = _mm256_unpacklo_epi32(bg1, ra1);
        let bgra3_ = _mm256_unpackhi_epi32(bg1, ra1);

        let bgra0 = _mm256_permute2x128_si256::<32>(bgra0_, bgra1_);
        let bgra2 = _mm256_permute2x128_si256::<49>(bgra0_, bgra1_);
        let bgra1 = _mm256_permute2x128_si256::<32>(bgra2_, bgra3_);
        let bgra3 = _mm256_permute2x128_si256::<49>(bgra2_, bgra3_);
        (bgra0, bgra1, bgra2, bgra3)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx_deinterleave_rgba_epi16(
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    unsafe {
        let sh = _mm256_setr_epi8(
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5,
            12, 13, 6, 7, 14, 15,
        );
        let p0 = _mm256_shuffle_epi8(a, sh);
        let p1 = _mm256_shuffle_epi8(b, sh);
        let p2 = _mm256_shuffle_epi8(c, sh);
        let p3 = _mm256_shuffle_epi8(d, sh);

        let p01l = _mm256_unpacklo_epi32(p0, p1);
        let p01h = _mm256_unpackhi_epi32(p0, p1);
        let p23l = _mm256_unpacklo_epi32(p2, p3);
        let p23h = _mm256_unpackhi_epi32(p2, p3);

        let pll = _mm256_permute2x128_si256::<32>(p01l, p23l);
        let plh = _mm256_permute2x128_si256::<49>(p01l, p23l);
        let phl = _mm256_permute2x128_si256::<32>(p01h, p23h);
        let phh = _mm256_permute2x128_si256::<49>(p01h, p23h);

        let b0 = _mm256_unpacklo_epi32(pll, plh);
        let g0 = _mm256_unpackhi_epi32(pll, plh);
        let r0 = _mm256_unpacklo_epi32(phl, phh);
        let a0 = _mm256_unpackhi_epi32(phl, phh);
        (b0, g0, r0, a0)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx_deinterleave_rgba_ps(
    p0: __m256,
    p1: __m256,
    p2: __m256,
    p3: __m256,
) -> (__m256, __m256, __m256, __m256) {
    unsafe {
        let reshaped = avx_deinterleave_rgba_epi32(
            _mm256_castps_si256(p0),
            _mm256_castps_si256(p1),
            _mm256_castps_si256(p2),
            _mm256_castps_si256(p3),
        );
        (
            _mm256_castsi256_ps(reshaped.0),
            _mm256_castsi256_ps(reshaped.1),
            _mm256_castsi256_ps(reshaped.2),
            _mm256_castsi256_ps(reshaped.3),
        )
    }
}

#[inline(always)]
pub(crate) unsafe fn avx_interleave_rgba_ps(
    p0: __m256,
    p1: __m256,
    p2: __m256,
    p3: __m256,
) -> (__m256, __m256, __m256, __m256) {
    unsafe {
        let reshaped = avx_interleave_rgba_epi32(
            _mm256_castps_si256(p0),
            _mm256_castps_si256(p1),
            _mm256_castps_si256(p2),
            _mm256_castps_si256(p3),
        );
        (
            _mm256_castsi256_ps(reshaped.0),
            _mm256_castsi256_ps(reshaped.1),
            _mm256_castsi256_ps(reshaped.2),
            _mm256_castsi256_ps(reshaped.3),
        )
    }
}

#[inline(always)]
pub(crate) unsafe fn avx2_interleave_rgba(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    a: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    unsafe {
        let bg0 = _mm256_unpacklo_epi8(r, g);
        let bg1 = _mm256_unpackhi_epi8(r, g);
        let ra0 = _mm256_unpacklo_epi8(b, a);
        let ra1 = _mm256_unpackhi_epi8(b, a);

        let rgba0_ = _mm256_unpacklo_epi16(bg0, ra0);
        let rgba1_ = _mm256_unpackhi_epi16(bg0, ra0);
        let rgba2_ = _mm256_unpacklo_epi16(bg1, ra1);
        let rgba3_ = _mm256_unpackhi_epi16(bg1, ra1);

        let rgba0 = _mm256_permute2x128_si256::<32>(rgba0_, rgba1_);
        let rgba2 = _mm256_permute2x128_si256::<49>(rgba0_, rgba1_);
        let rgba1 = _mm256_permute2x128_si256::<32>(rgba2_, rgba3_);
        let rgba3 = _mm256_permute2x128_si256::<49>(rgba2_, rgba3_);
        (rgba0, rgba1, rgba2, rgba3)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx2_pack_u16(s_1: __m256i, s_2: __m256i) -> __m256i {
    unsafe {
        let packed = _mm256_packus_epi16(s_1, s_2);
        const MASK: i32 = shuffle(3, 1, 2, 0);
        _mm256_permute4x64_epi64::<MASK>(packed)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx2_pack_u32(s_1: __m256i, s_2: __m256i) -> __m256i {
    unsafe {
        let packed = _mm256_packus_epi32(s_1, s_2);
        const MASK: i32 = shuffle(3, 1, 2, 0);
        _mm256_permute4x64_epi64::<MASK>(packed)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn avx_combine_ps(lo: __m128, hi: __m128) -> __m256 {
    unsafe { _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo), hi) }
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn avx_combine_epi(lo: __m128i, hi: __m128i) -> __m256i {
    unsafe {
        _mm256_castps_si256(_mm256_insertf128_ps::<1>(
            _mm256_castps128_ps256(_mm_castsi128_ps(lo)),
            _mm_castsi128_ps(hi),
        ))
    }
}

#[inline]
/// Arithmetic shift for i64, shifting with sign bits
pub(crate) unsafe fn _mm256_srai_epi64x<const IMM8: i32>(a: __m256i) -> __m256i {
    unsafe {
        let m = _mm256_set1_epi64x(1 << (64 - 1));
        let x = _mm256_srli_epi64::<IMM8>(a);
        _mm256_sub_epi64(_mm256_xor_si256(x, m), m)
    }
}

#[inline]
/// Pack 64bytes integers into 32 bytes using truncation
pub(crate) unsafe fn _mm256_packts_epi64(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        const SHUFFLE_1: i32 = shuffle(2, 0, 2, 0);
        let combined =
            _mm256_shuffle_ps::<SHUFFLE_1>(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b));
        const SHUFFLE_2: i32 = shuffle(3, 1, 2, 0);
        let ordered = _mm256_permute4x64_pd::<SHUFFLE_2>(_mm256_castps_pd(combined));
        _mm256_castpd_si256(ordered)
    }
}

#[inline]
#[allow(dead_code)]
/// Pack 64bytes integers into 32 bytes
pub(crate) unsafe fn _mm256_cvtepi64_epi32x(v: __m256i) -> __m128i {
    unsafe {
        let vf = _mm256_castsi256_ps(v);
        let hi = _mm256_extractf128_ps::<1>(vf);
        let lo = _mm256_castps256_ps128(vf);
        const FLAGS: i32 = shuffle(2, 0, 2, 0);
        let packed = _mm_shuffle_ps::<FLAGS>(lo, hi);
        _mm_castps_si128(packed)
    }
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn _mm256_dot16_avx_epi32<const HAS_DOT: bool>(
    a: __m256i,
    b: __m256i,
    c: __m256i,
) -> __m256i {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        {
            if HAS_DOT {
                _mm256_dpwssd_avx_epi32(a, b, c)
            } else {
                _mm256_add_epi32(a, _mm256_madd_epi16(b, c))
            }
        }
        #[cfg(not(feature = "nightly_avx512"))]
        {
            _mm256_add_epi32(a, _mm256_madd_epi16(b, c))
        }
    }
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn _mm_dot16_avx_epi32<const HAS_DOT: bool>(
    a: __m128i,
    b: __m128i,
    c: __m128i,
) -> __m128i {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        {
            if HAS_DOT {
                _mm_dpwssd_avx_epi32(a, b, c)
            } else {
                _mm_add_epi32(a, _mm_madd_epi16(b, c))
            }
        }
        #[cfg(not(feature = "nightly_avx512"))]
        {
            _mm_add_epi32(a, _mm_madd_epi16(b, c))
        }
    }
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn _mm_udot8_epi16<const DOT: bool>(
    a: __m128i,
    b: __m128i,
    c: __m128i,
) -> __m128i {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        if DOT {
            _mm_dpbusd_avx_epi32(a, b, c)
        } else {
            _mm_adds_epi16(a, _mm_maddubs_epi16(b, c))
        }
        #[cfg(not(feature = "nightly_avx512"))]
        {
            _mm_adds_epi16(a, _mm_maddubs_epi16(b, c))
        }
    }
}

#[inline]
pub(crate) unsafe fn _mm256_prefer_fma_ps<const FMA: bool>(
    a: __m256,
    b: __m256,
    c: __m256,
) -> __m256 {
    unsafe {
        if FMA {
            _mm256_fmadd_ps(b, c, a)
        } else {
            _mm256_add_ps(_mm256_mul_ps(b, c), a)
        }
    }
}

#[inline]
pub(crate) unsafe fn _mm_prefer_fma_ps<const FMA: bool>(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe {
        if FMA {
            _mm_fmadd_ps(b, c, a)
        } else {
            _mm_add_ps(_mm_mul_ps(b, c), a)
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_reduce_r_epi32<const PRECISION: i32>(x: __m128i) -> __m128i {
    unsafe {
        const FIRST_MASK: i32 = shuffle(1, 0, 3, 2);
        let hi64 = _mm_shuffle_epi32::<FIRST_MASK>(x);
        let sum64 = _mm_add_epi32(hi64, x);
        const SM: i32 = shuffle(1, 0, 3, 2);
        let hi32 = _mm_shufflelo_epi16::<SM>(sum64);
        _mm_srai_epi32::<PRECISION>(_mm_add_epi32(sum64, hi32))
    }
}

/// Sums all lanes in float32
#[inline(always)]
pub(crate) unsafe fn _mm_hsum_ps(v: __m128) -> __m128 {
    unsafe {
        let mut shuf = _mm_movehdup_ps(v);
        let sums = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        _mm_add_ss(sums, shuf)
    }
}
