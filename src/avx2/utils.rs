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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(not(target_feature = "fma"))]
#[inline]
pub unsafe fn _mm256_fma_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    return _mm256_add_ps(_mm256_mul_ps(b, c), a);
}

#[cfg(target_feature = "fma")]
#[inline]
pub unsafe fn _mm256_fma_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    return _mm256_fmadd_ps(b, c, a);
}

#[inline(always)]
pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub unsafe fn _mm256_select_si256(
    mask: __m256i,
    true_vals: __m256i,
    false_vals: __m256i,
) -> __m256i {
    _mm256_or_si256(
        _mm256_and_si256(mask, true_vals),
        _mm256_andnot_si256(mask, false_vals),
    )
}

#[inline(always)]
pub unsafe fn _mm256_selecti_ps(mask: __m256i, true_vals: __m256, false_vals: __m256) -> __m256 {
    _mm256_blendv_ps(false_vals, true_vals, _mm256_castsi256_ps(mask))
}

#[inline(always)]
pub unsafe fn avx2_div_by255(v: __m256i) -> __m256i {
    let addition = _mm256_set1_epi16(127);
    _mm256_srli_epi16::<8>(_mm256_add_epi16(
        _mm256_add_epi16(v, addition),
        _mm256_srli_epi16::<8>(v),
    ))
}

#[inline(always)]
pub unsafe fn avx2_deinterleave_rgba(
    rgba0: __m256i,
    rgba1: __m256i,
    rgba2: __m256i,
    rgba3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
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

#[inline(always)]
pub unsafe fn avx_deinterleave_rgba_epi32(
    p0: __m256i,
    p1: __m256i,
    p2: __m256i,
    p3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
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

#[inline(always)]
pub unsafe fn avx_interleave_rgba_epi32(
    p0: __m256i,
    p1: __m256i,
    p2: __m256i,
    p3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
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

#[inline(always)]
pub unsafe fn avx_interleave_rgba_epi16(
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
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

#[inline(always)]
pub unsafe fn avx_deinterleave_rgba_epi16(
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let sh = _mm256_setr_epi8(
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12,
        13, 6, 7, 14, 15,
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

#[inline(always)]
pub unsafe fn avx_deinterleave_rgba_ps(
    p0: __m256,
    p1: __m256,
    p2: __m256,
    p3: __m256,
) -> (__m256, __m256, __m256, __m256) {
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

#[inline(always)]
pub unsafe fn avx_interleave_rgba_ps(
    p0: __m256,
    p1: __m256,
    p2: __m256,
    p3: __m256,
) -> (__m256, __m256, __m256, __m256) {
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

#[inline(always)]
pub unsafe fn avx2_interleave_rgba(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    a: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
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

#[inline(always)]
pub unsafe fn avx2_pack_u16(s_1: __m256i, s_2: __m256i) -> __m256i {
    let packed = _mm256_packus_epi16(s_1, s_2);
    const MASK: i32 = shuffle(3, 1, 2, 0);
    return _mm256_permute4x64_epi64::<MASK>(packed);
}

#[inline(always)]
pub unsafe fn avx2_pack_s32(s_1: __m256i, s_2: __m256i) -> __m256i {
    let packed = _mm256_packs_epi32(s_1, s_2);
    const MASK: i32 = shuffle(3, 1, 2, 0);
    return _mm256_permute4x64_epi64::<MASK>(packed);
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx_combine_ps(lo: __m128, hi: __m128) -> __m256 {
    _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo), hi)
}

#[inline(always)]
pub unsafe fn avx_combine_epi(lo: __m128i, hi: __m128i) -> __m256i {
    _mm256_castps_si256(_mm256_insertf128_ps::<1>(
        _mm256_castps128_ps256(_mm_castsi128_ps(lo)),
        _mm_castsi128_ps(hi),
    ))
}
