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

#[inline]
#[cfg(target_feature = "avx2")]
pub(crate) unsafe fn _mm_srlv_epi32x(c: __m128i, n: __m128i) -> __m128i {
    _mm_srlv_epi32(c, n)
}

#[inline]
#[cfg(not(target_feature = "avx2"))]
pub(crate) unsafe fn _mm_srlv_epi32x(c: __m128i, n: __m128i) -> __m128i {
    _mm_setr_epi32(
        _mm_extract_epi32::<0>(c).wrapping_shr(_mm_extract_epi32::<0>(n) as u32),
        _mm_extract_epi32::<1>(c).wrapping_shr(_mm_extract_epi32::<1>(n) as u32),
        _mm_extract_epi32::<2>(c).wrapping_shr(_mm_extract_epi32::<2>(n) as u32),
        _mm_extract_epi32::<3>(c).wrapping_shr(_mm_extract_epi32::<3>(n) as u32),
    )
}

#[inline]
#[cfg(target_feature = "avx2")]
pub(crate) unsafe fn _mm_sllv_epi32x(c: __m128i, n: __m128i) -> __m128i {
    _mm_sllv_epi32(c, n)
}

#[inline]
#[cfg(not(target_feature = "avx2"))]
pub(crate) unsafe fn _mm_sllv_epi32x(c: __m128i, n: __m128i) -> __m128i {
    _mm_setr_epi32(
        _mm_extract_epi32::<0>(c).wrapping_shl(_mm_extract_epi32::<0>(n) as u32),
        _mm_extract_epi32::<1>(c).wrapping_shl(_mm_extract_epi32::<1>(n) as u32),
        _mm_extract_epi32::<2>(c).wrapping_shl(_mm_extract_epi32::<2>(n) as u32),
        _mm_extract_epi32::<3>(c).wrapping_shl(_mm_extract_epi32::<3>(n) as u32),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_blendv_epi32(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castps_si128(_mm_blendv_ps(
        _mm_castsi128_ps(xmm0),
        _mm_castsi128_ps(xmm1),
        _mm_castsi128_ps(mask),
    ))
}

#[inline(always)]
/// If mask then `true_vals` otherwise `false_val`
pub(crate) unsafe fn _mm_select_epi32(
    mask: __m128i,
    true_vals: __m128i,
    false_vals: __m128i,
) -> __m128i {
    _mm_blendv_epi32(false_vals, true_vals, mask)
}

#[inline]
unsafe fn _mm_cmpneq_epi32(a: __m128i, b: __m128i) -> __m128i {
    // Compare for equality
    let eq_mask = _mm_cmpeq_epi32(a, b);
    _mm_xor_si128(eq_mask, _mm_set1_epi32(-1)) // XOR with all 1s (0xFFFFFFFF) to invert
}

/**
    This is not fully IEEE complaint conversion, only more straight for fallback
**/
#[inline]
unsafe fn _mm_cvtph_ps_fallback(k: __m128i) -> __m128 {
    let h = _mm_unpacklo_epi16(k, _mm_setzero_si128());
    // Constants
    let exp_mask = _mm_set1_epi32(0x7C00);
    let mantissa_mask = _mm_set1_epi32(0x03FF);

    // Extract the exponent and mantissa
    let exp = _mm_srli_epi32::<10>(_mm_and_si128(h, exp_mask));
    let mantissa = _mm_slli_epi32::<13>(_mm_and_si128(h, mantissa_mask));
    let v = _mm_srli_epi32::<23>(_mm_castps_si128(_mm_cvtepi32_ps(mantissa)));
    let j1 = _mm_slli_epi32::<16>(_mm_and_si128(h, _mm_set1_epi32(0x8000)));
    let is_exp_zero = _mm_cmpeq_epi32(exp, _mm_setzero_si128());
    let j2 = _mm_select_epi32(
        is_exp_zero,
        _mm_setzero_si128(),
        _mm_or_si128(
            _mm_slli_epi32::<23>(_mm_add_epi32(exp, _mm_set1_epi32(112))),
            mantissa,
        ),
    );

    let pvm = _mm_slli_epi32::<23>(_mm_sub_epi32(v, _mm_set1_epi32(37)));
    let vgm = _mm_and_si128(
        _mm_sllv_epi32x(mantissa, _mm_sub_epi32(_mm_set1_epi32(150), v)),
        _mm_set1_epi32(0x007FE000),
    );

    let j3 = _mm_select_epi32(
        _mm_and_si128(is_exp_zero, _mm_cmpneq_epi32(mantissa, _mm_setzero_si128())),
        _mm_or_si128(pvm, vgm),
        _mm_setzero_si128(),
    );
    _mm_castsi128_ps(_mm_or_si128(_mm_or_si128(j1, j2), j3))
}

/**
   This is not fully IEEE complaint conversion, only more straight for fallback
**/
#[inline]
unsafe fn _mm_cvtps_ph_fallback(x: __m128) -> __m128i {
    let b = _mm_add_epi32(_mm_castps_si128(x), _mm_set1_epi32(0x00001000));
    let e = _mm_srli_epi32::<23>(_mm_and_si128(b, _mm_set1_epi32(0x7F800000)));
    let m = _mm_and_si128(b, _mm_set1_epi32(0x007FFFFF));

    let v_112 = _mm_set1_epi32(112);
    let j1 = _mm_select_epi32(
        _mm_cmpgt_epi32(e, v_112),
        _mm_or_si128(
            _mm_and_si128(
                _mm_slli_epi32::<10>(_mm_sub_epi32(e, v_112)),
                _mm_set1_epi32(0x7C00),
            ),
            _mm_srli_epi32::<13>(m),
        ),
        _mm_setzero_si128(),
    );

    let v2_count = _mm_sub_epi32(_mm_set1_epi32(125), e);
    let v2 = _mm_srli_epi32::<1>(_mm_add_epi32(
        _mm_srlv_epi32x(_mm_add_epi32(_mm_set1_epi32(0x007FF000), m), v2_count),
        _mm_set1_epi32(1),
    ));
    let j2 = _mm_select_epi32(
        _mm_and_si128(
            _mm_cmplt_epi32(e, _mm_set1_epi32(113)),
            _mm_cmpgt_epi32(e, _mm_set1_epi32(101)),
        ),
        v2,
        _mm_setzero_si128(),
    );
    let sat = _mm_mullo_epi32(
        _mm_select_epi32(
            _mm_cmpgt_epi32(e, _mm_set1_epi32(143)),
            _mm_set1_epi32(1),
            _mm_setzero_si128(),
        ),
        _mm_set1_epi32(0x7FFF),
    );
    let packed_32 = _mm_or_si128(_mm_or_si128(j1, j2), sat);
    _mm_packus_epi32(packed_32, _mm_setzero_si128())
}

#[inline]
#[target_feature(enable = "f16c")]
unsafe fn _mm_cvtps_phdx(x: __m128) -> __m128i {
    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x)
}

#[inline]
pub(crate) unsafe fn _mm_cvtps_phx<const F16C: bool>(x: __m128) -> __m128i {
    if F16C {
        _mm_cvtps_phdx(x)
    } else {
        _mm_cvtps_ph_fallback(x)
    }
}

#[inline]
#[target_feature(enable = "f16c")]
unsafe fn _mm_cvtph_psdx(x: __m128i) -> __m128 {
    _mm_cvtph_ps(x)
}

#[inline]
pub(crate) unsafe fn _mm_cvtph_psx<const F16C: bool>(x: __m128i) -> __m128 {
    if F16C {
        _mm_cvtph_ps(x)
    } else {
        _mm_cvtph_ps_fallback(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use f16;

    #[test]
    fn test_conversion_into_f16() {
        unsafe {
            // Test regular
            let value = _mm_set1_ps(24.);
            let converted = _mm_cvtps_phx(value);
            let flag = _mm_extract_epi16::<0>(converted) as u16;
            let bits = f16::from_f32(24.);
            assert_eq!(flag, bits.to_bits());
        }
    }

    #[test]
    fn test_srlv_sse() {
        unsafe {
            // Test regular
            let count = _mm_setr_epi32(4, 3, 2, 1);
            let n = _mm_setr_epi32(100, 75, 50, 25);
            let shifted = _mm_srlv_epi32x(n, count);
            let fist = _mm_extract_epi32::<0>(shifted) as u32;
            let sec = _mm_extract_epi32::<1>(shifted) as u32;
            let thi = _mm_extract_epi32::<2>(shifted) as u32;
            let fth = _mm_extract_epi32::<3>(shifted) as u32;
            assert_eq!(fist, 100u32.wrapping_shr(4));
            assert_eq!(sec, 75u32.wrapping_shr(3));
            assert_eq!(thi, 50u32.wrapping_shr(2));
            assert_eq!(fth, 25u32.wrapping_shr(1));
        }
    }
}
