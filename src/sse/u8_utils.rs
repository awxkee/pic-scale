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

use crate::support::PRECISION;

#[inline(always)]
pub(crate) fn compress_i32(x: __m128i) -> __m128i {
    unsafe {
        let store_32 = _mm_srai_epi32::<PRECISION>(x);
        let store_16 = _mm_packus_epi32(store_32, store_32);
        _mm_packus_epi16(store_16, store_16)
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
    let vl = i32::from_le_bytes([
        src_ptr.read_unaligned(),
        src_ptr.add(1).read_unaligned(),
        src_ptr.add(2).read_unaligned(),
        0,
    ]);
    let m_vl = _mm_cvtsi32_si128(vl);
    let lo = _mm_unpacklo_epi8(m_vl, _mm_setzero_si128());
    _mm_add_epi32(
        store_0,
        _mm_madd_epi16(_mm_unpacklo_epi16(lo, _mm_setzero_si128()), weight0),
    )
}
