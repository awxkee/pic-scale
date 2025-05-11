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

#[cfg(feature = "nightly_f16")]
macro_rules! load_4_weights_group_2_avx {
    ($src_ptr: expr) => {{
        (
            avx_combine_ps(
                _mm_broadcast_ss($src_ptr.get_unchecked(0)),
                _mm_broadcast_ss($src_ptr.get_unchecked(1)),
            ),
            avx_combine_ps(
                _mm_broadcast_ss($src_ptr.get_unchecked(2)),
                _mm_broadcast_ss($src_ptr.get_unchecked(3)),
            ),
        )
    }};
}

#[cfg(feature = "nightly_f16")]
pub(crate) use load_4_weights_group_2_avx;
#[cfg(feature = "nightly_f16")]
use std::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_shufflelo_epi16,
};

#[cfg(feature = "nightly_f16")]
macro_rules! load_8_weights_group_4_avx {
    ($src_ptr: expr) => {{
        (
            avx_combine_ps(
                _mm_broadcast_ss($src_ptr.get_unchecked(0)),
                _mm_broadcast_ss($src_ptr.get_unchecked(1)),
            ),
            avx_combine_ps(
                _mm_broadcast_ss($src_ptr.get_unchecked(2)),
                _mm_broadcast_ss($src_ptr.get_unchecked(3)),
            ),
            avx_combine_ps(
                _mm_broadcast_ss($src_ptr.get_unchecked(4)),
                _mm_broadcast_ss($src_ptr.get_unchecked(5)),
            ),
            avx_combine_ps(
                _mm_broadcast_ss($src_ptr.get_unchecked(6)),
                _mm_broadcast_ss($src_ptr.get_unchecked(7)),
            ),
        )
    }};
}

use crate::support::PRECISION;
#[cfg(feature = "nightly_f16")]
pub(crate) use load_8_weights_group_4_avx;

pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub(crate) unsafe fn _mm_hsum_epi32(x: __m128i) -> i32 {
    unsafe {
        const FIRST_MASK: i32 = shuffle(1, 0, 3, 2);
        let hi64 = _mm_shuffle_epi32::<FIRST_MASK>(x);
        let sum64 = _mm_add_epi32(hi64, x);
        const SM: i32 = shuffle(1, 0, 3, 2);
        let hi32 = _mm_shufflelo_epi16::<SM>(sum64);
        let sum32 = _mm_add_epi32(sum64, hi32);
        _mm_cvtsi128_si32(sum32)
    }
}

#[inline(always)]
pub(crate) fn compress_i32(x: __m128i) -> __m128i {
    unsafe {
        let store_32 = _mm_srai_epi32::<PRECISION>(x);
        _mm_packus_epi32(store_32, store_32)
    }
}
