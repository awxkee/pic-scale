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
        let weight = _mm_loadu_ps($src_ptr);
        const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
        let weight0 = _mm_shuffle_ps::<SHUFFLE_0>(weight, weight);
        const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
        let weight1 = _mm_shuffle_ps::<SHUFFLE_1>(weight, weight);
        const SHUFFLE_2: i32 = shuffle(2, 2, 2, 2);
        let weight2 = _mm_shuffle_ps::<SHUFFLE_2>(weight, weight);
        const SHUFFLE_3: i32 = shuffle(3, 3, 3, 3);
        let weight3 = _mm_shuffle_ps::<SHUFFLE_3>(weight, weight);
        (
            avx_combine_ps(weight0, weight1),
            avx_combine_ps(weight2, weight3),
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
        let weight_row_0 = _mm_loadu_ps($src_ptr);
        const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
        let weight0 = _mm_shuffle_ps::<SHUFFLE_0>(weight_row_0, weight_row_0);
        const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
        let weight1 = _mm_shuffle_ps::<SHUFFLE_1>(weight_row_0, weight_row_0);
        const SHUFFLE_2: i32 = shuffle(2, 2, 2, 2);
        let weight2 = _mm_shuffle_ps::<SHUFFLE_2>(weight_row_0, weight_row_0);
        const SHUFFLE_3: i32 = shuffle(3, 3, 3, 3);
        let weight3 = _mm_shuffle_ps::<SHUFFLE_3>(weight_row_0, weight_row_0);

        let weight_row_1 = _mm_loadu_ps($src_ptr.add(4));
        let weight4 = _mm_shuffle_ps::<SHUFFLE_0>(weight_row_1, weight_row_1);
        let weight5 = _mm_shuffle_ps::<SHUFFLE_1>(weight_row_1, weight_row_1);
        let weight6 = _mm_shuffle_ps::<SHUFFLE_2>(weight_row_1, weight_row_1);
        let weight7 = _mm_shuffle_ps::<SHUFFLE_3>(weight_row_1, weight_row_1);
        (
            avx_combine_ps(weight0, weight1),
            avx_combine_ps(weight2, weight3),
            avx_combine_ps(weight4, weight5),
            avx_combine_ps(weight6, weight7),
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
