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
use std::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn w_zeros() -> v128 {
    i32x4_splat(0)
}

/// Packs two u32x4 into one u16x8 using truncation
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn u32x4_pack_trunc_u16x8(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(a, b)
}

/// Packs two u16x8 into one u8x16 using truncation
#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn u16x8_pack_trunc_u8x16(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(a, b)
}

/// Packs two u16x8 into one u8x16 using unsigned saturation
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn u16x8_pack_sat_u8x16(a: v128, b: v128) -> v128 {
    let maxval = u16x8_splat(255);
    let a1 = v128_bitselect(maxval, a, u16x8_gt(a, maxval));
    let b1 = v128_bitselect(maxval, b, u16x8_gt(b, maxval));
    u8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(a1, b1)
}

#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_unpacklo_i8x16(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_unpackhi_i8x16(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b)
}
