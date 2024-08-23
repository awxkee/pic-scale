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
use crate::wasm32::utils::{wasm_unpackhi_i8x16, wasm_unpacklo_i8x16};
use std::arch::wasm32::*;

pub unsafe fn wasm_load_deinterleave_u8x4(ptr: *const u8) -> (v128, v128, v128, v128) {
    let u0 = v128_load(ptr as *const v128); // a0 b0 c0 d0 a1 b1 c1 d1 ...
    let u1 = v128_load(ptr.add(16) as *const v128); // a4 b4 c4 d4 ...
    let u2 = v128_load(ptr.add(32) as *const v128); // a8 b8 c8 d8 ...
    let u3 = v128_load(ptr.add(48) as *const v128); // a12 b12 c12 d12 ...

    let v0 = i8x16_shuffle::<0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29>(u0, u1);
    let v1 = i8x16_shuffle::<0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29>(u2, u3);
    let v2 = i8x16_shuffle::<2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31>(u0, u1);
    let v3 = i8x16_shuffle::<2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31>(u2, u3);

    let a = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(v0, v1);
    let b = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(v0, v1);
    let c = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(v2, v3);
    let d = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(v2, v3);
    (a, b, c, d)
}

pub unsafe fn wasm_store_interleave_u8x4(ptr: *mut u8, packed: (v128, v128, v128, v128)) {
    let a = packed.0;
    let b = packed.1;
    let c = packed.2;
    let d = packed.3;
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    let u0 = wasm_unpacklo_i8x16(a, c); // a0 c0 a1 c1 ...
    let u1 = wasm_unpackhi_i8x16(a, c); // a8 c8 a9 c9 ...
    let u2 = wasm_unpacklo_i8x16(b, d); // b0 d0 b1 d1 ...
    let u3 = wasm_unpackhi_i8x16(b, d); // b8 d8 b9 d9 ...

    let v0 = wasm_unpacklo_i8x16(u0, u2); // a0 b0 c0 d0 ...
    let v1 = wasm_unpackhi_i8x16(u0, u2); // a4 b4 c4 d4 ...
    let v2 = wasm_unpacklo_i8x16(u1, u3); // a8 b8 c8 d8 ...
    let v3 = wasm_unpackhi_i8x16(u1, u3); // a12 b12 c12 d12 ...

    v128_store(ptr as *mut v128, v0);
    v128_store(ptr.add(16) as *mut v128, v1);
    v128_store(ptr.add(32) as *mut v128, v2);
    v128_store(ptr.add(48) as *mut v128, v3);
}
