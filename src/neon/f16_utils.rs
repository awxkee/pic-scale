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
use std::arch::aarch64::*;
use std::arch::asm;

/// Provides basic support for f16

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub struct x_float16x4_t(pub(crate) uint16x4_t);

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub struct x_float16x8_t(pub(crate) uint16x8_t);

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct x_float16x8x2_t(pub(crate) x_float16x8_t, pub(crate) x_float16x8_t);

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct x_float16x8x4_t(
    pub(crate) x_float16x8_t,
    pub(crate) x_float16x8_t,
    pub(crate) x_float16x8_t,
    pub(crate) x_float16x8_t,
);

#[inline]
pub unsafe fn xvld_f16(ptr: *const half::f16) -> x_float16x4_t {
    let store: uint16x4_t = vld1_u16(std::mem::transmute(ptr));
    std::mem::transmute(store)
}

#[inline]
pub unsafe fn xvldq_f16(ptr: *const half::f16) -> x_float16x8_t {
    let store: uint16x8_t = vld1q_u16(std::mem::transmute(ptr));
    std::mem::transmute(store)
}

#[inline]
pub unsafe fn xvldq_f16_x2(ptr: *const half::f16) -> x_float16x8x2_t {
    let store = vld1q_u16_x2(std::mem::transmute(ptr));
    std::mem::transmute(store)
}

#[inline]
pub unsafe fn xvldq_f16_x4(ptr: *const half::f16) -> x_float16x8x4_t {
    let store = vld1q_u16_x4(std::mem::transmute(ptr));
    std::mem::transmute(store)
}

#[inline]
pub unsafe fn xvget_low_f16(x: x_float16x8_t) -> x_float16x4_t {
    std::mem::transmute(vget_low_u16(std::mem::transmute(x)))
}

#[inline]
pub unsafe fn xvget_high_f16(x: x_float16x8_t) -> x_float16x4_t {
    std::mem::transmute(vget_high_u16(std::mem::transmute(x)))
}

#[inline]
pub unsafe fn xcombine_f16(low: x_float16x4_t, high: x_float16x4_t) -> x_float16x8_t {
    std::mem::transmute(vcombine_u16(
        std::mem::transmute(low),
        std::mem::transmute(high),
    ))
}

#[inline]
pub unsafe fn xreinterpret_u16_f16(x: x_float16x4_t) -> uint16x4_t {
    std::mem::transmute(x)
}

#[inline]
pub unsafe fn xreinterpretq_u16_f16(x: x_float16x8_t) -> uint16x8_t {
    std::mem::transmute(x)
}

#[inline]
pub unsafe fn xreinterpret_f16_u16(x: uint16x4_t) -> x_float16x4_t {
    std::mem::transmute(x)
}

#[inline]
pub unsafe fn xreinterpretq_f16_u16(x: uint16x8_t) -> x_float16x8_t {
    std::mem::transmute(x)
}

#[inline]
pub(super) unsafe fn xvzerosq_f16() -> x_float16x8_t {
    xreinterpretq_f16_u16(vdupq_n_u16(0))
}

#[inline]
pub(super) unsafe fn xvzeros_f16() -> x_float16x4_t {
    xreinterpret_f16_u16(vdup_n_u16(0))
}

#[inline]
pub unsafe fn xvcvt_f32_f16(x: x_float16x4_t) -> float32x4_t {
    let src: uint16x4_t = xreinterpret_u16_f16(x);
    let dst: float32x4_t;
    asm!(
    "fcvtl {0:v}.4s, {1:v}.4h",
    out(vreg) dst,
    in(vreg) src,
    options(pure, nomem, nostack));
    dst
}

#[inline]
pub(super) unsafe fn xvcvt_f16_f32(v: float32x4_t) -> x_float16x4_t {
    let result: uint16x4_t;
    asm!(
    "fcvtn {0:v}.4h, {1:v}.4s",
    out(vreg) result,
    in(vreg) v,
    options(pure, nomem, nostack));
    xreinterpret_f16_u16(result)
}

// #[inline]
// pub(super) unsafe fn xvadd_f16(v1: x_float16x4_t, v2: x_float16x4_t) -> x_float16x4_t {
//     let result: uint16x4_t;
//     asm!(
//     "fadd {0:v}.4h, {1:v}.4h, {2:v}.4h",
//     out(vreg) result,
//     in(vreg) xreinterpret_u16_f16(v1),
//     in(vreg) xreinterpret_u16_f16(v2),
//     options(pure, nomem, nostack)
//     );
//     xreinterpret_f16_u16(result)
// }

// #[inline]
// pub(super) unsafe fn xvaddq_f16(v1: x_float16x8_t, v2: x_float16x8_t) -> x_float16x8_t {
//     let result: uint16x8_t;
//     asm!(
//     "fadd {0:v}.8h, {1:v}.8h, {2:v}.8h",
//     out(vreg) result,
//     in(vreg) xreinterpretq_u16_f16(v1),
//     in(vreg) xreinterpretq_u16_f16(v2),
//     options(pure, nomem, nostack)
//     );
//     xreinterpretq_f16_u16(result)
// }

#[inline]
pub(super) unsafe fn xvcombine_f16(v1: x_float16x4_t, v2: x_float16x4_t) -> x_float16x8_t {
    xreinterpretq_f16_u16(vcombine_u16(
        xreinterpret_u16_f16(v1),
        xreinterpret_u16_f16(v2),
    ))
}

// #[inline]
// pub(super) unsafe fn xvmul_f16(v1: x_float16x4_t, v2: x_float16x4_t) -> x_float16x4_t {
//     let result: uint16x4_t;
//     asm!(
//     "fmul {0:v}.4h, {1:v}.4h, {2:v}.4h",
//     out(vreg) result,
//     in(vreg) xreinterpret_u16_f16(v1),
//     in(vreg) xreinterpret_u16_f16(v2),
//     options(pure, nomem, nostack)
//     );
//     xreinterpret_f16_u16(result)
// }

#[inline]
pub(super) unsafe fn xvfmla_f16(
    a: x_float16x4_t,
    b: x_float16x4_t,
    c: x_float16x4_t,
) -> x_float16x4_t {
    let mut result: uint16x4_t = xreinterpret_u16_f16(a);
    asm!(
    "fmla {0:v}.4h, {1:v}.4h, {2:v}.4h",
    inout(vreg) result,
    in(vreg) xreinterpret_u16_f16(b),
    in(vreg) xreinterpret_u16_f16(c),
    options(pure, nomem, nostack)
    );
    xreinterpret_f16_u16(result)
}

#[inline]
pub(super) unsafe fn xvfmlaq_f16(
    a: x_float16x8_t,
    b: x_float16x8_t,
    c: x_float16x8_t,
) -> x_float16x8_t {
    let mut result: uint16x8_t = xreinterpretq_u16_f16(a);
    asm!(
    "fmla {0:v}.8h, {1:v}.8h, {2:v}.8h",
    inout(vreg) result,
    in(vreg) xreinterpretq_u16_f16(b),
    in(vreg) xreinterpretq_u16_f16(c),
    options(pure, nomem, nostack)
    );
    xreinterpretq_f16_u16(result)
}

// #[cfg(all(target_arch = "aarch64", target_feature = "fhm"))]
// #[inline]
// pub(super) unsafe fn p_xvmlaq_f16(
//     a: x_float16x8_t,
//     b: x_float16x8_t,
//     c: x_float16x8_t,
// ) -> x_float16x8_t {
//     xvfmlaq_f16(a, b, c)
// }

// #[inline]
// pub(super) unsafe fn xvmlaq_f16(
//     a: x_float16x8_t,
//     b: x_float16x8_t,
//     c: x_float16x8_t,
// ) -> x_float16x8_t {
//     xvaddq_f16(a, xvmulq_f16(b, c))
// }

// #[inline]
// pub(super) unsafe fn xvmla_f16(
//     a: x_float16x4_t,
//     b: x_float16x4_t,
//     c: x_float16x4_t,
// ) -> x_float16x4_t {
//     xvadd_f16(a, xvmul_f16(b, c))
// }

#[inline]
pub(super) unsafe fn xvmulq_f16(v1: x_float16x8_t, v2: x_float16x8_t) -> x_float16x8_t {
    let result: uint16x8_t;
    asm!(
    "fmul {0:v}.8h, {1:v}.8h, {2:v}.8h",
    out(vreg) result,
    in(vreg) xreinterpretq_u16_f16(v1),
    in(vreg) xreinterpretq_u16_f16(v2),
    options(pure, nomem, nostack)
    );
    xreinterpretq_f16_u16(result)
}

#[inline]
pub(super) unsafe fn xvdivq_f16(v1: x_float16x8_t, v2: x_float16x8_t) -> x_float16x8_t {
    let result: uint16x8_t;
    asm!(
    "fdiv {0:v}.8h, {1:v}.8h, {2:v}.8h",
    out(vreg) result,
    in(vreg) xreinterpretq_u16_f16(v1),
    in(vreg) xreinterpretq_u16_f16(v2),
    options(pure, nomem, nostack)
    );
    xreinterpretq_f16_u16(result)
}

#[inline]
pub(super) unsafe fn xvbslq_f16(
    a: uint16x8_t,
    b: x_float16x8_t,
    c: x_float16x8_t,
) -> x_float16x8_t {
    let mut result: uint16x8_t = a;
    asm!(
    "bsl {0:v}.16b, {1:v}.16b, {2:v}.16b",
    inout(vreg) result,
    in(vreg) xreinterpretq_u16_f16(b),
    in(vreg) xreinterpretq_u16_f16(c),
    options(pure, nomem, nostack)
    );
    xreinterpretq_f16_u16(result)
}

#[inline]
pub unsafe fn xvst_f16(ptr: *const half::f16, x: x_float16x4_t) {
    vst1_u16(std::mem::transmute(ptr), xreinterpret_u16_f16(x))
}

#[inline]
pub unsafe fn xvstq_f16(ptr: *const half::f16, x: x_float16x8_t) {
    vst1q_u16(std::mem::transmute(ptr), xreinterpretq_u16_f16(x))
}

#[inline]
pub unsafe fn xvstq_f16_x2(ptr: *const half::f16, x: x_float16x8x2_t) {
    vst1q_u16_x2(std::mem::transmute(ptr), std::mem::transmute(x))
}

#[inline]
pub unsafe fn xvstq_f16_x4(ptr: *const half::f16, x: x_float16x8x4_t) {
    vst1q_u16_x4(std::mem::transmute(ptr), std::mem::transmute(x))
}

#[inline]
pub unsafe fn xvdup_lane_f16<const N: i32>(a: x_float16x4_t) -> x_float16x4_t {
    xreinterpret_f16_u16(vdup_lane_u16::<N>(xreinterpret_u16_f16(a)))
}

#[inline]
pub unsafe fn xvdup_laneq_f16<const N: i32>(a: x_float16x8_t) -> x_float16x4_t {
    xreinterpret_f16_u16(vdup_laneq_u16::<N>(xreinterpretq_u16_f16(a)))
}

#[inline]
pub unsafe fn vceqzq_f16(a: x_float16x8_t) -> uint16x8_t {
    let mut result: uint16x8_t;
    asm!(
    "fcmeq {0:v}.8h, {1:v}.8h, #0",
    out(vreg) result,
    in(vreg) xreinterpretq_u16_f16(a),
    options(pure, nomem, nostack)
    );
    result
}
