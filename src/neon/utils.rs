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

#[inline(always)]
pub(crate) unsafe fn prefer_vfmaq_f32(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    #[cfg(target_arch = "aarch64")]
    {
        vfmaq_f32(a, b, c)
    }
    #[cfg(target_arch = "arm")]
    {
        vmlaq_f32(a, b, c)
    }
}

#[inline(always)]
pub(crate) unsafe fn load_3b_as_u16x4(src_ptr: *const u8) -> uint16x4_t {
    let v_new_value1 = u16::from_le_bytes([src_ptr.read_unaligned(), 0]);
    let v_new_value2 = u16::from_le_bytes([src_ptr.add(1).read_unaligned(), 0]);
    let v_new_value3 = u16::from_le_bytes([src_ptr.add(2).read_unaligned(), 0]);
    let arr = [v_new_value1, v_new_value2, v_new_value3, 0];
    vld1_u16(arr.as_ptr())
}

#[inline(always)]
pub(crate) unsafe fn load_4b_as_u16x4(src_ptr: *const u8) -> uint16x4_t {
    let v_new_value1 = u16::from_le_bytes([src_ptr.read_unaligned(), 0]);
    let v_new_value2 = u16::from_le_bytes([src_ptr.add(1).read_unaligned(), 0]);
    let v_new_value3 = u16::from_le_bytes([src_ptr.add(2).read_unaligned(), 0]);
    let v_new_value4 = u16::from_le_bytes([src_ptr.add(3).read_unaligned(), 0]);
    let arr = [v_new_value1, v_new_value2, v_new_value3, v_new_value4];
    vld1_u16(arr.as_ptr())
}

#[inline(always)]
pub(crate) unsafe fn vsplit_rgb_5(px: float32x4x4_t) -> Float32x5T {
    let first_pixel = px.0;
    let second_pixel = vextq_f32::<3>(px.0, px.1);
    let third_pixel = vextq_f32::<2>(px.1, px.2);
    let four_pixel = vextq_f32::<1>(px.2, px.3);
    Float32x5T(first_pixel, second_pixel, third_pixel, four_pixel, px.3)
}

pub(crate) struct Float32x5T(
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
);
