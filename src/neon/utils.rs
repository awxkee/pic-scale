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
#[cfg(feature = "rdm")]
pub(crate) unsafe fn expand8_to_14(row: uint8x8_t) -> int16x8_t {
    let row = vcombine_u8(row, row);
    vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_u8(vzip1q_u8(row, row))))
}

#[inline(always)]
#[cfg(feature = "rdm")]
pub(crate) unsafe fn expand8_high_to_14(row: uint8x16_t) -> int16x8_t {
    vreinterpretq_s16_u16(vshrq_n_u16::<2>(vreinterpretq_u16_u8(vzip2q_u8(row, row))))
}

#[inline(always)]
pub(crate) unsafe fn xvld1q_u8_x2(ptr: *const u8) -> uint8x16x2_t {
    uint8x16x2_t(vld1q_u8(ptr), vld1q_u8(ptr.add(16)))
}

#[inline(always)]
pub(crate) unsafe fn xvld1q_u8_x4(ptr: *const u8) -> uint8x16x4_t {
    uint8x16x4_t(
        vld1q_u8(ptr),
        vld1q_u8(ptr.add(16)),
        vld1q_u8(ptr.add(32)),
        vld1q_u8(ptr.add(48)),
    )
}

#[inline(always)]
pub(crate) unsafe fn xvld1q_u16_x4(a: *const u16) -> uint16x8x4_t {
    uint16x8x4_t(
        vld1q_u16(a),
        vld1q_u16(a.add(8)),
        vld1q_u16(a.add(16)),
        vld1q_u16(a.add(24)),
    )
}

#[inline(always)]
pub(crate) unsafe fn xvld1q_u16_x2(a: *const u16) -> uint16x8x2_t {
    uint16x8x2_t(vld1q_u16(a), vld1q_u16(a.add(8)))
}

#[inline(always)]
pub(crate) unsafe fn xvld1q_f32_x4(a: *const f32) -> float32x4x4_t {
    float32x4x4_t(
        vld1q_f32(a),
        vld1q_f32(a.add(4)),
        vld1q_f32(a.add(8)),
        vld1q_f32(a.add(12)),
    )
}

#[inline(always)]
pub(crate) unsafe fn xvld1q_f32_x2(a: *const f32) -> float32x4x2_t {
    float32x4x2_t(vld1q_f32(a), vld1q_f32(a.add(4)))
}

#[inline(always)]
pub(crate) unsafe fn xvst1q_u8_x2(ptr: *mut u8, b: uint8x16x2_t) {
    vst1q_u8(ptr, b.0);
    vst1q_u8(ptr.add(16), b.1);
}

#[inline(always)]
pub(crate) unsafe fn xvst1q_u8_x4(ptr: *mut u8, b: uint8x16x4_t) {
    vst1q_u8(ptr, b.0);
    vst1q_u8(ptr.add(16), b.1);
    vst1q_u8(ptr.add(32), b.2);
    vst1q_u8(ptr.add(48), b.3);
}

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
pub unsafe fn xvst1q_f32_x4(a: *mut f32, b: float32x4x4_t) {
    vst1q_f32(a, b.0);
    vst1q_f32(a.add(4), b.1);
    vst1q_f32(a.add(8), b.2);
    vst1q_f32(a.add(12), b.3);
}

#[inline(always)]
pub unsafe fn xvst1q_f32_x2(a: *mut f32, b: float32x4x2_t) {
    vst1q_f32(a, b.0);
    vst1q_f32(a.add(4), b.1);
}

#[inline(always)]
pub(crate) unsafe fn prefer_vfmaq_laneq_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    vfmaq_laneq_f32::<LANE>(a, b, c)
}

#[inline(always)]
pub(crate) unsafe fn prefer_vfmaq_lane_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x2_t,
) -> float32x4_t {
    vfmaq_lane_f32::<LANE>(a, b, c)
}

#[inline(always)]
pub(crate) unsafe fn load_3b_as_u16x4(src_ptr: *const u8) -> uint16x4_t {
    let mut v = vreinterpret_u8_u16(vld1_lane_u16::<0>(src_ptr as *const u16, vdup_n_u16(0)));
    v = vld1_lane_u8::<2>(src_ptr.add(2), v);
    vget_low_u16(vmovl_u8(v))
}

#[inline(always)]
#[cfg(feature = "rdm")]
pub(crate) unsafe fn load_3b_as_u8x16(src_ptr: *const u8) -> uint8x16_t {
    let v = vreinterpretq_u8_u16(vld1q_lane_u16::<0>(src_ptr as *const u16, vdupq_n_u16(0)));
    vld1q_lane_u8::<2>(src_ptr.add(2), v)
}

#[inline(always)]
pub(crate) unsafe fn load_4b_as_u16x4(src_ptr: *const u8) -> uint16x4_t {
    let j = vreinterpret_u8_u32(vld1_lane_u32::<0>(src_ptr as *const u32, vdup_n_u32(0)));
    vget_low_u16(vmovl_u8(j))
}

#[inline(always)]
#[cfg(feature = "rdm")]
pub(crate) unsafe fn load_4b_as_u8x8(src_ptr: *const u8) -> uint8x8_t {
    vreinterpret_u8_u32(vld1_lane_u32::<0>(src_ptr as *const u32, vdup_n_u32(0)))
}

#[inline(always)]
pub(crate) unsafe fn xvld1q_s16_x2(a: *const i16) -> int16x8x2_t {
    let v0 = vld1q_s16(a);
    let v1 = vld1q_s16(a.add(8));
    int16x8x2_t(v0, v1)
}

#[inline(always)]
#[cfg(feature = "rdm")]
pub(crate) unsafe fn xvld1q_s16_x4(a: *const i16) -> int16x8x4_t {
    let v0 = vld1q_s16(a);
    let v1 = vld1q_s16(a.add(8));
    let v2 = vld1q_s16(a.add(16));
    let v3 = vld1q_s16(a.add(24));
    int16x8x4_t(v0, v1, v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn vxmlal_high_lane_s16<const D: bool, const LANE: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x4_t,
) -> int32x4_t {
    if D {
        vqdmlal_high_lane_s16::<LANE>(a, b, c)
    } else {
        vmlal_high_lane_s16::<LANE>(a, b, c)
    }
}

#[inline(always)]
pub(crate) unsafe fn vxmlal_lane_s16<const D: bool, const LANE: i32>(
    a: int32x4_t,
    b: int16x4_t,
    c: int16x4_t,
) -> int32x4_t {
    if D {
        vqdmlal_lane_s16::<LANE>(a, b, c)
    } else {
        vmlal_lane_s16::<LANE>(a, b, c)
    }
}

#[inline(always)]
pub(crate) unsafe fn vxmlal_s16<const D: bool>(
    a: int32x4_t,
    b: int16x4_t,
    c: int16x4_t,
) -> int32x4_t {
    if D {
        vqdmlal_s16(a, b, c)
    } else {
        vmlal_s16(a, b, c)
    }
}

#[inline(always)]
pub(crate) unsafe fn vxmlal_high_s16<const D: bool>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x8_t,
) -> int32x4_t {
    if D {
        vqdmlal_high_s16(a, b, c)
    } else {
        vmlal_high_s16(a, b, c)
    }
}

#[inline(always)]
pub(crate) unsafe fn vxmlal_high_laneq_s16<const D: bool, const LANE: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x8_t,
) -> int32x4_t {
    if D {
        vqdmlal_high_laneq_s16::<LANE>(a, b, c)
    } else {
        vmlal_high_laneq_s16::<LANE>(a, b, c)
    }
}

#[inline(always)]
pub(crate) unsafe fn vxmlal_laneq_s16<const D: bool, const LANE: i32>(
    a: int32x4_t,
    b: int16x4_t,
    c: int16x8_t,
) -> int32x4_t {
    if D {
        vqdmlal_laneq_s16::<LANE>(a, b, c)
    } else {
        vmlal_laneq_s16::<LANE>(a, b, c)
    }
}

#[inline(always)]
pub(crate) unsafe fn xvld1q_u32_x2(a: *const u32) -> uint32x4x2_t {
    let v0 = vld1q_u32(a);
    let v1 = vld1q_u32(a.add(4));
    uint32x4x2_t(v0, v1)
}
