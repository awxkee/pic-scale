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
use crate::ar30::Rgb30;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn vrev128_u32(v: uint32x4_t) -> uint32x4_t {
    vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(v)))
}

#[inline(always)]
pub(crate) unsafe fn vunzips_4_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: uint32x4_t,
) -> int16x4x4_t {
    let mask = vdupq_n_u32(0x3ff);
    let ar_type: Rgb30 = AR30_TYPE.into();

    let v = if AR30_ORDER == 0 { v } else { vrev128_u32(v) };

    match ar_type {
        Rgb30::Ar30 => {
            let r = vmovn_u32(vandq_u32(v, mask));
            let g = vmovn_u32(vandq_u32(vshrq_n_u32::<10>(v), mask));
            let b = vmovn_u32(vandq_u32(vshrq_n_u32::<20>(v), mask));
            let va = vmovn_u32(vshrq_n_u32::<30>(v));
            let a = vorr_u16(
                vorr_u16(
                    vorr_u16(
                        vorr_u16(vshl_n_u16::<8>(va), vshl_n_u16::<6>(va)),
                        vshl_n_u16::<4>(va),
                    ),
                    vshl_n_u16::<2>(va),
                ),
                va,
            );
            int16x4x4_t(
                vreinterpret_s16_u16(r),
                vreinterpret_s16_u16(g),
                vreinterpret_s16_u16(b),
                vreinterpret_s16_u16(a),
            )
        }
        Rgb30::Ra30 => {
            let a_mask = vdupq_n_u32(0x3);
            let va = vmovn_u32(vandq_u32(v, a_mask));

            let a = vorr_u16(
                vorr_u16(
                    vorr_u16(
                        vorr_u16(vshl_n_u16::<8>(va), vshl_n_u16::<6>(va)),
                        vshl_n_u16::<4>(va),
                    ),
                    vshl_n_u16::<2>(va),
                ),
                va,
            );

            let r = vmovn_u32(vandq_u32(vshrq_n_u32::<22>(v), mask));
            let g = vmovn_u32(vandq_u32(vshrq_n_u32::<12>(v), mask));
            let b = vmovn_u32(vandq_u32(vshrq_n_u32::<2>(v), mask));
            int16x4x4_t(
                vreinterpret_s16_u16(r),
                vreinterpret_s16_u16(g),
                vreinterpret_s16_u16(b),
                vreinterpret_s16_u16(a),
            )
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn vunzip_4_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: uint32x4x2_t,
) -> int16x8x4_t {
    let mask = vdupq_n_u32(0x3ff);
    let ar_type: Rgb30 = AR30_TYPE.into();

    let v = if AR30_ORDER == 0 {
        v
    } else {
        uint32x4x2_t(vrev128_u32(v.0), vrev128_u32(v.1))
    };

    match ar_type {
        Rgb30::Ar30 => {
            let r = vcombine_u16(
                vmovn_u32(vandq_u32(v.0, mask)),
                vmovn_u32(vandq_u32(v.1, mask)),
            );
            let g = vcombine_u16(
                vmovn_u32(vandq_u32(vshrq_n_u32::<10>(v.0), mask)),
                vmovn_u32(vandq_u32(vshrq_n_u32::<10>(v.1), mask)),
            );
            let b = vcombine_u16(
                vmovn_u32(vandq_u32(vshrq_n_u32::<20>(v.0), mask)),
                vmovn_u32(vandq_u32(vshrq_n_u32::<20>(v.1), mask)),
            );
            let va = vcombine_u16(
                vmovn_u32(vshrq_n_u32::<30>(v.0)),
                vmovn_u32(vshrq_n_u32::<30>(v.1)),
            );
            let a = vorrq_u16(
                vorrq_u16(
                    vorrq_u16(
                        vorrq_u16(vshlq_n_u16::<8>(va), vshlq_n_u16::<6>(va)),
                        vshlq_n_u16::<4>(va),
                    ),
                    vshlq_n_u16::<2>(va),
                ),
                va,
            );
            int16x8x4_t(
                vreinterpretq_s16_u16(r),
                vreinterpretq_s16_u16(g),
                vreinterpretq_s16_u16(b),
                vreinterpretq_s16_u16(a),
            )
        }
        Rgb30::Ra30 => {
            let a_mask = vdupq_n_u32(0x3);
            let va = vcombine_u16(
                vmovn_u32(vandq_u32(v.0, a_mask)),
                vmovn_u32(vandq_u32(v.1, a_mask)),
            );

            let a = vorrq_u16(
                vorrq_u16(
                    vorrq_u16(
                        vorrq_u16(vshlq_n_u16::<8>(va), vshlq_n_u16::<6>(va)),
                        vshlq_n_u16::<4>(va),
                    ),
                    vshlq_n_u16::<2>(va),
                ),
                va,
            );

            let r = vcombine_u16(
                vmovn_u32(vandq_u32(vshrq_n_u32::<22>(v.0), mask)),
                vmovn_u32(vandq_u32(vshrq_n_u32::<22>(v.1), mask)),
            );
            let g = vcombine_u16(
                vmovn_u32(vandq_u32(vshrq_n_u32::<12>(v.0), mask)),
                vmovn_u32(vandq_u32(vshrq_n_u32::<12>(v.1), mask)),
            );
            let b = vcombine_u16(
                vmovn_u32(vandq_u32(vshrq_n_u32::<2>(v.0), mask)),
                vmovn_u32(vandq_u32(vshrq_n_u32::<2>(v.1), mask)),
            );
            int16x8x4_t(
                vreinterpretq_s16_u16(r),
                vreinterpretq_s16_u16(g),
                vreinterpretq_s16_u16(b),
                vreinterpretq_s16_u16(a),
            )
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn vunzip_4_ar30_separate<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: uint32x4x2_t,
) -> int16x8x4_t {
    let values = vunzip_4_ar30::<AR30_TYPE, AR30_ORDER>(v);
    let a0 = vtrnq_s16(values.0, values.1);
    let a1 = vtrnq_s16(values.2, values.3);
    let v1 = vtrnq_s32(vreinterpretq_s32_s16(a0.0), vreinterpretq_s32_s16(a1.0));
    let v2 = vtrnq_s32(vreinterpretq_s32_s16(a0.1), vreinterpretq_s32_s16(a1.1));
    let k0 = vreinterpretq_s16_s32(v1.0);
    let k1 = vreinterpretq_s16_s32(v2.0);
    let k2 = vreinterpretq_s16_s32(v1.1);
    let k3 = vreinterpretq_s16_s32(v2.1);
    int16x8x4_t(k0, k1, k2, k3)
}

#[inline(always)]
pub(crate) unsafe fn vunzips_4_ar30_separate<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: uint32x4_t,
) -> int16x8x2_t {
    let values = vunzips_4_ar30::<AR30_TYPE, AR30_ORDER>(v);
    let a0 = vtrn_s16(values.0, values.1);
    let a1 = vtrn_s16(values.2, values.3);
    let v1 = vtrn_s32(vreinterpret_s32_s16(a0.0), vreinterpret_s32_s16(a1.0));
    let v2 = vtrn_s32(vreinterpret_s32_s16(a0.1), vreinterpret_s32_s16(a1.1));
    let k0 = vreinterpret_s16_s32(v1.0);
    let k1 = vreinterpret_s16_s32(v2.0);
    let k2 = vreinterpret_s16_s32(v1.1);
    let k3 = vreinterpret_s16_s32(v2.1);
    int16x8x2_t(vcombine_s16(k0, k1), vcombine_s16(k2, k3))
}

#[inline(always)]
pub(crate) unsafe fn vzip_4_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: int16x8x4_t,
) -> uint32x4x2_t {
    let ar_type: Rgb30 = AR30_TYPE.into();
    let a_max = vdupq_n_s16(3);
    match ar_type {
        Rgb30::Ar30 => {
            let v3 = vminq_s16(vrshrq_n_s16::<8>(v.3), a_max);
            let mut a0 = vshlq_n_u32::<30>(vmovl_u16(vreinterpret_u16_s16(vget_low_s16(v3))));
            let mut a1 = vshlq_n_u32::<30>(vmovl_u16(vreinterpret_u16_s16(vget_high_s16(v3))));

            let r0 = vshlq_n_u32::<20>(vmovl_u16(vreinterpret_u16_s16(vget_low_s16(v.2))));
            let r1 = vshlq_n_u32::<20>(vmovl_u16(vreinterpret_u16_s16(vget_high_s16(v.2))));

            a0 = vorrq_u32(a0, r0);
            a1 = vorrq_u32(a1, r1);

            let g0 = vshlq_n_u32::<10>(vmovl_u16(vreinterpret_u16_s16(vget_low_s16(v.1))));
            let g1 = vshlq_n_u32::<10>(vmovl_u16(vreinterpret_u16_s16(vget_high_s16(v.1))));

            a0 = vorrq_u32(a0, g0);
            a1 = vorrq_u32(a1, g1);

            a0 = vorrq_u32(a0, vmovl_u16(vreinterpret_u16_s16(vget_low_s16(v.0))));
            a1 = vorrq_u32(a1, vmovl_u16(vreinterpret_u16_s16(vget_high_s16(v.0))));

            if AR30_ORDER == 0 {
                uint32x4x2_t(a0, a1)
            } else {
                uint32x4x2_t(vrev128_u32(a0), vrev128_u32(a1))
            }
        }
        Rgb30::Ra30 => {
            let v3 = vminq_s16(vrshrq_n_s16::<8>(v.3), a_max);
            let mut a0 = vmovl_u16(vreinterpret_u16_s16(vget_low_s16(v3)));
            let mut a1 = vmovl_u16(vreinterpret_u16_s16(vget_high_s16(v3)));

            let r0 = vshlq_n_u32::<22>(vmovl_u16(vreinterpret_u16_s16(vget_low_s16(v.0))));
            let r1 = vshlq_n_u32::<22>(vmovl_u16(vreinterpret_u16_s16(vget_high_s16(v.0))));

            a0 = vorrq_u32(a0, r0);
            a1 = vorrq_u32(a1, r1);

            let g0 = vshlq_n_u32::<12>(vmovl_u16(vreinterpret_u16_s16(vget_low_s16(v.1))));
            let g1 = vshlq_n_u32::<12>(vmovl_u16(vreinterpret_u16_s16(vget_high_s16(v.1))));

            a0 = vorrq_u32(a0, g0);
            a1 = vorrq_u32(a1, g1);

            a0 = vorrq_u32(
                a0,
                vshlq_n_u32::<2>(vmovl_u16(vreinterpret_u16_s16(vget_low_s16(v.2)))),
            );
            a1 = vorrq_u32(
                a1,
                vshlq_n_u32::<2>(vmovl_u16(vreinterpret_u16_s16(vget_high_s16(v.2)))),
            );

            if AR30_ORDER == 0 {
                uint32x4x2_t(a0, a1)
            } else {
                uint32x4x2_t(vrev128_u32(a0), vrev128_u32(a1))
            }
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn vld1_ar30_s16<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    arr: &[u8],
) -> int16x4_t {
    let item = u32::from_ne_bytes([
        *arr.get_unchecked(0),
        *arr.get_unchecked(1),
        *arr.get_unchecked(2),
        *arr.get_unchecked(3),
    ]);
    let ar_type: Rgb30 = AR30_TYPE.into();
    let vl = ar_type.unpack::<AR30_ORDER>(item);
    let a_rep = (vl.3 as i16) << 8;
    let temp = [vl.0 as i16, vl.1 as i16, vl.2 as i16, a_rep];
    vld1_s16(temp.as_ptr())
}

#[inline(always)]
pub(crate) unsafe fn vextract_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    v: uint16x4_t,
) -> u32 {
    let v0 = vreinterpret_u64_u16(v);
    let a_mask = vdup_n_u64(0x3);
    let v_mask = vdup_n_u64(0x3ff);
    let mut a = vand_u64(vshr_n_u64::<48>(v0), a_mask);
    let r = vand_u64(v0, v_mask);
    let g = vand_u64(vshr_n_u64::<16>(v0), v_mask);
    let b = vand_u64(vshr_n_u64::<32>(v0), v_mask);

    let ar_type: Rgb30 = AR30_TYPE.into();

    match ar_type {
        Rgb30::Ar30 => {
            a = vshl_n_u64::<30>(a);
            a = vorr_u64(a, vshl_n_u64::<20>(b));
            a = vorr_u64(a, vshl_n_u64::<10>(g));
            a = vorr_u64(a, r);
        }
        Rgb30::Ra30 => {
            a = vorr_u64(a, vshl_n_u64::<2>(b));
            a = vorr_u64(a, vshl_n_u64::<12>(g));
            a = vorr_u64(a, vshl_n_u64::<22>(r));
        }
    }

    if AR30_ORDER == 1 {
        a = vreinterpret_u64_u8(vrev32_u8(vreinterpret_u8_u64(a)));
    }
    let pairs = vreinterpret_u32_u64(a);
    vget_lane_u32::<0>(pairs)
}
