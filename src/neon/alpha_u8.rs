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

use crate::WorkloadStrategy;
use std::arch::aarch64::*;

#[inline]
pub fn neon_div_by_255_n(v: uint16x8_t) -> uint8x8_t {
    unsafe { vqrshrn_n_u16::<8>(vrsraq_n_u16::<8>(v, v)) }
}

macro_rules! premultiply_vec {
    ($v: expr, $a_values: expr) => {{
        let acc_hi = vmull_high_u8($v, $a_values);
        let acc_lo = vmull_u8(vget_low_u8($v), vget_low_u8($a_values));
        let hi = neon_div_by_255_n(acc_hi);
        let lo = neon_div_by_255_n(acc_lo);
        vcombine_u8(lo, hi)
    }};
}

#[target_feature(enable = "neon")]
fn neon_premultiply_alpha_rgba_impl_row(dst: &mut [u8], src: &[u8]) {
    let mut rem = dst;
    let mut src_rem = src;

    static TBL: [u8; 16] = [3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15];
    let shuf_tbl = unsafe { vld1q_u8(TBL.as_ptr()) };
    let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(u32::from_ne_bytes([0, 0, 0, 255])));

    for (dst, src) in rem
        .as_chunks_mut::<64>()
        .0
        .iter_mut()
        .zip(src_rem.as_chunks::<64>().0.iter())
    {
        let pixel0 = unsafe { vld1q_u8(src.as_ptr()) };
        let pixel1 = unsafe { vld1q_u8(src[16..].as_ptr()) };
        let pixel2 = unsafe { vld1q_u8(src[32..].as_ptr()) };
        let pixel3 = unsafe { vld1q_u8(src[48..].as_ptr()) };

        let alpha0 = vqtbl1q_u8(pixel0, shuf_tbl);
        let alpha1 = vqtbl1q_u8(pixel1, shuf_tbl);
        let alpha2 = vqtbl1q_u8(pixel2, shuf_tbl);
        let alpha3 = vqtbl1q_u8(pixel3, shuf_tbl);

        let mut new_px0 = premultiply_vec!(pixel0, alpha0);
        let mut new_px1 = premultiply_vec!(pixel1, alpha1);
        let mut new_px2 = premultiply_vec!(pixel2, alpha2);
        let mut new_px3 = premultiply_vec!(pixel3, alpha3);

        new_px0 = vbslq_u8(alpha_mask, pixel0, new_px0);
        new_px1 = vbslq_u8(alpha_mask, pixel1, new_px1);
        new_px2 = vbslq_u8(alpha_mask, pixel2, new_px2);
        new_px3 = vbslq_u8(alpha_mask, pixel3, new_px3);

        unsafe {
            vst1q_u8(dst.as_mut_ptr(), new_px0);
            vst1q_u8(dst[16..].as_mut_ptr(), new_px1);
            vst1q_u8(dst[32..].as_mut_ptr(), new_px2);
            vst1q_u8(dst[48..].as_mut_ptr(), new_px3);
        }
    }

    rem = rem.as_chunks_mut::<64>().1;
    src_rem = src_rem.as_chunks::<64>().1;

    for (dst, src) in rem
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(src_rem.as_chunks::<16>().0.iter())
    {
        let pixel = unsafe { vld1q_u8(src.as_ptr()) };
        let alpha = vqtbl1q_u8(pixel, shuf_tbl);
        let mut new_px = premultiply_vec!(pixel, alpha);
        new_px = vbslq_u8(alpha_mask, pixel, new_px);
        unsafe {
            vst1q_u8(dst.as_mut_ptr(), new_px);
        }
    }

    rem = rem.as_chunks_mut::<16>().1;
    src_rem = src_rem.as_chunks::<16>().1;

    if !rem.is_empty() && !src_rem.is_empty() {
        assert!(rem.len() < 16);
        let mut buffer: [u8; 16] = [0u8; 16];
        buffer[..src_rem.len()].copy_from_slice(src_rem);
        let pixel = unsafe { vld1q_u8(buffer.as_ptr()) };
        let alpha = vqtbl1q_u8(pixel, shuf_tbl);
        let mut new_px = premultiply_vec!(pixel, alpha);
        new_px = vbslq_u8(alpha_mask, pixel, new_px);
        unsafe {
            vst1q_u8(buffer.as_mut_ptr(), new_px);
        }
        rem.copy_from_slice(&buffer[..rem.len()]);
    }
}

pub(crate) fn neon_premultiply_alpha_rgba(dst: &mut [u8], src: &[u8]) {
    unsafe {
        neon_premultiply_alpha_rgba_impl_row(dst, src);
    }
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u8]);
}

#[derive(Default)]
struct NeonDisassociateAlpha {}

#[cfg(feature = "rdm")]
#[derive(Default)]
struct NeonDisassociateAlphaFast {}

impl DisassociateAlpha for NeonDisassociateAlpha {
    #[target_feature(enable = "neon")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        let mut rem = in_place;

        static ALPHA_MASK: [u8; 16] = [
            3, 255, 255, 255, 7, 255, 255, 255, 11, 255, 255, 255, 15, 255, 255, 255,
        ];
        let alpha_mask = unsafe { vld1q_u8(ALPHA_MASK.as_ptr()) };

        let recip_divider = vdupq_n_f32(1.);
        let scale = vdupq_n_u8(255);
        let copy_alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(u32::from_ne_bytes([0, 0, 0, 255])));

        for dst in rem.as_chunks_mut::<16>().0.iter_mut() {
            let src_ptr = dst.as_ptr();
            let pixel = unsafe { vld1q_u8(src_ptr) };
            let alpha_u32 = vreinterpretq_u32_u8(vqtbl1q_u8(pixel, alpha_mask));
            let alpha_f32 = vdivq_f32(recip_divider, vcvtq_f32_u32(alpha_u32));
            let lo16 = vmull_u8(vget_low_u8(pixel), vget_low_u8(scale));
            let hi16 = vmull_high_u8(pixel, scale);

            let is_alpha_zero = vreinterpretq_u8_u32(vceqzq_u32(alpha_u32));

            let mut p0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
            let mut p1 = vcvtq_f32_u32(vmovl_high_u16(lo16));
            let mut p2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
            let mut p3 = vcvtq_f32_u32(vmovl_high_u16(hi16));

            p0 = vmulq_laneq_f32::<0>(p0, alpha_f32);
            p1 = vmulq_laneq_f32::<1>(p1, alpha_f32);
            p2 = vmulq_laneq_f32::<2>(p2, alpha_f32);
            p3 = vmulq_laneq_f32::<3>(p3, alpha_f32);

            let packed0 = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(p0)),
                vqmovn_u32(vcvtaq_u32_f32(p1)),
            );
            let packed1 = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(p2)),
                vqmovn_u32(vcvtaq_u32_f32(p3)),
            );
            let mut packed = vcombine_u8(vqmovn_u16(packed0), vqmovn_u16(packed1));

            packed = vbslq_u8(is_alpha_zero, vdupq_n_u8(0), packed);
            packed = vbslq_u8(copy_alpha_mask, pixel, packed);

            unsafe {
                vst1q_u8(dst.as_mut_ptr(), packed);
            }
        }

        rem = rem.as_chunks_mut::<16>().1;

        if !rem.is_empty() {
            assert!(rem.len() < 16);
            let mut buffer: [u8; 16] = [0u8; 16];
            buffer[..rem.len()].copy_from_slice(rem);

            let pixel = unsafe { vld1q_u8(buffer.as_ptr()) };

            let alpha_u32 = vreinterpretq_u32_u8(vqtbl1q_u8(pixel, alpha_mask));
            let alpha_f32 = vdivq_f32(recip_divider, vcvtq_f32_u32(alpha_u32));
            let lo16 = vmull_u8(vget_low_u8(pixel), vget_low_u8(scale));
            let hi16 = vmull_high_u8(pixel, scale);

            let is_alpha_zero = vreinterpretq_u8_u32(vceqzq_u32(alpha_u32));

            let mut p0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
            let mut p1 = vcvtq_f32_u32(vmovl_high_u16(lo16));
            let mut p2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
            let mut p3 = vcvtq_f32_u32(vmovl_high_u16(hi16));

            p0 = vmulq_laneq_f32::<0>(p0, alpha_f32);
            p1 = vmulq_laneq_f32::<1>(p1, alpha_f32);
            p2 = vmulq_laneq_f32::<2>(p2, alpha_f32);
            p3 = vmulq_laneq_f32::<3>(p3, alpha_f32);

            let packed0 = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(p0)),
                vqmovn_u32(vcvtaq_u32_f32(p1)),
            );
            let packed1 = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(p2)),
                vqmovn_u32(vcvtaq_u32_f32(p3)),
            );
            let mut packed = vcombine_u8(vqmovn_u16(packed0), vqmovn_u16(packed1));

            packed = vbslq_u8(is_alpha_zero, vdupq_n_u8(0), packed);
            packed = vbslq_u8(copy_alpha_mask, pixel, packed);

            unsafe {
                vst1q_u8(buffer.as_mut_ptr(), packed);
            }

            rem.copy_from_slice(&buffer[..rem.len()]);
        }
    }
}

#[cfg(feature = "rdm")]
impl DisassociateAlpha for NeonDisassociateAlphaFast {
    #[target_feature(enable = "rdm")]
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        let mut rem = in_place;

        static ALPHA_MASK: [u8; 16] = [
            3, 255, 255, 255, 7, 255, 255, 255, 11, 255, 255, 255, 15, 255, 255, 255,
        ];
        let alpha_mask = unsafe { vld1q_u8(ALPHA_MASK.as_ptr()) };

        const Q: f32 = ((1i64 << 31) - 1) as f32;

        let q0_31_divider = vdupq_n_f32(Q);
        let scale = vdupq_n_u8(255);
        let copy_alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(u32::from_ne_bytes([0, 0, 0, 255])));

        for dst in rem.as_chunks_mut::<16>().0.iter_mut() {
            let src_ptr = dst.as_ptr();
            let pixel = unsafe { vld1q_u8(src_ptr) };
            let alpha_u32 = vreinterpretq_u32_u8(vqtbl1q_u8(pixel, alpha_mask));
            let alpha_f32 = vdivq_f32(q0_31_divider, vcvtq_f32_u32(alpha_u32));
            let reciprocal = vcvtq_s32_f32(alpha_f32);
            let lo16 = vmull_u8(vget_low_u8(pixel), vget_low_u8(scale));
            let hi16 = vmull_high_u8(pixel, scale);

            let is_alpha_zero = vreinterpretq_u8_u32(vceqzq_u32(alpha_u32));

            let mut p0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo16)));
            let mut p1 = vreinterpretq_s32_u32(vmovl_high_u16(lo16));
            let mut p2 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi16)));
            let mut p3 = vreinterpretq_s32_u32(vmovl_high_u16(hi16));

            p0 = vqrdmulhq_laneq_s32::<0>(p0, reciprocal);
            p1 = vqrdmulhq_laneq_s32::<1>(p1, reciprocal);
            p2 = vqrdmulhq_laneq_s32::<2>(p2, reciprocal);
            p3 = vqrdmulhq_laneq_s32::<3>(p3, reciprocal);

            let packed0 = vcombine_u16(vqmovun_s32(p0), vqmovun_s32(p1));
            let packed1 = vcombine_u16(vqmovun_s32(p2), vqmovun_s32(p3));
            let mut packed = vcombine_u8(vqmovn_u16(packed0), vqmovn_u16(packed1));

            packed = vbslq_u8(is_alpha_zero, vdupq_n_u8(0), packed);
            packed = vbslq_u8(copy_alpha_mask, pixel, packed);

            unsafe {
                vst1q_u8(dst.as_mut_ptr(), packed);
            }
        }

        rem = rem.as_chunks_mut::<16>().1;

        if !rem.is_empty() {
            assert!(rem.len() < 16);
            let mut buffer: [u8; 16] = [0u8; 16];
            buffer[..rem.len()].copy_from_slice(rem);

            let pixel = unsafe { vld1q_u8(buffer.as_ptr()) };
            let alpha_u32 = vreinterpretq_u32_u8(vqtbl1q_u8(pixel, alpha_mask));
            let alpha_f32 = vdivq_f32(q0_31_divider, vcvtq_f32_u32(alpha_u32));
            let reciprocal = vcvtq_s32_f32(alpha_f32);
            let lo16 = vmull_u8(vget_low_u8(pixel), vget_low_u8(scale));
            let hi16 = vmull_high_u8(pixel, scale);

            let is_alpha_zero = vreinterpretq_u8_u32(vceqzq_u32(alpha_u32));

            let mut p0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo16)));
            let mut p1 = vreinterpretq_s32_u32(vmovl_high_u16(lo16));
            let mut p2 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi16)));
            let mut p3 = vreinterpretq_s32_u32(vmovl_high_u16(hi16));

            p0 = vqrdmulhq_laneq_s32::<0>(p0, reciprocal);
            p1 = vqrdmulhq_laneq_s32::<1>(p1, reciprocal);
            p2 = vqrdmulhq_laneq_s32::<2>(p2, reciprocal);
            p3 = vqrdmulhq_laneq_s32::<3>(p3, reciprocal);

            let packed0 = vcombine_u16(vqmovun_s32(p0), vqmovun_s32(p1));
            let packed1 = vcombine_u16(vqmovun_s32(p2), vqmovun_s32(p3));
            let mut packed = vcombine_u8(vqmovn_u16(packed0), vqmovn_u16(packed1));

            packed = vbslq_u8(is_alpha_zero, vdupq_n_u8(0), packed);
            packed = vbslq_u8(copy_alpha_mask, pixel, packed);

            unsafe {
                vst1q_u8(buffer.as_mut_ptr(), packed);
            }

            rem.copy_from_slice(&buffer[..rem.len()]);
        }
    }
}

#[inline]
fn neon_dis_dispatch(in_place: &mut [u8], handler: impl DisassociateAlpha) {
    unsafe {
        handler.disassociate(in_place);
    }
}

fn neon_unpremultiply_alpha_rgba_impl_row(in_place: &mut [u8]) {
    neon_dis_dispatch(in_place, NeonDisassociateAlpha::default());
}

#[cfg(feature = "rdm")]
#[target_feature(enable = "rdm")]
fn neon_unpremultiply_alpha_rgba_impl_row_rdm(in_place: &mut [u8]) {
    neon_dis_dispatch(in_place, NeonDisassociateAlphaFast::default());
}

pub(crate) fn neon_unpremultiply_alpha_rgba(in_place: &mut [u8], _strategy: WorkloadStrategy) {
    let mut _executor: unsafe fn(&mut [u8]) = neon_unpremultiply_alpha_rgba_impl_row;
    #[cfg(feature = "rdm")]
    {
        if _strategy == WorkloadStrategy::PreferSpeed
            && std::arch::is_aarch64_feature_detected!("rdm")
        {
            _executor = neon_unpremultiply_alpha_rgba_impl_row_rdm;
        }
    }
    unsafe {
        _executor(in_place);
    }
}
