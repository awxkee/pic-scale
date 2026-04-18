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

impl NeonDisassociateAlpha {
    #[inline(always)]
    fn get_reciprocal(a_values: uint8x16_t) -> float32x4x4_t {
        unsafe {
            let a_hi = vmovl_high_u8(a_values);
            let a_lo = vmovl_u8(vget_low_u8(a_values));
            let a_lo_lo = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_lo))));
            let a_lo_hi = vrecpeq_f32(vcvtq_f32_u32(vmovl_high_u16(a_lo)));
            let a_hi_lo = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_hi))));
            let a_hi_ho = vrecpeq_f32(vcvtq_f32_u32(vmovl_high_u16(a_hi)));
            float32x4x4_t(a_lo_lo, a_lo_hi, a_hi_lo, a_hi_ho)
        }
    }

    #[inline(always)]
    fn get_reciprocalh(a_values: uint8x8_t) -> float32x4x2_t {
        unsafe {
            let a_lo = vmovl_u8(a_values);
            let a_lo_lo = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_lo))));
            let a_lo_hi = vrecpeq_f32(vcvtq_f32_u32(vmovl_high_u16(a_lo)));
            float32x4x2_t(a_lo_lo, a_lo_hi)
        }
    }

    #[inline(always)]
    fn unpremultiply_vec(v: uint8x16_t, mask: uint8x16_t, recip: float32x4x4_t) -> uint8x16_t {
        unsafe {
            let scale = vdupq_n_u8(255);
            let hi = vmull_high_u8(v, scale);
            let lo = vmull_u8(vget_low_u8(v), vget_low_u8(scale));
            let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
            let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo));
            let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
            let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi));

            let lo_lo = vcvtaq_u32_f32(vmulq_f32(lo_lo, recip.0));
            let lo_hi = vcvtaq_u32_f32(vmulq_f32(lo_hi, recip.1));
            let hi_lo = vcvtaq_u32_f32(vmulq_f32(hi_lo, recip.2));
            let hi_hi = vcvtaq_u32_f32(vmulq_f32(hi_hi, recip.3));
            let lo = vcombine_u16(vmovn_u32(lo_lo), vmovn_u32(lo_hi));
            let hi = vcombine_u16(vmovn_u32(hi_lo), vmovn_u32(hi_hi));
            vbslq_u8(
                mask,
                vdupq_n_u8(0),
                vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi)),
            )
        }
    }

    #[inline(always)]
    fn unpremultiply_vech(v: uint8x8_t, mask: uint8x8_t, reciprocal: float32x4x2_t) -> uint8x8_t {
        unsafe {
            let scale = vdupq_n_u8(255);
            let lo = vmull_u8(v, vget_low_u8(scale));
            let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
            let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo));

            let lo_lo = vcvtaq_u32_f32(vmulq_f32(lo_lo, reciprocal.0));
            let lo_hi = vcvtaq_u32_f32(vmulq_f32(lo_hi, reciprocal.1));
            let lo = vcombine_u16(vmovn_u32(lo_lo), vmovn_u32(lo_hi));
            vbsl_u8(mask, vdup_n_u8(0), vqmovn_u16(lo))
        }
    }
}

macro_rules! define_execution_rule {
    ($exec_name: ident) => {
        impl DisassociateAlpha for $exec_name {
            unsafe fn disassociate(&self, in_place: &mut [u8]) {
                unsafe {
                    let mut rem = in_place;

                    for dst in rem.chunks_exact_mut(16 * 4) {
                        let src_ptr = dst.as_ptr();
                        let mut pixel = vld4q_u8(src_ptr);
                        let reciprocal = Self::get_reciprocal(pixel.3);
                        let mask = vceqzq_u8(pixel.3);
                        pixel.0 = Self::unpremultiply_vec(pixel.0, mask, reciprocal);
                        pixel.1 = Self::unpremultiply_vec(pixel.1, mask, reciprocal);
                        pixel.2 = Self::unpremultiply_vec(pixel.2, mask, reciprocal);
                        let dst_ptr = dst.as_mut_ptr();
                        vst4q_u8(dst_ptr, pixel);
                    }

                    rem = rem.chunks_exact_mut(16 * 4).into_remainder();

                    for dst in rem.chunks_exact_mut(8 * 4) {
                        let src_ptr = dst.as_ptr();
                        let mut pixel = vld4_u8(src_ptr);
                        let mask = vceqz_u8(pixel.3);
                        let reciprocal = Self::get_reciprocalh(pixel.3);
                        pixel.0 = Self::unpremultiply_vech(pixel.0, mask, reciprocal);
                        pixel.1 = Self::unpremultiply_vech(pixel.1, mask, reciprocal);
                        pixel.2 = Self::unpremultiply_vech(pixel.2, mask, reciprocal);
                        vst4_u8(dst.as_mut_ptr(), pixel);
                    }

                    rem = rem.chunks_exact_mut(8 * 4).into_remainder();

                    if !rem.is_empty() {
                        assert!(rem.len() < 8 * 4);
                        let mut buffer: [u8; 8 * 4] = [0u8; 8 * 4];
                        std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

                        let mut pixel = vld4_u8(buffer.as_ptr());
                        let mask = vceqz_u8(pixel.3);
                        let reciprocal = Self::get_reciprocalh(pixel.3);
                        pixel.0 = Self::unpremultiply_vech(pixel.0, mask, reciprocal);
                        pixel.1 = Self::unpremultiply_vech(pixel.1, mask, reciprocal);
                        pixel.2 = Self::unpremultiply_vech(pixel.2, mask, reciprocal);
                        vst4_u8(buffer.as_mut_ptr(), pixel);

                        std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
                    }
                }
            }
        }
    };
}

define_execution_rule!(NeonDisassociateAlpha);
#[cfg(feature = "rdm")]
define_execution_rule!(NeonDisassociateAlphaFast);

#[cfg(feature = "rdm")]
impl NeonDisassociateAlphaFast {
    #[inline]
    #[target_feature(enable = "rdm")]
    fn get_reciprocal(alpha: uint8x16_t) -> int32x4x4_t {
        const Q: f32 = ((1i64 << 31) - 1) as f32;

        let q0_31_divider = vdupq_n_f32(Q);

        let a_hi = vmovl_high_u8(alpha);
        let a_lo = vmovl_u8(vget_low_u8(alpha));

        let a_lo_lo = vcvtq_s32_f32(vdivq_f32(
            q0_31_divider,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_lo))),
        ));
        let a_lo_hi = vcvtq_s32_f32(vdivq_f32(
            q0_31_divider,
            vcvtq_f32_u32(vmovl_high_u16(a_lo)),
        ));
        let a_hi_lo = vcvtq_s32_f32(vdivq_f32(
            q0_31_divider,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_hi))),
        ));
        let a_hi_ho = vcvtq_s32_f32(vdivq_f32(
            q0_31_divider,
            vcvtq_f32_u32(vmovl_high_u16(a_hi)),
        ));
        int32x4x4_t(a_lo_lo, a_lo_hi, a_hi_lo, a_hi_ho)
    }

    #[inline]
    #[target_feature(enable = "rdm")]
    fn get_reciprocalh(alpha: uint8x8_t) -> int32x4x2_t {
        const Q: f32 = ((1i64 << 31) - 1) as f32;

        let q0_31_divider = vdupq_n_f32(Q);

        let a_lo = vmovl_u8(alpha);

        let a_lo_lo = vcvtq_s32_f32(vdivq_f32(
            q0_31_divider,
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_lo))),
        ));
        let a_lo_hi = vcvtq_s32_f32(vdivq_f32(
            q0_31_divider,
            vcvtq_f32_u32(vmovl_high_u16(a_lo)),
        ));
        int32x4x2_t(a_lo_lo, a_lo_hi)
    }

    #[inline]
    #[target_feature(enable = "rdm")]
    fn unpremultiply_vec(v: uint8x16_t, mask: uint8x16_t, reciprocal: int32x4x4_t) -> uint8x16_t {
        let scale = vdupq_n_u8(255);
        let hi = vmull_high_u8(v, scale);
        let lo = vmull_u8(vget_low_u8(v), vget_low_u8(scale));
        let lo_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo)));
        let lo_hi = vreinterpretq_s32_u32(vmovl_high_u16(lo));
        let hi_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi)));
        let hi_hi = vreinterpretq_s32_u32(vmovl_high_u16(hi));

        let lo_lo = vqrdmulhq_s32(lo_lo, reciprocal.0);
        let lo_hi = vqrdmulhq_s32(lo_hi, reciprocal.1);
        let hi_lo = vqrdmulhq_s32(hi_lo, reciprocal.2);
        let hi_hi = vqrdmulhq_s32(hi_hi, reciprocal.3);
        let lo = vcombine_u16(vqmovun_s32(lo_lo), vqmovun_s32(lo_hi));
        let hi = vcombine_u16(vqmovun_s32(hi_lo), vqmovun_s32(hi_hi));
        vbslq_u8(
            mask,
            vdupq_n_u8(0),
            vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi)),
        )
    }

    #[inline]
    #[target_feature(enable = "rdm")]
    fn unpremultiply_vech(v: uint8x8_t, mask: uint8x8_t, reciprocal: int32x4x2_t) -> uint8x8_t {
        let scale = vdupq_n_u8(255);
        let lo = vmull_u8(v, vget_low_u8(scale));
        let lo_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo)));
        let lo_hi = vreinterpretq_s32_u32(vmovl_high_u16(lo));

        let lo_lo = vqrdmulhq_s32(lo_lo, reciprocal.0);
        let lo_hi = vqrdmulhq_s32(lo_hi, reciprocal.1);
        let lo = vcombine_u16(vqmovun_s32(lo_lo), vqmovun_s32(lo_hi));
        vbsl_u8(mask, vdup_n_u8(0), vqmovn_u16(lo))
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
    let mut executor: unsafe fn(&mut [u8]) = neon_unpremultiply_alpha_rgba_impl_row;
    #[cfg(feature = "rdm")]
    {
        if _strategy == WorkloadStrategy::PreferSpeed
            && std::arch::is_aarch64_feature_detected!("rdm")
        {
            executor = neon_unpremultiply_alpha_rgba_impl_row_rdm;
        }
    }
    unsafe {
        executor(in_place);
    }
}
