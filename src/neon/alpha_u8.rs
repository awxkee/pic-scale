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

use crate::alpha_handle_u8::premultiply_alpha_rgba_row_impl;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
use std::arch::aarch64::*;

#[inline]
pub unsafe fn neon_div_by_255_n(v: uint16x8_t) -> uint8x8_t {
    vqrshrn_n_u16::<8>(vrsraq_n_u16::<8>(v, v))
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

unsafe fn neon_premultiply_alpha_rgba_impl_row(dst: &mut [u8], src: &[u8]) {
    let mut rem = dst;
    let mut src_rem = src;

    for (dst, src) in rem
        .chunks_exact_mut(64 * 4)
        .zip(src_rem.chunks_exact(64 * 4))
    {
        let src_ptr = src.as_ptr();
        let mut pixel0 = vld4q_u8(src_ptr);
        let mut pixel1 = vld4q_u8(src_ptr.add(16 * 4));
        let mut pixel2 = vld4q_u8(src_ptr.add(16 * 4 * 2));
        let mut pixel3 = vld4q_u8(src_ptr.add(16 * 4 * 3));
        pixel0.0 = premultiply_vec!(pixel0.0, pixel0.3);
        pixel0.1 = premultiply_vec!(pixel0.1, pixel0.3);
        pixel0.2 = premultiply_vec!(pixel0.2, pixel0.3);

        pixel1.0 = premultiply_vec!(pixel1.0, pixel1.3);
        pixel1.1 = premultiply_vec!(pixel1.1, pixel1.3);
        pixel1.2 = premultiply_vec!(pixel1.2, pixel1.3);

        pixel2.0 = premultiply_vec!(pixel2.0, pixel2.3);
        pixel2.1 = premultiply_vec!(pixel2.1, pixel2.3);
        pixel2.2 = premultiply_vec!(pixel2.2, pixel2.3);

        pixel3.0 = premultiply_vec!(pixel3.0, pixel3.3);
        pixel3.1 = premultiply_vec!(pixel3.1, pixel3.3);
        pixel3.2 = premultiply_vec!(pixel3.2, pixel3.3);
        let dst_ptr = dst.as_mut_ptr();
        vst4q_u8(dst_ptr, pixel0);
        vst4q_u8(dst_ptr.add(16 * 4), pixel1);
        vst4q_u8(dst_ptr.add(16 * 4 * 2), pixel2);
        vst4q_u8(dst_ptr.add(16 * 4 * 3), pixel3);
    }

    rem = rem.chunks_exact_mut(64 * 4).into_remainder();
    src_rem = src_rem.chunks_exact(64 * 4).remainder();

    for (dst, src) in rem
        .chunks_exact_mut(16 * 4)
        .zip(src_rem.chunks_exact(16 * 4))
    {
        let src_ptr = src.as_ptr();
        let mut pixel = vld4q_u8(src_ptr);
        pixel.0 = premultiply_vec!(pixel.0, pixel.3);
        pixel.1 = premultiply_vec!(pixel.1, pixel.3);
        pixel.2 = premultiply_vec!(pixel.2, pixel.3);
        let dst_ptr = dst.as_mut_ptr();
        vst4q_u8(dst_ptr, pixel);
    }

    rem = rem.chunks_exact_mut(16 * 4).into_remainder();
    src_rem = src_rem.chunks_exact(16 * 4).remainder();

    premultiply_alpha_rgba_row_impl(rem, src_rem);
}

pub(crate) fn neon_premultiply_alpha_rgba(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    width: usize,
    _: usize,
    src_stride: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(dst_stride)
                .zip(src.par_chunks_exact(src_stride))
                .for_each(|(dst, src)| unsafe {
                    neon_premultiply_alpha_rgba_impl_row(&mut dst[..width * 4], &src[..width * 4]);
                });
        });
    } else {
        dst.chunks_exact_mut(dst_stride)
            .zip(src.chunks_exact(src_stride))
            .for_each(|(dst, src)| unsafe {
                neon_premultiply_alpha_rgba_impl_row(&mut dst[..width * 4], &src[..width * 4]);
            });
    }
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u8]);
}

#[derive(Default)]
struct NeonDisassociateAlpha {}

impl NeonDisassociateAlpha {
    #[inline(always)]
    unsafe fn unpremultiply_vec(v: uint8x16_t, a_values: uint8x16_t) -> uint8x16_t {
        let scale = vdupq_n_u8(255);
        let hi = vmull_high_u8(v, scale);
        let lo = vmull_u8(vget_low_u8(v), vget_low_u8(scale));
        let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
        let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo));
        let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
        let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi));
        let a_hi = vmovl_high_u8(a_values);
        let a_lo = vmovl_u8(vget_low_u8(a_values));
        let a_lo_lo = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_lo))));
        let a_lo_hi = vrecpeq_f32(vcvtq_f32_u32(vmovl_high_u16(a_lo)));
        let a_hi_lo = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_hi))));
        let a_hi_ho = vrecpeq_f32(vcvtq_f32_u32(vmovl_high_u16(a_hi)));

        let lo_lo = vcvtaq_u32_f32(vmulq_f32(lo_lo, a_lo_lo));
        let lo_hi = vcvtaq_u32_f32(vmulq_f32(lo_hi, a_lo_hi));
        let hi_lo = vcvtaq_u32_f32(vmulq_f32(hi_lo, a_hi_lo));
        let hi_hi = vcvtaq_u32_f32(vmulq_f32(hi_hi, a_hi_ho));
        let lo = vcombine_u16(vmovn_u32(lo_lo), vmovn_u32(lo_hi));
        let hi = vcombine_u16(vmovn_u32(hi_lo), vmovn_u32(hi_hi));
        vbslq_u8(
            vceqzq_u8(a_values),
            vdupq_n_u8(0),
            vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi)),
        )
    }

    #[inline(always)]
    unsafe fn unpremultiply_vech(v: uint8x8_t, a_values: uint8x8_t) -> uint8x8_t {
        let scale = vdupq_n_u8(255);
        let lo = vmull_u8(v, vget_low_u8(scale));
        let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
        let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo));
        let a_lo = vmovl_u8(a_values);
        let a_lo_lo = vrecpeq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_lo))));
        let a_lo_hi = vrecpeq_f32(vcvtq_f32_u32(vmovl_high_u16(a_lo)));

        let lo_lo = vcvtaq_u32_f32(vmulq_f32(lo_lo, a_lo_lo));
        let lo_hi = vcvtaq_u32_f32(vmulq_f32(lo_hi, a_lo_hi));
        let lo = vcombine_u16(vmovn_u32(lo_lo), vmovn_u32(lo_hi));
        vbsl_u8(vceqz_u8(a_values), vdup_n_u8(0), vqmovn_u16(lo))
    }
}

impl DisassociateAlpha for NeonDisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u8]) {
        let mut rem = in_place;

        for dst in rem.chunks_exact_mut(16 * 4) {
            let src_ptr = dst.as_ptr();
            let mut pixel = vld4q_u8(src_ptr);
            pixel.0 = Self::unpremultiply_vec(pixel.0, pixel.3);
            pixel.1 = Self::unpremultiply_vec(pixel.1, pixel.3);
            pixel.2 = Self::unpremultiply_vec(pixel.2, pixel.3);
            let dst_ptr = dst.as_mut_ptr();
            vst4q_u8(dst_ptr, pixel);
        }

        rem = rem.chunks_exact_mut(16 * 4).into_remainder();

        for dst in rem.chunks_exact_mut(8 * 4) {
            let src_ptr = dst.as_ptr();
            let mut pixel = vld4_u8(src_ptr);
            pixel.0 = Self::unpremultiply_vech(pixel.0, pixel.3);
            pixel.1 = Self::unpremultiply_vech(pixel.1, pixel.3);
            pixel.2 = Self::unpremultiply_vech(pixel.2, pixel.3);
            vst4_u8(dst.as_mut_ptr(), pixel);
        }

        rem = rem.chunks_exact_mut(8 * 4).into_remainder();

        if !rem.is_empty() {
            assert!(rem.len() < 8 * 4);
            let mut buffer: [u8; 8 * 4] = [0u8; 8 * 4];
            std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

            let mut pixel = vld4_u8(buffer.as_ptr());
            pixel.0 = Self::unpremultiply_vech(pixel.0, pixel.3);
            pixel.1 = Self::unpremultiply_vech(pixel.1, pixel.3);
            pixel.2 = Self::unpremultiply_vech(pixel.2, pixel.3);
            vst4_u8(buffer.as_mut_ptr(), pixel);

            std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

#[inline]
unsafe fn neon_dis_dispatch(in_place: &mut [u8], handler: impl DisassociateAlpha) {
    handler.disassociate(in_place);
}

unsafe fn neon_unpremultiply_alpha_rgba_impl_row(in_place: &mut [u8]) {
    neon_dis_dispatch(in_place, NeonDisassociateAlpha::default());
}

pub(crate) fn neon_unpremultiply_alpha_rgba(
    in_place: &mut [u8],
    width: usize,
    _: usize,
    stride: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            in_place
                .par_chunks_exact_mut(stride)
                .for_each(|row| unsafe {
                    neon_unpremultiply_alpha_rgba_impl_row(&mut row[..width * 4]);
                });
        });
    } else {
        in_place.chunks_exact_mut(stride).for_each(|row| unsafe {
            neon_unpremultiply_alpha_rgba_impl_row(&mut row[..width * 4]);
        });
    }
}
