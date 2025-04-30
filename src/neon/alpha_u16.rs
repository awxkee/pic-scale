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
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
use std::arch::aarch64::*;

#[inline]
unsafe fn neon_div_by_1023_n(v: uint32x4_t) -> uint16x4_t {
    vqrshrn_n_u32::<10>(vrsraq_n_u32::<10>(v, v))
}

#[inline]
unsafe fn neon_div_by_4095_n(v: uint32x4_t) -> uint16x4_t {
    vqrshrn_n_u32::<12>(vrsraq_n_u32::<12>(v, v))
}

#[inline]
unsafe fn neon_div_by_65535_n(v: uint32x4_t) -> uint16x4_t {
    vqrshrn_n_u32::<16>(vrsraq_n_u32::<16>(v, v))
}

#[inline(always)]
unsafe fn neon_div_by<const BIT_DEPTH: usize>(v: uint32x4_t) -> uint16x4_t {
    match BIT_DEPTH {
        10 => neon_div_by_1023_n(v),
        12 => neon_div_by_4095_n(v),
        16 => neon_div_by_65535_n(v),
        _ => neon_div_by_1023_n(v),
    }
}

trait NeonPremultiplyExecutor {
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize);
}

#[derive(Default)]
struct NeonPremultiplyExecutorDefault<const BIT_DEPTH: usize> {}

impl<const BIT_DEPTH: usize> NeonPremultiplyExecutor for NeonPremultiplyExecutorDefault<BIT_DEPTH> {
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        assert_ne!(bit_depth, 0, "Something goes wrong!");
        assert!((1..=16).contains(&bit_depth));

        let mut rem = dst;
        let mut src_rem = src;
        for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
            let pixel = vld4q_u16(src.as_ptr());

            let low_a = vget_low_u16(pixel.3);

            let new_r = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel.0), low_a)),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel.0, pixel.3)),
            );

            let new_g = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel.1), low_a)),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel.1, pixel.3)),
            );

            let new_b = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel.2), low_a)),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel.2, pixel.3)),
            );

            let new_px = uint16x8x4_t(new_r, new_g, new_b, pixel.3);

            vst4q_u16(dst.as_mut_ptr(), new_px);
        }

        rem = rem.chunks_exact_mut(8 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(8 * 4).remainder();

        if !rem.is_empty() {
            assert!(src_rem.len() < 8 * 4);
            assert!(rem.len() < 8 * 4);
            assert_eq!(src_rem.len(), rem.len());

            let mut buffer: [u16; 8 * 4] = [0u16; 8 * 4];
            std::ptr::copy_nonoverlapping(src_rem.as_ptr(), buffer.as_mut_ptr(), src_rem.len());

            let pixel = vld4q_u16(buffer.as_ptr());

            let low_a = vget_low_u16(pixel.3);

            let new_r = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel.0), low_a)),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel.0, pixel.3)),
            );

            let new_g = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel.1), low_a)),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel.1, pixel.3)),
            );

            let new_b = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel.2), low_a)),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel.2, pixel.3)),
            );

            let new_px = uint16x8x4_t(new_r, new_g, new_b, pixel.3);

            vst4q_u16(buffer.as_mut_ptr(), new_px);

            std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

#[derive(Default)]
struct NeonPremultiplyExecutorAnyBitDepth {}

impl NeonPremultiplyExecutor for NeonPremultiplyExecutorAnyBitDepth {
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        assert_ne!(bit_depth, 0, "Something goes wrong!");
        assert!((1..=16).contains(&bit_depth));
        let max_colors = (1u32 << bit_depth) - 1;
        let mut rem = dst;
        let mut src_rem = src;

        let v_max_colors_scale = vdupq_n_f32((1. / max_colors as f64) as f32);

        for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
            let pixel = vld4q_u16(src.as_ptr());

            let low_a = vmovl_u16(vget_low_u16(pixel.3));
            let high_a = vmovl_high_u16(pixel.3);

            let low_a = vmulq_f32(vcvtq_f32_u32(low_a), v_max_colors_scale);
            let hi_a = vmulq_f32(vcvtq_f32_u32(high_a), v_max_colors_scale);

            let new_r = v_scale_by_alpha(pixel.0, low_a, hi_a);
            let new_g = v_scale_by_alpha(pixel.1, low_a, hi_a);
            let new_b = v_scale_by_alpha(pixel.2, low_a, hi_a);

            let new_px = uint16x8x4_t(new_r, new_g, new_b, pixel.3);

            vst4q_u16(dst.as_mut_ptr(), new_px);
        }

        rem = rem.chunks_exact_mut(8 * 4).into_remainder();
        src_rem = src_rem.chunks_exact(8 * 4).remainder();

        if !rem.is_empty() {
            assert!(src_rem.len() < 8 * 4);
            assert!(rem.len() < 8 * 4);
            assert_eq!(src_rem.len(), rem.len());
            let mut buffer: [u16; 8 * 4] = [0u16; 8 * 4];
            std::ptr::copy_nonoverlapping(src_rem.as_ptr(), buffer.as_mut_ptr(), src_rem.len());

            let pixel = vld4q_u16(buffer.as_ptr());

            let low_a = vmovl_u16(vget_low_u16(pixel.3));
            let high_a = vmovl_high_u16(pixel.3);

            let low_a = vmulq_f32(vcvtq_f32_u32(low_a), v_max_colors_scale);
            let hi_a = vmulq_f32(vcvtq_f32_u32(high_a), v_max_colors_scale);

            let new_r = v_scale_by_alpha(pixel.0, low_a, hi_a);
            let new_g = v_scale_by_alpha(pixel.1, low_a, hi_a);
            let new_b = v_scale_by_alpha(pixel.2, low_a, hi_a);

            let new_px = uint16x8x4_t(new_r, new_g, new_b, pixel.3);

            vst4q_u16(buffer.as_mut_ptr(), new_px);

            std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

fn neon_premultiply_alpha_rgba_row_u16(dst: &mut [u16], src: &[u16], bit_depth: usize) {
    assert_ne!(bit_depth, 0, "Something goes wrong!");

    if bit_depth == 10 {
        neon_pa_dispatch(
            dst,
            src,
            bit_depth,
            NeonPremultiplyExecutorDefault::<10>::default(),
        )
    } else if bit_depth == 12 {
        neon_pa_dispatch(
            dst,
            src,
            bit_depth,
            NeonPremultiplyExecutorDefault::<12>::default(),
        )
    } else if bit_depth == 16 {
        neon_pa_dispatch(
            dst,
            src,
            bit_depth,
            NeonPremultiplyExecutorDefault::<16>::default(),
        )
    } else {
        neon_pa_dispatch(
            dst,
            src,
            bit_depth,
            NeonPremultiplyExecutorAnyBitDepth::default(),
        )
    }
}

#[inline]
fn neon_pa_dispatch(
    dst: &mut [u16],
    src: &[u16],
    bit_depth: usize,
    dispatch: impl NeonPremultiplyExecutor,
) {
    unsafe { dispatch.premultiply(dst, src, bit_depth) }
}

pub(crate) fn neon_premultiply_alpha_rgba_u16(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    width: usize,
    _: usize,
    src_stride: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(dst_stride)
                .zip(src.par_chunks_exact(src_stride))
                .for_each(|(dst, src)| {
                    neon_premultiply_alpha_rgba_row_u16(
                        &mut dst[..width * 4],
                        &src[..width * 4],
                        bit_depth,
                    );
                });
        });
    } else {
        dst.chunks_exact_mut(dst_stride)
            .zip(src.chunks_exact(src_stride))
            .for_each(|(dst, src)| {
                neon_premultiply_alpha_rgba_row_u16(
                    &mut dst[..width * 4],
                    &src[..width * 4],
                    bit_depth,
                );
            });
    }
}

#[inline]
unsafe fn v_scale_by_alpha(
    px: uint16x8_t,
    low_low_a: float32x4_t,
    low_high_a: float32x4_t,
) -> uint16x8_t {
    let low_px_u = vmovl_u16(vget_low_u16(px));
    let high_px_u = vmovl_high_u16(px);

    let low_px = vcvtq_f32_u32(low_px_u);
    let high_px = vcvtq_f32_u32(high_px_u);

    let new_ll = vcvtaq_u32_f32(vmulq_f32(low_px, low_low_a));
    let new_lh = vcvtaq_u32_f32(vmulq_f32(high_px, low_high_a));

    vcombine_u16(vqmovn_u32(new_ll), vqmovn_u32(new_lh))
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u16], bit_depth: usize);
}

#[derive(Default)]
struct NeonDisassociateAlpha {}

impl DisassociateAlpha for NeonDisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u16], bit_depth: usize) {
        let max_colors = (1u32 << bit_depth) - 1;

        let mut rem = in_place;

        let v_max_colors_f = vdupq_n_f32(max_colors as f32);
        let ones = vdupq_n_f32(1.);
        let v_max_test = vdupq_n_u16(max_colors as u16);

        for dst in rem.chunks_exact_mut(8 * 4) {
            let pixel = vld4q_u16(dst.as_ptr());

            let is_alpha_zero_mask = vceqzq_u16(pixel.3);

            let low_a = vmovl_u16(vget_low_u16(pixel.3));
            let high_a = vmovl_high_u16(pixel.3);

            let low_a = vmulq_f32(vdivq_f32(ones, vcvtq_f32_u32(low_a)), v_max_colors_f);
            let hi_a = vmulq_f32(vdivq_f32(ones, vcvtq_f32_u32(high_a)), v_max_colors_f);

            let mut new_r = vbslq_u16(
                is_alpha_zero_mask,
                pixel.0,
                v_scale_by_alpha(pixel.0, low_a, hi_a),
            );

            let mut new_g = vbslq_u16(
                is_alpha_zero_mask,
                pixel.1,
                v_scale_by_alpha(pixel.1, low_a, hi_a),
            );

            let mut new_b = vbslq_u16(
                is_alpha_zero_mask,
                pixel.2,
                v_scale_by_alpha(pixel.2, low_a, hi_a),
            );

            new_r = vminq_u16(new_r, v_max_test);
            new_g = vminq_u16(new_g, v_max_test);
            new_b = vminq_u16(new_b, v_max_test);

            let new_px = uint16x8x4_t(new_r, new_g, new_b, pixel.3);

            vst4q_u16(dst.as_mut_ptr(), new_px);
        }
        rem = rem.chunks_exact_mut(8 * 4).into_remainder();

        if !rem.is_empty() {
            assert!(rem.len() < 8 * 4);
            let mut buffer: [u16; 8 * 4] = [0u16; 8 * 4];
            std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

            let pixel = vld4q_u16(buffer.as_ptr());

            let is_alpha_zero_mask = vceqzq_u16(pixel.3);

            let low_a = vmovl_u16(vget_low_u16(pixel.3));
            let high_a = vmovl_high_u16(pixel.3);

            let low_a = vmulq_f32(vdivq_f32(ones, vcvtq_f32_u32(low_a)), v_max_colors_f);
            let hi_a = vmulq_f32(vdivq_f32(ones, vcvtq_f32_u32(high_a)), v_max_colors_f);

            let mut new_r = vbslq_u16(
                is_alpha_zero_mask,
                pixel.0,
                v_scale_by_alpha(pixel.0, low_a, hi_a),
            );

            let mut new_g = vbslq_u16(
                is_alpha_zero_mask,
                pixel.1,
                v_scale_by_alpha(pixel.1, low_a, hi_a),
            );

            let mut new_b = vbslq_u16(
                is_alpha_zero_mask,
                pixel.2,
                v_scale_by_alpha(pixel.2, low_a, hi_a),
            );

            new_r = vminq_u16(new_r, v_max_test);
            new_g = vminq_u16(new_g, v_max_test);
            new_b = vminq_u16(new_b, v_max_test);

            let new_px = uint16x8x4_t(new_r, new_g, new_b, pixel.3);

            vst4q_u16(buffer.as_mut_ptr(), new_px);

            std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

#[inline]
unsafe fn neon_un_row(in_place: &mut [u16], bit_depth: usize, handler: impl DisassociateAlpha) {
    handler.disassociate(in_place, bit_depth);
}

fn neon_unpremultiply_alpha_rgba_row_u16(in_place: &mut [u16], bit_depth: usize) {
    unsafe {
        neon_un_row(in_place, bit_depth, NeonDisassociateAlpha::default());
    }
}

pub(crate) fn neon_unpremultiply_alpha_rgba_u16(
    in_place: &mut [u16],
    src_stride: usize,
    width: usize,
    _: usize,
    bit_depth: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool.as_ref() {
        pool.install(|| {
            in_place.par_chunks_exact_mut(src_stride).for_each(|row| {
                neon_unpremultiply_alpha_rgba_row_u16(&mut row[..width * 4], bit_depth);
            });
        });
    } else {
        in_place.chunks_exact_mut(src_stride).for_each(|row| {
            neon_unpremultiply_alpha_rgba_row_u16(&mut row[..width * 4], bit_depth);
        });
    }
}
