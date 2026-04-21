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

#[inline]
fn neon_div_by_1023_n(v: uint32x4_t) -> uint16x4_t {
    unsafe { vqrshrn_n_u32::<10>(vrsraq_n_u32::<10>(v, v)) }
}

#[inline]
fn neon_div_by_4095_n(v: uint32x4_t) -> uint16x4_t {
    unsafe { vqrshrn_n_u32::<12>(vrsraq_n_u32::<12>(v, v)) }
}

#[inline]
fn neon_div_by_65535_n(v: uint32x4_t) -> uint16x4_t {
    unsafe { vqrshrn_n_u32::<16>(vrsraq_n_u32::<16>(v, v)) }
}

#[inline(always)]
fn neon_div_by<const BIT_DEPTH: usize>(v: uint32x4_t) -> uint16x4_t {
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
    #[target_feature(enable = "neon")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        assert_ne!(bit_depth, 0, "Something goes wrong!");
        assert!((1..=16).contains(&bit_depth));

        static TBL: [u8; 16] = [6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15];
        let shuf_tbl = unsafe { vld1q_u8(TBL.as_ptr()) };
        let alpha_mask = vreinterpretq_u16_u64(vdupq_n_u64(u64::from_ne_bytes([
            0, 0, 0, 0, 0, 0, 255, 255,
        ])));

        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(src_rem.as_chunks::<32>().0.iter())
        {
            let pixel0 = unsafe { vld1q_u16(src.as_ptr()) };
            let pixel1 = unsafe { vld1q_u16(src[8..].as_ptr()) };
            let pixel2 = unsafe { vld1q_u16(src[16..].as_ptr()) };
            let pixel3 = unsafe { vld1q_u16(src[24..].as_ptr()) };

            let alpha0 = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel0), shuf_tbl));
            let alpha1 = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel1), shuf_tbl));
            let alpha2 = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel2), shuf_tbl));
            let alpha3 = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel3), shuf_tbl));

            let mut new_px0 = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel0), vget_low_u16(alpha0))),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel0, alpha0)),
            );
            let mut new_px1 = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel1), vget_low_u16(alpha1))),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel1, alpha1)),
            );
            let mut new_px2 = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel2), vget_low_u16(alpha2))),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel2, alpha2)),
            );
            let mut new_px3 = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel3), vget_low_u16(alpha3))),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel3, alpha3)),
            );

            new_px0 = vbslq_u16(alpha_mask, pixel0, new_px0);
            new_px1 = vbslq_u16(alpha_mask, pixel1, new_px1);
            new_px2 = vbslq_u16(alpha_mask, pixel2, new_px2);
            new_px3 = vbslq_u16(alpha_mask, pixel3, new_px3);

            unsafe {
                vst1q_u16(dst.as_mut_ptr(), new_px0);
                vst1q_u16(dst[8..].as_mut_ptr(), new_px1);
                vst1q_u16(dst[16..].as_mut_ptr(), new_px2);
                vst1q_u16(dst[24..].as_mut_ptr(), new_px3);
            }
        }

        rem = rem.as_chunks_mut::<32>().1;
        src_rem = src_rem.as_chunks::<32>().1;

        for (dst, src) in rem
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(src_rem.as_chunks::<8>().0.iter())
        {
            let pixel = unsafe { vld1q_u16(src.as_ptr()) };

            let alpha = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel), shuf_tbl));

            let new_px = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel), vget_low_u16(alpha))),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel, alpha)),
            );

            let new_px = vbslq_u16(alpha_mask, pixel, new_px);

            unsafe {
                vst1q_u16(dst.as_mut_ptr(), new_px);
            }
        }

        rem = rem.as_chunks_mut::<8>().1;
        src_rem = src_rem.as_chunks::<8>().1;

        if !rem.is_empty() {
            let mut buffer: [u16; 8] = [0u16; 8];
            buffer[..src_rem.len()].copy_from_slice(src_rem);

            let pixel = unsafe { vld1q_u16(buffer.as_ptr()) };

            let alpha = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel), shuf_tbl));

            let new_px = vcombine_u16(
                neon_div_by::<BIT_DEPTH>(vmull_u16(vget_low_u16(pixel), vget_low_u16(alpha))),
                neon_div_by::<BIT_DEPTH>(vmull_high_u16(pixel, alpha)),
            );

            let new_px = vbslq_u16(alpha_mask, pixel, new_px);

            unsafe {
                vst1q_u16(buffer.as_mut_ptr(), new_px);
            }

            rem.copy_from_slice(&buffer[..rem.len()]);
        }
    }
}

#[derive(Default)]
struct NeonPremultiplyExecutorAnyBitDepth {}

impl NeonPremultiplyExecutor for NeonPremultiplyExecutorAnyBitDepth {
    #[target_feature(enable = "neon")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        assert_ne!(bit_depth, 0, "Something goes wrong!");
        assert!((1..=16).contains(&bit_depth));
        let max_colors = (1u32 << bit_depth) - 1;
        let mut rem = dst;
        let mut src_rem = src;

        let v_max_colors_scale = vdupq_n_f32((1. / max_colors as f64) as f32);
        let vmax_colors = vdupq_n_u16(max_colors as u16);

        static TBL: [u8; 16] = [6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15];
        let shuf_tbl = unsafe { vld1q_u8(TBL.as_ptr()) };
        let alpha_mask = vreinterpretq_u16_u64(vdupq_n_u64(u64::from_ne_bytes([
            0, 0, 0, 0, 0, 0, 255, 255,
        ])));

        for (dst, src) in rem
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(src_rem.as_chunks::<8>().0.iter())
        {
            let pixel = unsafe { vld1q_u16(src.as_ptr()) };

            let alpha = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel), shuf_tbl));

            let low_a = vmull_u16(vget_low_u16(pixel), vget_low_u16(alpha));
            let high_a = vmull_high_u16(pixel, alpha);

            let p0 = vcvtaq_u32_f32(vmulq_f32(vcvtq_f32_u32(low_a), v_max_colors_scale));
            let p1 = vcvtaq_u32_f32(vmulq_f32(vcvtq_f32_u32(high_a), v_max_colors_scale));

            let mut packed = vcombine_u16(vqmovn_u32(p0), vqmovn_u32(p1));
            packed = vminq_u16(packed, vmax_colors);

            let new_px = vbslq_u16(alpha_mask, pixel, packed);

            unsafe {
                vst1q_u16(dst.as_mut_ptr(), new_px);
            }
        }

        rem = rem.as_chunks_mut::<8>().1;
        src_rem = src_rem.as_chunks::<8>().1;

        if !rem.is_empty() {
            let mut buffer: [u16; 8] = [0u16; 8];
            buffer[..src_rem.len()].copy_from_slice(src_rem);

            let pixel = unsafe { vld1q_u16(buffer.as_ptr()) };

            let alpha = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel), shuf_tbl));

            let low_a = vmull_u16(vget_low_u16(pixel), vget_low_u16(alpha));
            let high_a = vmull_high_u16(pixel, alpha);

            let p0 = vcvtaq_u32_f32(vmulq_f32(vcvtq_f32_u32(low_a), v_max_colors_scale));
            let p1 = vcvtaq_u32_f32(vmulq_f32(vcvtq_f32_u32(high_a), v_max_colors_scale));

            let mut packed = vcombine_u16(vqmovn_u32(p0), vqmovn_u32(p1));
            packed = vminq_u16(packed, vmax_colors);

            let new_px = vbslq_u16(alpha_mask, pixel, packed);

            unsafe {
                vst1q_u16(buffer.as_mut_ptr(), new_px);
            }

            rem.copy_from_slice(&buffer[..rem.len()]);
        }
    }
}

#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
fn neon_pa_dispatch(
    dst: &mut [u16],
    src: &[u16],
    bit_depth: usize,
    dispatch: impl NeonPremultiplyExecutor,
) {
    unsafe { dispatch.premultiply(dst, src, bit_depth) }
}

pub(crate) fn neon_premultiply_alpha_rgba_u16(dst: &mut [u16], src: &[u16], bit_depth: usize) {
    unsafe {
        neon_premultiply_alpha_rgba_row_u16(dst, src, bit_depth);
    }
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u16], bit_depth: usize);
}

#[derive(Default)]
struct NeonDisassociateAlpha {}

impl DisassociateAlpha for NeonDisassociateAlpha {
    #[target_feature(enable = "neon")]
    unsafe fn disassociate(&self, in_place: &mut [u16], bit_depth: usize) {
        let max_colors = (1u32 << bit_depth) - 1;

        static TBL: [u8; 16] = [6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15];
        let shuf_tbl = unsafe { vld1q_u8(TBL.as_ptr()) };
        let copy_alpha_mask = vreinterpretq_u16_u64(vdupq_n_u64(u64::from_ne_bytes([
            0, 0, 0, 0, 0, 0, 255, 255,
        ])));

        let mut rem = in_place;

        let v_max_colors_f = vdupq_n_f32(max_colors as f32);
        let v_max_test = vdupq_n_u16(max_colors as u16);

        for dst in rem.as_chunks_mut::<16>().0.iter_mut() {
            let pixel0 = unsafe { vld1q_u16(dst.as_ptr()) };
            let pixel1 = unsafe { vld1q_u16(dst[8..].as_ptr()) };

            let alpha0 = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel0), shuf_tbl));
            let alpha1 = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel1), shuf_tbl));

            let pa0 = vzip1_u16(vget_low_u16(alpha0), vget_high_u16(alpha0));
            let pa1 = vzip1_u16(vget_low_u16(alpha1), vget_high_u16(alpha1));
            let pa_full = vreinterpret_u16_u32(vzip2_u32(
                vreinterpret_u32_u16(pa0),
                vreinterpret_u32_u16(pa1),
            ));

            let packed_alpha = vcvtq_f32_u32(vmovl_u16(pa_full));

            let recip_alphas = vdivq_f32(v_max_colors_f, packed_alpha);

            let is_alpha_zero_mask0 = vceqzq_u16(alpha0);
            let is_alpha_zero_mask1 = vceqzq_u16(alpha1);

            let low0 = vmovl_u16(vget_low_u16(pixel0));
            let high0 = vmovl_high_u16(pixel0);

            let low1 = vmovl_u16(vget_low_u16(pixel1));
            let high1 = vmovl_high_u16(pixel1);

            let low0 = vmulq_laneq_f32::<0>(vcvtq_f32_u32(low0), recip_alphas);
            let hi0 = vmulq_laneq_f32::<1>(vcvtq_f32_u32(high0), recip_alphas);
            let low1 = vmulq_laneq_f32::<2>(vcvtq_f32_u32(low1), recip_alphas);
            let hi1 = vmulq_laneq_f32::<3>(vcvtq_f32_u32(high1), recip_alphas);

            let mut packed0 = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(low0)),
                vqmovn_u32(vcvtaq_u32_f32(hi0)),
            );
            let mut packed1 = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(low1)),
                vqmovn_u32(vcvtaq_u32_f32(hi1)),
            );

            packed0 = vminq_u16(packed0, v_max_test);
            packed0 = vbslq_u16(is_alpha_zero_mask0, vdupq_n_u16(0), packed0);
            packed0 = vbslq_u16(copy_alpha_mask, pixel0, packed0);

            packed1 = vminq_u16(packed1, v_max_test);
            packed1 = vbslq_u16(is_alpha_zero_mask1, vdupq_n_u16(0), packed1);
            packed1 = vbslq_u16(copy_alpha_mask, pixel1, packed1);

            unsafe {
                vst1q_u16(dst.as_mut_ptr(), packed0);
                vst1q_u16(dst[8..].as_mut_ptr(), packed1);
            }
        }

        rem = rem.as_chunks_mut::<16>().1;

        for dst in rem.as_chunks_mut::<8>().0.iter_mut() {
            let pixel = unsafe { vld1q_u16(dst.as_ptr()) };

            let alpha = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel), shuf_tbl));
            let packed_alpha = vcvtq_f32_u32(vmovl_u16(vzip1_u16(
                vget_low_u16(alpha),
                vget_high_u16(alpha),
            )));
            let recip_alphas = vdivq_f32(v_max_colors_f, packed_alpha);

            let is_alpha_zero_mask = vceqzq_u16(alpha);

            let low = vmovl_u16(vget_low_u16(pixel));
            let high = vmovl_high_u16(pixel);

            let low = vmulq_laneq_f32::<0>(vcvtq_f32_u32(low), recip_alphas);
            let hi = vmulq_laneq_f32::<1>(vcvtq_f32_u32(high), recip_alphas);

            let mut packed = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(low)),
                vqmovn_u32(vcvtaq_u32_f32(hi)),
            );

            packed = vminq_u16(packed, v_max_test);
            packed = vbslq_u16(is_alpha_zero_mask, vdupq_n_u16(0), packed);
            packed = vbslq_u16(copy_alpha_mask, pixel, packed);

            unsafe {
                vst1q_u16(dst.as_mut_ptr(), packed);
            }
        }

        rem = rem.as_chunks_mut::<8>().1;

        if !rem.is_empty() {
            let mut buffer: [u16; 8] = [0u16; 8];
            buffer[..rem.len()].copy_from_slice(rem);

            let pixel = unsafe { vld1q_u16(buffer.as_ptr()) };

            let alpha = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(pixel), shuf_tbl));
            let packed_alpha = vcvtq_f32_u32(vmovl_u16(vzip1_u16(
                vget_low_u16(alpha),
                vget_high_u16(alpha),
            )));
            let recip_alphas = vdivq_f32(v_max_colors_f, packed_alpha);

            let is_alpha_zero_mask = vceqzq_u16(alpha);

            let low = vmovl_u16(vget_low_u16(pixel));
            let high = vmovl_high_u16(pixel);

            let low = vmulq_laneq_f32::<0>(vcvtq_f32_u32(low), recip_alphas);
            let hi = vmulq_laneq_f32::<1>(vcvtq_f32_u32(high), recip_alphas);

            let mut packed = vcombine_u16(
                vqmovn_u32(vcvtaq_u32_f32(low)),
                vqmovn_u32(vcvtaq_u32_f32(hi)),
            );

            packed = vminq_u16(packed, v_max_test);
            packed = vbslq_u16(is_alpha_zero_mask, vdupq_n_u16(0), packed);
            packed = vbslq_u16(copy_alpha_mask, pixel, packed);

            unsafe {
                vst1q_u16(buffer.as_mut_ptr(), packed);
            }

            rem.copy_from_slice(&buffer[..rem.len()]);
        }
    }
}

#[inline]
fn neon_un_row(in_place: &mut [u16], bit_depth: usize, handler: impl DisassociateAlpha) {
    unsafe {
        handler.disassociate(in_place, bit_depth);
    }
}

fn neon_unpremultiply_alpha_rgba_row_u16(in_place: &mut [u16], bit_depth: usize) {
    neon_un_row(in_place, bit_depth, NeonDisassociateAlpha::default());
}

pub(crate) fn neon_unpremultiply_alpha_rgba_u16(in_place: &mut [u16], bit_depth: usize) {
    neon_unpremultiply_alpha_rgba_row_u16(in_place, bit_depth);
}
