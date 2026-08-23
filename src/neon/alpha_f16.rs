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

use crate::alpha_handle_f16::{premultiply_pixel_f16_row, unpremultiply_pixel_f16_row};
use core::f16;
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
fn neon_premultiply_alpha_rgba_row_f16(dst: &mut [f16], src: &[f16]) {
    unsafe {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(src_rem.as_chunks::<32>().0.iter())
        {
            let src_ptr = src.as_ptr();
            let pixel = vld4q_u16(src_ptr.cast());

            let low_alpha = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.3)));
            let low_r = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.0))),
                low_alpha,
            );
            let low_g = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.1))),
                low_alpha,
            );
            let low_b = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.2))),
                low_alpha,
            );

            let high_alpha = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.3)));
            let high_r = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.0))),
                high_alpha,
            );
            let high_g = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.1))),
                high_alpha,
            );
            let high_b = vmulq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.2))),
                high_alpha,
            );
            let r_values = vcombine_u16(
                vreinterpret_u16_f16(vcvt_f16_f32(low_r)),
                vreinterpret_u16_f16(vcvt_f16_f32(high_r)),
            );
            let g_values = vcombine_u16(
                vreinterpret_u16_f16(vcvt_f16_f32(low_g)),
                vreinterpret_u16_f16(vcvt_f16_f32(high_g)),
            );
            let b_values = vcombine_u16(
                vreinterpret_u16_f16(vcvt_f16_f32(low_b)),
                vreinterpret_u16_f16(vcvt_f16_f32(high_b)),
            );

            let dst_ptr = dst.as_mut_ptr();
            let store_pixel = uint16x8x4_t(r_values, g_values, b_values, pixel.3);
            vst4q_u16(dst_ptr as *mut u16, store_pixel);
        }

        rem = rem.as_chunks_mut::<32>().1;
        src_rem = src_rem.as_chunks::<32>().1;

        premultiply_pixel_f16_row(rem, src_rem);
    }
}

pub(crate) fn neon_premultiply_alpha_rgba_f16(dst: &mut [f16], src: &[f16]) {
    unsafe {
        neon_premultiply_alpha_rgba_row_f16(dst, src);
    }
}

#[target_feature(enable = "neon")]
fn neon_unpremultiply_alpha_rgba_row_f16(in_place: &mut [f16]) {
    unsafe {
        let mut rem = in_place;

        for dst in rem.as_chunks_mut::<32>().0.iter_mut() {
            let src_ptr = dst.as_ptr();
            let pixel = vld4q_u16(src_ptr.cast());

            let zero_mask = vceqzq_u16(pixel.3);

            let low_alpha = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.3)));

            let low_r = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.0))),
                low_alpha,
            );
            let low_g = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.1))),
                low_alpha,
            );
            let low_b = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(pixel.2))),
                low_alpha,
            );

            let high_alpha = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.3)));

            let high_r = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.0))),
                high_alpha,
            );
            let high_g = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.1))),
                high_alpha,
            );
            let high_b = vdivq_f32(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(pixel.2))),
                high_alpha,
            );

            let u_zeros = vdupq_n_u16(0);

            let r_values = vbslq_u16(
                zero_mask,
                u_zeros,
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(low_r)),
                    vreinterpret_u16_f16(vcvt_f16_f32(high_r)),
                ),
            );
            let g_values = vbslq_u16(
                zero_mask,
                u_zeros,
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(low_g)),
                    vreinterpret_u16_f16(vcvt_f16_f32(high_g)),
                ),
            );
            let b_values = vbslq_u16(
                zero_mask,
                u_zeros,
                vcombine_u16(
                    vreinterpret_u16_f16(vcvt_f16_f32(low_b)),
                    vreinterpret_u16_f16(vcvt_f16_f32(high_b)),
                ),
            );

            let dst_ptr = dst.as_mut_ptr();
            let store_pixel = uint16x8x4_t(r_values, g_values, b_values, pixel.3);
            vst4q_u16(dst_ptr as *mut u16, store_pixel);
        }

        rem = rem.as_chunks_mut::<32>().1;

        unpremultiply_pixel_f16_row(rem);
    }
}

pub(crate) fn neon_unpremultiply_alpha_rgba_f16(in_place: &mut [f16]) {
    unsafe {
        neon_unpremultiply_alpha_rgba_row_f16(in_place);
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::test_utils::{XorShiftRng, assert_f16_slices_close};

    // f16 has only ~10 bits of mantissa, so the f32-domain rounding
    // differences between the scalar reference and the NEON (f16<->f32
    // conversion + multiply/divide) path can move the result by roughly one
    // f16 ULP. Random alpha is kept away from 0 (see `make_rgba`) so
    // unpremultiply's relative error stays small in absolute terms too.
    const ATOL: f32 = 1e-3;

    /// Builds an RGBA f16 buffer in `[0, 1)` that mixes alpha == 0, alpha == 1
    /// and fully random pixels (with random alpha kept away from 0) so both
    /// the vectorized bulk path and the scalar tail remainder exercise every
    /// branch the premultiply/unpremultiply math takes on alpha.
    fn make_rgba(rng: &mut XorShiftRng, pixels: usize) -> Vec<f16> {
        let mut buf = rng.fill_f16_unit(pixels * 4);
        for chunk in buf.as_chunks_mut::<4>().0 {
            chunk[3] = (0.05 + 0.95 * chunk[3] as f32) as f16;
        }
        if let Some(chunk) = buf.first_chunk_mut::<4>() {
            chunk[3] = 0.;
        }
        if pixels > 1 {
            buf[4 + 3] = 1.;
        }
        buf
    }

    #[test]
    fn neon_premultiply_matches_scalar_reference() {
        for pixels in [1usize, 4, 17, 256] {
            let mut rng = XorShiftRng::new(0xA11CE ^ pixels as u64);
            let src = make_rgba(&mut rng, pixels);

            let mut dst_scalar = vec![0f16; pixels * 4];
            let mut dst_neon = vec![0f16; pixels * 4];
            premultiply_pixel_f16_row(&mut dst_scalar, &src);
            neon_premultiply_alpha_rgba_f16(&mut dst_neon, &src);

            assert_f16_slices_close(
                &dst_neon,
                &dst_scalar,
                ATOL,
                &format!("{pixels} pixels: NEON premultiply"),
            );
        }
    }

    #[test]
    fn neon_unpremultiply_matches_scalar_reference() {
        for pixels in [1usize, 4, 17, 256] {
            let mut rng = XorShiftRng::new(0xB0BA ^ pixels as u64);
            let src = make_rgba(&mut rng, pixels);

            let mut scalar_buf = src.clone();
            let mut neon_buf = src;
            unpremultiply_pixel_f16_row(&mut scalar_buf);
            neon_unpremultiply_alpha_rgba_f16(&mut neon_buf);

            assert_f16_slices_close(
                &neon_buf,
                &scalar_buf,
                ATOL,
                &format!("{pixels} pixels: NEON unpremultiply"),
            );
        }
    }
}
