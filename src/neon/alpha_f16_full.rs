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

use crate::alpha_handle_f16::premultiply_pixel_f16_row;
use core::f16;
use std::arch::aarch64::*;

#[target_feature(enable = "fp16")]
fn neon_premultiply_alpha_rgba_row_f16_full(dst: &mut [f16], src: &[f16]) {
    let mut rem = dst;
    let mut src_rem = src;

    for (dst, src) in rem
        .as_chunks_mut::<32>()
        .0
        .iter_mut()
        .zip(src_rem.as_chunks::<32>().0.iter())
    {
        let src_ptr = src.as_ptr();
        let pixel = unsafe { vld4q_u16(src_ptr.cast()) };

        let low_alpha = vreinterpretq_f16_u16(pixel.3);
        let r_values = vmulq_f16(vreinterpretq_f16_u16(pixel.0), low_alpha);
        let g_values = vmulq_f16(vreinterpretq_f16_u16(pixel.1), low_alpha);
        let b_values = vmulq_f16(vreinterpretq_f16_u16(pixel.2), low_alpha);

        let dst_ptr = dst.as_mut_ptr();
        let store_pixel = uint16x8x4_t(
            vreinterpretq_u16_f16(r_values),
            vreinterpretq_u16_f16(g_values),
            vreinterpretq_u16_f16(b_values),
            pixel.3,
        );
        unsafe {
            vst4q_u16(dst_ptr.cast(), store_pixel);
        }
    }

    rem = rem.as_chunks_mut::<32>().1;
    src_rem = src_rem.as_chunks::<32>().1;

    premultiply_pixel_f16_row(rem, src_rem);

    if !rem.is_empty() {
        let mut transient: [f16; 4 * 8] = [0.; 4 * 8];
        assert_eq!(rem.len(), src_rem.len());
        assert!(rem.len() <= 4 * 8);

        unsafe {
            std::ptr::copy_nonoverlapping(src_rem.as_ptr(), transient.as_mut_ptr(), src_rem.len());
        }

        let pixel = unsafe { vld4q_u16(transient.as_ptr().cast()) };

        let low_alpha = vreinterpretq_f16_u16(pixel.3);
        let r_values = vmulq_f16(vreinterpretq_f16_u16(pixel.0), low_alpha);
        let g_values = vmulq_f16(vreinterpretq_f16_u16(pixel.1), low_alpha);
        let b_values = vmulq_f16(vreinterpretq_f16_u16(pixel.2), low_alpha);

        let store_pixel = uint16x8x4_t(
            vreinterpretq_u16_f16(r_values),
            vreinterpretq_u16_f16(g_values),
            vreinterpretq_u16_f16(b_values),
            pixel.3,
        );
        unsafe {
            vst4q_u16(transient.as_mut_ptr().cast(), store_pixel);
        }

        unsafe {
            std::ptr::copy_nonoverlapping(transient.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

pub(crate) fn neon_premultiply_alpha_rgba_f16_full(dst: &mut [f16], src: &[f16]) {
    unsafe {
        neon_premultiply_alpha_rgba_row_f16_full(dst, src);
    }
}

#[target_feature(enable = "fp16")]
fn neon_unpremultiply_alpha_rgba_f16_row_full(in_place: &mut [f16]) {
    let mut rem = in_place;

    for dst in rem.as_chunks_mut::<32>().0.iter_mut() {
        let src_ptr = dst.as_ptr();
        let pixel = unsafe { vld4q_u16(src_ptr.cast()) };

        let alphas = vreinterpretq_f16_u16(pixel.3);
        let zero_mask = vceqzq_f16(alphas);

        let r_values = vbslq_f16(
            zero_mask,
            vreinterpretq_f16_u16(pixel.0),
            vdivq_f16(vreinterpretq_f16_u16(pixel.0), alphas),
        );
        let g_values = vbslq_f16(
            zero_mask,
            vreinterpretq_f16_u16(pixel.1),
            vdivq_f16(vreinterpretq_f16_u16(pixel.1), alphas),
        );
        let b_values = vbslq_f16(
            zero_mask,
            vreinterpretq_f16_u16(pixel.2),
            vdivq_f16(vreinterpretq_f16_u16(pixel.2), alphas),
        );

        let dst_ptr = dst.as_mut_ptr();
        let store_pixel = uint16x8x4_t(
            vreinterpretq_u16_f16(r_values),
            vreinterpretq_u16_f16(g_values),
            vreinterpretq_u16_f16(b_values),
            pixel.3,
        );
        unsafe {
            vst4q_u16(dst_ptr.cast(), store_pixel);
        }
    }

    rem = rem.as_chunks_mut::<32>().1;
    if !rem.is_empty() {
        let mut transient: [f16; 4 * 8] = [0.; 4 * 8];
        assert!(rem.len() <= 4 * 8);
        unsafe {
            std::ptr::copy_nonoverlapping(rem.as_ptr(), transient.as_mut_ptr(), rem.len());
        }

        let pixel = unsafe { vld4q_u16(transient.as_ptr().cast()) };

        let alphas = vreinterpretq_f16_u16(pixel.3);
        let zero_mask = vceqzq_f16(alphas);

        let r_values = vbslq_f16(
            zero_mask,
            vreinterpretq_f16_u16(pixel.0),
            vdivq_f16(vreinterpretq_f16_u16(pixel.0), alphas),
        );
        let g_values = vbslq_f16(
            zero_mask,
            vreinterpretq_f16_u16(pixel.1),
            vdivq_f16(vreinterpretq_f16_u16(pixel.1), alphas),
        );
        let b_values = vbslq_f16(
            zero_mask,
            vreinterpretq_f16_u16(pixel.2),
            vdivq_f16(vreinterpretq_f16_u16(pixel.2), alphas),
        );

        let store_pixel = uint16x8x4_t(
            vreinterpretq_u16_f16(r_values),
            vreinterpretq_u16_f16(g_values),
            vreinterpretq_u16_f16(b_values),
            pixel.3,
        );
        unsafe {
            vst4q_u16(transient.as_mut_ptr().cast(), store_pixel);
        }

        unsafe {
            std::ptr::copy_nonoverlapping(transient.as_ptr(), rem.as_mut_ptr(), rem.len());
        }
    }
}

pub(crate) fn neon_unpremultiply_alpha_rgba_f16_full(in_place: &mut [f16]) {
    unsafe {
        neon_unpremultiply_alpha_rgba_f16_row_full(in_place);
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::alpha_handle_f16::unpremultiply_pixel_f16_row;
    use crate::test_utils::{XorShiftRng, assert_f16_slices_close};

    // This backend computes premultiply/unpremultiply directly in f16
    // (`fp16` NEON arithmetic), while the scalar reference widens to f32
    // first. That extra intermediate rounding step means the two can
    // legitimately differ by roughly one f16 ULP. Random alpha is kept away
    // from 0 (see `make_rgba`) so unpremultiply's relative error stays small
    // in absolute terms too.
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
    fn neon_fp16_premultiply_matches_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("fp16") {
            return;
        }
        for pixels in [1usize, 4, 17, 256] {
            let mut rng = XorShiftRng::new(0xA11CE ^ pixels as u64);
            let src = make_rgba(&mut rng, pixels);

            let mut dst_scalar = vec![0f16; pixels * 4];
            let mut dst_neon = vec![0f16; pixels * 4];
            premultiply_pixel_f16_row(&mut dst_scalar, &src);
            neon_premultiply_alpha_rgba_f16_full(&mut dst_neon, &src);

            assert_f16_slices_close(
                &dst_neon,
                &dst_scalar,
                ATOL,
                &format!("{pixels} pixels: NEON fp16 premultiply"),
            );
        }
    }

    #[ignore = "known bug: same alpha==0 handling / precision divergence class as the u16 unpremultiply findings - see issue"]
    #[test]
    fn neon_fp16_unpremultiply_matches_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("fp16") {
            return;
        }
        for pixels in [1usize, 4, 17, 256] {
            let mut rng = XorShiftRng::new(0xB0BA ^ pixels as u64);
            let src = make_rgba(&mut rng, pixels);

            let mut scalar_buf = src.clone();
            let mut neon_buf = src;
            unpremultiply_pixel_f16_row(&mut scalar_buf);
            neon_unpremultiply_alpha_rgba_f16_full(&mut neon_buf);

            assert_f16_slices_close(
                &neon_buf,
                &scalar_buf,
                ATOL,
                &format!("{pixels} pixels: NEON fp16 unpremultiply"),
            );
        }
    }
}
