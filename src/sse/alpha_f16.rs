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
use crate::sse::f16_utils::{_mm_cvtph_psx, _mm_cvtps_phx};
use crate::sse::{sse_deinterleave_rgba_epi16, sse_interleave_rgba_epi16};
use core::f16;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_premultiply_alpha_rgba_f16(dst: &mut [f16], src: &[f16]) {
    unsafe {
        sse_premultiply_alpha_rgba_f16_regular(dst, src);
    }
}

#[target_feature(enable = "sse4.1")]
fn sse_premultiply_alpha_rgba_f16_regular(dst: &mut [f16], src: &[f16]) {
    sse_premultiply_alpha_rgba_row_f16_impl::<false>(dst, src);
}

#[inline(always)]
fn sse_premultiply_alpha_rgba_row_f16_impl<const F16C: bool>(dst: &mut [f16], src: &[f16]) {
    unsafe {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(src_rem.as_chunks::<32>().0.iter())
        {
            let lane0 = _mm_loadu_si128(src.as_ptr().cast());
            let lane1 = _mm_loadu_si128(src[8..].as_ptr().cast());
            let lane2 = _mm_loadu_si128(src[16..].as_ptr().cast());
            let lane3 = _mm_loadu_si128(src[24..].as_ptr().cast());
            let pixel = sse_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let low_alpha = _mm_cvtph_psx::<F16C>(pixel.3);
            let low_r = _mm_mul_ps(_mm_cvtph_psx::<F16C>(pixel.0), low_alpha);
            let low_g = _mm_mul_ps(_mm_cvtph_psx::<F16C>(pixel.1), low_alpha);
            let low_b = _mm_mul_ps(_mm_cvtph_psx::<F16C>(pixel.2), low_alpha);

            let high_alpha = _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.3));
            let high_r = _mm_mul_ps(
                _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.0)),
                high_alpha,
            );
            let high_g = _mm_mul_ps(
                _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.1)),
                high_alpha,
            );
            let high_b = _mm_mul_ps(
                _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.2)),
                high_alpha,
            );
            let r_values =
                _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_r), _mm_cvtps_phx::<F16C>(high_r));
            let g_values =
                _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_g), _mm_cvtps_phx::<F16C>(high_g));
            let b_values =
                _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_b), _mm_cvtps_phx::<F16C>(high_b));
            let (d_lane0, d_lane1, d_lane2, d_lane3) =
                sse_interleave_rgba_epi16(r_values, g_values, b_values, pixel.3);
            _mm_storeu_si128(dst.as_mut_ptr().cast(), d_lane0);
            _mm_storeu_si128(dst[8..].as_mut_ptr().cast(), d_lane1);
            _mm_storeu_si128(dst[16..].as_mut_ptr().cast(), d_lane2);
            _mm_storeu_si128(dst[24..].as_mut_ptr().cast(), d_lane3);
        }

        rem = rem.as_chunks_mut::<32>().1;
        src_rem = src_rem.as_chunks::<32>().1;

        premultiply_pixel_f16_row(rem, src_rem);
    }
}

pub(crate) fn sse_unpremultiply_alpha_rgba_f16(in_place: &mut [f16]) {
    unsafe {
        if is_x86_feature_detected!("f16c") {
            sse_unpremultiply_alpha_rgba_f16c(in_place);
        } else {
            sse_unpremultiply_alpha_rgba_f16_regular(in_place);
        }
    }
}

#[target_feature(enable = "sse4.1")]
fn sse_unpremultiply_alpha_rgba_f16_regular(in_place: &mut [f16]) {
    sse_unpremultiply_alpha_rgba_f16_row_impl::<false>(in_place);
}

#[target_feature(enable = "sse4.1", enable = "f16c")]
fn sse_unpremultiply_alpha_rgba_f16c(in_place: &mut [f16]) {
    sse_unpremultiply_alpha_rgba_f16_row_impl::<true>(in_place);
}

#[inline(always)]
fn sse_unpremultiply_alpha_rgba_f16_row_impl<const F16C: bool>(in_place: &mut [f16]) {
    unsafe {
        let mut rem = in_place;

        for dst in rem.as_chunks_mut::<32>().0.iter_mut() {
            let lane0 = _mm_loadu_si128(dst.as_ptr().cast());
            let lane1 = _mm_loadu_si128(dst[8..].as_ptr().cast());
            let lane2 = _mm_loadu_si128(dst[16..].as_ptr().cast());
            let lane3 = _mm_loadu_si128(dst[24..].as_ptr().cast());
            let pixel = sse_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let low_alpha = _mm_cvtph_psx::<F16C>(pixel.3);
            let zeros = _mm_setzero_ps();
            let low_alpha_zero_mask = _mm_cmpeq_ps(low_alpha, zeros);
            let low_r = _mm_blendv_ps(
                _mm_div_ps(_mm_cvtph_psx::<F16C>(pixel.0), low_alpha),
                zeros,
                low_alpha_zero_mask,
            );
            let low_g = _mm_blendv_ps(
                _mm_div_ps(_mm_cvtph_psx::<F16C>(pixel.1), low_alpha),
                zeros,
                low_alpha_zero_mask,
            );
            let low_b = _mm_blendv_ps(
                _mm_div_ps(_mm_cvtph_psx::<F16C>(pixel.2), low_alpha),
                zeros,
                low_alpha_zero_mask,
            );

            let high_alpha = _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.3));
            let high_alpha_zero_mask = _mm_cmpeq_ps(high_alpha, zeros);
            let high_r = _mm_blendv_ps(
                _mm_div_ps(
                    _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.0)),
                    high_alpha,
                ),
                zeros,
                high_alpha_zero_mask,
            );
            let high_g = _mm_blendv_ps(
                _mm_div_ps(
                    _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.1)),
                    high_alpha,
                ),
                zeros,
                high_alpha_zero_mask,
            );
            let high_b = _mm_blendv_ps(
                _mm_div_ps(
                    _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(pixel.2)),
                    high_alpha,
                ),
                zeros,
                high_alpha_zero_mask,
            );
            let r_values =
                _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_r), _mm_cvtps_phx::<F16C>(high_r));
            let g_values =
                _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_g), _mm_cvtps_phx::<F16C>(high_g));
            let b_values =
                _mm_unpacklo_epi64(_mm_cvtps_phx::<F16C>(low_b), _mm_cvtps_phx::<F16C>(high_b));
            let (d_lane0, d_lane1, d_lane2, d_lane3) =
                sse_interleave_rgba_epi16(r_values, g_values, b_values, pixel.3);
            _mm_storeu_si128(dst.as_mut_ptr().cast(), d_lane0);
            _mm_storeu_si128(dst[8..].as_mut_ptr().cast(), d_lane1);
            _mm_storeu_si128(dst[16..].as_mut_ptr().cast(), d_lane2);
            _mm_storeu_si128(dst[24..].as_mut_ptr().cast(), d_lane3);
        }

        rem = rem.as_chunks_mut::<32>().1;

        unpremultiply_pixel_f16_row(rem);
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::test_utils::{XorShiftRng, assert_f16_slices_close};

    // f16 has only ~10 bits of mantissa, so the f32-domain rounding
    // differences between the scalar reference and the SSE (f16<->f32
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
    fn sse_premultiply_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        for pixels in [1usize, 4, 17, 256] {
            let mut rng = XorShiftRng::new(0xA11CE ^ pixels as u64);
            let src = make_rgba(&mut rng, pixels);

            let mut dst_scalar = vec![0f16; pixels * 4];
            let mut dst_sse = vec![0f16; pixels * 4];
            premultiply_pixel_f16_row(&mut dst_scalar, &src);
            sse_premultiply_alpha_rgba_f16(&mut dst_sse, &src);

            assert_f16_slices_close(
                &dst_sse,
                &dst_scalar,
                ATOL,
                &format!("{pixels} pixels: SSE premultiply"),
            );
        }
    }

    #[test]
    fn sse_unpremultiply_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        for pixels in [1usize, 4, 17, 256] {
            let mut rng = XorShiftRng::new(0xB0BA ^ pixels as u64);
            let src = make_rgba(&mut rng, pixels);

            let mut scalar_buf = src.clone();
            let mut sse_buf = src;
            unpremultiply_pixel_f16_row(&mut scalar_buf);
            sse_unpremultiply_alpha_rgba_f16(&mut sse_buf);

            assert_f16_slices_close(
                &sse_buf,
                &scalar_buf,
                ATOL,
                &format!("{pixels} pixels: SSE unpremultiply"),
            );
        }
    }
}
