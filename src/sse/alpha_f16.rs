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

use crate::sse::{sse_deinterleave_rgba_epi16, sse_interleave_rgba_epi16};
use crate::{premultiply_pixel_f16, unpremultiply_pixel_f16};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn sse_premultiply_alpha_rgba_f16(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;
    let src_stride = 4 * width;

    let mut offset = _cy * src_stride;

    for _ in _cy..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 8 < width {
                let px = _cx * 4;
                let src_ptr = src.as_ptr().add(offset + px);
                let lane0 = _mm_loadu_si128(src_ptr as *const __m128i);
                let lane1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
                let lane2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
                let lane3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
                let pixel = sse_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

                let low_alpha = _mm_cvtph_ps(pixel.3);
                let low_r = _mm_mul_ps(_mm_cvtph_ps(pixel.0), low_alpha);
                let low_g = _mm_mul_ps(_mm_cvtph_ps(pixel.1), low_alpha);
                let low_b = _mm_mul_ps(_mm_cvtph_ps(pixel.2), low_alpha);

                let high_alpha = _mm_cvtph_ps(_mm_srli_si128::<8>(pixel.3));
                let high_r = _mm_mul_ps(_mm_cvtph_ps(_mm_srli_si128::<8>(pixel.0)), high_alpha);
                let high_g = _mm_mul_ps(_mm_cvtph_ps(_mm_srli_si128::<8>(pixel.1)), high_alpha);
                let high_b = _mm_mul_ps(_mm_cvtph_ps(_mm_srli_si128::<8>(pixel.2)), high_alpha);
                let r_values = _mm_unpacklo_epi64(
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(low_r),
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(high_r),
                );
                let g_values = _mm_unpacklo_epi64(
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(low_g),
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(high_g),
                );
                let b_values = _mm_unpacklo_epi64(
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(low_b),
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(high_b),
                );
                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                let (d_lane0, d_lane1, d_lane2, d_lane3) =
                    sse_interleave_rgba_epi16(r_values, g_values, b_values, pixel.3);
                _mm_storeu_si128(dst_ptr as *mut __m128i, d_lane0);
                _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, d_lane1);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, d_lane2);
                _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, d_lane3);
                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            premultiply_pixel_f16!(dst, src, offset + px);
        }

        offset += 4 * width;
    }
}

pub fn sse_unpremultiply_alpha_rgba_f16(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;

    let mut offset = 0usize;
    offset += _cy * width * 4;

    for _ in _cy..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 8 < width {
                let px = _cx * 4;
                let src_ptr = src.as_ptr().add(offset + px);
                let lane0 = _mm_loadu_si128(src_ptr as *const __m128i);
                let lane1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
                let lane2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
                let lane3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
                let pixel = sse_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

                let low_alpha = _mm_cvtph_ps(pixel.3);
                let zeros = _mm_setzero_ps();
                let low_alpha_zero_mask = _mm_cmpeq_ps(low_alpha, zeros);
                let low_r = _mm_blendv_ps(
                    _mm_mul_ps(_mm_cvtph_ps(pixel.0), low_alpha),
                    zeros,
                    low_alpha_zero_mask,
                );
                let low_g = _mm_blendv_ps(
                    _mm_mul_ps(_mm_cvtph_ps(pixel.1), low_alpha),
                    zeros,
                    low_alpha_zero_mask,
                );
                let low_b = _mm_blendv_ps(
                    _mm_mul_ps(_mm_cvtph_ps(pixel.2), low_alpha),
                    zeros,
                    low_alpha_zero_mask,
                );

                let high_alpha = _mm_cvtph_ps(_mm_srli_si128::<8>(pixel.3));
                let high_alpha_zero_mask = _mm_cmpeq_ps(high_alpha, zeros);
                let high_r = _mm_blendv_ps(
                    _mm_mul_ps(_mm_cvtph_ps(_mm_srli_si128::<8>(pixel.0)), high_alpha),
                    zeros,
                    high_alpha_zero_mask,
                );
                let high_g = _mm_blendv_ps(
                    _mm_mul_ps(_mm_cvtph_ps(_mm_srli_si128::<8>(pixel.1)), high_alpha),
                    zeros,
                    high_alpha_zero_mask,
                );
                let high_b = _mm_blendv_ps(
                    _mm_mul_ps(_mm_cvtph_ps(_mm_srli_si128::<8>(pixel.2)), high_alpha),
                    zeros,
                    high_alpha_zero_mask,
                );
                let r_values = _mm_unpacklo_epi64(
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(low_r),
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(high_r),
                );
                let g_values = _mm_unpacklo_epi64(
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(low_g),
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(high_g),
                );
                let b_values = _mm_unpacklo_epi64(
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(low_b),
                    _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(high_b),
                );
                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                let (d_lane0, d_lane1, d_lane2, d_lane3) =
                    sse_interleave_rgba_epi16(r_values, g_values, b_values, pixel.3);
                _mm_storeu_si128(dst_ptr as *mut __m128i, d_lane0);
                _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, d_lane1);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, d_lane2);
                _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, d_lane3);
                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel_f16!(dst, src, pixel_offset);
        }

        offset += 4 * width;
    }
}
