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

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2"
))]
use crate::avx2_utils::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{neon_premultiply_alpha_rgba, neon_unpremultiply_alpha_rgba};
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_select_si128(mask: __m128i, true_vals: __m128i, false_vals: __m128i) -> __m128i {
    _mm_or_si128(
        _mm_and_si128(mask, true_vals),
        _mm_andnot_si128(mask, false_vals),
    )
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2"
))]
#[inline(always)]
pub unsafe fn avx2_unpremultiply_row(x: __m256i, a: __m256i) -> __m256i {
    let zeros = _mm256_setzero_si256();
    let lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(x));
    let hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(x));

    let scale = _mm256_set1_epi16(255);

    let is_zero_mask = _mm256_cmpeq_epi8(a, zeros);
    let a = _mm256_select_si256(is_zero_mask, scale, a);

    let scale_ps = _mm256_set1_ps(255f32);

    let lo_lo = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo))),
        scale_ps,
    );
    let lo_hi = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo))),
        scale_ps,
    );
    let hi_lo = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi))),
        scale_ps,
    );
    let hi_hi = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi))),
        scale_ps,
    );
    let a_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a));
    let a_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a));
    let a_lo_lo = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
        _mm256_castsi256_si128(a_lo),
    )));
    let a_lo_hi = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
        _mm256_extracti128_si256::<1>(a_lo),
    )));
    let a_hi_lo = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
        _mm256_castsi256_si128(a_hi),
    )));
    let a_hi_hi = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
        _mm256_extracti128_si256::<1>(a_hi),
    )));

    let lo_lo = _mm256_cvtps_epi32(_mm256_mul_ps(lo_lo, a_lo_lo));
    let lo_hi = _mm256_cvtps_epi32(_mm256_mul_ps(lo_hi, a_lo_hi));
    let hi_lo = _mm256_cvtps_epi32(_mm256_mul_ps(hi_lo, a_hi_lo));
    let hi_hi = _mm256_cvtps_epi32(_mm256_mul_ps(hi_hi, a_hi_hi));

    let lo = avx2_pack_s32(lo_lo, lo_hi);
    let hi = avx2_pack_s32(hi_lo, hi_hi);
    _mm256_select_si256(is_zero_mask, zeros, avx2_pack_u16(lo, hi))
}

#[macro_export]
macro_rules! unpremultiply_pixel {
    ($dst: expr, $src: expr, $pixel_offset: expr) => {{
        let mut r = *unsafe { $src.get_unchecked($pixel_offset) } as i32;
        let mut g = *unsafe { $src.get_unchecked($pixel_offset + 1) } as i32;
        let mut b = *unsafe { $src.get_unchecked($pixel_offset + 2) } as i32;
        let a = *unsafe { $src.get_unchecked($pixel_offset + 3) } as i32;
        if a != 0 {
            r = ((r * 255) / a).min(255).max(0);
            g = ((g * 255) / a).min(255).max(0);
            b = ((b * 255) / a).min(255).max(0);
        } else {
            r = 0;
            g = 0;
            b = 0;
        }
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = r as u8;
            *$dst.get_unchecked_mut($pixel_offset + 1) = g as u8;
            *$dst.get_unchecked_mut($pixel_offset + 2) = b as u8;
            *$dst.get_unchecked_mut($pixel_offset + 3) = a as u8;
        }
    }};
}

#[macro_export]
macro_rules! premultiply_pixel {
    ($dst: expr, $src: expr, $pixel_offset: expr) => {{
        let mut r = *unsafe { $src.get_unchecked($pixel_offset) } as i32;
        let mut g = *unsafe { $src.get_unchecked($pixel_offset + 1) } as i32;
        let mut b = *unsafe { $src.get_unchecked($pixel_offset + 2) } as i32;
        let a = *unsafe { $src.get_unchecked($pixel_offset + 3) } as i32;
        r *= a;
        g *= a;
        b *= a;
        r /= 255;
        g /= 255;
        b /= 255;
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = r as u8;
            *$dst.get_unchecked_mut($pixel_offset + 1) = g as u8;
            *$dst.get_unchecked_mut($pixel_offset + 2) = b as u8;
            *$dst.get_unchecked_mut($pixel_offset + 3) = a as u8;
        }
    }};
}

fn premultiply_alpha_rgba_impl(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
    let mut offset = 0usize;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _has_avx2 = false;

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx2"
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            _has_avx2 = true;
        }
    }

    for _ in 0..height {
        let mut _cx = 0usize;

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "avx2"
        ))]
        if _has_avx2 {
            unsafe {
                while _cx + 32 < width {
                    let px = _cx * 4;
                    let src_ptr = src.as_ptr().add(offset + px);
                    let rgba0 = _mm256_loadu_si256(src_ptr as *const __m256i);
                    let rgba1 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
                    let rgba2 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
                    let rgba3 = _mm256_loadu_si256(src_ptr.add(96) as *const __m256i);
                    let (rrr, ggg, bbb, aaa) = avx2_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

                    let mut rrr_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rrr));
                    let mut rrr_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(rrr));

                    let mut ggg_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ggg));
                    let mut ggg_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(ggg));

                    let mut bbb_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(bbb));
                    let mut bbb_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(bbb));

                    let aaa_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(aaa));
                    let aaa_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(aaa));

                    rrr_low = avx2_div_by255(_mm256_mullo_epi16(rrr_low, aaa_low));
                    rrr_high = avx2_div_by255(_mm256_mullo_epi16(rrr_high, aaa_high));
                    ggg_low = avx2_div_by255(_mm256_mullo_epi16(ggg_low, aaa_low));
                    ggg_high = avx2_div_by255(_mm256_mullo_epi16(ggg_high, aaa_high));
                    bbb_low = avx2_div_by255(_mm256_mullo_epi16(bbb_low, aaa_low));
                    bbb_high = avx2_div_by255(_mm256_mullo_epi16(bbb_high, aaa_high));

                    let rrr = avx2_pack_u16(rrr_low, rrr_high);
                    let ggg = avx2_pack_u16(ggg_low, ggg_high);
                    let bbb = avx2_pack_u16(bbb_low, bbb_high);

                    let (rgba0, rgba1, rgba2, rgba3) = avx2_interleave_rgba(rrr, ggg, bbb, aaa);
                    let dst_ptr = dst.as_mut_ptr().add(offset + px);
                    _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba0);
                    _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, rgba1);
                    _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, rgba2);
                    _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, rgba3);

                    _cx += 32;
                }
            }
        }

        for x in _cx..width {
            let px = x * 4;
            premultiply_pixel!(dst, src, offset + px);
        }

        offset += 4 * width;
    }
}

fn unpremultiply_alpha_rgba_impl(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
    let mut offset = 0usize;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _has_avx2 = false;

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx2"
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            _has_avx2 = true;
        }
    }

    for _ in 0..height {
        let mut _cx = 0usize;

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "avx2"
        ))]
        if _has_avx2 {
            unsafe {
                while _cx + 32 < width {
                    let px = _cx * 4;
                    let pixel_offset = offset + px;
                    let src_ptr = src.as_ptr().add(pixel_offset);
                    let rgba0 = _mm256_loadu_si256(src_ptr as *const __m256i);
                    let rgba1 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
                    let rgba2 = _mm256_loadu_si256(src_ptr.add(64) as *const __m256i);
                    let rgba3 = _mm256_loadu_si256(src_ptr.add(96) as *const __m256i);
                    let (rrr, ggg, bbb, aaa) = avx2_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

                    let rrr = avx2_unpremultiply_row(rrr, aaa);
                    let ggg = avx2_unpremultiply_row(ggg, aaa);
                    let bbb = avx2_unpremultiply_row(bbb, aaa);

                    let (rgba0, rgba1, rgba2, rgba3) = avx2_interleave_rgba(rrr, ggg, bbb, aaa);

                    let dst_ptr = dst.as_mut_ptr().add(pixel_offset);
                    _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba0);
                    _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, rgba1);
                    _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, rgba2);
                    _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, rgba3);

                    _cx += 32;
                }
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel!(dst, src, pixel_offset);
        }

        offset += 4 * width;
    }
}

pub fn premultiply_alpha_rgba(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
    let mut _dispatcher: fn(&mut [u8], &[u8], usize, usize) = premultiply_alpha_rgba_impl;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_premultiply_alpha_rgba;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = sse_premultiply_alpha_rgba;
    }
    _dispatcher(dst, src, width, height);
}

pub fn unpremultiply_alpha_rgba(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
    let mut _dispatcher: fn(&mut [u8], &[u8], usize, usize) = unpremultiply_alpha_rgba_impl;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_unpremultiply_alpha_rgba;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        _dispatcher = sse_unpremultiply_alpha_rgba;
    }
    _dispatcher(dst, src, width, height);
}
