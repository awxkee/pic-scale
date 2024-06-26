/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2"
))]
use crate::avx2_utils::*;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub unsafe fn neon_div_by_255(v: uint16x8_t) -> uint16x8_t {
    let rounding = vdupq_n_u16(1 << 7);
    let x = vqaddq_u16(v, rounding);
    let multiplier = vdupq_n_u16(0x8080);
    let hi = vmull_high_u16(x, multiplier);
    let lo = vmull_u16(vget_low_u16(x), vget_low_u16(multiplier));

    let hi_16 = vqshrn_n_u32::<7>(vshrq_n_u32::<16>(hi));
    let lo_16 = vqshrn_n_u32::<7>(vshrq_n_u32::<16>(lo));
    vcombine_u16(lo_16, hi_16)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub unsafe fn neon_premultiply_alpha(v: uint8x16_t, a_values: uint8x16_t) -> uint8x16_t {
    let acc_hi = vmull_high_u8(v, a_values);
    let acc_lo = vmull_u8(vget_low_u8(v), vget_low_u8(a_values));
    let hi = vqmovn_u16(neon_div_by_255(acc_hi));
    let lo = vqmovn_u16(neon_div_by_255(acc_lo));
    vcombine_u8(lo, hi)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub unsafe fn neon_umpremultiply_alpha(v: uint8x16_t, a_values: uint8x16_t) -> uint8x16_t {
    let scale = vdupq_n_u8(255);
    let hi = vmull_high_u8(v, scale);
    let lo = vmull_u8(vget_low_u8(v), vget_low_u8(scale));
    let lo_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
    let lo_hi = vcvtq_f32_u32(vmovl_high_u16(lo));
    let hi_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
    let hi_hi = vcvtq_f32_u32(vmovl_high_u16(hi));
    let zero_mask = vceqzq_u8(a_values);
    let a_values = vbslq_u8(zero_mask, scale, a_values);
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
        zero_mask,
        vdupq_n_u8(0),
        vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi)),
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_div_by255(v: __m128i) -> __m128i {
    let rounding = _mm_set1_epi16(1 << 7);
    let x = _mm_adds_epi16(v, rounding);
    let multiplier = _mm_set1_epi16(-32640);
    let r = _mm_mulhi_epu16(x, multiplier);
    return _mm_srli_epi16::<7>(r);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn _mm_select_si128(mask: __m128i, true_vals: __m128i, false_vals: __m128i) -> __m128i {
    _mm_or_si128(
        _mm_and_si128(mask, true_vals),
        _mm_andnot_si128(mask, false_vals),
    )
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_unpremultiply_row(x: __m128i, a: __m128i) -> __m128i {
    let zeros = _mm_setzero_si128();
    let lo = _mm_cvtepu8_epi16(x);
    let hi = _mm_unpackhi_epi8(x, zeros);

    let scale = _mm_set1_epi16(255);

    let is_zero_mask = _mm_cmpeq_epi8(a, zeros);
    let a = _mm_select_si128(is_zero_mask, scale, a);

    let scale_ps = _mm_set1_ps(255f32);

    let lo_lo = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(lo)), scale_ps);
    let lo_hi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(lo, zeros)), scale_ps);
    let hi_lo = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(hi)), scale_ps);
    let hi_hi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(hi, zeros)), scale_ps);
    let a_lo = _mm_cvtepu8_epi16(a);
    let a_hi = _mm_unpackhi_epi8(a, zeros);
    let a_lo_lo = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_lo)));
    let a_lo_hi = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a_lo, zeros)));
    let a_hi_lo = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(a_hi)));
    let a_hi_hi = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(a_hi, zeros)));

    let lo_lo = _mm_cvtps_epi32(_mm_mul_ps(lo_lo, a_lo_lo));
    let lo_hi = _mm_cvtps_epi32(_mm_mul_ps(lo_hi, a_lo_hi));
    let hi_lo = _mm_cvtps_epi32(_mm_mul_ps(hi_lo, a_hi_lo));
    let hi_hi = _mm_cvtps_epi32(_mm_mul_ps(hi_hi, a_hi_hi));

    let lo = _mm_packs_epi32(lo_lo, lo_hi);
    let hi = _mm_packs_epi32(hi_lo, hi_hi);
    _mm_select_si128(is_zero_mask, _mm_setzero_si128(), _mm_packus_epi16(lo, hi))
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

pub fn premultiply_alpha_rgba(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
    let mut offset = 0usize;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _has_sse = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _has_avx2 = false;

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        if is_x86_feature_detected!("sse4.1") {
            _has_sse = true;
        }
    }

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

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        if _has_sse {
            unsafe {
                let zeros = _mm_setzero_si128();
                while _cx + 16 < width {
                    let px = _cx * 4;
                    let src_ptr = src.as_ptr().add(offset + px);
                    let rgba0 = _mm_loadu_si128(src_ptr as *const __m128i);
                    let rgba1 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
                    let rgba2 = _mm_loadu_si128(src_ptr.add(32) as *const __m128i);
                    let rgba3 = _mm_loadu_si128(src_ptr.add(48) as *const __m128i);
                    let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

                    let mut rrr_low = _mm_cvtepu8_epi16(rrr);
                    let mut rrr_high = _mm_unpackhi_epi8(rrr, zeros);

                    let mut ggg_low = _mm_cvtepu8_epi16(ggg);
                    let mut ggg_high = _mm_unpackhi_epi8(ggg, zeros);

                    let mut bbb_low = _mm_cvtepu8_epi16(bbb);
                    let mut bbb_high = _mm_unpackhi_epi8(bbb, zeros);

                    let aaa_low = _mm_cvtepu8_epi16(aaa);
                    let aaa_high = _mm_unpackhi_epi8(aaa, zeros);

                    rrr_low = sse_div_by255(_mm_mullo_epi16(rrr_low, aaa_low));
                    rrr_high = sse_div_by255(_mm_mullo_epi16(rrr_high, aaa_high));
                    ggg_low = sse_div_by255(_mm_mullo_epi16(ggg_low, aaa_low));
                    ggg_high = sse_div_by255(_mm_mullo_epi16(ggg_high, aaa_high));
                    bbb_low = sse_div_by255(_mm_mullo_epi16(bbb_low, aaa_low));
                    bbb_high = sse_div_by255(_mm_mullo_epi16(bbb_high, aaa_high));

                    let rrr = _mm_packus_epi16(rrr_low, rrr_high);
                    let ggg = _mm_packus_epi16(ggg_low, ggg_high);
                    let bbb = _mm_packus_epi16(bbb_low, bbb_high);

                    let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba(rrr, ggg, bbb, aaa);

                    let dst_ptr = dst.as_mut_ptr().add(offset + px);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
                    _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);

                    _cx += 16;
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            while _cx + 16 < width {
                let px = _cx * 4;
                let src_ptr = src.as_ptr().add(offset + px);
                let mut pixel = vld4q_u8(src_ptr);
                pixel.0 = neon_premultiply_alpha(pixel.0, pixel.3);
                pixel.1 = neon_premultiply_alpha(pixel.1, pixel.3);
                pixel.2 = neon_premultiply_alpha(pixel.2, pixel.3);
                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                vst4q_u8(dst_ptr, pixel);
                _cx += 16;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let mut r = *unsafe { src.get_unchecked(offset + px) } as i32;
            let mut g = *unsafe { src.get_unchecked(offset + px + 1) } as i32;
            let mut b = *unsafe { src.get_unchecked(offset + px + 2) } as i32;
            let a = *unsafe { src.get_unchecked(offset + px + 3) } as i32;
            if a != 0 {
                r *= a;
                g *= a;
                b *= a;
                r /= 255;
                g /= 255;
                b /= 255;
            } else {
                r = 0;
                g = 0;
                b = 0;
            }
            unsafe {
                *dst.get_unchecked_mut(offset + px) = r as u8;
                *dst.get_unchecked_mut(offset + px + 1) = g as u8;
                *dst.get_unchecked_mut(offset + px + 2) = b as u8;
                *dst.get_unchecked_mut(offset + px + 3) = a as u8;
            }
        }

        offset += 4 * width;
    }
}

pub fn unpremultiply_alpha_rgba(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
    let mut offset = 0usize;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _has_sse = false;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _has_avx2 = false;

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "sse4.1"
    ))]
    {
        if is_x86_feature_detected!("sse4.1") {
            _has_sse = true;
        }
    }

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

        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        if _has_sse {
            unsafe {
                while _cx + 16 < width {
                    let px = _cx * 4;
                    let pixel_offset = offset + px;
                    let src_ptr = src.as_ptr().add(pixel_offset);
                    let rgba0 = _mm_loadu_si128(src_ptr as *const __m128i);
                    let rgba1 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
                    let rgba2 = _mm_loadu_si128(src_ptr.add(32) as *const __m128i);
                    let rgba3 = _mm_loadu_si128(src_ptr.add(48) as *const __m128i);
                    let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba(rgba0, rgba1, rgba2, rgba3);

                    let rrr = sse_unpremultiply_row(rrr, aaa);
                    let ggg = sse_unpremultiply_row(ggg, aaa);
                    let bbb = sse_unpremultiply_row(bbb, aaa);

                    let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba(rrr, ggg, bbb, aaa);

                    let dst_ptr = dst.as_mut_ptr().add(offset + px);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, rgba2);
                    _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, rgba3);

                    _cx += 16;
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            while _cx + 16 < width {
                let px = _cx * 4;
                let pixel_offset = offset + px;
                let src_ptr = src.as_ptr().add(pixel_offset);
                let mut pixel = vld4q_u8(src_ptr);
                pixel.0 = neon_umpremultiply_alpha(pixel.0, pixel.3);
                pixel.1 = neon_umpremultiply_alpha(pixel.1, pixel.3);
                pixel.2 = neon_umpremultiply_alpha(pixel.2, pixel.3);
                let dst_ptr = dst.as_mut_ptr().add(pixel_offset);
                vst4q_u8(dst_ptr, pixel);
                _cx += 16;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            let mut r = *unsafe { src.get_unchecked(pixel_offset) } as i32;
            let mut g = *unsafe { src.get_unchecked(pixel_offset + 1) } as i32;
            let mut b = *unsafe { src.get_unchecked(pixel_offset + 2) } as i32;
            let a = *unsafe { src.get_unchecked(pixel_offset + 3) } as i32;
            if a != 0 {
                r *= 255;
                g *= 255;
                b *= 255;
                r /= a;
                g /= a;
                b /= a;
            } else {
                r = 0;
                g = 0;
                b = 0;
            }
            unsafe {
                *dst.get_unchecked_mut(pixel_offset) = r as u8;
                *dst.get_unchecked_mut(pixel_offset + 1) = g as u8;
                *dst.get_unchecked_mut(pixel_offset + 2) = b as u8;
                *dst.get_unchecked_mut(pixel_offset + 3) = a as u8;
            }
        }

        offset += 4 * width;
    }
}
