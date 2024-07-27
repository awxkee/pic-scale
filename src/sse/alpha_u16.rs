use crate::{premultiply_pixel_u16, unpremultiply_pixel_u16};

use crate::sse::alpha_u8::_mm_select_si128;
use crate::sse::{sse_deinterleave_rgba_epi16, sse_interleave_rgba_epi16};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn sse_unpremultiply_row_u16(
    x: __m128i,
    is_zero_mask: __m128i,
    a_lo_f: __m128,
    a_hi_f: __m128,
    max_colors: __m128,
) -> __m128i {
    let zeros = _mm_setzero_si128();
    let lo = _mm_cvtepu16_epi32(x);
    let hi = _mm_unpackhi_epi16(x, zeros);

    let new_lo = _mm_cvtps_epi32(_mm_mul_ps(
        _mm_mul_ps(_mm_cvtepi32_ps(lo), a_lo_f),
        max_colors,
    ));
    let new_hi = _mm_cvtps_epi32(_mm_mul_ps(
        _mm_mul_ps(_mm_cvtepi32_ps(hi), a_hi_f),
        max_colors,
    ));

    let pixel = _mm_packs_epi32(new_lo, new_hi);
    _mm_select_si128(is_zero_mask, x, pixel)
}

pub fn unpremultiply_alpha_sse_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
) {
    let mut offset = 0usize;

    let max_colors = 2i64.pow(bit_depth as u32) - 1;

    let v_max_colors = unsafe { _mm_set1_ps(max_colors as f32) };

    for _ in 0..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 8 < width {
                let px = _cx * 4;
                let pixel_offset = offset + px;
                let src_ptr = src.as_ptr().add(pixel_offset);
                let row0 = _mm_loadu_si128(src_ptr as *const __m128i);
                let row1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
                let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
                let row3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
                let (rrrr, gggg, bbbb, aaaa) = sse_deinterleave_rgba_epi16(row0, row1, row2, row3);

                let is_zero_mask = _mm_cmpeq_epi16(aaaa, _mm_setzero_si128());
                let a_lo_f = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_cvtepu16_epi32(aaaa)));
                let a_hi_f = _mm_rcp_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(
                    aaaa,
                    _mm_setzero_si128(),
                )));

                let new_rrrr =
                    sse_unpremultiply_row_u16(rrrr, is_zero_mask, a_lo_f, a_hi_f, v_max_colors);
                let new_gggg =
                    sse_unpremultiply_row_u16(gggg, is_zero_mask, a_lo_f, a_hi_f, v_max_colors);
                let new_bbbb =
                    sse_unpremultiply_row_u16(bbbb, is_zero_mask, a_lo_f, a_hi_f, v_max_colors);

                let (rgba0, rgba1, rgba2, rgba3) =
                    sse_interleave_rgba_epi16(new_rrrr, new_gggg, new_bbbb, aaaa);

                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
                _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, rgba1);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba2);
                _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, rgba3);

                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel_u16!(dst, src, pixel_offset, max_colors);
        }

        offset += 4 * width;
    }
}

#[inline(always)]
pub unsafe fn sse_premultiply_row_u16(
    x: __m128i,
    a_lo_f: __m128,
    a_hi_f: __m128,
    v_max_colors_scale: __m128,
) -> __m128i {
    let zeros = _mm_setzero_si128();
    let lo = _mm_cvtepu16_epi32(x);
    let hi = _mm_unpackhi_epi16(x, zeros);

    let new_lo = _mm_cvtps_epi32(_mm_mul_ps(
        _mm_mul_ps(_mm_cvtepi32_ps(lo), v_max_colors_scale),
        a_lo_f,
    ));
    let new_hi = _mm_cvtps_epi32(_mm_mul_ps(
        _mm_mul_ps(_mm_cvtepi32_ps(hi), v_max_colors_scale),
        a_hi_f,
    ));

    _mm_packs_epi32(new_lo, new_hi)
}

pub fn premultiply_alpha_sse_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
) {
    let mut offset = 0usize;

    let max_colors = 2i64.pow(bit_depth as u32) - 1;

    let v_max_colors_scale = unsafe {
        _mm_div_ps(
            _mm_set1_ps(1.),
            _mm_cvtepi32_ps(_mm_set1_epi32(max_colors as i32)),
        )
    };

    for _ in 0..height {
        let mut _cx = 0usize;

        unsafe {
            while _cx + 8 < width {
                let px = _cx * 4;
                let pixel_offset = offset + px;
                let src_ptr = src.as_ptr().add(pixel_offset);
                let row0 = _mm_loadu_si128(src_ptr as *const __m128i);
                let row1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
                let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
                let row3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
                let (rrrr, gggg, bbbb, aaaa) = sse_deinterleave_rgba_epi16(row0, row1, row2, row3);

                let a_lo_f = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(aaaa));
                let a_hi_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(aaaa, _mm_setzero_si128()));

                let new_rrrr = sse_premultiply_row_u16(rrrr, a_lo_f, a_hi_f, v_max_colors_scale);
                let new_gggg = sse_premultiply_row_u16(gggg, a_lo_f, a_hi_f, v_max_colors_scale);
                let new_bbbb = sse_premultiply_row_u16(bbbb, a_lo_f, a_hi_f, v_max_colors_scale);

                let (rgba0, rgba1, rgba2, rgba3) =
                    sse_interleave_rgba_epi16(new_rrrr, new_gggg, new_bbbb, aaaa);

                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
                _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, rgba1);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba2);
                _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, rgba3);

                _cx += 8;
            }
        }

        for x in 0..width {
            let px = x * 4;
            premultiply_pixel_u16!(dst, src, offset + px, max_colors);
        }

        offset += 4 * width;
    }
}
