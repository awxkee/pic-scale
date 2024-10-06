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

use crate::avx2::utils::{
    _mm256_select_si256, avx2_pack_u32, avx_deinterleave_rgba_epi16, avx_interleave_rgba_epi16,
};
use crate::{premultiply_pixel_u16, unpremultiply_pixel_u16, ThreadingPolicy};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
unsafe fn _mm256_scale_by_alpha(px: __m256i, low_low_a: __m256, low_high_a: __m256) -> __m256i {
    let low_px = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(px)));
    let high_px = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(px)));

    let new_ll = _mm256_cvtps_epi32(_mm256_round_ps::<0x02>(_mm256_mul_ps(low_px, low_low_a)));
    let new_lh = _mm256_cvtps_epi32(_mm256_round_ps::<0x02>(_mm256_mul_ps(high_px, low_high_a)));

    avx2_pack_u32(new_ll, new_lh)
}

pub fn avx_premultiply_alpha_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
    threading_policy: ThreadingPolicy,
) {
    unsafe {
        avx_premultiply_alpha_rgba_u16_impl(dst, src, width, height, bit_depth, threading_policy);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_premultiply_alpha_rgba_u16_row(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    offset: usize,
    bit_depth: usize,
) {
    let max_colors = (1 << bit_depth) - 1;

    let v_scale_colors = unsafe { _mm256_set1_ps((1. / max_colors as f64) as f32) };

    let mut _cx = 0usize;

    unsafe {
        while _cx + 16 < width {
            let px = _cx * 4;
            let src_ptr = src.as_ptr().add(offset + px);
            let lane0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let lane1 = _mm256_loadu_si256(src_ptr.add(16) as *const __m256i);
            let lane2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let lane3 = _mm256_loadu_si256(src_ptr.add(48) as *const __m256i);

            let pixel = avx_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let low_alpha = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(pixel.3))),
                v_scale_colors,
            );
            let high_alpha = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(
                    pixel.3,
                ))),
                v_scale_colors,
            );

            let new_rrr = _mm256_scale_by_alpha(pixel.0, low_alpha, high_alpha);
            let new_ggg = _mm256_scale_by_alpha(pixel.1, low_alpha, high_alpha);
            let new_bbb = _mm256_scale_by_alpha(pixel.2, low_alpha, high_alpha);

            let dst_ptr = dst.as_mut_ptr().add(offset + px);

            let (d_lane0, d_lane1, d_lane2, d_lane3) =
                avx_interleave_rgba_epi16(new_rrr, new_ggg, new_bbb, pixel.3);

            _mm256_storeu_si256(dst_ptr as *mut __m256i, d_lane0);
            _mm256_storeu_si256(dst_ptr.add(16) as *mut __m256i, d_lane1);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, d_lane2);
            _mm256_storeu_si256(dst_ptr.add(48) as *mut __m256i, d_lane3);
            _cx += 16;
        }
    }

    for x in _cx..width {
        let px = x * 4;
        premultiply_pixel_u16!(dst, src, offset + px, max_colors);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_premultiply_alpha_rgba_u16_impl(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
    threading_policy: ThreadingPolicy,
) {
    let allowed_threading = threading_policy.allowed_threading();
    if allowed_threading {
        src.par_chunks_exact(width * 4)
            .zip(dst.par_chunks_exact_mut(width * 4))
            .for_each(|(src, dst)| unsafe {
                avx_premultiply_alpha_rgba_u16_row(dst, src, width, 0, bit_depth);
            });
    } else {
        let mut offset = 0;

        for _ in 0..height {
            unsafe {
                avx_premultiply_alpha_rgba_u16_row(dst, src, width, offset, bit_depth);
            }

            offset += 4 * width;
        }
    }
}

pub fn avx_unpremultiply_alpha_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
    threading_policy: ThreadingPolicy,
) {
    unsafe {
        avx_unpremultiply_alpha_rgba_u16_impl(dst, src, width, height, bit_depth, threading_policy);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_u16_row(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    offset: usize,
    bit_depth: usize,
) {
    let max_colors = (1 << bit_depth) - 1;

    let v_scale_colors = unsafe { _mm256_set1_ps(max_colors as f32) };

    let mut _cx = 0usize;

    unsafe {
        while _cx + 16 < width {
            let px = _cx * 4;
            let src_ptr = src.as_ptr().add(offset + px);
            let lane0 = _mm256_loadu_si256(src_ptr as *const __m256i);
            let lane1 = _mm256_loadu_si256(src_ptr.add(16) as *const __m256i);
            let lane2 = _mm256_loadu_si256(src_ptr.add(32) as *const __m256i);
            let lane3 = _mm256_loadu_si256(src_ptr.add(48) as *const __m256i);

            let pixel = avx_deinterleave_rgba_epi16(lane0, lane1, lane2, lane3);

            let is_zero_alpha_mask = _mm256_cmpeq_epi16(pixel.3, _mm256_setzero_si256());

            let mut low_alpha = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(
                _mm256_castsi256_si128(pixel.3),
            )));

            low_alpha = _mm256_mul_ps(low_alpha, v_scale_colors);

            let mut high_alpha = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(
                _mm256_extracti128_si256::<1>(pixel.3),
            )));

            high_alpha = _mm256_mul_ps(high_alpha, v_scale_colors);

            let mut new_rrr = _mm256_scale_by_alpha(pixel.0, low_alpha, high_alpha);
            new_rrr = _mm256_select_si256(is_zero_alpha_mask, pixel.0, new_rrr);
            let mut new_ggg = _mm256_scale_by_alpha(pixel.1, low_alpha, high_alpha);
            new_ggg = _mm256_select_si256(is_zero_alpha_mask, pixel.1, new_ggg);
            let mut new_bbb = _mm256_scale_by_alpha(pixel.2, low_alpha, high_alpha);
            new_bbb = _mm256_select_si256(is_zero_alpha_mask, pixel.2, new_bbb);

            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            let (d_lane0, d_lane1, d_lane2, d_lane3) =
                avx_interleave_rgba_epi16(new_rrr, new_ggg, new_bbb, pixel.3);

            _mm256_storeu_si256(dst_ptr as *mut __m256i, d_lane0);
            _mm256_storeu_si256(dst_ptr.add(16) as *mut __m256i, d_lane1);
            _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, d_lane2);
            _mm256_storeu_si256(dst_ptr.add(48) as *mut __m256i, d_lane3);
            _cx += 16;
        }
    }

    for x in _cx..width {
        let px = x * 4;
        let pixel_offset = offset + px;
        unpremultiply_pixel_u16!(dst, src, pixel_offset, max_colors);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_u16_impl(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
    threading_policy: ThreadingPolicy,
) {
    let allowed_threading = threading_policy.allowed_threading();
    if allowed_threading {
        src.par_chunks_exact(width * 4)
            .zip(dst.par_chunks_exact_mut(width * 4))
            .for_each(|(src, dst)| unsafe {
                avx_unpremultiply_alpha_rgba_u16_row(dst, src, width, 0, bit_depth);
            });
    } else {
        let mut offset = 0usize;

        for _ in 0..height {
            unsafe {
                avx_unpremultiply_alpha_rgba_u16_row(dst, src, width, offset, bit_depth);
            }

            offset += 4 * width;
        }
    }
}
