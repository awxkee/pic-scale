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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::avx2::utils::{avx_deinterleave_rgba_ps, avx_interleave_rgba_ps};
use crate::{premultiply_pixel_f32, unpremultiply_pixel_f32};

#[inline(always)]
pub unsafe fn avx_unpremultiply_row_f32(x: __m256, a: __m256) -> __m256 {
    let is_zero_mask = _mm256_cmp_ps::<_CMP_EQ_OS>(a, _mm256_setzero_ps());
    let rs = _mm256_div_ps(x, a);
    _mm256_blendv_ps(rs, _mm256_setzero_ps(), is_zero_mask)
}

pub fn avx_unpremultiply_alpha_rgba_f32(dst: &mut [f32], src: &[f32], width: usize, height: usize) {
    unsafe {
        avx_unpremultiply_alpha_rgba_f32_impl(dst, src, width, height);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx_unpremultiply_alpha_rgba_f32_impl(
    dst: &mut [f32],
    src: &[f32],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;

    let mut offset = 0usize;
    offset += _cy * width * 4;

    for _ in _cy..height {
        let mut _cx = 0usize;

        while _cx + 8 < width {
            let px = _cx * 4;
            let pixel_offset = offset + px;
            let src_ptr = src.as_ptr().add(pixel_offset);
            let rgba0 = _mm256_loadu_ps(src_ptr);
            let rgba1 = _mm256_loadu_ps(src_ptr.add(8));
            let rgba2 = _mm256_loadu_ps(src_ptr.add(16));
            let rgba3 = _mm256_loadu_ps(src_ptr.add(24));

            let (rrr, ggg, bbb, aaa) = avx_deinterleave_rgba_ps(rgba0, rgba1, rgba2, rgba3);

            let rrr = avx_unpremultiply_row_f32(rrr, aaa);
            let ggg = avx_unpremultiply_row_f32(ggg, aaa);
            let bbb = avx_unpremultiply_row_f32(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) = avx_interleave_rgba_ps(rrr, ggg, bbb, aaa);

            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            _mm256_storeu_ps(dst_ptr, rgba0);
            _mm256_storeu_ps(dst_ptr.add(8), rgba1);
            _mm256_storeu_ps(dst_ptr.add(16), rgba2);
            _mm256_storeu_ps(dst_ptr.add(24), rgba3);

            _cx += 8;
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel_f32!(dst, src, pixel_offset);
        }

        offset += 4 * width;
    }
}

pub fn avx_premultiply_alpha_rgba_f32(dst: &mut [f32], src: &[f32], width: usize, height: usize) {
    unsafe {
        avx_premultiply_alpha_rgba_f32_impl(dst, src, width, height);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx_premultiply_alpha_rgba_f32_impl(
    dst: &mut [f32],
    src: &[f32],
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
                let pixel_offset = offset + px;
                let src_ptr = src.as_ptr().add(pixel_offset);
                let rgba0 = _mm256_loadu_ps(src_ptr);
                let rgba1 = _mm256_loadu_ps(src_ptr.add(8));
                let rgba2 = _mm256_loadu_ps(src_ptr.add(16));
                let rgba3 = _mm256_loadu_ps(src_ptr.add(24));
                let (rrr, ggg, bbb, aaa) = avx_deinterleave_rgba_ps(rgba0, rgba1, rgba2, rgba3);

                let rrr = _mm256_mul_ps(rrr, aaa);
                let ggg = _mm256_mul_ps(ggg, aaa);
                let bbb = _mm256_mul_ps(bbb, aaa);

                let (rgba0, rgba1, rgba2, rgba3) = avx_interleave_rgba_ps(rrr, ggg, bbb, aaa);

                let dst_ptr = dst.as_mut_ptr().add(offset + px);
                _mm256_storeu_ps(dst_ptr, rgba0);
                _mm256_storeu_ps(dst_ptr.add(8), rgba1);
                _mm256_storeu_ps(dst_ptr.add(16), rgba2);
                _mm256_storeu_ps(dst_ptr.add(24), rgba3);

                _cx += 8;
            }
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            premultiply_pixel_f32!(dst, src, pixel_offset);
        }

        offset += 4 * width;
    }
}
