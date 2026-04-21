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

use crate::avx2::utils::shuffle;
use std::arch::x86_64::*;

pub(crate) fn avx_unpremultiply_alpha_rgba_f32(in_place: &mut [f32]) {
    unsafe {
        avx_unpremultiply_alpha_rgba_f32_row_impl(in_place);
    }
}

#[target_feature(enable = "avx2")]
fn avx_unpremultiply_alpha_rgba_f32_row_impl(in_place: &mut [f32]) {
    let mut rem = in_place;

    let copy_alpha_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, -1));
    let permute_mask = _mm256_setr_epi32(0, 4, 2, 6, 0, 4, 2, 6);
    let q0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    let q1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    let q2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
    let q3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);

    for dst in rem.as_chunks_mut::<32>().0.iter_mut() {
        let rgba0 = unsafe { _mm256_loadu_ps(dst.as_ptr().cast()) };
        let rgba1 = unsafe { _mm256_loadu_ps(dst[8..].as_ptr().cast()) };
        let rgba2 = unsafe { _mm256_loadu_ps(dst[16..].as_ptr().cast()) };
        let rgba3 = unsafe { _mm256_loadu_ps(dst[24..].as_ptr().cast()) };

        let a01 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba0, rgba0);
        let a23 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba1, rgba1);
        let a45 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba2, rgba2);
        let a67 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba3, rgba3);

        let mut a0123 = _mm256_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(a01, a23); // a0 a0 a2 a2 a1 a1 a3 a3
        a0123 = _mm256_permutevar8x32_ps(a0123, permute_mask);
        let mut a4567 = _mm256_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(a45, a67); // a0 a0 a2 a2 a1 a1 a3 a3
        a4567 = _mm256_permutevar8x32_ps(a4567, permute_mask);

        const LO_LO: i32 = 0b0010_0000;

        let a01234567 = _mm256_permute2f128_ps::<LO_LO>(a0123, a4567);

        let is_zero_mask0 = _mm256_cmp_ps::<_CMP_EQ_OS>(a01, _mm256_setzero_ps());
        let is_zero_mask1 = _mm256_cmp_ps::<_CMP_EQ_OS>(a23, _mm256_setzero_ps());
        let is_zero_mask2 = _mm256_cmp_ps::<_CMP_EQ_OS>(a45, _mm256_setzero_ps());
        let is_zero_mask3 = _mm256_cmp_ps::<_CMP_EQ_OS>(a67, _mm256_setzero_ps());

        let reciprocals = _mm256_div_ps(_mm256_set1_ps(1.), a01234567);

        let mut unscaled0 = _mm256_mul_ps(rgba0, _mm256_permutevar8x32_ps(reciprocals, q0));
        let mut unscaled1 = _mm256_mul_ps(rgba1, _mm256_permutevar8x32_ps(reciprocals, q1));
        let mut unscaled2 = _mm256_mul_ps(rgba2, _mm256_permutevar8x32_ps(reciprocals, q2));
        let mut unscaled3 = _mm256_mul_ps(rgba3, _mm256_permutevar8x32_ps(reciprocals, q3));

        unscaled0 = _mm256_blendv_ps(unscaled0, _mm256_setzero_ps(), is_zero_mask0);
        unscaled0 = _mm256_blendv_ps(unscaled0, rgba0, copy_alpha_mask);

        unscaled1 = _mm256_blendv_ps(unscaled1, _mm256_setzero_ps(), is_zero_mask1);
        unscaled1 = _mm256_blendv_ps(unscaled1, rgba1, copy_alpha_mask);

        unscaled2 = _mm256_blendv_ps(unscaled2, _mm256_setzero_ps(), is_zero_mask2);
        unscaled2 = _mm256_blendv_ps(unscaled2, rgba2, copy_alpha_mask);

        unscaled3 = _mm256_blendv_ps(unscaled3, _mm256_setzero_ps(), is_zero_mask3);
        unscaled3 = _mm256_blendv_ps(unscaled3, rgba3, copy_alpha_mask);

        unsafe {
            _mm256_storeu_ps(dst.as_mut_ptr().cast(), unscaled0);
            _mm256_storeu_ps(dst[8..].as_mut_ptr().cast(), unscaled1);
            _mm256_storeu_ps(dst[16..].as_mut_ptr().cast(), unscaled2);
            _mm256_storeu_ps(dst[24..].as_mut_ptr().cast(), unscaled3);
        }
    }

    rem = rem.as_chunks_mut::<32>().1;

    for dst in rem.as_chunks_mut::<8>().0.iter_mut() {
        let rgba = unsafe { _mm256_loadu_ps(dst.as_ptr().cast()) };

        let alphas = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba, rgba);

        let is_zero_mask = _mm256_cmp_ps::<_CMP_EQ_OS>(alphas, _mm256_setzero_ps());

        let mut unscaled = _mm256_div_ps(rgba, alphas);
        unscaled = _mm256_blendv_ps(unscaled, _mm256_setzero_ps(), is_zero_mask);
        unscaled = _mm256_blendv_ps(unscaled, rgba, copy_alpha_mask);

        unsafe {
            _mm256_storeu_ps(dst.as_mut_ptr().cast(), unscaled);
        }
    }

    rem = rem.as_chunks_mut::<8>().1;

    for dst in rem.as_chunks_mut::<4>().0.iter_mut() {
        let rgba = unsafe { _mm_loadu_ps(dst.as_ptr().cast()) };

        let alphas = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba, rgba);

        let is_zero_mask = _mm_cmp_ps::<_CMP_EQ_OS>(alphas, _mm_setzero_ps());

        let mut unscaled = _mm_div_ps(rgba, alphas);
        unscaled = _mm_blendv_ps(unscaled, _mm_setzero_ps(), is_zero_mask);
        unscaled = _mm_blendv_ps(unscaled, rgba, _mm256_castps256_ps128(copy_alpha_mask));

        unsafe {
            _mm_storeu_ps(dst.as_mut_ptr().cast(), unscaled);
        }
    }
}

pub(crate) fn avx_premultiply_alpha_rgba_f32(dst: &mut [f32], src: &[f32]) {
    unsafe {
        avx_premultiply_alpha_rgba_f32_row_impl(dst, src);
    }
}

#[target_feature(enable = "avx2")]
fn avx_premultiply_alpha_rgba_f32_row_impl(dst: &mut [f32], src: &[f32]) {
    let mut rem = dst;
    let mut src_rem = src;

    let copy_alpha_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, -1, 0, 0, 0, -1));

    for (dst, src) in rem
        .as_chunks_mut::<32>()
        .0
        .iter_mut()
        .zip(src_rem.as_chunks::<32>().0.iter())
    {
        let rgba0 = unsafe { _mm256_loadu_ps(src.as_ptr().cast()) };
        let rgba1 = unsafe { _mm256_loadu_ps(src[8..].as_ptr().cast()) };
        let rgba2 = unsafe { _mm256_loadu_ps(src[16..].as_ptr().cast()) };
        let rgba3 = unsafe { _mm256_loadu_ps(src[24..].as_ptr().cast()) };

        let alphas0 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba0, rgba0);
        let alphas1 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba1, rgba1);
        let alphas2 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba2, rgba2);
        let alphas3 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba3, rgba3);

        let mut new_px0 = _mm256_mul_ps(rgba0, alphas0);
        new_px0 = _mm256_blendv_ps(new_px0, rgba0, copy_alpha_mask);

        let mut new_px1 = _mm256_mul_ps(rgba1, alphas1);
        new_px1 = _mm256_blendv_ps(new_px1, rgba1, copy_alpha_mask);

        let mut new_px2 = _mm256_mul_ps(rgba2, alphas2);
        new_px2 = _mm256_blendv_ps(new_px2, rgba2, copy_alpha_mask);

        let mut new_px3 = _mm256_mul_ps(rgba3, alphas3);
        new_px3 = _mm256_blendv_ps(new_px3, rgba3, copy_alpha_mask);

        unsafe {
            _mm256_storeu_ps(dst.as_mut_ptr().cast(), new_px0);
            _mm256_storeu_ps(dst[8..].as_mut_ptr().cast(), new_px1);
            _mm256_storeu_ps(dst[16..].as_mut_ptr().cast(), new_px2);
            _mm256_storeu_ps(dst[24..].as_mut_ptr().cast(), new_px3);
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
        let rgba = unsafe { _mm256_loadu_ps(src.as_ptr().cast()) };
        let alphas = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba, rgba);
        let mut new_px = _mm256_mul_ps(rgba, alphas);
        new_px = _mm256_blendv_ps(new_px, rgba, copy_alpha_mask);
        unsafe {
            _mm256_storeu_ps(dst.as_mut_ptr().cast(), new_px);
        }
    }

    rem = rem.as_chunks_mut::<8>().1;
    src_rem = src_rem.as_chunks::<8>().1;

    for (dst, src) in rem
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(src_rem.as_chunks::<4>().0.iter())
    {
        let rgba = unsafe { _mm_loadu_ps(src.as_ptr().cast()) };
        let alphas = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(rgba, rgba);
        let mut new_px = _mm_mul_ps(rgba, alphas);
        new_px = _mm_blendv_ps(new_px, rgba, _mm256_castps256_ps128(copy_alpha_mask));
        unsafe {
            _mm_storeu_ps(dst.as_mut_ptr().cast(), new_px);
        }
    }
}
