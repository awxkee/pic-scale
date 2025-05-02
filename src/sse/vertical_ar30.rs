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
use crate::filter_weights::FilterBounds;
use crate::sse::{_mm_unzip_3_ar30, _mm_zip_4_ar30};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) fn sse_column_handler_fixed_point_ar30<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
>(
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    unsafe {
        sse_column_handler_fixed_point_ar30_impl::<AR30_TYPE, AR30_ORDER>(
            bounds, src, dst, src_stride, weight,
        );
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_column_handler_fixed_point_ar30_impl<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
>(
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    unsafe {
        let mut cx = 0usize;

        let total_width = dst.len() / 4;

        const PREC: i32 = 15;
        const RND_CONST: i32 = (1 << (PREC - 1)) - 1;

        while cx + 8 < total_width {
            let v_max = _mm_set1_epi16(1023);
            let filter = weight;
            let v_start_px = cx * 4;

            let mut v0 = _mm_set1_epi32(RND_CONST);
            let mut v1 = _mm_set1_epi32(RND_CONST);
            let mut v2 = _mm_set1_epi32(RND_CONST);
            let mut v3 = _mm_set1_epi32(RND_CONST);
            let mut v4 = _mm_set1_epi32(RND_CONST);
            let mut v5 = _mm_set1_epi32(RND_CONST);

            for (j, &k_weight) in filter.iter().take(bounds.size).enumerate() {
                let py = bounds.start + j;
                let weight = _mm_set1_epi16(k_weight);
                let offset = src_stride * py + v_start_px;
                let src_ptr = src.get_unchecked(offset..(offset + 8 * 4));

                let l0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
                let l1 = _mm_loadu_si128(src_ptr.as_ptr().add(4 * 4) as *const __m128i);

                let ps = _mm_unzip_3_ar30::<AR30_TYPE, AR30_ORDER>((l0, l1));
                v0 = _mm_add_epi32(
                    v0,
                    _mm_madd_epi16(_mm_unpacklo_epi16(ps.0, _mm_setzero_si128()), weight),
                );
                v1 = _mm_add_epi32(
                    v1,
                    _mm_madd_epi16(_mm_unpackhi_epi16(ps.0, _mm_setzero_si128()), weight),
                );
                v2 = _mm_add_epi32(
                    v2,
                    _mm_madd_epi16(_mm_unpacklo_epi16(ps.1, _mm_setzero_si128()), weight),
                );
                v3 = _mm_add_epi32(
                    v3,
                    _mm_madd_epi16(_mm_unpackhi_epi16(ps.1, _mm_setzero_si128()), weight),
                );
                v4 = _mm_add_epi32(
                    v4,
                    _mm_madd_epi16(_mm_unpacklo_epi16(ps.2, _mm_setzero_si128()), weight),
                );
                v5 = _mm_add_epi32(
                    v5,
                    _mm_madd_epi16(_mm_unpackhi_epi16(ps.2, _mm_setzero_si128()), weight),
                );
            }

            let v0 = _mm_srai_epi32::<PREC>(v0);
            let v1 = _mm_srai_epi32::<PREC>(v1);
            let v2 = _mm_srai_epi32::<PREC>(v2);
            let v3 = _mm_srai_epi32::<PREC>(v3);
            let v4 = _mm_srai_epi32::<PREC>(v4);
            let v5 = _mm_srai_epi32::<PREC>(v5);

            let r_v = _mm_min_epi16(_mm_packus_epi32(v0, v1), v_max);
            let g_v = _mm_min_epi16(_mm_packus_epi32(v2, v3), v_max);
            let b_v = _mm_min_epi16(_mm_packus_epi32(v4, v5), v_max);

            let v_dst = dst.get_unchecked_mut(v_start_px..(v_start_px + 8 * 4));

            let vals = _mm_zip_4_ar30::<AR30_TYPE, AR30_ORDER>((r_v, g_v, b_v, _mm_set1_epi16(3)));
            _mm_storeu_si128(v_dst.as_mut_ptr() as *mut _, vals.0);
            _mm_storeu_si128(v_dst.as_mut_ptr().add(4 * 4) as *mut _, vals.1);

            cx += 8;
        }

        if cx < total_width {
            let diff = total_width - cx;

            let mut src_transient: [u8; 4 * 8] = [0; 4 * 8];
            let mut dst_transient: [u8; 4 * 8] = [0; 4 * 8];

            let v_max = _mm_set1_epi16(1023);
            let filter = weight;
            let v_start_px = cx * 4;

            let mut v0 = _mm_set1_epi32(RND_CONST);
            let mut v1 = _mm_set1_epi32(RND_CONST);
            let mut v2 = _mm_set1_epi32(RND_CONST);
            let mut v3 = _mm_set1_epi32(RND_CONST);
            let mut v4 = _mm_set1_epi32(RND_CONST);
            let mut v5 = _mm_set1_epi32(RND_CONST);

            for (j, &k_weight) in filter.iter().take(bounds.size).enumerate() {
                let py = bounds.start + j;
                let weight = _mm_set1_epi16(k_weight);
                let offset = src_stride * py + v_start_px;
                let src_ptr = src.get_unchecked(offset..(offset + diff * 4));

                std::ptr::copy_nonoverlapping(
                    src_ptr.as_ptr(),
                    src_transient.as_mut_ptr(),
                    diff * 4,
                );

                let l0 = _mm_loadu_si128(src_transient.as_ptr() as *const __m128i);
                let l1 = _mm_loadu_si128(src_transient.as_ptr().add(4 * 4) as *const __m128i);

                let ps = _mm_unzip_3_ar30::<AR30_TYPE, AR30_ORDER>((l0, l1));
                v0 = _mm_add_epi32(
                    v0,
                    _mm_madd_epi16(_mm_unpacklo_epi16(ps.0, _mm_setzero_si128()), weight),
                );
                v1 = _mm_add_epi32(
                    v1,
                    _mm_madd_epi16(_mm_unpackhi_epi16(ps.0, _mm_setzero_si128()), weight),
                );
                v2 = _mm_add_epi32(
                    v2,
                    _mm_madd_epi16(_mm_unpacklo_epi16(ps.1, _mm_setzero_si128()), weight),
                );
                v3 = _mm_add_epi32(
                    v3,
                    _mm_madd_epi16(_mm_unpackhi_epi16(ps.1, _mm_setzero_si128()), weight),
                );
                v4 = _mm_add_epi32(
                    v4,
                    _mm_madd_epi16(_mm_unpacklo_epi16(ps.2, _mm_setzero_si128()), weight),
                );
                v5 = _mm_add_epi32(
                    v5,
                    _mm_madd_epi16(_mm_unpackhi_epi16(ps.2, _mm_setzero_si128()), weight),
                );
            }

            let v0 = _mm_srai_epi32::<PREC>(v0);
            let v1 = _mm_srai_epi32::<PREC>(v1);
            let v2 = _mm_srai_epi32::<PREC>(v2);
            let v3 = _mm_srai_epi32::<PREC>(v3);
            let v4 = _mm_srai_epi32::<PREC>(v4);
            let v5 = _mm_srai_epi32::<PREC>(v5);

            let r_v = _mm_min_epi16(_mm_packus_epi32(v0, v1), v_max);
            let g_v = _mm_min_epi16(_mm_packus_epi32(v2, v3), v_max);
            let b_v = _mm_min_epi16(_mm_packus_epi32(v4, v5), v_max);

            let vals = _mm_zip_4_ar30::<AR30_TYPE, AR30_ORDER>((r_v, g_v, b_v, _mm_set1_epi16(3)));
            _mm_storeu_si128(dst_transient.as_mut_ptr() as *mut _, vals.0);
            _mm_storeu_si128(dst_transient.as_mut_ptr().add(4 * 4) as *mut _, vals.1);

            let v_dst = dst.get_unchecked_mut(v_start_px..(v_start_px + diff * 4));
            std::ptr::copy_nonoverlapping(dst_transient.as_ptr(), v_dst.as_mut_ptr(), diff * 4);
        }
    }
}
