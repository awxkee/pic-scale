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
use crate::avx2::ar30_utils::{_mm_unzip_3_ar30, _mm_zip_4_ar30};
use crate::filter_weights::FilterBounds;
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) fn avx_column_handler_fixed_point_ar30<
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
        let unit = ExecutionUnit::<AR30_TYPE, AR30_ORDER>::default();
        unit.pass(bounds, src, dst, src_stride, weight);
    }
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit<const AR30_TYPE: usize, const AR30_ORDER: usize> {}

impl<const AR30_TYPE: usize, const AR30_ORDER: usize> ExecutionUnit<AR30_TYPE, AR30_ORDER> {
    #[target_feature(enable = "avx2")]
    unsafe fn pass(
        &self,
        bounds: &FilterBounds,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        weight: &[i16],
    ) {
        let mut cx = 0usize;

        let total_width = dst.len() / 4;

        const PREC: i32 = 15;
        const RND_CONST: i32 = (1 << (PREC - 1)) - 1;

        while cx + 8 < total_width {
            unsafe {
                let v_max = _mm_set1_epi16(1023);
                let filter = weight;
                let v_start_px = cx * 4;

                let mut v0 = _mm256_set1_epi32(RND_CONST);
                let mut v1 = _mm256_set1_epi32(RND_CONST);
                let mut v2 = _mm256_set1_epi32(RND_CONST);

                for (j, &k_weight) in filter.iter().take(bounds.size).enumerate() {
                    let py = bounds.start + j;
                    let weight = _mm256_set1_epi16(k_weight);
                    let offset = src_stride * py + v_start_px;
                    let src_ptr = src.get_unchecked(offset..(offset + 8 * 4));

                    let l0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
                    let l1 = _mm_loadu_si128(src_ptr.as_ptr().add(4 * 4) as *const __m128i);

                    let ps = _mm_unzip_3_ar30::<AR30_TYPE, AR30_ORDER>((l0, l1));

                    let ps0 = _mm256_cvtepu16_epi32(ps.0);
                    let ps1 = _mm256_cvtepu16_epi32(ps.1);
                    let ps2 = _mm256_cvtepu16_epi32(ps.2);

                    v0 = _mm256_add_epi32(v0, _mm256_madd_epi16(ps0, weight));
                    v1 = _mm256_add_epi32(v1, _mm256_madd_epi16(ps1, weight));
                    v2 = _mm256_add_epi32(v2, _mm256_madd_epi16(ps2, weight));
                }

                let v0 = _mm256_srai_epi32::<PREC>(v0);
                let v1 = _mm256_srai_epi32::<PREC>(v1);
                let v2 = _mm256_srai_epi32::<PREC>(v2);

                let r_v = _mm_min_epi16(
                    _mm_packus_epi32(
                        _mm256_castsi256_si128(v0),
                        _mm256_extracti128_si256::<1>(v0),
                    ),
                    v_max,
                );
                let g_v = _mm_min_epi16(
                    _mm_packus_epi32(
                        _mm256_castsi256_si128(v1),
                        _mm256_extracti128_si256::<1>(v1),
                    ),
                    v_max,
                );
                let b_v = _mm_min_epi16(
                    _mm_packus_epi32(
                        _mm256_castsi256_si128(v2),
                        _mm256_extracti128_si256::<1>(v2),
                    ),
                    v_max,
                );

                let v_dst = dst.get_unchecked_mut(v_start_px..(v_start_px + 8 * 4));

                let vals =
                    _mm_zip_4_ar30::<AR30_TYPE, AR30_ORDER>((r_v, g_v, b_v, _mm_set1_epi16(3)));
                _mm_storeu_si128(v_dst.as_mut_ptr() as *mut _, vals.0);
                _mm_storeu_si128(v_dst.as_mut_ptr().add(4 * 4) as *mut _, vals.1);

                cx += 8;
            }
        }

        if cx < total_width {
            let diff = total_width - cx;

            let mut src_transient: [u8; 4 * 8] = [0; 4 * 8];
            let mut dst_transient: [u8; 4 * 8] = [0; 4 * 8];

            let v_max = _mm_set1_epi16(1023);
            let filter = weight;
            let v_start_px = cx * 4;

            let mut v0 = _mm256_set1_epi32(RND_CONST);
            let mut v1 = _mm256_set1_epi32(RND_CONST);
            let mut v2 = _mm256_set1_epi32(RND_CONST);

            for (j, &k_weight) in filter.iter().take(bounds.size).enumerate() {
                let py = bounds.start + j;
                let weight = _mm256_set1_epi16(k_weight);
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

                let ps0 = _mm256_cvtepu16_epi32(ps.0);
                let ps1 = _mm256_cvtepu16_epi32(ps.1);
                let ps2 = _mm256_cvtepu16_epi32(ps.2);

                v0 = _mm256_add_epi32(v0, _mm256_madd_epi16(ps0, weight));
                v1 = _mm256_add_epi32(v1, _mm256_madd_epi16(ps1, weight));
                v2 = _mm256_add_epi32(v2, _mm256_madd_epi16(ps2, weight));
            }

            let v0 = _mm256_srai_epi32::<PREC>(v0);
            let v1 = _mm256_srai_epi32::<PREC>(v1);
            let v2 = _mm256_srai_epi32::<PREC>(v2);

            let r_v = _mm_min_epi16(
                _mm_packus_epi32(
                    _mm256_castsi256_si128(v0),
                    _mm256_extracti128_si256::<1>(v0),
                ),
                v_max,
            );
            let g_v = _mm_min_epi16(
                _mm_packus_epi32(
                    _mm256_castsi256_si128(v1),
                    _mm256_extracti128_si256::<1>(v1),
                ),
                v_max,
            );
            let b_v = _mm_min_epi16(
                _mm_packus_epi32(
                    _mm256_castsi256_si128(v2),
                    _mm256_extracti128_si256::<1>(v2),
                ),
                v_max,
            );

            let vals = _mm_zip_4_ar30::<AR30_TYPE, AR30_ORDER>((r_v, g_v, b_v, _mm_set1_epi16(3)));
            _mm_storeu_si128(dst_transient.as_mut_ptr() as *mut _, vals.0);
            _mm_storeu_si128(dst_transient.as_mut_ptr().add(4 * 4) as *mut _, vals.1);

            let v_dst = dst.get_unchecked_mut(v_start_px..(v_start_px + diff * 4));
            std::ptr::copy_nonoverlapping(dst_transient.as_ptr(), v_dst.as_mut_ptr(), diff * 4);
        }
    }
}
