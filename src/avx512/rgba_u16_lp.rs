/*
 * Copyright (c) Radzivon Bartoshyk 01/2025. All rights reserved.
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

use crate::filter_weights::FilterWeights;
use crate::support::{PRECISION, ROUNDING_CONST};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn acc_1_dot(
    start_x: usize,
    src: &[u16],
    w0: __m128i,
    store: __m128i,
    shuffle: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel = _mm_loadu_si64(src_ptr.as_ptr() as *const u8);
    _mm_dpwssd_avx_epi32(store, _mm_shuffle_epi8(rgba_pixel, shuffle), w0)
}

#[inline(always)]
unsafe fn acc_2_dot(
    start_x: usize,
    src: &[u16],
    w0: __m128i,
    store: __m128i,
    shuffle: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel = _mm_loadu_si128(src_ptr.as_ptr() as *const _);
    _mm_dpwssd_avx_epi32(store, _mm_shuffle_epi8(rgba_pixel, shuffle), w0)
}

#[inline(always)]
unsafe fn acc_4_dot(
    start_x: usize,
    src: &[u16],
    w0: __m256i,
    store: __m256i,
    shuffle: __m256i,
) -> __m256i {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel = _mm256_loadu_si256(src_ptr.as_ptr() as *const _);
    _mm256_dpwssd_avx_epi32(store, _mm256_shuffle_epi8(rgba_pixel, shuffle), w0)
}

#[inline(always)]
unsafe fn acc_8_dot(
    start_x: usize,
    src: &[u16],
    w0: __m256i,
    w1: __m256i,
    store: __m256i,
    shuffle: __m256i,
) -> __m256i {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const _);
    let rgba_pixel1 = _mm256_loadu_si256(src_ptr.get_unchecked(16..).as_ptr() as *const _);

    let p0 = _mm256_dpwssd_avx_epi32(store, _mm256_shuffle_epi8(rgba_pixel0, shuffle), w0);
    _mm256_dpwssd_avx_epi32(p0, _mm256_shuffle_epi8(rgba_pixel1, shuffle), w1)
}

pub(crate) fn convolve_horizontal_rgba_vnni_rows_4_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgba_vnni_rows_4_lb_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        );
    }
}

#[target_feature(enable = "avxvnni", enable = "avx2")]
unsafe fn convolve_horizontal_rgba_vnni_rows_4_lb_impl(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    const CHANNELS: usize = 4;

    let v_max_colors = _mm_set1_epi16((1 << bit_depth) - 1);

    let shuffle_weights_table = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);

    let shuffle_1_table = _mm_setr_epi8(0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 6, 7, -1, -1);

    let shuffle_2_table = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

    let permute_avx_weights = _mm256_setr_epi32(0, 2, 0, 0, 1, 3, 1, 1);

    let a_shuffle_2_table = _mm256_setr_epi8(
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12,
        13, 6, 7, 14, 15,
    );

    let a_shuffle_weights_table = _mm256_setr_epi8(
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
        2, 3,
    );

    let permute_avx_weights_hi = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);

    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
    let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
    let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
    let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

    for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
        .zip(iter_row1)
        .zip(iter_row2)
        .zip(iter_row3)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;
        let mut store_0 = _mm256_setr_epi32(
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            0,
            0,
            0,
            0,
        );
        let mut store_1 = _mm256_setr_epi32(
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            0,
            0,
            0,
            0,
        );
        let mut store_2 = _mm256_setr_epi32(
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            0,
            0,
            0,
            0,
        );
        let mut store_3 = _mm256_setr_epi32(
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            0,
            0,
            0,
            0,
        );

        let bounds_size = bounds.size;

        let src0 = src;
        let src1 = src0.get_unchecked(src_stride..);
        let src2 = src1.get_unchecked(src_stride..);
        let src3 = src2.get_unchecked(src_stride..);

        while jx + 8 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let wl = _mm256_castsi128_si256(_mm_loadu_si128(w_ptr.as_ptr() as *const _));
            let w0 = _mm256_shuffle_epi8(
                _mm256_permutevar8x32_epi32(wl, permute_avx_weights),
                a_shuffle_weights_table,
            );
            let w1 = _mm256_shuffle_epi8(
                _mm256_permutevar8x32_epi32(wl, permute_avx_weights_hi),
                a_shuffle_weights_table,
            );
            let bounds_start = bounds.start + jx;
            store_0 = acc_8_dot(bounds_start, src0, w0, w1, store_0, a_shuffle_2_table);
            store_1 = acc_8_dot(bounds_start, src1, w0, w1, store_1, a_shuffle_2_table);
            store_2 = acc_8_dot(bounds_start, src2, w0, w1, store_2, a_shuffle_2_table);
            store_3 = acc_8_dot(bounds_start, src3, w0, w1, store_3, a_shuffle_2_table);
            jx += 8;
        }

        while jx + 4 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let w0 = _mm256_shuffle_epi8(
                _mm256_permutevar8x32_epi32(
                    _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const _)),
                    permute_avx_weights,
                ),
                a_shuffle_weights_table,
            );
            store_0 = acc_4_dot(bounds_start, src0, w0, store_0, a_shuffle_2_table);
            store_1 = acc_4_dot(bounds_start, src1, w0, store_1, a_shuffle_2_table);
            store_2 = acc_4_dot(bounds_start, src2, w0, store_2, a_shuffle_2_table);
            store_3 = acc_4_dot(bounds_start, src3, w0, store_3, a_shuffle_2_table);
            jx += 4;
        }

        let mut store_0 = _mm_add_epi32(
            _mm256_castsi256_si128(store_0),
            _mm256_extracti128_si256::<1>(store_0),
        );
        let mut store_1 = _mm_add_epi32(
            _mm256_castsi256_si128(store_1),
            _mm256_extracti128_si256::<1>(store_1),
        );
        let mut store_2 = _mm_add_epi32(
            _mm256_castsi256_si128(store_2),
            _mm256_extracti128_si256::<1>(store_2),
        );
        let mut store_3 = _mm_add_epi32(
            _mm256_castsi256_si128(store_3),
            _mm256_extracti128_si256::<1>(store_3),
        );

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let w0 = _mm_shuffle_epi8(
                _mm_loadu_si32(w_ptr.as_ptr() as *const _),
                shuffle_weights_table,
            );
            store_0 = acc_2_dot(bounds_start, src0, w0, store_0, shuffle_2_table);
            store_1 = acc_2_dot(bounds_start, src1, w0, store_1, shuffle_2_table);
            store_2 = acc_2_dot(bounds_start, src2, w0, store_2, shuffle_2_table);
            store_3 = acc_2_dot(bounds_start, src3, w0, store_3, shuffle_2_table);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let bounds_start = bounds.start + jx;
            let w0 = _mm_shuffle_epi8(_mm_set1_epi16(w_ptr[0]), shuffle_weights_table);
            store_0 = acc_1_dot(bounds_start, src0, w0, store_0, shuffle_1_table);
            store_1 = acc_1_dot(bounds_start, src1, w0, store_1, shuffle_1_table);
            store_2 = acc_1_dot(bounds_start, src2, w0, store_2, shuffle_1_table);
            store_3 = acc_1_dot(bounds_start, src3, w0, store_3, shuffle_1_table);
            jx += 1;
        }

        store_0 = _mm_srai_epi32::<PRECISION>(store_0);
        store_1 = _mm_srai_epi32::<PRECISION>(store_1);
        store_2 = _mm_srai_epi32::<PRECISION>(store_2);
        store_3 = _mm_srai_epi32::<PRECISION>(store_3);

        let v_st0 = _mm_min_epi16(_mm_packus_epi32(store_0, store_0), v_max_colors);
        let v_st1 = _mm_min_epi16(_mm_packus_epi32(store_1, store_1), v_max_colors);
        let v_st2 = _mm_min_epi16(_mm_packus_epi32(store_2, store_2), v_max_colors);
        let v_st3 = _mm_min_epi16(_mm_packus_epi32(store_3, store_3), v_max_colors);

        _mm_storeu_si64(chunk0.as_mut_ptr() as *mut u8, v_st0);
        _mm_storeu_si64(chunk1.as_mut_ptr() as *mut u8, v_st1);
        _mm_storeu_si64(chunk2.as_mut_ptr() as *mut u8, v_st2);
        _mm_storeu_si64(chunk3.as_mut_ptr() as *mut u8, v_st3);
    }
}

pub(crate) fn convolve_horizontal_rgba_vnni_u16lp_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgba_vnni_u16_row_impl(src, dst, filter_weights, bit_depth);
    }
}

#[target_feature(enable = "avxvnni", enable = "avx2")]
unsafe fn convolve_horizontal_rgba_vnni_u16_row_impl(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    const CHANNELS: usize = 4;

    let v_max_colors = _mm_set1_epi16((1 << bit_depth) - 1);

    let shuffle_weights_table = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);

    let shuffle_1_table = _mm_setr_epi8(0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 6, 7, -1, -1);

    let shuffle_2_table = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

    let permute_avx_weights = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

    let permute_avx_weights_hi = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);

    let a_shuffle_2_table = _mm256_setr_epi8(
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12,
        13, 6, 7, 14, 15,
    );

    let a_shuffle_weights_table = _mm256_setr_epi8(
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
        2, 3,
    );

    for ((dst, bounds), weights) in dst
        .chunks_exact_mut(CHANNELS)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let bounds_size = bounds.size;
        let mut jx = 0usize;
        let mut store = _mm256_setr_epi32(
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            ROUNDING_CONST,
            0,
            0,
            0,
            0,
        );

        while jx + 8 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let wl = _mm256_castsi128_si256(_mm_loadu_si128(w_ptr.as_ptr() as *const _));
            let w0 = _mm256_shuffle_epi8(
                _mm256_permutevar8x32_epi32(wl, permute_avx_weights),
                a_shuffle_weights_table,
            );
            let w1 = _mm256_shuffle_epi8(
                _mm256_permutevar8x32_epi32(wl, permute_avx_weights_hi),
                a_shuffle_weights_table,
            );
            let bounds_start = bounds.start + jx;
            store = acc_8_dot(bounds_start, src, w0, w1, store, a_shuffle_2_table);
            jx += 8;
        }

        while jx + 4 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let w0 = _mm256_shuffle_epi8(
                _mm256_permutevar8x32_epi32(
                    _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const _)),
                    permute_avx_weights,
                ),
                a_shuffle_weights_table,
            );
            let bounds_start = bounds.start + jx;
            store = acc_4_dot(bounds_start, src, w0, store, a_shuffle_2_table);
            jx += 4;
        }

        let mut store = _mm_add_epi32(
            _mm256_castsi256_si128(store),
            _mm256_extracti128_si256::<1>(store),
        );

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let w0 = _mm_shuffle_epi8(
                _mm_loadu_si32(w_ptr.as_ptr() as *const _),
                shuffle_weights_table,
            );
            store = acc_2_dot(bounds_start, src, w0, store, shuffle_2_table);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let w0 = _mm_shuffle_epi8(_mm_set1_epi16(w_ptr[0]), shuffle_weights_table);
            let bounds_start = bounds.start + jx;
            store = acc_1_dot(bounds_start, src, w0, store, shuffle_1_table);
            jx += 1;
        }

        store = _mm_srai_epi32::<PRECISION>(store);

        let v_st = _mm_min_epi16(_mm_packus_epi32(store, store), v_max_colors);

        _mm_storeu_si64(dst.as_mut_ptr() as *mut u8, v_st);
    }
}
