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

use crate::avx2::routines::compress_i32;
use crate::filter_weights::FilterWeights;
use crate::support::{PRECISION, ROUNDING_CONST};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn convolve_horizontal_rgba_row_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_avx_row_4_impl(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_rgba_avx_row_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 4;

    let shuffle_weights_table = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);

    let shuffle_2_table = _mm_setr_epi8(0, -1, 4, -1, 1, -1, 5, -1, 2, -1, 6, -1, 3, -1, 7, -1);

    let shuffle_1_table = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);

    let a_shuffle_weights_table = _mm256_setr_epi8(
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
        2, 3,
    );
    let a_shuffle_weights_table_hi = _mm256_setr_epi8(
        4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5,
        6, 7,
    );
    let a_shuffle_2_table_hi = _mm256_setr_epi8(
        8, -1, 12, -1, 9, -1, 13, -1, 10, -1, 14, -1, 11, -1, 15, -1, 8, -1, 12, -1, 9, -1, 13, -1,
        10, -1, 14, -1, 11, -1, 15, -1,
    );

    let a_shuffle_2_table = _mm256_setr_epi8(
        0, -1, 4, -1, 1, -1, 5, -1, 2, -1, 6, -1, 3, -1, 7, -1, 0, -1, 4, -1, 1, -1, 5, -1, 2, -1,
        6, -1, 3, -1, 7, -1,
    );

    let permute_avx_weights = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

    let vld = _mm_set1_epi32(ROUNDING_CONST);

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
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;

        let src0 = src;
        let src1 = src0.get_unchecked(src_stride..);
        let src2 = src1.get_unchecked(src_stride..);
        let src3 = src2.get_unchecked(src_stride..);

        if bounds.size > 4 {
            let mut store_avx0 = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
            );

            let mut store_avx1 = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
            );

            let mut store_avx2 = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
            );

            let mut store_avx3 = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
            );

            while jx + 8 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;

                let weights = _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(
                    _mm_loadu_si128(w_ptr.as_ptr() as *const _),
                ));
                let weight01 = _mm256_shuffle_epi8(weights, a_shuffle_weights_table);
                let weight23 = _mm256_shuffle_epi8(weights, a_shuffle_weights_table_hi);

                let rgb_pixel0 = _mm256_loadu_si256(
                    src0.get_unchecked((bounds_start * CHANNELS)..).as_ptr() as *const _,
                );
                let rgb_pixel1 = _mm256_loadu_si256(
                    src1.get_unchecked((bounds_start * CHANNELS)..).as_ptr() as *const _,
                );
                let rgb_pixel2 = _mm256_loadu_si256(
                    src2.get_unchecked((bounds_start * CHANNELS)..).as_ptr() as *const _,
                );
                let rgb_pixel3 = _mm256_loadu_si256(
                    src3.get_unchecked((bounds_start * CHANNELS)..).as_ptr() as *const _,
                );

                let hi0 = _mm256_shuffle_epi8(rgb_pixel0, a_shuffle_2_table_hi);
                let lo0 = _mm256_shuffle_epi8(rgb_pixel0, a_shuffle_2_table);
                let hi1 = _mm256_shuffle_epi8(rgb_pixel1, a_shuffle_2_table_hi);
                let lo1 = _mm256_shuffle_epi8(rgb_pixel1, a_shuffle_2_table);
                let hi2 = _mm256_shuffle_epi8(rgb_pixel2, a_shuffle_2_table_hi);
                let lo2 = _mm256_shuffle_epi8(rgb_pixel2, a_shuffle_2_table);
                let hi3 = _mm256_shuffle_epi8(rgb_pixel3, a_shuffle_2_table_hi);
                let lo3 = _mm256_shuffle_epi8(rgb_pixel3, a_shuffle_2_table);

                store_avx0 = _mm256_add_epi32(store_avx0, _mm256_madd_epi16(lo0, weight01));
                store_avx0 = _mm256_add_epi32(store_avx0, _mm256_madd_epi16(hi0, weight23));

                store_avx1 = _mm256_add_epi32(store_avx1, _mm256_madd_epi16(lo1, weight01));
                store_avx1 = _mm256_add_epi32(store_avx1, _mm256_madd_epi16(hi1, weight23));

                store_avx2 = _mm256_add_epi32(store_avx2, _mm256_madd_epi16(lo2, weight01));
                store_avx2 = _mm256_add_epi32(store_avx2, _mm256_madd_epi16(hi2, weight23));

                store_avx3 = _mm256_add_epi32(store_avx3, _mm256_madd_epi16(lo3, weight01));
                store_avx3 = _mm256_add_epi32(store_avx3, _mm256_madd_epi16(hi3, weight23));

                jx += 8;
            }

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;

                let weights = _mm256_permutevar8x32_epi32(
                    _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const u8)),
                    permute_avx_weights,
                );
                let weight01 = _mm256_shuffle_epi8(weights, a_shuffle_weights_table);

                let rgb_pixel_0 = _mm_loadu_si128(
                    src0.get_unchecked((bounds_start * CHANNELS)..).as_ptr() as *const _,
                );
                let rgb_pixel_1 = _mm_loadu_si128(
                    src1.get_unchecked((bounds_start * CHANNELS)..).as_ptr() as *const _,
                );
                let rgb_pixel_2 = _mm_loadu_si128(
                    src2.get_unchecked((bounds_start * CHANNELS)..).as_ptr() as *const _,
                );
                let rgb_pixel_3 = _mm_loadu_si128(
                    src3.get_unchecked((bounds_start * CHANNELS)..).as_ptr() as *const _,
                );

                let rgb_pixel0 =
                    _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(rgb_pixel_0));
                let rgb_pixel1 =
                    _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(rgb_pixel_1));
                let rgb_pixel2 =
                    _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(rgb_pixel_2));
                let rgb_pixel3 =
                    _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(rgb_pixel_3));

                let lo0 = _mm256_shuffle_epi8(rgb_pixel0, a_shuffle_2_table);
                let lo1 = _mm256_shuffle_epi8(rgb_pixel1, a_shuffle_2_table);
                let lo2 = _mm256_shuffle_epi8(rgb_pixel2, a_shuffle_2_table);
                let lo3 = _mm256_shuffle_epi8(rgb_pixel3, a_shuffle_2_table);

                store_avx0 = _mm256_add_epi32(store_avx0, _mm256_madd_epi16(lo0, weight01));
                store_avx1 = _mm256_add_epi32(store_avx1, _mm256_madd_epi16(lo1, weight01));
                store_avx2 = _mm256_add_epi32(store_avx2, _mm256_madd_epi16(lo2, weight01));
                store_avx3 = _mm256_add_epi32(store_avx3, _mm256_madd_epi16(lo3, weight01));

                jx += 4;
            }

            store_0 = _mm_add_epi32(
                _mm256_castsi256_si128(store_avx0),
                _mm256_extracti128_si256::<1>(store_avx0),
            );

            store_1 = _mm_add_epi32(
                _mm256_castsi256_si128(store_avx1),
                _mm256_extracti128_si256::<1>(store_avx1),
            );

            store_2 = _mm_add_epi32(
                _mm256_castsi256_si128(store_avx2),
                _mm256_extracti128_si256::<1>(store_avx2),
            );

            store_3 = _mm_add_epi32(
                _mm256_castsi256_si128(store_avx3),
                _mm256_extracti128_si256::<1>(store_avx3),
            );
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);
            let bounds_start = bounds.start + jx;

            let weight01 = _mm_shuffle_epi8(
                _mm_loadu_si32(w_ptr.as_ptr() as *const _),
                shuffle_weights_table,
            );

            let rgb_pixel_0 =
                _mm_loadu_si64(src0.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
            let rgb_pixel_1 =
                _mm_loadu_si64(src1.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
            let rgb_pixel_2 =
                _mm_loadu_si64(src2.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
            let rgb_pixel_3 =
                _mm_loadu_si64(src3.get_unchecked((bounds_start * CHANNELS)..).as_ptr());

            let lo_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_2_table);
            let lo_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_2_table);
            let lo_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_2_table);
            let lo_3 = _mm_shuffle_epi8(rgb_pixel_3, shuffle_2_table);

            store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo_0, weight01));
            store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo_1, weight01));
            store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo_2, weight01));
            store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo_3, weight01));

            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);
            let weight0 = _mm_shuffle_epi8(
                _mm_loadu_si16(w_ptr.as_ptr() as *const _),
                shuffle_weights_table,
            );

            let bounds_start = bounds.start + jx;

            let src_ptr0 = src0.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr1 = src1.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr2 = src2.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr3 = src3.get_unchecked((bounds_start * CHANNELS)..);

            let rgba_pixel0 = _mm_loadu_si32(src_ptr0.as_ptr() as *const _);
            let rgba_pixel1 = _mm_loadu_si32(src_ptr1.as_ptr() as *const _);
            let rgba_pixel2 = _mm_loadu_si32(src_ptr2.as_ptr() as *const _);
            let rgba_pixel3 = _mm_loadu_si32(src_ptr3.as_ptr() as *const _);

            let lo0 = _mm_shuffle_epi8(rgba_pixel0, shuffle_1_table);
            let lo1 = _mm_shuffle_epi8(rgba_pixel1, shuffle_1_table);
            let lo2 = _mm_shuffle_epi8(rgba_pixel2, shuffle_1_table);
            let lo3 = _mm_shuffle_epi8(rgba_pixel3, shuffle_1_table);

            store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo0, weight0));
            store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo1, weight0));
            store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo2, weight0));
            store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo3, weight0));

            jx += 1;
        }

        let store_16_8_0 = compress_i32(store_0);
        let store_16_8_1 = compress_i32(store_1);
        let store_16_8_2 = compress_i32(store_2);
        let store_16_8_3 = compress_i32(store_3);

        _mm_storeu_si32(
            chunk0.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8_0, store_16_8_0),
        );
        _mm_storeu_si32(
            chunk1.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8_1, store_16_8_1),
        );
        _mm_storeu_si32(
            chunk2.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8_2, store_16_8_2),
        );
        _mm_storeu_si32(
            chunk3.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8_3, store_16_8_3),
        );
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_row_1(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_avx_rows_one_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_rgba_avx_rows_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 4;

    let shuffle_weights_table = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);

    let shuffle_2_table = _mm_setr_epi8(0, -1, 4, -1, 1, -1, 5, -1, 2, -1, 6, -1, 3, -1, 7, -1);

    let shuffle_1_table = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);

    let a_shuffle_weights_table = _mm256_setr_epi8(
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
        2, 3,
    );
    let a_shuffle_weights_table_hi = _mm256_setr_epi8(
        4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5,
        6, 7,
    );
    let a_shuffle_2_table_hi = _mm256_setr_epi8(
        8, -1, 12, -1, 9, -1, 13, -1, 10, -1, 14, -1, 11, -1, 15, -1, 8, -1, 12, -1, 9, -1, 13, -1,
        10, -1, 14, -1, 11, -1, 15, -1,
    );

    let a_shuffle_2_table = _mm256_setr_epi8(
        0, -1, 4, -1, 1, -1, 5, -1, 2, -1, 6, -1, 3, -1, 7, -1, 0, -1, 4, -1, 1, -1, 5, -1, 2, -1,
        6, -1, 3, -1, 7, -1,
    );

    let permute_avx_weights = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

    let vld = _mm_set1_epi32(PRECISION);

    for ((dst, bounds), weights) in dst
        .chunks_exact_mut(CHANNELS)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;
        let mut store = vld;

        if bounds.size > 4 {
            let mut store_avx = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
            );

            while jx + 8 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;

                let weights = _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(
                    _mm_loadu_si128(w_ptr.as_ptr() as *const _),
                ));
                let weight01 = _mm256_shuffle_epi8(weights, a_shuffle_weights_table);
                let weight23 = _mm256_shuffle_epi8(weights, a_shuffle_weights_table_hi);

                let src_ptr = src.get_unchecked((bounds_start * CHANNELS)..);

                let rgb_pixel = _mm256_loadu_si256(src_ptr.as_ptr() as *const _);

                let hi = _mm256_shuffle_epi8(rgb_pixel, a_shuffle_2_table_hi);
                let lo = _mm256_shuffle_epi8(rgb_pixel, a_shuffle_2_table);

                store_avx = _mm256_add_epi32(store_avx, _mm256_madd_epi16(lo, weight01));
                store_avx = _mm256_add_epi32(store_avx, _mm256_madd_epi16(hi, weight23));

                jx += 8;
            }

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;

                let weights = _mm256_permutevar8x32_epi32(
                    _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const u8)),
                    permute_avx_weights,
                );
                let weight01 = _mm256_shuffle_epi8(weights, a_shuffle_weights_table);

                let src_ptr = src.get_unchecked((bounds_start * CHANNELS)..);

                let rgb_pixel = _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(
                    _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i),
                ));

                let lo = _mm256_shuffle_epi8(rgb_pixel, a_shuffle_2_table);

                store_avx = _mm256_add_epi32(store_avx, _mm256_madd_epi16(lo, weight01));
                jx += 4;
            }

            store = _mm_add_epi32(
                _mm256_castsi256_si128(store_avx),
                _mm256_extracti128_si256::<1>(store_avx),
            );
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);
            let bounds_start = bounds.start + jx;

            let weight01 = _mm_shuffle_epi8(
                _mm_loadu_si32(w_ptr.as_ptr() as *const _),
                shuffle_weights_table,
            );

            let src_ptr = src.get_unchecked((bounds_start * CHANNELS)..);

            let rgb_pixel = _mm_loadu_si64(src_ptr.as_ptr());
            let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_2_table);
            store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight01));

            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);
            let weight0 = _mm_shuffle_epi8(
                _mm_loadu_si16(w_ptr.as_ptr() as *const _),
                shuffle_weights_table,
            );

            let bounds_start = bounds.start + jx;

            const COMPONENTS: usize = 4;
            let src_ptr = src.get_unchecked((bounds_start * COMPONENTS)..);

            let src_ptr_32 = src_ptr.as_ptr() as *const i32;
            let rgba_pixel = _mm_loadu_si32(src_ptr_32 as *const _);

            let lo = _mm_shuffle_epi8(rgba_pixel, shuffle_1_table);

            store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight0));

            jx += 1;
        }

        let store_0 = _mm_srai_epi32::<PRECISION>(store);

        let store_0 = _mm_packus_epi32(store_0, store_0);

        _mm_storeu_si32(
            dst.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_0, store_0),
        );
    }
}
