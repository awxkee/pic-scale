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

use crate::filter_weights::FilterWeights;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
fn compress_i32(x: __m128i) -> __m128i {
    unsafe {
        let store_32 = _mm_srai_epi32::<7>(x);
        _mm_packus_epi32(store_32, store_32)
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_rows_4_i8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgb_avx_rows_i8_4_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[inline(always)]
unsafe fn load_rgb_x2(src: &[u8]) -> __m128i {
    let mut rgb_pixel = _mm_setzero_si128();
    rgb_pixel = _mm_insert_epi32::<0>(rgb_pixel, (src.as_ptr() as *const i32).read_unaligned());
    rgb_pixel = _mm_insert_epi16::<2>(
        rgb_pixel,
        (src.get_unchecked(4..).as_ptr() as *const i16).read_unaligned() as i32,
    );
    rgb_pixel
}

#[inline(always)]
unsafe fn load_rgb_x4(src: &[u8]) -> __m128i {
    let mut rgb_pixel = _mm_loadu_si64(src.as_ptr());
    rgb_pixel = _mm_insert_epi32::<2>(
        rgb_pixel,
        (src.get_unchecked(8..).as_ptr() as *const i32).read_unaligned(),
    );
    rgb_pixel
}

#[inline(always)]
unsafe fn load_distr_x8_rgb(src: &[u8], shuf: __m256i) -> __m256i {
    let pixel_lo = _mm_loadu_si128(src.as_ptr() as *const _);
    let pixel_hi = _mm_loadu_si64(src.get_unchecked(16..).as_ptr() as *const _);

    make_tuple_x8(pixel_lo, pixel_hi, shuf)
}

#[inline(always)]
unsafe fn make_tuple_x8(pixel: __m128i, pixel2: __m128i, shuf: __m256i) -> __m256i {
    // Low part
    // [R0, G0, B0] [R1, G1, B1] [R2 G2 B2] [R3 G3 B3] [R4 G4 B4] [R5]
    // High part
    // [G5, B5] [R6, G6, B6] [R7, G7, B7]

    let hi_part = _mm_alignr_epi8::<12>(pixel2, pixel);

    _mm256_shuffle_epi8(
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(pixel), hi_part),
        shuf,
    )
}

#[target_feature(enable = "avx2", enable = "avxvnni")]
unsafe fn convolve_horizontal_rgb_avx_rows_i8_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    const CHANNELS: usize = 3;

    const PRECISION: i32 = 7;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

    let shuffle_v = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, -1, -1, -1, -1);

    let shuffle_weights = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);

    let weights_idx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

    let shuffle_weights01 = _mm256_setr_epi8(
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
        2, 3,
    );
    let shuffle_pixels_4 = _mm256_setr_epi8(
        0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, -1, -1, -1, -1, 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11,
        -1, -1, -1, -1,
    );

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

        // [R0, G0, B0] [R1, G1, B1] [R2 G2 B2] [R3 G3 B3]

        if bounds.size > 4 {
            let mut store0 = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
                0,
            );
            let mut store1 = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
                0,
            );
            let mut store2 = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
                0,
            );
            let mut store3 = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
                0,
            );

            while jx + 8 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 8));
                let full_weights =
                    _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const _));

                let w0 = _mm256_shuffle_epi8(
                    _mm256_permutevar8x32_epi32(full_weights, weights_idx),
                    shuffle_weights01,
                );

                let bounds_start = (bounds.start + jx) * CHANNELS;

                let rgb_pixel_0 =
                    load_distr_x8_rgb(src0.get_unchecked(bounds_start..), shuffle_pixels_4);
                let rgb_pixel_1 =
                    load_distr_x8_rgb(src1.get_unchecked(bounds_start..), shuffle_pixels_4);
                let rgb_pixel_2 =
                    load_distr_x8_rgb(src2.get_unchecked(bounds_start..), shuffle_pixels_4);
                let rgb_pixel_3 =
                    load_distr_x8_rgb(src3.get_unchecked(bounds_start..), shuffle_pixels_4);

                store0 = _mm256_dpbusd_avx_epi32(store0, rgb_pixel_0, w0);
                store1 = _mm256_dpbusd_avx_epi32(store1, rgb_pixel_1, w0);
                store2 = _mm256_dpbusd_avx_epi32(store2, rgb_pixel_2, w0);
                store3 = _mm256_dpbusd_avx_epi32(store3, rgb_pixel_3, w0);

                jx += 8;
            }

            store_0 = _mm_add_epi32(
                _mm256_castsi256_si128(store0),
                _mm256_extracti128_si256::<1>(store0),
            );
            store_1 = _mm_add_epi32(
                _mm256_castsi256_si128(store1),
                _mm256_extracti128_si256::<1>(store1),
            );
            store_2 = _mm_add_epi32(
                _mm256_castsi256_si128(store2),
                _mm256_extracti128_si256::<1>(store2),
            );
            store_3 = _mm_add_epi32(
                _mm256_castsi256_si128(store3),
                _mm256_extracti128_si256::<1>(store3),
            );
        }

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));

            let weight0 =
                _mm_shuffle_epi8(_mm_loadu_si32(w_ptr.as_ptr() as *const u8), shuffle_weights);
            let bounds_start = (bounds.start + jx) * CHANNELS;

            let rgb_pixel_0 = load_rgb_x4(src0.get_unchecked(bounds_start..));
            let rgb_pixel_1 = load_rgb_x4(src1.get_unchecked(bounds_start..));
            let rgb_pixel_2 = load_rgb_x4(src2.get_unchecked(bounds_start..));
            let rgb_pixel_4 = load_rgb_x4(src3.get_unchecked(bounds_start..));

            let lo_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_v);
            let lo_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_v);
            let lo_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_v);
            let lo_3 = _mm_shuffle_epi8(rgb_pixel_4, shuffle_v);

            store_0 = _mm_dpbusd_avx_epi32(store_0, lo_0, weight0);
            store_1 = _mm_dpbusd_avx_epi32(store_1, lo_1, weight0);
            store_2 = _mm_dpbusd_avx_epi32(store_2, lo_2, weight0);
            store_3 = _mm_dpbusd_avx_epi32(store_3, lo_3, weight0);

            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = (bounds.start + jx) * CHANNELS;
            let weight0 =
                _mm_shuffle_epi8(_mm_loadu_si16(w_ptr.as_ptr() as *const u8), shuffle_weights);

            let rgb_pixel_0 = load_rgb_x2(src0.get_unchecked(bounds_start..));
            let rgb_pixel_1 = load_rgb_x2(src1.get_unchecked(bounds_start..));
            let rgb_pixel_2 = load_rgb_x2(src2.get_unchecked(bounds_start..));
            let rgb_pixel_4 = load_rgb_x2(src3.get_unchecked(bounds_start..));

            let lo_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_v);
            let lo_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_v);
            let lo_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_v);
            let lo_3 = _mm_shuffle_epi8(rgb_pixel_4, shuffle_v);

            store_0 = _mm_dpbusd_avx_epi32(store_0, lo_0, weight0);
            store_1 = _mm_dpbusd_avx_epi32(store_1, lo_1, weight0);
            store_2 = _mm_dpbusd_avx_epi32(store_2, lo_2, weight0);
            store_3 = _mm_dpbusd_avx_epi32(store_3, lo_3, weight0);

            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let bounds_start = bounds.start + jx;

            let weight0 = _mm_shuffle_epi8(
                _mm_set1_epi8(w_ptr.as_ptr().read_unaligned()),
                shuffle_weights,
            );

            store_0 = add_one_weight(bounds_start, src0, weight0, store_0);
            store_1 = add_one_weight(bounds_start, src1, weight0, store_1);
            store_2 = add_one_weight(bounds_start, src2, weight0, store_2);
            store_3 = add_one_weight(bounds_start, src3, weight0, store_3);
            jx += 1;
        }

        let store_0_8 = compress_i32(store_0);
        let store_1_8 = compress_i32(store_1);
        let store_2_8 = compress_i32(store_2);
        let store_3_8 = compress_i32(store_3);

        let store_0_8 = _mm_packus_epi16(store_0_8, store_0_8);
        let store_1_8 = _mm_packus_epi16(store_1_8, store_1_8);
        let store_2_8 = _mm_packus_epi16(store_2_8, store_2_8);
        let store_3_8 = _mm_packus_epi16(store_3_8, store_3_8);

        let element_0 = _mm_extract_epi32::<0>(store_0_8);
        let element_1 = _mm_extract_epi32::<0>(store_1_8);
        let element_2 = _mm_extract_epi32::<0>(store_2_8);
        let element_3 = _mm_extract_epi32::<0>(store_3_8);

        let bytes = element_0.to_le_bytes();
        let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
        (chunk0.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
        *chunk0.get_unchecked_mut(2) = bytes[2];

        let bytes = element_1.to_le_bytes();
        let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
        (chunk1.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
        *chunk1.get_unchecked_mut(2) = bytes[2];

        let bytes = element_2.to_le_bytes();
        let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
        (chunk2.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
        *chunk2.get_unchecked_mut(2) = bytes[2];

        let bytes = element_3.to_le_bytes();
        let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
        (chunk3.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
        *chunk3.get_unchecked_mut(2) = bytes[2];
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_row_i8_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgb_avx_row_i8_one_impl(src, dst, filter_weights);
    }
}

#[inline(always)]
unsafe fn add_one_weight(
    start_x: usize,
    src: &[u8],
    weight0: __m128i,
    store_0: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..).as_ptr();
    let base_pixel = _mm_loadu_si16(src.as_ptr());
    let m_vl = _mm_insert_epi8::<2>(base_pixel, src_ptr.add(2).read_unaligned() as i32);
    let lo = _mm_unpacklo_epi8(m_vl, _mm_setzero_si128());
    _mm_dpbusd_avx_epi32(store_0, lo, weight0)
}

#[target_feature(enable = "avx2", enable = "avxvnni")]
unsafe fn convolve_horizontal_rgb_avx_row_i8_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    const CHANNELS: usize = 3;

    let shuffle_v = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, -1, -1, -1, -1);

    let shuffle_weights = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);

    let weights_idx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

    let shuffle_weights01 = _mm256_setr_epi8(
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
        2, 3,
    );
    let shuffle_pixels_4 = _mm256_setr_epi8(
        0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, -1, -1, -1, -1, 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11,
        -1, -1, -1, -1,
    );

    // Low part
    // [R0, G0, B0] [R1, G1, B1] [R2 G2 B2] [R3 G3 B3] [R4 G4 B4] [R5]
    // High part
    // [G5, B5] [R6, G6, B6] [R7, G7, B7]

    const PRECISION: i32 = 7;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

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
            0,
            0,
            0,
            0,
            0,
        );

        let mut store = if bounds_size > 4 {
            while jx + 8 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 8));
                let full_weights =
                    _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const _));

                let w0 = _mm256_shuffle_epi8(
                    _mm256_permutevar8x32_epi32(full_weights, weights_idx),
                    shuffle_weights01,
                );

                let bounds_start = bounds.start + jx;
                let src_ptr_0 = src.get_unchecked((bounds_start * CHANNELS)..);

                let pixel_lo = _mm_loadu_si128(src_ptr_0.as_ptr() as *const _);
                let pixel_hi = _mm_loadu_si64(src_ptr_0.get_unchecked(16..).as_ptr() as *const _);

                let px = make_tuple_x8(pixel_lo, pixel_hi, shuffle_pixels_4);

                store = _mm256_dpbusd_avx_epi32(store, px, w0);

                jx += 8;
            }

            _mm_add_epi32(
                _mm256_castsi256_si128(store),
                _mm256_extracti128_si256::<1>(store),
            )
        } else {
            _mm_set1_epi32(ROUNDING_CONST)
        };

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let weight0 =
                _mm_shuffle_epi8(_mm_loadu_si32(w_ptr.as_ptr() as *const u8), shuffle_weights);
            let src_ptr = src.get_unchecked(((bounds.start + jx) * 3)..);
            let rgb_pixel = load_rgb_x4(src_ptr);
            let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_v);
            store = _mm_dpbusd_avx_epi32(store, lo, weight0);
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let weight0 =
                _mm_shuffle_epi8(_mm_loadu_si16(w_ptr.as_ptr() as *const u8), shuffle_weights);
            let src_ptr = src.get_unchecked(((bounds.start + jx) * 3)..);
            let rgb_pixel = load_rgb_x2(src_ptr);
            let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_v);
            store = _mm_dpbusd_avx_epi32(store, lo, weight0);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weight0 = _mm_shuffle_epi8(
                _mm_set1_epi8(w_ptr.as_ptr().read_unaligned()),
                shuffle_weights,
            );
            store = add_one_weight(bounds.start + jx, src, weight0, store);
            jx += 1;
        }

        let store_16_8 = compress_i32(store);
        let store_16_8 = _mm_packus_epi16(store_16_8, store_16_8);

        let element = _mm_extract_epi32::<0>(store_16_8);
        let bytes = element.to_le_bytes();
        let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
        (dst.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
        *dst.get_unchecked_mut(2) = bytes[2];
    }
}
