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

use crate::avx2::utils::{_mm256_dot16_avx_epi32, _mm_dot16_avx_epi32};
use crate::filter_weights::FilterWeights;
use crate::support::{PRECISION, ROUNDING_CONST};
use std::arch::x86_64::*;

pub(crate) fn convolve_horizontal_rgb_avx_rows_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        {
            if std::arch::is_x86_feature_detected!("avxvnni") {
                return convolve_horizontal_rgb_avx_rows_4_vnni(
                    src,
                    src_stride,
                    dst,
                    dst_stride,
                    filter_weights,
                );
            }
        }
        convolve_horizontal_rgb_avx_rows_4_reg(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_rgb_avx_rows_4_reg(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    let unit = Row4ExecutionUnit::<false>::default();
    unit.pass(src, src_stride, dst, dst_stride, filter_weights);
}

#[cfg(feature = "nightly_avx512")]
#[target_feature(enable = "avx2", enable = "avxvnni")]
unsafe fn convolve_horizontal_rgb_avx_rows_4_vnni(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    let unit = Row4ExecutionUnit::<true>::default();
    unit.pass(src, src_stride, dst, dst_stride, filter_weights);
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
unsafe fn load_distr_x4_rgb(src: &[u8], shuf: __m256i) -> __m256i {
    let rgb_pixel = load_rgb_x4(src);

    // Extracting top pixel part
    let top_pixels = _mm_alignr_epi8::<6>(rgb_pixel, rgb_pixel);

    _mm256_shuffle_epi8(
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel), top_pixels),
        shuf,
    )
}

#[inline(always)]
unsafe fn load_distr_x8_rgb(src: &[u8], shuf: __m256i) -> (__m256i, __m256i) {
    let pixel_lo = _mm_loadu_si128(src.as_ptr() as *const _);
    let pixel_hi = _mm_loadu_si64(src.get_unchecked(16..).as_ptr() as *const _);

    let first_4 = make_first_4(pixel_lo, shuf);
    let second_4 = make_second_4(pixel_lo, pixel_hi, shuf);
    (first_4, second_4)
}

#[inline(always)]
unsafe fn make_first_4(pixel: __m128i, shuf: __m256i) -> __m256i {
    // Extracting top pixel part
    let top_pixels = _mm_alignr_epi8::<6>(pixel, pixel);

    _mm256_shuffle_epi8(
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(pixel), top_pixels),
        shuf,
    )
}

#[inline(always)]
unsafe fn make_second_4(pixel: __m128i, pixel2: __m128i, shuf: __m256i) -> __m256i {
    // Low part
    // [R0, G0, B0] [R1, G1, B1] [R2 G2 B2] [R3 G3 B3] [R4 G4 B4] [R5]
    // High part
    // [G5, B5] [R6, G6, B6] [R7, G7, B7]

    let low_part = _mm_alignr_epi8::<12>(pixel2, pixel);
    let hi_part = _mm_alignr_epi8::<2>(pixel2, pixel2);

    _mm256_shuffle_epi8(
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low_part), hi_part),
        shuf,
    )
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionUnit<const HAS_DOT: bool> {}

impl<const HAS_DOT: bool> Row4ExecutionUnit<HAS_DOT> {
    #[inline(always)]
    unsafe fn add_one_weight(
        &self,
        start_x: usize,
        src0: &[u8],
        src1: &[u8],
        weight0: __m256i,
        store_0: __m256i,
    ) -> __m256i {
        const COMPONENTS: usize = 3;
        let src_ptr0 = src0.get_unchecked((start_x * COMPONENTS)..).as_ptr();
        let src_ptr1 = src1.get_unchecked((start_x * COMPONENTS)..).as_ptr();
        let base_pixel0 = _mm_loadu_si16(src0.as_ptr());
        let base_pixel1 = _mm_loadu_si16(src1.as_ptr());
        let m_vl0 = _mm_insert_epi8::<2>(base_pixel0, src_ptr0.add(2).read_unaligned() as i32);
        let m_vl1 = _mm_insert_epi8::<2>(base_pixel1, src_ptr1.add(2).read_unaligned() as i32);
        let lo0 = _mm_unpacklo_epi8(m_vl0, _mm_setzero_si128());
        let lo1 = _mm_unpacklo_epi8(m_vl1, _mm_setzero_si128());
        let px = _mm_unpacklo_epi64(lo0, lo1);
        _mm256_dot16_avx_epi32::<HAS_DOT>(store_0, _mm256_cvtepu16_epi32(px), weight0)
    }

    #[inline(always)]
    unsafe fn pass(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
    ) {
        const CHANNELS: usize = 3;

        let shuffle_lo = _mm256_setr_epi8(
            0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1, 0, -1, 3, -1, 1, -1, 4, -1,
            2, -1, 5, -1, -1, -1, -1, -1,
        );

        let weights_idx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

        let shuffle_weights = _mm256_setr_epi8(
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, -1, -1, -1, -1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
            -1, -1, -1, -1,
        );
        let shuffle_pixels_4 = _mm256_setr_epi8(
            0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1, 0, -1, 3, -1, 1, -1, 4, -1,
            2, -1, 5, -1, -1, -1, -1, -1,
        );
        let weights_idx23 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);

        let vld = _mm256_set1_epi32(ROUNDING_CONST);

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
                    let w_ptr = weights.get_unchecked(jx..);
                    let full_weights =
                        _mm256_castsi128_si256(_mm_loadu_si128(w_ptr.as_ptr() as *const _));

                    let w0 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(full_weights, weights_idx),
                        shuffle_weights,
                    );
                    let w1 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(full_weights, weights_idx23),
                        shuffle_weights,
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

                    store0 = _mm256_dot16_avx_epi32::<HAS_DOT>(store0, rgb_pixel_0.0, w0);
                    store1 = _mm256_dot16_avx_epi32::<HAS_DOT>(store1, rgb_pixel_1.0, w0);
                    store2 = _mm256_dot16_avx_epi32::<HAS_DOT>(store2, rgb_pixel_2.0, w0);
                    store3 = _mm256_dot16_avx_epi32::<HAS_DOT>(store3, rgb_pixel_3.0, w0);

                    store0 = _mm256_dot16_avx_epi32::<HAS_DOT>(store0, rgb_pixel_0.1, w1);
                    store1 = _mm256_dot16_avx_epi32::<HAS_DOT>(store1, rgb_pixel_1.1, w1);
                    store2 = _mm256_dot16_avx_epi32::<HAS_DOT>(store2, rgb_pixel_2.1, w1);
                    store3 = _mm256_dot16_avx_epi32::<HAS_DOT>(store3, rgb_pixel_3.1, w1);

                    jx += 8;
                }

                while jx + 4 < bounds.size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let weights = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(
                            _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const u8)),
                            weights_idx,
                        ),
                        shuffle_weights,
                    );
                    let bounds_start = (bounds.start + jx) * CHANNELS;

                    let rgb_pixel_0 =
                        load_distr_x4_rgb(src0.get_unchecked(bounds_start..), shuffle_pixels_4);
                    let rgb_pixel_1 =
                        load_distr_x4_rgb(src1.get_unchecked(bounds_start..), shuffle_pixels_4);
                    let rgb_pixel_2 =
                        load_distr_x4_rgb(src2.get_unchecked(bounds_start..), shuffle_pixels_4);
                    let rgb_pixel_3 =
                        load_distr_x4_rgb(src3.get_unchecked(bounds_start..), shuffle_pixels_4);

                    store0 = _mm256_dot16_avx_epi32::<HAS_DOT>(store0, rgb_pixel_0, weights);
                    store1 = _mm256_dot16_avx_epi32::<HAS_DOT>(store1, rgb_pixel_1, weights);
                    store2 = _mm256_dot16_avx_epi32::<HAS_DOT>(store2, rgb_pixel_2, weights);
                    store3 = _mm256_dot16_avx_epi32::<HAS_DOT>(store3, rgb_pixel_3, weights);

                    jx += 4;
                }

                store_0 = _mm256_add_epi32(
                    _mm256_permute2x128_si256::<0x20>(store0, store1),
                    _mm256_permute2x128_si256::<0x31>(store0, store1),
                );

                store_1 = _mm256_add_epi32(
                    _mm256_permute2x128_si256::<0x20>(store2, store3),
                    _mm256_permute2x128_si256::<0x31>(store2, store3),
                );
            }

            while jx + 2 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = (bounds.start + jx) * CHANNELS;
                let weight01 = _mm256_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());

                let rgb_pixel_0 = load_rgb_x2(src0.get_unchecked(bounds_start..));
                let rgb_pixel_1 = load_rgb_x2(src1.get_unchecked(bounds_start..));
                let rgb_pixel_2 = load_rgb_x2(src2.get_unchecked(bounds_start..));
                let rgb_pixel_3 = load_rgb_x2(src3.get_unchecked(bounds_start..));

                let px0 =
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_0), rgb_pixel_1);
                let px1 =
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_2), rgb_pixel_3);

                let lo_0 = _mm256_shuffle_epi8(px0, shuffle_lo);
                let lo_1 = _mm256_shuffle_epi8(px1, shuffle_lo);

                store_0 = _mm256_dot16_avx_epi32::<HAS_DOT>(store_0, lo_0, weight01);
                store_1 = _mm256_dot16_avx_epi32::<HAS_DOT>(store_1, lo_1, weight01);

                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;

                let weight0 = _mm256_set1_epi32(w_ptr.as_ptr().read_unaligned() as i32);

                store_0 = self.add_one_weight(bounds_start, src0, src1, weight0, store_0);
                store_1 = self.add_one_weight(bounds_start, src2, src3, weight0, store_1);
                jx += 1;
            }

            store_0 = _mm256_srai_epi32::<PRECISION>(store_0);
            store_1 = _mm256_srai_epi32::<PRECISION>(store_1);

            let store_16_8_0 = _mm256_packus_epi32(store_0, store_0);
            let store_16_8_1 = _mm256_packus_epi32(store_1, store_1);

            let packed16_0 = _mm256_packus_epi16(store_16_8_0, store_16_8_0);
            let packed16_1 = _mm256_packus_epi16(store_16_8_1, store_16_8_1);

            let element_0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(packed16_0));
            let element_1 = _mm256_extract_epi32::<4>(packed16_0);
            let element_2 = _mm_cvtsi128_si32(_mm256_castsi256_si128(packed16_1));
            let element_3 = _mm256_extract_epi32::<4>(packed16_1);

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
}

pub(crate) fn convolve_horizontal_rgb_avx_row_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        {
            if std::arch::is_x86_feature_detected!("avxvnni") {
                return convolve_horizontal_rgb_avx_row_one_vnni(src, dst, filter_weights);
            }
        }
        convolve_horizontal_rgb_avx_row_one_reg(src, dst, filter_weights);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_rgb_avx_row_one_reg(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    let unit = Row1Execution::<false>::default();
    unit.pass(src, dst, filter_weights);
}

#[cfg(feature = "nightly_avx512")]
#[target_feature(enable = "avx2", enable = "avxvnni")]
unsafe fn convolve_horizontal_rgb_avx_row_one_vnni(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    let unit = Row1Execution::<true>::default();
    unit.pass(src, dst, filter_weights);
}

#[inline(always)]
unsafe fn add_one_weight<const HAS_DOT: bool>(
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
    _mm_dot16_avx_epi32::<HAS_DOT>(
        store_0,
        _mm_unpacklo_epi16(lo, _mm_setzero_si128()),
        weight0,
    )
}

#[derive(Copy, Clone, Default)]
struct Row1Execution<const HAS_DOT: bool> {}

impl<const HAS_DOT: bool> Row1Execution<HAS_DOT> {
    #[inline(always)]
    unsafe fn pass(&self, src: &[u8], dst: &mut [u8], filter_weights: &FilterWeights<i16>) {
        const CHANNELS: usize = 3;

        let shuffle_lo = _mm_setr_epi8(0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1);

        let weights_idx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
        let weights_idx23 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);

        let shuffle_weights01 = _mm256_setr_epi8(
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, -1, -1, -1, -1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
            -1, -1, -1, -1,
        );
        let shuffle_pixels_4 = _mm256_setr_epi8(
            0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1, 0, -1, 3, -1, 1, -1, 4, -1,
            2, -1, 5, -1, -1, -1, -1, -1,
        );

        // Low part
        // [R0, G0, B0] [R1, G1, B1] [R2 G2 B2] [R3 G3 B3] [R4 G4 B4] [R5]
        // High part
        // [G5, B5] [R6, G6, B6] [R7, G7, B7]

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
                    let w_ptr = weights.get_unchecked(jx..);
                    let full_weights =
                        _mm256_castsi128_si256(_mm_loadu_si128(w_ptr.as_ptr() as *const _));

                    let w0 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(full_weights, weights_idx),
                        shuffle_weights01,
                    );
                    let w1 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(full_weights, weights_idx23),
                        shuffle_weights01,
                    );

                    let bounds_start = bounds.start + jx;
                    let src_ptr_0 = src.get_unchecked((bounds_start * CHANNELS)..);

                    let pixel_lo = _mm_loadu_si128(src_ptr_0.as_ptr() as *const _);
                    let pixel_hi =
                        _mm_loadu_si64(src_ptr_0.get_unchecked(16..).as_ptr() as *const _);

                    let first_4 = make_first_4(pixel_lo, shuffle_pixels_4);
                    let second_4 = make_second_4(pixel_lo, pixel_hi, shuffle_pixels_4);

                    store = _mm256_dot16_avx_epi32::<HAS_DOT>(store, first_4, w0);
                    store = _mm256_dot16_avx_epi32::<HAS_DOT>(store, second_4, w1);

                    jx += 8;
                }

                while jx + 4 < bounds.size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let weights = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(
                            _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const u8)),
                            weights_idx,
                        ),
                        shuffle_weights01,
                    );

                    let bounds_start = bounds.start + jx;
                    let src_ptr_0 = src.get_unchecked((bounds_start * CHANNELS)..);

                    let rgb_pixel = load_distr_x4_rgb(src_ptr_0, shuffle_pixels_4);
                    store = _mm256_dot16_avx_epi32::<HAS_DOT>(store, rgb_pixel, weights);
                    jx += 4;
                }
                _mm_add_epi32(
                    _mm256_castsi256_si128(store),
                    _mm256_extracti128_si256::<1>(store),
                )
            } else {
                _mm_set1_epi32(ROUNDING_CONST)
            };

            while jx + 2 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                let src_ptr = src.get_unchecked(((bounds.start + jx) * 3)..);
                let rgb_pixel = load_rgb_x2(src_ptr);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
                store = _mm_dot16_avx_epi32::<HAS_DOT>(store, lo, weight0);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = _mm_set1_epi32(w_ptr.as_ptr().read_unaligned() as i32);
                store = add_one_weight::<HAS_DOT>(bounds.start + jx, src, weight0, store);
                jx += 1;
            }

            use crate::avx2::routines::compress_i32;
            let store_16_8 = compress_i32(store);
            let store_16_8 = _mm_packus_epi16(store_16_8, store_16_8);

            let element = _mm_extract_epi32::<0>(store_16_8);
            let bytes = element.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (dst.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
            *dst.get_unchecked_mut(2) = bytes[2];
        }
    }
}
