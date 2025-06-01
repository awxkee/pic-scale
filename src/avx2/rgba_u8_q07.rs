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

use crate::avx2::routines::shuffle;
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn c_one(
    start_x: usize,
    src: &[u8],
    weight0: __m128i,
    store_0: __m128i,
    shuffle: __m128i,
) -> __m128i {
    unsafe {
        const CN: usize = 4;
        let src_ptr = src.get_unchecked((start_x * CN)..);

        let src_ptr_32 = src_ptr.as_ptr() as *const i32;
        let rgba_pixel = _mm_cvtsi32_si128(src_ptr_32.read_unaligned());
        let lo = _mm_shuffle_epi8(rgba_pixel, shuffle);

        _mm_add_epi16(store_0, _mm_maddubs_epi16(lo, weight0))
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_rows_4_q07(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgba_avx_rows_4_impl(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[inline(always)]
unsafe fn hdot2(store: __m128i, v: __m128i, w0123: __m128i, v_data: __m128i) -> __m128i {
    unsafe {
        let v0 = _mm_shuffle_epi8(v, v_data);
        _mm_add_epi16(store, _mm_maddubs_epi16(v0, w0123))
    }
}

#[inline(always)]
unsafe fn hdot(store: __m128i, v: __m128i, w01: __m128i, v_data: __m128i) -> __m128i {
    unsafe {
        let lo = _mm_shuffle_epi8(v, v_data);
        _mm_add_epi16(store, _mm_maddubs_epi16(lo, w01))
    }
}
#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_rgba_avx_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        const CN: usize = 4;

        const ROUNDING: i16 = 1 << (7 - 1);

        let rnd = _mm256_setr_epi16(
            ROUNDING, ROUNDING, ROUNDING, ROUNDING, 0, 0, 0, 0, ROUNDING, ROUNDING, ROUNDING,
            ROUNDING, 0, 0, 0, 0,
        );

        let shuffle_weights = _mm256_setr_epi8(
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0,
            1, 2, 3,
        );
        let prepare_data = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6,
            10, 14, 3, 7, 11, 15,
        );

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.chunks_exact_mut(CN);
        let iter_row1 = row1_ref.chunks_exact_mut(CN);
        let iter_row2 = row2_ref.chunks_exact_mut(CN);
        let iter_row3 = row3_ref.chunks_exact_mut(CN);

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
            let mut store_0 = _mm256_setzero_si256();
            let mut store_1 = _mm256_setzero_si256();

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);

                let wz = _mm_loadu_si32(w_ptr.as_ptr() as *const u8);
                let w0101 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(wz), wz);
                let weight01 = _mm256_shuffle_epi8(w0101, shuffle_weights);

                let start_bounds = bounds.start + jx;

                let rgb_pixel_0 = _mm_loadu_si128(
                    src0.get_unchecked((start_bounds * CN)..).as_ptr() as *const __m128i,
                );
                let rgb_pixel_1 = _mm_loadu_si128(
                    src1.get_unchecked((start_bounds * CN)..).as_ptr() as *const __m128i,
                );
                let rgb_pixel_2 = _mm_loadu_si128(
                    src2.get_unchecked((start_bounds * CN)..).as_ptr() as *const __m128i,
                );
                let rgb_pixel_3 = _mm_loadu_si128(
                    src3.get_unchecked((start_bounds * CN)..).as_ptr() as *const __m128i,
                );

                let px01 =
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_0), rgb_pixel_1);
                let px23 =
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_2), rgb_pixel_3);

                let v01 = _mm256_shuffle_epi8(px01, prepare_data);
                let v23 = _mm256_shuffle_epi8(px23, prepare_data);
                store_0 = _mm256_add_epi16(store_0, _mm256_maddubs_epi16(v01, weight01));
                store_1 = _mm256_add_epi16(store_1, _mm256_maddubs_epi16(v23, weight01));

                jx += 4;
            }

            while jx + 2 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;

                let wz = _mm_loadu_si16(w_ptr.as_ptr() as *const u8);
                let w0101 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(wz), wz);
                let weight01 = _mm256_shuffle_epi8(w0101, shuffle_weights);

                let rgb_pixel_0 =
                    _mm_loadu_si64(src0.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgb_pixel_1 =
                    _mm_loadu_si64(src1.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgb_pixel_2 =
                    _mm_loadu_si64(src2.get_unchecked((bounds_start * CN)..).as_ptr());
                let rgb_pixel_3 =
                    _mm_loadu_si64(src3.get_unchecked((bounds_start * CN)..).as_ptr());

                let px01 =
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_0), rgb_pixel_1);
                let px23 =
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_2), rgb_pixel_3);

                let v01 = _mm256_shuffle_epi8(px01, prepare_data);
                let v23 = _mm256_shuffle_epi8(px23, prepare_data);
                store_0 = _mm256_add_epi16(store_0, _mm256_maddubs_epi16(v01, weight01));
                store_1 = _mm256_add_epi16(store_1, _mm256_maddubs_epi16(v23, weight01));

                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);

                let weight0 = _mm256_set1_epi8(w_ptr[0]);

                let start_bounds = bounds.start + jx;

                let rgb_pixel_0 = _mm_cvtsi32_si128(
                    (src0.get_unchecked((start_bounds * CN)..).as_ptr() as *const i32)
                        .read_unaligned(),
                );
                let rgb_pixel_1 = _mm_cvtsi32_si128(
                    (src1.get_unchecked((start_bounds * CN)..).as_ptr() as *const i32)
                        .read_unaligned(),
                );
                let rgb_pixel_2 = _mm_cvtsi32_si128(
                    (src2.get_unchecked((start_bounds * CN)..).as_ptr() as *const i32)
                        .read_unaligned(),
                );
                let rgb_pixel_3 = _mm_cvtsi32_si128(
                    (src3.get_unchecked((start_bounds * CN)..).as_ptr() as *const i32)
                        .read_unaligned(),
                );

                let px01 =
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_0), rgb_pixel_1);
                let px23 =
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_2), rgb_pixel_3);

                let v01 = _mm256_shuffle_epi8(px01, prepare_data);
                let v23 = _mm256_shuffle_epi8(px23, prepare_data);
                store_0 = _mm256_add_epi16(store_0, _mm256_maddubs_epi16(v01, weight0));
                store_1 = _mm256_add_epi16(store_1, _mm256_maddubs_epi16(v23, weight0));
                jx += 1;
            }

            store_0 = _mm256_adds_epi16(_mm256_hadd_epi16(store_0, store_0), rnd);
            store_1 = _mm256_adds_epi16(_mm256_hadd_epi16(store_1, store_1), rnd);

            const M: i32 = shuffle(3, 1, 2, 0);
            store_0 = _mm256_permute4x64_epi64::<M>(store_0);
            store_1 = _mm256_permute4x64_epi64::<M>(store_1);

            let store_16_8_0 = _mm_srai_epi16::<7>(_mm256_castsi256_si128(store_0));
            let store_16_8_1 = _mm_srai_epi16::<7>(_mm256_extracti128_si256::<1>(store_0));
            let store_16_8_2 = _mm_srai_epi16::<7>(_mm256_castsi256_si128(store_1));
            let store_16_8_3 = _mm_srai_epi16::<7>(_mm256_extracti128_si256::<1>(store_1));

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
}

pub(crate) fn convolve_horizontal_rgba_avx_rows_one_q07(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgba_avx2_rows_one_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_rgba_avx2_rows_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        const CN: usize = 4;

        let shuffle_weights = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);
        let prepare_data = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

        const ROUNDING: i16 = 1 << (7 - 1);

        let rnd = _mm_setr_epi16(ROUNDING, ROUNDING, ROUNDING, ROUNDING, 0, 0, 0, 0);

        for ((dst, bounds), weights) in dst
            .chunks_exact_mut(CN)
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut jx = 0usize;
            let mut store = _mm_setzero_si128();

            while jx + 4 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;

                let weight01 =
                    _mm_shuffle_epi8(_mm_loadu_si32(w_ptr.as_ptr() as *const u8), shuffle_weights);

                let src_ptr = src.get_unchecked((bounds_start * CN)..);

                let rgb_pixel = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);

                store = hdot2(store, rgb_pixel, weight01, prepare_data);

                jx += 4;
            }

            while jx + 2 < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;

                let weight01 =
                    _mm_shuffle_epi8(_mm_loadu_si16(w_ptr.as_ptr() as *const u8), shuffle_weights);

                let src_ptr = src.get_unchecked((bounds_start * CN)..);

                let rgb_pixel = _mm_loadu_si64(src_ptr.as_ptr());

                store = hdot(store, rgb_pixel, weight01, prepare_data);

                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..);
                let weight0 = _mm_set1_epi8(w_ptr[0]);

                let start_bounds = bounds.start + jx;

                store = c_one(start_bounds, src, weight0, store, prepare_data);
                jx += 1;
            }

            store = _mm_adds_epi16(_mm_hadd_epi16(store, store), rnd);

            let store_16_8 = _mm_srai_epi16::<7>(store);
            _mm_storeu_si32(
                dst.as_mut_ptr() as *mut _,
                _mm_packus_epi16(store_16_8, store_16_8),
            );
        }
    }
}
