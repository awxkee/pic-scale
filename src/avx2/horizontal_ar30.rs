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
use crate::avx2::ar30_utils::{
    _mm_extract_ar30, _mm_ld1_ar30_s16, _mm_unzip_4_ar30_separate, _mm_unzips_4_ar30_separate,
};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

pub(crate) fn avx_convolve_horizontal_rgba_rows_4_ar30<
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        let unit = Row4ExecutionUnit::<AR_TYPE, AR_ORDER>::default();
        unit.pass(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionUnit<const AR_TYPE: usize, const AR_ORDER: usize> {}

impl<const AR_TYPE: usize, const AR_ORDER: usize> Row4ExecutionUnit<AR_TYPE, AR_ORDER> {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn conv_horiz_rgba_8_u8_i16(
        &self,
        start_x: usize,
        src0: &[u8],
        src1: &[u8],
        w0: __m256i,
        w1: __m256i,
        w2: __m256i,
        w3: __m256i,
        store: __m256i,
    ) -> __m256i {
        unsafe {
            let src_ptr0 = src0.get_unchecked(start_x * 4..);
            let src_ptr1 = src1.get_unchecked(start_x * 4..);

            let l0_0 = _mm_loadu_si128(src_ptr0.as_ptr() as *const _);
            let l0_1 = _mm_loadu_si128(src_ptr0.get_unchecked(16..).as_ptr() as *const _);

            let l1_0 = _mm_loadu_si128(src_ptr1.as_ptr() as *const _);
            let l1_1 = _mm_loadu_si128(src_ptr1.get_unchecked(16..).as_ptr() as *const _);

            let rgba_pixel0 = _mm_unzip_4_ar30_separate::<AR_TYPE, AR_ORDER>((l0_0, l0_1));
            let rgba_pixel1 = _mm_unzip_4_ar30_separate::<AR_TYPE, AR_ORDER>((l1_0, l1_1));

            let sh1 = _mm256_setr_epi8(
                0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4,
                5, 12, 13, 6, 7, 14, 15,
            );

            let v0 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.0), rgba_pixel1.0),
                sh1,
            );
            let v1 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.1), rgba_pixel1.1),
                sh1,
            );
            let v2 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.2), rgba_pixel1.2),
                sh1,
            );
            let v3 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.3), rgba_pixel1.3),
                sh1,
            );

            let mut v = _mm256_add_epi32(store, _mm256_madd_epi16(v0, w0));
            v = _mm256_add_epi32(v, _mm256_madd_epi16(v1, w1));
            v = _mm256_add_epi32(v, _mm256_madd_epi16(v2, w2));
            _mm256_add_epi32(v, _mm256_madd_epi16(v3, w3))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn conv_horiz_rgba_4_u8_i16(
        &self,
        start_x: usize,
        src0: &[u8],
        src1: &[u8],
        w0: __m256i,
        w1: __m256i,
        store: __m256i,
    ) -> __m256i {
        unsafe {
            let src_ptr0 = src0.get_unchecked(start_x * 4..);
            let src_ptr1 = src1.get_unchecked(start_x * 4..);

            let rgba_pixel0 = _mm_unzips_4_ar30_separate::<AR_TYPE, AR_ORDER>(_mm_loadu_si128(
                src_ptr0.as_ptr() as *const _,
            ));
            let rgba_pixel1 = _mm_unzips_4_ar30_separate::<AR_TYPE, AR_ORDER>(_mm_loadu_si128(
                src_ptr1.as_ptr() as *const _,
            ));

            let sh1 = _mm256_setr_epi8(
                0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4,
                5, 12, 13, 6, 7, 14, 15,
            );

            let v0 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.0), rgba_pixel1.0),
                sh1,
            );
            let v1 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.1), rgba_pixel1.1),
                sh1,
            );

            let v = _mm256_add_epi32(store, _mm256_madd_epi16(v0, w0));
            _mm256_add_epi32(v, _mm256_madd_epi16(v1, w1))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn conv_horiz_rgba_1_u8_i16(
        &self,
        start_x: usize,
        src0: &[u8],
        src1: &[u8],
        w0: __m256i,
        store: __m256i,
    ) -> __m256i {
        unsafe {
            let src_ptr0 = src0.get_unchecked(start_x * 4..);
            let src_ptr1 = src1.get_unchecked(start_x * 4..);

            let ld0 = _mm_ld1_ar30_s16::<AR_TYPE, AR_ORDER>(src_ptr0);
            let ld1 = _mm_ld1_ar30_s16::<AR_TYPE, AR_ORDER>(src_ptr1);

            let full_lane = _mm_unpacklo_epi64(ld0, ld1);

            _mm256_add_epi32(
                store,
                _mm256_madd_epi16(_mm256_cvtepu16_epi32(full_lane), w0),
            )
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn pass(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
    ) {
        unsafe {
            const PRECISION: i32 = 15;
            const ROUNDING: i32 = 1 << (PRECISION - 1);

            let init = _mm256_set1_epi32(ROUNDING);

            let v_cut_off = _mm256_set1_epi16(1023);

            let (row0_ref, rest) = dst.split_at_mut(dst_stride);
            let (row1_ref, rest) = rest.split_at_mut(dst_stride);
            let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

            let iter_row0 = row0_ref.chunks_exact_mut(4);
            let iter_row1 = row1_ref.chunks_exact_mut(4);
            let iter_row2 = row2_ref.chunks_exact_mut(4);
            let iter_row3 = row3_ref.chunks_exact_mut(4);

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

                let bounds_size = bounds.size;

                let mut store_0 = init;
                let mut store_1 = init;

                let src0 = src;
                let src1 = src0.get_unchecked(src_stride..);
                let src2 = src1.get_unchecked(src_stride..);
                let src3 = src2.get_unchecked(src_stride..);

                while jx + 8 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm256_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                    let w1 =
                        _mm256_set1_epi32((w_ptr.as_ptr().add(2) as *const i32).read_unaligned());
                    let w2 =
                        _mm256_set1_epi32((w_ptr.as_ptr().add(4) as *const i32).read_unaligned());
                    let w3 =
                        _mm256_set1_epi32((w_ptr.as_ptr().add(6) as *const i32).read_unaligned());
                    store_0 = self.conv_horiz_rgba_8_u8_i16(
                        bounds_start,
                        src0,
                        src1,
                        w0,
                        w1,
                        w2,
                        w3,
                        store_0,
                    );
                    store_1 = self.conv_horiz_rgba_8_u8_i16(
                        bounds_start,
                        src2,
                        src3,
                        w0,
                        w1,
                        w2,
                        w3,
                        store_1,
                    );
                    jx += 8;
                }

                while jx + 4 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm256_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                    let w1 =
                        _mm256_set1_epi32((w_ptr.as_ptr().add(2) as *const i32).read_unaligned());
                    store_0 =
                        self.conv_horiz_rgba_4_u8_i16(bounds_start, src0, src1, w0, w1, store_0);
                    store_1 =
                        self.conv_horiz_rgba_4_u8_i16(bounds_start, src2, src3, w0, w1, store_1);
                    jx += 4;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx);
                    let bounds_start = bounds.start + jx;
                    let weight0 = _mm256_set1_epi16(*w_ptr);
                    store_0 =
                        self.conv_horiz_rgba_1_u8_i16(bounds_start, src0, src1, weight0, store_0);
                    store_1 =
                        self.conv_horiz_rgba_1_u8_i16(bounds_start, src2, src3, weight0, store_1);
                    jx += 1;
                }

                let store_0 = _mm256_srai_epi32::<PRECISION>(store_0);
                let store_1 = _mm256_srai_epi32::<PRECISION>(store_1);

                let store_0 = _mm256_packus_epi32(store_0, store_0);
                let store_1 = _mm256_packus_epi32(store_1, store_1);

                let ss0 = _mm256_min_epi16(store_0, v_cut_off);
                let ss1 = _mm256_min_epi16(store_1, v_cut_off);

                let packed0 = _mm_extract_ar30::<AR_TYPE, AR_ORDER>(_mm256_castsi256_si128(ss0));
                _mm_storeu_si32(chunk0.as_mut_ptr(), packed0);
                let packed1 =
                    _mm_extract_ar30::<AR_TYPE, AR_ORDER>(_mm256_extracti128_si256::<1>(ss0));
                _mm_storeu_si32(chunk1.as_mut_ptr(), packed1);
                let packed2 = _mm_extract_ar30::<AR_TYPE, AR_ORDER>(_mm256_castsi256_si128(ss1));
                _mm_storeu_si32(chunk2.as_mut_ptr(), packed2);
                let packed3 =
                    _mm_extract_ar30::<AR_TYPE, AR_ORDER>(_mm256_extracti128_si256::<1>(ss1));
                _mm_storeu_si32(chunk3.as_mut_ptr(), packed3);
            }
        }
    }
}

pub(crate) fn avx_convolve_horizontal_rgba_rows_ar30<
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        let unit = Row1ExecutionUnit::<AR_TYPE, AR_ORDER>::default();
        unit.pass(src, dst, filter_weights);
    }
}

#[derive(Copy, Clone, Default)]
struct Row1ExecutionUnit<const AR_TYPE: usize, const AR_ORDER: usize> {}

impl<const AR_TYPE: usize, const AR_ORDER: usize> Row1ExecutionUnit<AR_TYPE, AR_ORDER> {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn conv_horiz_rgba_1_u8_i16(
        start_x: usize,
        src: &[u8],
        w0: __m128i,
        store: __m128i,
    ) -> __m128i {
        unsafe {
            let src_ptr = src.get_unchecked(start_x * 4..);
            let ld = _mm_ld1_ar30_s16::<AR_TYPE, AR_ORDER>(src_ptr);
            _mm_add_epi32(
                store,
                _mm_madd_epi16(_mm_unpacklo_epi16(ld, _mm_setzero_si128()), w0),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn conv_horiz_rgba_8_u8_i16(
        &self,
        start_x: usize,
        src: &[u8],
        w0: __m128i,
        w1: __m128i,
        w2: __m128i,
        w3: __m128i,
        store: __m128i,
    ) -> __m128i {
        unsafe {
            let src_ptr = src.get_unchecked(start_x * 4..);

            let l0 = _mm_loadu_si128(src_ptr.as_ptr() as *const _);
            let l1 = _mm_loadu_si128(src_ptr.as_ptr().add(4 * 4) as *const _);

            let rgba_pixel = _mm_unzip_4_ar30_separate::<AR_TYPE, AR_ORDER>((l0, l1));

            let sh1 = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

            let v0 = _mm_shuffle_epi8(rgba_pixel.0, sh1);
            let v1 = _mm_shuffle_epi8(rgba_pixel.1, sh1);
            let v2 = _mm_shuffle_epi8(rgba_pixel.2, sh1);
            let v3 = _mm_shuffle_epi8(rgba_pixel.3, sh1);

            let mut v = _mm_add_epi32(store, _mm_madd_epi16(v0, w0));
            v = _mm_add_epi32(v, _mm_madd_epi16(v1, w1));
            v = _mm_add_epi32(v, _mm_madd_epi16(v2, w2));
            _mm_add_epi32(v, _mm_madd_epi16(v3, w3))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn conv_horiz_rgba_4_u8_i16(
        &self,
        start_x: usize,
        src: &[u8],
        w0: __m128i,
        w1: __m128i,
        store: __m128i,
    ) -> __m128i {
        unsafe {
            let src_ptr = src.get_unchecked(start_x * 4..);

            let rgba_pixel = _mm_unzips_4_ar30_separate::<AR_TYPE, AR_ORDER>(_mm_loadu_si128(
                src_ptr.as_ptr() as *const _,
            ));

            let sh1 = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

            let v0 = _mm_shuffle_epi8(rgba_pixel.0, sh1);
            let v1 = _mm_shuffle_epi8(rgba_pixel.1, sh1);

            let v = _mm_add_epi32(store, _mm_madd_epi16(v0, w0));
            _mm_add_epi32(v, _mm_madd_epi16(v1, w1))
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn pass(&self, src: &[u8], dst: &mut [u8], filter_weights: &FilterWeights<i16>) {
        unsafe {
            const PRECISION: i32 = 16;
            const ROUNDING: i32 = 1 << (PRECISION - 1);

            let init = _mm_set1_epi32(ROUNDING);

            let v_cut_off = _mm_set1_epi32(1023);

            for ((chunk0, &bounds), weights) in dst
                .chunks_exact_mut(4)
                .zip(filter_weights.bounds.iter())
                .zip(
                    filter_weights
                        .weights
                        .chunks_exact(filter_weights.aligned_size),
                )
            {
                let mut jx = 0usize;

                let bounds_size = bounds.size;

                let mut store_0 = init;

                let src0 = src;

                while jx + 8 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                    let w1 = _mm_set1_epi32((w_ptr.as_ptr().add(2) as *const i32).read_unaligned());
                    let w2 = _mm_set1_epi32((w_ptr.as_ptr().add(4) as *const i32).read_unaligned());
                    let w3 = _mm_set1_epi32((w_ptr.as_ptr().add(6) as *const i32).read_unaligned());
                    store_0 =
                        self.conv_horiz_rgba_8_u8_i16(bounds_start, src0, w0, w1, w2, w3, store_0);
                    jx += 8;
                }

                while jx + 4 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                    let w1 = _mm_set1_epi32((w_ptr.as_ptr().add(2) as *const i32).read_unaligned());
                    store_0 = self.conv_horiz_rgba_4_u8_i16(bounds_start, src0, w0, w1, store_0);
                    jx += 4;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx);
                    let bounds_start = bounds.start + jx;
                    let weight0 = _mm_set1_epi16(*w_ptr);
                    store_0 = Self::conv_horiz_rgba_1_u8_i16(bounds_start, src0, weight0, store_0);
                    jx += 1;
                }

                let store_0 = _mm_srai_epi32::<PRECISION>(store_0);

                let store_0 = _mm_packus_epi32(store_0, store_0);

                let store_16_0 = _mm_min_epi16(store_0, v_cut_off);

                let packed0 = _mm_extract_ar30::<AR_TYPE, AR_ORDER>(store_16_0);
                _mm_storeu_si32(chunk0.as_mut_ptr(), packed0);
            }
        }
    }
}
