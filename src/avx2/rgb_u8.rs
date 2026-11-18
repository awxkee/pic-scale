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

use crate::avx2::utils::{_mm_dot16_avx_epi32, _mm256_dot16_avx_epi32};
use crate::filter_weights::FilterWeights;
use crate::support::{PRECISION, ROUNDING_CONST};
use std::arch::x86_64::*;

pub(crate) fn convolve_horizontal_rgb_avx_rows_4<
    const EXACT_FOUR: bool,
    const BROADCAST_REMAINDER: bool,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        #[cfg(feature = "avx512")]
        {
            if std::arch::is_x86_feature_detected!("avxvnni") {
                return convolve_horizontal_rgb_avx_rows_4_vnni::<EXACT_FOUR, BROADCAST_REMAINDER>(
                    src,
                    src_stride,
                    dst,
                    dst_stride,
                    filter_weights,
                );
            }
        }
        convolve_horizontal_rgb_avx_rows_4_reg::<EXACT_FOUR, BROADCAST_REMAINDER>(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

#[target_feature(enable = "avx2")]
fn convolve_horizontal_rgb_avx_rows_4_reg<
    const EXACT_FOUR: bool,
    const BROADCAST_REMAINDER: bool,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    let unit = Row4ExecutionUnit::<false>::default();
    unit.pass::<EXACT_FOUR, BROADCAST_REMAINDER>(src, src_stride, dst, dst_stride, filter_weights);
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avx2", enable = "avxvnni")]
fn convolve_horizontal_rgb_avx_rows_4_vnni<
    const EXACT_FOUR: bool,
    const BROADCAST_REMAINDER: bool,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    let unit = Row4ExecutionUnit::<true>::default();
    unit.pass::<EXACT_FOUR, BROADCAST_REMAINDER>(src, src_stride, dst, dst_stride, filter_weights);
}

#[inline(always)]
fn load_rgb_x2(src: &[u8]) -> __m128i {
    unsafe {
        let mut rgb_pixel = _mm_setzero_si128();
        rgb_pixel = _mm_insert_epi32::<0>(rgb_pixel, (src.as_ptr() as *const i32).read_unaligned());
        rgb_pixel = _mm_insert_epi16::<2>(
            rgb_pixel,
            (src.get_unchecked(4..).as_ptr() as *const i16).read_unaligned() as i32,
        );
        rgb_pixel
    }
}

#[inline(always)]
fn load_rgb_x4(src: &[u8]) -> __m128i {
    unsafe {
        let mut rgb_pixel = _mm_loadu_si64(src.as_ptr());
        rgb_pixel = _mm_insert_epi32::<2>(
            rgb_pixel,
            (src.get_unchecked(8..).as_ptr() as *const i32).read_unaligned(),
        );
        rgb_pixel
    }
}

#[inline(always)]
fn load_distr_x4_rgb<const BROADCAST_REMAINDER: bool>(src: &[u8], shuf: __m256i) -> __m256i {
    unsafe {
        let rgb_pixel = load_rgb_x4(src);

        if BROADCAST_REMAINDER {
            // Select two pixel pairs from the same 12-byte load in separate lanes.
            let shuffle = _mm256_setr_epi8(
                0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1, 6, -1, 9, -1, 7, -1, 10,
                -1, 8, -1, 11, -1, -1, -1, -1, -1,
            );
            _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(rgb_pixel), shuffle)
        } else {
            let top_pixels = _mm_alignr_epi8::<6>(rgb_pixel, rgb_pixel);
            _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel), top_pixels),
                shuf,
            )
        }
    }
}

#[inline(always)]
fn load_distr_x8_rgb(src: &[u8]) -> (__m256i, __m256i) {
    unsafe {
        // After each broadcast, the low and high lanes select different pixel pairs.
        let shuffle_first = _mm256_setr_epi8(
            0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1, 6, -1, 9, -1, 7, -1, 10, -1,
            8, -1, 11, -1, -1, -1, -1, -1,
        );
        let shuffle_last = _mm256_setr_epi8(
            4, -1, 7, -1, 5, -1, 8, -1, 6, -1, 9, -1, -1, -1, -1, -1, 10, -1, 13, -1, 11, -1, 14,
            -1, 12, -1, 15, -1, -1, -1, -1, -1,
        );
        // The overlapping windows read exactly 24 bytes, even at the right edge.
        let first = _mm256_broadcastsi128_si256(_mm_loadu_si128(src.as_ptr().cast()));
        let last = _mm256_broadcastsi128_si256(_mm_loadu_si128(src.as_ptr().add(8).cast()));
        (
            _mm256_shuffle_epi8(first, shuffle_first),
            _mm256_shuffle_epi8(last, shuffle_last),
        )
    }
}

#[inline(always)]
fn load_distr_x4_rgb_wide<const LAST: bool>(src: &[u8]) -> __m256i {
    unsafe {
        let shuffle = if LAST {
            _mm256_setr_epi8(
                4, -1, 7, -1, 5, -1, 8, -1, 6, -1, 9, -1, -1, -1, -1, -1, 10, -1, 13, -1, 11, -1,
                14, -1, 12, -1, 15, -1, -1, -1, -1, -1,
            )
        } else {
            _mm256_setr_epi8(
                0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1, 6, -1, 9, -1, 7, -1, 10,
                -1, 8, -1, 11, -1, -1, -1, -1, -1,
            )
        };
        _mm256_shuffle_epi8(
            _mm256_broadcastsi128_si256(_mm_loadu_si128(src.as_ptr().cast())),
            shuffle,
        )
    }
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionUnit<const HAS_DOT: bool> {}

impl<const HAS_DOT: bool> Row4ExecutionUnit<HAS_DOT> {
    #[inline(always)]
    fn add_one_weight(
        &self,
        start_x: usize,
        src0: &[u8],
        src1: &[u8],
        weight0: __m256i,
        store_0: __m256i,
    ) -> __m256i {
        unsafe {
            const CN: usize = 3;
            let src_ptr0 = src0.get_unchecked((start_x * CN)..).as_ptr();
            let src_ptr1 = src1.get_unchecked((start_x * CN)..).as_ptr();
            let base_pixel0 = _mm_loadu_si16(src_ptr0);
            let base_pixel1 = _mm_loadu_si16(src_ptr1);
            let m_vl0 = _mm_insert_epi8::<2>(base_pixel0, src_ptr0.add(2).read_unaligned() as i32);
            let m_vl1 = _mm_insert_epi8::<2>(base_pixel1, src_ptr1.add(2).read_unaligned() as i32);
            let lo0 = _mm_unpacklo_epi8(m_vl0, _mm_setzero_si128());
            let lo1 = _mm_unpacklo_epi8(m_vl1, _mm_setzero_si128());
            let px = _mm_unpacklo_epi64(lo0, lo1);
            _mm256_dot16_avx_epi32::<HAS_DOT>(store_0, _mm256_cvtepu16_epi32(px), weight0)
        }
    }

    #[inline(always)]
    fn pass<const EXACT_FOUR: bool, const BROADCAST_REMAINDER: bool>(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
    ) {
        unsafe {
            const CN: usize = 3;

            let shuffle_lo = _mm256_setr_epi8(
                0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1, 0, -1, 3, -1, 1, -1, 4,
                -1, 2, -1, 5, -1, -1, -1, -1, -1,
            );

            let shuffle_hi = _mm256_setr_epi8(
                6, -1, 9, -1, 7, -1, 10, -1, 8, -1, 11, -1, -1, -1, -1, -1, 6, -1, 9, -1, 7, -1,
                10, -1, 8, -1, 11, -1, -1, -1, -1, -1,
            );

            let weights_idx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

            let shuffle_weights = _mm256_setr_epi8(
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, -1, -1, -1, -1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
                2, 3, -1, -1, -1, -1,
            );
            let shuffle_pixels_4 = _mm256_setr_epi8(
                0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1, 0, -1, 3, -1, 1, -1, 4,
                -1, 2, -1, 5, -1, -1, -1, -1, -1,
            );
            let weights_idx23 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);

            let vld = _mm256_set1_epi32(ROUNDING_CONST);

            let (row0_ref, rest) = dst.split_at_mut(dst_stride);
            let (row1_ref, rest) = rest.split_at_mut(dst_stride);
            let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

            let iter_row0 = row0_ref.as_chunks_mut::<CN>().0.iter_mut();
            let iter_row1 = row1_ref.as_chunks_mut::<CN>().0.iter_mut();
            let iter_row2 = row2_ref.as_chunks_mut::<CN>().0.iter_mut();
            let iter_row3 = row3_ref.as_chunks_mut::<CN>().0.iter_mut();

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

                if EXACT_FOUR && bounds.size == 4 {
                    let src_offset = bounds.start * CN;
                    let w_ptr = weights.as_ptr().cast::<i32>();
                    let w01 = _mm256_set1_epi32(w_ptr.read_unaligned());
                    let w23 = _mm256_set1_epi32(w_ptr.add(1).read_unaligned());
                    let rgb0 = load_rgb_x4(src0.get_unchecked(src_offset..));
                    let rgb1 = load_rgb_x4(src1.get_unchecked(src_offset..));
                    let rgb2 = load_rgb_x4(src2.get_unchecked(src_offset..));
                    let rgb3 = load_rgb_x4(src3.get_unchecked(src_offset..));
                    // Keep two rows in each vector while applying both coefficient pairs.
                    let rows_01 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb0), rgb1);
                    let rows_23 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb2), rgb3);
                    store_0 = _mm256_dot16_avx_epi32::<HAS_DOT>(
                        store_0,
                        _mm256_shuffle_epi8(rows_01, shuffle_lo),
                        w01,
                    );
                    store_0 = _mm256_dot16_avx_epi32::<HAS_DOT>(
                        store_0,
                        _mm256_shuffle_epi8(rows_01, shuffle_hi),
                        w23,
                    );
                    store_1 = _mm256_dot16_avx_epi32::<HAS_DOT>(
                        store_1,
                        _mm256_shuffle_epi8(rows_23, shuffle_lo),
                        w01,
                    );
                    store_1 = _mm256_dot16_avx_epi32::<HAS_DOT>(
                        store_1,
                        _mm256_shuffle_epi8(rows_23, shuffle_hi),
                        w23,
                    );
                    jx = 4;
                } else if bounds.size > 4 {
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

                    // Full 12-tap filters read exactly 36 RGB bytes in three windows.
                    if BROADCAST_REMAINDER && bounds.size == 12 {
                        let full_weights =
                            _mm256_castsi128_si256(_mm_loadu_si128(weights.as_ptr().cast()));
                        let w0 = _mm256_shuffle_epi8(
                            _mm256_permutevar8x32_epi32(full_weights, weights_idx),
                            shuffle_weights,
                        );
                        let w1 = _mm256_shuffle_epi8(
                            _mm256_permutevar8x32_epi32(full_weights, weights_idx23),
                            shuffle_weights,
                        );
                        let last_weights =
                            _mm256_castsi128_si256(_mm_loadu_si64(weights.as_ptr().add(8).cast()));
                        let w2 = _mm256_shuffle_epi8(
                            _mm256_permutevar8x32_epi32(last_weights, weights_idx),
                            shuffle_weights,
                        );
                        let start = bounds.start * CN;
                        let p0 = load_distr_x4_rgb_wide::<false>(src0.get_unchecked(start..));
                        let p1 = load_distr_x4_rgb_wide::<false>(src1.get_unchecked(start..));
                        let p2 = load_distr_x4_rgb_wide::<false>(src2.get_unchecked(start..));
                        let p3 = load_distr_x4_rgb_wide::<false>(src3.get_unchecked(start..));
                        store0 = _mm256_dot16_avx_epi32::<HAS_DOT>(store0, p0, w0);
                        store1 = _mm256_dot16_avx_epi32::<HAS_DOT>(store1, p1, w0);
                        store2 = _mm256_dot16_avx_epi32::<HAS_DOT>(store2, p2, w0);
                        store3 = _mm256_dot16_avx_epi32::<HAS_DOT>(store3, p3, w0);
                        let p0 = load_distr_x4_rgb_wide::<false>(src0.get_unchecked(start + 12..));
                        let p1 = load_distr_x4_rgb_wide::<false>(src1.get_unchecked(start + 12..));
                        let p2 = load_distr_x4_rgb_wide::<false>(src2.get_unchecked(start + 12..));
                        let p3 = load_distr_x4_rgb_wide::<false>(src3.get_unchecked(start + 12..));
                        store0 = _mm256_dot16_avx_epi32::<HAS_DOT>(store0, p0, w1);
                        store1 = _mm256_dot16_avx_epi32::<HAS_DOT>(store1, p1, w1);
                        store2 = _mm256_dot16_avx_epi32::<HAS_DOT>(store2, p2, w1);
                        store3 = _mm256_dot16_avx_epi32::<HAS_DOT>(store3, p3, w1);
                        let p0 = load_distr_x4_rgb_wide::<true>(src0.get_unchecked(start + 20..));
                        let p1 = load_distr_x4_rgb_wide::<true>(src1.get_unchecked(start + 20..));
                        let p2 = load_distr_x4_rgb_wide::<true>(src2.get_unchecked(start + 20..));
                        let p3 = load_distr_x4_rgb_wide::<true>(src3.get_unchecked(start + 20..));
                        store0 = _mm256_dot16_avx_epi32::<HAS_DOT>(store0, p0, w2);
                        store1 = _mm256_dot16_avx_epi32::<HAS_DOT>(store1, p1, w2);
                        store2 = _mm256_dot16_avx_epi32::<HAS_DOT>(store2, p2, w2);
                        store3 = _mm256_dot16_avx_epi32::<HAS_DOT>(store3, p3, w2);
                        jx = 12;
                    } else {
                        while jx + 8 <= bounds.size {
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

                            let bounds_start = (bounds.start + jx) * CN;

                            let rgb_pixel_0 = load_distr_x8_rgb(src0.get_unchecked(bounds_start..));
                            let rgb_pixel_1 = load_distr_x8_rgb(src1.get_unchecked(bounds_start..));
                            let rgb_pixel_2 = load_distr_x8_rgb(src2.get_unchecked(bounds_start..));
                            let rgb_pixel_3 = load_distr_x8_rgb(src3.get_unchecked(bounds_start..));

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

                        while jx + 4 <= bounds.size {
                            let w_ptr = weights.get_unchecked(jx..);
                            let weights = _mm256_shuffle_epi8(
                                _mm256_permutevar8x32_epi32(
                                    _mm256_castsi128_si256(_mm_loadu_si64(
                                        w_ptr.as_ptr() as *const u8
                                    )),
                                    weights_idx,
                                ),
                                shuffle_weights,
                            );
                            let bounds_start = (bounds.start + jx) * CN;

                            let rgb_pixel_0 = load_distr_x4_rgb::<BROADCAST_REMAINDER>(
                                src0.get_unchecked(bounds_start..),
                                shuffle_pixels_4,
                            );
                            let rgb_pixel_1 = load_distr_x4_rgb::<BROADCAST_REMAINDER>(
                                src1.get_unchecked(bounds_start..),
                                shuffle_pixels_4,
                            );
                            let rgb_pixel_2 = load_distr_x4_rgb::<BROADCAST_REMAINDER>(
                                src2.get_unchecked(bounds_start..),
                                shuffle_pixels_4,
                            );
                            let rgb_pixel_3 = load_distr_x4_rgb::<BROADCAST_REMAINDER>(
                                src3.get_unchecked(bounds_start..),
                                shuffle_pixels_4,
                            );

                            store0 =
                                _mm256_dot16_avx_epi32::<HAS_DOT>(store0, rgb_pixel_0, weights);
                            store1 =
                                _mm256_dot16_avx_epi32::<HAS_DOT>(store1, rgb_pixel_1, weights);
                            store2 =
                                _mm256_dot16_avx_epi32::<HAS_DOT>(store2, rgb_pixel_2, weights);
                            store3 =
                                _mm256_dot16_avx_epi32::<HAS_DOT>(store3, rgb_pixel_3, weights);

                            jx += 4;
                        }
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

                while jx + 2 <= bounds.size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let bounds_start = (bounds.start + jx) * CN;
                    let weight01 =
                        _mm256_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());

                    let rgb_pixel_0 = load_rgb_x2(src0.get_unchecked(bounds_start..));
                    let rgb_pixel_1 = load_rgb_x2(src1.get_unchecked(bounds_start..));
                    let rgb_pixel_2 = load_rgb_x2(src2.get_unchecked(bounds_start..));
                    let rgb_pixel_3 = load_rgb_x2(src3.get_unchecked(bounds_start..));

                    let px0 = _mm256_inserti128_si256::<1>(
                        _mm256_castsi128_si256(rgb_pixel_0),
                        rgb_pixel_1,
                    );
                    let px1 = _mm256_inserti128_si256::<1>(
                        _mm256_castsi128_si256(rgb_pixel_2),
                        rgb_pixel_3,
                    );

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
}

pub(crate) fn convolve_horizontal_rgb_avx_row_one<const BROADCAST_REMAINDER: bool>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        #[cfg(feature = "avx512")]
        {
            if std::arch::is_x86_feature_detected!("avxvnni") {
                return convolve_horizontal_rgb_avx_row_one_vnni::<BROADCAST_REMAINDER>(
                    src,
                    dst,
                    filter_weights,
                );
            }
        }
        convolve_horizontal_rgb_avx_row_one_reg::<BROADCAST_REMAINDER>(src, dst, filter_weights);
    }
}

#[target_feature(enable = "avx2")]
fn convolve_horizontal_rgb_avx_row_one_reg<const BROADCAST_REMAINDER: bool>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    let unit = Row1Execution::<false>::default();
    unit.pass::<BROADCAST_REMAINDER>(src, dst, filter_weights);
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avx2", enable = "avxvnni")]
fn convolve_horizontal_rgb_avx_row_one_vnni<const BROADCAST_REMAINDER: bool>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    let unit = Row1Execution::<true>::default();
    unit.pass::<BROADCAST_REMAINDER>(src, dst, filter_weights);
}

#[inline(always)]
fn add_one_weight<const HAS_DOT: bool>(
    start_x: usize,
    src: &[u8],
    weight0: __m128i,
    store_0: __m128i,
) -> __m128i {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..).as_ptr();
        let base_pixel = _mm_loadu_si16(src_ptr);
        let m_vl = _mm_insert_epi8::<2>(base_pixel, src_ptr.add(2).read_unaligned() as i32);
        let lo = _mm_unpacklo_epi8(m_vl, _mm_setzero_si128());
        _mm_dot16_avx_epi32::<HAS_DOT>(
            store_0,
            _mm_unpacklo_epi16(lo, _mm_setzero_si128()),
            weight0,
        )
    }
}

#[derive(Copy, Clone, Default)]
struct Row1Execution<const HAS_DOT: bool> {}

impl<const HAS_DOT: bool> Row1Execution<HAS_DOT> {
    #[inline(always)]
    fn pass<const BROADCAST_REMAINDER: bool>(
        &self,
        src: &[u8],
        dst: &mut [u8],
        filter_weights: &FilterWeights<i16>,
    ) {
        unsafe {
            const CN: usize = 3;

            let shuffle_lo =
                _mm_setr_epi8(0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1);

            let weights_idx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
            let weights_idx23 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);

            let shuffle_weights01 = _mm256_setr_epi8(
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, -1, -1, -1, -1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
                2, 3, -1, -1, -1, -1,
            );
            let shuffle_pixels_4 = _mm256_setr_epi8(
                0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1, 0, -1, 3, -1, 1, -1, 4,
                -1, 2, -1, 5, -1, -1, -1, -1, -1,
            );

            for ((dst, bounds), weights) in dst
                .as_chunks_mut::<CN>()
                .0
                .iter_mut()
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

                if BROADCAST_REMAINDER && bounds.size == 12 {
                    let full_weights =
                        _mm256_castsi128_si256(_mm_loadu_si128(weights.as_ptr().cast()));
                    let w0 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(full_weights, weights_idx),
                        shuffle_weights01,
                    );
                    let w1 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(full_weights, weights_idx23),
                        shuffle_weights01,
                    );
                    let last_weights =
                        _mm256_castsi128_si256(_mm_loadu_si64(weights.as_ptr().add(8).cast()));
                    let w2 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(last_weights, weights_idx),
                        shuffle_weights01,
                    );
                    let start = bounds.start * CN;
                    let p = load_distr_x4_rgb_wide::<false>(src.get_unchecked(start..));
                    store = _mm256_dot16_avx_epi32::<HAS_DOT>(store, p, w0);
                    let p = load_distr_x4_rgb_wide::<false>(src.get_unchecked(start + 12..));
                    store = _mm256_dot16_avx_epi32::<HAS_DOT>(store, p, w1);
                    let p = load_distr_x4_rgb_wide::<true>(src.get_unchecked(start + 20..));
                    store = _mm256_dot16_avx_epi32::<HAS_DOT>(store, p, w2);
                    jx = 12;
                } else {
                    while jx + 8 <= bounds.size {
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
                        let src_ptr_0 = src.get_unchecked((bounds_start * CN)..);

                        let (first_4, second_4) = load_distr_x8_rgb(src_ptr_0);

                        store = _mm256_dot16_avx_epi32::<HAS_DOT>(store, first_4, w0);
                        store = _mm256_dot16_avx_epi32::<HAS_DOT>(store, second_4, w1);

                        jx += 8;
                    }

                    while jx + 4 <= bounds.size {
                        let w_ptr = weights.get_unchecked(jx..);
                        let weights = _mm256_shuffle_epi8(
                            _mm256_permutevar8x32_epi32(
                                _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const u8)),
                                weights_idx,
                            ),
                            shuffle_weights01,
                        );

                        let bounds_start = bounds.start + jx;
                        let src_ptr_0 = src.get_unchecked((bounds_start * CN)..);

                        let rgb_pixel =
                            load_distr_x4_rgb::<BROADCAST_REMAINDER>(src_ptr_0, shuffle_pixels_4);
                        store = _mm256_dot16_avx_epi32::<HAS_DOT>(store, rgb_pixel, weights);
                        jx += 4;
                    }
                }

                let mut store = _mm_add_epi32(
                    _mm256_castsi256_si128(store),
                    _mm256_extracti128_si256::<1>(store),
                );

                while jx + 2 <= bounds.size {
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
}

#[cfg(test)]
mod tests {
    use super::{
        convolve_horizontal_rgb_avx_row_one, convolve_horizontal_rgb_avx_row_one_reg,
        convolve_horizontal_rgb_avx_rows_4, convolve_horizontal_rgb_avx_rows_4_reg,
    };
    use crate::filter_weights::{FilterBounds, FilterWeights};
    use crate::handler_provider::{handle_fixed_row_u8, handle_fixed_rows_4_u8};
    use crate::support::PRECISION;
    use crate::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy};

    fn check_odd_tap_tails(four_rows: bool) {
        for taps in (1usize..=17).step_by(2) {
            for start in [0, 1, 3] {
                let width = start + taps;
                let src_stride = width * 3 + 5;
                let dst_stride = 8;
                let rows = if four_rows { 4 } else { 1 };
                let mut src = vec![0xd3; src_stride * rows];
                for y in 0..rows {
                    for x in 0..width {
                        let offset = y * src_stride + x * 3;
                        src[offset..offset + 3].copy_from_slice(&[
                            ((17 + x * 31 + y * 47) % 251) as u8,
                            ((99 + x * 71 + y * 13) % 253) as u8,
                            ((153 + x * 19 + y * 53) % 255) as u8,
                        ]);
                    }
                }
                // Isolate the final tap after each combination of 8/4/2-pixel
                // loops. Its R/G must come from the same pixel as its blue.
                let aligned_size = taps.next_multiple_of(8);
                let mut weights = vec![0; aligned_size];
                weights[taps - 1] = 1 << (PRECISION - 1);
                let filter = FilterWeights::new(
                    weights,
                    taps,
                    aligned_size,
                    1,
                    taps as i32,
                    vec![FilterBounds::new(start, taps)],
                );
                let mut expected = vec![0xcc; dst_stride * rows];
                let mut actual = expected.clone();
                if four_rows {
                    handle_fixed_rows_4_u8::<3>(
                        &src,
                        src_stride,
                        &mut expected,
                        dst_stride,
                        &filter,
                        0,
                    );
                    unsafe {
                        convolve_horizontal_rgb_avx_rows_4_reg::<false, false>(
                            &src,
                            src_stride,
                            &mut actual,
                            dst_stride,
                            &filter,
                        );
                    }
                } else {
                    handle_fixed_row_u8::<3>(&src, &mut expected, &filter, 0);
                    unsafe {
                        convolve_horizontal_rgb_avx_row_one_reg::<false>(
                            &src,
                            &mut actual,
                            &filter,
                        );
                    }
                }
                assert_eq!(actual, expected, "taps={taps} start={start} rows={rows}");
            }
        }
    }

    #[test]
    fn rgb_single_row_odd_tap_tail_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        check_odd_tap_tails(false);
    }

    #[test]
    fn rgb_four_rows_odd_tap_tail_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        check_odd_tap_tails(true);
    }

    #[test]
    fn rgb_packed_tap_blocks_match_scalar_at_row_end() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        for taps in 4..=32 {
            for padding in [0, 5] {
                let src_row_len = (taps + 2) * 3;
                let src_stride = src_row_len + padding;
                let src: Vec<u8> = (0..3 * src_stride + src_row_len)
                    .map(|i| (i * 101 + i / src_stride * 11) as u8)
                    .collect();
                let bounds = if taps == 12 {
                    // The selected plan can have any clipped tap count at the image edges.
                    (1..=taps)
                        .map(|size| FilterBounds::new(taps + 2 - size, size))
                        .collect::<Vec<_>>()
                } else {
                    vec![
                        FilterBounds::new(0, 3),
                        FilterBounds::new(1, taps - 1),
                        FilterBounds::new(2, taps),
                    ]
                };
                let dst_row_len = bounds.len() * 3;
                let dst_stride = dst_row_len + padding;
                let weights = (0..taps * bounds.len())
                    .map(|i| (8293 + i * 10949) as i16)
                    .collect();
                let filter =
                    FilterWeights::new(weights, taps, taps, bounds.len(), taps as i32, bounds);

                let mut expected = vec![0xd3; 3 * dst_stride + dst_row_len];
                let mut actual = expected.clone();
                handle_fixed_rows_4_u8::<3>(
                    &src,
                    src_stride,
                    &mut expected,
                    dst_stride,
                    &filter,
                    0,
                );
                let convolve: fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i16>, u32) =
                    if taps == 4 {
                        convolve_horizontal_rgb_avx_rows_4::<true, false>
                    } else if taps == 12 {
                        convolve_horizontal_rgb_avx_rows_4::<false, true>
                    } else {
                        convolve_horizontal_rgb_avx_rows_4::<false, false>
                    };
                convolve(&src, src_stride, &mut actual, dst_stride, &filter, 0);
                assert_eq!(
                    actual, expected,
                    "four rows, taps={taps}, padding={padding}"
                );

                let src = &src[3 * src_stride..];
                let mut expected = vec![0xd3; dst_row_len];
                let mut actual = expected.clone();
                handle_fixed_row_u8::<3>(src, &mut expected, &filter, 0);
                let convolve = if taps == 12 {
                    convolve_horizontal_rgb_avx_row_one::<true>
                } else {
                    convolve_horizontal_rgb_avx_row_one::<false>
                };
                convolve(src, &mut actual, &filter, 0);
                assert_eq!(actual, expected, "one row, taps={taps}, padding={padding}");
            }
        }
    }

    #[test]
    fn hermite_rgb_issue_165() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let scaler =
            Scaler::new(ResamplingFunction::Hermite).set_threading_policy(ThreadingPolicy::Single);
        for height in [1, 4, 5, 9] {
            let pixels = [255u8, 0, 0, 0, 255, 0, 0, 0, 255].repeat(height);
            let src = ImageStore::<u8, 3>::from_slice(&pixels, 3, height).unwrap();
            for width in [2, 1] {
                let mut dst = ImageStoreMut::<u8, 3>::alloc(width, height);
                scaler
                    .plan_rgb_resampling(src.size(), dst.size())
                    .unwrap()
                    .resample(&src, &mut dst)
                    .unwrap();
                let expected: &[u8] = if width == 2 {
                    &[166, 89, 0, 0, 89, 166]
                } else {
                    &[76, 103, 76]
                };
                for row in dst.buffer.borrow().chunks_exact(width * 3) {
                    assert_eq!(row, expected, "3x{height} -> {width}x{height}");
                }
            }
        }
    }

    fn avx12_available() -> bool {
        let found = std::arch::is_x86_feature_detected!("avx2");
        assert!(
            found || std::env::var_os("PIC_SCALE_REQUIRE_AVX2").is_none(),
            "AVX2 required but unavailable"
        );
        found
    }

    // Q15 oracle: one rounding bias, full-width accumulation, then clamp.
    fn avx12_scalar(
        src: &[u8],
        ss: usize,
        ds: usize,
        width: usize,
        f: &FilterWeights<i16>,
    ) -> Vec<u8> {
        let mut out = vec![0xd3; ds * 3 + width * 3];
        for row in 0..4 {
            for x in 0..width {
                for c in 0..3 {
                    let b = &f.bounds[x];
                    let mut sum = 16384i64;
                    for j in 0..b.size {
                        sum += i64::from(src[row * ss + (b.start + j) * 3 + c])
                            * i64::from(f.weights[x * f.aligned_size + j]);
                    }
                    out[row * ds + x * 3 + c] = (sum >> 15).clamp(0, 255) as u8;
                }
            }
        }
        out
    }

    #[test]
    fn avx12_signed_coefficients_clipped_tight_rows_match_independent_scalar() {
        if !avx12_available() {
            return;
        }
        let mut cases = 0;
        for aligned in [12usize, 17] {
            for src_padding in [0usize, 1, 5] {
                for dst_padding in [0usize, 7] {
                    for prefix in [0usize, 1, 7] {
                        for width in [1usize, 3, 7, 13] {
                            let src_pixels = 12 + prefix;
                            let src_row_len = src_pixels * 3;
                            let ss = src_row_len + src_padding;
                            let mut storage = vec![0xadu8; prefix + ss * 3 + src_row_len];
                            let src = &mut storage[prefix..];
                            for row in 0..4 {
                                for x in 0..src_pixels {
                                    for c in 0..3 {
                                        src[row * ss + x * 3 + c] =
                                            ((row * 67 + x * 29 + c * 43 + x * c * 11) % 256) as u8;
                                    }
                                }
                            }
                            let bounds: Vec<_> = (0..width)
                                .map(|x| {
                                    let size = 13 - width + x;
                                    FilterBounds::new(src_pixels - size, size)
                                })
                                .collect();
                            let ds = width * 3 + dst_padding;
                            for pattern in 0..15 {
                                let mut weights = vec![32767i16; aligned * width];
                                for x in 0..width {
                                    for j in 0..bounds[x].size {
                                        weights[x * aligned + j] = match pattern {
                                            0 => (8251usize + j * 10949 + x * 8311) as i16,
                                            1 => {
                                                if j % 2 == 0 {
                                                    i16::MAX
                                                } else {
                                                    i16::MIN
                                                }
                                            }
                                            2 => i16::MAX,
                                            _ => {
                                                if j == pattern - 3 {
                                                    16384
                                                } else {
                                                    0
                                                }
                                            }
                                        };
                                    }
                                }
                                let f = FilterWeights::new(
                                    weights,
                                    12,
                                    aligned,
                                    width,
                                    12,
                                    bounds.clone(),
                                );
                                let expected = avx12_scalar(src, ss, ds, width, &f);
                                for specialized in [false, true] {
                                    let mut guarded = vec![0xd3; expected.len() + 8];
                                    if specialized {
                                        unsafe {
                                            convolve_horizontal_rgb_avx_rows_4_reg::<false, true>(
                                                src,
                                                ss,
                                                &mut guarded[3..expected.len() + 3],
                                                ds,
                                                &f,
                                            )
                                        }
                                    } else {
                                        unsafe {
                                            convolve_horizontal_rgb_avx_rows_4_reg::<false, false>(
                                                src,
                                                ss,
                                                &mut guarded[3..expected.len() + 3],
                                                ds,
                                                &f,
                                            )
                                        }
                                    }
                                    assert_eq!(
                                        &guarded[3..expected.len() + 3],
                                        &expected,
                                        "4rows aligned={aligned} srcpad={src_padding} dstpad={dst_padding} prefix={prefix} width={width} pattern={pattern} special={specialized}"
                                    );
                                    assert!(
                                        guarded[..3]
                                            .iter()
                                            .chain(&guarded[expected.len() + 3..])
                                            .all(|&x| x == 0xd3)
                                    );
                                    for row in 0..4 {
                                        let mut guarded = vec![0xd3; width * 3 + 8];
                                        let source = &src[row * ss..][..src_row_len];
                                        if specialized {
                                            unsafe {
                                                convolve_horizontal_rgb_avx_row_one_reg::<true>(
                                                    source,
                                                    &mut guarded[3..width * 3 + 3],
                                                    &f,
                                                )
                                            }
                                        } else {
                                            unsafe {
                                                convolve_horizontal_rgb_avx_row_one_reg::<false>(
                                                    source,
                                                    &mut guarded[3..width * 3 + 3],
                                                    &f,
                                                )
                                            }
                                        }
                                        assert_eq!(
                                            &guarded[3..width * 3 + 3],
                                            &expected[row * ds..][..width * 3],
                                            "1row row={row} aligned={aligned} srcpad={src_padding} dstpad={dst_padding} prefix={prefix} width={width} pattern={pattern} special={specialized}"
                                        );
                                        assert!(
                                            guarded[..3]
                                                .iter()
                                                .chain(&guarded[width * 3 + 3..])
                                                .all(|&x| x == 0xd3)
                                        );
                                    }
                                    cases += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        eprintln!("AVX12 Q15: {cases} matrices, each4row+four1row callbacks exact");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn avx12_full_and_clipped_source_ends_at_guard_page() {
        if !avx12_available() {
            return;
        }
        use std::ffi::c_void;
        unsafe extern "C" {
            fn mmap(
                addr: *mut c_void,
                len: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                off: isize,
            ) -> *mut c_void;
            fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
            fn munmap(addr: *mut c_void, len: usize) -> i32;
            fn getpagesize() -> i32;
        }
        unsafe {
            let page = getpagesize() as usize;
            assert!(page >= 144);
            // Linux: PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS.
            let memory = mmap(std::ptr::null_mut(), page * 2, 1 | 2, 2 | 0x20, -1, 0);
            assert_ne!(memory as isize, -1);
            assert_eq!(mprotect(memory.cast::<u8>().add(page).cast(), page, 0), 0);
            for taps in 1..=12 {
                let ss = taps * 3;
                let len = ss * 4;
                let p = memory.cast::<u8>().add(page - len);
                for i in 0..len {
                    p.add(i).write((17 + i * 43 + i / ss * 11) as u8);
                }
                let src = std::slice::from_raw_parts(p, len);
                for tap in 0..taps {
                    let mut w = vec![0i16; 12];
                    w[tap] = 16384;
                    let f = FilterWeights::new(w, 12, 12, 1, 12, vec![FilterBounds::new(0, taps)]);
                    let expected = avx12_scalar(src, ss, 3, 1, &f);
                    let mut actual = vec![0xd3; 12];
                    convolve_horizontal_rgb_avx_rows_4_reg::<false, true>(
                        src,
                        ss,
                        &mut actual,
                        3,
                        &f,
                    );
                    assert_eq!(actual, expected, "guard4rows taps={taps},isolated={tap}");
                    let mut actual = vec![0xd3; 3];
                    convolve_horizontal_rgb_avx_row_one_reg::<true>(
                        &src[ss * 3..],
                        &mut actual,
                        &f,
                    );
                    assert_eq!(
                        &actual,
                        &expected[9..],
                        "guard1row taps={taps},isolated={tap}"
                    );
                }
            }
            assert_eq!(munmap(memory, page * 2), 0);
        }
    }
}
