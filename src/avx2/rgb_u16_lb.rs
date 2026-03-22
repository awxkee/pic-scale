/*
 * Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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

#[inline(always)]
fn acc_1_dot_rgb<const D: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128i,
    store: __m128i,
    shuffle: __m128i,
) -> __m128i {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        // Load 3 u16 safely: 2 via u32 load + 1 via u16 load
        let v0 = _mm_loadu_si32(src_ptr.as_ptr().cast());
        let v1 = _mm_loadu_si16(src_ptr.get_unchecked(2..).as_ptr().cast());
        let pixel = _mm_unpacklo_epi32(v0, v1); // [r,g,b,0]
        _mm_dot16_avx_epi32::<D>(store, _mm_shuffle_epi8(pixel, shuffle), w0)
    }
}

#[inline(always)]
fn acc_2_dot_rgb<const D: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128i,
    store: __m128i,
    shuffle: __m128i,
) -> __m128i {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        // 2 pixels = 6 x u16 = 12 bytes — load as 8 bytes + 4 bytes
        let lo = _mm_loadu_si64(src_ptr.as_ptr().cast());
        let hi = _mm_loadu_si32(src_ptr.get_unchecked(4..).as_ptr().cast());
        let pixel = _mm_unpacklo_epi64(lo, hi); // [r0,g0,b0,r1,g1,b1,0,0]
        _mm_dot16_avx_epi32::<D>(store, _mm_shuffle_epi8(pixel, shuffle), w0)
    }
}

#[inline(always)]
fn acc_4_dot_rgb<const D: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m256i,
    store: __m256i,
    shuffle: __m256i,
) -> __m256i {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);

        let lo = _mm_loadu_si128(src_ptr.as_ptr().cast());
        let hi = _mm_loadu_si64(src_ptr.get_unchecked(8..).as_ptr().cast());

        let shuf = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1);
        let hi_src = _mm_alignr_epi8(hi, lo, 12);

        let lo_shuf = _mm_shuffle_epi8(lo, shuf);
        let hi_shuf = _mm_shuffle_epi8(hi_src, shuf);

        let pixel = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo_shuf), hi_shuf);
        let q_px = _mm256_shuffle_epi8(pixel, shuffle);
        _mm256_dot16_avx_epi32::<D>(store, q_px, w0)
    }
}

#[inline(always)]
fn acc_8_dot_rgb<const D: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m256i,
    w1: __m256i,
    store: __m256i,
    shuffle: __m256i,
) -> __m256i {
    let store = acc_4_dot_rgb::<D>(start_x, src, w0, store, shuffle);
    acc_4_dot_rgb::<D>(start_x + 4, src, w1, store, shuffle)
}

pub(crate) fn convolve_horizontal_rgb_avx_rows_4_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        #[cfg(feature = "avx512")]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return convolve_horizontal_rgb_avx_rows_4_lb_vn(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        }
        convolve_horizontal_rgb_avx_rows_4_lb_a(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        );
    }
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avxvnni", enable = "avx2")]
fn convolve_horizontal_rgb_avx_rows_4_lb_vn(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    Row4ExecutionHandlerRgb::<true>::default().pass(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
        bit_depth,
    );
}

#[target_feature(enable = "avx2")]
fn convolve_horizontal_rgb_avx_rows_4_lb_a(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    Row4ExecutionHandlerRgb::<false>::default().pass(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
        bit_depth,
    );
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionHandlerRgb<const D: bool> {}

impl<const D: bool> Row4ExecutionHandlerRgb<D> {
    #[inline(always)]
    fn acc_1_avx(
        &self,
        start_x: usize,
        src0: &[u16],
        src1: &[u16],
        w0: __m256i,
        store: __m256i,
        shuffle: __m256i,
    ) -> __m256i {
        unsafe {
            const CN: usize = 3;
            let s0 = src0.get_unchecked((start_x * CN)..);
            let s1 = src1.get_unchecked((start_x * CN)..);

            let v0 = _mm_loadu_si32(s0.as_ptr().cast());
            let v0b = _mm_loadu_si16(s0.get_unchecked(2..).as_ptr().cast());
            let p0 = _mm_unpacklo_epi32(v0, v0b);

            let v1 = _mm_loadu_si32(s1.as_ptr().cast());
            let v1b = _mm_loadu_si16(s1.get_unchecked(2..).as_ptr().cast());
            let p1 = _mm_unpacklo_epi32(v1, v1b);

            let pixel = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(p0), p1);
            _mm256_dot16_avx_epi32::<D>(store, _mm256_shuffle_epi8(pixel, shuffle), w0)
        }
    }

    #[inline(always)]
    fn acc_2_avx(
        &self,
        start_x: usize,
        src0: &[u16],
        src1: &[u16],
        w0: __m256i,
        store: __m256i,
        shuffle: __m256i,
    ) -> __m256i {
        unsafe {
            const CN: usize = 3;
            let s0 = src0.get_unchecked((start_x * CN)..);
            let s1 = src1.get_unchecked((start_x * CN)..);

            let lo0 = _mm_loadu_si64(s0.as_ptr() as *const u8);
            let hi0 = _mm_loadu_si32(s0.get_unchecked(4..).as_ptr().cast());
            let p0 = _mm_unpacklo_epi64(lo0, hi0);

            let lo1 = _mm_loadu_si64(s1.as_ptr() as *const u8);
            let hi1 = _mm_loadu_si32(s1.get_unchecked(4..).as_ptr().cast());
            let p1 = _mm_unpacklo_epi64(lo1, hi1);

            let pixel = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(p0), p1);
            _mm256_dot16_avx_epi32::<D>(store, _mm256_shuffle_epi8(pixel, shuffle), w0)
        }
    }

    #[inline(always)]
    fn pass(
        &self,
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) {
        unsafe {
            const CN: usize = 3;

            let v_max_colors = _mm256_set1_epi16((1 << bit_depth) - 1);

            let permute_avx_weights = _mm256_setr_epi32(0, 2, 0, 0, 1, 3, 1, 1);
            let permute_avx_weights_hi = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);

            let a_shuffle_weights_table = _mm256_setr_epi8(
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3,
            );

            // Shuffle for 2-pixel dot: interleave u16 pairs across halves
            // [r0,g0,b0,0, r1,g1,b1,0] -> dot-product friendly layout
            let a_shuffle_2_table = _mm256_setr_epi8(
                0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11, -1, -1, -1, -1, 0, 1, 6, 7, 2, 3, 8, 9, 4, 5,
                10, 11, -1, -1, -1, -1,
            );

            // Shuffle for 1-pixel: zero-extend each u16
            let a_shuffle_1_table = _mm256_setr_epi8(
                0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, -1,
                -1, 4, 5, -1, -1, -1, -1, -1, -1,
            );

            let shuffle_weights_table = _mm256_setr_epi8(
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3,
            );

            let init256 = _mm256_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                0,
                0,
                0,
                0,
                0,
            );

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
                let bounds_size = bounds.size;

                let src0 = src;
                let src1 = src0.get_unchecked(src_stride..);
                let src2 = src0.get_unchecked(src_stride * 2..);
                let src3 = src0.get_unchecked(src_stride * 3..);

                let mut astore_0 = init256;
                let mut astore_1 = init256;
                let mut astore_2 = init256;
                let mut astore_3 = init256;

                while jx + 8 <= bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let wl = _mm256_castsi128_si256(_mm_loadu_si128(w_ptr.as_ptr().cast()));
                    let w0 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(wl, permute_avx_weights),
                        a_shuffle_weights_table,
                    );
                    let w1 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(wl, permute_avx_weights_hi),
                        a_shuffle_weights_table,
                    );
                    let bounds_start = bounds.start + jx;
                    astore_0 =
                        acc_8_dot_rgb::<D>(bounds_start, src0, w0, w1, astore_0, a_shuffle_2_table);
                    astore_1 =
                        acc_8_dot_rgb::<D>(bounds_start, src1, w0, w1, astore_1, a_shuffle_2_table);
                    astore_2 =
                        acc_8_dot_rgb::<D>(bounds_start, src2, w0, w1, astore_2, a_shuffle_2_table);
                    astore_3 =
                        acc_8_dot_rgb::<D>(bounds_start, src3, w0, w1, astore_3, a_shuffle_2_table);
                    jx += 8;
                }

                while jx + 4 <= bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(
                            _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr().cast())),
                            permute_avx_weights,
                        ),
                        a_shuffle_weights_table,
                    );
                    astore_0 =
                        acc_4_dot_rgb::<D>(bounds_start, src0, w0, astore_0, a_shuffle_2_table);
                    astore_1 =
                        acc_4_dot_rgb::<D>(bounds_start, src1, w0, astore_1, a_shuffle_2_table);
                    astore_2 =
                        acc_4_dot_rgb::<D>(bounds_start, src2, w0, astore_2, a_shuffle_2_table);
                    astore_3 =
                        acc_4_dot_rgb::<D>(bounds_start, src3, w0, astore_3, a_shuffle_2_table);
                    jx += 4;
                }

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let mut store_0 = _mm256_add_epi32(
                    _mm256_permute2x128_si256::<LO_LO>(astore_0, astore_1),
                    _mm256_permute2x128_si256::<HI_HI>(astore_0, astore_1),
                );
                let mut store_1 = _mm256_add_epi32(
                    _mm256_permute2x128_si256::<LO_LO>(astore_2, astore_3),
                    _mm256_permute2x128_si256::<HI_HI>(astore_2, astore_3),
                );

                while jx + 2 <= bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let bounds_start = bounds.start + jx;
                    let ww0 = _mm_loadu_si32(w_ptr.as_ptr().cast());
                    let w0 = _mm256_shuffle_epi8(
                        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(ww0), ww0),
                        shuffle_weights_table,
                    );
                    store_0 =
                        self.acc_2_avx(bounds_start, src0, src1, w0, store_0, a_shuffle_2_table);
                    store_1 =
                        self.acc_2_avx(bounds_start, src2, src3, w0, store_1, a_shuffle_2_table);
                    jx += 2;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx);
                    let bounds_start = bounds.start + jx;
                    let ww0 = _mm_set1_epi16(*w_ptr);
                    let w0 = _mm256_shuffle_epi8(
                        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(ww0), ww0),
                        shuffle_weights_table,
                    );
                    store_0 =
                        self.acc_1_avx(bounds_start, src0, src1, w0, store_0, a_shuffle_1_table);
                    store_1 =
                        self.acc_1_avx(bounds_start, src2, src3, w0, store_1, a_shuffle_1_table);
                    jx += 1;
                }

                store_0 = _mm256_srai_epi32::<PRECISION>(store_0);
                store_1 = _mm256_srai_epi32::<PRECISION>(store_1);

                store_0 = _mm256_packus_epi32(store_0, store_0);
                store_1 = _mm256_packus_epi32(store_1, store_1);

                let v_st0 = _mm256_min_epi16(store_0, v_max_colors);
                let v_st1 = _mm256_min_epi16(store_1, v_max_colors);

                // Store 3 x u16 per pixel — use set_pixel pattern
                let lo0 = _mm256_castsi256_si128(v_st0);
                let hi0 = _mm256_extracti128_si256::<1>(v_st0);
                let lo1 = _mm256_castsi256_si128(v_st1);
                let hi1 = _mm256_extracti128_si256::<1>(v_st1);

                set_pixel_sse(chunk0, lo0);
                set_pixel_sse(chunk1, hi0);
                set_pixel_sse(chunk2, lo1);
                set_pixel_sse(chunk3, hi1);
            }
        }
    }
}

#[inline(always)]
fn set_pixel_sse(ptr: &mut [u16; 3], pixel: __m128i) {
    // Store lanes 0,1 as u32, lane 2 as u16
    unsafe {
        _mm_storeu_si32(ptr.as_mut_ptr().cast(), pixel);
        ptr[2] = _mm_extract_epi16::<2>(pixel) as u16;
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_u16lp_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        #[cfg(feature = "avx512")]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return convolve_horizontal_rgb_avx_u16_row_vn(src, dst, filter_weights, bit_depth);
        }
        convolve_horizontal_rgb_avx_u16_row_avx(src, dst, filter_weights, bit_depth);
    }
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avxvnni", enable = "avx2")]
fn convolve_horizontal_rgb_avx_u16_row_vn(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    OneRowExecutionUnitRgb::<true>::default().pass(src, dst, filter_weights, bit_depth);
}

#[target_feature(enable = "avx2")]
fn convolve_horizontal_rgb_avx_u16_row_avx(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    OneRowExecutionUnitRgb::<false>::default().pass(src, dst, filter_weights, bit_depth);
}

#[derive(Copy, Clone, Default)]
struct OneRowExecutionUnitRgb<const D: bool> {}

impl<const D: bool> OneRowExecutionUnitRgb<D> {
    #[inline(always)]
    fn pass(
        &self,
        src: &[u16],
        dst: &mut [u16],
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) {
        unsafe {
            const CN: usize = 3;

            let v_max_colors = _mm_set1_epi16((1 << bit_depth) - 1);

            let permute_avx_weights = _mm256_setr_epi32(0, 2, 0, 0, 1, 3, 1, 1);
            let permute_avx_weights_hi = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);

            let a_shuffle_weights_table = _mm256_setr_epi8(
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3,
            );

            let a_shuffle_2_table = _mm256_setr_epi8(
                0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11, -1, -1, -1, -1, 0, 1, 6, 7, 2, 3, 8, 9, 4, 5,
                10, 11, -1, -1, -1, -1,
            );

            let shuffle_2_table =
                _mm_setr_epi8(0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11, -1, -1, -1, -1);

            let shuffle_1_table =
                _mm_setr_epi8(0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, -1, -1, -1, -1);

            let shuffle_weights_table =
                _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);

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

                let mut store256 = _mm256_setr_epi32(
                    ROUNDING_CONST,
                    ROUNDING_CONST,
                    ROUNDING_CONST,
                    0,
                    0,
                    0,
                    0,
                    0,
                );

                while jx + 8 <= bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let wl = _mm256_castsi128_si256(_mm_loadu_si128(w_ptr.as_ptr().cast()));
                    let w0 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(wl, permute_avx_weights),
                        a_shuffle_weights_table,
                    );
                    let w1 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(wl, permute_avx_weights_hi),
                        a_shuffle_weights_table,
                    );
                    let bounds_start = bounds.start + jx;
                    store256 =
                        acc_8_dot_rgb::<D>(bounds_start, src, w0, w1, store256, a_shuffle_2_table);
                    jx += 8;
                }

                while jx + 4 <= bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(
                            _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr().cast())),
                            permute_avx_weights,
                        ),
                        a_shuffle_weights_table,
                    );
                    let bounds_start = bounds.start + jx;
                    store256 =
                        acc_4_dot_rgb::<D>(bounds_start, src, w0, store256, a_shuffle_2_table);
                    jx += 4;
                }

                let mut store = _mm_add_epi32(
                    _mm256_castsi256_si128(store256),
                    _mm256_extracti128_si256::<1>(store256),
                );

                while jx + 2 <= bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let bounds_start = bounds.start + jx;
                    let w0 = _mm_shuffle_epi8(
                        _mm_loadu_si32(w_ptr.as_ptr().cast()),
                        shuffle_weights_table,
                    );
                    store = acc_2_dot_rgb::<D>(bounds_start, src, w0, store, shuffle_2_table);
                    jx += 2;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx);
                    let w0 = _mm_shuffle_epi8(_mm_set1_epi16(*w_ptr), shuffle_weights_table);
                    let bounds_start = bounds.start + jx;
                    store = acc_1_dot_rgb::<D>(bounds_start, src, w0, store, shuffle_1_table);
                    jx += 1;
                }

                store = _mm_srai_epi32::<PRECISION>(store);
                let v_st = _mm_min_epi16(_mm_packus_epi32(store, store), v_max_colors);

                set_pixel_sse(dst, v_st);
            }
        }
    }
}
