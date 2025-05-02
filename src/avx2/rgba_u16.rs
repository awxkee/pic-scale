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
use crate::avx2::utils::{_mm256_prefer_fma_ps, _mm_prefer_fma_ps};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn conv_horiz_rgba_1_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    store: __m128,
) -> __m128 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel = _mm_loadu_si64(src_ptr.as_ptr() as *const u8);
    _mm_prefer_fma_ps::<FMA>(
        store,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel, _mm_setzero_si128())),
        w0,
    )
}

#[inline(always)]
unsafe fn conv_horiz_rgba_2_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    w1: __m128,
    store: __m128,
) -> __m128 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);

    let acc = _mm_prefer_fma_ps::<FMA>(
        store,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel, _mm_setzero_si128())),
        w0,
    );
    _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgba_pixel, _mm_setzero_si128())),
        w1,
    )
}

#[inline]
unsafe fn conv_horiz_rgba_4_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m256,
    w1: __m256,
    store: __m256,
) -> __m256 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = _mm256_loadu_si256(src_ptr.as_ptr() as *const __m256i);

    let acc = _mm256_prefer_fma_ps::<FMA>(
        store,
        _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rgba_pixel, _mm256_setzero_si256())),
        w0,
    );
    _mm256_prefer_fma_ps::<FMA>(
        acc,
        _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rgba_pixel, _mm256_setzero_si256())),
        w1,
    )
}

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w01: __m256,
    w23: __m256,
    w45: __m256,
    w67: __m256,
    store: __m256,
) -> __m256 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let z = _mm256_setzero_si256();

    let rgba_pixel0 = _mm256_loadu_si256(src_ptr.as_ptr() as *const _);
    let rgba_pixel1 = _mm256_loadu_si256(src_ptr.get_unchecked(16..).as_ptr() as *const _);

    let mut acc = _mm256_prefer_fma_ps::<FMA>(
        store,
        _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rgba_pixel1, z)),
        w67,
    );
    acc = _mm256_prefer_fma_ps::<FMA>(
        acc,
        _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rgba_pixel1, z)),
        w45,
    );
    acc = _mm256_prefer_fma_ps::<FMA>(
        acc,
        _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rgba_pixel0, z)),
        w23,
    );
    _mm256_prefer_fma_ps::<FMA>(
        acc,
        _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rgba_pixel0, z)),
        w01,
    )
}

pub(crate) fn convolve_horizontal_rgba_avx_rows_4_u16_f(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        if std::arch::is_x86_feature_detected!("fma") {
            convolve_horizontal_rgba_avx_rows_4_u16_fma(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        } else {
            convolve_horizontal_rgba_avx_rows_4_u16_def(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_rgba_avx_rows_4_u16_def(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    let unit = Row4ExecutionHandler::<false>::default();
    unit.pass(src, src_stride, dst, dst_stride, filter_weights, bit_depth);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_rgba_avx_rows_4_u16_fma(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    let unit = Row4ExecutionHandler::<true>::default();
    unit.pass(src, src_stride, dst, dst_stride, filter_weights, bit_depth);
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionHandler<const FMA: bool> {}

impl<const FMA: bool> Row4ExecutionHandler<FMA> {
    #[inline(always)]
    unsafe fn rgba_2_u16_avx(
        &self,
        start_x: usize,
        src0: &[u16],
        src1: &[u16],
        w0: __m256,
        w1: __m256,
        store: __m256,
    ) -> __m256 {
        const COMPONENTS: usize = 4;
        let src_ptr0 = src0.get_unchecked((start_x * COMPONENTS)..);
        let src_ptr1 = src1.get_unchecked((start_x * COMPONENTS)..);

        let rgba_pixel0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(
            src_ptr0.as_ptr() as *const _,
        )));
        let rgba_pixel1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(
            src_ptr1.as_ptr() as *const _,
        )));

        let lo = _mm256_insertf128_ps::<1>(
            _mm256_castps128_ps256(_mm256_castps256_ps128(rgba_pixel0)),
            _mm256_castps256_ps128(rgba_pixel1),
        );
        let hi = _mm256_insertf128_ps::<1>(
            _mm256_castps128_ps256(_mm256_extractf128_ps::<1>(rgba_pixel0)),
            _mm256_extractf128_ps::<1>(rgba_pixel1),
        );

        let acc = _mm256_prefer_fma_ps::<FMA>(store, lo, w0);
        _mm256_prefer_fma_ps::<FMA>(acc, hi, w1)
    }

    #[inline(always)]
    unsafe fn rgba_1_u16(
        &self,
        start_x: usize,
        src0: &[u16],
        src1: &[u16],
        w0: __m256,
        store: __m256,
    ) -> __m256 {
        const COMPONENTS: usize = 4;
        let src_ptr0 = src0.get_unchecked((start_x * COMPONENTS)..);
        let src_ptr1 = src1.get_unchecked((start_x * COMPONENTS)..);

        let rgba_pixel0 = _mm_loadu_si64(src_ptr0.as_ptr() as *const u8);
        let rgba_pixel1 = _mm_loadu_si64(src_ptr1.as_ptr() as *const u8);

        let full_pixel = _mm256_cvtepu16_epi32(_mm_unpacklo_epi64(rgba_pixel0, rgba_pixel1));

        let f_pixel = _mm256_cvtepi32_ps(full_pixel);

        _mm256_prefer_fma_ps::<FMA>(store, f_pixel, w0)
    }

    #[inline(always)]
    unsafe fn pass(
        &self,
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        filter_weights: &FilterWeights<f32>,
        bit_depth: u32,
    ) {
        const CHANNELS: usize = 4;

        let v_cap_colors = _mm256_set1_epi16((((1i32 << bit_depth) - 1) as u16) as i16);

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

            let mut store_0 = _mm256_setzero_ps();
            let mut store_1 = _mm256_setzero_ps();

            let bounds_size = bounds.size;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            if jx >= 4 {
                let mut astore_0 = _mm256_setzero_ps();
                let mut astore_1 = _mm256_setzero_ps();
                let mut astore_2 = _mm256_setzero_ps();
                let mut astore_3 = _mm256_setzero_ps();

                while jx + 8 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);

                    let w0 = _mm_load1_ps(w_ptr.as_ptr());
                    let w1 = _mm_load1_ps(w_ptr.as_ptr().add(1));
                    let w2 = _mm_load1_ps(w_ptr.as_ptr().add(2));
                    let w3 = _mm_load1_ps(w_ptr.as_ptr().add(3));
                    let w4 = _mm_load1_ps(w_ptr.as_ptr().add(4));
                    let w5 = _mm_load1_ps(w_ptr.as_ptr().add(5));
                    let w6 = _mm_load1_ps(w_ptr.as_ptr().add(6));
                    let w7 = _mm_load1_ps(w_ptr.as_ptr().add(7));

                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);
                    let w45 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w4), w5);
                    let w67 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w6), w7);

                    astore_0 = conv_horiz_rgba_8_u16::<FMA>(
                        bounds_start,
                        src0,
                        w01,
                        w23,
                        w45,
                        w67,
                        astore_0,
                    );
                    astore_1 = conv_horiz_rgba_8_u16::<FMA>(
                        bounds_start,
                        src1,
                        w01,
                        w23,
                        w45,
                        w67,
                        astore_1,
                    );
                    astore_2 = conv_horiz_rgba_8_u16::<FMA>(
                        bounds_start,
                        src2,
                        w01,
                        w23,
                        w45,
                        w67,
                        astore_2,
                    );
                    astore_3 = conv_horiz_rgba_8_u16::<FMA>(
                        bounds_start,
                        src3,
                        w01,
                        w23,
                        w45,
                        w67,
                        astore_3,
                    );
                    jx += 8;
                }

                while jx + 4 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_load1_ps(w_ptr.as_ptr());
                    let w1 = _mm_load1_ps(w_ptr.as_ptr().add(1));
                    let w2 = _mm_load1_ps(w_ptr.as_ptr().add(2));
                    let w3 = _mm_load1_ps(w_ptr.as_ptr().add(3));

                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);

                    astore_0 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src0, w01, w23, astore_0);
                    astore_1 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src1, w01, w23, astore_1);
                    astore_2 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src2, w01, w23, astore_2);
                    astore_3 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src3, w01, w23, astore_3);
                    jx += 4;
                }

                store_0 = _mm256_add_ps(
                    _mm256_permute2f128_ps::<0x20>(astore_0, astore_1),
                    _mm256_permute2f128_ps::<0x31>(astore_0, astore_1),
                );

                store_1 = _mm256_add_ps(
                    _mm256_permute2f128_ps::<0x20>(astore_2, astore_3),
                    _mm256_permute2f128_ps::<0x31>(astore_2, astore_3),
                );
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = _mm256_broadcast_ss(w_ptr.get_unchecked(0));
                let w1 = _mm256_broadcast_ss(w_ptr.get_unchecked(1));
                store_0 = self.rgba_2_u16_avx(bounds_start, src0, src1, w0, w1, store_0);
                store_1 = self.rgba_2_u16_avx(bounds_start, src2, src3, w0, w1, store_1);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = _mm256_broadcast_ss(w_ptr.get_unchecked(0));
                store_0 = self.rgba_1_u16(bounds_start, src0, src1, w0, store_0);
                store_1 = self.rgba_1_u16(bounds_start, src2, src3, w0, store_1);
                jx += 1;
            }

            let v_st0 = _mm256_cvtps_epi32(store_0);
            let v_st1 = _mm256_cvtps_epi32(store_1);

            let store_16_0 = _mm256_min_epu16(_mm256_packus_epi32(v_st0, v_st0), v_cap_colors);
            let store_16_1 = _mm256_min_epu16(_mm256_packus_epi32(v_st1, v_st1), v_cap_colors);

            _mm_storeu_si64(
                chunk0.as_mut_ptr() as *mut u8,
                _mm256_castsi256_si128(store_16_0),
            );
            _mm_storeu_si64(
                chunk1.as_mut_ptr() as *mut u8,
                _mm256_extracti128_si256::<1>(store_16_0),
            );
            _mm_storeu_si64(
                chunk2.as_mut_ptr() as *mut u8,
                _mm256_castsi256_si128(store_16_1),
            );
            _mm_storeu_si64(
                chunk3.as_mut_ptr() as *mut u8,
                _mm256_extracti128_si256::<1>(store_16_1),
            );
        }
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_u16_row_f(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        if std::arch::is_x86_feature_detected!("fma") {
            convolve_horizontal_rgba_avx_u16_row_fma(src, dst, filter_weights, bit_depth);
        } else {
            convolve_horizontal_rgba_avx_u16_row_def(src, dst, filter_weights, bit_depth);
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_rgba_avx_u16_row_def(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    let unit = OneRowExecutionHandler::<false>::default();
    unit.pass(src, dst, filter_weights, bit_depth);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_rgba_avx_u16_row_fma(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    let unit = OneRowExecutionHandler::<true>::default();
    unit.pass(src, dst, filter_weights, bit_depth);
}

#[derive(Copy, Clone, Default)]
struct OneRowExecutionHandler<const FMA: bool> {}

impl<const FMA: bool> OneRowExecutionHandler<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        src: &[u16],
        dst: &mut [u16],
        filter_weights: &FilterWeights<f32>,
        bit_depth: u32,
    ) {
        const CHANNELS: usize = 4;

        let v_cap_colors = _mm_set1_epi16((((1i32 << bit_depth) - 1) as u16) as i16);

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
            let mut store = _mm_setzero_ps();

            if jx >= 4 {
                let mut astore = _mm256_setzero_ps();

                while jx + 8 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_load1_ps(w_ptr.as_ptr());
                    let w1 = _mm_load1_ps(w_ptr.as_ptr().add(1));
                    let w2 = _mm_load1_ps(w_ptr.as_ptr().add(2));
                    let w3 = _mm_load1_ps(w_ptr.as_ptr().add(3));
                    let w4 = _mm_load1_ps(w_ptr.as_ptr().add(4));
                    let w5 = _mm_load1_ps(w_ptr.as_ptr().add(5));
                    let w6 = _mm_load1_ps(w_ptr.as_ptr().add(6));
                    let w7 = _mm_load1_ps(w_ptr.as_ptr().add(7));
                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);
                    let w45 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w4), w5);
                    let w67 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w6), w7);
                    astore =
                        conv_horiz_rgba_8_u16::<FMA>(bounds_start, src, w01, w23, w45, w67, astore);
                    jx += 8;
                }

                while jx + 4 < bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_load1_ps(w_ptr.as_ptr());
                    let w1 = _mm_load1_ps(w_ptr.as_ptr().add(1));
                    let w2 = _mm_load1_ps(w_ptr.as_ptr().add(2));
                    let w3 = _mm_load1_ps(w_ptr.as_ptr().add(3));
                    let bounds_start = bounds.start + jx;

                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);

                    astore = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src, w01, w23, astore);
                    jx += 4;
                }

                store = _mm_add_ps(
                    _mm256_castps256_ps128(astore),
                    _mm256_extractf128_ps::<1>(astore),
                );
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = _mm_load1_ps(w_ptr.as_ptr());
                let w1 = _mm_load1_ps(w_ptr.as_ptr().add(1));
                store = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src, w0, w1, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = _mm_load1_ps(w_ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src, w0, store);
                jx += 1;
            }

            let v_st = _mm_cvtps_epi32(store);

            let store_16_0 = _mm_min_epu16(_mm_packus_epi32(v_st, v_st), v_cap_colors);
            _mm_storeu_si64(dst.as_mut_ptr() as *mut u8, store_16_0);
        }
    }
}
