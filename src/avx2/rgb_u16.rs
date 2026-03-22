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

use crate::avx2::utils::{_mm_prefer_fma_ps, _mm256_prefer_fma_ps, shuffle};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

#[inline(always)]
fn load_rgb_u16_1(src: &[u16]) -> __m128i {
    // Load 3 x u16 = 6 bytes safely
    unsafe {
        let lo = _mm_loadu_si32(src.as_ptr().cast()); // r, g
        let hi = _mm_loadu_si16(src.get_unchecked(2..).as_ptr().cast()); // b
        _mm_unpacklo_epi32(lo, hi) // [r, g, b, 0]
    }
}

#[inline(always)]
fn load_rgb_u16_2(src: &[u16]) -> __m128i {
    unsafe {
        // Load 2 x RGB = 6 x u16 = 12 bytes
        let lo = _mm_loadu_si64(src.as_ptr().cast()); // [r0,g0,b0,r1]
        let hi = _mm_loadu_si32(src.get_unchecked(4..).as_ptr().cast()); // [g1,b1]
        _mm_unpacklo_epi64(lo, hi) // [r0,g0,b0,r1,g1,b1,0,0]
    }
}

#[inline(always)]
fn conv_horiz_rgb_1_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    store: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let pixel = load_rgb_u16_1(src_ptr);
        _mm_prefer_fma_ps::<FMA>(store, _mm_cvtepi32_ps(_mm_cvtepu16_epi32(pixel)), w0)
    }
}

#[inline(always)]
fn conv_horiz_rgb_2_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    w1: __m128,
    store: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        let pixel = load_rgb_u16_2(src_ptr); // [r0,g0,b0,r1,g1,b1,0,0] as u16

        let p0 = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(pixel)); // [r0,g0,b0,r1] as f32
        let p1 = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(_mm_srli_si128::<6>(pixel))); // [g1,b1,0,0] as f32

        let acc = _mm_prefer_fma_ps::<FMA>(store, p0, w0);
        _mm_prefer_fma_ps::<FMA>(acc, p1, w1)
    }
}

#[inline(always)]
fn conv_horiz_rgb_4_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m256,
    w1: __m256,
    store: __m256,
) -> __m256 {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked((start_x * CN)..);
        // 4 x RGB = 12 x u16 = 24 bytes
        // Load 16 bytes + 8 bytes
        let lo = _mm_loadu_si128(src_ptr.as_ptr().cast()); // [r0,g0,b0,r1,g1,b1,r2,g2]
        let hi = _mm_loadu_si64(src_ptr.get_unchecked(8..).as_ptr().cast()); // [b2,r3,g3,b3]

        // pixel 0: [r0,g0,b0,0], pixel 1: [r1,g1,b1,0]
        // pixel 2: [r2,g2,b2,0], pixel 3: [r3,g3,b3,0]
        let shuf = _mm_setr_epi8(0, 1, 2, 3, 4, 5, -1, -1, 6, 7, 8, 9, 10, 11, -1, -1);
        let hi_src = _mm_alignr_epi8(hi, lo, 12); // [r2,g2,b2,r3,g3,b3,0,0]

        let lo_shuf = _mm_shuffle_epi8(lo, shuf); // [r0,g0,b0,0, r1,g1,b1,0]
        let hi_shuf = _mm_shuffle_epi8(hi_src, shuf); // [r2,g2,b2,0, r3,g3,b3,0]

        // Convert u16 -> f32 for each pair
        // lo_shuf as u16: [r0,g0,b0,0, r1,g1,b1,0]
        let p01 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(lo_shuf)); // [r0,g0,b0,0, r1,g1,b1,0] as f32
        let p23 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(hi_shuf)); // [r2,g2,b2,0, r3,g3,b3,0] as f32

        // w0 covers pixels 0,1 (lo128=p0, hi128=p1)
        // w1 covers pixels 2,3 (lo128=p2, hi128=p3)
        // But _mm256_cvtepu16_epi32 takes lo 8 x u16 -> 8 x i32
        // lo_shuf has [r0,g0,b0,0,r1,g1,b1,0] so cvtepu16_epi32 gives all 8 as i32
        // p01 lo128 = [r0,g0,b0,0], hi128 = [r1,g1,b1,0] ✓

        let acc = _mm256_prefer_fma_ps::<FMA>(store, p01, w0);
        _mm256_prefer_fma_ps::<FMA>(acc, p23, w1)
    }
}

#[inline(always)]
fn conv_horiz_rgb_8_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w01: __m256,
    w23: __m256,
    w45: __m256,
    w67: __m256,
    store: __m256,
) -> __m256 {
    let acc = conv_horiz_rgb_4_u16::<FMA>(start_x, src, w01, w23, store);
    conv_horiz_rgb_4_u16::<FMA>(start_x + 4, src, w45, w67, acc)
}

#[inline(always)]
fn set_pixel_f32(ptr: &mut [u16; 3], pixel: __m128) {
    unsafe {
        let v = _mm_packus_epi32(_mm_cvtps_epi32(pixel), _mm_setzero_si128());
        _mm_storeu_si32(ptr.as_mut_ptr().cast(), v);
        ptr[2] = _mm_extract_epi16::<2>(v) as u16;
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_rows_4_u16_default(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgb_avx_rows_4_u16_def(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        );
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_rows_4_u16_fma(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgb_avx_rows_4_u16_fma_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        );
    }
}

#[target_feature(enable = "avx2")]
fn convolve_horizontal_rgb_avx_rows_4_u16_def(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
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

#[target_feature(enable = "avx2", enable = "fma")]
fn convolve_horizontal_rgb_avx_rows_4_u16_fma_impl(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
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

#[derive(Copy, Clone, Default)]
struct Row4ExecutionHandlerRgb<const FMA: bool> {}

impl<const FMA: bool> Row4ExecutionHandlerRgb<FMA> {
    #[inline(always)]
    fn rgb_2_u16_avx(
        &self,
        start_x: usize,
        src0: &[u16],
        src1: &[u16],
        w0: __m256,
        w1: __m256,
        store: __m256,
    ) -> __m256 {
        unsafe {
            const CN: usize = 3;
            let s0 = src0.get_unchecked((start_x * CN)..);
            let s1 = src1.get_unchecked((start_x * CN)..);

            let px0 = load_rgb_u16_2(s0); // [r0,g0,b0,r1,g1,b1,0,0]
            let px1 = load_rgb_u16_2(s1);

            // p0_lo = first pixel of src0, p0_hi = second pixel of src0
            let p0_lo = _mm_cvtepu16_epi32(px0);
            let p0_hi = _mm_cvtepu16_epi32(_mm_srli_si128::<6>(px0));
            let p1_lo = _mm_cvtepu16_epi32(px1);
            let p1_hi = _mm_cvtepu16_epi32(_mm_srli_si128::<6>(px1));

            let p0m = _mm256_cvtepi32_ps(_mm256_setr_m128i(p0_lo, p1_lo));
            let p1m = _mm256_cvtepi32_ps(_mm256_setr_m128i(p0_hi, p1_hi));

            let acc = _mm256_prefer_fma_ps::<FMA>(store, p0m, w0);
            _mm256_prefer_fma_ps::<FMA>(acc, p1m, w1)
        }
    }

    #[inline(always)]
    fn rgb_1_u16_avx(
        &self,
        start_x: usize,
        src0: &[u16],
        src1: &[u16],
        w0: __m256,
        store: __m256,
    ) -> __m256 {
        unsafe {
            const CN: usize = 3;
            let s0 = src0.get_unchecked((start_x * CN)..);
            let s1 = src1.get_unchecked((start_x * CN)..);

            let p0 = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(load_rgb_u16_1(s0)));
            let p1 = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(load_rgb_u16_1(s1)));

            let pixel = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(p0), p1);
            _mm256_prefer_fma_ps::<FMA>(store, pixel, w0)
        }
    }

    #[inline(always)]
    fn pass(
        &self,
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        filter_weights: &FilterWeights<f32>,
        bit_depth: u32,
    ) {
        unsafe {
            const CN: usize = 3;

            let v_max_colors = (1u32 << bit_depth) - 1;
            let v_cap_f = _mm256_set1_ps(v_max_colors as f32);

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

                let mut astore_0 = _mm256_setzero_ps();
                let mut astore_1 = _mm256_setzero_ps();
                let mut astore_2 = _mm256_setzero_ps();
                let mut astore_3 = _mm256_setzero_ps();

                while jx + 8 <= bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w = _mm256_loadu_ps(w_ptr.as_ptr());
                    let w_lo = _mm256_castps256_ps128(w);
                    let w_hi = _mm256_extractf128_ps::<1>(w);

                    let w0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w_lo, w_lo);
                    let w1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w_lo, w_lo);
                    let w2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w_lo, w_lo);
                    let w3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w_lo, w_lo);
                    let w4 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w_hi, w_hi);
                    let w5 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w_hi, w_hi);
                    let w6 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w_hi, w_hi);
                    let w7 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w_hi, w_hi);

                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);
                    let w45 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w4), w5);
                    let w67 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w6), w7);

                    astore_0 = conv_horiz_rgb_8_u16::<FMA>(
                        bounds_start,
                        src0,
                        w01,
                        w23,
                        w45,
                        w67,
                        astore_0,
                    );
                    astore_1 = conv_horiz_rgb_8_u16::<FMA>(
                        bounds_start,
                        src1,
                        w01,
                        w23,
                        w45,
                        w67,
                        astore_1,
                    );
                    astore_2 = conv_horiz_rgb_8_u16::<FMA>(
                        bounds_start,
                        src2,
                        w01,
                        w23,
                        w45,
                        w67,
                        astore_2,
                    );
                    astore_3 = conv_horiz_rgb_8_u16::<FMA>(
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

                while jx + 4 <= bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w = _mm_loadu_ps(w_ptr.as_ptr());

                    let w0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w, w);
                    let w1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w, w);
                    let w2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w, w);
                    let w3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w, w);

                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);

                    astore_0 = conv_horiz_rgb_4_u16::<FMA>(bounds_start, src0, w01, w23, astore_0);
                    astore_1 = conv_horiz_rgb_4_u16::<FMA>(bounds_start, src1, w01, w23, astore_1);
                    astore_2 = conv_horiz_rgb_4_u16::<FMA>(bounds_start, src2, w01, w23, astore_2);
                    astore_3 = conv_horiz_rgb_4_u16::<FMA>(bounds_start, src3, w01, w23, astore_3);
                    jx += 4;
                }

                let mut store_0 = _mm256_add_ps(
                    _mm256_permute2f128_ps::<0x20>(astore_0, astore_1),
                    _mm256_permute2f128_ps::<0x31>(astore_0, astore_1),
                );
                let mut store_1 = _mm256_add_ps(
                    _mm256_permute2f128_ps::<0x20>(astore_2, astore_3),
                    _mm256_permute2f128_ps::<0x31>(astore_2, astore_3),
                );

                while jx + 2 <= bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let bounds_start = bounds.start + jx;
                    let w0 = _mm256_broadcast_ss(w_ptr.get_unchecked(0));
                    let w1 = _mm256_broadcast_ss(w_ptr.get_unchecked(1));
                    store_0 = self.rgb_2_u16_avx(bounds_start, src0, src1, w0, w1, store_0);
                    store_1 = self.rgb_2_u16_avx(bounds_start, src2, src3, w0, w1, store_1);
                    jx += 2;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let bounds_start = bounds.start + jx;
                    let w0 = _mm256_broadcast_ss(w_ptr.get_unchecked(0));
                    store_0 = self.rgb_1_u16_avx(bounds_start, src0, src1, w0, store_0);
                    store_1 = self.rgb_1_u16_avx(bounds_start, src2, src3, w0, store_1);
                    jx += 1;
                }

                // Clamp and store
                store_0 = _mm256_min_ps(store_0, v_cap_f);
                store_1 = _mm256_min_ps(store_1, v_cap_f);

                let lo0 = _mm256_castps256_ps128(store_0);
                let hi0 = _mm256_extractf128_ps::<1>(store_0);
                let lo1 = _mm256_castps256_ps128(store_1);
                let hi1 = _mm256_extractf128_ps::<1>(store_1);

                set_pixel_f32(chunk0, lo0);
                set_pixel_f32(chunk1, hi0);
                set_pixel_f32(chunk2, lo1);
                set_pixel_f32(chunk3, hi1);
            }
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_u16_row_default(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgb_avx_u16_row_def(src, dst, filter_weights, bit_depth);
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_u16_row_fma(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        convolve_horizontal_rgb_avx_u16_row_fma_impl(src, dst, filter_weights, bit_depth);
    }
}

#[target_feature(enable = "avx2")]
fn convolve_horizontal_rgb_avx_u16_row_def(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    OneRowExecutionHandlerRgb::<false>::default().pass(src, dst, filter_weights, bit_depth);
}

#[target_feature(enable = "avx2", enable = "fma")]
fn convolve_horizontal_rgb_avx_u16_row_fma_impl(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    OneRowExecutionHandlerRgb::<true>::default().pass(src, dst, filter_weights, bit_depth);
}

#[derive(Copy, Clone, Default)]
struct OneRowExecutionHandlerRgb<const FMA: bool> {}

impl<const FMA: bool> OneRowExecutionHandlerRgb<FMA> {
    #[inline(always)]
    fn pass(
        &self,
        src: &[u16],
        dst: &mut [u16],
        filter_weights: &FilterWeights<f32>,
        bit_depth: u32,
    ) {
        unsafe {
            const CN: usize = 3;

            let v_max_colors = (1u32 << bit_depth) - 1;
            let v_cap_f = _mm_set1_ps(v_max_colors as f32);

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

                let mut astore = _mm256_setzero_ps();

                while jx + 8 <= bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w = _mm256_loadu_ps(w_ptr.as_ptr());
                    let w_lo = _mm256_castps256_ps128(w);
                    let w_hi = _mm256_extractf128_ps::<1>(w);

                    let w0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w_lo, w_lo);
                    let w1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w_lo, w_lo);
                    let w2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w_lo, w_lo);
                    let w3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w_lo, w_lo);
                    let w4 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w_hi, w_hi);
                    let w5 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w_hi, w_hi);
                    let w6 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w_hi, w_hi);
                    let w7 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w_hi, w_hi);

                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);
                    let w45 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w4), w5);
                    let w67 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w6), w7);

                    astore =
                        conv_horiz_rgb_8_u16::<FMA>(bounds_start, src, w01, w23, w45, w67, astore);
                    jx += 8;
                }

                while jx + 4 <= bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let w = _mm_loadu_ps(w_ptr.as_ptr());

                    let w0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(w, w);
                    let w1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(w, w);
                    let w2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(w, w);
                    let w3 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(w, w);

                    let w01 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w1);
                    let w23 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w2), w3);

                    let bounds_start = bounds.start + jx;
                    astore = conv_horiz_rgb_4_u16::<FMA>(bounds_start, src, w01, w23, astore);
                    jx += 4;
                }

                let mut store = _mm_add_ps(
                    _mm256_castps256_ps128(astore),
                    _mm256_extractf128_ps::<1>(astore),
                );

                while jx + 2 <= bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let bounds_start = bounds.start + jx;
                    let w0 = _mm_broadcast_ss(w_ptr.get_unchecked(0));
                    let w1 = _mm_broadcast_ss(w_ptr.get_unchecked(1));
                    store = conv_horiz_rgb_2_u16::<FMA>(bounds_start, src, w0, w1, store);
                    jx += 2;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_broadcast_ss(w_ptr.get_unchecked(0));
                    let bounds_start = bounds.start + jx;
                    store = conv_horiz_rgb_1_u16::<FMA>(bounds_start, src, w0, store);
                    jx += 1;
                }

                store = _mm_min_ps(store, v_cap_f);
                set_pixel_f32(dst, store);
            }
        }
    }
}
