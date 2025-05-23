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
use crate::avx2::utils::{_mm_hsum_ps, _mm_prefer_fma_ps};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn conv_horiz_rgba_1_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    store: __m128,
) -> __m128 {
    unsafe {
        const COMPONENTS: usize = 1;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
        let rgba_pixel = _mm_loadu_si16(src_ptr.as_ptr() as *const _);
        _mm_prefer_fma_ps::<FMA>(
            store,
            _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel, _mm_setzero_si128())),
            w0,
        )
    }
}

#[inline(always)]
unsafe fn conv_horiz_rgba_2_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    store: __m128,
) -> __m128 {
    unsafe {
        const COMPONENTS: usize = 1;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let rgba_pixel = _mm_loadu_si32(src_ptr.as_ptr() as *const _);

        _mm_prefer_fma_ps::<FMA>(
            store,
            _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel, _mm_setzero_si128())),
            w0,
        )
    }
}

#[inline]
unsafe fn conv_horiz_rgba_4_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    store: __m128,
) -> __m128 {
    unsafe {
        const COMPONENTS: usize = 1;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let rgba_pixel = _mm_loadu_si64(src_ptr.as_ptr() as *const _);

        _mm_prefer_fma_ps::<FMA>(
            store,
            _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel, _mm_setzero_si128())),
            w0,
        )
    }
}

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w: __m256,
    store: __m128,
) -> __m128 {
    unsafe {
        const COMPONENTS: usize = 1;
        let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

        let px = _mm_loadu_si128(src_ptr.as_ptr() as *const _);

        let acc = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(px)), w);
        let z0 = _mm_add_ps(store, _mm256_castps256_ps128(acc));
        _mm_add_ps(z0, _mm256_extractf128_ps::<1>(acc))
    }
}

pub(crate) fn convolve_horizontal_plane_avx_rows_4_u16_f(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        if std::arch::is_x86_feature_detected!("fma") {
            convolve_horizontal_plane_avx_rows_4_u16_fma(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        } else {
            convolve_horizontal_plane_avx_rows_4_u16_def(
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
unsafe fn convolve_horizontal_plane_avx_rows_4_u16_def(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        let unit = Row4ExecutionHandler::<false>::default();
        unit.pass(src, src_stride, dst, dst_stride, filter_weights, bit_depth);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_plane_avx_rows_4_u16_fma(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        let unit = Row4ExecutionHandler::<true>::default();
        unit.pass(src, src_stride, dst, dst_stride, filter_weights, bit_depth);
    }
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionHandler<const FMA: bool> {}

impl<const FMA: bool> Row4ExecutionHandler<FMA> {
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
        unsafe {
            let v_max_colors = _mm_set1_epi32((1i32 << bit_depth) - 1);

            let (row0_ref, rest) = dst.split_at_mut(dst_stride);
            let (row1_ref, rest) = rest.split_at_mut(dst_stride);
            let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

            let iter_row0 = row0_ref.iter_mut();
            let iter_row1 = row1_ref.iter_mut();
            let iter_row2 = row2_ref.iter_mut();
            let iter_row3 = row3_ref.iter_mut();

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

                let mut store_0 = _mm_setzero_ps();
                let mut store_1 = _mm_setzero_ps();
                let mut store_2 = _mm_setzero_ps();
                let mut store_3 = _mm_setzero_ps();

                let bounds_size = bounds.size;

                let src0 = src;
                let src1 = src0.get_unchecked(src_stride..);
                let src2 = src1.get_unchecked(src_stride..);
                let src3 = src2.get_unchecked(src_stride..);

                while jx + 8 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);

                    let w0 = _mm256_loadu_ps(w_ptr.as_ptr());

                    store_0 = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src0, w0, store_0);
                    store_1 = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src1, w0, store_1);
                    store_2 = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src2, w0, store_2);
                    store_3 = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src3, w0, store_3);
                    jx += 8;
                }

                while jx + 4 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_loadu_ps(w_ptr.as_ptr());

                    store_0 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src0, w0, store_0);
                    store_1 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src1, w0, store_1);
                    store_2 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src2, w0, store_2);
                    store_3 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src3, w0, store_3);
                    jx += 4;
                }

                while jx + 2 < bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let bounds_start = bounds.start + jx;
                    let w0 = _mm_castsi128_ps(_mm_loadu_si64(w_ptr.as_ptr() as *const _));
                    store_0 = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src0, w0, store_0);
                    store_1 = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src1, w0, store_1);
                    store_2 = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src2, w0, store_2);
                    store_3 = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src3, w0, store_3);
                    jx += 2;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let bounds_start = bounds.start + jx;
                    let w0 = _mm_load_ss(w_ptr.as_ptr());
                    store_0 = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src0, w0, store_0);
                    store_1 = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src1, w0, store_1);
                    store_2 = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src2, w0, store_2);
                    store_3 = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src3, w0, store_3);
                    jx += 1;
                }

                store_0 = _mm_hsum_ps(store_0);
                store_1 = _mm_hsum_ps(store_1);
                store_2 = _mm_hsum_ps(store_2);
                store_3 = _mm_hsum_ps(store_3);

                let v_st0 = _mm_min_epi32(
                    _mm_cvtps_epi32(_mm_max_ps(store_0, _mm_setzero_ps())),
                    v_max_colors,
                );
                let v_st1 = _mm_min_epi32(
                    _mm_cvtps_epi32(_mm_max_ps(store_1, _mm_setzero_ps())),
                    v_max_colors,
                );
                let v_st2 = _mm_min_epi32(
                    _mm_cvtps_epi32(_mm_max_ps(store_2, _mm_setzero_ps())),
                    v_max_colors,
                );
                let v_st3 = _mm_min_epi32(
                    _mm_cvtps_epi32(_mm_max_ps(store_3, _mm_setzero_ps())),
                    v_max_colors,
                );

                let store_16_0 = _mm_packus_epi32(v_st0, v_st1);
                let store_16_1 = _mm_packus_epi32(v_st2, v_st3);

                *chunk0 = _mm_extract_epi16::<0>(store_16_0) as u16;
                *chunk1 = _mm_extract_epi16::<4>(store_16_0) as u16;
                *chunk2 = _mm_extract_epi16::<0>(store_16_1) as u16;
                *chunk3 = _mm_extract_epi16::<4>(store_16_1) as u16;
            }
        }
    }
}

pub(crate) fn convolve_horizontal_plane_avx_u16_row_f(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        if std::arch::is_x86_feature_detected!("fma") {
            convolve_horizontal_plane_avx_u16_row_fma(src, dst, filter_weights, bit_depth);
        } else {
            convolve_horizontal_plane_avx_u16_row_def(src, dst, filter_weights, bit_depth);
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_plane_avx_u16_row_def(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        let unit = OneRowExecutionHandler::<false>::default();
        unit.pass(src, dst, filter_weights, bit_depth);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_plane_avx_u16_row_fma(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        let unit = OneRowExecutionHandler::<true>::default();
        unit.pass(src, dst, filter_weights, bit_depth);
    }
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
        unsafe {
            let v_max_colors = _mm_set1_epi32((1i32 << bit_depth) - 1);

            for ((dst, bounds), weights) in dst.iter_mut().zip(filter_weights.bounds.iter()).zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            ) {
                let bounds_size = bounds.size;
                let mut jx = 0usize;
                let mut store = _mm_setzero_ps();

                while jx + 8 < bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm256_loadu_ps(w_ptr.as_ptr());
                    store = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src, w0, store);
                    jx += 8;
                }

                while jx + 4 < bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_loadu_ps(w_ptr.as_ptr());
                    let bounds_start = bounds.start + jx;

                    store = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src, w0, store);
                    jx += 4;
                }

                while jx + 2 < bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let bounds_start = bounds.start + jx;
                    let w0 = _mm_castsi128_ps(_mm_loadu_si64(w_ptr.as_ptr() as *const _));
                    store = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src, w0, store);
                    jx += 2;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_load_ss(w_ptr.as_ptr());
                    let bounds_start = bounds.start + jx;
                    store = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src, w0, store);
                    jx += 1;
                }

                store = _mm_hsum_ps(store);

                let v_st = _mm_min_epi32(
                    _mm_cvtps_epi32(_mm_max_ps(store, _mm_setzero_ps())),
                    v_max_colors,
                );

                let store_16_0 = _mm_packus_epi32(v_st, v_st);
                *dst = _mm_extract_epi16::<0>(store_16_0) as u16;
            }
        }
    }
}
