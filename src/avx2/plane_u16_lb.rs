/*
 * Copyright (c) Radzivon Bartoshyk 04/2025. All rights reserved.
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

use crate::avx2::utils::{_mm_dot16_avx_epi32, _mm_reduce_r_epi32};
use crate::filter_weights::FilterWeights;
use crate::support::{PRECISION, ROUNDING_CONST};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn acc_1_dot<const D: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128i,
    store: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 1;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let px = _mm_loadu_si16(src_ptr.as_ptr() as *const _);
    _mm_dot16_avx_epi32::<D>(store, px, w0)
}

#[inline(always)]
unsafe fn acc_2_dot<const D: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128i,
    store: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 1;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let px = _mm_loadu_si32(src_ptr.as_ptr() as *const _);
    _mm_dot16_avx_epi32::<D>(store, px, w0)
}

#[inline(always)]
unsafe fn acc_4_dot<const D: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128i,
    store: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 1;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let px = _mm_loadu_si64(src_ptr.as_ptr() as *const _);
    _mm_dot16_avx_epi32::<D>(store, px, w0)
}

#[inline(always)]
unsafe fn acc_8_dot<const D: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128i,
    store: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 1;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let px = _mm_loadu_si128(src_ptr.as_ptr() as *const _);

    _mm_dot16_avx_epi32::<D>(store, px, w0)
}

pub(crate) fn convolve_horizontal_plane_avx_rows_4_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return convolve_horizontal_plane_avx_rows_4_lb_vn(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        }
        convolve_horizontal_plane_avx_rows_4_lb_a(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
            bit_depth,
        );
    }
}

#[cfg(feature = "nightly_avx512")]
#[target_feature(enable = "avxvnni", enable = "avx2")]
unsafe fn convolve_horizontal_plane_avx_rows_4_lb_vn(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    let unit = Row4ExecutionHandler::<true>::default();
    unit.pass(src, src_stride, dst, dst_stride, filter_weights, bit_depth);
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_plane_avx_rows_4_lb_a(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    let unit = Row4ExecutionHandler::<false>::default();
    unit.pass(src, src_stride, dst, dst_stride, filter_weights, bit_depth);
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionHandler<const D: bool> {}

impl<const D: bool> Row4ExecutionHandler<D> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) {
        let v_max_colors = _mm_set1_epi16((1i16 << bit_depth) - 1);

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
            let mut store_0 = _mm_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
            );
            let mut store_1 = _mm_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
            );
            let mut store_2 = _mm_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
            );
            let mut store_3 = _mm_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
            );

            let bounds_size = bounds.size;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while jx + 8 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let wl = _mm_loadu_si128(w_ptr.as_ptr() as *const _);
                let bounds_start = bounds.start + jx;
                store_0 = acc_8_dot::<D>(bounds_start, src0, wl, store_0);
                store_1 = acc_8_dot::<D>(bounds_start, src1, wl, store_1);
                store_2 = acc_8_dot::<D>(bounds_start, src2, wl, store_2);
                store_3 = acc_8_dot::<D>(bounds_start, src3, wl, store_3);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let bounds_start = bounds.start + jx;
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = _mm_loadu_si64(w_ptr.as_ptr() as *const _);
                store_0 = acc_4_dot::<D>(bounds_start, src0, w0, store_0);
                store_1 = acc_4_dot::<D>(bounds_start, src1, w0, store_1);
                store_2 = acc_4_dot::<D>(bounds_start, src2, w0, store_2);
                store_3 = acc_4_dot::<D>(bounds_start, src3, w0, store_3);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = _mm_loadu_si32(w_ptr.as_ptr() as *const _);
                store_0 = acc_2_dot::<D>(bounds_start, src0, w0, store_0);
                store_1 = acc_2_dot::<D>(bounds_start, src1, w0, store_1);
                store_2 = acc_2_dot::<D>(bounds_start, src2, w0, store_2);
                store_3 = acc_2_dot::<D>(bounds_start, src3, w0, store_3);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = _mm_loadu_si16(w_ptr.as_ptr() as *const _);
                store_0 = acc_1_dot::<D>(bounds_start, src0, w0, store_0);
                store_1 = acc_1_dot::<D>(bounds_start, src1, w0, store_1);
                store_2 = acc_1_dot::<D>(bounds_start, src2, w0, store_2);
                store_3 = acc_1_dot::<D>(bounds_start, src3, w0, store_3);
                jx += 1;
            }

            let v_st0 = _mm_reduce_r_epi32::<PRECISION>(store_0);
            let v_st1 = _mm_reduce_r_epi32::<PRECISION>(store_1);
            let v_st2 = _mm_reduce_r_epi32::<PRECISION>(store_2);
            let v_st3 = _mm_reduce_r_epi32::<PRECISION>(store_3);

            let v_zst0 = _mm_min_epi16(_mm_packus_epi32(v_st0, v_st1), v_max_colors);
            let v_zst1 = _mm_min_epi16(_mm_packus_epi32(v_st2, v_st3), v_max_colors);

            *chunk0 = _mm_extract_epi16::<0>(v_zst0) as u16;
            *chunk1 = _mm_extract_epi16::<4>(v_zst0) as u16;
            *chunk2 = _mm_extract_epi16::<0>(v_zst1) as u16;
            *chunk3 = _mm_extract_epi16::<4>(v_zst1) as u16;
        }
    }
}

pub(crate) fn convolve_horizontal_plane_avx_u16lp_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return convolve_horizontal_plane_avx_u16_row_vn(src, dst, filter_weights, bit_depth);
        }
        convolve_horizontal_plane_avx_u16_row_avx(src, dst, filter_weights, bit_depth);
    }
}

#[cfg(feature = "nightly_avx512")]
#[target_feature(enable = "avxvnni", enable = "avx2")]
unsafe fn convolve_horizontal_plane_avx_u16_row_vn(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    let unit = OneRowExecutionUnit::<true>::default();
    unit.pass(src, dst, filter_weights, bit_depth);
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_plane_avx_u16_row_avx(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<i16>,
    bit_depth: u32,
) {
    let unit = OneRowExecutionUnit::<false>::default();
    unit.pass(src, dst, filter_weights, bit_depth);
}

#[derive(Copy, Clone, Default)]
struct OneRowExecutionUnit<const D: bool> {}

impl<const D: bool> OneRowExecutionUnit<D> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        src: &[u16],
        dst: &mut [u16],
        filter_weights: &FilterWeights<i16>,
        bit_depth: u32,
    ) {
        let v_max_colors = _mm_set1_epi16((1 << bit_depth) - 1);

        for ((dst, bounds), weights) in dst.iter_mut().zip(filter_weights.bounds.iter()).zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        ) {
            let bounds_size = bounds.size;
            let mut jx = 0usize;
            let mut store = _mm_setr_epi32(
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
                ROUNDING_CONST,
            );

            while jx + 8 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let wl = _mm_loadu_si128(w_ptr.as_ptr() as *const _);
                let bounds_start = bounds.start + jx;
                store = acc_8_dot::<D>(bounds_start, src, wl, store);
                jx += 8;
            }

            while jx + 4 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = _mm_loadu_si64(w_ptr.as_ptr() as *const _);
                let bounds_start = bounds.start + jx;
                store = acc_4_dot::<D>(bounds_start, src, w0, store);
                jx += 4;
            }

            while jx + 2 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let bounds_start = bounds.start + jx;
                let w0 = _mm_loadu_si32(w_ptr.as_ptr() as *const _);
                store = acc_2_dot::<D>(bounds_start, src, w0, store);
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..);
                let w0 = _mm_loadu_si16(w_ptr.as_ptr() as *const _);
                let bounds_start = bounds.start + jx;
                store = acc_1_dot::<D>(bounds_start, src, w0, store);
                jx += 1;
            }

            let v_st0 = _mm_reduce_r_epi32::<PRECISION>(store);

            let v_zst1 = _mm_min_epi16(_mm_packus_epi32(v_st0, _mm_setzero_si128()), v_max_colors);
            _mm_storeu_si16(dst as *mut u16 as *mut _, v_zst1);
        }
    }
}
