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

use crate::avx2::utils::_mm256_fma_pd;
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn ld4_rgb(src: &[f32]) -> (__m128, __m128, __m128, __m128) {
    unsafe {
        let px0 = _mm_loadu_ps(src.as_ptr());
        let px1 = _mm_loadu_ps(src.get_unchecked(3..).as_ptr());
        let px2 = _mm_loadu_ps(src.get_unchecked(6..).as_ptr());
        let px3 = _mm_setr_ps(
            *src.get_unchecked(9),
            *src.get_unchecked(10),
            *src.get_unchecked(11),
            0.,
        );
        (px0, px1, px2, px3)
    }
}

#[inline(always)]
unsafe fn ld2_rgb(src: &[f32]) -> (__m128, __m128) {
    unsafe {
        let px0 = _mm_loadu_ps(src.as_ptr());
        let px1 = _mm_setr_ps(
            *src.get_unchecked(3),
            *src.get_unchecked(4),
            *src.get_unchecked(5),
            0.,
        );
        (px0, px1)
    }
}

#[inline(always)]
unsafe fn ld1_rgb(src: &[f32]) -> __m128 {
    unsafe {
        _mm_setr_ps(
            *src.get_unchecked(0),
            *src.get_unchecked(1),
            *src.get_unchecked(2),
            0.,
        )
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_row_one_f32_f64<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgb_avx_row_one_f32_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        } else {
            convolve_horizontal_rgb_avx_row_one_f32_regular(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgb_avx_row_one_f32_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        let unit = ExecutionUnit1Row::<false>::default();
        unit.pass(dst_width, src_width, filter_weights, src, dst);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgb_avx_row_one_f32_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        let unit = ExecutionUnit1Row::<true>::default();
        unit.pass(dst_width, src_width, filter_weights, src, dst);
    }
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit1Row<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit1Row<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        dst_width: usize,
        _: usize,
        filter_weights: &FilterWeights<f64>,
        src: &[f32],
        dst: &mut [f32],
    ) {
        unsafe {
            const CN: usize = 3;
            let mut filter_offset = 0usize;
            let weights = &filter_weights.weights;

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store = _mm256_setzero_pd();

                while jx + 4 < bounds.size {
                    let ptr = weights.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm256_set1_pd(*ptr.get_unchecked(0));
                    let weight1 = _mm256_set1_pd(*ptr.get_unchecked(1));
                    let weight2 = _mm256_set1_pd(*ptr.get_unchecked(2));
                    let weight3 = _mm256_set1_pd(*ptr.get_unchecked(3));

                    let filter_start = jx + bounds.start;
                    let px = ld4_rgb(src.get_unchecked(filter_start * CN..));
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(px.0), weight0);
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(px.1), weight1);
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(px.2), weight2);
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(px.3), weight3);
                    jx += 4;
                }

                while jx + 2 < bounds.size {
                    let ptr = weights.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm256_set1_pd(*ptr.get_unchecked(0));
                    let weight1 = _mm256_set1_pd(*ptr.get_unchecked(1));
                    let filter_start = jx + bounds.start;
                    let px = ld2_rgb(src.get_unchecked(filter_start * CN..));
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(px.0), weight0);
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(px.1), weight1);
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = weights.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm256_set1_pd(*ptr.get_unchecked(0));
                    let filter_start = jx + bounds.start;
                    let px = ld1_rgb(src.get_unchecked(filter_start * CN..));
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(px), weight0);
                    jx += 1;
                }

                let z = _mm256_cvtpd_ps(store);

                let px = x * CN;
                let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
                _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(z));
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(z));

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_rows_4_f32_f64<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgb_avx_rows_4_f32_f64_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        } else {
            convolve_horizontal_rgb_avx_rows_4_f32_f64_regular(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgb_avx_rows_4_f32_f64_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let unit = ExecutionUnit4Row::<false>::default();
        unit.pass(
            dst_width,
            src_width,
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgb_avx_rows_4_f32_f64_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let unit = ExecutionUnit4Row::<true>::default();
        unit.pass(
            dst_width,
            src_width,
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit4Row<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit4Row<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        dst_width: usize,
        _: usize,
        filter_weights: &FilterWeights<f64>,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        dst_stride: usize,
    ) {
        unsafe {
            const CN: usize = 3;
            let mut filter_offset = 0usize;

            let weights_ptr = &filter_weights.weights;

            let src1 = src.get_unchecked(src_stride..);
            let src2 = src.get_unchecked(src_stride * 2..);
            let src3 = src.get_unchecked(src_stride * 3..);

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store_0 = _mm256_setzero_pd();
                let mut store_1 = _mm256_setzero_pd();
                let mut store_2 = _mm256_setzero_pd();
                let mut store_3 = _mm256_setzero_pd();

                while jx + 2 < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm256_set1_pd(*ptr.get_unchecked(0));
                    let weight1 = _mm256_set1_pd(*ptr.get_unchecked(1));
                    let filter_start = jx + bounds.start;
                    let px0 = ld2_rgb(src.get_unchecked(filter_start * CN..));
                    let px1 = ld2_rgb(src1.get_unchecked(filter_start * CN..));
                    let px2 = ld2_rgb(src2.get_unchecked(filter_start * CN..));
                    let px3 = ld2_rgb(src3.get_unchecked(filter_start * CN..));

                    store_0 = _mm256_fma_pd::<FMA>(store_0, _mm256_cvtps_pd(px0.0), weight0);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, _mm256_cvtps_pd(px1.0), weight0);
                    store_2 = _mm256_fma_pd::<FMA>(store_2, _mm256_cvtps_pd(px2.0), weight0);
                    store_3 = _mm256_fma_pd::<FMA>(store_3, _mm256_cvtps_pd(px3.0), weight0);

                    store_0 = _mm256_fma_pd::<FMA>(store_0, _mm256_cvtps_pd(px0.1), weight1);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, _mm256_cvtps_pd(px1.1), weight1);
                    store_2 = _mm256_fma_pd::<FMA>(store_2, _mm256_cvtps_pd(px2.1), weight1);
                    store_3 = _mm256_fma_pd::<FMA>(store_3, _mm256_cvtps_pd(px3.1), weight1);
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm256_set1_pd(*ptr.get_unchecked(0));
                    let filter_start = jx + bounds.start;
                    let px0 = ld1_rgb(src.get_unchecked(filter_start * CN..));
                    let px1 = ld1_rgb(src1.get_unchecked(filter_start * CN..));
                    let px2 = ld1_rgb(src2.get_unchecked(filter_start * CN..));
                    let px3 = ld1_rgb(src3.get_unchecked(filter_start * CN..));

                    store_0 = _mm256_fma_pd::<FMA>(store_0, _mm256_cvtps_pd(px0), weight0);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, _mm256_cvtps_pd(px1), weight0);
                    store_2 = _mm256_fma_pd::<FMA>(store_2, _mm256_cvtps_pd(px2), weight0);
                    store_3 = _mm256_fma_pd::<FMA>(store_3, _mm256_cvtps_pd(px3), weight0);

                    jx += 1;
                }

                let z0 = _mm256_cvtpd_ps(store_0);
                let z1 = _mm256_cvtpd_ps(store_1);
                let z2 = _mm256_cvtpd_ps(store_2);
                let z3 = _mm256_cvtpd_ps(store_3);

                let px = x * CN;
                let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
                _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(z0));
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(z0));

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
                _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(z1));
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(z1));

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
                _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(z2));
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(z2));

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
                _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(z3));
                (dest_ptr as *mut i32)
                    .add(2)
                    .write_unaligned(_mm_extract_ps::<2>(z3));

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}
