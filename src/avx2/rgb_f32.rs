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
unsafe fn ch_parts_4_rgb_f32_sse<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m128,
    weight1: __m128,
    weight2: __m128,
    weight3: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..);

    let rgb_pixel_0 = _mm_loadu_ps(src_ptr.as_ptr());
    let rgb_pixel_1 = _mm_loadu_ps(src_ptr.get_unchecked(3..).as_ptr());
    let rgb_pixel_2 = _mm_loadu_ps(src_ptr.get_unchecked(6..).as_ptr());
    let rgb_pixel_3 = _mm_setr_ps(
        *src_ptr.get_unchecked(9),
        *src_ptr.get_unchecked(10),
        *src_ptr.get_unchecked(11),
        0.,
    );

    let acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
    let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
    let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_2, weight2);
    _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_3, weight3)
}

#[inline(always)]
unsafe fn ch_parts_4_rgb_f32_avx<const FMA: bool>(
    start_x: usize,
    src0: &[f32],
    src1: &[f32],
    weight0: __m256,
    weight1: __m256,
    weight2: __m256,
    weight3: __m256,
    store_0: __m256,
) -> __m256 {
    const COMPONENTS: usize = 3;
    let src_ptr0 = src0.get_unchecked(start_x * COMPONENTS..);
    let src_ptr1 = src1.get_unchecked(start_x * COMPONENTS..);

    let rgb_pixel_0_0 = _mm_loadu_ps(src_ptr0.as_ptr());
    let rgb_pixel_0_1 = _mm_loadu_ps(src_ptr0.get_unchecked(3..).as_ptr());
    let rgb_pixel_0_2 = _mm_loadu_ps(src_ptr0.get_unchecked(6..).as_ptr());
    let rgb_pixel_0_3 = _mm_setr_ps(
        *src_ptr0.get_unchecked(9),
        *src_ptr0.get_unchecked(10),
        *src_ptr0.get_unchecked(11),
        0.,
    );

    let rgb_pixel_1_0 = _mm_loadu_ps(src_ptr1.as_ptr());
    let rgb_pixel_1_1 = _mm_loadu_ps(src_ptr1.get_unchecked(3..).as_ptr());
    let rgb_pixel_1_2 = _mm_loadu_ps(src_ptr1.get_unchecked(6..).as_ptr());
    let rgb_pixel_1_3 = _mm_setr_ps(
        *src_ptr1.get_unchecked(9),
        *src_ptr1.get_unchecked(10),
        *src_ptr1.get_unchecked(11),
        0.,
    );

    let rgb_pixel_0 =
        _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_0), rgb_pixel_1_0);
    let rgb_pixel_1 =
        _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_1), rgb_pixel_1_1);
    let rgb_pixel_2 =
        _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_2), rgb_pixel_1_2);
    let rgb_pixel_3 =
        _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_3), rgb_pixel_1_3);

    let acc = _mm256_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
    let acc = _mm256_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
    let acc = _mm256_prefer_fma_ps::<FMA>(acc, rgb_pixel_2, weight2);
    _mm256_prefer_fma_ps::<FMA>(acc, rgb_pixel_3, weight3)
}

#[inline(always)]
unsafe fn ch_parts_2_rgb_f32_avx<const FMA: bool>(
    start_x: usize,
    src0: &[f32],
    src1: &[f32],
    weight0: __m256,
    weight1: __m256,
    store_0: __m256,
) -> __m256 {
    const COMPONENTS: usize = 3;
    let src_ptr0 = src0.get_unchecked(start_x * COMPONENTS..);
    let src_ptr1 = src1.get_unchecked(start_x * COMPONENTS..);

    let orig0 = _mm_loadu_ps(src_ptr0.as_ptr());
    let orig1 = _mm_loadu_ps(src_ptr1.as_ptr());

    let rgb_pixel_0_0 = orig0;
    let rgb_pixel_0_1 = _mm_setr_ps(
        *src_ptr0.get_unchecked(3),
        *src_ptr0.get_unchecked(4),
        *src_ptr0.get_unchecked(5),
        0.,
    );

    let rgb_pixel_1_0 = orig1;
    let rgb_pixel_1_1 = _mm_setr_ps(
        *src_ptr1.get_unchecked(3),
        *src_ptr1.get_unchecked(4),
        *src_ptr1.get_unchecked(5),
        0.,
    );

    let rgb_pixel_0 =
        _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_0), rgb_pixel_1_0);
    let rgb_pixel_1 =
        _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel_0_1), rgb_pixel_1_1);

    let mut acc = _mm256_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
    acc = _mm256_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
    acc
}

#[inline(always)]
unsafe fn ch_parts_2_rgb_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m128,
    weight1: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..);

    let orig1 = _mm_loadu_ps(src_ptr.as_ptr());
    let rgb_pixel_0 = orig1;
    let rgb_pixel_1 = _mm_setr_ps(
        *src_ptr.get_unchecked(3),
        *src_ptr.get_unchecked(4),
        *src_ptr.get_unchecked(5),
        0.,
    );

    let mut acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
    acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
    acc
}

#[inline(always)]
unsafe fn ch_parts_one_rgb_f32<const FMA: bool>(
    start_x: usize,
    src: &[f32],
    weight0: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.get_unchecked(start_x * COMPONENTS..).as_ptr();
    let rgb_pixel = _mm_setr_ps(
        src_ptr.add(0).read_unaligned(),
        src_ptr.add(1).read_unaligned(),
        src_ptr.add(2).read_unaligned(),
        0f32,
    );
    _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel, weight0)
}

#[inline(always)]
unsafe fn ch_parts_one_rgb_f32_avx<const FMA: bool>(
    start_x: usize,
    src0: &[f32],
    src1: &[f32],
    weight0: __m256,
    store_0: __m256,
) -> __m256 {
    const COMPONENTS: usize = 3;
    let src_ptr0 = src0.get_unchecked(start_x * COMPONENTS..);
    let src_ptr1 = src1.get_unchecked(start_x * COMPONENTS..);

    let rgb_pixel0 = _mm_setr_ps(
        *src_ptr0.get_unchecked(0),
        *src_ptr0.get_unchecked(1),
        *src_ptr0.get_unchecked(2),
        0.,
    );

    let rgb_pixel1 = _mm_setr_ps(
        *src_ptr1.get_unchecked(0),
        *src_ptr1.get_unchecked(1),
        *src_ptr1.get_unchecked(2),
        0.,
    );

    let rgb_pixel = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel0), rgb_pixel1);

    _mm256_prefer_fma_ps::<FMA>(store_0, rgb_pixel, weight0)
}

pub(crate) fn convolve_horizontal_rgb_avx_row_one_f32<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
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
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = ExecutionUnit1Row::<false>::default();
    unit.pass(dst_width, src_width, filter_weights, src, dst);
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgb_avx_row_one_f32_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    let unit = ExecutionUnit1Row::<true>::default();
    unit.pass(dst_width, src_width, filter_weights, src, dst);
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit1Row<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit1Row<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        dst_width: usize,
        _: usize,
        filter_weights: &FilterWeights<f32>,
        src: &[f32],
        dst: &mut [f32],
    ) {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;
        let weights = &filter_weights.weights;

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = _mm_setzero_ps();

            while jx + 4 < bounds.size {
                let ptr = weights.get_unchecked(jx + filter_offset..);
                let weight0 = _mm_broadcast_ss(ptr.get_unchecked(0));
                let weight1 = _mm_broadcast_ss(ptr.get_unchecked(1));
                let weight2 = _mm_broadcast_ss(ptr.get_unchecked(2));
                let weight3 = _mm_broadcast_ss(ptr.get_unchecked(3));

                let filter_start = jx + bounds.start;
                store = ch_parts_4_rgb_f32_sse::<FMA>(
                    filter_start,
                    src,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights.get_unchecked(jx + filter_offset..);
                let weight0 = _mm_broadcast_ss(ptr.get_unchecked(0));
                let weight1 = _mm_broadcast_ss(ptr.get_unchecked(1));
                let filter_start = jx + bounds.start;
                store = ch_parts_2_rgb_f32::<FMA>(filter_start, src, weight0, weight1, store);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights.get_unchecked(jx + filter_offset..);
                let weight0 = _mm_broadcast_ss(ptr.get_unchecked(0));
                let filter_start = jx + bounds.start;
                store = ch_parts_one_rgb_f32::<FMA>(filter_start, src, weight0, store);
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(store));
            (dest_ptr as *mut i32)
                .add(2)
                .write_unaligned(_mm_extract_ps::<2>(store));

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_avx_rows_4_f32<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        if FMA {
            convolve_horizontal_rgb_avx_rows_4_f32_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        } else {
            convolve_horizontal_rgb_avx_rows_4_f32_regular(
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
unsafe fn convolve_horizontal_rgb_avx_rows_4_f32_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
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

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch
unsafe fn convolve_horizontal_rgb_avx_rows_4_f32_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
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

#[derive(Copy, Clone, Default)]
struct ExecutionUnit4Row<const FMA: bool> {}

impl<const FMA: bool> ExecutionUnit4Row<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        dst_width: usize,
        _: usize,
        filter_weights: &FilterWeights<f32>,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        dst_stride: usize,
    ) {
        const CHANNELS: usize = 3;
        let mut filter_offset = 0usize;

        let weights_ptr = &filter_weights.weights;

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = _mm256_setzero_ps();
            let mut store_1 = _mm256_setzero_ps();

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);

                let weight0 = _mm256_broadcast_ss(ptr.get_unchecked(0));
                let weight1 = _mm256_broadcast_ss(ptr.get_unchecked(1));
                let weight2 = _mm256_broadcast_ss(ptr.get_unchecked(2));
                let weight3 = _mm256_broadcast_ss(ptr.get_unchecked(3));

                let filter_start = jx + bounds.start;
                store_0 = ch_parts_4_rgb_f32_avx::<FMA>(
                    filter_start,
                    src,
                    src.get_unchecked(src_stride..),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_0,
                );
                store_1 = ch_parts_4_rgb_f32_avx::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..),
                    src.get_unchecked(src_stride * 3..),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_1,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let weight0 = _mm256_broadcast_ss(ptr.get_unchecked(0));
                let weight1 = _mm256_broadcast_ss(ptr.get_unchecked(1));
                let filter_start = jx + bounds.start;
                store_0 = ch_parts_2_rgb_f32_avx::<FMA>(
                    filter_start,
                    src,
                    src.get_unchecked(src_stride..),
                    weight0,
                    weight1,
                    store_0,
                );
                store_1 = ch_parts_2_rgb_f32_avx::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..),
                    src.get_unchecked(src_stride * 3..),
                    weight0,
                    weight1,
                    store_1,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let weight0 = _mm256_broadcast_ss(ptr.get_unchecked(0));
                let filter_start = jx + bounds.start;
                store_0 = ch_parts_one_rgb_f32_avx::<FMA>(
                    filter_start,
                    src,
                    src.get_unchecked(src_stride..),
                    weight0,
                    store_0,
                );
                store_1 = ch_parts_one_rgb_f32_avx::<FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..),
                    src.get_unchecked(src_stride * 3..),
                    weight0,
                    store_1,
                );
                jx += 1;
            }

            let px = x * CHANNELS;
            let dest_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
            _mm_storeu_si64(
                dest_ptr as *mut u8,
                _mm_castps_si128(_mm256_castps256_ps128(store_0)),
            );
            (dest_ptr as *mut i32)
                .add(2)
                .write_unaligned(_mm_extract_ps::<2>(_mm256_castps256_ps128(store_0)));

            let ss1 = _mm256_extractf128_ps::<1>(store_0);

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride..).as_mut_ptr();
            _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(ss1));
            (dest_ptr as *mut i32)
                .add(2)
                .write_unaligned(_mm_extract_ps::<2>(ss1));

            let ss2 = _mm256_castps256_ps128(store_1);

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2..).as_mut_ptr();
            _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(ss2));
            (dest_ptr as *mut i32)
                .add(2)
                .write_unaligned(_mm_extract_ps::<2>(ss2));

            let ss3 = _mm256_extractf128_ps::<1>(store_1);

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3..).as_mut_ptr();
            _mm_storeu_si64(dest_ptr as *mut u8, _mm_castps_si128(ss3));
            (dest_ptr as *mut i32)
                .add(2)
                .write_unaligned(_mm_extract_ps::<2>(ss3));

            filter_offset += filter_weights.aligned_size;
        }
    }
}
