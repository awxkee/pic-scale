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

use crate::avx2::utils::_mm_hsum_ps;
use crate::filter_weights::FilterWeights;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

macro_rules! conv_horiz_plane_4_f32 {
    ($start_x: expr, $src: expr, $set1: expr,  $store: expr, $fma: expr) => {{
        let src_ptr = $src.get_unchecked($start_x..).as_ptr();

        let rgb_pixel = _mm_loadu_ps(src_ptr);

        _mm_prefer_fma_ps::<$fma>($store, rgb_pixel, $set1)
    }};
}

macro_rules! conv_horiz_plane_4_f32_avx {
    ($start_x: expr, $src0: expr, $src1: expr, $set1: expr,  $store: expr, $fma: expr) => {{
        let src_ptr0 = $src0.get_unchecked($start_x..).as_ptr();
        let src_ptr1 = $src1.get_unchecked($start_x..).as_ptr();

        let rgb_pixel0 = _mm_loadu_ps(src_ptr0);
        let rgb_pixel1 = _mm_loadu_ps(src_ptr1);

        let rgb_pixel = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel0), rgb_pixel1);

        _mm256_prefer_fma_ps::<$fma>($store, rgb_pixel, $set1)
    }};
}

macro_rules! conv_horiz_plane_2_f32 {
    ($start_x: expr, $src: expr, $set: expr,  $store: expr, $fma: expr) => {{
        let src_ptr = $src.get_unchecked($start_x..).as_ptr();

        let rgb_pixel = _mm_castsi128_ps(_mm_loadu_si64(src_ptr as *const _));

        _mm_prefer_fma_ps::<$fma>($store, rgb_pixel, $set)
    }};
}

macro_rules! conv_horiz_plane_2_f32_avx {
    ($start_x: expr, $src0: expr, $src1: expr, $set: expr,  $store: expr, $fma: expr) => {{
        let src_ptr0 = $src0.get_unchecked($start_x..).as_ptr();
        let src_ptr1 = $src1.get_unchecked($start_x..).as_ptr();

        let rgb_pixel0 = _mm_castsi128_ps(_mm_loadu_si64(src_ptr0 as *const _));
        let rgb_pixel1 = _mm_castsi128_ps(_mm_loadu_si64(src_ptr1 as *const _));

        let rgb_pixel = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel0), rgb_pixel1);

        _mm256_prefer_fma_ps::<$fma>($store, rgb_pixel, $set)
    }};
}

macro_rules! conv_horiz_plane_1_f32 {
    ($start_x: expr, $src: expr, $set: expr,  $store: expr, $fma: expr) => {{
        let src_ptr = $src.get_unchecked($start_x..).as_ptr();
        let rgb_pixel = _mm_load_ss(src_ptr);
        _mm_prefer_fma_ps::<$fma>($store, rgb_pixel, $set)
    }};
}

macro_rules! conv_horiz_plane_1_f32_avx {
    ($start_x: expr, $src0: expr, $src1: expr, $set: expr,  $store: expr, $fma: expr) => {{
        let src_ptr0 = $src0.get_unchecked($start_x..).as_ptr();
        let src_ptr1 = $src1.get_unchecked($start_x..).as_ptr();

        let rgb_pixel0 = _mm_load_ss(src_ptr0);
        let rgb_pixel1 = _mm_load_ss(src_ptr1);

        let rgb_pixel = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(rgb_pixel0), rgb_pixel1);

        _mm256_prefer_fma_ps::<$fma>($store, rgb_pixel, $set)
    }};
}

pub(crate) fn convolve_horizontal_plane_avx_row_one_f32<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        if FMA {
            convolve_horizontal_plane_avx_row_one_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        } else {
            convolve_horizontal_plane_avx_row_one_regular(
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
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_plane_avx_row_one_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    convolve_horizontal_plane_avx_row_one_impl::<false>(
        dst_width,
        src_width,
        filter_weights,
        src,
        dst,
    );
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_plane_avx_row_one_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    convolve_horizontal_plane_avx_row_one_impl::<true>(
        dst_width,
        src_width,
        filter_weights,
        src,
        dst,
    );
}

#[inline(always)]
unsafe fn convolve_horizontal_plane_avx_row_one_impl<const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    let mut filter_offset = 0usize;
    let weights_ptr = &filter_weights.weights;

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store = _mm_setzero_ps();

        use crate::avx2::utils::_mm_prefer_fma_ps;

        while jx + 4 < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
            let read_weights = _mm_loadu_ps(ptr.as_ptr());
            store = conv_horiz_plane_4_f32!(bounds_start, src, read_weights, store, FMA);
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let bounds_start = bounds.start + jx;
            let w = weights_ptr.get_unchecked(jx + filter_offset..);
            let weights = _mm_castsi128_ps(_mm_loadu_si64(w.as_ptr() as *const _));
            store = conv_horiz_plane_2_f32!(bounds_start, src, weights, store, FMA);
            jx += 2;
        }

        while jx < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
            let weight0 = _mm_load_ss(ptr.as_ptr());
            store = conv_horiz_plane_1_f32!(bounds_start, src, weight0, store, FMA);
            jx += 1;
        }

        let px = x;
        let dest_ptr = dst.get_unchecked_mut(px);
        _mm_store_ss(dest_ptr, _mm_hsum_ps(store));

        filter_offset += filter_weights.aligned_size;
    }
}

pub(crate) fn convolve_horizontal_plane_avx_rows_4_f32<const FMA: bool>(
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
            convolve_horizontal_plane_avx_rows_4_fma(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        } else {
            convolve_horizontal_plane_avx_rows_4_regular(
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
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_plane_avx_rows_4_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    convolve_horizontal_plane_avx_rows_4_impl::<false>(
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
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_plane_avx_rows_4_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    convolve_horizontal_plane_avx_rows_4_impl::<true>(
        dst_width,
        src_width,
        filter_weights,
        src,
        src_stride,
        dst,
        dst_stride,
    );
}

#[inline(always)]
unsafe fn convolve_horizontal_plane_avx_rows_4_impl<const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let mut filter_offset = 0usize;
    let weights_ptr = &filter_weights.weights;

    use crate::avx2::utils::_mm256_prefer_fma_ps;

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store_0 = _mm256_setzero_ps();
        let mut store_1 = _mm256_setzero_ps();

        while jx + 4 < bounds.size {
            let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
            let w0 = _mm_loadu_ps(ptr.as_ptr());

            let weights = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w0);

            let bounds_start = bounds.start + jx;
            let s_ptr_1 = src.get_unchecked(src_stride..);
            store_0 =
                conv_horiz_plane_4_f32_avx!(bounds_start, src, s_ptr_1, weights, store_0, FMA);
            let s_ptr2 = src.get_unchecked(src_stride * 2..);
            let s_ptr3 = src.get_unchecked(src_stride * 3..);
            store_1 =
                conv_horiz_plane_4_f32_avx!(bounds_start, s_ptr2, s_ptr3, weights, store_1, FMA);
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w = weights_ptr.get_unchecked(jx + filter_offset..);
            let w0 = _mm_castsi128_ps(_mm_loadu_si64(w.as_ptr() as *const _));
            let weights = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w0);
            let bounds_start = bounds.start + jx;
            let s_ptr_1 = src.get_unchecked(src_stride..);
            store_0 =
                conv_horiz_plane_2_f32_avx!(bounds_start, src, s_ptr_1, weights, store_0, FMA);
            let s_ptr2 = src.get_unchecked(src_stride * 2..);
            let s_ptr3 = src.get_unchecked(src_stride * 3..);
            store_1 =
                conv_horiz_plane_2_f32_avx!(bounds_start, s_ptr2, s_ptr3, weights, store_1, FMA);
            jx += 2;
        }

        while jx < bounds.size {
            let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
            let w0 = _mm_load_ss(ptr.as_ptr());
            let weights = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(w0), w0);

            let bounds_start = bounds.start + jx;
            let s_ptr_1 = src.get_unchecked(src_stride..);
            store_0 =
                conv_horiz_plane_1_f32_avx!(bounds_start, src, s_ptr_1, weights, store_0, FMA);
            let s_ptr2 = src.get_unchecked(src_stride * 2..);
            let s_ptr3 = src.get_unchecked(src_stride * 3..);
            store_1 =
                conv_horiz_plane_1_f32_avx!(bounds_start, s_ptr2, s_ptr3, weights, store_1, FMA);
            jx += 1;
        }

        let px = x;
        let dest_ptr = dst.get_unchecked_mut(px);
        _mm_store_ss(dest_ptr, _mm_hsum_ps(_mm256_castps256_ps128(store_0)));

        let dest_ptr = dst.get_unchecked_mut(px + dst_stride);
        _mm_store_ss(dest_ptr, _mm_hsum_ps(_mm256_extractf128_ps::<1>(store_0)));

        let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2);
        _mm_store_ss(dest_ptr, _mm_hsum_ps(_mm256_castps256_ps128(store_1)));

        let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3);
        _mm_store_ss(dest_ptr, _mm_hsum_ps(_mm256_extractf128_ps::<1>(store_1)));

        filter_offset += filter_weights.aligned_size;
    }
}
