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

use crate::filter_weights::FilterWeights;
use crate::sse::{_mm_hsum_ps, _mm_prefer_fma_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

macro_rules! conv_horiz_plane_16_f32 {
    ($start_x: expr, $src: expr, $set: expr, $store: expr, $fma: expr) => {{
        let src_ptr = $src.add($start_x);

        let rgb_pixel0 = _mm_loadu_ps(src_ptr);
        let rgb_pixel1 = _mm_loadu_ps(src_ptr.add(4));
        let rgb_pixel2 = _mm_loadu_ps(src_ptr.add(8));
        let rgb_pixel3 = _mm_loadu_ps(src_ptr.add(12));

        let mut acc = _mm_prefer_fma_ps::<$fma>($store, rgb_pixel0, $set.0);
        acc = _mm_prefer_fma_ps::<$fma>(acc, rgb_pixel1, $set.1);
        acc = _mm_prefer_fma_ps::<$fma>(acc, rgb_pixel2, $set.2);
        acc = _mm_prefer_fma_ps::<$fma>(acc, rgb_pixel3, $set.3);
        acc
    }};
}

macro_rules! conv_horiz_plane_8_f32 {
    ($start_x: expr, $src: expr, $set1: expr, $set2: expr, $store: expr, $fma: expr) => {{
        let src_ptr = $src.add($start_x);

        let rgb_pixel0 = _mm_loadu_ps(src_ptr);
        let rgb_pixel1 = _mm_loadu_ps(src_ptr.add(4));

        let mut acc = _mm_prefer_fma_ps::<$fma>($store, rgb_pixel0, $set1);
        acc = _mm_prefer_fma_ps::<$fma>(acc, rgb_pixel1, $set2);
        acc
    }};
}

macro_rules! conv_horiz_plane_4_f32 {
    ($start_x: expr, $src: expr, $set1: expr,  $store: expr, $fma: expr) => {{
        let src_ptr = $src.add($start_x);

        let rgb_pixel = _mm_loadu_ps(src_ptr);

        _mm_prefer_fma_ps::<$fma>($store, rgb_pixel, $set1)
    }};
}

macro_rules! conv_horiz_plane_2_f32 {
    ($start_x: expr, $src: expr, $set: expr,  $store: expr, $fma: expr) => {{
        let src_ptr = $src.add($start_x);

        let rgb_pixel = _mm_setr_ps(
            src_ptr.read_unaligned(),
            src_ptr.add(1).read_unaligned(),
            0.,
            0.,
        );

        _mm_prefer_fma_ps::<$fma>($store, rgb_pixel, $set)
    }};
}

macro_rules! conv_horiz_plane_1_f32 {
    ($start_x: expr, $src: expr, $set: expr,  $store: expr, $fma: expr) => {{
        let src_ptr = $src.add($start_x);
        let rgb_pixel = _mm_setr_ps(src_ptr.read_unaligned(), 0., 0., 0.);
        _mm_prefer_fma_ps::<$fma>($store, rgb_pixel, $set)
    }};
}

pub fn convolve_horizontal_plane_sse_row_one<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    unsafe {
        if FMA {
            convolve_horizontal_plane_sse_row_one_fma(
                dst_width,
                src_width,
                filter_weights,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
            );
        } else {
            convolve_horizontal_plane_sse_row_one_regular(
                dst_width,
                src_width,
                filter_weights,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_plane_sse_row_one_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    convolve_horizontal_plane_sse_row_one_impl::<false>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        unsafe_destination_ptr_0,
    );
}

#[inline]
#[target_feature(enable = "sse4.1,fma")]
unsafe fn convolve_horizontal_plane_sse_row_one_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    convolve_horizontal_plane_sse_row_one_impl::<true>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        unsafe_destination_ptr_0,
    );
}

#[inline]
unsafe fn convolve_horizontal_plane_sse_row_one_impl<const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    let mut filter_offset = 0usize;
    let weights_ptr = filter_weights.weights.as_ptr();

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store = _mm_setzero_ps();

        while jx + 16 < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights0 = _mm_loadu_ps(ptr);
            let read_weights1 = _mm_loadu_ps(ptr.add(4));
            let read_weights2 = _mm_loadu_ps(ptr.add(8));
            let read_weights3 = _mm_loadu_ps(ptr.add(12));
            let weights = (read_weights0, read_weights1, read_weights2, read_weights3);
            store =
                conv_horiz_plane_16_f32!(bounds_start, unsafe_source_ptr_0, weights, store, FMA);
            jx += 8;
        }

        while jx + 8 < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights0 = _mm_loadu_ps(ptr);
            let read_weights1 = _mm_loadu_ps(ptr.add(4));
            let read_weights = (read_weights0, read_weights1);
            store = conv_horiz_plane_8_f32!(
                bounds_start,
                unsafe_source_ptr_0,
                read_weights.0,
                read_weights.1,
                store,
                FMA
            );
            jx += 8;
        }

        while jx + 4 < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.add(jx + filter_offset);
            let read_weights = _mm_loadu_ps(ptr);
            store = conv_horiz_plane_4_f32!(
                bounds_start,
                unsafe_source_ptr_0,
                read_weights,
                store,
                FMA
            );
            jx += 4;
        }

        while jx + 2 < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.add(jx + filter_offset);
            let weights = _mm_setr_ps(ptr.read_unaligned(), ptr.add(1).read_unaligned(), 0., 0.);
            store = conv_horiz_plane_2_f32!(bounds_start, unsafe_source_ptr_0, weights, store, FMA);
            jx += 2;
        }

        while jx < bounds.size {
            let bounds_start = bounds.start + jx;
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight0 = _mm_load1_ps(ptr);
            store = conv_horiz_plane_1_f32!(bounds_start, unsafe_source_ptr_0, weight0, store, FMA);
            jx += 1;
        }

        let px = x;
        let dest_ptr = unsafe_destination_ptr_0.add(px);
        dest_ptr.write_unaligned(_mm_hsum_ps(store));

        filter_offset += filter_weights.aligned_size;
    }
}

pub fn convolve_horizontal_plane_sse_rows_4<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f32,
    dst_stride: usize,
) {
    unsafe {
        if FMA {
            convolve_horizontal_plane_sse_rows_4_fma(
                dst_width,
                src_width,
                filter_weights,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                dst_stride,
            );
        } else {
            convolve_horizontal_plane_sse_rows_4_regular(
                dst_width,
                src_width,
                filter_weights,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                dst_stride,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_plane_sse_rows_4_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f32,
    dst_stride: usize,
) {
    convolve_horizontal_plane_sse_rows_4_impl::<false>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        src_stride,
        unsafe_destination_ptr_0,
        dst_stride,
    );
}

#[inline]
#[target_feature(enable = "sse4.1,fma")]
unsafe fn convolve_horizontal_plane_sse_rows_4_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f32,
    dst_stride: usize,
) {
    convolve_horizontal_plane_sse_rows_4_impl::<true>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        src_stride,
        unsafe_destination_ptr_0,
        dst_stride,
    );
}

#[inline]
unsafe fn convolve_horizontal_plane_sse_rows_4_impl<const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f32,
    dst_stride: usize,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let zeros = _mm_setzero_ps();
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 16 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights0 = _mm_loadu_ps(ptr);
                let read_weights1 = _mm_loadu_ps(ptr.add(4));
                let read_weights2 = _mm_loadu_ps(ptr.add(8));
                let read_weights3 = _mm_loadu_ps(ptr.add(12));
                let weights = (read_weights0, read_weights1, read_weights2, read_weights3);
                let bounds_start = bounds.start + jx;
                store_0 =
                    conv_horiz_plane_16_f32!(bounds_start, unsafe_source_ptr_0, weights, store_0, FMA);
                let s_ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_plane_16_f32!(bounds_start, s_ptr_1, weights, store_1, FMA);
                let s_ptr2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_plane_16_f32!(bounds_start, s_ptr2, weights, store_2, FMA);
                let s_ptr3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_plane_16_f32!(bounds_start, s_ptr3, weights, store_3, FMA);
                jx += 16;
            }

            while jx + 8 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights0 = _mm_loadu_ps(ptr);
                let read_weights1 = _mm_loadu_ps(ptr.add(4));
                let read_weights = (read_weights0, read_weights1);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_8_f32!(
                    bounds_start,
                    unsafe_source_ptr_0,
                    read_weights.0,
                    read_weights.1,
                    store_0,
                    FMA
                );
                let s_ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_plane_8_f32!(
                    bounds_start,
                    s_ptr_1,
                    read_weights.0,
                    read_weights.1,
                    store_1,
                    FMA
                );
                let s_ptr2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_plane_8_f32!(
                    bounds_start,
                    s_ptr2,
                    read_weights.0,
                    read_weights.1,
                    store_2,
                    FMA
                );
                let s_ptr3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_plane_8_f32!(
                    bounds_start,
                    s_ptr3,
                    read_weights.0,
                    read_weights.1,
                    store_3,
                    FMA
                );
                jx += 8;
            }

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let read_weights = _mm_loadu_ps(ptr);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_4_f32!(
                    bounds_start,
                    unsafe_source_ptr_0,
                    read_weights,
                    store_0,
                    FMA
                );
                let s_ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 =
                    conv_horiz_plane_4_f32!(bounds_start, s_ptr_1, read_weights, store_1, FMA);
                let s_ptr2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_plane_4_f32!(bounds_start, s_ptr2, read_weights, store_2, FMA);
                let s_ptr3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_plane_4_f32!(bounds_start, s_ptr3, read_weights, store_3, FMA);
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weights =
                    _mm_setr_ps(ptr.read_unaligned(), ptr.add(1).read_unaligned(), 0., 0.);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_2_f32!(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weights,
                    store_0,
                    FMA
                );
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_plane_2_f32!(bounds_start, ptr_1, weights, store_1, FMA);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_plane_2_f32!(bounds_start, ptr_2, weights, store_2, FMA);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_plane_2_f32!(bounds_start, ptr_3, weights, store_3, FMA);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = _mm_set1_ps(ptr.read_unaligned());
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_1_f32!(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    store_0,
                    FMA
                );
                let ptr_1 = unsafe_source_ptr_0.add(src_stride);
                store_1 = conv_horiz_plane_1_f32!(bounds_start, ptr_1, weight0, store_1, FMA);
                let ptr_2 = unsafe_source_ptr_0.add(src_stride * 2);
                store_2 = conv_horiz_plane_1_f32!(bounds_start, ptr_2, weight0, store_2, FMA);
                let ptr_3 = unsafe_source_ptr_0.add(src_stride * 3);
                store_3 = conv_horiz_plane_1_f32!(bounds_start, ptr_3, weight0, store_3, FMA);
                jx += 1;
            }

            let px = x;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            dest_ptr.write_unaligned(_mm_hsum_ps(store_0));

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            dest_ptr.write_unaligned(_mm_hsum_ps(store_1));

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            dest_ptr.write_unaligned(_mm_hsum_ps(store_2));

            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            dest_ptr.write_unaligned(_mm_hsum_ps(store_3));

            filter_offset += filter_weights.aligned_size;
        }
    }
}
