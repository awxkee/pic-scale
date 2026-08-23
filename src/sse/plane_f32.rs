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
        let src_ptr = $src.get_unchecked($start_x..).as_ptr();

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
        let src_ptr = $src.get_unchecked($start_x..).as_ptr();

        let rgb_pixel0 = _mm_loadu_ps(src_ptr);
        let rgb_pixel1 = _mm_loadu_ps(src_ptr.add(4));

        let mut acc = _mm_prefer_fma_ps::<$fma>($store, rgb_pixel0, $set1);
        acc = _mm_prefer_fma_ps::<$fma>(acc, rgb_pixel1, $set2);
        acc
    }};
}

macro_rules! conv_horiz_plane_4_f32 {
    ($start_x: expr, $src: expr, $set1: expr,  $store: expr, $fma: expr) => {{
        let src_ptr = $src.get_unchecked($start_x..).as_ptr();

        let rgb_pixel = _mm_loadu_ps(src_ptr);

        _mm_prefer_fma_ps::<$fma>($store, rgb_pixel, $set1)
    }};
}

macro_rules! conv_horiz_plane_2_f32 {
    ($start_x: expr, $src: expr, $set: expr,  $store: expr, $fma: expr) => {{
        let src_ptr = $src.get_unchecked($start_x..).as_ptr();

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
        let src_ptr = $src.get_unchecked($start_x..).as_ptr();
        let rgb_pixel = _mm_load_ss(src_ptr);
        _mm_prefer_fma_ps::<$fma>($store, rgb_pixel, $set)
    }};
}

pub(crate) fn convolve_horizontal_plane_sse_row_one(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_plane_sse_row_one_regular(filter_weights, src, dst);
    }
}

#[target_feature(enable = "sse4.1")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_horizontal_plane_sse_row_one_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    convolve_horizontal_plane_sse_row_one_impl::<false>(filter_weights, src, dst);
}

#[inline(always)]
fn convolve_horizontal_plane_sse_row_one_impl<const FMA: bool>(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = &filter_weights.weights;

        let dst_width = filter_weights.bounds.len();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = _mm_setzero_ps();

            while jx + 16 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let read_weights0 = _mm_loadu_ps(ptr.as_ptr());
                let read_weights1 = _mm_loadu_ps(ptr.get_unchecked(4..).as_ptr());
                let read_weights2 = _mm_loadu_ps(ptr.get_unchecked(8..).as_ptr());
                let read_weights3 = _mm_loadu_ps(ptr.get_unchecked(12..).as_ptr());
                let weights = (read_weights0, read_weights1, read_weights2, read_weights3);
                store = conv_horiz_plane_16_f32!(bounds_start, src, weights, store, FMA);
                jx += 16;
            }

            while jx + 8 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let read_weights0 = _mm_loadu_ps(ptr.as_ptr());
                let read_weights1 = _mm_loadu_ps(ptr.get_unchecked(4..).as_ptr());
                let read_weights = (read_weights0, read_weights1);
                store = conv_horiz_plane_8_f32!(
                    bounds_start,
                    src,
                    read_weights.0,
                    read_weights.1,
                    store,
                    FMA
                );
                jx += 8;
            }

            while jx + 4 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let read_weights = _mm_loadu_ps(ptr.as_ptr());
                store = conv_horiz_plane_4_f32!(bounds_start, src, read_weights, store, FMA);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let w = weights_ptr.get_unchecked(jx + filter_offset..);
                let weights = _mm_setr_ps(*w.get_unchecked(0), *w.get_unchecked(1), 0., 0.);
                store = conv_horiz_plane_2_f32!(bounds_start, src, weights, store, FMA);
                jx += 2;
            }

            while jx < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let weight0 = _mm_load1_ps(ptr.as_ptr());
                store = conv_horiz_plane_1_f32!(bounds_start, src, weight0, store, FMA);
                jx += 1;
            }

            let px = x;
            let dest_ptr = dst.get_unchecked_mut(px);
            *dest_ptr = _mm_hsum_ps(store);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_plane_sse_rows_4(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_plane_sse_rows_4_regular(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "sse4.1")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_horizontal_plane_sse_rows_4_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    convolve_horizontal_plane_sse_rows_4_impl::<false>(
        filter_weights,
        src,
        src_stride,
        dst,
        dst_stride,
    );
}

#[inline(always)]
fn convolve_horizontal_plane_sse_rows_4_impl<const FMA: bool>(
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let zeros = _mm_setzero_ps();
        let weights_ptr = &filter_weights.weights;

        let dst_width = filter_weights.bounds.len();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 16 <= bounds.size {
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let read_weights0 = _mm_loadu_ps(ptr.as_ptr());
                let read_weights1 = _mm_loadu_ps(ptr.get_unchecked(4..).as_ptr());
                let read_weights2 = _mm_loadu_ps(ptr.get_unchecked(8..).as_ptr());
                let read_weights3 = _mm_loadu_ps(ptr.get_unchecked(12..).as_ptr());
                let weights = (read_weights0, read_weights1, read_weights2, read_weights3);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_16_f32!(bounds_start, src, weights, store_0, FMA);
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_plane_16_f32!(bounds_start, s_ptr_1, weights, store_1, FMA);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_16_f32!(bounds_start, s_ptr2, weights, store_2, FMA);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_plane_16_f32!(bounds_start, s_ptr3, weights, store_3, FMA);
                jx += 16;
            }

            while jx + 8 <= bounds.size {
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let read_weights0 = _mm_loadu_ps(ptr.as_ptr());
                let read_weights1 = _mm_loadu_ps(ptr.get_unchecked(4..).as_ptr());
                let read_weights = (read_weights0, read_weights1);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_8_f32!(
                    bounds_start,
                    src,
                    read_weights.0,
                    read_weights.1,
                    store_0,
                    FMA
                );
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_plane_8_f32!(
                    bounds_start,
                    s_ptr_1,
                    read_weights.0,
                    read_weights.1,
                    store_1,
                    FMA
                );
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_8_f32!(
                    bounds_start,
                    s_ptr2,
                    read_weights.0,
                    read_weights.1,
                    store_2,
                    FMA
                );
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
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

            while jx + 4 <= bounds.size {
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let read_weights = _mm_loadu_ps(ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_4_f32!(bounds_start, src, read_weights, store_0, FMA);
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 =
                    conv_horiz_plane_4_f32!(bounds_start, s_ptr_1, read_weights, store_1, FMA);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_4_f32!(bounds_start, s_ptr2, read_weights, store_2, FMA);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_plane_4_f32!(bounds_start, s_ptr3, read_weights, store_3, FMA);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w = weights_ptr.get_unchecked(jx + filter_offset..);
                let weights = _mm_setr_ps(*w.get_unchecked(0), *w.get_unchecked(1), 0., 0.);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_2_f32!(bounds_start, src, weights, store_0, FMA);
                let ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_plane_2_f32!(bounds_start, ptr_1, weights, store_1, FMA);
                let ptr_2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_2_f32!(bounds_start, ptr_2, weights, store_2, FMA);
                let ptr_3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_plane_2_f32!(bounds_start, ptr_3, weights, store_3, FMA);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                let weight0 = _mm_load1_ps(ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_1_f32!(bounds_start, src, weight0, store_0, FMA);
                let ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_plane_1_f32!(bounds_start, ptr_1, weight0, store_1, FMA);
                let ptr_2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_1_f32!(bounds_start, ptr_2, weight0, store_2, FMA);
                let ptr_3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_plane_1_f32!(bounds_start, ptr_3, weight0, store_3, FMA);
                jx += 1;
            }

            let px = x;
            let dest_ptr = dst.get_unchecked_mut(px);
            *dest_ptr = _mm_hsum_ps(store_0);

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride);
            *dest_ptr = _mm_hsum_ps(store_1);

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2);
            *dest_ptr = _mm_hsum_ps(store_2);

            let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3);
            *dest_ptr = _mm_hsum_ps(store_3);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::ResamplingFunction;
    use crate::convolve_naive_f32::{
        convolve_horizontal_native_row_f32, convolve_horizontal_rgba_4_row_f32,
    };
    use crate::test_utils::{XorShiftRng, assert_f32_slices_close, make_row_filter_weights_f32};

    // Unlike rgba_f32.rs (one channel per SIMD lane, no intra-lane reduction),
    // a single-channel plane accumulates multiple taps within the same lane
    // and needs a horizontal reduction, reordering the sum vs the scalar
    // loop's strict left-to-right accumulation - not bit-exact (observed diffs
    // are a single f32 ULP).
    const ATOL: f32 = 1e-5;

    #[test]
    fn sse_row_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64), (256, 256), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_f32(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xC0FFEE ^ (in_size as u64) << 32 ^ out_size as u64);
                let src = rng.fill_f32_unit(in_size);

                let mut dst_scalar = vec![0f32; out_size];
                let mut dst_sse = vec![0f32; out_size];
                convolve_horizontal_native_row_f32::<1>(&src, &mut dst_scalar, &filter_weights, 8);
                convolve_horizontal_plane_sse_row_one(&src, &mut dst_sse, &filter_weights, 8);

                assert_f32_slices_close(
                    &dst_sse,
                    &dst_scalar,
                    ATOL,
                    &format!("{resampling:?} {in_size}->{out_size}: SSE plane f32 single-row"),
                );
            }
        }
    }

    #[test]
    fn sse_rows_4_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        const ROWS: usize = 4;
        for resampling in [ResamplingFunction::Bilinear, ResamplingFunction::Lanczos3] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64)] {
                let filter_weights =
                    make_row_filter_weights_f32(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xBADF00D ^ (in_size as u64) << 32 ^ out_size as u64);
                let src_stride = in_size;
                let dst_stride = out_size;
                let src = rng.fill_f32_unit(src_stride * ROWS);

                let mut dst_scalar = vec![0f32; dst_stride * ROWS];
                let mut dst_sse = vec![0f32; dst_stride * ROWS];
                convolve_horizontal_rgba_4_row_f32::<1>(
                    &src,
                    src_stride,
                    &mut dst_scalar,
                    dst_stride,
                    &filter_weights,
                    8,
                );
                convolve_horizontal_plane_sse_rows_4(
                    &src,
                    src_stride,
                    &mut dst_sse,
                    dst_stride,
                    &filter_weights,
                    8,
                );

                assert_f32_slices_close(
                    &dst_sse,
                    &dst_scalar,
                    ATOL,
                    &format!("{resampling:?} {in_size}->{out_size}: SSE plane f32 4-row"),
                );
            }
        }
    }
}
