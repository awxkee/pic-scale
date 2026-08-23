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
use crate::neon::utils::{prefer_vfmaq_f32, xvld1q_f32_x2, xvld1q_f32_x4};
use std::arch::aarch64::*;

#[inline(always)]
fn conv_horiz_plane_16_f32(
    start_x: usize,
    src: &[f32],
    set: float32x4x4_t,
    store: float32x4_t,
) -> float32x4_t {
    let src_ptr = unsafe { src.get_unchecked(start_x..).as_ptr() };
    let rgb_pixel = unsafe { xvld1q_f32_x4(src_ptr) };
    let mut acc = prefer_vfmaq_f32(store, rgb_pixel.0, set.0);
    acc = prefer_vfmaq_f32(acc, rgb_pixel.1, set.1);
    acc = prefer_vfmaq_f32(acc, rgb_pixel.2, set.2);
    acc = prefer_vfmaq_f32(acc, rgb_pixel.3, set.3);
    acc
}

#[inline(always)]
fn conv_horiz_plane_8_f32(
    start_x: usize,
    src: &[f32],
    set1: float32x4_t,
    set2: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    let src_ptr = unsafe { src.get_unchecked(start_x..) };
    let rgb_pixel = unsafe { xvld1q_f32_x2(src_ptr.as_ptr()) };
    let mut acc = prefer_vfmaq_f32(store, rgb_pixel.0, set1);
    acc = prefer_vfmaq_f32(acc, rgb_pixel.1, set2);
    acc
}

#[inline(always)]
fn conv_horiz_plane_4_f32(
    start_x: usize,
    src: &[f32],
    set1: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    let src_ptr = unsafe { src.get_unchecked(start_x..) };
    let rgb_pixel = unsafe { vld1q_f32(src_ptr.as_ptr()) };
    prefer_vfmaq_f32(store, rgb_pixel, set1)
}

#[inline(always)]
fn conv_horiz_plane_2_f32(
    start_x: usize,
    src: &[f32],
    set: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    let src_ptr = unsafe { src.get_unchecked(start_x..) };
    let rgb_pixel_0 = unsafe { vld1_f32(src_ptr.as_ptr()) };
    let rgb_pixel = unsafe { vcombine_f32(rgb_pixel_0, vdup_n_f32(0.)) };
    prefer_vfmaq_f32(store, rgb_pixel, set)
}

#[inline(always)]
fn conv_horiz_plane_1_f32(
    start_x: usize,
    src: &[f32],
    set: float32x4_t,
    store: float32x4_t,
) -> float32x4_t {
    unsafe {
        let src_ptr = src.get_unchecked(start_x..);
        let rgb_pixel = vld1q_lane_f32::<0>(src_ptr.as_ptr(), vdupq_n_f32(0.));
        prefer_vfmaq_f32(store, rgb_pixel, set)
    }
}

pub(crate) fn convolve_horizontal_plane_neon_row_one(
    src: &[f32],
    dst: &mut [f32],
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let dst_width = filter_weights.bounds.len();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vdupq_n_f32(0f32);

            let local_weights = filter_weights.weights.get_unchecked(filter_offset..);

            while jx + 16 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = local_weights.get_unchecked(jx..);
                let read_weights = xvld1q_f32_x4(ptr.as_ptr());
                store = conv_horiz_plane_16_f32(bounds_start, src, read_weights, store);
                jx += 16;
            }

            while jx + 8 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = local_weights.get_unchecked(jx..);
                let read_weights = xvld1q_f32_x2(ptr.as_ptr());
                store = conv_horiz_plane_8_f32(
                    bounds_start,
                    src,
                    read_weights.0,
                    read_weights.1,
                    store,
                );
                jx += 8;
            }

            while jx + 4 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = local_weights.get_unchecked(jx..);
                let read_weights = vld1q_f32(ptr.as_ptr());
                store = conv_horiz_plane_4_f32(bounds_start, src, read_weights, store);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = local_weights.get_unchecked(jx..);
                let weights0 = vld1_f32(ptr.as_ptr());
                let weights = vcombine_f32(weights0, vdup_n_f32(0.));
                store = conv_horiz_plane_2_f32(bounds_start, src, weights, store);
                jx += 2;
            }

            while jx < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = local_weights.get_unchecked(jx);
                let weight0 = vld1q_dup_f32(ptr);
                store = conv_horiz_plane_1_f32(bounds_start, src, weight0, store);
                jx += 1;
            }

            let dest_ptr = dst.get_unchecked_mut(x);
            *dest_ptr = vaddvq_f32(store);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub(crate) fn convolve_horizontal_plane_neon_rows_4(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let zeros = vdupq_n_f32(0f32);

        let dst_width = filter_weights.bounds.len();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            let local_weights = filter_weights.weights.get_unchecked(filter_offset..);

            while jx + 16 <= bounds.size {
                let ptr = local_weights.get_unchecked(jx..);
                let read_weights = xvld1q_f32_x4(ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_16_f32(bounds_start, src, read_weights, store_0);
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_plane_16_f32(bounds_start, s_ptr_1, read_weights, store_1);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_16_f32(bounds_start, s_ptr2, read_weights, store_2);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_plane_16_f32(bounds_start, s_ptr3, read_weights, store_3);
                jx += 16;
            }

            while jx + 8 <= bounds.size {
                let ptr = local_weights.get_unchecked(jx..);
                let read_weights = xvld1q_f32_x2(ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_8_f32(
                    bounds_start,
                    src,
                    read_weights.0,
                    read_weights.1,
                    store_0,
                );
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_plane_8_f32(
                    bounds_start,
                    s_ptr_1,
                    read_weights.0,
                    read_weights.1,
                    store_1,
                );
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_8_f32(
                    bounds_start,
                    s_ptr2,
                    read_weights.0,
                    read_weights.1,
                    store_2,
                );
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_plane_8_f32(
                    bounds_start,
                    s_ptr3,
                    read_weights.0,
                    read_weights.1,
                    store_3,
                );
                jx += 8;
            }

            while jx + 4 <= bounds.size {
                let ptr = local_weights.get_unchecked(jx..);
                let read_weights = vld1q_f32(ptr.as_ptr());
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_4_f32(bounds_start, src, read_weights, store_0);
                let s_ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_plane_4_f32(bounds_start, s_ptr_1, read_weights, store_1);
                let s_ptr2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_4_f32(bounds_start, s_ptr2, read_weights, store_2);
                let s_ptr3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_plane_4_f32(bounds_start, s_ptr3, read_weights, store_3);
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let ptr = local_weights.get_unchecked(jx..);
                let weights0 = vld1_f32(ptr.as_ptr());
                let weights = vcombine_f32(weights0, vdup_n_f32(0.));
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_2_f32(bounds_start, src, weights, store_0);
                let ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_plane_2_f32(bounds_start, ptr_1, weights, store_1);
                let ptr_2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_2_f32(bounds_start, ptr_2, weights, store_2);
                let ptr_3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_plane_2_f32(bounds_start, ptr_3, weights, store_3);
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = local_weights.get_unchecked(jx);
                let weight0 = vld1q_dup_f32(ptr);
                let bounds_start = bounds.start + jx;
                store_0 = conv_horiz_plane_1_f32(bounds_start, src, weight0, store_0);
                let ptr_1 = src.get_unchecked(src_stride..);
                store_1 = conv_horiz_plane_1_f32(bounds_start, ptr_1, weight0, store_1);
                let ptr_2 = src.get_unchecked(src_stride * 2..);
                store_2 = conv_horiz_plane_1_f32(bounds_start, ptr_2, weight0, store_2);
                let ptr_3 = src.get_unchecked(src_stride * 3..);
                store_3 = conv_horiz_plane_1_f32(bounds_start, ptr_3, weight0, store_3);
                jx += 1;
            }

            let packed = vpaddq_f32(vpaddq_f32(store_0, store_1), vpaddq_f32(store_2, store_3));

            let dest_ptr0 = dst.get_unchecked_mut(x);
            vst1q_lane_f32::<0>(dest_ptr0, packed);
            let dest_ptr1 = dst.get_unchecked_mut(x + dst_stride);
            vst1q_lane_f32::<1>(dest_ptr1, packed);
            let dest_ptr2 = dst.get_unchecked_mut(x + dst_stride * 2);
            vst1q_lane_f32::<2>(dest_ptr2, packed);
            let dest_ptr3 = dst.get_unchecked_mut(x + dst_stride * 3);
            vst1q_lane_f32::<3>(dest_ptr3, packed);

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

    // Same reduction-reordering + hardware-FMA rationale as sse/plane_f32.rs
    // and avx2/plane_f32.rs: a single-channel plane accumulates multiple taps
    // per SIMD lane and reduces at the end, unlike the scalar loop's strict
    // left-to-right accumulation - not bit-exact (observed diffs are 1 ULP).
    const ATOL: f32 = 1e-5;

    #[test]
    fn neon_row_matches_scalar_reference() {
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
                let mut dst_neon = vec![0f32; out_size];
                convolve_horizontal_native_row_f32::<1>(&src, &mut dst_scalar, &filter_weights, 8);
                convolve_horizontal_plane_neon_row_one(&src, &mut dst_neon, &filter_weights, 8);

                assert_f32_slices_close(
                    &dst_neon,
                    &dst_scalar,
                    ATOL,
                    &format!("{resampling:?} {in_size}->{out_size}: NEON plane f32 single-row"),
                );
            }
        }
    }

    #[test]
    fn neon_rows_4_matches_scalar_reference() {
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
                let mut dst_neon = vec![0f32; dst_stride * ROWS];
                convolve_horizontal_rgba_4_row_f32::<1>(
                    &src,
                    src_stride,
                    &mut dst_scalar,
                    dst_stride,
                    &filter_weights,
                    8,
                );
                convolve_horizontal_plane_neon_rows_4(
                    &src,
                    src_stride,
                    &mut dst_neon,
                    dst_stride,
                    &filter_weights,
                    8,
                );

                assert_f32_slices_close(
                    &dst_neon,
                    &dst_scalar,
                    ATOL,
                    &format!("{resampling:?} {in_size}->{out_size}: NEON plane f32 4-row"),
                );
            }
        }
    }
}
