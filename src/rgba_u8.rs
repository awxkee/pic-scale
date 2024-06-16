/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
use std::sync::Arc;

use rayon::ThreadPool;

use crate::acceleration_feature::AccelerationFeature;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::filter_weights::FilterWeights;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon_simd_u8::*;
use crate::rgb_u8::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse_rgb_u8::sse_rgb::*;
use crate::support::{PRECISION, ROUNDING_APPROX};
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageStore;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn convolve_horizontal_rgba_sse(
    image_store: &ImageStore<u8, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 4>,
    pool: &Option<ThreadPool>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;
    let dst_width = destination.width;

    let mut yy = 0usize;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        let destination_height = destination.height;
        let dst_width = destination.width;
        pool.scope(|scope| {
            let mut yy = 0usize;
            for y in (0..destination_height.saturating_sub(4)).step_by(4) {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 =
                        unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * y) };
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    unsafe {
                        convolve_horizontal_rgba_sse_rows_4(
                            dst_width,
                            &weights,
                            unsafe_source_ptr_0,
                            src_stride,
                            unsafe_destination_ptr_0,
                            dst_stride,
                        );
                    }
                });
                yy = y;
            }
            for y in (yy..destination.height).step_by(4) {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 =
                        unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * y) };
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    unsafe {
                        convolve_horizontal_rgba_sse_rows_one(
                            dst_width,
                            &weights,
                            unsafe_source_ptr_0,
                            unsafe_destination_ptr_0,
                        );
                    }
                });
            }
        });
    } else {
        while yy < destination.height.saturating_sub(4) {
            unsafe {
                convolve_horizontal_rgba_sse_rows_4(
                    dst_width,
                    &approx_weights,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    dst_stride,
                );
            }
            unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride * 4) };
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride * 4) };

            yy += 4;
        }

        for _ in yy..destination.height {
            unsafe {
                convolve_horizontal_rgba_sse_rows_one(
                    dst_width,
                    &approx_weights,
                    unsafe_source_ptr_0,
                    unsafe_destination_ptr_0,
                );
            }

            unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub unsafe fn convolve_horizontal_rgba_neon_row(
    dst_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    const CHANNELS: usize = 4;
    let mut filter_offset = 0usize;

    let weights_ptr = approx_weights.weights.as_ptr();

    for x in 0..dst_width {
        let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
        let mut jx = 0usize;
        let mut store = unsafe { vdupq_n_s32(ROUNDING_APPROX) };

        while jx + 4 < bounds.size {
            let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
            unsafe {
                let weight0 = vdup_n_s16(ptr.read_unaligned());
                let weight1 = vdupq_n_s16(ptr.add(1).read_unaligned());
                let weight2 = vdup_n_s16(ptr.add(2).read_unaligned());
                let weight3 = vdupq_n_s16(ptr.add(3).read_unaligned());
                store = neon_convolve_u8::convolve_horizontal_parts_4_rgba(
                    bounds.start + jx,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store,
                );
            }
            jx += 4;
        }

        while jx < bounds.size {
            let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
            unsafe {
                let weight0 = vdup_n_s16(ptr.read_unaligned());
                store = neon_convolve_u8::convolve_horizontal_parts_one_rgba(
                    bounds.start + jx,
                    unsafe_source_ptr_0,
                    weight0,
                    store,
                );
            }
            jx += 1;
        }

        let store_16 = unsafe { vqshrun_n_s32::<12>(vmaxq_s32(store, vdupq_n_s32(0i32))) };
        let store_16_8 = unsafe { vqmovn_u16(vcombine_u16(store_16, store_16)) };

        let px = x * CHANNELS;
        let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
        let value = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8)) };
        let dest_ptr_32 = dest_ptr as *mut u32;
        unsafe {
            dest_ptr_32.write_unaligned(value);
        }

        filter_offset += approx_weights.aligned_size;
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn convolve_horizontal_rgba_neon(
    image_store: &ImageStore<u8, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 4>,
    pool: &Option<ThreadPool>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    let dst_width = destination.width;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        let destination_height = destination.height;
        let dst_width = destination.width;
        pool.scope(|_| {
            let weights = arc_weights.clone();
            for y in 0..destination_height {
                let unsafe_source_ptr_0 =
                    unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * y) };
                let dst_ptr = unsafe_slice.mut_ptr();
                let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                unsafe {
                    convolve_horizontal_rgba_neon_row(
                        dst_width,
                        &weights,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                    );
                }
            }
        });
    } else {
        for _ in 0..destination.height {
            unsafe {
                convolve_horizontal_rgba_neon_row(
                    dst_width,
                    &approx_weights,
                    unsafe_source_ptr_0,
                    unsafe_destination_ptr_0,
                );
            }

            unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

fn convolve_horizontal_rgba_native_row(
    dst_width: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    const CHANNELS: usize = 4;
    let mut filter_offset = 0usize;
    let weights_ptr = filter_weights.weights.as_ptr();

    for x in 0..dst_width {
        let mut sum_r = ROUNDING_APPROX;
        let mut sum_g = ROUNDING_APPROX;
        let mut sum_b = ROUNDING_APPROX;
        let mut sum_a = ROUNDING_APPROX;

        let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
        let start_x = bounds.start;
        for j in 0..bounds.size {
            let px = (start_x + j) * CHANNELS;
            let weight = unsafe { weights_ptr.add(j + filter_offset).read_unaligned() } as i32;
            let src = unsafe { unsafe_source_ptr_0.add(px) };
            sum_r += unsafe { src.read_unaligned() } as i32 * weight;
            sum_g += unsafe { src.add(1).read_unaligned() } as i32 * weight;
            sum_b += unsafe { src.add(2).read_unaligned() } as i32 * weight;
            sum_a += unsafe { src.add(3).read_unaligned() } as i32 * weight;
        }

        let px = x * CHANNELS;

        let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

        unsafe {
            dest_ptr.write_unaligned((sum_r >> PRECISION).min(255).max(0) as u8);
            dest_ptr
                .add(1)
                .write_unaligned((sum_g >> PRECISION).min(255).max(0) as u8);
            dest_ptr
                .add(2)
                .write_unaligned((sum_b >> PRECISION).min(255).max(0) as u8);
            dest_ptr
                .add(3)
                .write_unaligned((sum_a >> PRECISION).min(255).max(0) as u8);
        }

        filter_offset += filter_weights.aligned_size;
    }
}

fn convolve_horizontal_rgba_native(
    image_store: &ImageStore<u8, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 4>,
    _pool: &Option<ThreadPool>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let dst_width = destination.width;

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    if let Some(pool) = _pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        pool.scope(|scope| {
            for y in 0..destination.height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 =
                        unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * y) };
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    convolve_horizontal_rgba_native_row(
                        dst_width,
                        &weights,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                    );
                });
            }
        });
    } else {
        for _ in 0..destination.height {
            convolve_horizontal_rgba_native_row(
                dst_width,
                &approx_weights,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
            );

            unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

impl<'a> HorizontalConvolutionPass<u8, 4> for ImageStore<'a, u8, 4> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 4>,
        _pool: &Option<ThreadPool>,
    ) {
        let mut _using_feature = AccelerationFeature::Native;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _using_feature = AccelerationFeature::Neon;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _using_feature = AccelerationFeature::Sse;
            }
        }
        match _using_feature {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            AccelerationFeature::Neon => {
                convolve_horizontal_rgba_neon(self, filter_weights, destination, _pool);
            }
            AccelerationFeature::Native => {
                convolve_horizontal_rgba_native(self, filter_weights, destination, _pool);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {
                convolve_horizontal_rgba_sse(self, filter_weights, destination, _pool);
            }
        }
    }
}

impl<'a> VerticalConvolutionPass<u8, 4> for ImageStore<'a, u8, 4> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 4>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _using_feature = AccelerationFeature::Native;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _using_feature = AccelerationFeature::Neon;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _using_feature = AccelerationFeature::Sse;
            }
        }
        match _using_feature {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            AccelerationFeature::Neon => {
                convolve_vertical_rgb_neon(self, filter_weights, destination, pool);
            }
            AccelerationFeature::Native => {
                convolve_vertical_rgb_native_8(self, filter_weights, destination, pool);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {
                convolve_vertical_rgb_sse_8(self, filter_weights, destination, pool);
            }
        }
    }
}
