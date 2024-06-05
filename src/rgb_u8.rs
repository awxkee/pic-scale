/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use rayon::ThreadPool;
use std::sync::Arc;

use crate::acceleration_feature::AccelerationFeature;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_u8::*;
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_store::ImageStore;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon_rgb_u8::neon_rgb::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse_rgb_u8::sse_rgb::*;
use crate::support::{PRECISION, ROUNDING_APPROX};
use crate::unsafe_slice::UnsafeSlice;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn convolve_horizontal_rgb_sse(
    image_store: &ImageStore<u8, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 3>,
    pool: &Option<ThreadPool>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        let destination_height = destination.height;
        let dst_width = destination.width;
        pool.scope(|scope| {
            let mut yy = 0usize;
            while yy + 4 < destination_height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 =
                        unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * yy) };
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * yy) };
                    unsafe {
                        convolve_horizontal_rgb_sse_rows_4(
                            image_store.width,
                            dst_width,
                            &weights,
                            unsafe_source_ptr_0,
                            src_stride,
                            unsafe_destination_ptr_0,
                            dst_stride,
                        );
                    }
                });
                yy += 4;
            }
            for y in (yy..destination.height).step_by(4) {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 =
                        unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * y) };
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    unsafe {
                        convolve_horizontal_rgb_sse_row_one(
                            image_store.width,
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
        let mut yy = 0usize;

        while yy + 4 < destination.height {
            unsafe {
                convolve_horizontal_rgb_sse_rows_4(
                    image_store.width,
                    destination.width,
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
                convolve_horizontal_rgb_sse_row_one(
                    image_store.width,
                    destination.width,
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
fn convolve_horizontal_rgb_neon(
    image_store: &ImageStore<u8, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 3>,
    pool: &Option<ThreadPool>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;
    let dst_width = destination.width;
    let src_width = image_store.width;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        pool.scope(|scope| {
            let mut yy = 0usize;
            for y in (0..destination.height.saturating_sub(4)).step_by(4) {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 =
                        unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * y) };
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    unsafe {
                        convolve_horizontal_rgb_neon_rows_4(
                            dst_width,
                            src_width,
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
                        convolve_horizontal_rgb_neon_row_one(
                            dst_width,
                            src_width,
                            &weights,
                            unsafe_source_ptr_0,
                            unsafe_destination_ptr_0,
                        );
                    }
                });
            }
        });
    } else {
        let mut yy = 0usize;
        while yy + 4 < destination.height {
            unsafe {
                convolve_horizontal_rgb_neon_rows_4(
                    dst_width,
                    src_width,
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
                convolve_horizontal_rgb_neon_row_one(
                    dst_width,
                    src_width,
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

fn convolve_horizontal_rgb_native_row(
    dst_width: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    const CHANNELS: usize = 3;
    let mut filter_offset = 0usize;
    let weights_ptr = filter_weights.weights.as_ptr();
    for x in 0..dst_width {
        let mut sum_r = ROUNDING_APPROX;
        let mut sum_g = ROUNDING_APPROX;
        let mut sum_b = ROUNDING_APPROX;

        let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
        let start_x = bounds.start;
        for j in 0..bounds.size {
            let px = (start_x + j) * CHANNELS;
            let weight = unsafe { weights_ptr.add(j + filter_offset).read_unaligned() } as i32;
            let src = unsafe { unsafe_source_ptr_0.add(px) };
            sum_r += unsafe { src.read_unaligned() } as i32 * weight;
            sum_g += unsafe { src.add(1).read_unaligned() } as i32 * weight;
            sum_b += unsafe { src.add(2).read_unaligned() } as i32 * weight;
        }

        let px = x * CHANNELS;

        let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

        unsafe {
            *dest_ptr = (sum_r >> PRECISION).min(255).max(0) as u8;
            *dest_ptr.add(1) = (sum_g >> PRECISION).min(255).max(0) as u8;
            *dest_ptr.add(2) = (sum_b >> PRECISION).min(255).max(0) as u8;
        }

        filter_offset += filter_weights.aligned_size;
    }
}

fn convolve_horizontal_rgb_native(
    image_store: &ImageStore<u8, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 3>,
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
        pool.scope(|scope| {
            for y in 0..destination.height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 =
                        unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * y) };
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    convolve_horizontal_rgb_native_row(
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
            convolve_horizontal_rgb_native_row(
                destination.width,
                &approx_weights,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
            );

            unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) fn convolve_vertical_rgb_sse_8<const COMPONENTS: usize>(
    image_store: &ImageStore<u8, COMPONENTS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, COMPONENTS>,
    pool: &Option<ThreadPool>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);
    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();
    let src_stride = image_store.width * image_store.channels;
    let mut filter_offset = 0usize;
    let dst_stride = destination.width * image_store.channels;
    let total_width = destination.width * image_store.channels;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        pool.scope(|scope| {
            for y in 0..destination.height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let bounds = unsafe { weights.bounds.get_unchecked(y) };
                    let weight_ptr =
                        unsafe { weights.weights.as_ptr().add(weights.aligned_size * y) };
                    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    convolve_vertical_rgb_sse_row(
                        total_width,
                        &bounds,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                        src_stride,
                        weight_ptr,
                    );
                });
            }
        });
    } else {
        for y in 0..destination.height {
            let bounds = unsafe { approx_weights.bounds.get_unchecked(y) };
            let weight_ptr = unsafe { approx_weights.weights.as_ptr().add(filter_offset) };
            convolve_vertical_rgb_sse_row(
                total_width,
                &bounds,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                src_stride,
                weight_ptr,
            );
            filter_offset += approx_weights.aligned_size;
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub(crate) fn convolve_vertical_rgb_neon<const CHANNELS: usize>(
    image_store: &ImageStore<u8, CHANNELS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, CHANNELS>,
    pool: &Option<ThreadPool>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);
    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();
    let src_stride = image_store.width * image_store.channels;
    let mut filter_offset = 0usize;
    let dst_stride = destination.width * image_store.channels;
    let total_width = destination.width * image_store.channels;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        pool.scope(|scope| {
            for y in 0..destination.height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let bounds = unsafe { weights.bounds.get_unchecked(y) };
                    let weight_ptr =
                        unsafe { weights.weights.as_ptr().add(weights.aligned_size * y) };
                    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    convolve_vertical_rgb_neon_row(
                        total_width,
                        bounds,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                        src_stride,
                        weight_ptr,
                    );
                });
            }
        });
    } else {
        for y in 0..destination.height {
            let bounds = unsafe { approx_weights.bounds.get_unchecked(y) };
            let weight_ptr = unsafe { approx_weights.weights.as_ptr().add(filter_offset) };
            convolve_vertical_rgb_neon_row(
                total_width,
                bounds,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                src_stride,
                weight_ptr,
            );
            filter_offset += approx_weights.aligned_size;
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

#[inline(always)]
pub(crate) fn convolve_vertical_rgb_native_row(
    dst_width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
    src_stride: usize,
    weight_ptr: *const i16,
) {
    let mut cx = 0usize;
    while cx + 12 < dst_width {
        unsafe {
            convolve_vertical_part::<12, 3>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 12;
    }

    while cx + 8 < dst_width {
        unsafe {
            convolve_vertical_part::<8, 3>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 8;
    }

    while cx < dst_width {
        unsafe {
            convolve_vertical_part::<1, 3>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 1;
    }
}

#[inline(always)]
pub(crate) fn convolve_vertical_rgb_native_8<'a, const COMPONENTS: usize>(
    image_store: &ImageStore<u8, COMPONENTS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<'a, u8, COMPONENTS>,
    pool: &Option<ThreadPool>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    let dst_width = destination.width;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        pool.scope(|scope| {
            for y in 0..destination.height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let bounds = unsafe { weights.bounds.get_unchecked(y) };
                    let weight_ptr =
                        unsafe { weights.weights.as_ptr().add(weights.aligned_size * y) };
                    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    convolve_vertical_rgb_native_row(
                        dst_width,
                        bounds,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                        src_stride,
                        weight_ptr,
                    );
                });
            }
        });
    } else {
        let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
        let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();
        let mut filter_offset = 0usize;
        for y in 0..destination.height {
            let bounds = unsafe { approx_weights.bounds.get_unchecked(y) };
            let weight_ptr = unsafe { approx_weights.weights.as_ptr().add(filter_offset) };
            convolve_vertical_rgb_native_row(
                dst_width,
                bounds,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                src_stride,
                weight_ptr,
            );

            filter_offset += approx_weights.aligned_size;
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

impl<'a> HorizontalConvolutionPass<u8, 3> for ImageStore<'a, u8, 3> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 3>,
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
                convolve_horizontal_rgb_neon(self, filter_weights, destination, pool);
            }
            AccelerationFeature::Native => {
                convolve_horizontal_rgb_native(self, filter_weights, destination, pool);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {
                convolve_horizontal_rgb_sse(self, filter_weights, destination, pool);
            }
        }
    }
}

impl<'a> VerticalConvolutionPass<u8, 3> for ImageStore<'a, u8, 3> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 3>,
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
