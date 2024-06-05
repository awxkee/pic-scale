/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::acceleration_feature::AccelerationFeature;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_f32::*;
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_store::ImageStore;
use crate::neon_rgb_f32::neon_convolve_floats;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse_rgb_f32::sse_convolve_f32::*;
use crate::unsafe_slice::UnsafeSlice;
use rayon::ThreadPool;
use std::sync::Arc;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn convolve_horizontal_neon(
    image_store: &ImageStore<f32, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 3>,
    pool: &Option<ThreadPool>,
) {
    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;
    let dst_width = destination.width;
    let src_width = image_store.width;

    let mut yy = 0usize;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(filter_weights);
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
                        neon_convolve_floats::convolve_horizontal_rgb_neon_rows_4(
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
                        neon_convolve_floats::convolve_horizontal_rgb_neon_row_one(
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
        while yy + 4 < destination.height {
            unsafe {
                neon_convolve_floats::convolve_horizontal_rgb_neon_rows_4(
                    dst_width,
                    src_width,
                    &filter_weights,
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
                neon_convolve_floats::convolve_horizontal_rgb_neon_row_one(
                    dst_width,
                    &filter_weights,
                    unsafe_source_ptr_0,
                    unsafe_destination_ptr_0,
                );
            }
            unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

#[inline(always)]
fn convolve_horizontal_rgb_native_row(
    dst_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    let mut filter_offset = 0usize;
    let weights_ptr = filter_weights.weights.as_ptr();

    const CHANNELS: usize = 3;
    for x in 0..dst_width {
        let mut sum_r = 0f32;
        let mut sum_g = 0f32;
        let mut sum_b = 0f32;

        let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
        let start_x = bounds.start;
        for j in 0..bounds.size {
            let px = (start_x + j) * CHANNELS;
            let weight = unsafe { weights_ptr.add(j + filter_offset).read_unaligned() };
            let src = unsafe { unsafe_source_ptr_0.add(px) };
            sum_r += unsafe { src.read_unaligned() } * weight;
            sum_g += unsafe { src.add(1).read_unaligned() } * weight;
            sum_b += unsafe { src.add(2).read_unaligned() } * weight;
        }

        let px = x * CHANNELS;

        let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

        unsafe {
            *dest_ptr = sum_r;
            *dest_ptr.add(1) = sum_g;
            *dest_ptr.add(2) = sum_b;
        }

        filter_offset += filter_weights.aligned_size;
    }
}

fn convolve_horizontal_native(
    image_store: &ImageStore<f32, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 3>,
    pool: &Option<ThreadPool>,
) {
    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    let dst_width = destination.width;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(filter_weights);
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
                dst_width,
                &filter_weights,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
            );
            unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn convolve_vertical_rgb_native_row(
    total_width: usize,
    src_stride: usize,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    weight_ptr: *const f32,
    bounds: &FilterBounds,
) {
    let mut cx = 0usize;

    while cx + 16 < total_width {
        unsafe {
            convolve_vertical_part_neon_16_f32(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 16;
    }
    while cx + 8 < total_width {
        unsafe {
            convolve_vertical_part_neon_8_f32::<false>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
                8,
            );
        }

        cx += 8;
    }

    let left = total_width - cx;

    if left > 0 {
        unsafe {
            convolve_vertical_part_neon_8_f32::<true>(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
                left,
            );
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn convolve_vertical_neon(
    image_store: &ImageStore<f32, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 3>,
    pool: &Option<ThreadPool>,
) {
    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();
    let src_stride = image_store.width * image_store.channels;
    let mut filter_offset = 0usize;
    let dst_stride = destination.width * image_store.channels;
    let total_width = destination.width * image_store.channels;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(filter_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        pool.scope(|scope| {
            for y in 0..destination.height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    let bounds = unsafe { weights.bounds.get_unchecked(y) };
                    let weight_ptr = unsafe { weights.weights.as_ptr().add(filter_offset) };
                    convolve_vertical_rgb_native_row(
                        total_width,
                        src_stride,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                        weight_ptr,
                        bounds,
                    );
                });
            }
        });
    } else {
        for y in 0..destination.height {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(y) };
            let weight_ptr = unsafe { filter_weights.weights.as_ptr().add(filter_offset) };

            convolve_vertical_rgb_native_row(
                total_width,
                src_stride,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );

            filter_offset += filter_weights.aligned_size;
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
fn convolve_horizontal_rgb_sse_row(
    total_width: usize,
    src_stride: usize,
    bounds: &FilterBounds,
    weight_ptr: *const f32,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    let mut cx = 0usize;

    while cx + 16 < total_width {
        unsafe {
            convolve_vertical_part_sse_16_f32(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 16;
    }

    while cx + 8 < total_width {
        unsafe {
            convolve_vertical_part_sse_8_f32(
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

    while cx + 4 < total_width {
        unsafe {
            convolve_vertical_part_sse_4_f32(
                bounds.start,
                cx,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                weight_ptr,
                bounds,
            );
        }

        cx += 4;
    }

    while cx < total_width {
        unsafe {
            convolve_vertical_part_sse_f32(
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) fn convolve_vertical_sse_rgb_f32<const COMPONENTS: usize>(
    image_store: &ImageStore<f32, COMPONENTS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, COMPONENTS>,
    pool: &Option<ThreadPool>,
) {
    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();
    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;
    let total_width = destination.width * image_store.channels;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(filter_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        pool.scope(|scope| {
            for y in 0..destination.height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    let filter_offset = y * weights.aligned_size;
                    let bounds = unsafe { weights.bounds.get_unchecked(y) };
                    let weight_ptr = unsafe { weights.weights.as_ptr().add(filter_offset) };
                    convolve_horizontal_rgb_sse_row(
                        total_width,
                        src_stride,
                        &bounds,
                        weight_ptr,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                    );
                });
            }
        });
    } else {
        let mut filter_offset = 0usize;
        for y in 0..destination.height {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(y) };
            let weight_ptr = unsafe { filter_weights.weights.as_ptr().add(filter_offset) };

            convolve_horizontal_rgb_sse_row(
                total_width,
                src_stride,
                &bounds,
                weight_ptr,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
            );

            filter_offset += filter_weights.aligned_size;
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

#[inline(always)]
fn convolve_vertical_rgb_native_row_f32<const COMPONENTS: usize>(
    dst_width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    src_stride: usize,
    weight_ptr: *const f32,
) {
    let mut cx = 0usize;
    while cx + 12 < dst_width {
        unsafe {
            convolve_vertical_part_f32::<12, COMPONENTS>(
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
            convolve_vertical_part_f32::<8, COMPONENTS>(
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
            convolve_vertical_part_f32::<1, COMPONENTS>(
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

pub(crate) fn convolve_vertical_native_f32<const COMPONENTS: usize>(
    image_store: &ImageStore<f32, COMPONENTS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, COMPONENTS>,
    pool: &Option<ThreadPool>,
) {
    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;

    let mut filter_offset = 0usize;

    let dst_stride = destination.width * image_store.channels;
    let dst_width = destination.width;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(filter_weights);
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
                    convolve_vertical_rgb_native_row_f32::<COMPONENTS>(
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
        for y in 0..destination.height {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(y) };
            let weight_ptr = unsafe { filter_weights.weights.as_ptr().add(filter_offset) };

            convolve_vertical_rgb_native_row_f32::<COMPONENTS>(
                dst_width,
                bounds,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                src_stride,
                weight_ptr,
            );

            filter_offset += filter_weights.aligned_size;
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

impl<'a> HorizontalConvolutionPass<f32, 3> for ImageStore<'a, f32, 3> {
    #[inline(always)]
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 3>,
        pool: &Option<ThreadPool>,
    ) {
        #[allow(unused_assignments)]
        #[allow(unused_mut)]
        let mut using_feature = AccelerationFeature::Native;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            using_feature = AccelerationFeature::Neon;
        }
        match using_feature {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            AccelerationFeature::Neon => {
                convolve_horizontal_neon(self, filter_weights, destination, pool);
            }
            AccelerationFeature::Native => {
                convolve_horizontal_native(self, filter_weights, destination, pool);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {}
        }
    }
}

impl<'a> VerticalConvolutionPass<f32, 3> for ImageStore<'a, f32, 3> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 3>,
        pool: &Option<ThreadPool>,
    ) {
        #[allow(unused_assignments)]
        #[allow(unused_mut)]
        let mut using_feature = AccelerationFeature::Native;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            using_feature = AccelerationFeature::Neon;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            if is_x86_feature_detected!("sse4.1") {
                using_feature = AccelerationFeature::Sse;
            }
        }
        match using_feature {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            AccelerationFeature::Neon => {
                convolve_vertical_neon(self, filter_weights, destination, pool);
            }
            AccelerationFeature::Native => {
                convolve_vertical_native_f32(self, filter_weights, destination, pool);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {
                convolve_vertical_sse_rgb_f32(self, filter_weights, destination, pool);
            }
        }
    }
}
