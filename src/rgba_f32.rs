#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
use std::sync::Arc;

use rayon::ThreadPool;

use crate::acceleration_feature::AccelerationFeature;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::convolve_f32::*;
use crate::filter_weights::*;
use crate::rgb_f32::convolve_vertical_native_f32;
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageStore;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn convolve_horizontal_rgba_f32_neon(
    image_store: &ImageStore<f32, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 4>,
) {
    let weights_ptr = filter_weights.weights.as_ptr();

    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    let zeros = unsafe { vdupq_n_f32(0f32) };

    let mut yy = 0usize;

    while yy + 4 < destination.height {
        let mut filter_offset = 0usize;

        for x in 0..destination.width {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 4 < bounds.size && x + 6 < destination.width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                let weight2 = unsafe { ptr.add(2).read_unaligned() };
                let weight3 = unsafe { ptr.add(3).read_unaligned() };
                unsafe {
                    store_0 = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_0,
                    );
                    store_1 = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_1,
                    );
                    store_2 = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_2,
                    );
                    store_3 = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_3,
                    );
                }
                jx += 4;
            }
            while jx < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                unsafe {
                    store_0 = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0,
                        weight0,
                        store_0,
                    );
                    store_1 = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        store_1,
                    );
                    store_2 = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        store_2,
                    );
                    store_3 = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        store_3,
                    );
                }
                jx += 1;
            }

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                vst1q_f32(dest_ptr, store_0);
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride) };
            unsafe {
                vst1q_f32(dest_ptr, store_1);
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 2) };
            unsafe {
                vst1q_f32(dest_ptr, store_2);
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 3) };
            unsafe {
                vst1q_f32(dest_ptr, store_3);
            }

            filter_offset += filter_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride * 4) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride * 4) };

        yy += 4;
    }

    for _ in yy..destination.height {
        let mut filter_offset = 0usize;

        for x in 0..destination.width {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store = unsafe { vdupq_n_f32(0f32) };

            while jx + 4 < bounds.size && x + 6 < destination.width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                let weight2 = unsafe { ptr.add(2).read_unaligned() };
                let weight3 = unsafe { ptr.add(3).read_unaligned() };
                unsafe {
                    store = convolve_horizontal_parts_4_rgba_f32(
                        bounds.start,
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
                let weight0 = unsafe { ptr.read_unaligned() };
                unsafe {
                    store = convolve_horizontal_parts_one_rgba_f32(
                        bounds.start,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                    );
                }
                jx += 1;
            }

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                vst1q_f32(dest_ptr, store);
            }

            filter_offset += filter_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

#[inline(always)]
fn convolve_horizontal_rgb_native_row(
    dst_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    const CHANNELS: usize = 4;
    let weights_ptr = filter_weights.weights.as_ptr();
    let mut filter_offset = 0usize;

    for x in 0..dst_width {
        let mut sum_r = 0f32;
        let mut sum_g = 0f32;
        let mut sum_b = 0f32;
        let mut sum_a = 0f32;

        let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
        let start_x = bounds.start;
        for j in 0..bounds.size {
            let px = (start_x + j) * CHANNELS;
            let weight = unsafe { weights_ptr.add(j + filter_offset).read_unaligned() };
            let src = unsafe { unsafe_source_ptr_0.add(px) };
            sum_r += unsafe { src.read_unaligned() } * weight;
            sum_g += unsafe { src.add(1).read_unaligned() } * weight;
            sum_b += unsafe { src.add(2).read_unaligned() } * weight;
            sum_a += unsafe { src.add(3).read_unaligned() } * weight;
        }

        let px = x * CHANNELS;

        let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

        unsafe {
            *dest_ptr = sum_r;
            *dest_ptr.add(1) = sum_g;
            *dest_ptr.add(2) = sum_b;
            *dest_ptr.add(3) = sum_a;
        }

        filter_offset += filter_weights.aligned_size;
    }
}

fn convolve_horizontal_rgba_f32_native(
    image_store: &ImageStore<f32, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 4>,
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
fn convolve_vertical_rgba_f32_neon(
    image_store: &ImageStore<f32, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 4>,
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

                    convolve_vertical_rgb_native_row(
                        total_width,
                        src_stride,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                        weight_ptr,
                        &bounds,
                    );
                });
            }
        });
    } else {
        let mut filter_offset = 0usize;
        for y in 0..destination.height {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(y) };
            let weight_ptr = unsafe { filter_weights.weights.as_ptr().add(filter_offset) };

            convolve_vertical_rgb_native_row(
                total_width,
                src_stride,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                weight_ptr,
                &bounds,
            );

            filter_offset += filter_weights.aligned_size;
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

impl<'a> HorizontalConvolutionPass<f32, 4> for ImageStore<'a, f32, 4> {
    #[inline(always)]
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 4>,
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
                convolve_horizontal_rgba_f32_neon(self, filter_weights, destination);
            }
            AccelerationFeature::Native => {
                convolve_horizontal_rgba_f32_native(self, filter_weights, destination, pool);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {}
        }
    }
}

impl<'a> VerticalConvolutionPass<f32, 4> for ImageStore<'a, f32, 4> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 4>,
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
                convolve_vertical_rgba_f32_neon(self, filter_weights, destination, pool);
            }
            AccelerationFeature::Native => {
                convolve_vertical_native_f32(self, filter_weights, destination, pool);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {
                crate::rgb_f32::convolve_vertical_sse_rgb_f32(
                    self,
                    filter_weights,
                    destination,
                    pool,
                );
            }
        }
    }
}
