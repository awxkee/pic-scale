#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
use std::sync::Arc;

use crate::acceleration_feature::AccelerationFeature;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::filter_weights::FilterWeights;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon_simd_u8::*;
use crate::rgb_u8::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse_rgb_u8::sse_rgb::*;
use crate::threading_policy::ThreadingPolicy;
use crate::ImageStore;
use crate::unsafe_slice::UnsafeSlice;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn convolve_horizontal_rgba_sse(
    image_store: &ImageStore<u8, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 4>,
    threading_policy: ThreadingPolicy,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;
    let dst_width = destination.width;

    let threads_count = threading_policy.get_threads_count(destination.get_size());

    let mut yy = 0usize;

    if threads_count == 1 {
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
    } else {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        let destination_height = destination.height;
        let dst_width = destination.width;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads_count)
            .build()
            .unwrap();
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
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn convolve_horizontal_rgba_neon(
    image_store: &ImageStore<u8, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 4>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let weights_ptr = approx_weights.weights.as_ptr();

    let mut unsafe_source_ptr_0 = image_store.buffer.as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    for _ in 0..destination.height {
        let mut filter_offset = 0usize;

        for x in 0..destination.width {
            let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store = unsafe { vdupq_n_s32(0i32) };

            while jx + 4 < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                let weight2 = unsafe { ptr.add(2).read_unaligned() };
                let weight3 = unsafe { ptr.add(3).read_unaligned() };
                unsafe {
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
                let weight0 = unsafe { ptr.read_unaligned() };
                unsafe {
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

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            let value = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8)) };
            let dest_ptr_32 = dest_ptr as *mut u32;
            unsafe {
                *dest_ptr_32 = value;
            }

            filter_offset += approx_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

fn convolve_horizontal_rgba_native(
    image_store: &ImageStore<u8, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 4>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let weights_ptr = approx_weights.weights.as_ptr();

    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    for _ in 0..destination.height {
        let mut filter_offset = 0usize;

        for x in 0..destination.width {
            let mut sum_r = 0i32;
            let mut sum_g = 0i32;
            let mut sum_b = 0i32;
            let mut sum_a = 0i32;

            let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * image_store.channels;
                let weight = unsafe { weights_ptr.add(j + filter_offset).read_unaligned() } as i32;
                let src = unsafe { unsafe_source_ptr_0.add(px) };
                sum_r += unsafe { src.read_unaligned() } as i32 * weight;
                sum_g += unsafe { src.add(1).read_unaligned() } as i32 * weight;
                sum_b += unsafe { src.add(2).read_unaligned() } as i32 * weight;
                sum_a += unsafe { src.add(3).read_unaligned() } as i32 * weight;
            }

            let px = x * image_store.channels;

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

            unsafe {
                *dest_ptr = (sum_r >> 12).min(255).max(0) as u8;
                *dest_ptr.add(1) = (sum_g >> 12).min(255).max(0) as u8;
                *dest_ptr.add(2) = (sum_b >> 12).min(255).max(0) as u8;
                *dest_ptr.add(3) = (sum_a >> 12).min(255).max(0) as u8;
            }

            filter_offset += approx_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn convolve_vertical_rgba_neon(
    image_store: &ImageStore<u8, 4>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 4>,
    threading_policy: ThreadingPolicy,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let unsafe_source_ptr_0 = image_store.buffer.as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;

    let mut filter_offset = 0usize;

    let dst_stride = destination.width * image_store.channels;

    let total_width = destination.width * image_store.channels;

    for y in 0..destination.height {
        let mut cx = 0usize;
        let bounds = unsafe { approx_weights.bounds.get_unchecked(y) };
        let weight_ptr = unsafe { approx_weights.weights.as_ptr().add(filter_offset) };

        while cx + 32 < total_width {
            unsafe {
                neon_convolve_u8::convolve_vertical_part_neon_32(
                    bounds.start,
                    cx,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    weight_ptr,
                    bounds,
                );
            }

            cx += 32;
        }

        while cx + 16 < total_width {
            unsafe {
                neon_convolve_u8::convolve_vertical_part_neon_16(
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
                neon_convolve_u8::convolve_vertical_part_neon_8::<false>(
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
                neon_convolve_u8::convolve_vertical_part_neon_8::<true>(
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

        filter_offset += approx_weights.aligned_size;
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

impl HorizontalConvolutionPass<u8, 4> for ImageStore<'static, u8, 4> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 4>,
        threading_policy: ThreadingPolicy,
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
                convolve_horizontal_rgba_neon(self, filter_weights, destination);
            }
            AccelerationFeature::Native => {
                convolve_horizontal_rgba_native(self, filter_weights, destination);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {
                convolve_horizontal_rgba_sse(self, filter_weights, destination, threading_policy);
            }
        }
    }
}

impl VerticalConvolutionPass<u8, 4> for ImageStore<'static, u8, 4> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 4>,
        threading_policy: ThreadingPolicy,
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
                convolve_vertical_rgba_neon(self, filter_weights, destination, threading_policy);
            }
            AccelerationFeature::Native => {
                convolve_vertical_rgb_native_8(self, filter_weights, destination, threading_policy);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {
                convolve_vertical_rgb_sse_8(self, filter_weights, destination, threading_policy);
            }
        }
    }
}
