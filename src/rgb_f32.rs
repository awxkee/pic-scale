use crate::acceleration_feature::AccelerationFeature;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_f32::*;
use crate::filter_weights::FilterWeights;
use crate::image_store::ImageStore;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn convolve_horizontal_neon(
    image_store: &ImageStore<f32, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 3>,
) {
    let weights_ptr = filter_weights.weights.as_ptr();

    let mut unsafe_source_ptr_0 = image_store.buffer.as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    let mask = unsafe { vld1q_f32([1f32, 1f32, 1f32, 0f32].as_ptr()) };

    for _ in 0..destination.height {
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
                    store = convolve_horizontal_parts_4_rgb_f32(
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
                    store = convolve_horizontal_parts_one_rgb_f32(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                        mask,
                    );
                }
                jx += 1;
            }

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            if x + 2 < destination.width {
                unsafe {
                    vst1q_f32(dest_ptr, store);
                }
            } else {
                unsafe {
                    let mut transient: [f32; 4] = [0f32; 4];
                    vst1q_f32(transient.as_mut_ptr(), store);
                    std::ptr::copy_nonoverlapping(transient.as_ptr(), dest_ptr, 3);
                }
            }

            filter_offset += filter_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

#[inline(always)]
fn convolve_horizontal_native(
    image_store: &ImageStore<f32, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 3>,
) {
    let weights_ptr = filter_weights.weights.as_ptr();

    let mut unsafe_source_ptr_0 = image_store.buffer.as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    for _ in 0..destination.height {
        let mut filter_offset = 0usize;

        for x in 0..destination.width {
            let mut sum_r = 0f32;
            let mut sum_g = 0f32;
            let mut sum_b = 0f32;

            let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * image_store.channels;
                let weight = unsafe { weights_ptr.add(j + filter_offset).read_unaligned() };
                let src = unsafe { unsafe_source_ptr_0.add(px) };
                sum_r += unsafe { src.read_unaligned() } * weight;
                sum_g += unsafe { src.add(1).read_unaligned() } * weight;
                sum_b += unsafe { src.add(2).read_unaligned() } * weight;
            }

            let px = x * image_store.channels;

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

            unsafe {
                *dest_ptr = sum_r;
                *dest_ptr.add(1) = sum_g;
                *dest_ptr.add(2) = sum_b;
            }

            filter_offset += filter_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn convolve_vertical_neon(
    image_store: &ImageStore<f32, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 3>,
) {
    let unsafe_source_ptr_0 = image_store.buffer.as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.as_mut_ptr();
    let src_stride = image_store.width * image_store.channels;
    let mut filter_offset = 0usize;
    let dst_stride = destination.width * image_store.channels;
    let total_width = destination.width * image_store.channels;

    for y in 0..destination.height {
        let mut cx = 0usize;
        let bounds = unsafe { filter_weights.bounds.get_unchecked(y) };
        let weight_ptr = unsafe { filter_weights.weights.as_ptr().add(filter_offset) };

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

        filter_offset += filter_weights.aligned_size;
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

fn convolve_vertical_native(
    image_store: &ImageStore<f32, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<f32, 3>,
) {
    let unsafe_source_ptr_0 = image_store.buffer.as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;

    let mut filter_offset = 0usize;

    let dst_stride = destination.width * image_store.channels;

    for y in 0..destination.height {
        let mut cx = 0usize;
        let bounds = unsafe { filter_weights.bounds.get_unchecked(y) };
        let weight_ptr = unsafe { filter_weights.weights.as_ptr().add(filter_offset) };

        while cx + 12 < destination.width {
            unsafe {
                convolve_vertical_part_f32::<12, 3>(
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

        while cx + 8 < destination.width {
            unsafe {
                convolve_vertical_part_f32::<8, 3>(
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

        while cx < destination.width {
            unsafe {
                convolve_vertical_part_f32::<1, 3>(
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

        filter_offset += filter_weights.aligned_size;
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

impl HorizontalConvolutionPass<f32, 3> for ImageStore<f32, 3> {
    #[inline(always)]
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 3>,
    ) {
        #[allow(unused_assignments)]
        let mut using_feature = AccelerationFeature::Native;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            using_feature = AccelerationFeature::Neon;
        }
        match using_feature {
            AccelerationFeature::Neon => {
                convolve_horizontal_neon(self, filter_weights, destination);
            }
            AccelerationFeature::Native => {
                convolve_horizontal_native(self, filter_weights, destination);
            }
        }
    }
}

impl VerticalConvolutionPass<f32, 3> for ImageStore<f32, 3> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<f32, 3>,
    ) {
        #[allow(unused_assignments)]
        let mut using_feature = AccelerationFeature::Native;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            using_feature = AccelerationFeature::Neon;
        }
        match using_feature {
            AccelerationFeature::Neon => {
                convolve_vertical_neon(self, filter_weights, destination);
            }
            AccelerationFeature::Native => {
                convolve_vertical_native(self, filter_weights, destination);
            }
        }
    }
}
