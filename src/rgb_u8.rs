use crate::acceleration_feature::AccelerationFeature;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_u8::*;
use crate::filter_weights::FilterWeights;
use crate::image_store::ImageStore;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon_simd_u8::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse_simd_u8::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn convolve_horizontal_rgb_sse(
    image_store: &ImageStore<u8, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 3>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let weights_ptr = approx_weights.weights.as_ptr();

    let mut unsafe_source_ptr_0 = image_store.buffer.as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    #[rustfmt::skip]
    let shuffle = unsafe { _mm_setr_epi8(0, 1, 2, -1, 3, 4,
                                            5, -1, 6, 7, 8, -1,
                                            9, 10, 11, -1) };

    #[rustfmt::skip]
    let shuffle_2 = unsafe {
        _mm_setr_epi8(0, 1,2, -1, 3,4,5, -1, -1, -1, -1,-1,-1,-1,-1,-1)
    };

    let mut yy = 0usize;

    while yy + 4 < destination.height {
        let mut filter_offset = 0usize;

        for x in 0..destination.width {
            let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store_0 = unsafe { _mm_setzero_si128() };
            let mut store_1 = unsafe { _mm_setzero_si128() };
            let mut store_2 = unsafe { _mm_setzero_si128() };
            let mut store_3 = unsafe { _mm_setzero_si128() };

            while jx + 4 < bounds.size && x + 6 < destination.width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                let weight2 = unsafe { ptr.add(2).read_unaligned() };
                let weight3 = unsafe { ptr.add(3).read_unaligned() };
                unsafe {
                    store_0 = sse_convolve_u8::convolve_horizontal_parts_4_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_0,
                        shuffle,
                    );
                    store_1 = sse_convolve_u8::convolve_horizontal_parts_4_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_2,
                        shuffle,
                    );
                    store_2 = sse_convolve_u8::convolve_horizontal_parts_4_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_2,
                        shuffle,
                    );
                    store_3 = sse_convolve_u8::convolve_horizontal_parts_4_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_3,
                        shuffle,
                    );
                }
                jx += 4;
            }

            while jx + 2 < bounds.size && x + 3 < image_store.width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                unsafe {
                    store_0 = sse_convolve_u8::convolve_horizontal_parts_two_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        store_0,
                        shuffle_2,
                    );
                    store_1 = sse_convolve_u8::convolve_horizontal_parts_two_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        weight1,
                        store_1,
                        shuffle_2,
                    );
                    store_2 = sse_convolve_u8::convolve_horizontal_parts_two_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        weight1,
                        store_2,
                        shuffle_2,
                    );
                    store_3 = sse_convolve_u8::convolve_horizontal_parts_two_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        weight1,
                        store_3,
                        shuffle_2,
                    );
                }
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                unsafe {
                    store_0 = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        store_0,
                    );
                    store_1 = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        store_1,
                    );
                    store_2 = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        store_2,
                    );
                    store_3 = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        store_3,
                    );
                }
                jx += 1;
            }
            let store_0_8 = sse_convolve_u8::compress_i32(store_0);

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

            let element = unsafe { _mm_extract_epi32::<0>(store_0_8) };
            let bytes = element.to_le_bytes();
            unsafe {
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            let store_1_8 = sse_convolve_u8::compress_i32(store_1);

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride) };

            let element = unsafe { _mm_extract_epi32::<0>(store_1_8) };
            let bytes = element.to_le_bytes();
            unsafe {
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            let store_2_8 = sse_convolve_u8::compress_i32(store_2);

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 2) };

            let element = unsafe { _mm_extract_epi32::<0>(store_2_8) };
            let bytes = element.to_le_bytes();
            unsafe {
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            let store_3_8 = sse_convolve_u8::compress_i32(store_3);

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 3) };

            let element = unsafe { _mm_extract_epi32::<0>(store_3_8) };
            let bytes = element.to_le_bytes();
            unsafe {
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            filter_offset += approx_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride * 4) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride * 4) };
        yy += 4;
    }

    for _ in yy..destination.height {
        let mut filter_offset = 0usize;

        for x in 0..destination.width {
            let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store = unsafe { _mm_setzero_si128() };

            while jx + 4 < bounds.size && x + 6 < destination.width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                let weight2 = unsafe { ptr.add(2).read_unaligned() };
                let weight3 = unsafe { ptr.add(3).read_unaligned() };
                unsafe {
                    store = sse_convolve_u8::convolve_horizontal_parts_4_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store,
                        shuffle,
                    );
                }
                jx += 4;
            }

            while jx + 2 < bounds.size && x + 3 < image_store.width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                unsafe {
                    store = sse_convolve_u8::convolve_horizontal_parts_two_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        store,
                        shuffle_2,
                    );
                }
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                unsafe {
                    store = sse_convolve_u8::convolve_horizontal_parts_one_sse_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                    );
                }
                jx += 1;
            }

            let store_16_8 = sse_convolve_u8::compress_i32(store);

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

            let element = unsafe { _mm_extract_epi32::<0>(store_16_8) };
            let bytes = element.to_le_bytes();
            unsafe {
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }
            unsafe {
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            filter_offset += approx_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn convolve_horizontal_rgb_neon(
    image_store: &ImageStore<u8, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 3>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);

    let weights_ptr = approx_weights.weights.as_ptr();

    let mut unsafe_source_ptr_0 = image_store.buffer.as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    let shuf_table_1: [u8; 8] = [0, 1, 2, 255, 3, 4, 5, 255];
    let shuffle_1 = unsafe { vld1_u8(shuf_table_1.as_ptr()) };
    let shuf_table_2: [u8; 8] = [6, 7, 8, 255, 9, 10, 11, 255];
    let shuffle_2 = unsafe { vld1_u8(shuf_table_2.as_ptr()) };
    let shuffle = unsafe { vcombine_u8(shuffle_1, shuffle_2) };

    let zeros = unsafe { vdupq_n_s32(0i32) };

    let mut yy = 0usize;

    while yy + 4 < destination.height {
        let mut filter_offset = 0usize;

        for x in 0..destination.width {
            let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
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
                    store_0 = neon_convolve_u8::convolve_horizontal_parts_4_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_0,
                        shuffle,
                    );
                    store_1 = neon_convolve_u8::convolve_horizontal_parts_4_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_1,
                        shuffle,
                    );
                    store_2 = neon_convolve_u8::convolve_horizontal_parts_4_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_1,
                        shuffle,
                    );
                    store_3 = neon_convolve_u8::convolve_horizontal_parts_4_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store_1,
                        shuffle,
                    );
                }
                jx += 4;
            }

            while jx + 2 < bounds.size && x + 3 < destination.width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                unsafe {
                    store_0 = neon_convolve_u8::convolve_horizontal_parts_2_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        store_0,
                        shuffle_1,
                    );
                    store_1 = neon_convolve_u8::convolve_horizontal_parts_2_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        weight1,
                        store_1,
                        shuffle_1,
                    );
                    store_2 = neon_convolve_u8::convolve_horizontal_parts_2_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        weight1,
                        store_2,
                        shuffle_1,
                    );
                    store_3 = neon_convolve_u8::convolve_horizontal_parts_2_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        weight1,
                        store_3,
                        shuffle_1,
                    );
                }
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                unsafe {
                    store_0 = neon_convolve_u8::convolve_horizontal_parts_one_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        store_0,
                    );
                    store_1 = neon_convolve_u8::convolve_horizontal_parts_one_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride),
                        weight0,
                        store_1,
                    );
                    store_2 = neon_convolve_u8::convolve_horizontal_parts_one_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 2),
                        weight0,
                        store_2,
                    );
                    store_3 = neon_convolve_u8::convolve_horizontal_parts_one_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0.add(src_stride * 3),
                        weight0,
                        store_2,
                    );
                }
                jx += 1;
            }

            let store_16 = unsafe { vqshrun_n_s32::<12>(vmaxq_s32(store_0, zeros)) };
            let store_16_8 = unsafe { vqmovn_u16(vcombine_u16(store_16, store_16)) };

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            let store_16 = unsafe { vqshrun_n_s32::<12>(vmaxq_s32(store_1, zeros)) };
            let store_16_8 = unsafe { vqmovn_u16(vcombine_u16(store_16, store_16)) };

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride) };
            unsafe {
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            let store_16 = unsafe { vqshrun_n_s32::<12>(vmaxq_s32(store_2, zeros)) };
            let store_16_8 = unsafe { vqmovn_u16(vcombine_u16(store_16, store_16)) };

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 2) };
            unsafe {
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            let store_16 = unsafe { vqshrun_n_s32::<12>(vmaxq_s32(store_3, zeros)) };
            let store_16_8 = unsafe { vqmovn_u16(vcombine_u16(store_16, store_16)) };

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 3) };
            unsafe {
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            filter_offset += approx_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };

        yy += 4;
    }

    for _ in yy..destination.height {
        let mut filter_offset = 0usize;

        for x in 0..destination.width {
            let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store = zeros;

            while jx + 4 < bounds.size && x + 6 < destination.width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                let weight2 = unsafe { ptr.add(2).read_unaligned() };
                let weight3 = unsafe { ptr.add(3).read_unaligned() };
                unsafe {
                    store = neon_convolve_u8::convolve_horizontal_parts_4_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        weight2,
                        weight3,
                        store,
                        shuffle,
                    );
                }
                jx += 4;
            }

            while jx + 2 < bounds.size && x + 3 < destination.width {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                let weight1 = unsafe { ptr.add(1).read_unaligned() };
                unsafe {
                    store = neon_convolve_u8::convolve_horizontal_parts_2_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        weight1,
                        store,
                        shuffle_1,
                    );
                }
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { ptr.read_unaligned() };
                unsafe {
                    store = neon_convolve_u8::convolve_horizontal_parts_one_rgb(
                        bounds.start + jx,
                        unsafe_source_ptr_0,
                        weight0,
                        store,
                    );
                }
                jx += 1;
            }

            let store_16 = unsafe { vqshrun_n_s32::<12>(vmaxq_s32(store, zeros)) };
            let store_16_8 = unsafe { vqmovn_u16(vcombine_u16(store_16, store_16)) };

            let px = x * image_store.channels;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
                let bytes = pixel.to_le_bytes();
                *dest_ptr = bytes[0];
                *dest_ptr.add(1) = bytes[1];
                *dest_ptr.add(2) = bytes[2];
            }

            filter_offset += approx_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

fn convolve_horizontal_rgb_native(
    image_store: &ImageStore<u8, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 3>,
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
            let mut sum_r = 0i32;
            let mut sum_g = 0i32;
            let mut sum_b = 0i32;

            let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
            let start_x = bounds.start;
            for j in 0..bounds.size {
                let px = (start_x + j) * image_store.channels;
                let weight = unsafe { weights_ptr.add(j + filter_offset).read_unaligned() } as i32;
                let src = unsafe { unsafe_source_ptr_0.add(px) };
                sum_r += unsafe { src.read_unaligned() } as i32 * weight;
                sum_g += unsafe { src.add(1).read_unaligned() } as i32 * weight;
                sum_b += unsafe { src.add(2).read_unaligned() } as i32 * weight;
            }

            let px = x * image_store.channels;

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

            unsafe {
                *dest_ptr = (sum_r >> 12).min(255).max(0) as u8;
                *dest_ptr.add(1) = (sum_g >> 12).min(255).max(0) as u8;
                *dest_ptr.add(2) = (sum_b >> 12).min(255).max(0) as u8;
            }

            filter_offset += approx_weights.aligned_size;
        }

        unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn convolve_vertical_rgb_sse(
    image_store: &ImageStore<u8, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 3>,
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
                sse_convolve_u8::convolve_vertical_part_sse_32(
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
                sse_convolve_u8::convolve_vertical_part_sse_16(
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
                sse_convolve_u8::convolve_vertical_part_sse_8::<false>(
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
                sse_convolve_u8::convolve_vertical_part_sse_8::<true>(
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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn convolve_vertical_rgb_neon(
    image_store: &ImageStore<u8, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 3>,
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

#[inline(always)]
fn convolve_vertical_rgb_native(
    image_store: &ImageStore<u8, 3>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u8, 3>,
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<12>(0);
    let unsafe_source_ptr_0 = image_store.buffer.as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.as_mut_ptr();
    let src_stride = image_store.width * image_store.channels;
    let mut filter_offset = 0usize;
    let dst_stride = destination.width * image_store.channels;

    for y in 0..destination.height {
        let mut cx = 0usize;
        let bounds = unsafe { approx_weights.bounds.get_unchecked(y) };
        let weight_ptr = unsafe { approx_weights.weights.as_ptr().add(filter_offset) };

        while cx + 12 < destination.width {
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

        while cx + 8 < destination.width {
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

        while cx < destination.width {
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

        filter_offset += approx_weights.aligned_size;
        unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
    }
}

impl HorizontalConvolutionPass<u8, 3> for ImageStore<u8, 3> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 3>,
    ) {
        #[allow(unused_assignments)]
        let mut using_feature = AccelerationFeature::Native;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            using_feature = AccelerationFeature::Neon;
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                using_feature = AccelerationFeature::Sse;
            }
        }
        match using_feature {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            AccelerationFeature::Neon => {
                convolve_horizontal_rgb_neon(self, filter_weights, destination);
            }
            AccelerationFeature::Native => {
                convolve_horizontal_rgb_native(self, filter_weights, destination);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {
                convolve_horizontal_rgb_sse(self, filter_weights, destination);
            }
        }
    }
}

impl VerticalConvolutionPass<u8, 3> for ImageStore<u8, 3> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u8, 3>,
    ) {
        #[allow(unused_assignments)]
        let mut using_feature = AccelerationFeature::Native;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            using_feature = AccelerationFeature::Neon;
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                using_feature = AccelerationFeature::Sse;
            }
        }
        match using_feature {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            AccelerationFeature::Neon => {
                convolve_vertical_rgb_neon(self, filter_weights, destination);
            }
            AccelerationFeature::Native => {
                convolve_vertical_rgb_native(self, filter_weights, destination);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            AccelerationFeature::Sse => {
                convolve_vertical_rgb_sse(self, filter_weights, destination);
            }
        }
    }
}
