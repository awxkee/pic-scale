use crate::filter_weights::FilterWeights;
use crate::neon::utils::neon_convolve_u8::{
    convolve_horizontal_parts_2_rgba, convolve_horizontal_parts_4_rgba,
    convolve_horizontal_parts_one_rgba,
};
use crate::support::ROUNDING_APPROX;
use std::arch::aarch64::*;

pub fn convolve_horizontal_rgba_neon_rows_4_u8(
    dst_width: usize,
    _: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = approx_weights.weights.as_ptr();
        const CHANNELS: usize = 4;
        let zeros = vdupq_n_s32(0i32);
        let init = vdupq_n_s32(ROUNDING_APPROX);
        for x in 0..dst_width {
            let bounds = approx_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store_0 = init;
            let mut store_1 = init;
            let mut store_2 = init;
            let mut store_3 = init;

            while jx + 4 < bounds.size {
                let bounds_start = bounds.start + jx;
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vdup_n_s16(ptr.read_unaligned());
                let weight1 = vdupq_n_s16(ptr.add(1).read_unaligned());
                let weight2 = vdup_n_s16(ptr.add(2).read_unaligned());
                let weight3 = vdupq_n_s16(ptr.add(3).read_unaligned());
                store_0 = convolve_horizontal_parts_4_rgba(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_4_rgba(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_4_rgba(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_4_rgba(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 3),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_3,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vdup_n_s16(ptr.read_unaligned());
                let weight1 = vdupq_n_s16(ptr.add(1).read_unaligned());
                store_0 = convolve_horizontal_parts_2_rgba(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_2_rgba(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    weight1,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_2_rgba(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    weight1,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_2_rgba(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 3),
                    weight0,
                    weight1,
                    store_3,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vdup_n_s16(ptr.read_unaligned());
                store_0 = convolve_horizontal_parts_one_rgba(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_one_rgba(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride),
                    weight0,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_one_rgba(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 2),
                    weight0,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_one_rgba(
                    bounds_start,
                    unsafe_source_ptr_0.add(src_stride * 3),
                    weight0,
                    store_3,
                );
                jx += 1;
            }

            let store_16 = vqshrun_n_s32::<12>(vmaxq_s32(store_0, zeros));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            let store_16 = vqshrun_n_s32::<12>(vmaxq_s32(store_1, zeros));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            let store_16 = vqshrun_n_s32::<12>(vmaxq_s32(store_2, zeros));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            let store_16 = vqshrun_n_s32::<12>(vmaxq_s32(store_3, zeros));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
            let pixel = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(pixel);

            filter_offset += approx_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgba_neon_row(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    unsafe {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;

        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let mut jx = 0usize;
            let mut store = vdupq_n_s32(ROUNDING_APPROX);

            while jx + 4 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vdup_n_s16(ptr.read_unaligned());
                let weight1 = vdupq_n_s16(ptr.add(1).read_unaligned());
                let weight2 = vdup_n_s16(ptr.add(2).read_unaligned());
                let weight3 = vdupq_n_s16(ptr.add(3).read_unaligned());
                store = convolve_horizontal_parts_4_rgba(
                    bounds.start + jx,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store,
                );
                jx += 4;
            }

            while jx + 2 < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let bounds_start = bounds.start + jx;
                let weight0 = vdup_n_s16(ptr.read_unaligned());
                let weight1 = vdupq_n_s16(ptr.add(1).read_unaligned());
                store = convolve_horizontal_parts_2_rgba(
                    bounds_start,
                    unsafe_source_ptr_0,
                    weight0,
                    weight1,
                    store,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let ptr = weights_ptr.add(jx + filter_offset);
                let weight0 = vdup_n_s16(ptr.read_unaligned());
                store = convolve_horizontal_parts_one_rgba(
                    bounds.start + jx,
                    unsafe_source_ptr_0,
                    weight0,
                    store,
                );
                jx += 1;
            }

            let store_16 = vqshrun_n_s32::<12>(vmaxq_s32(store, vdupq_n_s32(0i32)));
            let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

            let px = x * CHANNELS;
            let dest_ptr = unsafe_destination_ptr_0.add(px);
            let value = vget_lane_u32::<0>(vreinterpret_u32_u8(store_16_8));
            let dest_ptr_32 = dest_ptr as *mut u32;
            dest_ptr_32.write_unaligned(value);

            filter_offset += filter_weights.aligned_size;
        }
    }
}
