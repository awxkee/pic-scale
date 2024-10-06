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
use crate::support::{PRECISION, ROUNDING_CONST};
use crate::wasm32::utils::{u16x8_pack_sat_u8x16, u32x4_pack_trunc_u16x8, w_zeros};
use std::arch::wasm32::*;

#[inline]
unsafe fn convolve_horizontal_parts_one_wasm_rgb(
    start_x: usize,
    src: *const u8,
    weight0: v128,
    store_0: v128,
) -> v128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);
    let vl = i32::from_le_bytes([
        src_ptr.read_unaligned(),
        src_ptr.add(1).read_unaligned(),
        src_ptr.add(2).read_unaligned(),
        0,
    ]);
    let m_vl = i32x4_replace_lane::<0>(w_zeros(), vl);
    let lo = u16x8_extend_low_u8x16(m_vl);
    i32x4_add(store_0, i32x4_extmul_low_i16x8(lo, weight0))
}

pub fn convolve_horizontal_rgb_wasm_rows_4(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    unsafe {
        convolve_horizontal_rgb_wasm_rows_4_impl(
            dst_width,
            src_width,
            approx_weights,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            dst_stride,
        );
    }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn convolve_horizontal_rgb_wasm_rows_4_impl(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut u8,
    dst_stride: usize,
) {
    const CHANNELS: usize = 3;
    let mut filter_offset = 0usize;
    let weights_ptr = approx_weights.weights.as_ptr();

    let vld = i32x4_splat(ROUNDING_CONST);

    let zeros = w_zeros();

    for x in 0..dst_width {
        let bounds = approx_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;

        // Will make step in 4 items however since it is RGB it is necessary to make a safe offset
        while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight01 = v128_load32_splat(ptr as *const u32);
            let weight23 = v128_load32_splat(ptr.add(2) as *const u32);
            let start_bounds = bounds.start + jx;

            let src_ptr = unsafe_source_ptr_0.add(start_bounds * CHANNELS);

            let rgb_pixel_0 = v128_load(src_ptr as *const v128);
            let rgb_pixel_1 = v128_load(src_ptr.add(src_stride) as *const v128);
            let rgb_pixel_2 = v128_load(src_ptr.add(src_stride * 2) as *const v128);
            let rgb_pixel_3 = v128_load(src_ptr.add(src_stride * 3) as *const v128);

            let hi_0 = i8x16_shuffle::<6, 16, 9, 16, 7, 16, 10, 16, 8, 16, 11, 16, 16, 16, 16, 16>(
                rgb_pixel_0,
                zeros,
            );
            let lo_0 = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel_0,
                zeros,
            );
            let hi_1 = i8x16_shuffle::<6, 16, 9, 16, 7, 16, 10, 16, 8, 16, 11, 16, 16, 16, 16, 16>(
                rgb_pixel_1,
                zeros,
            );
            let lo_1 = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel_1,
                zeros,
            );
            let hi_2 = i8x16_shuffle::<6, 16, 9, 16, 7, 16, 10, 16, 8, 16, 11, 16, 16, 16, 16, 16>(
                rgb_pixel_2,
                zeros,
            );
            let lo_2 = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel_2,
                zeros,
            );
            let hi_3 = i8x16_shuffle::<6, 16, 9, 16, 7, 16, 10, 16, 8, 16, 11, 16, 16, 16, 16, 16>(
                rgb_pixel_3,
                zeros,
            );
            let lo_3 = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel_3,
                zeros,
            );

            store_0 = i32x4_add(store_0, i32x4_dot_i16x8(lo_0, weight01));
            store_0 = i32x4_add(store_0, i32x4_dot_i16x8(hi_0, weight23));

            store_1 = i32x4_add(store_1, i32x4_dot_i16x8(lo_1, weight01));
            store_1 = i32x4_add(store_1, i32x4_dot_i16x8(hi_1, weight23));

            store_2 = i32x4_add(store_2, i32x4_dot_i16x8(lo_2, weight01));
            store_2 = i32x4_add(store_2, i32x4_dot_i16x8(hi_2, weight23));

            store_3 = i32x4_add(store_3, i32x4_dot_i16x8(lo_3, weight01));
            store_3 = i32x4_add(store_3, i32x4_dot_i16x8(hi_3, weight23));
            jx += 4;
        }

        while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
            let ptr = weights_ptr.add(jx + filter_offset);
            let bounds_start = bounds.start + jx;
            let weight01 = v128_load32_splat(ptr as *const u32);

            let src_ptr = unsafe_source_ptr_0.add(bounds_start * CHANNELS);

            let rgb_pixel_0 = v128_load(src_ptr as *const v128);
            let rgb_pixel_1 = v128_load(src_ptr.add(src_stride) as *const v128);
            let rgb_pixel_2 = v128_load(src_ptr.add(src_stride * 2) as *const v128);
            let rgb_pixel_3 = v128_load(src_ptr.add(src_stride * 3) as *const v128);

            let lo_0 = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel_0,
                zeros,
            );
            let lo_1 = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel_1,
                zeros,
            );
            let lo_2 = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel_2,
                zeros,
            );
            let lo_3 = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel_3,
                zeros,
            );

            store_0 = i32x4_add(store_0, i32x4_dot_i16x8(lo_0, weight01));
            store_1 = i32x4_add(store_1, i32x4_dot_i16x8(lo_1, weight01));
            store_2 = i32x4_add(store_2, i32x4_dot_i16x8(lo_2, weight01));
            store_3 = i32x4_add(store_3, i32x4_dot_i16x8(lo_3, weight01));

            jx += 2;
        }

        while jx < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let bounds_start = bounds.start + jx;

            let weight0 = v128_load16_splat(ptr as *const u16);

            store_0 = convolve_horizontal_parts_one_wasm_rgb(
                bounds_start,
                unsafe_source_ptr_0,
                weight0,
                store_0,
            );
            store_1 = convolve_horizontal_parts_one_wasm_rgb(
                bounds_start,
                unsafe_source_ptr_0.add(src_stride),
                weight0,
                store_1,
            );
            store_2 = convolve_horizontal_parts_one_wasm_rgb(
                bounds_start,
                unsafe_source_ptr_0.add(src_stride * 2),
                weight0,
                store_2,
            );
            store_3 = convolve_horizontal_parts_one_wasm_rgb(
                bounds_start,
                unsafe_source_ptr_0.add(src_stride * 3),
                weight0,
                store_3,
            );
            jx += 1;
        }

        store_0 = i32x4_max(store_0, zeros);
        store_1 = i32x4_max(store_1, zeros);
        store_2 = i32x4_max(store_2, zeros);
        store_3 = i32x4_max(store_3, zeros);
        store_0 = i32x4_shr(store_0, PRECISION as u32);
        store_1 = i32x4_shr(store_1, PRECISION as u32);
        store_2 = i32x4_shr(store_2, PRECISION as u32);
        store_3 = i32x4_shr(store_3, PRECISION as u32);

        let store_16_8_0 = u32x4_pack_trunc_u16x8(store_0, store_0);
        let store_16_8_1 = u32x4_pack_trunc_u16x8(store_1, store_1);
        let store_16_8_2 = u32x4_pack_trunc_u16x8(store_2, store_2);
        let store_16_8_3 = u32x4_pack_trunc_u16x8(store_3, store_3);
        let store_8_16_0 = u16x8_pack_sat_u8x16(store_16_8_0, store_16_8_0);
        let store_8_16_1 = u16x8_pack_sat_u8x16(store_16_8_1, store_16_8_1);
        let store_8_16_2 = u16x8_pack_sat_u8x16(store_16_8_2, store_16_8_2);
        let store_8_16_3 = u16x8_pack_sat_u8x16(store_16_8_3, store_16_8_3);
        let pixel_0 = i32x4_extract_lane::<0>(store_8_16_0);
        let pixel_1 = i32x4_extract_lane::<0>(store_8_16_1);
        let pixel_2 = i32x4_extract_lane::<0>(store_8_16_2);
        let pixel_3 = i32x4_extract_lane::<0>(store_8_16_3);

        let element_0 = pixel_0.to_le_bytes();
        let element_1 = pixel_1.to_le_bytes();
        let element_2 = pixel_2.to_le_bytes();
        let element_3 = pixel_3.to_le_bytes();

        let px = x * CHANNELS;
        let dest_ptr = unsafe_destination_ptr_0.add(px);

        let first_byte = u16::from_le_bytes([element_0[0], element_0[1]]);
        (dest_ptr as *mut u16).write_unaligned(first_byte);
        dest_ptr.add(2).write_unaligned(element_0[2]);

        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);

        let first_byte = u16::from_le_bytes([element_1[0], element_1[1]]);
        (dest_ptr as *mut u16).write_unaligned(first_byte);
        dest_ptr.add(2).write_unaligned(element_1[2]);

        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);

        let first_byte = u16::from_le_bytes([element_2[0], element_2[1]]);
        (dest_ptr as *mut u16).write_unaligned(first_byte);
        dest_ptr.add(2).write_unaligned(element_2[2]);

        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);

        let first_byte = u16::from_le_bytes([element_3[0], element_3[1]]);
        (dest_ptr as *mut u16).write_unaligned(first_byte);
        dest_ptr.add(2).write_unaligned(element_3[2]);

        filter_offset += approx_weights.aligned_size;
    }
}

pub fn convolve_horizontal_rgb_wasm_row_one(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    unsafe {
        convolve_horizontal_rgb_wasm_row_one_impl(
            dst_width,
            src_width,
            approx_weights,
            unsafe_source_ptr_0,
            unsafe_destination_ptr_0,
        );
    }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn convolve_horizontal_rgb_wasm_row_one_impl(
    dst_width: usize,
    src_width: usize,
    approx_weights: &FilterWeights<i16>,
    unsafe_source_ptr_0: *const u8,
    unsafe_destination_ptr_0: *mut u8,
) {
    const CHANNELS: usize = 3;
    let mut filter_offset = 0usize;
    let weights_ptr = approx_weights.weights.as_ptr();

    let zeros = w_zeros();

    for x in 0..dst_width {
        let bounds = unsafe { approx_weights.bounds.get_unchecked(x) };
        let mut jx = 0usize;
        let mut store = zeros;

        while jx + 4 < bounds.size && bounds.start + jx + 6 < src_width {
            let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
            let weight01 = v128_load32_splat(ptr as *const u32);
            let weight23 = v128_load32_splat(ptr.add(2) as *const u32);

            let bounds_start = bounds.start + jx;
            let src_ptr_0 = unsafe_source_ptr_0.add(bounds_start * CHANNELS);

            let rgb_pixel = v128_load(src_ptr_0 as *const v128);
            let hi = i8x16_shuffle::<6, 16, 9, 16, 7, 16, 10, 16, 8, 16, 11, 16, 16, 16, 16, 16>(
                rgb_pixel, zeros,
            );
            let lo = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel, zeros,
            );

            store = i32x4_add(store, i32x4_dot_i16x8(lo, weight01));
            store = i32x4_add(store, i32x4_dot_i16x8(hi, weight23));
            jx += 4;
        }

        while jx + 2 < bounds.size && bounds.start + jx + 3 < src_width {
            let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
            let weight0 = v128_load32_splat(ptr as *const u32);
            let src_ptr = unsafe_source_ptr_0.add((bounds.start + jx) * 3);
            let mut rgb_pixel = zeros;
            rgb_pixel =
                i64x2_replace_lane::<0>(rgb_pixel, (src_ptr as *const i64).read_unaligned());
            rgb_pixel = i8x16_shuffle::<0, 16, 3, 16, 1, 16, 4, 16, 2, 16, 5, 16, 16, 16, 16, 16>(
                rgb_pixel, zeros,
            );
            store = i32x4_add(store, i32x4_dot_i16x8(rgb_pixel, weight0));
            jx += 2;
        }

        while jx < bounds.size {
            let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
            let weight0 = v128_load16_splat(ptr as *const u16);
            store = convolve_horizontal_parts_one_wasm_rgb(
                bounds.start + jx,
                unsafe_source_ptr_0,
                weight0,
                store,
            );
            jx += 1;
        }

        store = i32x4_max(store, zeros);
        store = i32x4_shr(store, PRECISION as u32);

        let px = x * CHANNELS;
        let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };

        let store_16_8 = u32x4_pack_trunc_u16x8(store, store);
        let store_8_16 = u16x8_pack_sat_u8x16(store_16_8, store_16_8);
        let pixel = i32x4_extract_lane::<0>(store_8_16);
        let bytes = pixel.to_le_bytes();

        let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
        (dest_ptr as *mut u16).write_unaligned(first_byte);
        dest_ptr.add(2).write_unaligned(bytes[2]);

        filter_offset += approx_weights.aligned_size;
    }
}
