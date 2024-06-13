/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod sse_convolve_f32 {
    use crate::filter_weights::{FilterBounds, FilterWeights};
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline(always)]
    pub(crate) unsafe fn convolve_horizontal_parts_4_rgba_f32(
        start_x: usize,
        src: *const f32,
        weight0: __m128,
        weight1: __m128,
        weight2: __m128,
        weight3: __m128,
        store_0: __m128,
    ) -> __m128 {
        const COMPONENTS: usize = 4;
        let src_ptr = src.add(start_x * COMPONENTS);

        let rgb_pixel_0 = _mm_loadu_ps(src_ptr);
        let rgb_pixel_1 = _mm_loadu_ps(src_ptr.add(4));
        let rgb_pixel_2 = _mm_loadu_ps(src_ptr.add(8));
        let rgb_pixel_3 = _mm_loadu_ps(src_ptr.add(12));

        let acc = _mm_prefer_fma_ps(store_0, rgb_pixel_0, weight0);
        let acc = _mm_prefer_fma_ps(acc, rgb_pixel_1, weight1);
        let acc = _mm_prefer_fma_ps(acc, rgb_pixel_2, weight2);
        let acc = _mm_prefer_fma_ps(acc, rgb_pixel_3, weight3);
        acc
    }
    
    #[inline(always)]
    pub(crate) unsafe fn convolve_horizontal_parts_one_rgba_f32(
        start_x: usize,
        src: *const f32,
        weight0: __m128,
        store_0: __m128,
    ) -> __m128 {
        const COMPONENTS: usize = 4;
        let src_ptr = src.add(start_x * COMPONENTS);
        let rgb_pixel = _mm_loadu_ps(src_ptr);
        let acc = _mm_prefer_fma_ps(store_0, rgb_pixel, weight0);
        acc
    }

    pub unsafe fn convolve_horizontal_rgba_sse_rows_4(
        dst_width: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        src_stride: usize,
        unsafe_destination_ptr_0: *mut f32,
        dst_stride: usize,
    ) {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let zeros = unsafe { _mm_setzero_ps() };
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 4 < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { _mm_set1_ps(ptr.read_unaligned()) };
                let weight1 = unsafe { _mm_set1_ps(ptr.add(1).read_unaligned()) };
                let weight2 = unsafe { _mm_set1_ps(ptr.add(2).read_unaligned()) };
                let weight3 = unsafe { _mm_set1_ps(ptr.add(3).read_unaligned()) };
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
                let weight0 = unsafe { _mm_set1_ps(ptr.read_unaligned()) };
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

            let px = x * CHANNELS;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                _mm_storeu_ps(dest_ptr, store_0);
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride) };
            unsafe {
                _mm_storeu_ps(dest_ptr, store_1);
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 2) };
            unsafe {
                _mm_storeu_ps(dest_ptr, store_2);
            }

            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px + dst_stride * 3) };
            unsafe {
                _mm_storeu_ps(dest_ptr, store_3);
            }

            filter_offset += filter_weights.aligned_size;
        }
    }

    pub unsafe fn convolve_horizontal_rgba_sse_row_one(
        dst_width: usize,
        filter_weights: &FilterWeights<f32>,
        unsafe_source_ptr_0: *const f32,
        unsafe_destination_ptr_0: *mut f32,
    ) {
        const CHANNELS: usize = 4;
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = unsafe { filter_weights.bounds.get_unchecked(x) };
            let mut jx = 0usize;
            let mut store = unsafe { _mm_setzero_ps() };

            while jx + 4 < bounds.size {
                let ptr = unsafe { weights_ptr.add(jx + filter_offset) };
                let weight0 = unsafe { _mm_set1_ps(ptr.read_unaligned()) };
                let weight1 = unsafe { _mm_set1_ps(ptr.add(1).read_unaligned()) };
                let weight2 = unsafe { _mm_set1_ps(ptr.add(2).read_unaligned()) };
                let weight3 = unsafe { _mm_set1_ps(ptr.add(3).read_unaligned()) };
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
                let weight0 = unsafe { _mm_set1_ps(ptr.read_unaligned()) };
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

            let px = x * CHANNELS;
            let dest_ptr = unsafe { unsafe_destination_ptr_0.add(px) };
            unsafe {
                _mm_storeu_ps(dest_ptr, store);
            }

            filter_offset += filter_weights.aligned_size;
        }
    }
    
    #[allow(unused)]
    #[inline(always)]
    pub(crate) unsafe fn convolve_vertical_part_sse_16_f32(
        start_y: usize,
        start_x: usize,
        src: *const f32,
        src_stride: usize,
        dst: *mut f32,
        filter: *const f32,
        bounds: &FilterBounds,
    ) {
        let mut store_0 = _mm_setzero_ps();
        let mut store_1 = _mm_setzero_ps();
        let mut store_2 = _mm_setzero_ps();
        let mut store_3 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = unsafe { filter.add(j).read_unaligned() };
            let v_weight = _mm_set1_ps(weight);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row_0 = _mm_loadu_ps(s_ptr);
            let item_row_1 = _mm_loadu_ps(s_ptr.add(4));
            let item_row_2 = _mm_loadu_ps(s_ptr.add(8));
            let item_row_3 = _mm_loadu_ps(s_ptr.add(12));

            store_0 = _mm_prefer_fma_ps(store_0, item_row_0, v_weight);
            store_1 = _mm_prefer_fma_ps(store_1, item_row_1, v_weight);
            store_2 = _mm_prefer_fma_ps(store_2, item_row_2, v_weight);
            store_3 = _mm_prefer_fma_ps(store_3, item_row_3, v_weight);
        }

        let dst_ptr = dst.add(px);
        _mm_storeu_ps(dst_ptr, store_0);
        _mm_storeu_ps(dst_ptr.add(4), store_1);
        _mm_storeu_ps(dst_ptr.add(8), store_2);
        _mm_storeu_ps(dst_ptr.add(12), store_3);
    }

    #[allow(unused)]
    #[inline(always)]
    pub(crate) unsafe fn convolve_vertical_part_sse_8_f32(
        start_y: usize,
        start_x: usize,
        src: *const f32,
        src_stride: usize,
        dst: *mut f32,
        filter: *const f32,
        bounds: &FilterBounds,
    ) {
        let mut store_0 = _mm_setzero_ps();
        let mut store_1 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = unsafe { filter.add(j).read_unaligned() };
            let v_weight = _mm_set1_ps(weight);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row_0 = _mm_loadu_ps(s_ptr);
            let item_row_1 = _mm_loadu_ps(s_ptr.add(4));

            store_0 = _mm_prefer_fma_ps(store_0, item_row_0, v_weight);
            store_1 = _mm_prefer_fma_ps(store_1, item_row_1, v_weight);
        }

        let dst_ptr = dst.add(px);
        _mm_storeu_ps(dst_ptr, store_0);
        _mm_storeu_ps(dst_ptr.add(4), store_1);
    }

    #[allow(unused)]
    #[inline(always)]
    pub(crate) unsafe fn convolve_vertical_part_sse_4_f32(
        start_y: usize,
        start_x: usize,
        src: *const f32,
        src_stride: usize,
        dst: *mut f32,
        filter: *const f32,
        bounds: &FilterBounds,
    ) {
        let mut store_0 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = unsafe { filter.add(j).read_unaligned() };
            let v_weight = _mm_set1_ps(weight);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row_0 = _mm_loadu_ps(s_ptr);

            store_0 = _mm_prefer_fma_ps(store_0, item_row_0, v_weight);
        }

        let dst_ptr = dst.add(px);
        _mm_storeu_ps(dst_ptr, store_0);
    }

    #[allow(unused)]
    #[inline(always)]
    pub(crate) unsafe fn convolve_vertical_part_sse_f32(
        start_y: usize,
        start_x: usize,
        src: *const f32,
        src_stride: usize,
        dst: *mut f32,
        filter: *const f32,
        bounds: &FilterBounds,
    ) {
        let mut store_0 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = unsafe { filter.add(j).read_unaligned() };
            let v_weight = _mm_set1_ps(weight);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row_0 = _mm_set1_ps(s_ptr.read_unaligned());

            store_0 = _mm_prefer_fma_ps(store_0, item_row_0, v_weight);
        }

        let dst_ptr = dst.add(px);
        dst_ptr.write_unaligned(f32::from_bits(_mm_extract_ps::<0>(store_0) as u32));
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[cfg(not(target_feature = "fma"))]
    #[inline]
    #[allow(dead_code)]
    pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
        return _mm_add_ps(_mm_mul_ps(b, c), a);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[cfg(target_feature = "fma")]
    #[inline]
    #[allow(dead_code)]
    pub unsafe fn _mm_prefer_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
        return _mm_fmadd_ps(b, c, a);
    }

}