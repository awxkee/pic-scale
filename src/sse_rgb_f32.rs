/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod sse_convolve_f32 {
    use crate::filter_weights::FilterBounds;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

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
        let mut store_0 = _mm_set1_ps(0f32);
        let mut store_1 = _mm_set1_ps(0f32);
        let mut store_2 = _mm_set1_ps(0f32);
        let mut store_3 = _mm_set1_ps(0f32);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = *unsafe { filter.add(j) };
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
        let mut store_0 = _mm_set1_ps(0f32);
        let mut store_1 = _mm_set1_ps(0f32);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = *unsafe { filter.add(j) };
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
        let mut store_0 = _mm_set1_ps(0f32);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = *unsafe { filter.add(j) };
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
        let mut store_0 = _mm_set1_ps(0f32);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = *unsafe { filter.add(j) };
            let v_weight = _mm_set1_ps(weight);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row_0 = _mm_set1_ps(*s_ptr);

            store_0 = _mm_prefer_fma_ps(store_0, item_row_0, v_weight);
        }

        let dst_ptr = dst.add(px);
        *dst_ptr = f32::from_bits(_mm_extract_ps::<0>(store_0) as u32);
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