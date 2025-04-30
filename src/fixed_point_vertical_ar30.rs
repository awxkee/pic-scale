/*
 * Copyright (c) Radzivon Bartoshyk, 10/2024. All rights reserved.
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
#![forbid(unsafe_code)]
use crate::color_group::{load_ar30_p, ColorGroup};
use crate::filter_weights::FilterBounds;
use crate::support::ROUNDING_CONST;

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
pub(crate) fn convolve_column_handler_fip_db_ar30<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
    const BUFFER_SIZE: usize,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    filter: &[i16],
    bounds: &FilterBounds,
    x: usize,
) {
    if filter.is_empty() {
        return;
    }
    let mut direct_store: [ColorGroup<4, i32>; BUFFER_SIZE] =
        [ColorGroup::<4, i32>::dup(ROUNDING_CONST); BUFFER_SIZE];

    let v_start_px = x;

    let py = bounds.start;
    let weight = filter[0] as i32;
    let offset = src_stride * py + v_start_px * 4;
    let src_ptr = &src[offset..(offset + BUFFER_SIZE * 4)];

    for (dst, src) in direct_store.iter_mut().zip(src_ptr.chunks_exact(4)) {
        *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
    }

    for (j, &k_weight) in filter.iter().take(bounds.size).skip(1).enumerate() {
        // Adding 1 is necessary because skip do not incrementing value on values that skipped
        let py = bounds.start + j + 1;
        let weight = k_weight as i32;
        let offset = src_stride * py + v_start_px * 4;
        let src_ptr = &src[offset..(offset + BUFFER_SIZE * 4)];

        for (dst, src) in direct_store.iter_mut().zip(src_ptr.chunks_exact(4)) {
            *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
        }
    }

    let v_dst = &mut dst[v_start_px * 4..(v_start_px * 4 + BUFFER_SIZE * 4)];
    for (dst, src) in v_dst.chunks_exact_mut(4).zip(direct_store) {
        let saturated = src
            .saturate_ar30()
            .to_ar30::<AR30_TYPE, AR30_ORDER>()
            .to_ne_bytes();
        dst[0] = saturated[0];
        dst[1] = saturated[1];
        dst[2] = saturated[2];
        dst[3] = saturated[3];
    }
}

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
fn convolve_column_handler_fixed_point_direct_buffer_double<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
    const BUFFER_SIZE: usize,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    filter: &[i16],
    bounds: &FilterBounds,
    x: usize,
) {
    if filter.is_empty() {
        return;
    }
    let mut direct_store0: [ColorGroup<4, i32>; BUFFER_SIZE] =
        [ColorGroup::<4, i32>::dup(ROUNDING_CONST); BUFFER_SIZE];
    let mut direct_store1: [ColorGroup<4, i32>; BUFFER_SIZE] =
        [ColorGroup::<4, i32>::dup(ROUNDING_CONST); BUFFER_SIZE];

    let v_start_px = x;

    let py = bounds.start;
    let weight = filter[0] as i32;
    let offset = src_stride * py + v_start_px * 4;
    let src_ptr0 = &src[offset..(offset + BUFFER_SIZE * 4)];
    let src_ptr1 = &src[(offset + BUFFER_SIZE * 4)..(offset + BUFFER_SIZE * 2 * 4)];

    for (dst, src) in direct_store0.iter_mut().zip(src_ptr0.chunks_exact(4)) {
        *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
    }

    for (dst, src) in direct_store1.iter_mut().zip(src_ptr1.chunks_exact(4)) {
        *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
    }

    for (j, &k_weight) in filter.iter().take(bounds.size).skip(1).enumerate() {
        // Adding 1 is necessary because skip do not incrementing value on values that skipped
        let py = bounds.start + j + 1;
        let weight = k_weight as i32;
        let offset = src_stride * py + v_start_px * 4;
        let src_ptr0 = &src[offset..(offset + BUFFER_SIZE * 4)];
        let src_ptr1 = &src[(offset + BUFFER_SIZE * 4)..(offset + BUFFER_SIZE * 2 * 4)];

        for (dst, src) in direct_store0.iter_mut().zip(src_ptr0.chunks_exact(4)) {
            *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
        }
        for (dst, src) in direct_store1.iter_mut().zip(src_ptr1.chunks_exact(4)) {
            *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
        }
    }

    let v_dst0 = &mut dst[v_start_px * 4..(v_start_px * 4 + BUFFER_SIZE * 4)];
    for (dst, src) in v_dst0.chunks_exact_mut(4).zip(direct_store0) {
        let saturated = src
            .saturate_ar30()
            .to_ar30::<AR30_TYPE, AR30_ORDER>()
            .to_ne_bytes();
        dst[0] = saturated[0];
        dst[1] = saturated[1];
        dst[2] = saturated[2];
        dst[3] = saturated[3];
    }

    let v_dst1 =
        &mut dst[(v_start_px * 4 + BUFFER_SIZE * 4)..(v_start_px * 4 + BUFFER_SIZE * 2 * 4)];
    for (dst, src) in v_dst1.chunks_exact_mut(4).zip(direct_store1) {
        let saturated = src
            .saturate_ar30()
            .to_ar30::<AR30_TYPE, AR30_ORDER>()
            .to_ne_bytes();
        dst[0] = saturated[0];
        dst[1] = saturated[1];
        dst[2] = saturated[2];
        dst[3] = saturated[3];
    }
}

#[inline(always)]
/// # Generics
/// `T` - template buffer type
/// `J` - accumulator type
fn convolve_column_handler_fixed_point_direct_buffer_four<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
    const BUFFER_SIZE: usize,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    filter: &[i16],
    bounds: &FilterBounds,
    x: usize,
) {
    if filter.is_empty() {
        return;
    }
    let mut direct_store0: [ColorGroup<4, i32>; BUFFER_SIZE] =
        [ColorGroup::<4, i32>::dup(ROUNDING_CONST); BUFFER_SIZE];
    let mut direct_store1: [ColorGroup<4, i32>; BUFFER_SIZE] =
        [ColorGroup::<4, i32>::dup(ROUNDING_CONST); BUFFER_SIZE];
    let mut direct_store2: [ColorGroup<4, i32>; BUFFER_SIZE] =
        [ColorGroup::<4, i32>::dup(ROUNDING_CONST); BUFFER_SIZE];
    let mut direct_store3: [ColorGroup<4, i32>; BUFFER_SIZE] =
        [ColorGroup::<4, i32>::dup(ROUNDING_CONST); BUFFER_SIZE];

    let v_start_px = x * 4;

    let py = bounds.start;
    let weight = filter[0] as i32;
    let offset = src_stride * py + v_start_px;
    let src_ptr0 = &src[offset..(offset + BUFFER_SIZE * 4)];
    let src_ptr1 = &src[(offset + BUFFER_SIZE * 4)..(offset + BUFFER_SIZE * 2 * 4)];
    let src_ptr2 = &src[(offset + BUFFER_SIZE * 2 * 4)..(offset + BUFFER_SIZE * 3 * 4)];
    let src_ptr3 = &src[(offset + BUFFER_SIZE * 3 * 4)..(offset + BUFFER_SIZE * 4 * 4)];

    for (dst, src) in direct_store0.iter_mut().zip(src_ptr0.chunks_exact(4)) {
        *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
    }

    for (dst, src) in direct_store1.iter_mut().zip(src_ptr1.chunks_exact(4)) {
        *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
    }

    for (dst, src) in direct_store2.iter_mut().zip(src_ptr2.chunks_exact(4)) {
        *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
    }

    for (dst, src) in direct_store3.iter_mut().zip(src_ptr3.chunks_exact(4)) {
        *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
    }

    for (j, &k_weight) in filter.iter().take(bounds.size).skip(1).enumerate() {
        // Adding 1 is necessary because skip do not incrementing value on values that skipped
        let py = bounds.start + j + 1;
        let weight = k_weight as i32;
        let offset = src_stride * py + v_start_px;
        let src_ptr0 = &src[offset..(offset + BUFFER_SIZE * 4)];
        let src_ptr1 = &src[(offset + BUFFER_SIZE * 4)..(offset + BUFFER_SIZE * 2 * 4)];
        let src_ptr2 = &src[(offset + BUFFER_SIZE * 2 * 4)..(offset + BUFFER_SIZE * 3 * 4)];
        let src_ptr3 = &src[(offset + BUFFER_SIZE * 3 * 4)..(offset + BUFFER_SIZE * 4 * 4)];

        for (dst, src) in direct_store0.iter_mut().zip(src_ptr0.chunks_exact(4)) {
            *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
        }
        for (dst, src) in direct_store1.iter_mut().zip(src_ptr1.chunks_exact(4)) {
            *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
        }
        for (dst, src) in direct_store2.iter_mut().zip(src_ptr2.chunks_exact(4)) {
            *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
        }
        for (dst, src) in direct_store3.iter_mut().zip(src_ptr3.chunks_exact(4)) {
            *dst += load_ar30_p!(src, AR30_TYPE, AR30_ORDER) * weight;
        }
    }

    let v_dst0 = &mut dst[v_start_px..(v_start_px + BUFFER_SIZE * 4)];
    for (dst, src) in v_dst0.chunks_exact_mut(4).zip(direct_store0) {
        let saturated = src
            .saturate_ar30()
            .to_ar30::<AR30_TYPE, AR30_ORDER>()
            .to_ne_bytes();
        dst[0] = saturated[0];
        dst[1] = saturated[1];
        dst[2] = saturated[2];
        dst[3] = saturated[3];
    }

    let v_dst1 = &mut dst[(v_start_px + BUFFER_SIZE * 4)..(v_start_px + BUFFER_SIZE * 2 * 4)];
    for (dst, src) in v_dst1.chunks_exact_mut(4).zip(direct_store1) {
        let saturated = src
            .saturate_ar30()
            .to_ar30::<AR30_TYPE, AR30_ORDER>()
            .to_ne_bytes();
        dst[0] = saturated[0];
        dst[1] = saturated[1];
        dst[2] = saturated[2];
        dst[3] = saturated[3];
    }

    let v_dst2 = &mut dst[(v_start_px + BUFFER_SIZE * 2 * 4)..(v_start_px + BUFFER_SIZE * 3 * 4)];
    for (dst, src) in v_dst2.chunks_exact_mut(4).zip(direct_store2) {
        let saturated = src
            .saturate_ar30()
            .to_ar30::<AR30_TYPE, AR30_ORDER>()
            .to_ne_bytes();
        dst[0] = saturated[0];
        dst[1] = saturated[1];
        dst[2] = saturated[2];
        dst[3] = saturated[3];
    }

    let v_dst3 = &mut dst[(v_start_px + BUFFER_SIZE * 3 * 4)..(v_start_px + BUFFER_SIZE * 4 * 4)];
    for (dst, src) in v_dst3.chunks_exact_mut(4).zip(direct_store3) {
        let saturated = src
            .saturate_ar30()
            .to_ar30::<AR30_TYPE, AR30_ORDER>()
            .to_ne_bytes();
        dst[0] = saturated[0];
        dst[1] = saturated[1];
        dst[2] = saturated[2];
        dst[3] = saturated[3];
    }
}

pub(crate) fn column_handler_fixed_point_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    let mut cx = 0usize;

    let total_width = dst.len() / 4;

    while cx + 64 < total_width {
        convolve_column_handler_fixed_point_direct_buffer_four::<AR30_TYPE, AR30_ORDER, 16>(
            src, src_stride, dst, weight, bounds, cx,
        );

        cx += 64;
    }

    while cx + 32 < total_width {
        convolve_column_handler_fixed_point_direct_buffer_double::<AR30_TYPE, AR30_ORDER, 16>(
            src, src_stride, dst, weight, bounds, cx,
        );

        cx += 32;
    }

    while cx + 16 < total_width {
        convolve_column_handler_fip_db_ar30::<AR30_TYPE, AR30_ORDER, 16>(
            src, src_stride, dst, weight, bounds, cx,
        );

        cx += 16;
    }

    while cx + 8 < total_width {
        convolve_column_handler_fip_db_ar30::<AR30_TYPE, AR30_ORDER, 8>(
            src, src_stride, dst, weight, bounds, cx,
        );

        cx += 8;
    }

    while cx < total_width {
        convolve_column_handler_fip_db_ar30::<AR30_TYPE, AR30_ORDER, 1>(
            src, src_stride, dst, weight, bounds, cx,
        );

        cx += 1;
    }
}
