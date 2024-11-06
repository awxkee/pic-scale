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
use crate::sse::f16_utils::{_mm_cvtph_psx, _mm_cvtps_phx};
use crate::sse::{_mm_prefer_fma_ps, load_4_weights, shuffle};
use half::f16;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn convolve_horizontal_parts_4_rgb_f16<const F16C: bool, const FMA: bool>(
    start_x: usize,
    src: *const f16,
    weight0: __m128,
    weight1: __m128,
    weight2: __m128,
    weight3: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let first_set = _mm_loadu_si128(src_ptr as *const __m128i); // First 8 elements
    let second_set = _mm_loadu_si64(src_ptr.add(8) as *const u8); // Last 4 elements

    let rgbr_pixel_0 = _mm_cvtph_psx::<F16C>(first_set);
    let gbrg_pixel_1 = _mm_cvtph_psx::<F16C>(_mm_srli_si128::<8>(first_set));
    let brgb_pixel_2 = _mm_cvtph_psx::<F16C>(second_set);

    let rgb_pixel_0 = rgbr_pixel_0;
    const RESHUFFLE_FIRST_FLAG: i32 = shuffle(3, 2, 1, 3);
    const MOVE_AWAY_FROM_FIRST: i32 = shuffle(2, 1, 0, 0);
    let rgb_pixel_1 = _mm_move_ss(
        _mm_castsi128_ps(_mm_shuffle_epi32::<MOVE_AWAY_FROM_FIRST>(_mm_castps_si128(
            gbrg_pixel_1,
        ))),
        _mm_castsi128_ps(_mm_shuffle_epi32::<RESHUFFLE_FIRST_FLAG>(_mm_castps_si128(
            rgbr_pixel_0,
        ))),
    );
    let rgb_pixel_2 = _mm_movelh_ps(
        _mm_castsi128_ps(_mm_unpackhi_epi64(
            _mm_castps_si128(gbrg_pixel_1),
            _mm_castps_si128(gbrg_pixel_1),
        )),
        brgb_pixel_2,
    );
    const SKIP_FIRST: i32 = shuffle(3, 3, 2, 1);
    let rgb_pixel_3 = _mm_castsi128_ps(_mm_shuffle_epi32::<SKIP_FIRST>(_mm_castps_si128(
        brgb_pixel_2,
    )));

    let acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
    let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
    let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_2, weight2);
    let acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_3, weight3);
    acc
}

#[inline(always)]
unsafe fn convolve_horizontal_parts_2_rgb_f16<const F16C: bool, const FMA: bool>(
    start_x: usize,
    src: *const f16,
    weight0: __m128,
    weight1: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let orig1 = _mm_cvtph_psx::<F16C>(_mm_loadu_si64(src_ptr as *const u8));
    const SHUFFLE_FLAG: i32 = shuffle(2, 1, 0, 0);
    let orig2 = _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_FLAG>(_mm_castps_si128(
        _mm_cvtph_psx::<F16C>(_mm_setr_epi32(
            (src_ptr.add(4) as *const i32).read_unaligned(),
            0,
            0,
            0,
        )),
    )));
    let rgb_pixel_0 = orig1;
    const RESHUFFLE_FIRST_FLAG: i32 = shuffle(3, 2, 1, 3);
    let shuffled_first = _mm_castsi128_ps(_mm_shuffle_epi32::<RESHUFFLE_FIRST_FLAG>(
        _mm_castps_si128(orig1),
    ));
    let rgb_pixel_1 = _mm_move_ss(orig2, shuffled_first);

    let mut acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel_0, weight0);
    acc = _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_1, weight1);
    acc
}

#[inline(always)]
unsafe fn convolve_horizontal_parts_one_rgb_f16<const F16C: bool, const FMA: bool>(
    start_x: usize,
    src: *const f16,
    weight0: __m128,
    store_0: __m128,
) -> __m128 {
    const COMPONENTS: usize = 3;
    let src_ptr = src.add(start_x * COMPONENTS);

    let read_first = (src_ptr as *const u32).read_unaligned().to_le_bytes();
    let read_last = src_ptr.add(2).read_unaligned();

    let rgb_pixel = _mm_cvtph_psx::<F16C>(_mm_setr_epi16(
        i16::from_le_bytes([read_first[0], read_first[1]]),
        i16::from_le_bytes([read_first[2], read_first[3]]),
        read_last.to_bits() as i16,
        0,
        0,
        0,
        0,
        0,
    ));
    let acc = _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel, weight0);
    acc
}

pub fn convolve_horizontal_rgb_sse_row_one_f16<const F16C: bool, const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    unsafe_destination_ptr_0: *mut f16,
) {
    unsafe {
        if F16C {
            if FMA {
                convolve_horizontal_rgb_sse_row_one_f16c_fma(
                    dst_width,
                    src_width,
                    filter_weights,
                    unsafe_source_ptr_0,
                    unsafe_destination_ptr_0,
                );
            } else {
                convolve_horizontal_rgb_sse_row_one_f16c(
                    dst_width,
                    src_width,
                    filter_weights,
                    unsafe_source_ptr_0,
                    unsafe_destination_ptr_0,
                );
            }
        } else {
            convolve_horizontal_rgb_sse_row_one_f16_regular(
                dst_width,
                src_width,
                filter_weights,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgb_sse_row_one_f16_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    unsafe_destination_ptr_0: *mut f16,
) {
    convolve_horizontal_rgb_sse_row_one_f16_impl::<false, false>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        unsafe_destination_ptr_0,
    );
}

#[inline]
#[target_feature(enable = "sse4.1,f16c")]
unsafe fn convolve_horizontal_rgb_sse_row_one_f16c(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    unsafe_destination_ptr_0: *mut f16,
) {
    convolve_horizontal_rgb_sse_row_one_f16_impl::<true, false>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        unsafe_destination_ptr_0,
    );
}

#[inline]
#[target_feature(enable = "sse4.1,f16c,fma")]
unsafe fn convolve_horizontal_rgb_sse_row_one_f16c_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    unsafe_destination_ptr_0: *mut f16,
) {
    convolve_horizontal_rgb_sse_row_one_f16_impl::<true, true>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        unsafe_destination_ptr_0,
    );
}

#[inline]
unsafe fn convolve_horizontal_rgb_sse_row_one_f16_impl<const F16C: bool, const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    unsafe_destination_ptr_0: *mut f16,
) {
    const CHANNELS: usize = 3;
    let mut filter_offset = 0usize;
    let weights_ptr = filter_weights.weights.as_ptr();

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store = _mm_setzero_ps();

        while jx + 4 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let (weight0, weight1, weight2, weight3) = load_4_weights!(ptr);
            let filter_start = jx + bounds.start;
            store = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                filter_start,
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
            let weights = _mm_castsi128_ps(_mm_loadu_si64(ptr as *const u8));
            const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
            let weight0 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_0>(_mm_castps_si128(weights)));
            const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
            let weight1 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_1>(_mm_castps_si128(weights)));
            let filter_start = jx + bounds.start;
            store = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0,
                weight0,
                weight1,
                store,
            );
            jx += 2;
        }

        while jx < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight0 = _mm_load1_ps(ptr);
            let filter_start = jx + bounds.start;
            store = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0,
                weight0,
                store,
            );
            jx += 1;
        }

        let store_ph = _mm_cvtps_phx::<F16C>(store);

        let px = x * CHANNELS;
        let dest_ptr = unsafe_destination_ptr_0.add(px);
        (dest_ptr as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph));
        (dest_ptr as *mut i16)
            .add(2)
            .write_unaligned(_mm_extract_epi16::<2>(store_ph) as i16);

        filter_offset += filter_weights.aligned_size;
    }
}

pub fn convolve_horizontal_rgb_sse_rows_4_f16<const F16C: bool, const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f16,
    dst_stride: usize,
) {
    unsafe {
        if F16C {
            if FMA {
                convolve_horizontal_rgb_sse_rows_4_f16c_fma(
                    dst_width,
                    src_width,
                    filter_weights,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    dst_stride,
                );
            } else {
                convolve_horizontal_rgb_sse_rows_4_f16c(
                    dst_width,
                    src_width,
                    filter_weights,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    dst_stride,
                );
            }
        } else {
            convolve_horizontal_rgb_sse_rows_4_f16_regular(
                dst_width,
                src_width,
                filter_weights,
                unsafe_source_ptr_0,
                src_stride,
                unsafe_destination_ptr_0,
                dst_stride,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgb_sse_rows_4_f16_regular(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f16,
    dst_stride: usize,
) {
    convolve_horizontal_rgb_sse_rows_4_f16_impl::<false, false>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        src_stride,
        unsafe_destination_ptr_0,
        dst_stride,
    );
}

#[inline]
#[target_feature(enable = "sse4.1,f16c")]
unsafe fn convolve_horizontal_rgb_sse_rows_4_f16c(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f16,
    dst_stride: usize,
) {
    convolve_horizontal_rgb_sse_rows_4_f16_impl::<true, false>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        src_stride,
        unsafe_destination_ptr_0,
        dst_stride,
    );
}

#[inline]
#[target_feature(enable = "sse4.1,f16c,fma")]
unsafe fn convolve_horizontal_rgb_sse_rows_4_f16c_fma(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f16,
    dst_stride: usize,
) {
    convolve_horizontal_rgb_sse_rows_4_f16_impl::<true, true>(
        dst_width,
        src_width,
        filter_weights,
        unsafe_source_ptr_0,
        src_stride,
        unsafe_destination_ptr_0,
        dst_stride,
    );
}

#[inline]
unsafe fn convolve_horizontal_rgb_sse_rows_4_f16_impl<const F16C: bool, const FMA: bool>(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f16,
    dst_stride: usize,
) {
    const CHANNELS: usize = 3;
    let mut filter_offset = 0usize;
    let zeros = _mm_setzero_ps();
    let weights_ptr = filter_weights.weights.as_ptr();

    for x in 0..dst_width {
        let bounds = filter_weights.bounds.get_unchecked(x);
        let mut jx = 0usize;
        let mut store_0 = zeros;
        let mut store_1 = zeros;
        let mut store_2 = zeros;
        let mut store_3 = zeros;

        while jx + 4 < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let (weight0, weight1, weight2, weight3) = load_4_weights!(ptr);
            let filter_start = jx + bounds.start;
            store_0 = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0,
                weight0,
                weight1,
                weight2,
                weight3,
                store_0,
            );
            store_1 = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0.add(src_stride),
                weight0,
                weight1,
                weight2,
                weight3,
                store_1,
            );
            store_2 = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0.add(src_stride * 2),
                weight0,
                weight1,
                weight2,
                weight3,
                store_2,
            );
            store_3 = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                filter_start,
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
            let weights = _mm_castsi128_ps(_mm_loadu_si64(ptr as *const u8));
            const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
            let weight0 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_0>(_mm_castps_si128(weights)));
            const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
            let weight1 =
                _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_1>(_mm_castps_si128(weights)));
            let filter_start = jx + bounds.start;
            store_0 = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0,
                weight0,
                weight1,
                store_0,
            );
            store_1 = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0.add(src_stride),
                weight0,
                weight1,
                store_1,
            );
            store_2 = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0.add(src_stride * 2),
                weight0,
                weight1,
                store_2,
            );
            store_3 = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0.add(src_stride * 3),
                weight0,
                weight1,
                store_3,
            );
            jx += 2;
        }

        while jx < bounds.size {
            let ptr = weights_ptr.add(jx + filter_offset);
            let weight0 = _mm_load1_ps(ptr);
            let filter_start = jx + bounds.start;
            store_0 = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0,
                weight0,
                store_0,
            );
            store_1 = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0.add(src_stride),
                weight0,
                store_1,
            );
            store_2 = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0.add(src_stride * 2),
                weight0,
                store_2,
            );
            store_3 = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                filter_start,
                unsafe_source_ptr_0.add(src_stride * 3),
                weight0,
                store_3,
            );
            jx += 1;
        }

        let store_ph_0 = _mm_cvtps_phx::<F16C>(store_0);
        let store_ph_1 = _mm_cvtps_phx::<F16C>(store_1);
        let store_ph_2 = _mm_cvtps_phx::<F16C>(store_2);
        let store_ph_3 = _mm_cvtps_phx::<F16C>(store_3);

        let px = x * CHANNELS;
        let dest_ptr = unsafe_destination_ptr_0.add(px);
        (dest_ptr as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph_0));
        (dest_ptr as *mut i16)
            .add(2)
            .write_unaligned(_mm_extract_epi16::<2>(store_ph_0) as i16);

        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride);
        (dest_ptr as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph_1));
        (dest_ptr as *mut i16)
            .add(2)
            .write_unaligned(_mm_extract_epi16::<2>(store_ph_1) as i16);

        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 2);
        (dest_ptr as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph_2));
        (dest_ptr as *mut i16)
            .add(2)
            .write_unaligned(_mm_extract_epi16::<2>(store_ph_2) as i16);

        let dest_ptr = unsafe_destination_ptr_0.add(px + dst_stride * 3);
        (dest_ptr as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph_3));
        (dest_ptr as *mut i16)
            .add(2)
            .write_unaligned(_mm_extract_epi16::<2>(store_ph_3) as i16);

        filter_offset += filter_weights.aligned_size;
    }
}
