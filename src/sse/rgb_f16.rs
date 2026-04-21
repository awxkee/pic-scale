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
use core::f16;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
fn convolve_horizontal_parts_4_rgb_f16<const F16C: bool, const FMA: bool>(
    start_x: usize,
    src: &[f16],
    weight0: __m128,
    weight1: __m128,
    weight2: __m128,
    weight3: __m128,
    store_0: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked(start_x * CN..);

        let first_set = _mm_loadu_si128(src_ptr.as_ptr().cast()); // First 8 elements
        let second_set = _mm_loadu_si64(src_ptr.get_unchecked(8..).as_ptr().cast()); // Last 4 elements

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
        _mm_prefer_fma_ps::<FMA>(acc, rgb_pixel_3, weight3)
    }
}

#[inline(always)]
fn convolve_horizontal_parts_2_rgb_f16<const F16C: bool, const FMA: bool>(
    start_x: usize,
    src: &[f16],
    weight0: __m128,
    weight1: __m128,
    store_0: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked(start_x * CN..);

        let orig1 = _mm_cvtph_psx::<F16C>(_mm_loadu_si64(src_ptr.as_ptr().cast()));
        const SHUFFLE_FLAG: i32 = shuffle(2, 1, 0, 0);
        let orig2 = _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_FLAG>(_mm_castps_si128(
            _mm_cvtph_psx::<F16C>(_mm_setr_epi32(
                (src_ptr.get_unchecked(4..).as_ptr() as *const i32).read_unaligned(),
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
}

#[inline(always)]
fn convolve_horizontal_parts_one_rgb_f16<const F16C: bool, const FMA: bool>(
    start_x: usize,
    src: &[f16],
    weight0: __m128,
    store_0: __m128,
) -> __m128 {
    unsafe {
        const CN: usize = 3;
        let src_ptr = src.get_unchecked(start_x * CN..);

        let read_first = (src_ptr.as_ptr() as *const u32)
            .read_unaligned()
            .to_le_bytes();
        let read_last = *src_ptr.get_unchecked(2);

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
        _mm_prefer_fma_ps::<FMA>(store_0, rgb_pixel, weight0)
    }
}

pub(crate) fn convolve_horizontal_rgb_sse_row_one_f16(
    src: &[f16],
    dst: &mut [f16],
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_sse_row_one_f16_regular(filter_weights, src, dst);
    }
}

#[target_feature(enable = "sse4.1")]
/// This inlining is required to activate all features for runtime dispatch
fn convolve_horizontal_rgb_sse_row_one_f16_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    convolve_horizontal_rgb_sse_row_one_f16_impl::<false, false>(filter_weights, src, dst);
}

#[inline(always)]
fn convolve_horizontal_rgb_sse_row_one_f16_impl<const F16C: bool, const FMA: bool>(
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    unsafe {
        const CN: usize = 3;
        for ((dst, bounds), weights) in dst
            .as_chunks_mut::<CN>()
            .0
            .iter_mut()
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut jx = 0usize;
            let mut store = _mm_setzero_ps();

            while jx + 4 <= bounds.size {
                let w_s = weights.get_unchecked(jx..);
                let (weight0, weight1, weight2, weight3) = load_4_weights!(w_s.as_ptr());
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store,
                );
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w_s = weights.get_unchecked(jx..);
                let weights = _mm_castsi128_ps(_mm_loadu_si64(w_s.as_ptr().cast()));
                const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
                let weight0 =
                    _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_0>(_mm_castps_si128(weights)));
                const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
                let weight1 =
                    _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_1>(_mm_castps_si128(weights)));
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src,
                    weight0,
                    weight1,
                    store,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let w_s = weights.get_unchecked(jx..);
                let weight0 = _mm_load1_ps(w_s.as_ptr());
                let filter_start = jx + bounds.start;
                store = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src,
                    weight0,
                    store,
                );
                jx += 1;
            }

            let store_ph = _mm_cvtps_phx::<F16C>(store);

            (dst.as_mut_ptr() as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph));
            (dst[2..].as_mut_ptr() as *mut i16)
                .write_unaligned(_mm_extract_epi16::<2>(store_ph) as i16);
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_sse_rows_4_f16(
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_sse_rows_4_f16_regular(
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "sse4.1")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_horizontal_rgb_sse_rows_4_f16_regular(
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    convolve_horizontal_rgb_sse_rows_4_f16_impl::<false, false>(
        filter_weights,
        src,
        src_stride,
        dst,
        dst_stride,
    );
}

#[inline(always)]
fn convolve_horizontal_rgb_sse_rows_4_f16_impl<const F16C: bool, const FMA: bool>(
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    unsafe {
        const CN: usize = 3;
        let zeros = _mm_setzero_ps();

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
        let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
        let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
        let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

        for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
            .iter_mut()
            .zip(iter_row1.iter_mut())
            .zip(iter_row2.iter_mut())
            .zip(iter_row3.iter_mut())
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let mut jx = 0usize;
            let mut store_0 = zeros;
            let mut store_1 = zeros;
            let mut store_2 = zeros;
            let mut store_3 = zeros;

            while jx + 4 <= bounds.size {
                let w_s = weights.get_unchecked(jx..);
                let (weight0, weight1, weight2, weight3) = load_4_weights!(w_s.as_ptr());
                let filter_start = jx + bounds.start;
                store_0 = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src,
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src.get_unchecked(src_stride..),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_4_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 3..),
                    weight0,
                    weight1,
                    weight2,
                    weight3,
                    store_3,
                );
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w_s = weights.get_unchecked(jx..);
                let weights = _mm_castsi128_ps(_mm_loadu_si64(w_s.as_ptr().cast()));
                const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
                let weight0 =
                    _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_0>(_mm_castps_si128(weights)));
                const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
                let weight1 =
                    _mm_castsi128_ps(_mm_shuffle_epi32::<SHUFFLE_1>(_mm_castps_si128(weights)));
                let filter_start = jx + bounds.start;
                store_0 = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src,
                    weight0,
                    weight1,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src.get_unchecked(src_stride..),
                    weight0,
                    weight1,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..),
                    weight0,
                    weight1,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_2_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 3..),
                    weight0,
                    weight1,
                    store_3,
                );
                jx += 2;
            }

            while jx < bounds.size {
                let w_s = weights.get_unchecked(jx..);
                let weight0 = _mm_load1_ps(w_s.as_ptr());
                let filter_start = jx + bounds.start;
                store_0 = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src,
                    weight0,
                    store_0,
                );
                store_1 = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src.get_unchecked(src_stride..),
                    weight0,
                    store_1,
                );
                store_2 = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 2..),
                    weight0,
                    store_2,
                );
                store_3 = convolve_horizontal_parts_one_rgb_f16::<F16C, FMA>(
                    filter_start,
                    src.get_unchecked(src_stride * 3..),
                    weight0,
                    store_3,
                );
                jx += 1;
            }

            let store_ph_0 = _mm_cvtps_phx::<F16C>(store_0);
            let store_ph_1 = _mm_cvtps_phx::<F16C>(store_1);
            let store_ph_2 = _mm_cvtps_phx::<F16C>(store_2);
            let store_ph_3 = _mm_cvtps_phx::<F16C>(store_3);

            (chunk0.as_mut_ptr() as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph_0));
            (chunk0[2..].as_mut_ptr() as *mut i16)
                .write_unaligned(_mm_extract_epi16::<2>(store_ph_0) as i16);

            (chunk1.as_mut_ptr() as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph_1));
            (chunk1[2..].as_mut_ptr() as *mut i16)
                .write_unaligned(_mm_extract_epi16::<2>(store_ph_1) as i16);

            (chunk2.as_mut_ptr() as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph_2));
            (chunk2[2..].as_mut_ptr() as *mut i16)
                .write_unaligned(_mm_extract_epi16::<2>(store_ph_2) as i16);

            (chunk3.as_mut_ptr() as *mut i32).write_unaligned(_mm_extract_epi32::<0>(store_ph_3));
            (chunk3[2..].as_mut_ptr() as *mut i16)
                .write_unaligned(_mm_extract_epi16::<2>(store_ph_3) as i16);
        }
    }
}
