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
use std::arch::aarch64::*;

/// Checking NEON `i8mm` availability is required before a call.
///
/// `i8mm` feature has slightly lower precision and won't work really well on huge kernel which
/// edges fades out fast. Therefore, it would be reasonable to avoid using feature for huge downscaling.
///
/// # Safety
/// - Check `i8mm` availability before the call.
pub(crate) fn convolve_horizontal_rgba_neon_row_dot(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgba_neon_row_dot_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve_horizontal_rgba_neon_row_dot_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    const ROUNDING: i16 = 1 << (7 - 1);
    const CHANNELS: usize = 4;

    for ((dst, bounds), weights) in dst
        .chunks_exact_mut(CHANNELS)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let bounds_size = bounds.size;
        let mut jx = 0usize;
        let mut store = vdupq_n_s32(ROUNDING as i32);

        let q_shuf_weights: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let q_shuf_4: [u8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
        let q_shuf_2: [u8; 16] = [
            0, 4, 255, 255, 1, 5, 255, 255, 2, 6, 255, 255, 3, 7, 255, 255,
        ];
        let q_shuf_1: [u8; 16] = [
            0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 3, 255, 255, 255,
        ];
        let shuffle_weights_table = vld1q_u8(q_shuf_weights.as_ptr());
        let shuffle_4_table = vld1q_u8(q_shuf_4.as_ptr());
        let shuffle_2_table = vld1q_u8(q_shuf_2.as_ptr());
        let shuffle_1_table = vld1q_u8(q_shuf_1.as_ptr());

        while jx + 4 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let mut weights = vreinterpretq_u8_u32(vld1q_dup_u32(w_ptr.as_ptr() as *const _));
            weights = vqtbl1q_u8(weights, shuffle_weights_table);
            let bounds_start = bounds.start + jx;

            const COMPONENTS: usize = 4;
            let src_ptr = src.get_unchecked((bounds_start * COMPONENTS)..);

            let rgba_pixel = vqtbl1q_u8(vld1q_u8(src_ptr.as_ptr()), shuffle_4_table);
            store = vusdotq_s32(store, rgba_pixel, vreinterpretq_s8_u8(weights));

            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let weights = vqtbl1q_s8(
                vreinterpretq_s8_s16(vld1q_dup_s16(w_ptr.as_ptr() as *const _)),
                shuffle_weights_table,
            );

            const COMPONENTS: usize = 4;
            let src_ptr = src.get_unchecked((bounds_start * COMPONENTS)..);

            let rgba0 = vld1_u32(src_ptr.as_ptr() as *const _);
            let rgbaj = vreinterpretq_u8_u32(vcombine_u32(rgba0, rgba0));
            let rgba = vqtbl1q_u8(rgbaj, shuffle_2_table);
            store = vusdotq_s32(store, rgba, weights);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weights = vqtbl1q_s8(vld1q_dup_s8(w_ptr.as_ptr()), shuffle_weights_table);
            let bounds_start = bounds.start + jx;
            const COMPONENTS: usize = 4;
            let src_ptr = src.get_unchecked((bounds_start * COMPONENTS)..);

            let rgba = vqtbl1q_u8(
                vreinterpretq_u8_u32(vld1q_dup_u32(src_ptr.as_ptr() as *const _)),
                shuffle_1_table,
            );
            store = vusdotq_s32(store, rgba, weights);

            jx += 1;
        }

        let store_16 = vqshrun_n_s32::<7>(store);
        let store_16_8 = vqmovn_u16(vcombine_u16(store_16, store_16));

        vst1_lane_u32::<0>(
            dst.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8),
        );
    }
}

/// Checking NEON `i8mm` availability is required before a call.
///
/// RDM feature has slightly lower precision and won't work really well on huge kernel which
/// edges fades out fast. Therefore, it would be reasonable to avoid using feature for huge downscaling.
///
/// # Safety
/// - Check `i8mm` availability before the call.
pub(crate) fn convolve_horizontal_rgba_neon_rows_4_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        convolve_horizontal_rgba_neon_rows_4_dot_impl(
            src,
            src_stride,
            dst,
            dst_stride,
            filter_weights,
        );
    }
}

/// Slightly lower precision scale option
///
/// # Safety
/// - Check `i8mm` availability before the call.
#[target_feature(enable = "i8mm")]
unsafe fn convolve_horizontal_rgba_neon_rows_4_dot_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    const CHANNELS: usize = 4;
    const SCALE: i32 = 7;
    const ROUNDING: i16 = 1 << (SCALE - 1);
    let init = vdupq_n_s32(ROUNDING as i32);

    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
    let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
    let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
    let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

    let q_shuf_weights: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    let q_shuf_4: [u8; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];
    let q_shuf_2: [u8; 16] = [
        0, 4, 255, 255, 1, 5, 255, 255, 2, 6, 255, 255, 3, 7, 255, 255,
    ];
    let q_shuf_1: [u8; 16] = [
        0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 3, 255, 255, 255,
    ];
    let shuffle_weights_table = vld1q_u8(q_shuf_weights.as_ptr());
    let shuffle_4_table = vld1q_u8(q_shuf_4.as_ptr());
    let shuffle_2_table = vld1q_u8(q_shuf_2.as_ptr());
    let shuffle_1_table = vld1q_u8(q_shuf_1.as_ptr());

    for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
        .zip(iter_row1)
        .zip(iter_row2)
        .zip(iter_row3)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;

        let bounds_size = bounds.size;

        let mut store_0 = init;
        let mut store_1 = init;
        let mut store_2 = init;
        let mut store_3 = init;

        let src0 = src;
        let src1 = src0.get_unchecked(src_stride..);
        let src2 = src1.get_unchecked(src_stride..);
        let src3 = src2.get_unchecked(src_stride..);

        while jx + 4 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));

            let mut weights = vreinterpretq_u8_u32(vld1q_dup_u32(w_ptr.as_ptr() as *const _));
            weights = vqtbl1q_u8(weights, shuffle_weights_table);
            let bounds_start = bounds.start + jx;

            let src_ptr0 = src0.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr1 = src1.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr2 = src2.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr3 = src3.get_unchecked((bounds_start * CHANNELS)..);

            let rgba_pixel0 = vqtbl1q_u8(vld1q_u8(src_ptr0.as_ptr()), shuffle_4_table);
            store_0 = vusdotq_s32(store_0, rgba_pixel0, vreinterpretq_s8_u8(weights));
            let rgba_pixel1 = vqtbl1q_u8(vld1q_u8(src_ptr1.as_ptr()), shuffle_4_table);
            store_1 = vusdotq_s32(store_1, rgba_pixel1, vreinterpretq_s8_u8(weights));
            let rgba_pixel2 = vqtbl1q_u8(vld1q_u8(src_ptr2.as_ptr()), shuffle_4_table);
            store_2 = vusdotq_s32(store_2, rgba_pixel2, vreinterpretq_s8_u8(weights));
            let rgba_pixel3 = vqtbl1q_u8(vld1q_u8(src_ptr3.as_ptr()), shuffle_4_table);
            store_3 = vusdotq_s32(store_3, rgba_pixel3, vreinterpretq_s8_u8(weights));

            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let weights = vqtbl1q_s8(
                vreinterpretq_s8_s16(vld1q_dup_s16(w_ptr.as_ptr() as *const _)),
                shuffle_weights_table,
            );

            let src_ptr0 = src0.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr1 = src1.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr2 = src2.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr3 = src3.get_unchecked((bounds_start * CHANNELS)..);

            let rgba0 = vld1_u8(src_ptr0.as_ptr());
            let rgba_pixel0 = vqtbl1q_u8(vcombine_u8(rgba0, rgba0), shuffle_2_table);
            store_0 = vusdotq_s32(store_0, rgba_pixel0, weights);

            let rgba1 = vld1_u8(src_ptr1.as_ptr());
            let rgba_pixel1 = vqtbl1q_u8(vcombine_u8(rgba1, rgba1), shuffle_2_table);
            store_1 = vusdotq_s32(store_1, rgba_pixel1, weights);

            let rgba2 = vld1_u8(src_ptr2.as_ptr());
            let rgba_pixel2 = vqtbl1q_u8(vcombine_u8(rgba2, rgba2), shuffle_2_table);
            store_2 = vusdotq_s32(store_2, rgba_pixel2, weights);

            let rgba3 = vld1_u8(src_ptr3.as_ptr());
            let rgba_pixel3 = vqtbl1q_u8(vcombine_u8(rgba3, rgba3), shuffle_2_table);
            store_3 = vusdotq_s32(store_3, rgba_pixel3, weights);

            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));

            let weights = vqtbl1q_s8(vld1q_dup_s8(w_ptr.as_ptr()), shuffle_weights_table);
            let bounds_start = bounds.start + jx;

            let src_ptr0 = src0.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr1 = src1.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr2 = src2.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr3 = src3.get_unchecked((bounds_start * CHANNELS)..);

            let rgba0 = vqtbl1q_u8(
                vreinterpretq_u8_u32(vld1q_dup_u32(src_ptr0.as_ptr() as *const _)),
                shuffle_1_table,
            );
            store_0 = vusdotq_s32(store_0, rgba0, weights);

            let rgba1 = vqtbl1q_u8(
                vreinterpretq_u8_u32(vld1q_dup_u32(src_ptr1.as_ptr() as *const _)),
                shuffle_1_table,
            );
            store_1 = vusdotq_s32(store_1, rgba1, weights);

            let rgba2 = vqtbl1q_u8(
                vreinterpretq_u8_u32(vld1q_dup_u32(src_ptr2.as_ptr() as *const _)),
                shuffle_1_table,
            );
            store_2 = vusdotq_s32(store_2, rgba2, weights);

            let rgba3 = vqtbl1q_u8(
                vreinterpretq_u8_u32(vld1q_dup_u32(src_ptr3.as_ptr() as *const _)),
                shuffle_1_table,
            );
            store_3 = vusdotq_s32(store_3, rgba3, weights);

            jx += 1;
        }

        let store_16_0 = vqshrun_n_s32::<SCALE>(store_0);
        let store_16_1 = vqshrun_n_s32::<SCALE>(store_1);
        let store_16_2 = vqshrun_n_s32::<SCALE>(store_2);
        let store_16_3 = vqshrun_n_s32::<SCALE>(store_3);

        let store_16_8_0 = vqmovn_u16(vcombine_u16(store_16_0, store_16_0));
        let store_16_8_1 = vqmovn_u16(vcombine_u16(store_16_1, store_16_1));
        let store_16_8_2 = vqmovn_u16(vcombine_u16(store_16_2, store_16_2));
        let store_16_8 = vqmovn_u16(vcombine_u16(store_16_3, store_16_3));

        vst1_lane_u32::<0>(
            chunk0.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8_0),
        );
        vst1_lane_u32::<0>(
            chunk1.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8_1),
        );
        vst1_lane_u32::<0>(
            chunk2.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8_2),
        );
        vst1_lane_u32::<0>(
            chunk3.as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(store_16_8),
        );
    }
}
