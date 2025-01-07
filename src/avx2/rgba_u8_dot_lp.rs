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
use crate::avx2::utils::{_mm256_reduce_dot_epi16, _mm256_udot8_epi16, _mm_udot8_epi16};
use crate::filter_weights::FilterWeights;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Will use `avxvnni` if available, if not `maddubs16`
///
/// `avxvnni` feature has slightly lower precision and won't work really well on huge kernel which
/// edges fades out fast. Therefore, it would be reasonable to avoid using feature for huge downscaling.
///
/// # Safety
/// - Check `avx2` availability before the call.
pub(crate) fn convolve_horizontal_rgba_row_dot(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return convolve_horizontal_rgba_vnni_row_dot_impl(src, dst, filter_weights);
        }
        convolve_horizontal_rgba_ubs_row_dot_impl(src, dst, filter_weights);
    }
}

#[cfg(feature = "nightly_avx512")]
#[target_feature(enable = "avxvnni", enable = "avx2")]
unsafe fn convolve_horizontal_rgba_vnni_row_dot_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    convolve_horizontal_rgba_row_dot_impl::<true>(src, dst, filter_weights);
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_rgba_ubs_row_dot_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    convolve_horizontal_rgba_row_dot_impl::<false>(src, dst, filter_weights);
}

#[inline(always)]
fn compress_i32<const DOT: bool>(x: __m128i) -> __m128i {
    unsafe {
        if DOT {
            let store_32 = _mm_srai_epi32::<7>(x);
            _mm_packus_epi32(store_32, store_32)
        } else {
            _mm_srai_epi16::<7>(_mm_hadds_epi16(x, x))
        }
    }
}

#[inline(always)]
unsafe fn convolve_horizontal_rgba_row_dot_impl<const DOT: bool>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    const ROUNDING: i16 = 1 << (7 - 1);
    const CHANNELS: usize = 4;

    let shuffle_weights_table = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);
    let shuffle_4_table = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let shuffle_2_table = _mm_setr_epi8(0, 4, -1, -1, 1, 5, -1, -1, 2, 6, -1, -1, 3, 7, -1, -1);
    let shuffle_1_table = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);

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
        let mut store = if DOT {
            _mm_set1_epi32(ROUNDING as i32)
        } else {
            _mm_setr_epi16(ROUNDING, 0, ROUNDING, 0, ROUNDING, 0, ROUNDING, 0)
        };

        if bounds_size > 8 {
            let shuffle_avx_weights = _mm256_setr_epi8(
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3,
            );
            let permute_avx_weights = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
            let shuffle_8_table = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2,
                6, 10, 14, 3, 7, 11, 15,
            );

            let mut store_avx = if DOT {
                _mm256_setr_epi32(
                    ROUNDING as i32,
                    ROUNDING as i32,
                    ROUNDING as i32,
                    ROUNDING as i32,
                    0,
                    0,
                    0,
                    0,
                )
            } else {
                _mm256_setr_epi16(
                    ROUNDING, 0, ROUNDING, 0, ROUNDING, 0, ROUNDING, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                )
            };

            while jx + 8 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 8));
                let mut weights = _mm256_permutevar8x32_epi32(
                    _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const _)),
                    permute_avx_weights,
                );

                weights = _mm256_shuffle_epi8(weights, shuffle_avx_weights);
                let bounds_start = bounds.start + jx;

                const COMPONENTS: usize = 4;
                let src_ptr = src.get_unchecked((bounds_start * COMPONENTS)..);

                let rgba_pixel = _mm256_shuffle_epi8(
                    _mm256_loadu_si256(src_ptr.as_ptr() as *const _),
                    shuffle_8_table,
                );
                store_avx = _mm256_udot8_epi16::<DOT>(store_avx, rgba_pixel, weights);

                jx += 8;
            }

            store = _mm256_reduce_dot_epi16::<DOT>(store_avx);
        }

        while jx + 4 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let mut weights = _mm_loadu_si32(w_ptr.as_ptr() as *const _);
            weights = _mm_shuffle_epi8(weights, shuffle_weights_table);
            let bounds_start = bounds.start + jx;

            const COMPONENTS: usize = 4;
            let src_ptr = src.get_unchecked((bounds_start * COMPONENTS)..);

            let rgba_pixel = _mm_shuffle_epi8(
                _mm_loadu_si128(src_ptr.as_ptr() as *const _),
                shuffle_4_table,
            );
            store = _mm_udot8_epi16::<DOT>(store, rgba_pixel, weights);

            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let weights = _mm_shuffle_epi8(
                _mm_loadu_si16(w_ptr.as_ptr() as *const _),
                shuffle_weights_table,
            );

            const COMPONENTS: usize = 4;
            let src_ptr = src.get_unchecked((bounds_start * COMPONENTS)..);

            let rgba0 = _mm_loadu_si64(src_ptr.as_ptr() as *const _);
            let rgba = _mm_shuffle_epi8(rgba0, shuffle_2_table);
            store = _mm_udot8_epi16::<DOT>(store, rgba, weights);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let weights = _mm_shuffle_epi8(_mm_set1_epi8(w_ptr[0]), shuffle_weights_table);
            let bounds_start = bounds.start + jx;
            const COMPONENTS: usize = 4;
            let src_ptr = src.get_unchecked((bounds_start * COMPONENTS)..);

            let rgba = _mm_shuffle_epi8(
                _mm_loadu_si32(src_ptr.as_ptr() as *const _),
                shuffle_1_table,
            );
            store = _mm_udot8_epi16::<DOT>(store, rgba, weights);

            jx += 1;
        }

        let store_16_8 = compress_i32::<DOT>(store);

        _mm_storeu_si32(
            dst.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8, store_16_8),
        );
    }
}

/// Will use `avxvnni` if available, if not `maddubs16`
///
/// `avxvnni` feature has slightly lower precision and won't work really well on huge kernel which
/// edges fades out fast. Therefore, it would be reasonable to avoid using feature for huge downscaling.
///
/// # Safety
/// - Check `avx2` availability before the call.
pub(crate) fn convolve_horizontal_rgba_rows_4_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return convolve_horizontal_rgba_vnni_rows_4_dot(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
            );
        }
        convolve_horizontal_rgba_vnni_rows_4_ubs(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[cfg(feature = "nightly_avx512")]
#[target_feature(enable = "avxvnni", enable = "avx2")]
unsafe fn convolve_horizontal_rgba_vnni_rows_4_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    convolve_horizontal_rgba_vnni_rows_4_dot_impl::<true>(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
    );
}

#[target_feature(enable = "avx2")]
unsafe fn convolve_horizontal_rgba_vnni_rows_4_ubs(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    convolve_horizontal_rgba_vnni_rows_4_dot_impl::<false>(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
    );
}

#[inline(always)]
unsafe fn convolve_horizontal_rgba_vnni_rows_4_dot_impl<const DOT: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    const CHANNELS: usize = 4;
    const SCALE: i32 = 7;
    const ROUNDING: i16 = 1 << (SCALE - 1);

    let init = if DOT {
        _mm_set1_epi32(ROUNDING as i32)
    } else {
        _mm_setr_epi16(ROUNDING, 0, ROUNDING, 0, ROUNDING, 0, ROUNDING, 0)
    };

    let init_avx = if DOT {
        _mm256_setr_epi32(
            ROUNDING as i32,
            ROUNDING as i32,
            ROUNDING as i32,
            ROUNDING as i32,
            0,
            0,
            0,
            0,
        )
    } else {
        _mm256_setr_epi16(
            ROUNDING, 0, ROUNDING, 0, ROUNDING, 0, ROUNDING, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        )
    };

    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
    let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
    let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
    let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

    let shuffle_weights_table = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);
    let shuffle_4_table = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let shuffle_2_table = _mm_setr_epi8(0, 4, -1, -1, 1, 5, -1, -1, 2, 6, -1, -1, 3, 7, -1, -1);
    let shuffle_1_table = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);

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

        if bounds_size > 8 {
            let shuffle_avx_weights = _mm256_setr_epi8(
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3,
            );
            let permute_avx_weights = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
            let shuffle_8_table = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2,
                6, 10, 14, 3, 7, 11, 15,
            );

            let mut store_avx0 = init_avx;
            let mut store_avx1 = init_avx;
            let mut store_avx2 = init_avx;
            let mut store_avx3 = init_avx;

            while jx + 8 < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 8));
                let mut weights = _mm256_permutevar8x32_epi32(
                    _mm256_castsi128_si256(_mm_loadu_si64(w_ptr.as_ptr() as *const _)),
                    permute_avx_weights,
                );

                weights = _mm256_shuffle_epi8(weights, shuffle_avx_weights);
                let bounds_start = bounds.start + jx;

                let src_ptr0 = src0.get_unchecked((bounds_start * CHANNELS)..);
                let src_ptr1 = src1.get_unchecked((bounds_start * CHANNELS)..);
                let src_ptr2 = src2.get_unchecked((bounds_start * CHANNELS)..);
                let src_ptr3 = src3.get_unchecked((bounds_start * CHANNELS)..);

                let rgba_pixel0 = _mm256_shuffle_epi8(
                    _mm256_loadu_si256(src_ptr0.as_ptr() as *const _),
                    shuffle_8_table,
                );
                let rgba_pixel1 = _mm256_shuffle_epi8(
                    _mm256_loadu_si256(src_ptr1.as_ptr() as *const _),
                    shuffle_8_table,
                );
                let rgba_pixel2 = _mm256_shuffle_epi8(
                    _mm256_loadu_si256(src_ptr2.as_ptr() as *const _),
                    shuffle_8_table,
                );
                let rgba_pixel3 = _mm256_shuffle_epi8(
                    _mm256_loadu_si256(src_ptr3.as_ptr() as *const _),
                    shuffle_8_table,
                );
                store_avx0 = _mm256_udot8_epi16::<DOT>(store_avx0, rgba_pixel0, weights);
                store_avx1 = _mm256_udot8_epi16::<DOT>(store_avx1, rgba_pixel1, weights);
                store_avx2 = _mm256_udot8_epi16::<DOT>(store_avx2, rgba_pixel2, weights);
                store_avx3 = _mm256_udot8_epi16::<DOT>(store_avx3, rgba_pixel3, weights);

                jx += 8;
            }

            store_0 = _mm256_reduce_dot_epi16::<DOT>(store_avx0);
            store_1 = _mm256_reduce_dot_epi16::<DOT>(store_avx1);
            store_2 = _mm256_reduce_dot_epi16::<DOT>(store_avx2);
            store_3 = _mm256_reduce_dot_epi16::<DOT>(store_avx3);
        }

        while jx + 4 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));

            let mut weights = _mm_loadu_si32(w_ptr.as_ptr() as *const _);
            weights = _mm_shuffle_epi8(weights, shuffle_weights_table);
            let bounds_start = bounds.start + jx;

            let src_ptr0 = src0.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr1 = src1.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr2 = src2.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr3 = src3.get_unchecked((bounds_start * CHANNELS)..);

            let rgba_pixel0 = _mm_shuffle_epi8(
                _mm_loadu_si128(src_ptr0.as_ptr() as *const _),
                shuffle_4_table,
            );
            store_0 = _mm_udot8_epi16::<DOT>(store_0, rgba_pixel0, weights);
            let rgba_pixel1 = _mm_shuffle_epi8(
                _mm_loadu_si128(src_ptr1.as_ptr() as *const _),
                shuffle_4_table,
            );
            store_1 = _mm_udot8_epi16::<DOT>(store_1, rgba_pixel1, weights);
            let rgba_pixel2 = _mm_shuffle_epi8(
                _mm_loadu_si128(src_ptr2.as_ptr() as *const _),
                shuffle_4_table,
            );
            store_2 = _mm_udot8_epi16::<DOT>(store_2, rgba_pixel2, weights);
            let rgba_pixel3 = _mm_shuffle_epi8(
                _mm_loadu_si128(src_ptr3.as_ptr() as *const _),
                shuffle_4_table,
            );
            store_3 = _mm_udot8_epi16::<DOT>(store_3, rgba_pixel3, weights);

            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let weights = _mm_shuffle_epi8(
                _mm_loadu_si16(w_ptr.as_ptr() as *const _),
                shuffle_weights_table,
            );

            let src_ptr0 = src0.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr1 = src1.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr2 = src2.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr3 = src3.get_unchecked((bounds_start * CHANNELS)..);

            let rgba0 = _mm_loadu_si64(src_ptr0.as_ptr());
            let rgba_pixel0 = _mm_shuffle_epi8(rgba0, shuffle_2_table);
            store_0 = _mm_udot8_epi16::<DOT>(store_0, rgba_pixel0, weights);

            let rgba1 = _mm_loadu_si64(src_ptr1.as_ptr());
            let rgba_pixel1 = _mm_shuffle_epi8(rgba1, shuffle_2_table);
            store_1 = _mm_udot8_epi16::<DOT>(store_1, rgba_pixel1, weights);

            let rgba2 = _mm_loadu_si64(src_ptr2.as_ptr());
            let rgba_pixel2 = _mm_shuffle_epi8(rgba2, shuffle_2_table);
            store_2 = _mm_udot8_epi16::<DOT>(store_2, rgba_pixel2, weights);

            let rgba3 = _mm_loadu_si64(src_ptr3.as_ptr());
            let rgba_pixel3 = _mm_shuffle_epi8(rgba3, shuffle_2_table);
            store_3 = _mm_udot8_epi16::<DOT>(store_3, rgba_pixel3, weights);

            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));

            let weights = _mm_shuffle_epi8(_mm_set1_epi8(w_ptr[0]), shuffle_weights_table);
            let bounds_start = bounds.start + jx;

            let src_ptr0 = src0.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr1 = src1.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr2 = src2.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr3 = src3.get_unchecked((bounds_start * CHANNELS)..);

            let rgba0 = _mm_shuffle_epi8(
                _mm_loadu_si32(src_ptr0.as_ptr() as *const _),
                shuffle_1_table,
            );
            store_0 = _mm_udot8_epi16::<DOT>(store_0, rgba0, weights);

            let rgba1 = _mm_shuffle_epi8(
                _mm_loadu_si32(src_ptr1.as_ptr() as *const _),
                shuffle_1_table,
            );
            store_1 = _mm_udot8_epi16::<DOT>(store_1, rgba1, weights);

            let rgba2 = _mm_shuffle_epi8(
                _mm_loadu_si32(src_ptr2.as_ptr() as *const _),
                shuffle_1_table,
            );
            store_2 = _mm_udot8_epi16::<DOT>(store_2, rgba2, weights);

            let rgba3 = _mm_shuffle_epi8(
                _mm_loadu_si32(src_ptr3.as_ptr() as *const _),
                shuffle_1_table,
            );
            store_3 = _mm_udot8_epi16::<DOT>(store_3, rgba3, weights);

            jx += 1;
        }

        let store_16_8_0 = compress_i32::<DOT>(store_0);
        let store_16_8_1 = compress_i32::<DOT>(store_1);
        let store_16_8_2 = compress_i32::<DOT>(store_2);
        let store_16_8 = compress_i32::<DOT>(store_3);

        _mm_storeu_si32(
            chunk0.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8_0, store_16_8_0),
        );
        _mm_storeu_si32(
            chunk1.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8_1, store_16_8_1),
        );
        _mm_storeu_si32(
            chunk2.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8_2, store_16_8_2),
        );
        _mm_storeu_si32(
            chunk3.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8, store_16_8),
        );
    }
}
