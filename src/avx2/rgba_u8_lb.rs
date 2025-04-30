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

use crate::avx2::utils::shuffle;
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn convolve_horizontal_parts_one_rgba_sse(
    start_x: usize,
    src: &[u8],
    weight0: __m128i,
    store_0: __m128i,
) -> __m128i {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let src_ptr_32 = src_ptr.as_ptr() as *const i32;
    let rgba_pixel = _mm_cvtsi32_si128(src_ptr_32.read_unaligned());
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(rgba_pixel, rgba_pixel));

    _mm_add_epi16(store_0, _mm_mulhrs_epi16(lo, weight0))
}

pub(crate) fn convolve_horizontal_rgba_avx_rows_4_lb(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_avx_rows_4_impl(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[inline(always)]
unsafe fn hdot4(store: __m256i, v0: __m256i, w01: __m256i, w23: __m256i) -> __m256i {
    let lo0 = _mm256_srli_epi16::<2>(_mm256_unpacklo_epi8(v0, v0));
    let hi0 = _mm256_srli_epi16::<2>(_mm256_unpackhi_epi8(v0, v0));
    let mut p = _mm256_mulhrs_epi16(lo0, w01);
    p = _mm256_add_epi16(p, _mm256_mulhrs_epi16(hi0, w23));
    _mm256_add_epi16(store, p)
}

#[inline(always)]
unsafe fn hdot2(store: __m256i, v: __m256i, w0123: __m256i) -> __m256i {
    let lo = _mm256_srli_epi16::<2>(_mm256_unpacklo_epi8(v, v));
    _mm256_add_epi16(store, _mm256_mulhrs_epi16(lo, w0123))
}

#[inline(always)]
unsafe fn hdot(store: __m128i, v: __m128i, w01: __m128i) -> __m128i {
    let lo = _mm_srli_epi16::<2>(_mm_unpacklo_epi8(v, v));
    _mm_add_epi16(store, _mm_mulhrs_epi16(lo, w01))
}

#[inline(always)]
unsafe fn _mm_add_hi_lo_epi16(v: __m128i) -> __m128i {
    let p = _mm_unpackhi_epi64(v, v);
    _mm_add_epi16(v, p)
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_rgba_avx_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 4;

    const SCALE: i32 = 6;
    const V_SHR: i32 = SCALE;
    const ROUNDING: i16 = 1 << (V_SHR - 1);

    let vld = _mm256_setr_epi16(
        ROUNDING, ROUNDING, ROUNDING, ROUNDING, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );

    let shuffle_weights = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3);

    let distr_weights = _mm256_setr_epi8(
        0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3,
        2, 3,
    );

    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
    let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
    let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
    let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

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

        let src0 = src;
        let src1 = src0.get_unchecked(src_stride..);
        let src2 = src1.get_unchecked(src_stride..);
        let src3 = src2.get_unchecked(src_stride..);

        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;

        while jx + 8 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);

            let w01 = _mm_loadu_si32(w_ptr.as_ptr() as *const _);
            let w23 = _mm_loadu_si32(w_ptr.get_unchecked(2..).as_ptr() as *const _);
            let w45 = _mm_loadu_si32(w_ptr.get_unchecked(4..).as_ptr() as *const _);
            let w67 = _mm_loadu_si32(w_ptr.get_unchecked(6..).as_ptr() as *const _);

            let w0145 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w01), w45),
                distr_weights,
            );
            let w2367 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w23), w67),
                distr_weights,
            );

            let start_bounds = bounds.start + jx;

            let rgb_pixel_0 = _mm256_loadu_si256(
                src0.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m256i,
            );
            let rgb_pixel_1 = _mm256_loadu_si256(
                src1.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m256i,
            );
            let rgb_pixel_2 = _mm256_loadu_si256(
                src2.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m256i,
            );
            let rgb_pixel_3 = _mm256_loadu_si256(
                src3.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m256i,
            );

            store_0 = hdot4(store_0, rgb_pixel_0, w0145, w2367);
            store_1 = hdot4(store_1, rgb_pixel_1, w0145, w2367);
            store_2 = hdot4(store_2, rgb_pixel_2, w0145, w2367);
            store_3 = hdot4(store_3, rgb_pixel_3, w0145, w2367);

            jx += 8;
        }

        let mut store_0 = _mm256_add_epi16(
            _mm256_permute2x128_si256::<0x20>(store_0, store_1),
            _mm256_permute2x128_si256::<0x31>(store_0, store_1),
        );

        let mut store_1 = _mm256_add_epi16(
            _mm256_permute2x128_si256::<0x20>(store_2, store_3),
            _mm256_permute2x128_si256::<0x31>(store_2, store_3),
        );

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);

            let w01 = _mm_cvtsi32_si128((w_ptr.as_ptr() as *const i32).read_unaligned());
            let w23 = _mm_cvtsi32_si128(
                (w_ptr.get_unchecked(2..).as_ptr() as *const i32).read_unaligned(),
            );

            let weights01 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w01), w01),
                distr_weights,
            );
            let weights23 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w23), w23),
                distr_weights,
            );

            let start_bounds = bounds.start + jx;

            let rgb_pixel_0 = _mm_loadu_si128(
                src0.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
            );

            let rgb_pixel_1 = _mm_loadu_si128(
                src1.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
            );
            let rgb_pixel_2 = _mm_loadu_si128(
                src2.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
            );
            let rgb_pixel_3 = _mm_loadu_si128(
                src3.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m128i,
            );

            let px0 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_0), rgb_pixel_1);
            let px1 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_2), rgb_pixel_3);

            let lo0 = _mm256_srli_epi16::<2>(_mm256_unpacklo_epi8(px0, px0));
            let lo1 = _mm256_srli_epi16::<2>(_mm256_unpacklo_epi8(px1, px1));

            store_0 = _mm256_add_epi16(store_0, _mm256_mulhrs_epi16(lo0, weights01));
            store_1 = _mm256_add_epi16(store_1, _mm256_mulhrs_epi16(lo1, weights01));

            let hi0 = _mm256_srli_epi16::<2>(_mm256_unpackhi_epi8(px0, px0));
            let hi1 = _mm256_srli_epi16::<2>(_mm256_unpackhi_epi8(px1, px1));

            store_0 = _mm256_add_epi16(store_0, _mm256_mulhrs_epi16(hi0, weights23));
            store_1 = _mm256_add_epi16(store_1, _mm256_mulhrs_epi16(hi1, weights23));

            jx += 4;
        }

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);
            let bounds_start = bounds.start + jx;

            let w01 = _mm_shuffle_epi8(
                _mm_cvtsi32_si128((w_ptr.as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let rgb_pixel_0 =
                _mm_loadu_si64(src0.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
            let rgb_pixel_1 =
                _mm_loadu_si64(src1.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
            let rgb_pixel_2 =
                _mm_loadu_si64(src2.get_unchecked((bounds_start * CHANNELS)..).as_ptr());
            let rgb_pixel_3 =
                _mm_loadu_si64(src3.get_unchecked((bounds_start * CHANNELS)..).as_ptr());

            let weight01 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w01), w01);

            let mut px0 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_0), rgb_pixel_1);
            let mut px1 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgb_pixel_2), rgb_pixel_3);

            px0 = _mm256_unpacklo_epi8(px0, px0);
            px1 = _mm256_unpacklo_epi8(px1, px1);

            px0 = _mm256_srli_epi16::<2>(px0);
            px1 = _mm256_srli_epi16::<2>(px1);

            store_0 = _mm256_add_epi16(store_0, _mm256_mulhrs_epi16(px0, weight01));
            store_1 = _mm256_add_epi16(store_1, _mm256_mulhrs_epi16(px1, weight01));

            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx);

            let weight0 = _mm256_set1_epi16(*w_ptr);

            let bounds_start = bounds.start + jx;

            let src_ptr0 = src0.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr1 = src1.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr2 = src2.get_unchecked((bounds_start * CHANNELS)..);
            let src_ptr3 = src3.get_unchecked((bounds_start * CHANNELS)..);

            let rgba_pixel0 = _mm_cvtsi32_si128((src_ptr0.as_ptr() as *const i32).read_unaligned());
            let rgba_pixel1 = _mm_cvtsi32_si128((src_ptr1.as_ptr() as *const i32).read_unaligned());
            let rgba_pixel2 = _mm_cvtsi32_si128((src_ptr2.as_ptr() as *const i32).read_unaligned());
            let rgba_pixel3 = _mm_cvtsi32_si128((src_ptr3.as_ptr() as *const i32).read_unaligned());

            let mut px0 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0), rgba_pixel1);
            let mut px1 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel2), rgba_pixel3);

            px0 = _mm256_unpacklo_epi8(px0, px0);
            px1 = _mm256_unpacklo_epi8(px1, px1);

            px0 = _mm256_srli_epi16::<2>(px0);
            px1 = _mm256_srli_epi16::<2>(px1);

            store_0 = _mm256_add_epi16(store_0, _mm256_mulhrs_epi16(px0, weight0));
            store_1 = _mm256_add_epi16(store_1, _mm256_mulhrs_epi16(px1, weight0));
            jx += 1;
        }

        let hi_s0 = _mm256_permute4x64_epi64::<{ shuffle(2, 3, 0, 1) }>(store_0);
        let hi_s1 = _mm256_permute4x64_epi64::<{ shuffle(2, 3, 0, 1) }>(store_1);

        store_0 = _mm256_add_epi16(store_0, hi_s0);
        store_1 = _mm256_add_epi16(store_1, hi_s1);

        store_0 = _mm256_srai_epi16::<V_SHR>(store_0);
        store_1 = _mm256_srai_epi16::<V_SHR>(store_1);

        let packed8_0 = _mm256_packus_epi16(store_0, store_0);
        let packed8_1 = _mm256_packus_epi16(store_1, store_1);

        _mm_storeu_si32(
            chunk0.as_mut_ptr() as *mut _,
            _mm256_castsi256_si128(packed8_0),
        );
        _mm_storeu_si32(
            chunk1.as_mut_ptr() as *mut _,
            _mm256_extracti128_si256::<1>(packed8_0),
        );
        _mm_storeu_si32(
            chunk2.as_mut_ptr() as *mut _,
            _mm256_castsi256_si128(packed8_1),
        );
        _mm_storeu_si32(
            chunk3.as_mut_ptr() as *mut _,
            _mm256_extracti128_si256::<1>(packed8_1),
        );
    }
}

pub(crate) fn convolve_horizontal_rgba_avx_rows_one_lb(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        convolve_horizontal_rgba_avx_rows_one_impl(src, dst, filter_weights);
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_rgba_avx_rows_one_impl(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    const CHANNELS: usize = 4;

    let shuffle_weights = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3);

    const SCALE: i32 = 6;
    const V_SHR: i32 = SCALE;
    const ROUNDING: i16 = 1 << (V_SHR - 1);

    let vld = _mm256_setr_epi16(
        ROUNDING, ROUNDING, ROUNDING, ROUNDING, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );

    let distr_weights = _mm256_setr_epi8(
        0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3,
        2, 3,
    );

    for ((dst, bounds), weights) in dst
        .chunks_exact_mut(CHANNELS)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;
        let mut store = vld;

        while jx + 8 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);

            let w01 = _mm_loadu_si32(w_ptr.as_ptr() as *const _);
            let w23 = _mm_loadu_si32(w_ptr.get_unchecked(2..).as_ptr() as *const _);
            let w45 = _mm_loadu_si32(w_ptr.get_unchecked(4..).as_ptr() as *const _);
            let w67 = _mm_loadu_si32(w_ptr.get_unchecked(6..).as_ptr() as *const _);

            let w0145 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w01), w45),
                distr_weights,
            );
            let w2367 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w23), w67),
                distr_weights,
            );

            let start_bounds = bounds.start + jx;

            let rgb_pixel_0 = _mm256_loadu_si256(
                src.get_unchecked((start_bounds * CHANNELS)..).as_ptr() as *const __m256i,
            );

            store = hdot4(store, rgb_pixel_0, w0145, w2367);

            jx += 8;
        }

        while jx + 4 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);
            let bounds_start = bounds.start + jx;

            let w01 = _mm_loadu_si32(w_ptr.as_ptr() as *const _);
            let w23 = _mm_loadu_si32(w_ptr.get_unchecked(2..).as_ptr() as *const _);

            let weights = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(w01), w23),
                distr_weights,
            );

            let src_ptr = src.get_unchecked((bounds_start * CHANNELS)..);

            let rgb_pixel = _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(
                _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i),
            ));

            store = hdot2(store, rgb_pixel, weights);

            jx += 4;
        }

        let mut store = _mm_add_epi16(
            _mm256_castsi256_si128(store),
            _mm256_extracti128_si256::<1>(store),
        );

        while jx + 2 < bounds.size {
            let w_ptr = weights.get_unchecked(jx..);
            let bounds_start = bounds.start + jx;

            let weight01 = _mm_shuffle_epi8(
                _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned()),
                shuffle_weights,
            );

            let src_ptr = src.get_unchecked((bounds_start * CHANNELS)..);

            let rgb_pixel = _mm_loadu_si64(src_ptr.as_ptr());

            store = hdot(store, rgb_pixel, weight01);

            jx += 2;
        }

        while jx < bounds.size {
            let w_ptr = weights.get_unchecked(jx);
            let weight0 = _mm_set1_epi16(*w_ptr);

            let start_bounds = bounds.start + jx;

            store = convolve_horizontal_parts_one_rgba_sse(start_bounds, src, weight0, store);
            jx += 1;
        }

        store = _mm_add_hi_lo_epi16(store);

        let store_16_8 = _mm_srai_epi16::<V_SHR>(store);
        _mm_storeu_si32(
            dst.as_mut_ptr() as *mut _,
            _mm_packus_epi16(store_16_8, store_16_8),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::arch::x86_64::*;

    #[test]
    fn test_sum_avx_lanes() {
        unsafe {
            let store_0_s = _mm_set1_epi16(5);
            let store_1_s = _mm_set1_epi16(17);
            let store_0 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(store_0_s), store_1_s);
            let store_0_s = _mm_set1_epi16(22);
            let store_1_s = _mm_set1_epi16(15);
            let store_1 =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(store_0_s), store_1_s);
            let new_store = _mm256_add_epi16(
                _mm256_permute2x128_si256::<0x20>(store_0, store_1),
                _mm256_permute2x128_si256::<0x31>(store_0, store_1),
            );
            let original_0 = _mm256_extract_epi16::<0>(new_store);
            let original_1 = _mm256_extract_epi16::<8>(new_store);
            assert_eq!(original_0, 5 + 17);
            assert_eq!(original_1, 22 + 15);
        }
    }
}
