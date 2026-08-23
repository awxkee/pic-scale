/*
 * Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::avx2::ar30_utils::{
    _mm_extract_ar30, _mm_ld1_ar30_s16, _mm_unzip_4_ar30_separate, _mm_unzips_4_ar30_separate,
};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

pub(crate) fn avx_convolve_horizontal_rgba_rows_4_ar30<
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        let unit = Row4ExecutionUnit::<AR_TYPE, AR_ORDER>::default();
        unit.pass(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionUnit<const AR_TYPE: usize, const AR_ORDER: usize> {}

impl<const AR_TYPE: usize, const AR_ORDER: usize> Row4ExecutionUnit<AR_TYPE, AR_ORDER> {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn conv_horiz_rgba_8_u8_i16(
        &self,
        start_x: usize,
        src0: &[u8],
        src1: &[u8],
        w0: __m256i,
        w1: __m256i,
        w2: __m256i,
        w3: __m256i,
        store: __m256i,
    ) -> __m256i {
        unsafe {
            let src_ptr0 = src0.get_unchecked(start_x * 4..);
            let src_ptr1 = src1.get_unchecked(start_x * 4..);

            let l0 = _mm256_loadu_si256(src_ptr0.as_ptr().cast());
            let l1 = _mm256_loadu_si256(src_ptr1.as_ptr().cast());

            let rgba_pixel0 = _mm_unzip_4_ar30_separate::<AR_TYPE, AR_ORDER>((
                _mm256_castsi256_si128(l0),
                _mm256_extractf128_si256::<1>(l0),
            ));
            let rgba_pixel1 = _mm_unzip_4_ar30_separate::<AR_TYPE, AR_ORDER>((
                _mm256_castsi256_si128(l1),
                _mm256_extractf128_si256::<1>(l1),
            ));

            let sh1 = _mm256_setr_epi8(
                0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4,
                5, 12, 13, 6, 7, 14, 15,
            );

            let v0 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.0), rgba_pixel1.0),
                sh1,
            );
            let v1 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.1), rgba_pixel1.1),
                sh1,
            );
            let v2 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.2), rgba_pixel1.2),
                sh1,
            );
            let v3 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.3), rgba_pixel1.3),
                sh1,
            );

            let mut v = _mm256_add_epi32(store, _mm256_madd_epi16(v0, w0));
            v = _mm256_add_epi32(v, _mm256_madd_epi16(v1, w1));
            v = _mm256_add_epi32(v, _mm256_madd_epi16(v2, w2));
            _mm256_add_epi32(v, _mm256_madd_epi16(v3, w3))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn conv_horiz_rgba_4_u8_i16(
        &self,
        start_x: usize,
        src0: &[u8],
        src1: &[u8],
        w0: __m256i,
        w1: __m256i,
        store: __m256i,
    ) -> __m256i {
        unsafe {
            let src_ptr0 = src0.get_unchecked(start_x * 4..);
            let src_ptr1 = src1.get_unchecked(start_x * 4..);

            let rgba_pixel0 = _mm_unzips_4_ar30_separate::<AR_TYPE, AR_ORDER>(_mm_loadu_si128(
                src_ptr0.as_ptr().cast(),
            ));
            let rgba_pixel1 = _mm_unzips_4_ar30_separate::<AR_TYPE, AR_ORDER>(_mm_loadu_si128(
                src_ptr1.as_ptr().cast(),
            ));

            let sh1 = _mm256_setr_epi8(
                0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4,
                5, 12, 13, 6, 7, 14, 15,
            );

            let v0 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.0), rgba_pixel1.0),
                sh1,
            );
            let v1 = _mm256_shuffle_epi8(
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(rgba_pixel0.1), rgba_pixel1.1),
                sh1,
            );

            let v = _mm256_add_epi32(store, _mm256_madd_epi16(v0, w0));
            _mm256_add_epi32(v, _mm256_madd_epi16(v1, w1))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn conv_horiz_rgba_1_u8_i16(
        &self,
        start_x: usize,
        src0: &[u8],
        src1: &[u8],
        w0: __m256i,
        store: __m256i,
    ) -> __m256i {
        unsafe {
            let src_ptr0 = src0.get_unchecked(start_x * 4..);
            let src_ptr1 = src1.get_unchecked(start_x * 4..);

            let ld0 = _mm_ld1_ar30_s16::<AR_TYPE, AR_ORDER>(src_ptr0);
            let ld1 = _mm_ld1_ar30_s16::<AR_TYPE, AR_ORDER>(src_ptr1);

            let full_lane = _mm_unpacklo_epi64(ld0, ld1);

            _mm256_add_epi32(
                store,
                _mm256_madd_epi16(_mm256_cvtepu16_epi32(full_lane), w0),
            )
        }
    }

    #[target_feature(enable = "avx2")]
    fn pass(
        &self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filter_weights: &FilterWeights<i16>,
    ) {
        unsafe {
            const PRECISION: i32 = 15;
            const ROUNDING: i32 = 1 << (PRECISION - 1);

            let init = _mm256_set1_epi32(ROUNDING);

            let v_cut_off = _mm256_set1_epi16(1023);

            let (row0_ref, rest) = dst.split_at_mut(dst_stride);
            let (row1_ref, rest) = rest.split_at_mut(dst_stride);
            let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

            let iter_row0 = row0_ref.as_chunks_mut::<4>().0.iter_mut();
            let iter_row1 = row1_ref.as_chunks_mut::<4>().0.iter_mut();
            let iter_row2 = row2_ref.as_chunks_mut::<4>().0.iter_mut();
            let iter_row3 = row3_ref.as_chunks_mut::<4>().0.iter_mut();

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

                let src0 = src;
                let src1 = src0.get_unchecked(src_stride..);
                let src2 = src1.get_unchecked(src_stride..);
                let src3 = src2.get_unchecked(src_stride..);

                while jx + 8 <= bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm256_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                    let w1 =
                        _mm256_set1_epi32((w_ptr.as_ptr().add(2) as *const i32).read_unaligned());
                    let w2 =
                        _mm256_set1_epi32((w_ptr.as_ptr().add(4) as *const i32).read_unaligned());
                    let w3 =
                        _mm256_set1_epi32((w_ptr.as_ptr().add(6) as *const i32).read_unaligned());
                    store_0 = self.conv_horiz_rgba_8_u8_i16(
                        bounds_start,
                        src0,
                        src1,
                        w0,
                        w1,
                        w2,
                        w3,
                        store_0,
                    );
                    store_1 = self.conv_horiz_rgba_8_u8_i16(
                        bounds_start,
                        src2,
                        src3,
                        w0,
                        w1,
                        w2,
                        w3,
                        store_1,
                    );
                    jx += 8;
                }

                while jx + 4 <= bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm256_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                    let w1 =
                        _mm256_set1_epi32((w_ptr.as_ptr().add(2) as *const i32).read_unaligned());
                    store_0 =
                        self.conv_horiz_rgba_4_u8_i16(bounds_start, src0, src1, w0, w1, store_0);
                    store_1 =
                        self.conv_horiz_rgba_4_u8_i16(bounds_start, src2, src3, w0, w1, store_1);
                    jx += 4;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx);
                    let bounds_start = bounds.start + jx;
                    let weight0 = _mm256_set1_epi16(*w_ptr);
                    store_0 =
                        self.conv_horiz_rgba_1_u8_i16(bounds_start, src0, src1, weight0, store_0);
                    store_1 =
                        self.conv_horiz_rgba_1_u8_i16(bounds_start, src2, src3, weight0, store_1);
                    jx += 1;
                }

                let store_0 = _mm256_srai_epi32::<PRECISION>(store_0);
                let store_1 = _mm256_srai_epi32::<PRECISION>(store_1);

                let store_0 = _mm256_packus_epi32(store_0, store_0);
                let store_1 = _mm256_packus_epi32(store_1, store_1);

                let ss0 = _mm256_min_epi16(store_0, v_cut_off);
                let ss1 = _mm256_min_epi16(store_1, v_cut_off);

                let packed0 = _mm_extract_ar30::<AR_TYPE, AR_ORDER>(_mm256_castsi256_si128(ss0));
                _mm_storeu_si32(chunk0.as_mut_ptr(), packed0);
                let packed1 =
                    _mm_extract_ar30::<AR_TYPE, AR_ORDER>(_mm256_extracti128_si256::<1>(ss0));
                _mm_storeu_si32(chunk1.as_mut_ptr(), packed1);
                let packed2 = _mm_extract_ar30::<AR_TYPE, AR_ORDER>(_mm256_castsi256_si128(ss1));
                _mm_storeu_si32(chunk2.as_mut_ptr(), packed2);
                let packed3 =
                    _mm_extract_ar30::<AR_TYPE, AR_ORDER>(_mm256_extracti128_si256::<1>(ss1));
                _mm_storeu_si32(chunk3.as_mut_ptr(), packed3);
            }
        }
    }
}

pub(crate) fn avx_convolve_horizontal_rgba_rows_ar30<
    const AR_TYPE: usize,
    const AR_ORDER: usize,
>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        let unit = Row1ExecutionUnit::<AR_TYPE, AR_ORDER>::default();
        unit.pass(src, dst, filter_weights);
    }
}

#[derive(Copy, Clone, Default)]
struct Row1ExecutionUnit<const AR_TYPE: usize, const AR_ORDER: usize> {}

impl<const AR_TYPE: usize, const AR_ORDER: usize> Row1ExecutionUnit<AR_TYPE, AR_ORDER> {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn conv_horiz_rgba_1_u8_i16(
        start_x: usize,
        src: &[u8],
        w0: __m128i,
        store: __m128i,
    ) -> __m128i {
        unsafe {
            let src_ptr = src.get_unchecked(start_x * 4..);
            let ld = _mm_ld1_ar30_s16::<AR_TYPE, AR_ORDER>(src_ptr);
            _mm_add_epi32(
                store,
                _mm_madd_epi16(_mm_unpacklo_epi16(ld, _mm_setzero_si128()), w0),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn conv_horiz_rgba_8_u8_i16(
        &self,
        start_x: usize,
        src: &[u8],
        w0: __m128i,
        w1: __m128i,
        w2: __m128i,
        w3: __m128i,
        store: __m128i,
    ) -> __m128i {
        unsafe {
            let src_ptr = src.get_unchecked(start_x * 4..);

            let l0 = _mm_loadu_si128(src_ptr.as_ptr().cast());
            let l1 = _mm_loadu_si128(src_ptr.as_ptr().add(4 * 4).cast());

            let rgba_pixel = _mm_unzip_4_ar30_separate::<AR_TYPE, AR_ORDER>((l0, l1));

            let sh1 = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

            let v0 = _mm_shuffle_epi8(rgba_pixel.0, sh1);
            let v1 = _mm_shuffle_epi8(rgba_pixel.1, sh1);
            let v2 = _mm_shuffle_epi8(rgba_pixel.2, sh1);
            let v3 = _mm_shuffle_epi8(rgba_pixel.3, sh1);

            let mut v = _mm_add_epi32(store, _mm_madd_epi16(v0, w0));
            v = _mm_add_epi32(v, _mm_madd_epi16(v1, w1));
            v = _mm_add_epi32(v, _mm_madd_epi16(v2, w2));
            _mm_add_epi32(v, _mm_madd_epi16(v3, w3))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn conv_horiz_rgba_4_u8_i16(
        &self,
        start_x: usize,
        src: &[u8],
        w0: __m128i,
        w1: __m128i,
        store: __m128i,
    ) -> __m128i {
        unsafe {
            let src_ptr = src.get_unchecked(start_x * 4..);

            let rgba_pixel = _mm_unzips_4_ar30_separate::<AR_TYPE, AR_ORDER>(_mm_loadu_si128(
                src_ptr.as_ptr().cast(),
            ));

            let sh1 = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

            let v0 = _mm_shuffle_epi8(rgba_pixel.0, sh1);
            let v1 = _mm_shuffle_epi8(rgba_pixel.1, sh1);

            let v = _mm_add_epi32(store, _mm_madd_epi16(v0, w0));
            _mm_add_epi32(v, _mm_madd_epi16(v1, w1))
        }
    }

    #[target_feature(enable = "avx2")]
    fn pass(&self, src: &[u8], dst: &mut [u8], filter_weights: &FilterWeights<i16>) {
        unsafe {
            // Must match `crate::support::PRECISION`, which is what the weights were
            // quantized with; shifting by 16 halves every channel.
            const PRECISION: i32 = 15;
            const ROUNDING: i32 = 1 << (PRECISION - 1);

            let init = _mm_set1_epi32(ROUNDING);

            // 16-bit lanes: the clamp is applied with `_mm_min_epi16` after
            // `_mm_packus_epi32`. Building it with `_mm_set1_epi32` leaves every odd
            // 16-bit lane zero, which clamps green to 0.
            let v_cut_off = _mm_set1_epi16(1023);

            for ((chunk0, &bounds), weights) in dst
                .as_chunks_mut::<4>()
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

                let bounds_size = bounds.size;

                let mut store_0 = init;

                let src0 = src;

                while jx + 8 <= bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                    let w1 = _mm_set1_epi32((w_ptr.as_ptr().add(2) as *const i32).read_unaligned());
                    let w2 = _mm_set1_epi32((w_ptr.as_ptr().add(4) as *const i32).read_unaligned());
                    let w3 = _mm_set1_epi32((w_ptr.as_ptr().add(6) as *const i32).read_unaligned());
                    store_0 =
                        self.conv_horiz_rgba_8_u8_i16(bounds_start, src0, w0, w1, w2, w3, store_0);
                    jx += 8;
                }

                while jx + 4 <= bounds_size {
                    let bounds_start = bounds.start + jx;
                    let w_ptr = weights.get_unchecked(jx..);
                    let w0 = _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                    let w1 = _mm_set1_epi32((w_ptr.as_ptr().add(2) as *const i32).read_unaligned());
                    store_0 = self.conv_horiz_rgba_4_u8_i16(bounds_start, src0, w0, w1, store_0);
                    jx += 4;
                }

                while jx < bounds_size {
                    let w_ptr = weights.get_unchecked(jx);
                    let bounds_start = bounds.start + jx;
                    let weight0 = _mm_set1_epi16(*w_ptr);
                    store_0 = Self::conv_horiz_rgba_1_u8_i16(bounds_start, src0, weight0, store_0);
                    jx += 1;
                }

                let store_0 = _mm_srai_epi32::<PRECISION>(store_0);

                let store_0 = _mm_packus_epi32(store_0, store_0);

                let store_16_0 = _mm_min_epi16(store_0, v_cut_off);

                let packed0 = _mm_extract_ar30::<AR_TYPE, AR_ORDER>(store_16_0);
                _mm_storeu_si32(chunk0.as_mut_ptr(), packed0);
            }
        }
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::ResamplingFunction;
    use crate::factory::{Ar30ByteOrder, Rgb30};
    use crate::test_utils::{XorShiftRng, make_row_filter_weights_u8};

    /// AR30 is a packed 10-10-10-2 format with no separate per-channel byte
    /// layout, so `handler_provider`'s scalar row-convolution reference (which
    /// operates on flat `u8` channel arrays) doesn't apply here. Rebuild the
    /// same math directly on top of the crate's own `Rgb30::unpack`/`pack_w_a`,
    /// using the identical Q15 weights (`support::PRECISION`) and rounding the
    /// AVX2 kernel uses.
    fn scalar_reference_ar30_row<const AR_TYPE: usize, const AR_ORDER: usize>(
        src: &[u8],
        dst: &mut [u8],
        filter_weights: &FilterWeights<i16>,
    ) {
        const PRECISION: i32 = 15;
        const ROUNDING: i32 = 1 << (PRECISION - 1);
        let rgb_type: Rgb30 = AR_TYPE.into();

        for (dst_chunk, (&bounds, weights)) in dst.as_chunks_mut::<4>().0.iter_mut().zip(
            filter_weights
                .bounds
                .iter()
                .zip(filter_weights.weights.chunks_exact(filter_weights.aligned_size)),
        ) {
            let mut acc_r = ROUNDING;
            let mut acc_g = ROUNDING;
            let mut acc_b = ROUNDING;
            for (k, &weight) in weights.iter().enumerate().take(bounds.size) {
                let w = weight as i32;
                let px = (bounds.start + k) * 4;
                let word = u32::from_ne_bytes(src[px..px + 4].try_into().unwrap());
                let (r, g, b, _a) = rgb_type.unpack::<AR_ORDER>(word);
                acc_r += r as i32 * w;
                acc_g += g as i32 * w;
                acc_b += b as i32 * w;
            }
            let r = (acc_r >> PRECISION).clamp(0, 1023);
            let g = (acc_g >> PRECISION).clamp(0, 1023);
            let b = (acc_b >> PRECISION).clamp(0, 1023);
            let packed = rgb_type.pack_w_a::<AR_ORDER>(r, g, b, 3);
            dst_chunk.copy_from_slice(&packed.to_ne_bytes());
        }
    }

    fn make_ar30_src<const AR_ORDER: usize>(
        rgb_type: Rgb30,
        rng: &mut XorShiftRng,
        pixel_count: usize,
    ) -> Vec<u8> {
        let components = rng.fill_u16(pixel_count * 3, 1023);
        let mut out = Vec::with_capacity(pixel_count * 4);
        for chunk in components.as_chunks::<3>().0 {
            let packed =
                rgb_type.pack_w_a::<AR_ORDER>(chunk[0] as i32, chunk[1] as i32, chunk[2] as i32, 3);
            out.extend_from_slice(&packed.to_ne_bytes());
        }
        out
    }

    fn run_row_test<const AR_TYPE: usize, const AR_ORDER: usize>(label: &str) {
        let rgb_type: Rgb30 = AR_TYPE.into();
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64), (256, 256), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_u8(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xC0FFEE ^ (in_size as u64) << 32 ^ out_size as u64);
                let src = make_ar30_src::<AR_ORDER>(rgb_type, &mut rng, in_size);

                let mut dst_scalar = vec![0u8; out_size * 4];
                let mut dst_avx = vec![0u8; out_size * 4];
                scalar_reference_ar30_row::<AR_TYPE, AR_ORDER>(
                    &src,
                    &mut dst_scalar,
                    &filter_weights,
                );
                avx_convolve_horizontal_rgba_rows_ar30::<AR_TYPE, AR_ORDER>(
                    &src,
                    &mut dst_avx,
                    &filter_weights,
                    8,
                );

                assert_eq!(
                    dst_scalar, dst_avx,
                    "{label} {resampling:?} {in_size}->{out_size}: AVX2 AR30 single-row output diverges from the scalar reference"
                );
            }
        }
    }

    fn run_rows_4_test<const AR_TYPE: usize, const AR_ORDER: usize>(label: &str) {
        let rgb_type: Rgb30 = AR_TYPE.into();
        const ROWS: usize = 4;
        for resampling in [ResamplingFunction::Bilinear, ResamplingFunction::Lanczos3] {
            for (in_size, out_size) in [(64usize, 37usize), (37, 64)] {
                let filter_weights =
                    make_row_filter_weights_u8(resampling, in_size, out_size).unwrap();
                let mut rng = XorShiftRng::new(0xBADF00D ^ (in_size as u64) << 32 ^ out_size as u64);
                let src_stride = in_size * 4;
                let dst_stride = out_size * 4;
                let src = make_ar30_src::<AR_ORDER>(rgb_type, &mut rng, in_size * ROWS);

                let mut dst_scalar = vec![0u8; dst_stride * ROWS];
                let mut dst_avx = vec![0u8; dst_stride * ROWS];
                for row in 0..ROWS {
                    scalar_reference_ar30_row::<AR_TYPE, AR_ORDER>(
                        &src[row * src_stride..],
                        &mut dst_scalar[row * dst_stride..(row + 1) * dst_stride],
                        &filter_weights,
                    );
                }
                avx_convolve_horizontal_rgba_rows_4_ar30::<AR_TYPE, AR_ORDER>(
                    &src,
                    src_stride,
                    &mut dst_avx,
                    dst_stride,
                    &filter_weights,
                    8,
                );

                assert_eq!(
                    dst_scalar, dst_avx,
                    "{label} {resampling:?} {in_size}->{out_size}: AVX2 AR30 4-row output diverges from the scalar reference"
                );
            }
        }
    }

    #[test]
    fn avx2_row_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        run_row_test::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Host as usize }>("AR30 host");
        run_row_test::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Network as usize }>(
            "AR30 network",
        );
        run_row_test::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Host as usize }>("RA30 host");
        run_row_test::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Network as usize }>(
            "RA30 network",
        );
    }

    #[test]
    fn avx2_rows_4_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        run_rows_4_test::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Host as usize }>("AR30 host");
        run_rows_4_test::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Network as usize }>(
            "AR30 network",
        );
        run_rows_4_test::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Host as usize }>("RA30 host");
        run_rows_4_test::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Network as usize }>(
            "RA30 network",
        );
    }
}
