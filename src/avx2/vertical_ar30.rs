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
use crate::avx2::ar30_utils::{_mm_unzip_3_ar30, _mm_zip_4_ar30};
use crate::filter_weights::FilterBounds;
use std::arch::x86_64::*;

pub(crate) fn avx_column_handler_fixed_point_ar30<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
>(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
    _: u32,
) {
    unsafe {
        let unit = ExecutionUnit::<AR30_TYPE, AR30_ORDER>::default();
        unit.pass(bounds, src, dst, src_stride, weight);
    }
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit<const AR30_TYPE: usize, const AR30_ORDER: usize> {}

impl<const AR30_TYPE: usize, const AR30_ORDER: usize> ExecutionUnit<AR30_TYPE, AR30_ORDER> {
    #[target_feature(enable = "avx2")]
    fn pass(
        &self,
        bounds: &FilterBounds,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        weights: &[i16],
    ) {
        unsafe {
            let mut cx = 0usize;

            let total_width = dst.len() / 4;

            let weights = &weights[..bounds.size];

            const PREC: i32 = 15;
            const RND_CONST: i32 = 1 << (PREC - 1);

            while cx + 8 <= total_width {
                let v_max = _mm_set1_epi16(1023);
                let filter = weights;
                let v_start_px = cx * 4;

                let mut v0 = _mm256_set1_epi32(RND_CONST);
                let mut v1 = _mm256_set1_epi32(RND_CONST);
                let mut v2 = _mm256_set1_epi32(RND_CONST);

                for (j, &k_weight) in filter.iter().enumerate() {
                    let py = bounds.start + j;
                    let weight = _mm256_set1_epi16(k_weight);
                    let offset = src_stride * py + v_start_px;
                    let src_ptr = src.get_unchecked(offset..(offset + 8 * 4));

                    let l0 = _mm_loadu_si128(src_ptr.as_ptr().cast());
                    let l1 = _mm_loadu_si128(src_ptr.as_ptr().add(4 * 4).cast());

                    let ps = _mm_unzip_3_ar30::<AR30_TYPE, AR30_ORDER>((l0, l1));

                    let ps0 = _mm256_cvtepu16_epi32(ps.0);
                    let ps1 = _mm256_cvtepu16_epi32(ps.1);
                    let ps2 = _mm256_cvtepu16_epi32(ps.2);

                    v0 = _mm256_add_epi32(v0, _mm256_madd_epi16(ps0, weight));
                    v1 = _mm256_add_epi32(v1, _mm256_madd_epi16(ps1, weight));
                    v2 = _mm256_add_epi32(v2, _mm256_madd_epi16(ps2, weight));
                }

                let v0 = _mm256_srai_epi32::<PREC>(v0);
                let v1 = _mm256_srai_epi32::<PREC>(v1);
                let v2 = _mm256_srai_epi32::<PREC>(v2);

                let r_v = _mm_min_epi16(
                    _mm_packus_epi32(
                        _mm256_castsi256_si128(v0),
                        _mm256_extracti128_si256::<1>(v0),
                    ),
                    v_max,
                );
                let g_v = _mm_min_epi16(
                    _mm_packus_epi32(
                        _mm256_castsi256_si128(v1),
                        _mm256_extracti128_si256::<1>(v1),
                    ),
                    v_max,
                );
                let b_v = _mm_min_epi16(
                    _mm_packus_epi32(
                        _mm256_castsi256_si128(v2),
                        _mm256_extracti128_si256::<1>(v2),
                    ),
                    v_max,
                );

                let v_dst = dst.get_unchecked_mut(v_start_px..(v_start_px + 8 * 4));

                let vals =
                    _mm_zip_4_ar30::<AR30_TYPE, AR30_ORDER>((r_v, g_v, b_v, _mm_set1_epi16(3)));
                _mm_storeu_si128(v_dst.as_mut_ptr().cast(), vals.0);
                _mm_storeu_si128(v_dst.as_mut_ptr().add(4 * 4).cast(), vals.1);

                cx += 8;
            }

            if cx < total_width {
                let diff = total_width - cx;

                let mut src_transient: [u8; 4 * 8] = [0; 4 * 8];
                let mut dst_transient: [u8; 4 * 8] = [0; 4 * 8];

                let v_max = _mm_set1_epi16(1023);
                let filter = weights;
                let v_start_px = cx * 4;

                let mut v0 = _mm256_set1_epi32(RND_CONST);
                let mut v1 = _mm256_set1_epi32(RND_CONST);
                let mut v2 = _mm256_set1_epi32(RND_CONST);

                for (j, &k_weight) in filter.iter().take(bounds.size).enumerate() {
                    let py = bounds.start + j;
                    let weight = _mm256_set1_epi16(k_weight);
                    let offset = src_stride * py + v_start_px;
                    let src_ptr = src.get_unchecked(offset..(offset + diff * 4));

                    std::ptr::copy_nonoverlapping(
                        src_ptr.as_ptr(),
                        src_transient.as_mut_ptr(),
                        diff * 4,
                    );

                    let l0 = _mm_loadu_si128(src_transient.as_ptr().cast());
                    let l1 = _mm_loadu_si128(src_transient.as_ptr().add(4 * 4).cast());

                    let ps = _mm_unzip_3_ar30::<AR30_TYPE, AR30_ORDER>((l0, l1));

                    let ps0 = _mm256_cvtepu16_epi32(ps.0);
                    let ps1 = _mm256_cvtepu16_epi32(ps.1);
                    let ps2 = _mm256_cvtepu16_epi32(ps.2);

                    v0 = _mm256_add_epi32(v0, _mm256_madd_epi16(ps0, weight));
                    v1 = _mm256_add_epi32(v1, _mm256_madd_epi16(ps1, weight));
                    v2 = _mm256_add_epi32(v2, _mm256_madd_epi16(ps2, weight));
                }

                let v0 = _mm256_srai_epi32::<PREC>(v0);
                let v1 = _mm256_srai_epi32::<PREC>(v1);
                let v2 = _mm256_srai_epi32::<PREC>(v2);

                let r_v = _mm_min_epi16(
                    _mm_packus_epi32(
                        _mm256_castsi256_si128(v0),
                        _mm256_extracti128_si256::<1>(v0),
                    ),
                    v_max,
                );
                let g_v = _mm_min_epi16(
                    _mm_packus_epi32(
                        _mm256_castsi256_si128(v1),
                        _mm256_extracti128_si256::<1>(v1),
                    ),
                    v_max,
                );
                let b_v = _mm_min_epi16(
                    _mm_packus_epi32(
                        _mm256_castsi256_si128(v2),
                        _mm256_extracti128_si256::<1>(v2),
                    ),
                    v_max,
                );

                let vals =
                    _mm_zip_4_ar30::<AR30_TYPE, AR30_ORDER>((r_v, g_v, b_v, _mm_set1_epi16(3)));
                _mm_storeu_si128(dst_transient.as_mut_ptr().cast(), vals.0);
                _mm_storeu_si128(dst_transient.as_mut_ptr().add(4 * 4).cast(), vals.1);

                let v_dst = dst.get_unchecked_mut(v_start_px..(v_start_px + diff * 4));
                std::ptr::copy_nonoverlapping(dst_transient.as_ptr(), v_dst.as_mut_ptr(), diff * 4);
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

    /// Vertical counterpart of the scalar reference in `horizontal_ar30.rs`:
    /// same Q15 fixed-point math (`support::PRECISION`), but each output
    /// pixel accumulates taps from `bounds.size` different SOURCE ROWS
    /// (via `src_stride`) instead of `bounds.size` taps within one row.
    fn scalar_reference_ar30_column<const AR_TYPE: usize, const AR_ORDER: usize>(
        width: usize,
        bounds: &FilterBounds,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        weight: &[i16],
    ) {
        const PRECISION: i32 = 15;
        const ROUNDING: i32 = 1 << (PRECISION - 1);
        let rgb_type: Rgb30 = AR_TYPE.into();

        for x in 0..width {
            let mut acc_r = ROUNDING;
            let mut acc_g = ROUNDING;
            let mut acc_b = ROUNDING;
            for (j, &w) in weight.iter().enumerate().take(bounds.size) {
                let py = bounds.start + j;
                let offset = src_stride * py + x * 4;
                let word = u32::from_ne_bytes(src[offset..offset + 4].try_into().unwrap());
                let (r, g, b, _a) = rgb_type.unpack::<AR_ORDER>(word);
                acc_r += r as i32 * w as i32;
                acc_g += g as i32 * w as i32;
                acc_b += b as i32 * w as i32;
            }
            let r = (acc_r >> PRECISION).clamp(0, 1023);
            let g = (acc_g >> PRECISION).clamp(0, 1023);
            let b = (acc_b >> PRECISION).clamp(0, 1023);
            let packed = rgb_type.pack_w_a::<AR_ORDER>(r, g, b, 3);
            let dst_offset = x * 4;
            dst[dst_offset..dst_offset + 4].copy_from_slice(&packed.to_ne_bytes());
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

    fn max_source_rows_needed(bounds: &[FilterBounds]) -> usize {
        bounds.iter().map(|b| b.start + b.size).max().unwrap_or(0)
    }

    fn run_vertical_test<const AR_TYPE: usize, const AR_ORDER: usize>(label: &str) {
        let rgb_type: Rgb30 = AR_TYPE.into();
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_height, out_height) in [(64usize, 37usize), (37, 64), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_u8(resampling, in_height, out_height).unwrap();
                let needed_rows = max_source_rows_needed(&filter_weights.bounds);

                for &width in &[9usize, 53usize] {
                    let mut rng = XorShiftRng::new(
                        0xC0FFEE ^ (in_height as u64) << 32 ^ (out_height as u64) << 16 ^ width as u64,
                    );
                    let src_stride = width * 4;
                    let src = make_ar30_src::<AR_ORDER>(rgb_type, &mut rng, width * needed_rows);

                    for (y, bounds) in filter_weights.bounds.iter().enumerate() {
                        let filter_offset = y * filter_weights.aligned_size;
                        let weights = &filter_weights.weights[filter_offset..];

                        let mut dst_scalar = vec![0u8; width * 4];
                        let mut dst_avx = vec![0u8; width * 4];
                        scalar_reference_ar30_column::<AR_TYPE, AR_ORDER>(
                            width,
                            bounds,
                            &src,
                            &mut dst_scalar,
                            src_stride,
                            weights,
                        );
                        avx_column_handler_fixed_point_ar30::<AR_TYPE, AR_ORDER>(
                            width,
                            bounds,
                            &src,
                            &mut dst_avx,
                            src_stride,
                            weights,
                            8,
                        );

                        assert_eq!(
                            dst_scalar, dst_avx,
                            "{label} {resampling:?} {in_height}->{out_height} row {y} width {width}: AVX2 AR30 vertical output diverges from the scalar reference"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn avx2_vertical_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        run_vertical_test::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Host as usize }>(
            "AR30 host",
        );
        run_vertical_test::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Network as usize }>(
            "AR30 network",
        );
        run_vertical_test::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Host as usize }>(
            "RA30 host",
        );
        run_vertical_test::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Network as usize }>(
            "RA30 network",
        );
    }
}
