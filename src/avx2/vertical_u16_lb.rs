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
use crate::avx2::utils::{_mm_dot16_avx_epi32, _mm256_dot16_avx_epi32, avx2_pack_u32};
use crate::filter_weights::FilterBounds;
use crate::support::{PRECISION, ROUNDING_CONST};
use std::arch::x86_64::*;

/// This is fixed point path for bit-depth's lower or equal to 12
pub(crate) fn convolve_column_lb_avx2_u16(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    unsafe {
        convolve_column_lb_avx_u16_reg(bounds, src, dst, src_stride, weight, bit_depth);
    }
}

#[cfg(feature = "avx512")]
pub(crate) fn convolve_column_lb_avx2_u16_vnni(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    unsafe {
        convolve_column_lb_avx_u16_dot(bounds, src, dst, src_stride, weight, bit_depth);
    }
}

#[cfg(feature = "avx512")]
#[target_feature(enable = "avx2", enable = "avxvnni")]
fn convolve_column_lb_avx_u16_dot(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    convolve_column_lb_avx_u16_impl::<true>(bounds, src, dst, src_stride, weight, bit_depth);
}

#[target_feature(enable = "avx2")]
fn convolve_column_lb_avx_u16_reg(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    convolve_column_lb_avx_u16_impl::<false>(bounds, src, dst, src_stride, weight, bit_depth);
}

#[inline(always)]
fn accumulate_pair<const HAS_DOT: bool>(
    sum: &mut [__m256i; 2],
    first: __m256i,
    second: __m256i,
    weights: __m256i,
) {
    unsafe {
        sum[0] = _mm256_dot16_avx_epi32::<HAS_DOT>(
            sum[0],
            _mm256_unpacklo_epi16(first, second),
            weights,
        );
        sum[1] = _mm256_dot16_avx_epi32::<HAS_DOT>(
            sum[1],
            _mm256_unpackhi_epi16(first, second),
            weights,
        );
    }
}

#[inline(always)]
fn convolve_column_lb_32<const HAS_DOT: bool>(
    chunks: &mut [[u16; 32]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
    cx: usize,
) -> usize {
    unsafe {
        let mut cx = cx;
        let max_colors = _mm256_set1_epi16((1 << bit_depth) - 1);
        let rounding = _mm256_set1_epi32(ROUNDING_CONST);
        let zero = _mm256_setzero_si256();
        let (pairs, tail) = weights.as_chunks::<2>();

        for dst in chunks {
            let mut low = [rounding; 2];
            let mut high = [rounding; 2];
            let start = src.as_ptr().wrapping_add(bounds.start * src_stride + cx);
            // Pair each sample with the one under it so one dot product covers two taps.
            for (index, coefficients) in pairs.iter().enumerate() {
                let weights = _mm256_broadcastd_epi32(_mm_loadu_si32(coefficients.as_ptr().cast()));
                let first = start.add(index * 2 * src_stride);
                let second = first.add(src_stride);
                accumulate_pair::<HAS_DOT>(
                    &mut low,
                    _mm256_loadu_si256(first.cast()),
                    _mm256_loadu_si256(second.cast()),
                    weights,
                );
                accumulate_pair::<HAS_DOT>(
                    &mut high,
                    _mm256_loadu_si256(first.add(16).cast()),
                    _mm256_loadu_si256(second.add(16).cast()),
                    weights,
                );
            }
            if let [weight] = *tail {
                let weights = _mm256_set1_epi32(i32::from(weight as u16));
                let row = start.add(pairs.len() * 2 * src_stride);
                accumulate_pair::<HAS_DOT>(&mut low, _mm256_loadu_si256(row.cast()), zero, weights);
                accumulate_pair::<HAS_DOT>(
                    &mut high,
                    _mm256_loadu_si256(row.add(16).cast()),
                    zero,
                    weights,
                );
            }

            for (index, [low, high]) in [low, high].into_iter().enumerate() {
                let low = _mm256_srai_epi32::<PRECISION>(low);
                let high = _mm256_srai_epi32::<PRECISION>(high);
                let packed = _mm256_min_epi16(_mm256_packus_epi32(low, high), max_colors);
                _mm256_storeu_si256(dst.as_mut_ptr().add(index * 16).cast(), packed);
            }
            cx += 32;
        }
        cx
    }
}

#[inline(always)]
fn convolve_column_lb_16<const HAS_DOT: bool>(
    chunks: &mut [[u16; 16]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
    cx: usize,
) -> usize {
    unsafe {
        let v_px = cx;
        let mut cx = cx;
        let max_colors = (1 << bit_depth) - 1;
        let v_max_colors = _mm256_set1_epi16(max_colors);
        let bounds_size = bounds.size;

        for (x, dst) in chunks.iter_mut().enumerate() {
            let mut store0 = _mm256_set1_epi32(ROUNDING_CONST);
            let mut store1 = _mm256_set1_epi32(ROUNDING_CONST);

            let v_dx = v_px + x * 16;

            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm256_set1_epi16(k_weight);

                let item_row0 = _mm256_loadu_si256(src_ptr.as_ptr().cast());

                store0 = _mm256_dot16_avx_epi32::<HAS_DOT>(
                    store0,
                    _mm256_unpacklo_epi16(item_row0, _mm256_setzero_si256()),
                    v_weight,
                );
                store1 = _mm256_dot16_avx_epi32::<HAS_DOT>(
                    store1,
                    _mm256_unpackhi_epi16(item_row0, _mm256_setzero_si256()),
                    v_weight,
                );
            }

            let v_st0 = _mm256_srai_epi32::<PRECISION>(store0);
            let v_st1 = _mm256_srai_epi32::<PRECISION>(store1);

            let item0 = _mm256_min_epi16(_mm256_packus_epi32(v_st0, v_st1), v_max_colors);

            _mm256_storeu_si256(dst.as_mut_ptr().cast(), item0);

            cx += 16;
        }
        cx
    }
}

#[inline(always)]
fn convolve_column_lb_8<const HAS_DOT: bool>(
    chunks: &mut [[u16; 8]],
    bounds: &FilterBounds,
    src: &[u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
    cx: usize,
) -> usize {
    unsafe {
        let v_px = cx;
        let mut cx = cx;
        let max_colors = (1 << bit_depth) - 1;
        let v_max_colors = _mm256_set1_epi16(max_colors);
        let bounds_size = bounds.size;

        for (x, dst) in chunks.iter_mut().enumerate() {
            let mut store0 = _mm256_set1_epi32(ROUNDING_CONST);

            let v_dx = v_px + x * 8;

            for (j, &k_weight) in weight.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm256_set1_epi16(k_weight);

                let item_row = _mm256_permute4x64_epi64::<0x50>(_mm256_castsi128_si256(
                    _mm_loadu_si128(src_ptr.as_ptr().cast()),
                ));

                store0 = _mm256_dot16_avx_epi32::<HAS_DOT>(
                    store0,
                    _mm256_unpacklo_epi16(item_row, _mm256_setzero_si256()),
                    v_weight,
                );
            }

            let v_st0 = _mm256_srai_epi32::<PRECISION>(store0);

            let item = _mm256_min_epi16(avx2_pack_u32(v_st0, _mm256_setzero_si256()), v_max_colors);
            _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm256_castsi256_si128(item));

            cx += 8;
        }
        cx
    }
}

#[inline(always)]
fn convolve_column_lb_avx_u16_impl<const HAS_DOT: bool>(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
) {
    unsafe {
        assert!((1..=16).contains(&bit_depth));
        let max_colors = (1 << bit_depth) - 1;
        let mut cx = 0usize;

        let bounds_size = bounds.size;
        let weights = &weights[..bounds.size];

        let zeros = _mm_setzero_si128();
        let v_max_colors_sse = _mm_set1_epi16(max_colors);

        cx = convolve_column_lb_32::<HAS_DOT>(
            dst.as_chunks_mut::<32>().0,
            bounds,
            src,
            src_stride,
            weights,
            bit_depth,
            cx,
        );

        let mut rem = dst.as_chunks_mut::<32>().1;

        cx = convolve_column_lb_16::<HAS_DOT>(
            rem.as_chunks_mut::<16>().0,
            bounds,
            src,
            src_stride,
            weights,
            bit_depth,
            cx,
        );

        rem = rem.as_chunks_mut::<16>().1;

        cx = convolve_column_lb_8::<HAS_DOT>(
            rem.as_chunks_mut::<8>().0,
            bounds,
            src,
            src_stride,
            weights,
            bit_depth,
            cx,
        );

        let tail8 = rem.as_chunks_mut::<8>().1;
        let iter4 = tail8.as_chunks_mut::<4>().0.iter_mut();

        let v_cx = cx;

        for (x, dst) in iter4.enumerate() {
            let mut store0 = _mm_set1_epi32(ROUNDING_CONST);

            let v_dx = v_cx + x * 4;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm_set1_epi16(k_weight);

                let item_row = _mm_loadu_si64(src_ptr.as_ptr() as *const u8);

                store0 = _mm_dot16_avx_epi32::<HAS_DOT>(
                    store0,
                    _mm_unpacklo_epi16(item_row, zeros),
                    v_weight,
                );
            }

            let v_st = _mm_srai_epi32::<PRECISION>(store0);

            let u_store0 = _mm_min_epi16(_mm_packus_epi32(v_st, v_st), v_max_colors_sse);
            _mm_storeu_si64(dst.as_mut_ptr() as *mut u8, u_store0);

            cx += 4;
        }

        let tail4 = tail8.as_chunks_mut::<4>().1;

        let a_px = cx;

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = ROUNDING_CONST;

            let v_px = a_px + x;

            for (j, &k_weight) in weights.iter().take(bounds_size).enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_ptr = src.get_unchecked(offset);

                store0 = store0.wrapping_add((*src_ptr as i32).wrapping_mul(k_weight as i32));
            }

            *dst = (store0 >> PRECISION).max(0).min(max_colors as i32) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::convolve_column_lb_avx2_u16;
    use crate::filter_weights::FilterBounds;
    use crate::support::{PRECISION, ROUNDING_CONST};

    type Kernel = fn(usize, &FilterBounds, &[u16], &mut [u16], usize, &[i16], u32);

    fn check_q15_rows(kernel: Kernel) {
        for depth in [10, 12] {
            let max = (1 << depth) - 1;
            for width in 1..=101 {
                let stride = width + (width % 2) * 5;
                for taps in 1..=32 {
                    let bounds = FilterBounds::new(taps % 3, taps);
                    // The last tap reaches the final active sample, without row-end padding.
                    let len = (bounds.start + taps - 1) * stride + width;
                    let source: Vec<_> = (0..len)
                        .map(|i| ((i * 317 + i / 17 * 89) as u16) & max)
                        .collect();
                    let mut weights: Vec<_> = (0..taps)
                        .map(|i| ((i * 199 + taps * 229) % 1024) as i16 - 512)
                        .collect();
                    if taps == 1 {
                        weights[0] = i16::MAX;
                    } else {
                        let tail_sum: i32 = weights[2..].iter().map(|&w| i32::from(w)).sum();
                        weights[0] = (16384 - tail_sum) as i16;
                        weights[1] = 16384;
                    }
                    let mut output = vec![0; width];
                    kernel(
                        width,
                        &bounds,
                        &source,
                        &mut output,
                        stride,
                        &weights,
                        depth,
                    );
                    for (x, actual) in output.into_iter().enumerate() {
                        let expected =
                            weights
                                .iter()
                                .enumerate()
                                .fold(ROUNDING_CONST, |sum, (tap, &w)| {
                                    sum.wrapping_add(
                                        i32::from(source[(bounds.start + tap) * stride + x])
                                            * i32::from(w),
                                    )
                                });
                        let expected = (expected >> PRECISION).clamp(0, i32::from(max)) as u16;
                        assert_eq!(
                            actual, expected,
                            "depth {depth}, width {width}, {taps} taps, x {x}"
                        );
                    }
                }
            }
        }

        // Seventeen maximum positive products wrap the accumulator below zero.
        let bounds = FilterBounds::new(0, 17);
        let source = vec![4095; 17 * 65];
        let mut output = [u16::MAX; 65];
        kernel(65, &bounds, &source, &mut output, 65, &[32767; 17], 12);
        assert_eq!(output, [0; 65]);
    }

    #[test]
    fn avx2_q15_vertical_pairs_match_scalar() {
        if std::is_x86_feature_detected!("avx2") {
            check_q15_rows(convolve_column_lb_avx2_u16);
        }
        #[cfg(feature = "avx512")]
        if std::is_x86_feature_detected!("avxvnni") {
            check_q15_rows(super::convolve_column_lb_avx2_u16_vnni);
        }
    }
}
