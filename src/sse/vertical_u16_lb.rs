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
use crate::filter_weights::FilterBounds;
use crate::support::{PRECISION, ROUNDING_CONST};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
/// This is fixed point path for bit-depth's lower or equal to 12
pub(crate) fn convolve_column_lb_sse_u16(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    unsafe {
        convolve_column_lb_u16_impl(bounds, src, dst, src_stride, weight, bit_depth);
    }
}

#[target_feature(enable = "sse4.1")]
fn convolve_column_lb_u16_impl(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    unsafe {
        assert!((1..=16).contains(&bit_depth));
        let max_colors = (1 << bit_depth) - 1;
        let mut cx = 0usize;

        let zeros = _mm_setzero_si128();
        let initial_store = _mm_set1_epi32(ROUNDING_CONST);

        let v_max_colors = _mm_set1_epi16(max_colors);

        let v_px = cx;

        let iter16 = dst.as_chunks_mut::<16>();

        let weights = &weight[..bounds.size];

        for (x, dst) in iter16.0.iter_mut().enumerate() {
            let mut store0 = initial_store;
            let mut store1 = initial_store;
            let mut store2 = initial_store;
            let mut store3 = initial_store;

            let v_dx = v_px + x * 16;

            for (j, &k_weight) in weights.iter().enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm_set1_epi16(k_weight);

                let item_row0 = _mm_loadu_si128(src_ptr.as_ptr().cast());
                let item_row1 = _mm_loadu_si128(src_ptr.as_ptr().add(8).cast());

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpacklo_epi16(item_row0, zeros), v_weight),
                );
                store1 = _mm_add_epi32(
                    store1,
                    _mm_madd_epi16(_mm_unpackhi_epi16(item_row0, zeros), v_weight),
                );
                store2 = _mm_add_epi32(
                    store2,
                    _mm_madd_epi16(_mm_unpacklo_epi16(item_row1, zeros), v_weight),
                );
                store3 = _mm_add_epi32(
                    store3,
                    _mm_madd_epi16(_mm_unpackhi_epi16(item_row1, zeros), v_weight),
                );
            }

            let v_st0 = _mm_srai_epi32::<PRECISION>(store0);
            let v_st1 = _mm_srai_epi32::<PRECISION>(store1);
            let v_st2 = _mm_srai_epi32::<PRECISION>(store2);
            let v_st3 = _mm_srai_epi32::<PRECISION>(store3);

            let item0 = _mm_min_epi16(_mm_packus_epi32(v_st0, v_st1), v_max_colors);
            let item1 = _mm_min_epi16(_mm_packus_epi32(v_st2, v_st3), v_max_colors);

            _mm_storeu_si128(dst.as_mut_ptr().cast(), item0);
            _mm_storeu_si128(dst.as_mut_ptr().add(8).cast(), item1);

            cx += 16;
        }

        let tail16 = dst.as_chunks_mut::<16>().1;
        let iter8 = tail16.as_chunks_mut::<8>();

        let v_px = cx;

        for (x, dst) in iter8.0.iter_mut().enumerate() {
            let mut store0 = initial_store;
            let mut store1 = initial_store;

            let v_dx = v_px + x * 8;

            for (j, &k_weight) in weights.iter().enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm_set1_epi16(k_weight);

                let item_row = _mm_loadu_si128(src_ptr.as_ptr().cast());

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpacklo_epi16(item_row, zeros), v_weight),
                );
                store1 = _mm_add_epi32(
                    store1,
                    _mm_madd_epi16(_mm_unpackhi_epi16(item_row, zeros), v_weight),
                );
            }

            let v_st0 = _mm_srai_epi32::<PRECISION>(store0);
            let v_st1 = _mm_srai_epi32::<PRECISION>(store1);

            let item = _mm_min_epi16(_mm_packus_epi32(v_st0, v_st1), v_max_colors);
            _mm_storeu_si128(dst.as_mut_ptr().cast(), item);

            cx += 8;
        }

        let tail8 = tail16.as_chunks_mut::<8>().1;
        let iter4 = tail8.as_chunks_mut::<4>();

        let v_cx = cx;

        for (x, dst) in iter4.0.iter_mut().enumerate() {
            let mut store0 = initial_store;

            let v_dx = v_cx + x * 4;

            for (j, &k_weight) in weights.iter().enumerate() {
                let py = bounds.start + j;
                let src_ptr = src.get_unchecked((src_stride * py + v_dx)..);

                let v_weight = _mm_set1_epi16(k_weight);

                let item_row = _mm_loadu_si64(src_ptr.as_ptr() as *const u8);

                store0 = _mm_add_epi32(
                    store0,
                    _mm_madd_epi16(_mm_unpacklo_epi16(item_row, zeros), v_weight),
                );
            }

            let v_st = _mm_srai_epi32::<PRECISION>(store0);

            let u_store0 = _mm_min_epi16(_mm_packus_epi32(v_st, v_st), v_max_colors);
            _mm_storeu_si64(dst.as_mut_ptr() as *mut u8, u_store0);

            cx += 4;
        }

        let tail4 = tail8.as_chunks_mut::<4>().1;

        let a_px = cx;

        for (x, dst) in tail4.iter_mut().enumerate() {
            let mut store0 = ROUNDING_CONST;

            let v_px = a_px + x;

            for (j, &k_weight) in weights.iter().enumerate() {
                let py = bounds.start + j;
                let offset = src_stride * py + v_px;
                let src_ptr = src.get_unchecked(offset..(offset + 1));

                store0 = store0.wrapping_add((src_ptr[0] as i32).wrapping_mul(k_weight as i32));
            }

            *dst = (store0 >> PRECISION).max(0).min(max_colors as i32) as u16;
        }
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::ResamplingFunction;
    use crate::fixed_point_vertical::column_handler_fixed_point;
    use crate::test_utils::{XorShiftRng, make_row_filter_weights_u16};

    // Same idea as the u8 vertical tests in `src/sse/vertical_u8.rs`: for
    // every output row the accelerated low-bit-depth (<=12 bit) fixed-point
    // path must agree bit-for-bit with `column_handler_fixed_point::<u16, i32>`,
    // the same scalar reference `default_u16_column_plan` falls back to.
    fn max_source_rows_needed(bounds: &[FilterBounds]) -> usize {
        bounds.iter().map(|b| b.start + b.size).max().unwrap_or(0)
    }

    #[test]
    fn sse_vertical_lb_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        const BIT_DEPTH: u32 = 12;
        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_height, out_height) in [(64usize, 37usize), (37, 64), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_u16(resampling, in_height, out_height).unwrap();
                let needed_rows = max_source_rows_needed(&filter_weights.bounds);

                for &width in &[9usize, 53usize] {
                    let src_stride = width;
                    let mut rng = XorShiftRng::new(
                        0xC0FFEE ^ (in_height as u64) << 32 ^ (out_height as u64) << 16 ^ width as u64,
                    );
                    let src = rng.fill_u16(src_stride * needed_rows, (1u16 << BIT_DEPTH) - 1);

                    for (y, bounds) in filter_weights.bounds.iter().enumerate() {
                        let filter_offset = y * filter_weights.aligned_size;
                        let weights = &filter_weights.weights[filter_offset..];

                        let mut dst_scalar = vec![0u16; width];
                        let mut dst_sse = vec![0u16; width];
                        column_handler_fixed_point::<u16, i32>(
                            width,
                            bounds,
                            &src,
                            &mut dst_scalar,
                            src_stride,
                            weights,
                            BIT_DEPTH,
                        );
                        convolve_column_lb_sse_u16(
                            width,
                            bounds,
                            &src,
                            &mut dst_sse,
                            src_stride,
                            weights,
                            BIT_DEPTH,
                        );

                        assert_eq!(
                            dst_scalar, dst_sse,
                            "{resampling:?} {in_height}->{out_height} row {y} width {width}: SSE vertical lb output diverges from the scalar reference"
                        );
                    }
                }
            }
        }
    }
}
