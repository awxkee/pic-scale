/*
 * Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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

pub(crate) fn sve_convolve_horizontal_rgb_neon_rows_4_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_neon_rows_4_impl(src, src_stride, dst, dst_stride, filter_weights);
    }
}

#[target_feature(enable = "sve2,sve,i8mm")]
fn convolve_horizontal_rgb_neon_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    static TBL0: [u8; 16] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255];
    let v_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), TBL0.as_ptr()) };

    let rnd_const: i32 = 1 << 6;

    const CN: usize = 3;
    let init = svdup_n_s32(rnd_const);
    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
    let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
    let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
    let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

    let pg4 = svwhilelt_b8_u32(0u32, 4u32);
    let pg_weights12 = svwhilelt_b8_u32(0u32, 12u32);
    let pg3 = svwhilelt_b32_u32(0u32, 3u32);
    let pg12 = svwhilelt_b8_u32(0u32, 12u32);

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
        let mut store_0 = init;
        let mut store_1 = init;
        let mut store_2 = init;
        let mut store_3 = init;

        let src0 = src;
        let src1 = unsafe { src0.get_unchecked(src_stride..) };
        let src2 = unsafe { src1.get_unchecked(src_stride..) };
        let src3 = unsafe { src2.get_unchecked(src_stride..) };

        // Three indexed dot products share one coefficient load for 12/24-tap filters.
        while jx + 12 <= bounds.size {
            let bounds_start = (bounds.start + jx) * CN;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let w_ld = unsafe { svld1_s8(pg_weights12, w_ptr.as_ptr()) };
            let pixels_00 = unsafe { svld1_u8(pg12, src0.get_unchecked(bounds_start..).as_ptr()) };
            let pixels_01 = unsafe { svld1_u8(pg12, src1.get_unchecked(bounds_start..).as_ptr()) };
            let pixels_02 = unsafe { svld1_u8(pg12, src2.get_unchecked(bounds_start..).as_ptr()) };
            let pixels_03 = unsafe { svld1_u8(pg12, src3.get_unchecked(bounds_start..).as_ptr()) };
            let pixels_10 =
                unsafe { svld1_u8(pg12, src0.get_unchecked(bounds_start + 12..).as_ptr()) };
            let pixels_11 =
                unsafe { svld1_u8(pg12, src1.get_unchecked(bounds_start + 12..).as_ptr()) };
            let pixels_12 =
                unsafe { svld1_u8(pg12, src2.get_unchecked(bounds_start + 12..).as_ptr()) };
            let pixels_13 =
                unsafe { svld1_u8(pg12, src3.get_unchecked(bounds_start + 12..).as_ptr()) };
            let pixels_20 =
                unsafe { svld1_u8(pg12, src0.get_unchecked(bounds_start + 24..).as_ptr()) };
            let pixels_21 =
                unsafe { svld1_u8(pg12, src1.get_unchecked(bounds_start + 24..).as_ptr()) };
            let pixels_22 =
                unsafe { svld1_u8(pg12, src2.get_unchecked(bounds_start + 24..).as_ptr()) };
            let pixels_23 =
                unsafe { svld1_u8(pg12, src3.get_unchecked(bounds_start + 24..).as_ptr()) };
            let channels_00 = svtbl_u8(pixels_00, v_tbl);
            let channels_01 = svtbl_u8(pixels_01, v_tbl);
            let channels_02 = svtbl_u8(pixels_02, v_tbl);
            let channels_03 = svtbl_u8(pixels_03, v_tbl);
            let channels_10 = svtbl_u8(pixels_10, v_tbl);
            let channels_11 = svtbl_u8(pixels_11, v_tbl);
            let channels_12 = svtbl_u8(pixels_12, v_tbl);
            let channels_13 = svtbl_u8(pixels_13, v_tbl);
            let channels_20 = svtbl_u8(pixels_20, v_tbl);
            let channels_21 = svtbl_u8(pixels_21, v_tbl);
            let channels_22 = svtbl_u8(pixels_22, v_tbl);
            let channels_23 = svtbl_u8(pixels_23, v_tbl);
            store_0 = svusdot_lane_s32::<0>(store_0, channels_00, w_ld);
            store_1 = svusdot_lane_s32::<0>(store_1, channels_01, w_ld);
            store_2 = svusdot_lane_s32::<0>(store_2, channels_02, w_ld);
            store_3 = svusdot_lane_s32::<0>(store_3, channels_03, w_ld);
            store_0 = svusdot_lane_s32::<1>(store_0, channels_10, w_ld);
            store_1 = svusdot_lane_s32::<1>(store_1, channels_11, w_ld);
            store_2 = svusdot_lane_s32::<1>(store_2, channels_12, w_ld);
            store_3 = svusdot_lane_s32::<1>(store_3, channels_13, w_ld);
            store_0 = svusdot_lane_s32::<2>(store_0, channels_20, w_ld);
            store_1 = svusdot_lane_s32::<2>(store_1, channels_21, w_ld);
            store_2 = svusdot_lane_s32::<2>(store_2, channels_22, w_ld);
            store_3 = svusdot_lane_s32::<2>(store_3, channels_23, w_ld);
            jx += 12;
        }

        while jx + 4 <= bounds.size {
            let bounds_start = bounds.start + jx;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let w_ld = unsafe { svld1_s8(pg4, w_ptr.as_ptr()) };

            let rgb_pixel0 =
                unsafe { svld1_u8(pg12, src0.get_unchecked(bounds_start * CN..).as_ptr()) };
            let rgb_pixel1 =
                unsafe { svld1_u8(pg12, src1.get_unchecked(bounds_start * CN..).as_ptr()) };

            let rgb_pixel2 =
                unsafe { svld1_u8(pg12, src2.get_unchecked(bounds_start * CN..).as_ptr()) };
            let rgb_pixel3 =
                unsafe { svld1_u8(pg12, src3.get_unchecked(bounds_start * CN..).as_ptr()) };

            let r0 = svtbl_u8(rgb_pixel0, v_tbl);
            let r1 = svtbl_u8(rgb_pixel1, v_tbl);
            let r2 = svtbl_u8(rgb_pixel2, v_tbl);
            let r3 = svtbl_u8(rgb_pixel3, v_tbl);

            store_0 = svusdot_lane_s32::<0>(store_0, r0, w_ld);
            store_1 = svusdot_lane_s32::<0>(store_1, r1, w_ld);
            store_2 = svusdot_lane_s32::<0>(store_2, r2, w_ld);
            store_3 = svusdot_lane_s32::<0>(store_3, r3, w_ld);
            jx += 4;
        }

        if jx < bounds.size {
            let pq = svwhilelt_b8_u64(jx as u64, bounds.size as u64);
            let pqb = svwhilelt_b8_u64(0u64, (bounds.size - jx) as u64 * 3);

            let bounds_start = bounds.start + jx;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let w_ld = unsafe { svld1_s8(pq, w_ptr.as_ptr()) };

            let rgb_pixel0 =
                unsafe { svld1_u8(pqb, src0.get_unchecked(bounds_start * CN..).as_ptr()) };
            let rgb_pixel1 =
                unsafe { svld1_u8(pqb, src1.get_unchecked(bounds_start * CN..).as_ptr()) };

            let rgb_pixel2 =
                unsafe { svld1_u8(pqb, src2.get_unchecked(bounds_start * CN..).as_ptr()) };
            let rgb_pixel3 =
                unsafe { svld1_u8(pqb, src3.get_unchecked(bounds_start * CN..).as_ptr()) };

            let r0 = svtbl_u8(rgb_pixel0, v_tbl);
            let r1 = svtbl_u8(rgb_pixel1, v_tbl);
            let r2 = svtbl_u8(rgb_pixel2, v_tbl);
            let r3 = svtbl_u8(rgb_pixel3, v_tbl);
            store_0 = svusdot_lane_s32::<0>(store_0, r0, w_ld);
            store_1 = svusdot_lane_s32::<0>(store_1, r1, w_ld);
            store_2 = svusdot_lane_s32::<0>(store_2, r2, w_ld);
            store_3 = svusdot_lane_s32::<0>(store_3, r3, w_ld);
        }

        // Each result occupies the low byte of its original 32-bit lane.
        // Narrow stores compact the three active lanes into exactly three bytes.
        let n0 = svqshrunb_n_s32::<7>(store_0);
        let n1 = svqshrunb_n_s32::<7>(store_1);
        let n2 = svqshrunb_n_s32::<7>(store_2);
        let n3 = svqshrunb_n_s32::<7>(store_3);
        let b0 = svreinterpret_u32_u8(svqxtnb_u16(n0));
        let b1 = svreinterpret_u32_u8(svqxtnb_u16(n1));
        let b2 = svreinterpret_u32_u8(svqxtnb_u16(n2));
        let b3 = svreinterpret_u32_u8(svqxtnb_u16(n3));
        unsafe {
            svst1b_u32(pg3, chunk0.as_mut_ptr(), b0);
            svst1b_u32(pg3, chunk1.as_mut_ptr(), b1);
            svst1b_u32(pg3, chunk2.as_mut_ptr(), b2);
            svst1b_u32(pg3, chunk3.as_mut_ptr(), b3);
        }
    }
}

pub(crate) fn sve_convolve_horizontal_rgb_neon_row_one_dot(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_neon_row_one_impl_dot(src, dst, filter_weights);
    }
}

#[target_feature(enable = "sve2,sve,i8mm")]
fn convolve_horizontal_rgb_neon_row_one_impl_dot(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    const CN: usize = 3;

    static TBL0: [u8; 16] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255];
    let v_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), TBL0.as_ptr()) };

    let rnd_const: i32 = 1 << 6;
    let init = svdup_n_s32(rnd_const);

    let pg4 = svwhilelt_b8_u32(0u32, 4u32);
    let pg_weights12 = svwhilelt_b8_u32(0u32, 12u32);
    let pg3 = svwhilelt_b32_u32(0u32, 3u32);
    let pg12 = svwhilelt_b8_u32(0u32, 12u32);

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
        let mut store_0 = init;

        while jx + 12 <= bounds.size {
            let bounds_start = (bounds.start + jx) * CN;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let w_ld = unsafe { svld1_s8(pg_weights12, w_ptr.as_ptr()) };
            let pixels_0 = unsafe { svld1_u8(pg12, src.get_unchecked(bounds_start..).as_ptr()) };
            let pixels_1 =
                unsafe { svld1_u8(pg12, src.get_unchecked(bounds_start + 12..).as_ptr()) };
            let pixels_2 =
                unsafe { svld1_u8(pg12, src.get_unchecked(bounds_start + 24..).as_ptr()) };
            let channels_0 = svtbl_u8(pixels_0, v_tbl);
            let channels_1 = svtbl_u8(pixels_1, v_tbl);
            let channels_2 = svtbl_u8(pixels_2, v_tbl);
            store_0 = svusdot_lane_s32::<0>(store_0, channels_0, w_ld);
            store_0 = svusdot_lane_s32::<1>(store_0, channels_1, w_ld);
            store_0 = svusdot_lane_s32::<2>(store_0, channels_2, w_ld);
            jx += 12;
        }

        while jx + 4 <= bounds.size {
            let bounds_start = bounds.start + jx;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let w_ld = unsafe { svld1_s8(pg4, w_ptr.as_ptr()) };

            let rgb_pixel0 =
                unsafe { svld1_u8(pg12, src.get_unchecked(bounds_start * CN..).as_ptr()) };
            store_0 = svusdot_lane_s32::<0>(store_0, svtbl_u8(rgb_pixel0, v_tbl), w_ld);
            jx += 4;
        }

        if jx < bounds.size {
            let pq = svwhilelt_b8_u64(jx as u64, bounds.size as u64);
            let pqb = svwhilelt_b8_u64(0u64, (bounds.size - jx) as u64 * 3);

            let bounds_start = bounds.start + jx;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let w_ld = unsafe { svld1_s8(pq, w_ptr.as_ptr()) };

            let rgb_pixel0 =
                unsafe { svld1_u8(pqb, src.get_unchecked(bounds_start * CN..).as_ptr()) };
            store_0 = svusdot_lane_s32::<0>(store_0, svtbl_u8(rgb_pixel0, v_tbl), w_ld);
        }

        let n = svqshrunb_n_s32::<7>(store_0);
        let b = svreinterpret_u32_u8(svqxtnb_u16(n));
        unsafe {
            svst1b_u32(pg3, dst.as_mut_ptr(), b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_weights::FilterBounds;

    fn check_horizontal(four_rows: bool) {
        let Some(vl) = crate::sve2::test_vector_length(true) else {
            return;
        };
        for width in [1usize, 2, 3, 7, 19, 33] {
            let dst_stride = width * 3 + 13;
            for taps in [
                1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 31, 32,
                33, 47, 48, 49, 63, 64, 65,
            ] {
                let src_width = taps + width - 1;
                let src_stride = src_width * 3 + 11;
                // The last source row has no padding, including the final tap.
                let mut src = vec![0x9b; src_stride * 3 + src_width * 3];
                for row in 0..4 {
                    for x in 0..src_width {
                        for c in 0..3 {
                            src[row * src_stride + x * 3 + c] =
                                ((row * 67 + x * 29 + c * 43 + x * c * 11) % 256) as u8;
                        }
                    }
                }
                let aligned = taps.next_multiple_of(4);
                for clipped in [false, true] {
                    let bounds: Vec<_> = (0..width)
                        .map(|x| {
                            let size = if clipped && x == 0 {
                                taps.div_ceil(2)
                            } else if clipped {
                                taps - (x % 4).min(taps - 1)
                            } else {
                                taps
                            };
                            let start = if x == width - 1 { src_width - size } else { x };
                            FilterBounds::new(start, size)
                        })
                        .collect();
                    for pattern in 0..3 {
                        let mut weights = vec![-128i8; aligned * width];
                        for x in 0..width {
                            for j in 0..bounds[x].size {
                                weights[x * aligned + j] = match pattern {
                                    0 => ((j * 23 + x * 13) % 127) as i8 - 63,
                                    1 => {
                                        if j < bounds[x].size.div_ceil(2) {
                                            127
                                        } else {
                                            -128
                                        }
                                    }
                                    _ => {
                                        if j % 2 == 0 {
                                            -128
                                        } else {
                                            127
                                        }
                                    }
                                };
                            }
                            if pattern == 0 {
                                weights[x * aligned] = 96;
                            }
                        }
                        let filters =
                            FilterWeights::new(weights, taps, aligned, width, 0, bounds.clone());
                        let mut expected = vec![0xa5; dst_stride * 3 + width * 3];
                        let mut actual = expected.clone();
                        for row in 0..4 {
                            for x in 0..width {
                                let b = &filters.bounds[x];
                                for c in 0..3 {
                                    let mut sum = 64i64;
                                    for j in 0..b.size {
                                        sum += i64::from(
                                            src[row * src_stride + (b.start + j) * 3 + c],
                                        ) * i64::from(filters.weights[x * aligned + j]);
                                    }
                                    expected[row * dst_stride + x * 3 + c] =
                                        (sum >> 7).clamp(0, 255) as u8;
                                }
                            }
                            if !four_rows {
                                sve_convolve_horizontal_rgb_neon_row_one_dot(
                                    &src[row * src_stride..][..src_width * 3],
                                    &mut actual[row * dst_stride..][..width * 3],
                                    &filters,
                                    8,
                                );
                            }
                        }
                        if four_rows {
                            sve_convolve_horizontal_rgb_neon_rows_4_dot(
                                &src,
                                src_stride,
                                &mut actual,
                                dst_stride,
                                &filters,
                                8,
                            );
                        }
                        for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                actual,
                                expected,
                                "VL={}, width={width}, taps={taps}, clipped={clipped}, pattern={pattern}, row={}, byte={}",
                                vl * 8,
                                i / dst_stride,
                                i % dst_stride
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sve2_rows_4_matches_scalar_reference() {
        check_horizontal(true);
    }
    #[test]
    fn sve2_row_one_matches_scalar_reference() {
        check_horizontal(false);
    }
}
