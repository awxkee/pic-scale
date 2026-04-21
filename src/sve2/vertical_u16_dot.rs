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
use crate::filter_weights::FilterBounds;
use crate::support::ROUNDING_CONST;
use std::arch::aarch64::*;

pub(crate) fn convolve_vertical_sve2_u16_dot(
    width: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weight: &[i16],
    bit_depth: u32,
) {
    unsafe {
        convolve_vertical_sve2_u16_row(width, bounds, src, dst, src_stride, weight, bit_depth);
    }
}

macro_rules! pack_4_rows_sve_u16 {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {{
        let ab_lo = svzip1_u16($a, $b);
        let ab_hi = svzip2_u16($a, $b);

        let cd_lo = svzip1_u16($c, $d);
        let cd_hi = svzip2_u16($c, $d);

        let lo0 = svzip1_u32(svreinterpret_u32_u16(ab_lo), svreinterpret_u32_u16(cd_lo));
        let lo1 = svzip2_u32(svreinterpret_u32_u16(ab_lo), svreinterpret_u32_u16(cd_lo));
        let hi0 = svzip1_u32(svreinterpret_u32_u16(ab_hi), svreinterpret_u32_u16(cd_hi));
        let hi1 = svzip2_u32(svreinterpret_u32_u16(ab_hi), svreinterpret_u32_u16(cd_hi));

        [
            svreinterpret_u16_u32(lo0),
            svreinterpret_u16_u32(lo1),
            svreinterpret_u16_u32(hi0),
            svreinterpret_u16_u32(hi1),
        ]
    }};
}

macro_rules! narrow_i64_to_u16 {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {{
        let q0 = svqrshrunb_n_s64::<{ crate::support::PRECISION }>($a);
        let q1 = svqrshrunb_n_s64::<{ crate::support::PRECISION }>($b);
        let q2 = svqrshrunb_n_s64::<{ crate::support::PRECISION }>($c);
        let q3 = svqrshrunb_n_s64::<{ crate::support::PRECISION }>($d);
        let s01 = svuzp1_u32(q0, q1);
        let s23 = svuzp1_u32(q2, q3);
        svuzp1_u16(svreinterpret_u16_u32(s01), svreinterpret_u16_u32(s23))
    }};
}

#[target_feature(enable = "sve,sve2")]
fn work_wide_chunks_u16(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
) -> usize {
    let vl = svcnth() as usize;
    let mut cx = 0usize;
    let len = dst.len();

    let pg_full = svptrue_b16();

    let pg8 = svwhilelt_b16_u32(0u32, 8u32);
    let pg4 = svwhilelt_b16_u32(0u32, 4u32);
    let pg2 = svwhilelt_b16_u32(0u32, 2u32);
    let pg1 = svwhilelt_b16_u32(0u32, 1u32);

    let max_colors = svdup_n_u16(((1u32 << bit_depth) - 1) as u16);

    let shuf4 = svreinterpret_u8_s64(svdup_n_s64(i64::from_ne_bytes([0, 1, 2, 3, 4, 5, 6, 7])));
    let shuf4hi = svreinterpret_u8_s64(svdup_n_s64(i64::from_ne_bytes([
        8, 9, 10, 11, 12, 13, 14, 15,
    ])));

    while cx + vl * 2 <= len {
        let rounding = svdup_n_s64(ROUNDING_CONST as i64);

        let mut acc_lo_0 = rounding;
        let mut acc_lo_1 = rounding;
        let mut acc_lo_2 = rounding;
        let mut acc_lo_3 = rounding;
        let mut acc_hi_0 = rounding;
        let mut acc_hi_1 = rounding;
        let mut acc_hi_2 = rounding;
        let mut acc_hi_3 = rounding;

        let mut j = 0usize;

        while j + 8 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };

            let wq = unsafe { svreinterpret_s8_s16(svld1_s16(pg8, w.as_ptr())) };
            let vw0 = svreinterpret_s16_s8(svtbl_s8(wq, shuf4));
            let vw1 = svreinterpret_s16_s8(svtbl_s8(wq, shuf4hi));

            let base0 = src_stride * py + cx;
            let base1 = base0 + src_stride * 4;

            // Lower strip, first 4 rows
            let row0_lo = unsafe { svld1_u16(pg_full, src.get_unchecked(base0..).as_ptr()) };
            let row1_lo =
                unsafe { svld1_u16(pg_full, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row2_lo = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 2..).as_ptr(),
                )
            };
            let row3_lo = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 3..).as_ptr(),
                )
            };

            // Upper strip, first 4 rows
            let row0_hi = unsafe { svld1_u16(pg_full, src.get_unchecked(base0 + vl..).as_ptr()) };
            let row1_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride + vl..).as_ptr(),
                )
            };
            let row2_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 2 + vl..).as_ptr(),
                )
            };
            let row3_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 3 + vl..).as_ptr(),
                )
            };

            // Lower strip, second 4 rows
            let row4_lo = unsafe { svld1_u16(pg_full, src.get_unchecked(base1..).as_ptr()) };
            let row5_lo =
                unsafe { svld1_u16(pg_full, src.get_unchecked(base1 + src_stride..).as_ptr()) };
            let row6_lo = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base1 + src_stride * 2..).as_ptr(),
                )
            };
            let row7_lo = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base1 + src_stride * 3..).as_ptr(),
                )
            };

            // Upper strip, second 4 rows
            let row4_hi = unsafe { svld1_u16(pg_full, src.get_unchecked(base1 + vl..).as_ptr()) };
            let row5_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base1 + src_stride + vl..).as_ptr(),
                )
            };
            let row6_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base1 + src_stride * 2 + vl..).as_ptr(),
                )
            };
            let row7_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base1 + src_stride * 3 + vl..).as_ptr(),
                )
            };

            let [packed_lo_0, packed_lo_1, packed_lo_2, packed_lo_3] =
                pack_4_rows_sve_u16!(row0_lo, row1_lo, row2_lo, row3_lo);
            let [packed_hi_0, packed_hi_1, packed_hi_2, packed_hi_3] =
                pack_4_rows_sve_u16!(row0_hi, row1_hi, row2_hi, row3_hi);
            let [packed_lo_4, packed_lo_5, packed_lo_6, packed_lo_7] =
                pack_4_rows_sve_u16!(row4_lo, row5_lo, row6_lo, row7_lo);
            let [packed_hi_4, packed_hi_5, packed_hi_6, packed_hi_7] =
                pack_4_rows_sve_u16!(row4_hi, row5_hi, row6_hi, row7_hi);

            acc_lo_0 = svdot_s64(acc_lo_0, svreinterpret_s16_u16(packed_lo_0), vw0);
            acc_lo_1 = svdot_s64(acc_lo_1, svreinterpret_s16_u16(packed_lo_1), vw0);
            acc_lo_2 = svdot_s64(acc_lo_2, svreinterpret_s16_u16(packed_lo_2), vw0);
            acc_lo_3 = svdot_s64(acc_lo_3, svreinterpret_s16_u16(packed_lo_3), vw0);
            acc_hi_0 = svdot_s64(acc_hi_0, svreinterpret_s16_u16(packed_hi_0), vw0);
            acc_hi_1 = svdot_s64(acc_hi_1, svreinterpret_s16_u16(packed_hi_1), vw0);
            acc_hi_2 = svdot_s64(acc_hi_2, svreinterpret_s16_u16(packed_hi_2), vw0);
            acc_hi_3 = svdot_s64(acc_hi_3, svreinterpret_s16_u16(packed_hi_3), vw0);

            acc_lo_0 = svdot_s64(acc_lo_0, svreinterpret_s16_u16(packed_lo_4), vw1);
            acc_lo_1 = svdot_s64(acc_lo_1, svreinterpret_s16_u16(packed_lo_5), vw1);
            acc_lo_2 = svdot_s64(acc_lo_2, svreinterpret_s16_u16(packed_lo_6), vw1);
            acc_lo_3 = svdot_s64(acc_lo_3, svreinterpret_s16_u16(packed_lo_7), vw1);
            acc_hi_0 = svdot_s64(acc_hi_0, svreinterpret_s16_u16(packed_hi_4), vw1);
            acc_hi_1 = svdot_s64(acc_hi_1, svreinterpret_s16_u16(packed_hi_5), vw1);
            acc_hi_2 = svdot_s64(acc_hi_2, svreinterpret_s16_u16(packed_hi_6), vw1);
            acc_hi_3 = svdot_s64(acc_hi_3, svreinterpret_s16_u16(packed_hi_7), vw1);

            j += 8;
        }

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_s8(svtbl_s8(
                unsafe { svreinterpret_s8_s16(svld1_s16(pg4, w.as_ptr())) },
                shuf4,
            ));

            let base0 = src_stride * py + cx;

            // Lower strip
            let row0_lo = unsafe { svld1_u16(pg_full, src.get_unchecked(base0..).as_ptr()) };
            let row1_lo =
                unsafe { svld1_u16(pg_full, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row2_lo = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 2..).as_ptr(),
                )
            };
            let row3_lo = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 3..).as_ptr(),
                )
            };

            // Upper strip
            let row0_hi = unsafe { svld1_u16(pg_full, src.get_unchecked(base0 + vl..).as_ptr()) };
            let row1_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride + vl..).as_ptr(),
                )
            };
            let row2_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 2 + vl..).as_ptr(),
                )
            };
            let row3_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 3 + vl..).as_ptr(),
                )
            };

            let [packed_lo_0, packed_lo_1, packed_lo_2, packed_lo_3] =
                pack_4_rows_sve_u16!(row0_lo, row1_lo, row2_lo, row3_lo);
            let [packed_hi_0, packed_hi_1, packed_hi_2, packed_hi_3] =
                pack_4_rows_sve_u16!(row0_hi, row1_hi, row2_hi, row3_hi);

            acc_lo_0 = svdot_s64(acc_lo_0, svreinterpret_s16_u16(packed_lo_0), vw);
            acc_lo_1 = svdot_s64(acc_lo_1, svreinterpret_s16_u16(packed_lo_1), vw);
            acc_lo_2 = svdot_s64(acc_lo_2, svreinterpret_s16_u16(packed_lo_2), vw);
            acc_lo_3 = svdot_s64(acc_lo_3, svreinterpret_s16_u16(packed_lo_3), vw);
            acc_hi_0 = svdot_s64(acc_hi_0, svreinterpret_s16_u16(packed_hi_0), vw);
            acc_hi_1 = svdot_s64(acc_hi_1, svreinterpret_s16_u16(packed_hi_1), vw);
            acc_hi_2 = svdot_s64(acc_hi_2, svreinterpret_s16_u16(packed_hi_2), vw);
            acc_hi_3 = svdot_s64(acc_hi_3, svreinterpret_s16_u16(packed_hi_3), vw);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_s8(svtbl_s8(
                unsafe { svreinterpret_s8_s16(svld1_s16(pg2, w.as_ptr())) },
                shuf4,
            ));

            let base0 = src_stride * py + cx;
            let row0_lo = unsafe { svld1_u16(pg_full, src.get_unchecked(base0..).as_ptr()) };
            let row1_lo =
                unsafe { svld1_u16(pg_full, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row0_hi = unsafe { svld1_u16(pg_full, src.get_unchecked(base0 + vl..).as_ptr()) };
            let row1_hi = unsafe {
                svld1_u16(
                    pg_full,
                    src.get_unchecked(base0 + src_stride + vl..).as_ptr(),
                )
            };

            let zero = svdup_n_u16(0);
            let [packed_lo_0, packed_lo_1, packed_lo_2, packed_lo_3] =
                pack_4_rows_sve_u16!(row0_lo, row1_lo, zero, zero);
            let [packed_hi_0, packed_hi_1, packed_hi_2, packed_hi_3] =
                pack_4_rows_sve_u16!(row0_hi, row1_hi, zero, zero);

            acc_lo_0 = svdot_s64(acc_lo_0, svreinterpret_s16_u16(packed_lo_0), vw);
            acc_lo_1 = svdot_s64(acc_lo_1, svreinterpret_s16_u16(packed_lo_1), vw);
            acc_lo_2 = svdot_s64(acc_lo_2, svreinterpret_s16_u16(packed_lo_2), vw);
            acc_lo_3 = svdot_s64(acc_lo_3, svreinterpret_s16_u16(packed_lo_3), vw);
            acc_hi_0 = svdot_s64(acc_hi_0, svreinterpret_s16_u16(packed_hi_0), vw);
            acc_hi_1 = svdot_s64(acc_hi_1, svreinterpret_s16_u16(packed_hi_1), vw);
            acc_hi_2 = svdot_s64(acc_hi_2, svreinterpret_s16_u16(packed_hi_2), vw);
            acc_hi_3 = svdot_s64(acc_hi_3, svreinterpret_s16_u16(packed_hi_3), vw);

            j += 2;
        }

        while j < bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_s8(svtbl_s8(
                unsafe { svreinterpret_s8_s16(svld1_s16(pg1, w.as_ptr())) },
                shuf4,
            ));

            let base0 = src_stride * py + cx;
            let row_lo = unsafe { svld1_u16(pg_full, src.get_unchecked(base0..).as_ptr()) };
            let row_hi = unsafe { svld1_u16(pg_full, src.get_unchecked(base0 + vl..).as_ptr()) };
            let zero = svdup_n_u16(0);

            let [packed_lo_0, packed_lo_1, packed_lo_2, packed_lo_3] =
                pack_4_rows_sve_u16!(row_lo, zero, zero, zero);
            let [packed_hi_0, packed_hi_1, packed_hi_2, packed_hi_3] =
                pack_4_rows_sve_u16!(row_hi, zero, zero, zero);

            acc_lo_0 = svdot_s64(acc_lo_0, svreinterpret_s16_u16(packed_lo_0), vw);
            acc_lo_1 = svdot_s64(acc_lo_1, svreinterpret_s16_u16(packed_lo_1), vw);
            acc_lo_2 = svdot_s64(acc_lo_2, svreinterpret_s16_u16(packed_lo_2), vw);
            acc_lo_3 = svdot_s64(acc_lo_3, svreinterpret_s16_u16(packed_lo_3), vw);
            acc_hi_0 = svdot_s64(acc_hi_0, svreinterpret_s16_u16(packed_hi_0), vw);
            acc_hi_1 = svdot_s64(acc_hi_1, svreinterpret_s16_u16(packed_hi_1), vw);
            acc_hi_2 = svdot_s64(acc_hi_2, svreinterpret_s16_u16(packed_hi_2), vw);
            acc_hi_3 = svdot_s64(acc_hi_3, svreinterpret_s16_u16(packed_hi_3), vw);

            j += 1;
        }

        let mut result_lo = narrow_i64_to_u16!(acc_lo_0, acc_lo_1, acc_lo_2, acc_lo_3);
        let mut result_hi = narrow_i64_to_u16!(acc_hi_0, acc_hi_1, acc_hi_2, acc_hi_3);
        result_lo = svmin_u16_x(pg_full, result_lo, max_colors);
        result_hi = svmin_u16_x(pg_full, result_hi, max_colors);

        unsafe {
            svst1_u16(pg_full, dst.get_unchecked_mut(cx..).as_mut_ptr(), result_lo);
            svst1_u16(
                pg_full,
                dst.get_unchecked_mut(cx + vl..).as_mut_ptr(),
                result_hi,
            );
        }

        cx += vl * 2;
    }
    cx
}

#[target_feature(enable = "sve,sve2")]
fn convolve_vertical_sve2_u16_row(
    _: usize,
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
) {
    let vl = svcnth() as usize;
    let mut cx = work_wide_chunks_u16(bounds, src, dst, src_stride, weights, bit_depth);
    let len = dst.len();

    let pg8 = svwhilelt_b16_u32(0u32, 8u32);
    let pg4 = svwhilelt_b16_u32(0u32, 4u32);
    let pg2 = svwhilelt_b16_u32(0u32, 2u32);
    let pg1 = svwhilelt_b16_u32(0u32, 1u32);

    let shuf4 = svreinterpret_u8_s64(svdup_n_s64(i64::from_ne_bytes([0, 1, 2, 3, 4, 5, 6, 7])));
    let shuf4hi = svreinterpret_u8_s64(svdup_n_s64(i64::from_ne_bytes([
        8, 9, 10, 11, 12, 13, 14, 15,
    ])));

    let max_colors = svdup_n_u16(((1u32 << bit_depth) - 1) as u16);

    while cx < len {
        let pg = svwhilelt_b16_u64(cx as u64, len as u64);

        let rounding = svdup_n_s64(ROUNDING_CONST as i64);
        let mut acc_0 = rounding;
        let mut acc_1 = rounding;
        let mut acc_2 = rounding;
        let mut acc_3 = rounding;

        let mut j = 0usize;

        while j + 8 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };

            let wq = unsafe { svreinterpret_s8_s16(svld1_s16(pg8, w.as_ptr())) };
            let vw0 = svreinterpret_s16_s8(svtbl_s8(wq, shuf4));
            let vw1 = svreinterpret_s16_s8(svtbl_s8(wq, shuf4hi));

            let base0 = src_stride * py + cx;
            // Load 8 rows
            let row0 = unsafe { svld1_u16(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row2 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 2..).as_ptr()) };
            let row3 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 3..).as_ptr()) };
            let row4 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 4..).as_ptr()) };
            let row5 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 5..).as_ptr()) };
            let row6 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 6..).as_ptr()) };
            let row7 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 7..).as_ptr()) };

            let [packed0_0, packed1_0, packed2_0, packed3_0] =
                pack_4_rows_sve_u16!(row0, row1, row2, row3);
            let [packed0_1, packed1_1, packed2_1, packed3_1] =
                pack_4_rows_sve_u16!(row4, row5, row6, row7);

            acc_0 = svdot_s64(acc_0, svreinterpret_s16_u16(packed0_0), vw0);
            acc_1 = svdot_s64(acc_1, svreinterpret_s16_u16(packed1_0), vw0);
            acc_2 = svdot_s64(acc_2, svreinterpret_s16_u16(packed2_0), vw0);
            acc_3 = svdot_s64(acc_3, svreinterpret_s16_u16(packed3_0), vw0);

            acc_0 = svdot_s64(acc_0, svreinterpret_s16_u16(packed0_1), vw1);
            acc_1 = svdot_s64(acc_1, svreinterpret_s16_u16(packed1_1), vw1);
            acc_2 = svdot_s64(acc_2, svreinterpret_s16_u16(packed2_1), vw1);
            acc_3 = svdot_s64(acc_3, svreinterpret_s16_u16(packed3_1), vw1);

            j += 8;
        }

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_s8(svtbl_s8(
                unsafe { svreinterpret_s8_s16(svld1_s16(pg4, w.as_ptr())) },
                shuf4,
            ));

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u16(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row2 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 2..).as_ptr()) };
            let row3 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 3..).as_ptr()) };

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve_u16!(row0, row1, row2, row3);

            acc_0 = svdot_s64(acc_0, svreinterpret_s16_u16(packed0), vw);
            acc_1 = svdot_s64(acc_1, svreinterpret_s16_u16(packed1), vw);
            acc_2 = svdot_s64(acc_2, svreinterpret_s16_u16(packed2), vw);
            acc_3 = svdot_s64(acc_3, svreinterpret_s16_u16(packed3), vw);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_s8(svtbl_s8(
                unsafe { svreinterpret_s8_s16(svld1_s16(pg2, w.as_ptr())) },
                shuf4,
            ));

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u16(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let zero = svdup_n_u16(0);

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve_u16!(row0, row1, zero, zero);

            acc_0 = svdot_s64(acc_0, svreinterpret_s16_u16(packed0), vw);
            acc_1 = svdot_s64(acc_1, svreinterpret_s16_u16(packed1), vw);
            acc_2 = svdot_s64(acc_2, svreinterpret_s16_u16(packed2), vw);
            acc_3 = svdot_s64(acc_3, svreinterpret_s16_u16(packed3), vw);

            j += 2;
        }

        while j < bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_s8(svtbl_s8(
                unsafe { svreinterpret_s8_s16(svld1_s16(pg1, w.as_ptr())) },
                shuf4,
            ));

            let base0 = src_stride * py + cx;
            let row = unsafe { svld1_u16(pg, src.get_unchecked(base0..).as_ptr()) };
            let zero = svdup_n_u16(0);

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve_u16!(row, zero, zero, zero);

            acc_0 = svdot_s64(acc_0, svreinterpret_s16_u16(packed0), vw);
            acc_1 = svdot_s64(acc_1, svreinterpret_s16_u16(packed1), vw);
            acc_2 = svdot_s64(acc_2, svreinterpret_s16_u16(packed2), vw);
            acc_3 = svdot_s64(acc_3, svreinterpret_s16_u16(packed3), vw);

            j += 1;
        }

        let mut result = narrow_i64_to_u16!(acc_0, acc_1, acc_2, acc_3);
        result = svmin_u16_x(pg, result, max_colors);

        unsafe {
            svst1_u16(pg, dst.get_unchecked_mut(cx..).as_mut_ptr(), result);
        }

        cx += vl;
    }
}
