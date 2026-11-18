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
    ([$lo0:ident, $lo1:ident, $hi0:ident, $hi1:ident], $a:expr, $b:expr, $c:expr, $d:expr) => {
        let ab_lo = svtrn1_u16($a, $b);
        let ab_hi = svtrn2_u16($a, $b);

        let cd_lo = svtrn1_u16($c, $d);
        let cd_hi = svtrn2_u16($c, $d);

        let lo0 = svtrn1_u32(svreinterpret_u32_u16(ab_lo), svreinterpret_u32_u16(cd_lo));
        let lo1 = svtrn2_u32(svreinterpret_u32_u16(ab_lo), svreinterpret_u32_u16(cd_lo));
        let hi0 = svtrn1_u32(svreinterpret_u32_u16(ab_hi), svreinterpret_u32_u16(cd_hi));
        let hi1 = svtrn2_u32(svreinterpret_u32_u16(ab_hi), svreinterpret_u32_u16(cd_hi));

        // The four outputs contain columns congruent to [0, 2, 1, 3] modulo four.
        // Keep each scalable vector in its own local to preserve all lanes.
        let $lo0 = svreinterpret_u16_u32(lo0);
        let $lo1 = svreinterpret_u16_u32(lo1);
        let $hi0 = svreinterpret_u16_u32(hi0);
        let $hi1 = svreinterpret_u16_u32(hi1);
    };
}

macro_rules! narrow_i64_to_u16 {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {{
        let q0 = svqshrunb_n_s64::<{ crate::support::PRECISION }>($a);
        let q01 = svqshrunt_n_s64::<{ crate::support::PRECISION }>(q0, $b);
        let q2 = svqshrunb_n_s64::<{ crate::support::PRECISION }>($c);
        let q23 = svqshrunt_n_s64::<{ crate::support::PRECISION }>(q2, $d);
        // Interleave the even and odd columns, saturating before narrowing to u16.
        svqxtnt_u32(svqxtnb_u32(q01), q23)
    }};
}

#[target_feature(enable = "sve,sve2")]
fn work_wide_chunks_u16<const BOUNDS: usize>(
    bounds: &FilterBounds,
    src: &[u16],
    dst: &mut [u16],
    src_stride: usize,
    weights: &[i16],
    bit_depth: u32,
) -> usize {
    let bounds_size = if BOUNDS == 0 { bounds.size } else { BOUNDS };
    let vl = svcnth() as usize;
    let mut cx = 0usize;
    let len = dst.len();

    let pg_full = svptrue_b16();

    let max_colors = svdup_n_u16(((1u32 << bit_depth) - 1) as u16);

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

        while j + 8 <= bounds_size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };

            let vw0 = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().cast::<u64>().read_unaligned()
            }));
            let vw1 = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().add(4).cast::<u64>().read_unaligned()
            }));

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

            pack_4_rows_sve_u16!(
                [packed_lo_0, packed_lo_1, packed_lo_2, packed_lo_3],
                row0_lo,
                row1_lo,
                row2_lo,
                row3_lo
            );
            pack_4_rows_sve_u16!(
                [packed_hi_0, packed_hi_1, packed_hi_2, packed_hi_3],
                row0_hi,
                row1_hi,
                row2_hi,
                row3_hi
            );
            pack_4_rows_sve_u16!(
                [packed_lo_4, packed_lo_5, packed_lo_6, packed_lo_7],
                row4_lo,
                row5_lo,
                row6_lo,
                row7_lo
            );
            pack_4_rows_sve_u16!(
                [packed_hi_4, packed_hi_5, packed_hi_6, packed_hi_7],
                row4_hi,
                row5_hi,
                row6_hi,
                row7_hi
            );

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

        while j + 4 <= bounds_size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().cast::<u64>().read_unaligned()
            }));

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

            pack_4_rows_sve_u16!(
                [packed_lo_0, packed_lo_1, packed_lo_2, packed_lo_3],
                row0_lo,
                row1_lo,
                row2_lo,
                row3_lo
            );
            pack_4_rows_sve_u16!(
                [packed_hi_0, packed_hi_1, packed_hi_2, packed_hi_3],
                row0_hi,
                row1_hi,
                row2_hi,
                row3_hi
            );

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

        while j + 2 <= bounds_size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().cast::<u32>().read_unaligned() as u64
            }));

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
            pack_4_rows_sve_u16!(
                [packed_lo_0, packed_lo_1, packed_lo_2, packed_lo_3],
                row0_lo,
                row1_lo,
                zero,
                zero
            );
            pack_4_rows_sve_u16!(
                [packed_hi_0, packed_hi_1, packed_hi_2, packed_hi_3],
                row0_hi,
                row1_hi,
                zero,
                zero
            );

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

        while j < bounds_size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().cast::<u16>().read_unaligned() as u64
            }));

            let base0 = src_stride * py + cx;
            let row_lo = unsafe { svld1_u16(pg_full, src.get_unchecked(base0..).as_ptr()) };
            let row_hi = unsafe { svld1_u16(pg_full, src.get_unchecked(base0 + vl..).as_ptr()) };
            let zero = svdup_n_u16(0);

            pack_4_rows_sve_u16!(
                [packed_lo_0, packed_lo_1, packed_lo_2, packed_lo_3],
                row_lo,
                zero,
                zero,
                zero
            );
            pack_4_rows_sve_u16!(
                [packed_hi_0, packed_hi_1, packed_hi_2, packed_hi_3],
                row_hi,
                zero,
                zero,
                zero
            );

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
    // Unroll 12 taps so coefficients and row addresses can stay outside the column loop.
    let mut cx = match bounds.size {
        12 => work_wide_chunks_u16::<12>(bounds, src, dst, src_stride, weights, bit_depth),
        _ => work_wide_chunks_u16::<0>(bounds, src, dst, src_stride, weights, bit_depth),
    };
    let len = dst.len();

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

            let vw0 = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().cast::<u64>().read_unaligned()
            }));
            let vw1 = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().add(4).cast::<u64>().read_unaligned()
            }));

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

            pack_4_rows_sve_u16!(
                [packed0_0, packed1_0, packed2_0, packed3_0],
                row0,
                row1,
                row2,
                row3
            );
            pack_4_rows_sve_u16!(
                [packed0_1, packed1_1, packed2_1, packed3_1],
                row4,
                row5,
                row6,
                row7
            );

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
            let vw = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().cast::<u64>().read_unaligned()
            }));

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u16(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row2 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 2..).as_ptr()) };
            let row3 =
                unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride * 3..).as_ptr()) };

            pack_4_rows_sve_u16!([packed0, packed1, packed2, packed3], row0, row1, row2, row3);

            acc_0 = svdot_s64(acc_0, svreinterpret_s16_u16(packed0), vw);
            acc_1 = svdot_s64(acc_1, svreinterpret_s16_u16(packed1), vw);
            acc_2 = svdot_s64(acc_2, svreinterpret_s16_u16(packed2), vw);
            acc_3 = svdot_s64(acc_3, svreinterpret_s16_u16(packed3), vw);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().cast::<u32>().read_unaligned() as u64
            }));

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u16(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u16(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let zero = svdup_n_u16(0);

            pack_4_rows_sve_u16!([packed0, packed1, packed2, packed3], row0, row1, zero, zero);

            acc_0 = svdot_s64(acc_0, svreinterpret_s16_u16(packed0), vw);
            acc_1 = svdot_s64(acc_1, svreinterpret_s16_u16(packed1), vw);
            acc_2 = svdot_s64(acc_2, svreinterpret_s16_u16(packed2), vw);
            acc_3 = svdot_s64(acc_3, svreinterpret_s16_u16(packed3), vw);

            j += 2;
        }

        while j < bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s16_u64(svdup_n_u64(unsafe {
                w.as_ptr().cast::<u16>().read_unaligned() as u64
            }));

            let base0 = src_stride * py + cx;
            let row = unsafe { svld1_u16(pg, src.get_unchecked(base0..).as_ptr()) };
            let zero = svdup_n_u16(0);

            pack_4_rows_sve_u16!([packed0, packed1, packed2, packed3], row, zero, zero, zero);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point_vertical::column_handler_fixed_point;

    #[test]
    fn sve2_vertical_rounds_once() {
        let Some(bytes) = crate::sve2::test_vector_length(false) else {
            return;
        };
        let vl = bytes / 2;
        for width in [1, vl, 2 * vl, 2 * vl + 1] {
            // 2 * 0.5 is exactly 1; adding two rounding biases incorrectly gives 2.
            let src = vec![2; width];
            let mut dst = vec![0; width];
            convolve_vertical_sve2_u16_dot(
                width,
                &FilterBounds::new(0, 1),
                &src,
                &mut dst,
                width,
                &[16384],
                15,
            );
            assert!(dst.iter().all(|&v| v == 1), "width={width}, dst={dst:?}");
        }
    }

    #[test]
    fn sve2_vertical_u16_saturates_before_removing_upper_halfwords() {
        let Some(bytes) = crate::sve2::test_vector_length(false) else {
            return;
        };
        let vl = bytes / 2;
        for depth in [13, 14, 15] {
            let max = ((1u32 << depth) - 1) as u16;
            let full = 1usize << (16 - depth);
            let mut weights = vec![i16::MAX; full + 1];
            weights.extend(std::iter::repeat_n(i16::MIN, full - 1));
            weights.push(full as i16 - i16::MAX);
            assert_eq!(weights.iter().map(|&w| i32::from(w)).sum::<i32>(), 32768);
            let tap_counts = [weights.len(), weights.len().max(12)];
            for taps in tap_counts {
                weights.resize(taps, 0);
                for width in [
                    1,
                    vl - 1,
                    vl,
                    vl + 1,
                    2 * vl - 1,
                    2 * vl,
                    2 * vl + 1,
                    4 * vl + 3,
                ] {
                    for padding in [0, 5] {
                        let stride = width + padding;
                        let start = 2;
                        let mut src = vec![0; stride * (start + taps - 1) + width];
                        for row in 0..full {
                            src[(start + row) * stride..][..width].fill(max);
                        }
                        for x in 0..width {
                            src[(start + full) * stride + x] = (full + 2 + x % 5) as u16;
                        }
                        let expected: Vec<_> = (0..width)
                            .map(|x| {
                                let sum: i64 = weights
                                    .iter()
                                    .enumerate()
                                    .map(|(j, &w)| {
                                        i64::from(src[(start + j) * stride + x]) * i64::from(w)
                                    })
                                    .sum();
                                let rounded = (sum + (1 << 14)) >> 15;
                                assert!(rounded >= 65536, "exercise narrowing overflow");
                                rounded.clamp(0, i64::from(max)) as u16
                            })
                            .collect();
                        let mut actual = vec![0xbeef; width + 2];
                        convolve_vertical_sve2_u16_dot(
                            width,
                            &FilterBounds::new(start, taps),
                            &src,
                            &mut actual[1..width + 1],
                            stride,
                            &weights,
                            depth,
                        );
                        assert_eq!(
                            &actual[1..width + 1],
                            expected,
                            "VL={}, depth={depth}, taps={taps}, width={width}, padding={padding}",
                            bytes * 8
                        );
                        assert_eq!(actual[0], 0xbeef);
                        assert_eq!(actual[width + 1], 0xbeef);
                    }
                }
            }
        }
    }

    #[test]
    fn sve2_vertical_matches_scalar_reference() {
        let Some(bytes) = crate::sve2::test_vector_length(false) else {
            return;
        };
        let vl = bytes / 2;
        for bit_depth in [13, 14, 15] {
            let max = (1u32 << bit_depth) - 1;
            for width in [
                0,
                1,
                vl - 1,
                vl,
                vl + 1,
                2 * vl - 1,
                2 * vl,
                2 * vl + 1,
                4 * vl + 3,
            ] {
                let stride = width + 7;
                for taps in 1..=17 {
                    let bounds = FilterBounds::new(2, taps);
                    let src: Vec<_> = (0..stride * (taps + 2))
                        .map(|i| match i % 7 {
                            0 => 0,
                            1 => max as u16,
                            _ => ((i as u32 * 7919 + 1237) & max) as u16,
                        })
                        .collect();
                    let mut weights: Vec<_> = (0..taps)
                        .map(|j| ((j * 997 + taps * 53) % 4096) as i16 - 2048)
                        .collect();
                    weights[0] = 30000;
                    let mut expected = vec![0xbeef; width + 2];
                    let mut actual = expected.clone();
                    column_handler_fixed_point::<u16, i32>(
                        width,
                        &bounds,
                        &src,
                        &mut expected[1..width + 1],
                        stride,
                        &weights,
                        bit_depth,
                    );
                    convolve_vertical_sve2_u16_dot(
                        width,
                        &bounds,
                        &src,
                        &mut actual[1..width + 1],
                        stride,
                        &weights,
                        bit_depth,
                    );
                    for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            actual,
                            expected,
                            "VL={}, depth={bit_depth}, width={width}, taps={taps}, index={i}",
                            bytes * 8,
                        );
                    }
                }
            }
        }
    }
}
