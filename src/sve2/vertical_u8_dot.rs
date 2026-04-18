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
use std::arch::aarch64::*;

const SCALE: i32 = 7;
const ROUNDING: i32 = 1 << (SCALE - 1);

/// Vertical convolution using SVE2 i8mm.
///
/// Requires SVE2 + i8mm feature flags. Uses scalable vectors so the main loop
/// handles any width without a scalar tail — the final iteration is simply
/// predicated to the remaining lanes.
///
/// # Safety
/// - Caller must verify `sve2` and `i8mm` CPU features before calling.
pub(crate) fn convolve_vertical_sve2_i8_dot(
    width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i8],
    _: u32,
) {
    unsafe {
        convolve_vertical_sve2_row(width, bounds, src, dst, src_stride, weight);
    }
}

macro_rules! pack_4_rows_sve {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {{
        let ab_lo = svzip1_u8($a, $b);
        let ab_hi = svzip2_u8($a, $b);

        let cd_lo = svzip1_u8($c, $d);
        let cd_hi = svzip2_u8($c, $d);

        let lo0 = svreinterpret_u8_u16(svzip1_u16(
            svreinterpret_u16_u8(ab_lo),
            svreinterpret_u16_u8(cd_lo),
        ));
        let lo1 = svreinterpret_u8_u16(svzip2_u16(
            svreinterpret_u16_u8(ab_lo),
            svreinterpret_u16_u8(cd_lo),
        ));
        let hi0 = svreinterpret_u8_u16(svzip1_u16(
            svreinterpret_u16_u8(ab_hi),
            svreinterpret_u16_u8(cd_hi),
        ));
        let hi1 = svreinterpret_u8_u16(svzip2_u16(
            svreinterpret_u16_u8(ab_hi),
            svreinterpret_u16_u8(cd_hi),
        ));

        [lo0, lo1, hi0, hi1]
    }};
}

#[target_feature(enable = "sve,sve2,i8mm")]
fn work_32_chunks(
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i8],
) -> usize {
    let vl = svcntb() as usize;

    let mut cx = 0usize;

    let len = dst.len();

    let pg4 = svwhilelt_b8_u32(0u32, 4u32);
    let pg2 = svwhilelt_b8_u32(0u32, 2u32);
    let pg1 = svwhilelt_b8_u32(0u32, 1u32);

    let shuf4 = svreinterpret_u8_s32(svdup_n_s32(i32::from_ne_bytes([0, 1, 2, 3])));

    let pg_full = svptrue_b8();

    while cx + vl * 2 <= len {
        let rounding = svdup_n_s32(ROUNDING);
        let mut acc_0 = rounding;
        let mut acc_1 = rounding;
        let mut acc_2 = rounding;
        let mut acc_3 = rounding;
        let mut acc_4 = rounding;
        let mut acc_5 = rounding;
        let mut acc_6 = rounding;
        let mut acc_7 = rounding;

        let mut j = 0usize;

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svtbl_s8(unsafe { svld1_s8(pg4, w.as_ptr()) }, shuf4);

            let base0 = src_stride * py + cx;
            let row0_lo = unsafe { svld1_u8(pg_full, src.get_unchecked(base0..).as_ptr()) };
            let row1_lo =
                unsafe { svld1_u8(pg_full, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row2_lo = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 2..).as_ptr(),
                )
            };
            let row3_lo = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 3..).as_ptr(),
                )
            };

            let row0_hi = unsafe { svld1_u8(pg_full, src.get_unchecked(base0 + vl..).as_ptr()) };
            let row1_hi = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base0 + src_stride + vl..).as_ptr(),
                )
            };
            let row2_hi = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 2 + vl..).as_ptr(),
                )
            };
            let row3_hi = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base0 + src_stride * 3 + vl..).as_ptr(),
                )
            };

            let [packed0, packed1, packed2, packed3] =
                pack_4_rows_sve!(row0_lo, row1_lo, row2_lo, row3_lo);
            let [packed4, packed5, packed6, packed7] =
                pack_4_rows_sve!(row0_hi, row1_hi, row2_hi, row3_hi);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);
            acc_4 = svusdot_s32(acc_4, packed4, vw);
            acc_5 = svusdot_s32(acc_5, packed5, vw);
            acc_6 = svusdot_s32(acc_6, packed6, vw);
            acc_7 = svusdot_s32(acc_7, packed7, vw);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svtbl_s8(unsafe { svld1_s8(pg2, w.as_ptr()) }, shuf4);

            let base0 = src_stride * py + cx;
            let row0_lo = unsafe { svld1_u8(pg_full, src.get_unchecked(base0..).as_ptr()) };
            let row1_lo =
                unsafe { svld1_u8(pg_full, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row0_hi = unsafe { svld1_u8(pg_full, src.get_unchecked(base0 + vl..).as_ptr()) };
            let row1_hi = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base0 + src_stride + vl..).as_ptr(),
                )
            };
            let zero = svdup_n_u8(0);

            let [packed0, packed1, packed2, packed3] =
                pack_4_rows_sve!(row0_lo, row1_lo, zero, zero);
            let [packed4, packed5, packed6, packed7] =
                pack_4_rows_sve!(row0_hi, row1_hi, zero, zero);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);
            acc_4 = svusdot_s32(acc_4, packed4, vw);
            acc_5 = svusdot_s32(acc_5, packed5, vw);
            acc_6 = svusdot_s32(acc_6, packed6, vw);
            acc_7 = svusdot_s32(acc_7, packed7, vw);

            j += 2;
        }

        while j < bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j) };
            let vw = svtbl_s8(unsafe { svld1_s8(pg1, w) }, shuf4);

            let base0 = src_stride * py + cx;
            let row_lo = unsafe { svld1_u8(pg_full, src.get_unchecked(base0..).as_ptr()) };
            let row_hi = unsafe { svld1_u8(pg_full, src.get_unchecked(base0 + vl..).as_ptr()) };
            let zero = svdup_n_u8(0);

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve!(row_lo, zero, zero, zero);
            let [packed4, packed5, packed6, packed7] = pack_4_rows_sve!(row_hi, zero, zero, zero);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);
            acc_4 = svusdot_s32(acc_4, packed4, vw);
            acc_5 = svusdot_s32(acc_5, packed5, vw);
            acc_6 = svusdot_s32(acc_6, packed6, vw);
            acc_7 = svusdot_s32(acc_7, packed7, vw);

            j += 1;
        }

        let n0 = svqshrunb_n_s32::<SCALE>(acc_0);
        let n1 = svqshrunb_n_s32::<SCALE>(acc_1);
        let n2 = svqshrunb_n_s32::<SCALE>(acc_2);
        let n3 = svqshrunb_n_s32::<SCALE>(acc_3);
        let result_lo = svuzp1_u8(
            svqxtnb_u16(svuzp1_u16(n0, n1)),
            svqxtnb_u16(svuzp1_u16(n2, n3)),
        );
        unsafe { svst1_u8(pg_full, dst.get_unchecked_mut(cx..).as_mut_ptr(), result_lo) };

        let n4 = svqshrunb_n_s32::<SCALE>(acc_4);
        let n5 = svqshrunb_n_s32::<SCALE>(acc_5);
        let n6 = svqshrunb_n_s32::<SCALE>(acc_6);
        let n7 = svqshrunb_n_s32::<SCALE>(acc_7);
        let result_hi = svuzp1_u8(
            svqxtnb_u16(svuzp1_u16(n4, n5)),
            svqxtnb_u16(svuzp1_u16(n6, n7)),
        );
        unsafe {
            svst1_u8(
                pg_full,
                dst.get_unchecked_mut(cx + vl..).as_mut_ptr(),
                result_hi,
            )
        };

        cx += vl * 2;
    }
    cx
}

#[target_feature(enable = "sve,sve2,i8mm")]
fn convolve_vertical_sve2_row(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i8],
) {
    let vl = svcntb() as usize;

    let mut cx = work_32_chunks(bounds, src, dst, src_stride, weights);

    let len = dst.len();

    let pg4 = svwhilelt_b8_u32(0u32, 4u32);
    let pg2 = svwhilelt_b8_u32(0u32, 2u32);
    let pg1 = svwhilelt_b8_u32(0u32, 1u32);

    let shuf4 = svreinterpret_u8_s32(svdup_n_s32(i32::from_ne_bytes([0, 1, 2, 3])));

    while cx < dst.len() {
        let pg = svwhilelt_b8_u64(cx as u64, len as u64);

        let rounding = svdup_n_s32(ROUNDING);
        let mut acc_0 = rounding;
        let mut acc_1 = rounding;
        let mut acc_2 = rounding;
        let mut acc_3 = rounding;

        let mut j = 0usize;

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };

            let vw = svtbl_s8(unsafe { svld1_s8(pg4, w.as_ptr()) }, shuf4);

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u8(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row2 =
                unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride * 2..).as_ptr()) };
            let row3 =
                unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride * 3..).as_ptr()) };

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve!(row0, row1, row2, row3);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };

            let vw = svtbl_s8(unsafe { svld1_s8(pg2, w.as_ptr()) }, shuf4);

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u8(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let zero = svdup_n_u8(0);

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve!(row0, row1, zero, zero);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);

            j += 2;
        }

        while j < bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j) };
            let vw = svtbl_s8(unsafe { svld1_s8(pg1, w) }, shuf4);

            let base0 = src_stride * py + cx;
            let row = unsafe { svld1_u8(pg, src.get_unchecked(base0..).as_ptr()) };

            let zero = svdup_n_u8(0);

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve!(row, zero, zero, zero);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);

            j += 1;
        }

        let n0 = svqshrunb_n_s32::<SCALE>(acc_0);
        let n1 = svqshrunb_n_s32::<SCALE>(acc_1);
        let n2 = svqshrunb_n_s32::<SCALE>(acc_2);
        let n3 = svqshrunb_n_s32::<SCALE>(acc_3);

        let s01_packed = svuzp1_u16(n0, n1);
        let s23_packed = svuzp1_u16(n2, n3);

        let b01 = svqxtnb_u16(s01_packed);
        let b23 = svqxtnb_u16(s23_packed);

        let result = svuzp1_u8(b01, b23);

        unsafe {
            svst1_u8(pg, dst.get_unchecked_mut(cx..).as_mut_ptr(), result);
        }

        cx += vl;
    }
}
