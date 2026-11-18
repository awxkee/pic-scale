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
    ([$lo0:ident, $lo1:ident, $hi0:ident, $hi1:ident],
     $a:expr, $b:expr, $c:expr, $d:expr) => {
        // Bind scalable vectors separately: arrays use fixed-size stack slots in debug builds.
        let ab_lo = svtrn1_u8($a, $b);
        let ab_hi = svtrn2_u8($a, $b);

        let cd_lo = svtrn1_u8($c, $d);
        let cd_hi = svtrn2_u8($c, $d);

        // Each dot group is [a, b, c, d], with columns ordered modulo four:
        // lo0: 0,4,...; lo1: 2,6,...; hi0: 1,5,...; hi1: 3,7,... .
        // This order lets the bottom/top narrows interleave the final bytes.
        let $lo0 = svreinterpret_u8_u16(svtrn1_u16(
            svreinterpret_u16_u8(ab_lo),
            svreinterpret_u16_u8(cd_lo),
        ));
        let $lo1 = svreinterpret_u8_u16(svtrn2_u16(
            svreinterpret_u16_u8(ab_lo),
            svreinterpret_u16_u8(cd_lo),
        ));
        let $hi0 = svreinterpret_u8_u16(svtrn1_u16(
            svreinterpret_u16_u8(ab_hi),
            svreinterpret_u16_u8(cd_hi),
        ));
        let $hi1 = svreinterpret_u8_u16(svtrn2_u16(
            svreinterpret_u16_u8(ab_hi),
            svreinterpret_u16_u8(cd_hi),
        ));
    };
}

macro_rules! pack_8_rows_sve {
    ([$lo0:ident, $lo1:ident, $hi0:ident, $hi1:ident,
      $lo4:ident, $lo5:ident, $hi4:ident, $hi5:ident],
     $a:expr, $b:expr, $c:expr, $d:expr,
     $e:expr, $f:expr, $g:expr, $h:expr) => {
        // Bind scalable vectors separately: arrays use fixed-size stack slots in debug builds.
        let ab_lo = svtrn1_u8($a, $b);
        let ef_lo = svtrn1_u8($e, $f);

        let ab_hi = svtrn2_u8($a, $b);
        let ef_hi = svtrn2_u8($e, $f);

        let cd_lo = svtrn1_u8($c, $d);
        let gh_lo = svtrn1_u8($g, $h);

        let cd_hi = svtrn2_u8($c, $d);
        let gh_hi = svtrn2_u8($g, $h);

        let $lo0 = svreinterpret_u8_u16(svtrn1_u16(
            svreinterpret_u16_u8(ab_lo),
            svreinterpret_u16_u8(cd_lo),
        ));
        let $lo4 = svreinterpret_u8_u16(svtrn1_u16(
            svreinterpret_u16_u8(ef_lo),
            svreinterpret_u16_u8(gh_lo),
        ));

        let $lo1 = svreinterpret_u8_u16(svtrn2_u16(
            svreinterpret_u16_u8(ab_lo),
            svreinterpret_u16_u8(cd_lo),
        ));
        let $lo5 = svreinterpret_u8_u16(svtrn2_u16(
            svreinterpret_u16_u8(ef_lo),
            svreinterpret_u16_u8(gh_lo),
        ));

        let $hi0 = svreinterpret_u8_u16(svtrn1_u16(
            svreinterpret_u16_u8(ab_hi),
            svreinterpret_u16_u8(cd_hi),
        ));
        let $hi4 = svreinterpret_u8_u16(svtrn1_u16(
            svreinterpret_u16_u8(ef_hi),
            svreinterpret_u16_u8(gh_hi),
        ));

        let $hi1 = svreinterpret_u8_u16(svtrn2_u16(
            svreinterpret_u16_u8(ab_hi),
            svreinterpret_u16_u8(cd_hi),
        ));
        let $hi5 = svreinterpret_u8_u16(svtrn2_u16(
            svreinterpret_u16_u8(ef_hi),
            svreinterpret_u16_u8(gh_hi),
        ));

    };
}

#[target_feature(enable = "sve,sve2,i8mm")]
fn work_32_chunks<const BOUNDS: usize>(
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i8],
) -> usize {
    let bounds_size = if BOUNDS == 0 { bounds.size } else { BOUNDS };
    let vl = svcntb() as usize;

    let mut cx = 0usize;

    let len = dst.len();

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

        while j + 8 <= bounds_size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw0 = svreinterpret_s8_s32(svdup_n_s32(unsafe {
                w.as_ptr().cast::<i32>().read_unaligned()
            }));
            let vw1 = svreinterpret_s8_s32(svdup_n_s32(unsafe {
                w.as_ptr().add(4).cast::<i32>().read_unaligned()
            }));

            let base0 = src_stride * py + cx;
            let base1 = src_stride * (py + 4) + cx;

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

            pack_8_rows_sve!(
                [
                    packed0, packed1, packed2, packed3, packed4, packed5, packed6, packed7
                ],
                row0_lo,
                row1_lo,
                row2_lo,
                row3_lo,
                row0_hi,
                row1_hi,
                row2_hi,
                row3_hi
            );

            acc_0 = svusdot_s32(acc_0, packed0, vw0);
            acc_1 = svusdot_s32(acc_1, packed1, vw0);
            acc_2 = svusdot_s32(acc_2, packed2, vw0);
            acc_3 = svusdot_s32(acc_3, packed3, vw0);
            acc_4 = svusdot_s32(acc_4, packed4, vw0);
            acc_5 = svusdot_s32(acc_5, packed5, vw0);
            acc_6 = svusdot_s32(acc_6, packed6, vw0);
            acc_7 = svusdot_s32(acc_7, packed7, vw0);

            let row0_lo = unsafe { svld1_u8(pg_full, src.get_unchecked(base1..).as_ptr()) };
            let row1_lo =
                unsafe { svld1_u8(pg_full, src.get_unchecked(base1 + src_stride..).as_ptr()) };
            let row2_lo = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base1 + src_stride * 2..).as_ptr(),
                )
            };
            let row3_lo = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base1 + src_stride * 3..).as_ptr(),
                )
            };
            let row0_hi = unsafe { svld1_u8(pg_full, src.get_unchecked(base1 + vl..).as_ptr()) };
            let row1_hi = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base1 + src_stride + vl..).as_ptr(),
                )
            };
            let row2_hi = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base1 + src_stride * 2 + vl..).as_ptr(),
                )
            };
            let row3_hi = unsafe {
                svld1_u8(
                    pg_full,
                    src.get_unchecked(base1 + src_stride * 3 + vl..).as_ptr(),
                )
            };

            pack_8_rows_sve!(
                [
                    packed0, packed1, packed2, packed3, packed4, packed5, packed6, packed7
                ],
                row0_lo,
                row1_lo,
                row2_lo,
                row3_lo,
                row0_hi,
                row1_hi,
                row2_hi,
                row3_hi
            );

            acc_0 = svusdot_s32(acc_0, packed0, vw1);
            acc_1 = svusdot_s32(acc_1, packed1, vw1);
            acc_2 = svusdot_s32(acc_2, packed2, vw1);
            acc_3 = svusdot_s32(acc_3, packed3, vw1);
            acc_4 = svusdot_s32(acc_4, packed4, vw1);
            acc_5 = svusdot_s32(acc_5, packed5, vw1);
            acc_6 = svusdot_s32(acc_6, packed6, vw1);
            acc_7 = svusdot_s32(acc_7, packed7, vw1);

            j += 8;
        }

        while j + 4 <= bounds_size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s8_s32(svdup_n_s32(unsafe {
                w.as_ptr().cast::<i32>().read_unaligned()
            }));

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

            pack_8_rows_sve!(
                [
                    packed0, packed1, packed2, packed3, packed4, packed5, packed6, packed7
                ],
                row0_lo,
                row1_lo,
                row2_lo,
                row3_lo,
                row0_hi,
                row1_hi,
                row2_hi,
                row3_hi
            );

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

        while j + 2 <= bounds_size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s8_s32(svdup_n_s32(unsafe {
                w.as_ptr().cast::<u16>().read_unaligned() as i32
            }));

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

            pack_4_rows_sve!(
                [packed0, packed1, packed2, packed3],
                row0_lo,
                row1_lo,
                zero,
                zero
            );
            pack_4_rows_sve!(
                [packed4, packed5, packed6, packed7],
                row0_hi,
                row1_hi,
                zero,
                zero
            );

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

        while j < bounds_size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j) };
            let vw = svreinterpret_s8_s32(svdup_n_s32(*w as u8 as i32));

            let base0 = src_stride * py + cx;
            let row_lo = unsafe { svld1_u8(pg_full, src.get_unchecked(base0..).as_ptr()) };
            let row_hi = unsafe { svld1_u8(pg_full, src.get_unchecked(base0 + vl..).as_ptr()) };
            let zero = svdup_n_u8(0);

            pack_4_rows_sve!(
                [packed0, packed1, packed2, packed3],
                row_lo,
                zero,
                zero,
                zero
            );
            pack_4_rows_sve!(
                [packed4, packed5, packed6, packed7],
                row_hi,
                zero,
                zero,
                zero
            );

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

        let even = svqshrunt_n_s32::<SCALE>(svqshrunb_n_s32::<SCALE>(acc_0), acc_1);
        let odd = svqshrunt_n_s32::<SCALE>(svqshrunb_n_s32::<SCALE>(acc_2), acc_3);
        let result_lo = svqxtnt_u16(svqxtnb_u16(even), odd);
        unsafe { svst1_u8(pg_full, dst.get_unchecked_mut(cx..).as_mut_ptr(), result_lo) };

        let even = svqshrunt_n_s32::<SCALE>(svqshrunb_n_s32::<SCALE>(acc_4), acc_5);
        let odd = svqshrunt_n_s32::<SCALE>(svqshrunb_n_s32::<SCALE>(acc_6), acc_7);
        let result_hi = svqxtnt_u16(svqxtnb_u16(even), odd);
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

    // Unroll 12 taps so coefficients and row addresses can stay outside the column loop.
    // Larger specializations increase register pressure on 128-bit SVE machines.
    let mut cx = match bounds.size {
        12 => work_32_chunks::<12>(bounds, src, dst, src_stride, weights),
        _ => work_32_chunks::<0>(bounds, src, dst, src_stride, weights),
    };

    let len = dst.len();

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
            let vw = svreinterpret_s8_s32(svdup_n_s32(unsafe {
                w.as_ptr().cast::<i32>().read_unaligned()
            }));

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u8(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row2 =
                unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride * 2..).as_ptr()) };
            let row3 =
                unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride * 3..).as_ptr()) };

            pack_4_rows_sve!([packed0, packed1, packed2, packed3], row0, row1, row2, row3);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };
            let vw = svreinterpret_s8_s32(svdup_n_s32(unsafe {
                w.as_ptr().cast::<u16>().read_unaligned() as i32
            }));

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u8(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };

            let zero = svdup_n_u8(0);

            pack_4_rows_sve!([packed0, packed1, packed2, packed3], row0, row1, zero, zero);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);

            j += 2;
        }

        while j < bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j) };
            let vw = svreinterpret_s8_s32(svdup_n_s32(*w as u8 as i32));

            let base0 = src_stride * py + cx;
            let row = unsafe { svld1_u8(pg, src.get_unchecked(base0..).as_ptr()) };

            let zero = svdup_n_u8(0);

            pack_4_rows_sve!([packed0, packed1, packed2, packed3], row, zero, zero, zero);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);

            j += 1;
        }

        let even = svqshrunt_n_s32::<SCALE>(svqshrunb_n_s32::<SCALE>(acc_0), acc_1);
        let odd = svqshrunt_n_s32::<SCALE>(svqshrunb_n_s32::<SCALE>(acc_2), acc_3);
        let result = svqxtnt_u16(svqxtnb_u16(even), odd);

        unsafe {
            svst1_u8(pg, dst.get_unchecked_mut(cx..).as_mut_ptr(), result);
        }

        cx += vl;
    }
}

#[cfg(test)]
mod tests {
    use super::convolve_vertical_sve2_i8_dot;
    use crate::filter_weights::FilterBounds;

    #[test]
    fn sve2_vertical_u8_matches_scalar_reference() {
        let Some(vl) = crate::sve2::test_vector_length(true) else {
            return;
        };
        let mut widths = vec![
            0,
            1,
            vl / 4 - 1,
            vl / 2 - 1,
            vl / 2 + 1,
            vl - 1,
            vl,
            vl + 1,
            vl * 2 - 1,
            vl * 2,
            vl * 2 + 1,
            vl * 3 + 3,
            vl * 5 + 7,
        ];
        widths.sort_unstable();
        widths.dedup();
        let mut cases = 0;
        for width in widths {
            for padding in [0, 5] {
                let stride = width + padding;
                for taps in [
                    0usize, 1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 48, 63, 64, 127,
                ] {
                    let start = 2;
                    let rows = (start + taps).max(1);
                    let mut src = vec![0xBC; stride * (rows - 1) + width];
                    for y in 0..rows {
                        for x in 0..width {
                            src[y * stride + x] = (x * 97 + y * 53 + (x ^ (y * 17)) * 13) as u8;
                        }
                    }
                    for mode in 0..4 {
                        let mut weights: Vec<i8> = (0..taps)
                            .map(|i| match mode {
                                0 => ((i * 43 + 19) % 65) as i8 - 32,
                                1 => {
                                    if i % 2 == 0 {
                                        -128
                                    } else {
                                        127
                                    }
                                }
                                2 => 127,
                                _ => -128,
                            })
                            .collect();
                        if mode == 0 {
                            let mut residual =
                                128 - weights.iter().map(|&w| i32::from(w)).sum::<i32>();
                            for w in &mut weights {
                                let delta =
                                    residual.clamp(-128 - i32::from(*w), 127 - i32::from(*w));
                                *w = (i32::from(*w) + delta) as i8;
                                residual -= delta;
                            }
                        }
                        let expected: Vec<u8> = (0..width)
                            .map(|x| {
                                let mut sum = 64i32;
                                for (i, &w) in weights.iter().enumerate() {
                                    sum += i32::from(src[(start + i) * stride + x]) * i32::from(w);
                                }
                                (sum >> 7).clamp(0, 255) as u8
                            })
                            .collect();
                        let mut dst = vec![0xA5; width + 8];
                        convolve_vertical_sve2_i8_dot(
                            width,
                            &FilterBounds { start, size: taps },
                            &src,
                            &mut dst[1..width + 1],
                            stride,
                            &weights,
                            0,
                        );
                        assert_eq!(
                            &dst[1..width + 1],
                            expected,
                            "VL={vl}, width={width}, padding={padding}, taps={taps}, mode={mode}"
                        );
                        assert_eq!(dst[0], 0xA5);
                        assert!(dst[width + 1..].iter().all(|&v| v == 0xA5));
                        cases += 1;
                    }
                }
            }
        }
        eprintln!(
            "vertical u8 Q7: {cases} exact scalar cases at {} bits",
            vl * 8
        );
    }
}
