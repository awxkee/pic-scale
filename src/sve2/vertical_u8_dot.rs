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

#[inline]
#[target_feature(enable = "sve2")]
fn pack_4_rows_sve(a: svuint8_t, b: svuint8_t, c: svuint8_t, d: svuint8_t) -> [svuint8_t; 4] {
    let ab_lo = svzip1_u8(a, b);
    let ab_hi = svzip2_u8(a, b);

    let cd_lo = svzip1_u8(c, d);
    let cd_hi = svzip2_u8(c, d);

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
}

#[inline(never)]
#[target_feature(enable = "sve2,i8mm")]
fn convolve_vertical_sve2_row(
    _: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i8],
) {
    let vl = svcntb() as usize;

    let mut cx = 0usize;

    let len = dst.len();

    while cx < dst.len() {
        let pg = svwhilelt_b8_u64(cx as u64, len as u64);

        let pg32_0 = svwhilelt_b32_u64(cx as u64, len as u64);
        let pg32_1 = svwhilelt_b32_u64((cx + vl / 4) as u64, len as u64);
        let pg32_2 = svwhilelt_b32_u64((cx + vl / 2) as u64, len as u64);
        let pg32_3 = svwhilelt_b32_u64((cx + 3 * (vl / 4)) as u64, len as u64);

        let rounding = svdup_n_s32(ROUNDING);
        let mut acc_0 = svsel_s32(pg32_0, rounding, svdup_n_s32(0));
        let mut acc_1 = svsel_s32(pg32_1, rounding, svdup_n_s32(0));
        let mut acc_2 = svsel_s32(pg32_2, rounding, svdup_n_s32(0));
        let mut acc_3 = svsel_s32(pg32_3, rounding, svdup_n_s32(0));

        let mut j = 0usize;

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..j + 4) };

            let w32 = i32::from_le_bytes([w[0] as u8, w[1] as u8, w[2] as u8, w[3] as u8]);
            let vw = svreinterpret_s8_s32(svdup_n_s32(w32));

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u8(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let row2 =
                unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride * 2..).as_ptr()) };
            let row3 =
                unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride * 3..).as_ptr()) };

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve(row0, row1, row2, row3);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let w = unsafe { weights.get_unchecked(j..) };

            let w32 = i32::from_le_bytes([w[0] as u8, w[1] as u8, 0, 0]);
            let vw = svreinterpret_s8_s32(svdup_n_s32(w32));

            let base0 = src_stride * py + cx;
            let row0 = unsafe { svld1_u8(pg, src.get_unchecked(base0..).as_ptr()) };
            let row1 = unsafe { svld1_u8(pg, src.get_unchecked(base0 + src_stride..).as_ptr()) };
            let zero = svdup_n_u8(0);

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve(row0, row1, zero, zero);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);

            j += 2;
        }

        while j < bounds.size {
            let py = bounds.start + j;
            let w = unsafe { *weights.get_unchecked(j) };

            let base0 = src_stride * py + cx;
            let row = unsafe { svld1_u8(pg, src.get_unchecked(base0..).as_ptr()) };
            let zero = svdup_n_u8(0);

            let vw = svdup_n_s8(w);

            let [packed0, packed1, packed2, packed3] = pack_4_rows_sve(row, zero, zero, zero);

            acc_0 = svusdot_s32(acc_0, packed0, vw);
            acc_1 = svusdot_s32(acc_1, packed1, vw);
            acc_2 = svusdot_s32(acc_2, packed2, vw);
            acc_3 = svusdot_s32(acc_3, packed3, vw);

            j += 1;
        }

        let shifted0 = svasr_n_s32_x(pg32_0, acc_0, SCALE as u32);
        let shifted1 = svasr_n_s32_x(pg32_1, acc_1, SCALE as u32);
        let shifted2 = svasr_n_s32_x(pg32_2, acc_2, SCALE as u32);
        let shifted3 = svasr_n_s32_x(pg32_3, acc_3, SCALE as u32);

        let n0 = svqxtunb_s32(shifted0);
        let n1 = svqxtunb_s32(shifted1);
        let n2 = svqxtunb_s32(shifted2);
        let n3 = svqxtunb_s32(shifted3);

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
