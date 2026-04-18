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

/// Pack four SVE uint8 vectors (one per row) into the interleaved layout
/// expected by `svdot`: result[4i..4i+4] = [a[i], b[i], c[i], d[i]].
///
/// SVE2 `svdot[_lane]_s32` with unsigned×signed computes, per group of 4 bytes:
///   acc += u8[0]*i8[0] + u8[1]*i8[1] + u8[2]*i8[2] + u8[3]*i8[3]
///
/// So we interleave 4 source rows into that layout — same idea as the NEON
/// `packq_4_rows` helper, but predicated and scalable.
#[inline]
#[target_feature(enable = "sve2,i8mm")]
unsafe fn pack_4_rows_sve(
    a: svuint8_t,
    b: svuint8_t,
    c: svuint8_t,
    d: svuint8_t,
) -> [svuint8_t; 2] {
    // Interleave pairs at byte level, then at 16-bit level.
    let ab_lo = svzip1_u8(a, b); // even bytes: a0 b0 a1 b1 ...
    let ab_hi = svzip2_u8(a, b); // odd bytes

    let cd_lo = svzip1_u8(c, d);
    let cd_hi = svzip2_u8(c, d);

    let lo = svreinterpret_u8_u16(svzip1_u16(
        svreinterpret_u16_u8(ab_lo),
        svreinterpret_u16_u8(cd_lo),
    ));
    let hi = svreinterpret_u8_u16(svzip1_u16(
        svreinterpret_u16_u8(ab_hi),
        svreinterpret_u16_u8(cd_hi),
    ));
    [lo, hi]
}

/// Core predicated loop: processes `vl` pixels per iteration (vl = SVE vector
/// byte length, e.g. 16 on SVE-128, 32 on SVE-256, 64 on SVE-512).
///
/// For each horizontal tile of `vl` pixels we accumulate over all filter rows,
/// processing 4 rows at a time with `svdotq`, then 2 rows, then 1.
#[inline(never)]
#[target_feature(enable = "sve2")]
unsafe fn convolve_vertical_sve2_row(
    width: usize,
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weights: &[i8],
) {
    // SVE vector length in bytes (determined at runtime).
    let vl = svcntb() as usize;

    let mut cx = 0usize;

    while cx < width {
        let pg = svwhilelt_b8_u64(cx as u64, width as u64);

        let pg32_lo = svwhilelt_b32_u64(cx as u64, width as u64);
        let pg32_hi = svwhilelt_b32_u64((cx + vl / 2) as u64, width as u64);

        let rounding = svdup_n_s32(ROUNDING);
        let mut acc_lo = svsel_s32(pg32_lo, rounding, svdup_n_s32(0));
        let mut acc_hi = svsel_s32(pg32_hi, rounding, svdup_n_s32(0));

        let mut j = 0usize;

        while j + 4 <= bounds.size {
            let py = bounds.start + j;
            let w = weights.get_unchecked(j..j + 4);

            let w32 = i32::from_le_bytes([w[0] as u8, w[1] as u8, w[2] as u8, w[3] as u8]);
            let vw = svreinterpret_s8_s32(svdup_n_s32(w32));

            let base0 = src_stride * py + cx;
            let row0 = svld1_u8(pg, src.as_ptr().add(base0));
            let row1 = svld1_u8(pg, src.as_ptr().add(base0 + src_stride));
            let row2 = svld1_u8(pg, src.as_ptr().add(base0 + src_stride * 2));
            let row3 = svld1_u8(pg, src.as_ptr().add(base0 + src_stride * 3));

            let [packed_lo, packed_hi] = pack_4_rows_sve(row0, row1, row2, row3);

            // svdot: acc[i] += packed[4i]*w[0] + packed[4i+1]*w[1]
            //                + packed[4i+2]*w[2] + packed[4i+3]*w[3]
            acc_lo = svdot_s32(acc_lo, svreinterpret_s8_u8(packed_lo), vw);
            acc_hi = svdot_s32(acc_hi, svreinterpret_s8_u8(packed_hi), vw);

            j += 4;
        }

        // ------------------------------------------------------------------
        // 2-row-at-a-time path
        // ------------------------------------------------------------------
        while j + 2 <= bounds.size {
            let py = bounds.start + j;
            let w = weights.get_unchecked(j..j + 2);

            let w32 = i32::from_le_bytes([w[0] as u8, w[1] as u8, 0, 0]);
            let vw = svreinterpret_s8_s32(svdup_n_s32(w32));

            let base0 = src_stride * py + cx;
            let row0 = svld1_u8(pg, src.as_ptr().add(base0));
            let row1 = svld1_u8(pg, src.as_ptr().add(base0 + src_stride));
            let zero = svdup_n_u8(0);

            let [packed_lo, packed_hi] = pack_4_rows_sve(row0, row1, zero, zero);

            acc_lo = svdot_s32(acc_lo, svreinterpret_s8_u8(packed_lo), vw);
            acc_hi = svdot_s32(acc_hi, svreinterpret_s8_u8(packed_hi), vw);

            j += 2;
        }

        // ------------------------------------------------------------------
        // Scalar tail (0 or 1 remaining rows)
        // ------------------------------------------------------------------
        while j < bounds.size {
            let py = bounds.start + j;
            let w = *weights.get_unchecked(j) as i32;

            let base0 = src_stride * py + cx;
            let row = svld1_u8(pg, src.as_ptr().add(base0));
            let zero = svdup_n_u8(0);

            // Use 1-row pack (only w[0] contributes; w[1..3] == 0).
            let w32 = i32::from_le_bytes([w as u8, 0, 0, 0]);
            let vw = svreinterpret_s8_s32(svdup_n_s32(w32));

            let [packed_lo, packed_hi] = pack_4_rows_sve(row, zero, zero, zero);

            acc_lo = svdot_s32(acc_lo, svreinterpret_s8_u8(packed_lo), vw);
            acc_hi = svdot_s32(acc_hi, svreinterpret_s8_u8(packed_hi), vw);

            j += 1;
        }

        let shifted_lo = svasr_n_s32_x(pg32_lo, acc_lo, SCALE as u32);
        let shifted_hi = svasr_n_s32_x(pg32_hi, acc_hi, SCALE as u32);

        let narrow_lo = svqxtunt_s32(svdup_n_u16(0), shifted_lo);
        let narrow_hi = svqxtunb_s32(shifted_hi);

        let bytes_lo = svqxtnb_u16(narrow_lo);
        let bytes_hi = svqxtnb_u16(narrow_hi);

        let result = svuzp1_u8(bytes_lo, bytes_hi);
        svst1_u8(pg, dst.as_mut_ptr().add(cx), result);

        cx += vl;
    }
}
