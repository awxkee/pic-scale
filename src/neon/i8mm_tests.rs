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

use super::{
    convolve_horizontal_rgb_neon_row_one_dot, convolve_horizontal_rgb_neon_rows_4_dot,
    convolve_horizontal_rgba_neon_row_dot, convolve_horizontal_rgba_neon_rows_4_u8_dot,
    convolve_vertical_neon_i8_dot,
};
use crate::filter_weights::{FilterBounds, FilterWeights};

fn available() -> bool {
    let found = std::arch::is_aarch64_feature_detected!("i8mm");
    assert!(
        found || std::env::var_os("PIC_SCALE_REQUIRE_I8MM").is_none(),
        "i8mm required but unavailable"
    );
    if !found {
        eprintln!("Skipping: i8mm unavailable");
    }
    found
}

type Row1 = fn(&[u8], &mut [u8], &FilterWeights<i8>, u32);
type Row4 = fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i8>, u32);

fn check_horizontal<const N: usize>(one: Row1, four: Row4) {
    if !available() {
        return;
    }
    let mut cases = 0;
    for width in [1usize, 2, 3, 7, 19, 33] {
        let ds = width * N + 13;
        for taps in [
            0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65,
            127,
        ] {
            let sw = (taps + width - 1).max(1);
            let ss = sw * N + 11;
            // The final row ends at its last pixel; earlier rows include padding.
            let mut src = vec![0x9b; ss * 3 + sw * N];
            for row in 0..4 {
                for x in 0..sw {
                    for c in 0..N {
                        src[row * ss + x * N + c] =
                            ((row * 67 + x * 29 + c * 43 + x * c * 11) % 256) as u8;
                    }
                }
            }
            let aligned = taps.next_multiple_of(4).max(4);
            for clipped in [false, true] {
                let bounds: Vec<_> = (0..width)
                    .map(|x| {
                        let size = if clipped && x == 0 {
                            taps.div_ceil(2)
                        } else if clipped && taps != 0 {
                            taps - (x % 4).min(taps - 1)
                        } else {
                            taps
                        };
                        let start = if x == width - 1 { sw - size } else { x.min(sw) };
                        FilterBounds::new(start, size)
                    })
                    .collect();
                for pattern in 0..4 {
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
                                2 => {
                                    if j % 2 == 0 {
                                        -128
                                    } else {
                                        127
                                    }
                                }
                                _ => 127,
                            };
                        }
                    }
                    let f = FilterWeights::new(weights, taps, aligned, width, 0, bounds.clone());
                    let mut expected = vec![0xa5; ds * 3 + width * N];
                    for row in 0..4 {
                        for x in 0..width {
                            for c in 0..N {
                                let b = &f.bounds[x];
                                let mut sum = 64i64;
                                for j in 0..b.size {
                                    sum += i64::from(src[row * ss + (b.start + j) * N + c])
                                        * i64::from(f.weights[x * aligned + j]);
                                }
                                expected[row * ds + x * N + c] = (sum >> 7).clamp(0, 255) as u8;
                            }
                        }
                    }
                    for rows4 in [false, true] {
                        let mut actual = vec![0xa5; expected.len()];
                        if rows4 {
                            four(&src, ss, &mut actual, ds, &f, 8)
                        } else {
                            for row in 0..4 {
                                one(
                                    &src[row * ss..][..sw * N],
                                    &mut actual[row * ds..][..width * N],
                                    &f,
                                    8,
                                );
                            }
                        }
                        assert_eq!(
                            actual, expected,
                            "N={N}, width={width}, taps={taps}, clipped={clipped}, pattern={pattern}, rows4={rows4}"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("NEON i8mm {N}-channel horizontal Q7: {cases} exact scalar cases");
}

#[test]
fn neon_i8mm_rgb_q7_scalar() {
    check_horizontal::<3>(
        convolve_horizontal_rgb_neon_row_one_dot,
        convolve_horizontal_rgb_neon_rows_4_dot,
    );
}

#[test]
fn neon_i8mm_rgba_q7_scalar() {
    check_horizontal::<4>(
        convolve_horizontal_rgba_neon_row_dot,
        convolve_horizontal_rgba_neon_rows_4_u8_dot,
    );
}

#[test]
fn neon_i8mm_vertical_q7_scalar() {
    if !available() {
        return;
    }
    let mut cases = 0;
    for width in [
        0usize, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 95, 96, 127, 128,
        129,
    ] {
        for padding in [0usize, 5] {
            let ss = width + padding;
            for taps in [
                0usize, 1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 24, 31, 32, 63, 127,
            ] {
                let start = 2;
                let rows = (start + taps).max(1);
                let mut src = vec![0xbc; ss * (rows - 1) + width];
                for y in 0..rows {
                    for x in 0..width {
                        src[y * ss + x] = (x * 97 + y * 53 + (x ^ (y * 17)) * 13) as u8;
                    }
                }
                for pattern in 0..4 {
                    let w: Vec<i8> = (0..taps)
                        .map(|j| match pattern {
                            0 => ((j * 43 + 19) % 65) as i8 - 32,
                            1 => {
                                if j % 2 == 0 {
                                    -128
                                } else {
                                    127
                                }
                            }
                            2 => 127,
                            _ => -128,
                        })
                        .collect();
                    let expected: Vec<u8> = (0..width)
                        .map(|x| {
                            let mut sum = 64i64;
                            for j in 0..taps {
                                sum += i64::from(src[(start + j) * ss + x]) * i64::from(w[j]);
                            }
                            (sum >> 7).clamp(0, 255) as u8
                        })
                        .collect();
                    for band in [false, true] {
                        let mut guarded = vec![0xa5; width + 8];
                        let (s, b) = if band {
                            (&src[ss..], FilterBounds::new(start - 1, taps))
                        } else {
                            (&src[..], FilterBounds::new(start, taps))
                        };
                        convolve_vertical_neon_i8_dot(
                            width,
                            &b,
                            s,
                            &mut guarded[3..width + 3],
                            ss,
                            &w,
                            8,
                        );
                        assert_eq!(
                            &guarded[3..width + 3],
                            &expected,
                            "width={width}, taps={taps}, pad={padding}, pattern={pattern}, band={band}"
                        );
                        assert!(
                            guarded[..3]
                                .iter()
                                .chain(&guarded[width + 3..])
                                .all(|&x| x == 0xa5)
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("NEON i8mm vertical Q7: {cases} exact scalar cases");
}
