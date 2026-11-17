/*
 * Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice, this
 *  list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *  contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use super::vertical_pair::PairedVertical;
use crate::convolution::ColumnFilter;
use crate::plan::VerticalFiltering;
use std::arch::aarch64::*;
use std::sync::Arc;

pub(crate) fn paired_vertical_f32<const N: usize>(
    filter: VerticalFiltering<f32, f32, N>,
) -> Arc<dyn ColumnFilter<f32, N> + Send + Sync> {
    match filter.filter_weights.kernel_size {
        12 => Arc::new(PairedVertical::<_, _, N, 12, 2, 16> {
            filter,
            convolve_pair: convolve_pair_f32::<12, 2, 4>,
        }),
        24 => Arc::new(PairedVertical::<_, _, N, 24, 4, 32> {
            filter,
            convolve_pair: convolve_pair_f32::<24, 4, 8>,
        }),
        _ => Arc::new(filter),
    }
}

pub(crate) fn paired_vertical_f64<const N: usize>(
    filter: VerticalFiltering<f32, f64, N>,
) -> Arc<dyn ColumnFilter<f32, N> + Send + Sync> {
    match filter.filter_weights.kernel_size {
        12 => Arc::new(PairedVertical::<_, _, N, 12, 2, 16> {
            filter,
            convolve_pair: convolve_pair_f64::<12, 2>,
        }),
        24 => Arc::new(PairedVertical::<_, _, N, 24, 4, 16> {
            filter,
            convolve_pair: convolve_pair_f64::<24, 4>,
        }),
        _ => Arc::new(filter),
    }
}

#[inline(always)]
unsafe fn load_inputs<const VECTORS: usize>(ptr: *const f32) -> [float32x4_t; VECTORS] {
    std::array::from_fn(|i| unsafe { vld1q_f32(ptr.add(i * 4)) })
}

#[inline(always)]
fn accumulate_f32<const VECTORS: usize>(
    acc: &mut [float32x4_t; VECTORS],
    pixels: [float32x4_t; VECTORS],
    weight: f32,
) {
    unsafe {
        let weight = vdupq_n_f32(weight);
        for (acc, pixel) in acc.iter_mut().zip(pixels) {
            *acc = vfmaq_f32(*acc, pixel, weight);
        }
    }
}

#[target_feature(enable = "neon")]
fn convolve_pair_f32<const TAPS: usize, const STEP: usize, const VECTORS: usize>(
    input: &[f32],
    stride: usize,
    first: &mut [f32],
    second: &mut [f32],
    first_weights: &[f32; TAPS],
    second_weights: &[f32; TAPS],
    _: u32,
) {
    let zero = vdupq_n_f32(0.0);
    for x in (0..first.len()).step_by(4 * VECTORS) {
        let mut first_acc = [zero; VECTORS];
        let mut second_acc = [zero; VECTORS];
        unsafe {
            let base = input.as_ptr().add(x);
            for (j, &weight) in first_weights[..STEP].iter().enumerate() {
                accumulate_f32(&mut first_acc, load_inputs(base.add(j * stride)), weight);
            }
            // The overlapping input rows feed both sums in their original tap order.
            for j in STEP..TAPS {
                let pixels = load_inputs(base.add(j * stride));
                accumulate_f32(&mut first_acc, pixels, first_weights[j]);
                accumulate_f32(&mut second_acc, pixels, second_weights[j - STEP]);
            }
            for j in TAPS..TAPS + STEP {
                accumulate_f32(
                    &mut second_acc,
                    load_inputs(base.add(j * stride)),
                    second_weights[j - STEP],
                );
            }
            for (out, acc) in [(&mut *first, first_acc), (&mut *second, second_acc)] {
                for (i, value) in acc.into_iter().enumerate() {
                    vst1q_f32(out.as_mut_ptr().add(x + i * 4), value);
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn widen_inputs(ptr: *const f32) -> [float64x2_t; 8] {
    unsafe {
        let [a, b, c, d] = load_inputs(ptr);
        [
            vcvt_f64_f32(vget_low_f32(a)),
            vcvt_high_f64_f32(a),
            vcvt_f64_f32(vget_low_f32(b)),
            vcvt_high_f64_f32(b),
            vcvt_f64_f32(vget_low_f32(c)),
            vcvt_high_f64_f32(c),
            vcvt_f64_f32(vget_low_f32(d)),
            vcvt_high_f64_f32(d),
        ]
    }
}

#[inline(always)]
unsafe fn accumulate_f64(acc: &mut [float64x2_t; 8], pixels: [float64x2_t; 8], weight: f64) {
    unsafe {
        let weight = vdupq_n_f64(weight);
        for (acc, pixel) in acc.iter_mut().zip(pixels) {
            *acc = vfmaq_f64(*acc, pixel, weight);
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn convolve_pair_f64<const TAPS: usize, const STEP: usize>(
    input: &[f32],
    stride: usize,
    first: &mut [f32],
    second: &mut [f32],
    first_weights: &[f64; TAPS],
    second_weights: &[f64; TAPS],
    _: u32,
) {
    let zero = vdupq_n_f64(0.0);
    for x in (0..first.len()).step_by(16) {
        let mut first_acc = [zero; 8];
        let mut second_acc = [zero; 8];
        unsafe {
            let base = input.as_ptr().add(x);
            for (j, &weight) in first_weights[..STEP].iter().enumerate() {
                accumulate_f64(&mut first_acc, widen_inputs(base.add(j * stride)), weight);
            }
            // Share both the input loads and their conversions to f64.
            for j in STEP..TAPS {
                let pixels = widen_inputs(base.add(j * stride));
                accumulate_f64(&mut first_acc, pixels, first_weights[j]);
                accumulate_f64(&mut second_acc, pixels, second_weights[j - STEP]);
            }
            for j in TAPS..TAPS + STEP {
                accumulate_f64(
                    &mut second_acc,
                    widen_inputs(base.add(j * stride)),
                    second_weights[j - STEP],
                );
            }
            for (out, acc) in [(&mut *first, first_acc), (&mut *second, second_acc)] {
                for (i, pair) in acc.as_chunks::<2>().0.iter().enumerate() {
                    vst1q_f32(
                        out.as_mut_ptr().add(x + i * 4),
                        vcvt_high_f32_f64(vcvt_f32_f64(pair[0]), pair[1]),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThreadingPolicy;
    use crate::convolution::ColumnSource;
    use crate::filter_weights::{FilterBounds, FilterWeights};
    use crate::neon::{convolve_vertical_neon_row_f32_f64, convolve_vertical_rgb_neon_row_f32};

    fn check_rows<F: Copy + Send + Sync>(
        taps: usize,
        step: usize,
        filter_row: fn(usize, &FilterBounds, &[f32], &mut [f32], usize, &[F], u32),
        paired: impl Fn(VerticalFiltering<f32, F, 4>) -> Arc<dyn ColumnFilter<f32, 4> + Send + Sync>,
        weight: impl Fn(usize) -> F,
    ) {
        let bounds = [
            FilterBounds::new(0, taps - 1),
            FilterBounds::new(3, taps),
            FilterBounds::new(3 + step, taps),
            FilterBounds::new(3 + 2 * step, taps - 1),
            FilterBounds::new(3 + 3 * step, taps),
            FilterBounds::new(3 + 4 * step, taps),
            FilterBounds::new(4 + 5 * step, taps),
            FilterBounds::new(4 + 6 * step, taps),
        ];
        let make_filter = || VerticalFiltering {
            filter_weights: FilterWeights::new(
                (0..taps * bounds.len()).map(&weight).collect(),
                taps,
                taps,
                bounds.len(),
                (taps / 2) as i32,
                bounds.to_vec(),
            ),
            filter_row,
            threading_policy: ThreadingPolicy::Single,
        };
        let baseline = make_filter();
        let filter = paired(make_filter());

        for width in [8, 9, 20] {
            let row_len = width * 4;
            let stride = row_len + 5;
            let last = bounds.last().unwrap();
            let pixels = (0..(last.start + last.size - 1) * stride + row_len)
                .map(|i| match (i * 997) % 127 {
                    1 => f32::from_bits(0x7fc0_1234),
                    2 => f32::from_bits(0xffc0_4567),
                    3 => f32::from_bits(0x7f80_1234),
                    4 => f32::INFINITY,
                    5 => f32::NEG_INFINITY,
                    6 => -0.0,
                    7 => f32::from_bits(1),
                    8 => f32::MAX,
                    _ => (i % 509) as f32 * 0.013 - 2.0,
                })
                .collect::<Vec<_>>();
            let source = ColumnSource {
                pixels: &pixels[3 * stride..],
                width,
                stride,
                first_row: 3,
            };
            let mut expected = vec![0.0; 7 * row_len];
            let mut actual = vec![0.0; 7 * row_len];
            baseline.run_on_band_rows(&source, &mut expected, 1, 0);
            filter.run_on_band_rows(&source, &mut actual, 1, 0);
            let compare = |actual: &[f32], expected: &[f32]| {
                for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "width={width}, taps={taps}, sample={i}"
                    );
                }
            };
            compare(&actual, &expected);

            let actual = &mut actual[..4 * row_len];
            actual.fill(0.0);
            filter.run_on_4_rows(&pixels, actual, width, stride, 1, 0);
            compare(actual, &expected[..4 * row_len]);
        }
    }

    #[test]
    fn paired_f32_rows_keep_ieee_results() {
        for (taps, step) in [(12, 2), (24, 4)] {
            check_rows(
                taps,
                step,
                convolve_vertical_rgb_neon_row_f32,
                paired_vertical_f32,
                |i| ((i * 19 % 31) as f32 - 15.0) * (1.0 / 64.0),
            );
        }
    }

    #[test]
    fn paired_f64_rows_keep_ieee_results() {
        for (taps, step) in [(12, 2), (24, 4)] {
            check_rows(
                taps,
                step,
                convolve_vertical_neon_row_f32_f64,
                paired_vertical_f64,
                |i| ((i * 19 % 31) as f64 - 15.0) * (1.0 / 64.0),
            );
        }
    }
}
