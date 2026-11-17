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
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::math::gaussian::Exponential;
use crate::math::kaiser::BesselI0;
use crate::math::sinc::{Sinc, Trigonometry};
use crate::math::{ConstPI, ConstSqrt2, Jinc};
use crate::validation::try_vec;
use crate::{PicScaleError, ResamplingFunction};
use num_traits::{AsPrimitive, Float, MulAdd, Signed};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

pub(crate) trait WeightsGenerator<R> {
    fn make_weights(
        function: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<R>, PicScaleError>;
}

impl WeightsGenerator<f32> for u8 {
    fn make_weights(
        function: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<f32>, PicScaleError> {
        generate_weights::<f32, f64>(function, in_size, out_size)
    }
}

impl WeightsGenerator<f32> for u16 {
    fn make_weights(
        function: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<f32>, PicScaleError> {
        generate_weights::<f32, f64>(function, in_size, out_size)
    }
}

impl WeightsGenerator<f32> for i16 {
    fn make_weights(
        function: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<f32>, PicScaleError> {
        generate_weights::<f32, f64>(function, in_size, out_size)
    }
}

impl WeightsGenerator<f32> for f32 {
    fn make_weights(
        function: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<f32>, PicScaleError> {
        generate_weights::<f32, f64>(function, in_size, out_size)
    }
}

impl WeightsGenerator<f64> for f32 {
    fn make_weights(
        function: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<f64>, PicScaleError> {
        generate_weights::<f64, DoubleDouble>(function, in_size, out_size)
    }
}

use crate::math::dd::DoubleDouble;
#[cfg(feature = "nightly_f16")]
use core::f16;

#[cfg(feature = "nightly_f16")]
impl WeightsGenerator<f32> for f16 {
    fn make_weights(
        function: ResamplingFunction,
        in_size: usize,
        out_size: usize,
    ) -> Result<FilterWeights<f32>, PicScaleError> {
        generate_weights::<f32, f64>(function, in_size, out_size)
    }
}

fn generate_bilinear_weights<T>(
    in_size: usize,
    out_size: usize,
) -> Result<FilterWeights<T>, PicScaleError>
where
    T: Copy + Default + AsPrimitive<f64> + 'static,
    f64: AsPrimitive<T>,
{
    let scale = in_size as f64 / out_size as f64;
    let support = 2 * in_size as u128;
    let kernel_size = usize::try_from(support.div_ceil(out_size as u128).max(2))
        .map_err(|_| PicScaleError::DestinationImageIsTooLarge)?;
    let weight_count = kernel_size
        .checked_mul(out_size)
        .ok_or(PicScaleError::DestinationImageIsTooLarge)?;
    let mut weights = try_vec![T::default(); weight_count];
    let mut bounds = try_vec![FilterBounds::new(0, 0); out_size];
    let kernels = bounds.iter_mut().zip(weights.chunks_exact_mut(kernel_size));

    if in_size < out_size {
        let last = in_size - 1;
        for (i, (bound, kernel)) in kernels.enumerate() {
            let center = ((i as f64 + 0.5) * scale - 0.5).clamp(0.0, last as f64);
            let left = center.floor() as usize;
            let fraction = center - left as f64;
            if left == last || fraction == 0.0 {
                *bound = FilterBounds::new(left, 1);
                kernel[0] = 1.0.as_();
            } else {
                *bound = FilterBounds::new(left, 2);
                kernel[0] = (1.0 - fraction).as_();
                kernel[1] = fraction.as_();
            }
        }
    } else {
        // Express centers and the radius in units of 1 / (2 * out_size).
        // Integer bounds keep even the smallest positive taps at fractional ratios.
        let denominator = 2 * out_size as u128;
        let mut center = (in_size - out_size) as u128;
        for (bound, kernel) in kernels {
            let start = center
                .checked_sub(support)
                .map_or(0, |lower| lower / denominator + 1) as usize;
            let end = (center + support)
                .div_ceil(denominator)
                .min(in_size as u128) as usize;
            let size = end - start;
            *bound = FilterBounds::new(start, size);
            let values = (start..end)
                .map(|sample| (support - center.abs_diff(sample as u128 * denominator)) as f64);
            let inverse_sum = values.clone().sum::<f64>().recip();
            for (weight, value) in kernel[..size].iter_mut().zip(values) {
                *weight = (value * inverse_sum).as_();
            }
            center += support;
        }
    }

    Ok(FilterWeights::new(
        weights,
        kernel_size,
        kernel_size,
        out_size,
        scale.max(1.0) as i32,
        bounds,
    ))
}

fn generate_weights<
    T,
    W: AddAssign
        + Mul<W, Output = W>
        + Add<W, Output = W>
        + Sub<W, Output = W>
        + Div<W, Output = W>
        + 'static
        + Copy
        + Clone
        + AsPrimitive<T>
        + AsPrimitive<f32>
        + PartialOrd
        + PartialEq,
>(
    function: ResamplingFunction,
    in_size: usize,
    out_size: usize,
) -> Result<FilterWeights<T>, PicScaleError>
where
    T: Copy
        + Neg
        + Signed
        + Float
        + 'static
        + ConstPI
        + MulAssign<T>
        + AddAssign<T>
        + AsPrimitive<f64>
        + AsPrimitive<usize>
        + Jinc<T>
        + ConstSqrt2
        + Default
        + AsPrimitive<i32>
        + Trigonometry
        + Exponential
        + Sinc
        + BesselI0
        + MulAdd<Output = T>
        + AsPrimitive<W>,
    f32: AsPrimitive<T> + AsPrimitive<W>,
    f64: AsPrimitive<T>,
    i64: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    if function == ResamplingFunction::Bilinear {
        if in_size == 0 || out_size == 0 {
            return Err(PicScaleError::ZeroImageDimensions);
        }
        // Integer downscales already use the full support and keep their existing coefficients.
        if !in_size.is_multiple_of(out_size) {
            return generate_bilinear_weights(in_size, out_size);
        }
    }
    let resampling_filter = function.get_resampling_filter();
    let scale = in_size.as_() / out_size.as_();
    let is_resizable_kernel = resampling_filter.is_resizable_kernel;
    let filter_scale_cutoff = match is_resizable_kernel {
        true => scale.max(1f32.as_()),
        false => 1f32.as_(),
    };
    let filter_base_size: T = (resampling_filter.min_kernel_size as f64 * 2f64).as_();
    let resampling_function = resampling_filter.kernel;

    let is_area = resampling_filter.is_area && scale < 1.as_();

    let mut bounds: Vec<FilterBounds> = try_vec![FilterBounds::new(0, 0); out_size];

    if !is_area {
        let window_func = resampling_filter.window;
        let base_size: usize = (filter_base_size * filter_scale_cutoff).round().as_();
        let kernel_size = base_size;
        let filter_radius = base_size.as_() / 2.as_();
        let ones_t: T = 1f32.as_();
        let filter_scale = ones_t / filter_scale_cutoff;
        let mut weights: Vec<T> = try_vec![T::default(); kernel_size * out_size];
        let mut local_filters = try_vec![T::default(); kernel_size];
        let mut filter_position = 0usize;
        // At these integral scales, all tap coordinates and centers are exactly
        // representable, including half-integer centers in the f32 generator.
        let replay_phase =
            kernel_size >= 8 && in_size <= (1 << 22) && in_size.is_multiple_of(out_size);
        let mut previous_phase = None;
        let mut repeated_phase: Option<std::ops::Range<usize>> = None;
        let blur_scale: T = match window_func {
            None => 1f32.as_(),
            Some(window) => {
                let window_blur_t: T = window.blur.as_();
                if window_blur_t > 0f32.as_() {
                    ones_t / window_blur_t
                } else {
                    0f32.as_()
                }
            }
        };
        for (i, bound) in bounds.iter_mut().enumerate() {
            let center_x = ((i.as_() + 0.5.as_()) * scale).min(in_size.as_());
            let mut weights_sum: W = 0f32.as_();
            let mut local_filter_iteration = 0usize;

            let start: usize = (center_x - filter_radius).floor().max(0f32.as_()).as_();
            let end: usize = (center_x + filter_radius)
                .ceil()
                .min(start.as_() + kernel_size.as_())
                .min(in_size.as_())
                .as_();

            let center = center_x - 0.5.as_();
            let size = end - start;
            *bound = FilterBounds::new(start, size);

            if replay_phase {
                let phase = (start.as_() - center, size);
                if previous_phase == Some(phase) {
                    weights.copy_within(
                        filter_position - kernel_size..filter_position,
                        filter_position,
                    );
                    filter_position += kernel_size;

                    match &mut repeated_phase {
                        Some(span) if span.end == i => span.end += 1,
                        None => repeated_phase = Some(i - 1..i + 1),
                        _ => {}
                    }
                    continue;
                }
                previous_phase = Some(phase);
            }

            for (k, filter) in (start..end).zip(local_filters.iter_mut()) {
                let dx = k.as_() - center;
                let weight;
                if let Some(resampling_window) = window_func {
                    let mut x = dx.abs();
                    let resampling_blur_t: T = resampling_window.blur.as_();
                    let resampling_taper_t: T = resampling_window.taper.as_();
                    x = if resampling_blur_t > 0f32.as_() {
                        x * blur_scale
                    } else {
                        x
                    };
                    x = if x <= resampling_taper_t {
                        0f32.as_()
                    } else {
                        (x - resampling_taper_t) / (ones_t - resampling_taper_t)
                    };
                    let window_producer = resampling_window.window;
                    let x_kernel_scaled = x * filter_scale;
                    let window = if x < resampling_window.window_size.as_() {
                        window_producer(x_kernel_scaled * resampling_window.window_size.as_())
                    } else {
                        0f32.as_()
                    };
                    weight = window * resampling_function(x_kernel_scaled);
                } else {
                    let dx = dx.abs();
                    weight = resampling_function(dx * filter_scale);
                }
                weights_sum += weight.as_();
                *filter = weight;
                local_filter_iteration += 1;
            }

            if resampling_filter.is_ewa && !local_filters.is_empty() {
                weights_sum = 0f32.as_();
                for (j, filter) in local_filters
                    .iter_mut()
                    .take(local_filter_iteration)
                    .enumerate()
                {
                    // recompute weight using 2D radial distance from center
                    // dx is the 1D offset we already computed per tap; treat it as the radial r
                    let tap_pos: T = (start + j).as_();
                    let dx = (tap_pos - center).abs();
                    // r² = dx² — in 1D separable EWA pass, r = |dx|, but we weight
                    // using jinc(r) * window(r) so it behaves as a slice through
                    // the 2D circular kernel rather than an outer product of 1D kernels
                    let r = dx * filter_scale;
                    let ewa_weight = if r < resampling_filter.min_kernel_size.as_() {
                        T::jinc(r)
                            * match window_func {
                                Some(w) => {
                                    let wr = r / resampling_filter.min_kernel_size.as_();
                                    (w.window)(wr * w.window_size.as_())
                                }
                                None => 1f32.as_(),
                            }
                    } else {
                        0f32.as_()
                    };
                    *filter = ewa_weight;
                    weights_sum += ewa_weight.as_();
                }
            }

            if weights_sum != 0f32.as_() {
                let ones_w: W = 1f32.as_();
                let recpeq = ones_w / weights_sum;

                for (dst, src) in weights
                    .iter_mut()
                    .skip(filter_position)
                    .take(size)
                    .zip(local_filters.iter().take(size))
                {
                    let src_w: W = src.as_();
                    *dst = (src_w * recpeq).as_();
                }
            }

            filter_position += kernel_size;
        }

        let mut filter_weights = FilterWeights::<T>::new(
            weights,
            kernel_size,
            kernel_size,
            out_size,
            filter_radius.as_(),
            bounds,
        );
        filter_weights.repeated_phase = repeated_phase;
        Ok(filter_weights)
    } else {
        // Simulating INTER_AREA from OpenCV, for up scaling here,
        // this is necessary because weight computation is different
        // from any other func
        let inv_scale: T = 1.as_() / scale;
        let kernel_size = 2;
        let filter_radius: T = 1.as_();
        let mut weights: Vec<T> = try_vec![T::default(); kernel_size * out_size];
        let mut local_filters = try_vec![T::default(); kernel_size];
        let mut filter_position = 0usize;

        for (i, bound) in bounds.iter_mut().enumerate() {
            let mut weights_sum: W = 0f32.as_();

            let sx: T = (i.as_() * scale).floor();
            let fx = (i as i64 + 1).as_() - (sx + 1.as_()) * inv_scale;
            let dx = if fx <= 0.as_() {
                0.as_()
            } else {
                fx - fx.floor()
            };
            let dx = dx.abs();
            let weight0 = 1.as_() - dx;
            let weight1: T = dx;
            local_filters[0] = weight0;
            local_filters[1] = weight1;

            let start: usize = sx.floor().max(0f32.as_()).as_();
            let end: usize = (sx + kernel_size.as_())
                .ceil()
                .min(start.as_() + kernel_size.as_())
                .min(in_size.as_())
                .as_();

            let size = end - start;

            weights_sum += weight0.as_();
            if size > 1 {
                weights_sum += weight1.as_();
            }
            *bound = FilterBounds::new(start, size);

            if weights_sum != 0f32.as_() {
                let ones_w: W = 1f32.as_();
                let recpeq = ones_w / weights_sum;

                for (dst, src) in weights
                    .iter_mut()
                    .skip(filter_position)
                    .take(size)
                    .zip(local_filters.iter().take(size))
                {
                    let src_w: W = src.as_();
                    *dst = (src_w * recpeq).as_();
                }
            } else {
                weights[filter_position] = 1.as_();
            }

            filter_position += kernel_size;
        }

        Ok(FilterWeights::new(
            weights,
            kernel_size,
            kernel_size,
            out_size,
            filter_radius.as_(),
            bounds,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bilinear(in_size: usize, out_size: usize) -> FilterWeights<f32> {
        u8::make_weights(ResamplingFunction::Bilinear, in_size, out_size).unwrap()
    }

    fn assert_taps(weights: &FilterWeights<f32>, output: usize, start: usize, expected: &[f32]) {
        assert_eq!(
            weights.bounds[output],
            FilterBounds::new(start, expected.len())
        );
        let offset = output * weights.aligned_size;
        for (&actual, &expected) in weights.weights[offset..].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn bilinear_upscale_uses_both_interior_taps_and_clamps_edges() {
        let weights = bilinear(4, 6);
        assert_eq!(weights.kernel_size, 2);
        assert_taps(&weights, 0, 0, &[1.0]);
        assert_taps(&weights, 2, 1, &[5.0 / 6.0, 1.0 / 6.0]);
        assert_taps(&weights, 3, 1, &[1.0 / 6.0, 5.0 / 6.0]);
        assert_taps(&weights, 5, 3, &[1.0]);
    }

    #[test]
    fn bilinear_fractional_downscale_keeps_full_triangle() {
        let weights = bilinear(15, 10);
        assert_taps(&weights, 0, 0, &[5.0 / 8.0, 3.0 / 8.0]);
        assert_taps(&weights, 1, 1, &[1.0 / 3.0, 5.0 / 9.0, 1.0 / 9.0]);
        assert_taps(&weights, 9, 13, &[3.0 / 8.0, 5.0 / 8.0]);

        let weights = bilinear(12, 10);
        assert_eq!(weights.kernel_size, 3);
        assert_taps(&weights, 4, 4, &[1.0 / 5.0, 11.0 / 15.0, 1.0 / 15.0]);
    }

    #[test]
    fn bilinear_integer_downscale_keeps_existing_kernel() {
        let weights = bilinear(8, 4);
        assert_eq!(weights.kernel_size, 4);
        assert_taps(&weights, 0, 0, &[3.0 / 7.0, 3.0 / 7.0, 1.0 / 7.0]);
        assert_taps(&weights, 1, 1, &[0.125, 0.375, 0.375, 0.125]);
    }

    #[test]
    fn bilinear_handles_single_sample_and_rejects_empty_axes() {
        let weights = bilinear(1, 5);
        for output in 0..5 {
            assert_taps(&weights, output, 0, &[1.0]);
        }
        for (input, output) in [(0, 5), (5, 0)] {
            assert!(matches!(
                u8::make_weights(ResamplingFunction::Bilinear, input, output),
                Err(PicScaleError::ZeroImageDimensions)
            ));
        }
    }
}
