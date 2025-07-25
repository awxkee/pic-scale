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
#![forbid(unsafe_code)]
use num_traits::{AsPrimitive, Bounded};

#[derive(Debug, Clone)]
pub(crate) struct FilterWeights<T> {
    pub weights: Vec<T>,
    pub bounds: Vec<FilterBounds>,
    pub kernel_size: usize,
    pub aligned_size: usize,
    pub distinct_elements: usize,
    pub coeffs_size: i32,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct FilterBounds {
    pub start: usize,
    pub size: usize,
}

impl FilterBounds {
    pub(crate) fn new(start: usize, size: usize) -> FilterBounds {
        FilterBounds { start, size }
    }
}

impl<T> FilterWeights<T> {
    pub(crate) fn new(
        slice_ref: Vec<T>,
        kernel_size: usize,
        aligned_size: usize,
        distinct_elements: usize,
        coeffs_size: i32,
        bounds: Vec<FilterBounds>,
    ) -> FilterWeights<T> {
        FilterWeights::<T> {
            weights: slice_ref,
            bounds,
            kernel_size,
            aligned_size,
            distinct_elements,
            coeffs_size,
        }
    }

    pub(crate) fn cast<F: 'static + Copy>(&self) -> FilterWeights<F>
    where
        T: AsPrimitive<F> + 'static + Copy,
    {
        FilterWeights {
            weights: self.weights.iter().map(|&x| x.as_()).collect(),
            bounds: self.bounds.clone(),
            kernel_size: self.kernel_size,
            aligned_size: self.aligned_size,
            distinct_elements: self.distinct_elements,
            coeffs_size: self.coeffs_size,
        }
    }
}

impl FilterWeights<f32> {
    pub(crate) fn numerical_approximation_i16<const PRECISION: i32>(
        &self,
        alignment: usize,
    ) -> FilterWeights<i16> {
        self.numerical_approximation::<i16, PRECISION>(alignment)
    }

    pub(crate) fn numerical_approximation<
        J: Clone + Default + Copy + 'static + Bounded + AsPrimitive<f64>,
        const PRECISION: i32,
    >(
        &self,
        alignment: usize,
    ) -> FilterWeights<J>
    where
        f64: AsPrimitive<J>,
    {
        let align = if alignment != 0 {
            (self.kernel_size.div_ceil(alignment)) * alignment
        } else {
            self.kernel_size
        };
        let precision_scale: f64 = (1i64 << PRECISION) as f64;

        let mut output_kernel = vec![J::default(); self.distinct_elements * align];

        let lower_bound = J::min_value().as_();
        let upper_bound = J::max_value().as_();

        for (chunk, kernel_chunk) in self
            .weights
            .chunks_exact(self.kernel_size)
            .zip(output_kernel.chunks_exact_mut(align))
        {
            for (&weight, kernel) in chunk.iter().zip(kernel_chunk) {
                *kernel = (weight as f64 * precision_scale)
                    .min(upper_bound)
                    .max(lower_bound)
                    .as_();
            }
        }

        let mut new_bounds = vec![FilterBounds::new(0, 0); self.bounds.len()];

        for (dst, src) in new_bounds.iter_mut().zip(self.bounds.iter()) {
            *dst = *src;
        }

        FilterWeights::new(
            output_kernel,
            self.kernel_size,
            align,
            self.distinct_elements,
            self.coeffs_size,
            new_bounds,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn numerical_approximation_q0_7(&self, alignment: usize) -> FilterWeights<i8> {
        let align = if alignment != 0 {
            (self.kernel_size.div_ceil(alignment)) * alignment
        } else {
            self.kernel_size
        };
        let precision_scale: f64 = (1i64 << 7) as f64;

        let mut output_kernel = vec![0i8; self.distinct_elements * align];

        for (chunk, kernel_chunk) in self
            .weights
            .chunks_exact(self.kernel_size)
            .zip(output_kernel.chunks_exact_mut(align))
        {
            let mut local_sum = 0i32;
            for (&weight, kernel) in chunk.iter().zip(kernel_chunk.iter_mut()) {
                let new_element = (weight as f64 * precision_scale)
                    .min(i8::MAX as f64)
                    .max(i8::MIN as f64) as i8;
                *kernel = new_element;
                local_sum += new_element as i32;
            }
            if local_sum > 128 {
                let len = kernel_chunk.len() / 2;
                while local_sum > 128 {
                    local_sum -= 1;
                    kernel_chunk[len] = kernel_chunk[len].saturating_sub(1);
                }
            } else if local_sum < 128 {
                let len = kernel_chunk.len() / 2;
                while local_sum < 128 {
                    kernel_chunk[len] = kernel_chunk[len].saturating_add(1);
                    local_sum += 1;
                }
            }
        }

        let mut new_bounds = vec![FilterBounds::new(0, 0); self.bounds.len()];

        for (dst, src) in new_bounds.iter_mut().zip(self.bounds.iter()) {
            *dst = *src;
        }

        FilterWeights::new(
            output_kernel,
            self.kernel_size,
            align,
            self.distinct_elements,
            self.coeffs_size,
            new_bounds,
        )
    }
}

pub(crate) trait WeightsConverter<V> {
    fn prepare_weights(&self, weights: &FilterWeights<f32>) -> FilterWeights<V>;
}

#[derive(Default)]
pub(crate) struct DefaultWeightsConverter {}

impl<V: Default + Copy + 'static + Clone + Bounded + AsPrimitive<f64>> WeightsConverter<V>
    for DefaultWeightsConverter
where
    f64: AsPrimitive<V>,
{
    fn prepare_weights(&self, weights: &FilterWeights<f32>) -> FilterWeights<V> {
        use crate::support::PRECISION;
        weights.numerical_approximation::<V, PRECISION>(0)
    }
}

#[derive(Default)]
#[cfg(feature = "nightly_f16")]
pub(crate) struct PassthroughWeightsConverter {}

#[cfg(feature = "nightly_f16")]
impl WeightsConverter<f32> for PassthroughWeightsConverter {
    fn prepare_weights(&self, weights: &FilterWeights<f32>) -> FilterWeights<f32> {
        weights.clone()
    }
}

#[derive(Default)]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[cfg(feature = "nightly_f16")]
pub(crate) struct WeightFloat16Converter {}

#[cfg(feature = "nightly_f16")]
#[allow(unused)]
use core::f16;

#[cfg(feature = "nightly_f16")]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
impl WeightsConverter<f16> for WeightFloat16Converter {
    fn prepare_weights(&self, weights: &FilterWeights<f32>) -> FilterWeights<f16> {
        use crate::neon::convert_weights_to_f16_fhm;
        let converted_weights = convert_weights_to_f16_fhm(&weights.weights);

        let new_bounds = weights.bounds.to_vec();

        FilterWeights::new(
            converted_weights,
            weights.kernel_size,
            weights.kernel_size,
            weights.distinct_elements,
            weights.coeffs_size,
            new_bounds,
        )
    }
}
