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

use crate::chunking::chunked;

#[derive(Debug, Clone)]
pub struct FilterWeights<T> {
    pub weights: Vec<T>,
    pub bounds: Vec<FilterBounds>,
    pub kernel_size: usize,
    pub aligned_size: usize,
    pub distinct_elements: usize,
    pub coeffs_size: i32,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct FilterBounds {
    pub start: usize,
    pub size: usize,
}

impl FilterBounds {
    pub fn new(start: usize, size: usize) -> FilterBounds {
        FilterBounds { start, size }
    }
}

impl<T> FilterWeights<T> {
    pub fn new(
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
}

impl FilterWeights<f32> {
    pub fn numerical_approximation_i16<const PRECISION: i32>(
        &self,
        alignment: usize,
    ) -> FilterWeights<i16> {
        let align = if alignment != 0 {
            ((self.kernel_size + (alignment - 1)) / alignment) * alignment
        } else {
            self.kernel_size
        };
        let precision_scale: f32 = (1 << PRECISION) as f32;

        let mut output_kernel = vec![0i16; self.distinct_elements * align];

        let mut chunk_position = 0usize;

        for chunk in chunked(&self.weights, self.kernel_size) {
            for (i, _) in chunk.iter().enumerate() {
                let k = chunk_position + i;
                unsafe {
                    *output_kernel.get_unchecked_mut(k) =
                        ((*chunk[i]) * precision_scale).round() as i16;
                }
            }
            chunk_position += align;
        }

        let mut new_bounds = vec![FilterBounds::new(0, 0); self.bounds.len()];
        for i in 0..self.bounds.len() {
            unsafe {
                *new_bounds.get_unchecked_mut(i) = *self.bounds.get_unchecked(i);
            }
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
