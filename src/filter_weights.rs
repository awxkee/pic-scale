/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
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
            for i in 0..chunk.len() {
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

        return FilterWeights::new(
            output_kernel,
            self.kernel_size,
            align,
            self.distinct_elements,
            self.coeffs_size,
            new_bounds,
        );
    }
}
