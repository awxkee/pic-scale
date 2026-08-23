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
//! Shared harness for cross-backend differential tests: every accelerated
//! (SSE/AVX2/AVX-512/NEON/SVE2/WASM) row-convolution function is built to the
//! same signature as the portable scalar fallback in `handler_provider`, so
//! feeding both the identical weights and source pixels and diffing the
//! outputs catches a backend that has drifted from the others - regardless of
//! which architecture the test suite happens to run on.
#![cfg(test)]

use crate::filter_weights::FilterWeights;
use crate::math::WeightsGenerator;
use crate::support::PRECISION;
use crate::{PicScaleError, ResamplingFunction};
#[cfg(feature = "nightly_f16")]
use core::f16;

/// Builds the same fixed-point row filter weights the real resize pipeline
/// would use for `u8` horizontal convolution, so backend-comparison tests
/// exercise realistic weight patterns instead of ad-hoc synthetic ones.
pub(crate) fn make_row_filter_weights_u8(
    resampling: ResamplingFunction,
    in_size: usize,
    out_size: usize,
) -> Result<FilterWeights<i16>, PicScaleError> {
    let weights_f32 = <u8 as WeightsGenerator<f32>>::make_weights(resampling, in_size, out_size)?;
    Ok(weights_f32.numerical_approximation::<i16, PRECISION>(0))
}

/// Builds the real (non-quantized) `f32` row filter weights the floating-point
/// pipeline uses directly - no fixed-point conversion involved. Also the
/// correct weights to use for `f16` convolution: that pipeline weights in
/// `FilterWeights<f32>` too (only the pixel storage type is `f16`).
pub(crate) fn make_row_filter_weights_f32(
    resampling: ResamplingFunction,
    in_size: usize,
    out_size: usize,
) -> Result<FilterWeights<f32>, PicScaleError> {
    <f32 as WeightsGenerator<f32>>::make_weights(resampling, in_size, out_size)
}

/// Same as `make_row_filter_weights_u8`, but for `u16` fixed-point convolution
/// (still quantized to `i16` weights - `u16` pixels just carry more bit depth).
pub(crate) fn make_row_filter_weights_u16(
    resampling: ResamplingFunction,
    in_size: usize,
    out_size: usize,
) -> Result<FilterWeights<i16>, PicScaleError> {
    let weights_f32 = <u16 as WeightsGenerator<f32>>::make_weights(resampling, in_size, out_size)?;
    Ok(weights_f32.numerical_approximation::<i16, PRECISION>(0))
}

/// Minimal dependency-free xorshift64 PRNG so backend-comparison tests are
/// reproducible without pulling in a `rand` dev-dependency.
pub(crate) struct XorShiftRng(u64);

impl XorShiftRng {
    pub(crate) fn new(seed: u64) -> Self {
        XorShiftRng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(crate) fn fill_u8(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| (self.next_u64() % 256) as u8).collect()
    }

    /// Uniform `f32` in `[0, 1)`, the conventional range for normalized pixel data.
    pub(crate) fn fill_f32_unit(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
            .collect()
    }

    pub(crate) fn fill_u16(&mut self, len: usize, max_value: u16) -> Vec<u16> {
        (0..len)
            .map(|_| (self.next_u64() % (max_value as u64 + 1)) as u16)
            .collect()
    }

    /// Uniform `f16` in `[0, 1)` - built from the same unit-`f32` stream so the
    /// two are drawn from equivalent distributions, then narrowed to `f16`.
    #[cfg(feature = "nightly_f16")]
    pub(crate) fn fill_f16_unit(&mut self, len: usize) -> Vec<f16> {
        self.fill_f32_unit(len)
            .into_iter()
            .map(|v| v as f16)
            .collect()
    }
}

/// Floating-point backends legitimately differ from the scalar reference by a
/// few ULPs (different accumulation order under vectorization), so exact
/// equality is the wrong check - unlike the fixed-point `u8` paths, where
/// integer arithmetic must be bit-exact across every backend.
pub(crate) fn assert_f32_slices_close(actual: &[f32], expected: &[f32], atol: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= atol,
            "{context}: index {i}: {a} vs {e} (diff {diff} > atol {atol})"
        );
    }
}

/// Same idea as `assert_f32_slices_close`, comparing `f16` values by widening
/// to `f32` first (fp16 has no ergonomic arithmetic ops on its own here).
#[cfg(feature = "nightly_f16")]
pub(crate) fn assert_f16_slices_close(
    actual: &[f16],
    expected: &[f16],
    atol: f32,
    context: &str,
) {
    let actual_f32: Vec<f32> = actual.iter().map(|&v| v as f32).collect();
    let expected_f32: Vec<f32> = expected.iter().map(|&v| v as f32).collect();
    assert_f32_slices_close(&actual_f32, &expected_f32, atol, context);
}
