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

use crate::math::consts::ConstPI;
use num_traits::{AsPrimitive, Float};
use pxfm::f_expf;
use std::ops::{Mul, Neg};

pub(crate) trait Exponential {
    fn f_exp(self) -> Self;
}

impl Exponential for f32 {
    #[inline]
    fn f_exp(self) -> Self {
        f_expf(self)
    }
}

impl Exponential for f64 {
    fn f_exp(self) -> Self {
        pxfm::f_exp(self)
    }
}

pub(crate) fn gaussian<
    V: ConstPI + Copy + Neg<Output = V> + Mul<Output = V> + 'static + Float + Exponential,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    let sigma: V = 0.35f32.as_();
    let pi = V::const_pi();
    let den = 2f32.as_() * sigma * sigma;
    (1f32.as_() / ((2f32.as_() * pi).sqrt() * sigma)) * (-(x * x) / den).f_exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The kernel builds sigma from an `f32` literal, so the reference must use
    // the same widened value (0.3499999940395355) rather than an exact 0.35.
    const SIGMA: f64 = 0.35f32 as f64;

    /// Independent transcription of the normal density.
    fn normal_pdf(x: f64) -> f64 {
        (-0.5 * (x / SIGMA).powi(2)).exp() / (SIGMA * (2.0 * std::f64::consts::PI).sqrt())
    }

    #[test]
    fn matches_the_normal_density() {
        for i in 0..=4000 {
            let x = i as f64 / 1000.0;
            let (got, want) = (gaussian(x), normal_pdf(x));
            assert!((got - want).abs() < 1e-9, "x = {x}: got {got}, want {want}");
        }
    }

    /// A Gaussian is flat at its centre. The pre-fix kernel used `exp(-x)`
    /// rather than `exp(-x*x)`, which put a cusp there instead.
    #[test]
    fn is_a_bell_not_a_cusp() {
        let h = 1e-6;
        let slope = (gaussian(h) - gaussian(0.0f64)) / h;
        assert!(slope.abs() < 1e-3, "nonzero slope at centre: {slope}");
    }

    /// Pins both sigma and the squared exponent: w(sigma)/w(0) == exp(-1/2).
    #[test]
    fn one_sigma_falloff() {
        let ratio = gaussian(SIGMA) / gaussian(0.0f64);
        assert!(
            (ratio - (-0.5f64).exp()).abs() < 1e-9,
            "w(sigma)/w(0) = {ratio}, expected {}",
            (-0.5f64).exp()
        );
    }

    /// The pre-fix kernel decayed so fast (w(1)/w(0) = 5.8e-8) that after
    /// normalization the filter collapsed to nearest-neighbour.
    #[test]
    fn neighbouring_taps_carry_real_weight() {
        let ratio = gaussian(1.0f64) / gaussian(0.0f64);
        assert!(
            ratio > 1e-3,
            "neighbour tap is negligible: w(1)/w(0) = {ratio}"
        );
    }

    #[test]
    fn is_symmetric_and_decreasing() {
        let mut prev = f64::INFINITY;
        for i in 0..=2000 {
            let x = i as f64 / 1000.0;
            let w = gaussian(x);
            assert!(w <= prev + 1e-12, "not decreasing at {x}");
            assert!((w - gaussian(-x)).abs() < 1e-12, "not symmetric at {x}");
            prev = w;
        }
    }
}
