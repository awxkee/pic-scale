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
use crate::math::mla;
use crate::math::sinc::Trigonometry;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::ops::{Add, Mul, Neg, Sub};

pub(crate) fn bartlett<
    V: Copy
        + Sub<Output = V>
        + Mul<Output = V>
        + 'static
        + PartialOrd
        + MulAdd<V, Output = V>
        + Add<V, Output = V>
        + Neg<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    if x >= 0f32.as_() && x <= 1f32.as_() {
        return 2f32.as_() * x;
    }
    mla(-2f32.as_(), x, 2f32.as_())
}

pub(crate) fn bartlett_hann<
    V: Copy
        + Sub<Output = V>
        + Mul<Output = V>
        + Float
        + ConstPI
        + 'static
        + Trigonometry
        + Add<V, Output = V>
        + MulAdd<V, Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    let x = x.abs();
    if x > 2f32.as_() {
        return 0f32.as_();
    }
    // The window is defined on an index `t = n/(N-1)` in [0, 1] as
    // `0.62 - 0.48*|t - 0.5| - 0.38*cos(2*pi*t)`, but resampling passes a distance
    // from the centre. Re-expressed over the normalized distance `d = |x|/R`,
    // `|t - 0.5|` becomes `d/2` (halving the linear coefficient) and the cosine
    // folds to `+0.38*cos(pi*d)`, giving w(0) = 1 and w(R) = 0.
    let d = x * 0.5f32.as_();
    mla(
        0.38f32.as_(),
        d.f_cospi(),
        mla(-0.24f32.as_(), d, 0.62f32.as_()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference Bartlett-Hann window in its textbook index form,
    /// `t = n/(N-1)` over [0, 1].
    fn reference_index(t: f64) -> f64 {
        0.62 - 0.48 * (t - 0.5).abs() - 0.38 * (2.0 * std::f64::consts::PI * t).cos()
    }

    #[test]
    fn bartlett_hann_matches_reference_window() {
        // Distance `d` from the centre corresponds to index `0.5 + d/2`.
        for i in 0..=1000 {
            let d = i as f64 / 1000.0;
            let got = bartlett_hann(d * 2.0);
            let want = reference_index(0.5 + d / 2.0);
            assert!((got - want).abs() < 1e-6, "d = {d}: got {got}, want {want}");
        }
    }

    #[test]
    fn bartlett_hann_endpoints_and_symmetry() {
        assert!((bartlett_hann(0.0f64) - 1.0).abs() < 1e-6);
        assert!(bartlett_hann(2.0f64).abs() < 1e-6);
        assert_eq!(bartlett_hann(2.5f64), 0.0);

        for i in 0..=200 {
            let x = i as f64 / 100.0;
            assert!(
                (bartlett_hann(x) - bartlett_hann(-x)).abs() < 1e-12,
                "not symmetric at {x}"
            );
        }
    }

    #[test]
    fn bartlett_hann_is_non_negative_and_decreasing() {
        let mut prev = f64::INFINITY;
        for i in 0..=200 {
            let x = i as f64 / 100.0;
            let w = bartlett_hann(x);
            assert!(w >= -1e-6, "negative tap {w} at {x}");
            assert!(w <= prev + 1e-9, "not decreasing at {x}");
            prev = w;
        }
    }
}
