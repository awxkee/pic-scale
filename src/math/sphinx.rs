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
use num_traits::{AsPrimitive, Float, MulAdd, Signed};

pub(crate) fn sphinx<
    V: Copy + ConstPI + Signed + Float + 'static + Trigonometry + MulAdd<V, Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    // sphinx(x) = 3 * (sin(pi*x) - pi*x*cos(pi*x)) / (pi*x)^3.
    // `f_sincospi` folds pi into its own argument, but the `pi*x` factor on the
    // cosine and the `(pi*x)^3` denominator still have to carry it explicitly.
    let px = x * V::const_pi();
    let t2 = px * px;
    if t2 < 0.01f32.as_() {
        // For |pi*x| < 0.1 the subtraction `sin t - t*cos t` cancels to roughly
        // t^3/3, losing ~3*eps/t^2 of relative precision — at x = 1e-8 that is a
        // total loss. The Maclaurin series is accurate to ~4e-14 across this
        // range and, unlike a flat `return 1`, leaves no discontinuity.
        return 1f32.as_() - t2 / 10f32.as_() + t2 * t2 / 280f32.as_()
            - t2 * t2 * t2 / 15120f32.as_();
    }
    let (x_sin, x_cos) = x.f_sincospi();
    3.0f32.as_() * mla(-px, x_cos, x_sin) / (px * px * px)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Independent transcription: 3 * (sin(t) - t*cos(t)) / t^3, with t = pi*x.
    fn reference(x: f64) -> f64 {
        let t = PI * x;
        3.0 * (t.sin() - t * t.cos()) / (t * t * t)
    }

    #[test]
    fn matches_the_reference_kernel() {
        // Compared above the series crossover, where direct evaluation of the
        // reference is itself free of cancellation.
        for i in 100..=4000 {
            let x = i as f64 / 1000.0;
            let (got, want) = (sphinx(x), reference(x));
            assert!((got - want).abs() < 1e-9, "x = {x}: got {got}, want {want}");
        }
    }

    /// The kernel tends to 1 at the centre. The pre-fix version dropped the
    /// `pi` factors, so it diverged like 3*(pi-1)/x^2 instead.
    #[test]
    fn tends_to_one_at_the_centre() {
        assert_eq!(sphinx(0.0f64), 1.0);
        for x in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3] {
            let w = sphinx(x);
            assert!((w - 1.0).abs() < 1e-5, "sphinx({x}) = {w}, expected ~1");
        }
    }

    /// The old `|x| < 1e-8 -> 1` guard left a 36% cliff at the boundary
    /// (1.0 just inside, 0.640 just outside). Nothing may jump anywhere.
    #[test]
    fn has_no_discontinuity() {
        let mut prev = sphinx(0.0f64);
        for i in 1..=200_000 {
            // dense sweep through the series/direct crossover at |pi*x| = 0.1
            let x = i as f64 * 1e-5;
            let w = sphinx(x);
            assert!(
                (w - prev).abs() < 1e-4,
                "jump of {} at x = {x} ({prev} -> {w})",
                (w - prev).abs()
            );
            prev = w;
        }
        // Across the decades spanned by the old guard the kernel must stay
        // pinned near 1. This is what used to fail on both counts: the missing
        // `pi` gave 6.4e6 at x = 1e-3, and the guard gave 0.640 at x = 1e-8.
        for e in 20..=120 {
            let x = 10f64.powf(-(e as f64) / 10.0); // 1e-2 down to 1e-12
            let w = sphinx(x);
            assert!((w - 1.0).abs() < 1e-3, "sphinx({x}) = {w}, expected ~1");
        }
    }

    /// Closed forms at the half-integer and integer: 24/pi^3 and 3/pi^2.
    #[test]
    fn known_closed_form_values() {
        assert!((sphinx(0.5f64) - 24.0 / PI.powi(3)).abs() < 1e-12);
        assert!((sphinx(1.0f64) - 3.0 / PI.powi(2)).abs() < 1e-12);
    }

    /// A normalized kernel never exceeds its centre tap.
    #[test]
    fn is_bounded_by_the_centre_tap() {
        for i in 0..=4000 {
            let x = i as f64 / 1000.0;
            let w = sphinx(x);
            assert!(w <= 1.0 + 1e-12, "sphinx({x}) = {w} exceeds 1");
            assert!((w - sphinx(-x)).abs() < 1e-12, "not symmetric at {x}");
        }
    }
}
