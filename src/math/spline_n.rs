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

use num_traits::{AsPrimitive, Float};
use std::ops::{Add, Div, Mul, Sub};

pub(crate) fn spline16<
    V: Copy
        + Div<Output = V>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + 'static
        + PartialOrd
        + Float
        + Div<Output = V>,
>(
    x: V,
) -> V
where
    f64: AsPrimitive<V>,
    f32: AsPrimitive<V>,
{
    // The piecewise polynomial is only defined on |x| < 2; past that the last
    // branch keeps extrapolating instead of clamping. `weights.rs` evaluates
    // taps out to |dx| = radius + 0.5, so the tail is reached in practice.
    let x = x.abs();
    if x >= 2f32.as_() {
        return 0f32.as_();
    }
    if x < 1.0.as_() {
        ((x - 9.0.as_() / 5.0.as_()) * x - 1.0.as_() / 5.0.as_()) * x + 1.0.as_()
    } else {
        (((-1.0).as_() / 3.0.as_() * (x - 1f32.as_()) + 4.0.as_() / 5.0.as_()) * (x - 1f32.as_())
            - 7.0.as_() / 15.0.as_())
            * (x - 1f32.as_())
    }
}

pub(crate) fn spline36<
    V: Copy
        + Div<Output = V>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + 'static
        + PartialOrd
        + Float
        + Div<Output = V>,
>(
    x: V,
) -> V
where
    f64: AsPrimitive<V>,
    f32: AsPrimitive<V>,
{
    // Defined on |x| < 3 only; see the note on `spline16`.
    let x = x.abs();
    if x >= 3f32.as_() {
        return 0f32.as_();
    }
    if x < 1.0.as_() {
        ((13.0.as_() / 11.0.as_() * x - 453.0.as_() / 209.0.as_()) * x - 3.0.as_() / 209.0.as_())
            * x
            + 1.0.as_()
    } else if x < 2.0.as_() {
        (((-6.0).as_() / 11.0.as_() * (x - 1f32.as_()) + 270.0.as_() / 209.0.as_())
            * (x - 1f32.as_())
            - 156.0.as_() / 209.0.as_())
            * (x - 1f32.as_())
    } else {
        ((1.0.as_() / 11.0.as_() * (x - 2f32.as_()) - 45.0.as_() / 209.0.as_()) * (x - 2f32.as_())
            + 26.0.as_() / 209.0.as_())
            * (x - 2f32.as_())
    }
}

pub(crate) fn spline64<
    V: Copy
        + Div<Output = V>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + 'static
        + PartialOrd
        + Float
        + Div<Output = V>,
>(
    x: V,
) -> V
where
    f64: AsPrimitive<V>,
    f32: AsPrimitive<V>,
{
    // Defined on |x| < 4 only; see the note on `spline16`.
    let x = x.abs();
    if x >= 4f32.as_() {
        return 0f32.as_();
    }
    if x < 1.0.as_() {
        ((49.0.as_() / 41.0.as_() * x - 6387.0.as_() / 2911.0.as_()) * x - 3.0.as_() / 2911.0.as_())
            * x
            + 1.0.as_()
    } else if x < 2.0.as_() {
        (((-24.0).as_() / 41.0.as_() * (x - 1f32.as_()) + 4032.0.as_() / 2911.0.as_())
            * (x - 1f32.as_())
            - 2328.0.as_() / 2911.0.as_())
            * (x - 1f32.as_())
    } else if x < 3.0.as_() {
        ((6.0.as_() / 41.0.as_() * (x - 2f32.as_()) - 1008.0.as_() / 2911.0.as_())
            * (x - 2f32.as_())
            + 582.0.as_() / 2911.0.as_())
            * (x - 2f32.as_())
    } else {
        (((-1.0).as_() / 41.0.as_() * (x - 3f32.as_()) + 168.0.as_() / 2911.0.as_())
            * (x - 3f32.as_())
            - 97.0.as_() / 2911.0.as_())
            * (x - 3f32.as_())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (name, kernel, support radius)
    const SPLINES: [(&str, fn(f64) -> f64, f64); 3] = [
        ("spline16", spline16, 2.0),
        ("spline36", spline36, 3.0),
        ("spline64", spline64, 4.0),
    ];

    /// The polynomials are only defined inside their support; outside they used
    /// to keep extrapolating (spline64(6) reached -0.239).
    #[test]
    fn vanishes_outside_the_support() {
        for (name, kernel, support) in SPLINES {
            for i in 0..=4000 {
                let x = support + i as f64 / 1000.0;
                let w = kernel(x);
                assert!(w == 0.0, "{name}({x}) = {w}, expected 0");
            }
        }
    }

    /// Clamping must not introduce a step: each polynomial already reaches 0
    /// at its support edge.
    #[test]
    fn is_continuous_at_the_support_edge() {
        for (name, kernel, support) in SPLINES {
            let inside = kernel(support - 1e-9);
            assert!(
                inside.abs() < 1e-8,
                "{name}: w(support-) = {inside}, so clamping to 0 creates a step"
            );
        }
    }

    #[test]
    fn centre_tap_is_unity_and_kernel_is_even() {
        for (name, kernel, support) in SPLINES {
            let w0 = kernel(0.0);
            assert!((w0 - 1.0).abs() < 1e-12, "{name}(0) = {w0}, expected 1");
            for i in 0..=((support + 1.0) * 1000.0) as i32 {
                let x = i as f64 / 1000.0;
                assert!(
                    (kernel(x) - kernel(-x)).abs() < 1e-12,
                    "{name}: not symmetric at {x}"
                );
            }
        }
    }

    /// A resampling kernel must reconstruct a constant exactly at any phase.
    #[test]
    fn is_a_partition_of_unity() {
        for (name, kernel, support) in SPLINES {
            let r = support.ceil() as i32;
            for i in 0..64 {
                let phase = i as f64 / 64.0;
                let sum: f64 = (-r..=r).map(|k| kernel(k as f64 + phase)).sum();
                assert!(
                    (sum - 1.0).abs() < 1e-9,
                    "{name} at phase {phase}: sum = {sum}"
                );
            }
        }
    }

    /// The taps beyond the support were not small: for spline64 the largest
    /// spurious value (0.455) exceeded the deepest genuine lobe (-0.136).
    #[test]
    fn no_tap_beyond_support_can_rival_the_real_lobes() {
        for (name, kernel, support) in SPLINES {
            // radius + 0.5 is the furthest weights.rs ever evaluates
            let reach = support + 2.5;
            let mut worst: f64 = 0.0;
            let mut x = support;
            while x <= reach {
                worst = worst.max(kernel(x).abs());
                x += 0.001;
            }
            assert!(worst == 0.0, "{name}: spurious tap of {worst} past support");
        }
    }
}
