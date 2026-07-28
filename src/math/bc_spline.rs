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

use crate::math::consts::ConstSqrt2;
use crate::math::mla;
use num_traits::{AsPrimitive, MulAdd};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[inline]
fn bc_spline<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>
        + MulAdd<Output = V>,
>(
    d: V,
    b: V,
    c: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    let mut x = d;
    if x < 0.0f32.as_() {
        x = -x;
    }
    let dp = x * x;
    let tp = dp * x;
    if x < 1f32.as_() {
        return (mla(-6f32.as_(), c, mla(-9f32.as_(), b, 12f32.as_())) * tp
            + mla(6f32.as_(), c, mla(12f32.as_(), b, -18f32.as_())) * dp
            + mla(-2f32.as_(), b, 6f32.as_()))
            * (1f32.as_() / 6f32.as_());
    } else if x < 2f32.as_() {
        return (mla(-6f32.as_(), c, -b) * tp
            + mla(6f32.as_(), b, 30f32.as_() * c) * dp
            + mla((-12f32).as_(), b, -48f32.as_() * c) * x
            + mla(8f32.as_(), b, 24f32.as_() * c))
            * (1f32.as_() / 6f32.as_());
    }
    0f32.as_()
}

pub(crate) fn hermite_spline<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>
        + MulAdd<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    bc_spline(x, 0f32.as_(), 0f32.as_())
}

pub(crate) fn b_spline<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>
        + MulAdd<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    bc_spline(x, 1f32.as_(), 0f32.as_())
}

pub(crate) fn mitchell_netravalli<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>
        + MulAdd<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    bc_spline(x, 1f32.as_() / 3f32.as_(), 1f32.as_() / 3f32.as_())
}

pub(crate) fn catmull_rom<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>
        + MulAdd<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    bc_spline(x, 0f32.as_(), 0.5f32.as_())
}

pub(crate) fn robidoux<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>
        + ConstSqrt2
        + MulAdd<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    bc_spline(
        x,
        12f32.as_() / mla(9f32.as_(), V::const_sqrt2(), 19f32.as_()),
        113f32.as_() / mla(216f32.as_(), V::const_sqrt2(), 58f32.as_()),
    )
}

pub(crate) fn robidoux_sharp<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>
        + ConstSqrt2
        + MulAdd<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    bc_spline(
        x,
        6f32.as_() / mla(7f32.as_(), V::const_sqrt2(), 13f32.as_()),
        7f32.as_() / mla(12f32.as_(), V::const_sqrt2(), 2f32.as_()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recovers `(B, C)` from a BC-spline kernel by sampling it.
    ///
    /// `w(0) = (6 - 2B)/6` carries only `B`, and `w(1.5) = (0.125B - 0.75C)/6`
    /// then pins `C`. This lets the tests assert the defining constants without
    /// having to expose them.
    fn recover_bc(kernel: impl Fn(f64) -> f64) -> (f64, f64) {
        let b = 3.0 * (1.0 - kernel(0.0));
        let c = (0.125 * b - 6.0 * kernel(1.5)) / 0.75;
        (b, c)
    }

    /// Every filter in the Mitchell family that is meant to be a Keys cubic
    /// satisfies `B + 2C == 1`. Robidoux and Robidoux-Sharp both are.
    #[test]
    fn robidoux_filters_satisfy_keys_invariant() {
        for (name, (b, c)) in [
            ("robidoux", recover_bc(robidoux::<f64>)),
            ("robidoux_sharp", recover_bc(robidoux_sharp::<f64>)),
        ] {
            assert!(
                (b + 2.0 * c - 1.0).abs() < 1e-9,
                "{name}: B + 2C = {} (B = {b}, C = {c}), expected 1",
                b + 2.0 * c
            );
        }
    }

    #[test]
    fn robidoux_matches_published_constants() {
        let (b, c) = recover_bc(robidoux::<f64>);
        assert!((b - 0.378215755093998_67).abs() < 1e-9, "B = {b}");
        assert!((c - 0.310892122453000_67).abs() < 1e-9, "C = {c}");
    }

    #[test]
    fn robidoux_sharp_matches_published_constants() {
        let (b, c) = recover_bc(robidoux_sharp::<f64>);
        assert!((b - 0.2620145123991189).abs() < 1e-9, "B = {b}");
        assert!((c - 0.3689927438004406).abs() < 1e-9, "C = {c}");
    }

    /// A BC-spline must reconstruct a constant signal exactly at any phase.
    #[test]
    fn bc_splines_are_a_partition_of_unity() {
        let kernels: [(&str, fn(f64) -> f64); 6] = [
            ("hermite_spline", hermite_spline),
            ("b_spline", b_spline),
            ("mitchell_netravalli", mitchell_netravalli),
            ("catmull_rom", catmull_rom),
            ("robidoux", robidoux),
            ("robidoux_sharp", robidoux_sharp),
        ];
        for (name, kernel) in kernels {
            for i in 0..64 {
                let phase = i as f64 / 64.0;
                let sum: f64 = (-2..=2).map(|k| kernel(k as f64 + phase)).sum();
                assert!(
                    (sum - 1.0).abs() < 1e-9,
                    "{name} at phase {phase}: sum = {sum}"
                );
            }
        }
    }

    /// Robidoux is a sharpening filter: it must have a real negative lobe.
    /// The pre-fix constant left one ~46x too shallow to do anything.
    #[test]
    fn robidoux_has_a_negative_lobe() {
        let min = (0..=2000)
            .map(|i| robidoux(i as f64 / 1000.0))
            .fold(f64::INFINITY, f64::min);
        assert!(min < -0.02, "negative lobe too shallow: min = {min}");
    }
}
