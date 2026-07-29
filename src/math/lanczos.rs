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
#![allow(clippy::excessive_precision)]
use crate::Jinc;
use crate::math::consts::ConstPI;
use crate::math::sinc::{Sinc, Trigonometry};
use crate::sinc::sinc;
use num_traits::{AsPrimitive, Float};
use std::ops::Div;

#[inline]
pub fn lanczos_jinc<
    V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI + AsPrimitive<f64> + Jinc<V>,
>(
    x: V,
    a: V,
) -> V
where
    f32: AsPrimitive<V>,
    f64: AsPrimitive<V>,
{
    let scale_a: V = 1f32.as_() / a;
    // No special case for `x == 0`: `jinc` is well defined there and pxfm
    // returns 1, so the centre tap correctly evaluates to `jinc(0) * jinc(0) == 1`.
    if x > 16.247661874700962f32.as_() {
        return 0f32.as_();
    }
    if x.abs() < a {
        return V::jinc(x) * V::jinc(x * scale_a);
    }
    0f32.as_()
}

pub(crate) fn lanczos3_jinc<
    V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI + AsPrimitive<f64> + Jinc<V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
    f64: AsPrimitive<V>,
{
    lanczos_jinc(x, 3f32.as_())
}

pub(crate) fn lanczos2_jinc<
    V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI + AsPrimitive<f64> + Jinc<V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
    f64: AsPrimitive<V>,
{
    lanczos_jinc(x, 2f32.as_())
}

pub(crate) fn lanczos4_jinc<
    V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI + AsPrimitive<f64> + Jinc<V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
    f64: AsPrimitive<V>,
{
    lanczos_jinc(x, 4f32.as_())
}

pub(crate) fn lanczos5_jinc<
    V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI + AsPrimitive<f64> + Jinc<V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
    f64: AsPrimitive<V>,
{
    lanczos_jinc(x, 5f32.as_())
}

pub(crate) fn lanczos6_jinc<
    V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI + AsPrimitive<f64> + Jinc<V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
    f64: AsPrimitive<V>,
{
    const A: f32 = 6f32;
    lanczos_jinc(x, A.as_())
}

pub(crate) fn lanczos_sinc<
    V: Copy + PartialEq + Div<Output = V> + 'static + Trigonometry + Float + ConstPI + Sinc,
>(
    x: V,
    a: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    let scale_a: V = 1f32.as_() / a;
    if x.abs() < a {
        return sinc(x) * sinc(x * scale_a);
    }
    0f32.as_()
}

pub(crate) fn lanczos3<
    V: Copy + PartialEq + Div<Output = V> + 'static + Float + Trigonometry + ConstPI + Sinc,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
    f64: AsPrimitive<V>,
{
    lanczos_sinc(x, 3f32.as_())
}

pub(crate) fn lanczos4<
    V: Copy + PartialEq + Div<Output = V> + 'static + Trigonometry + Float + ConstPI + Sinc,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    lanczos_sinc(x, 4f32.as_())
}

pub(crate) fn lanczos5<
    V: Copy + PartialEq + Div<Output = V> + 'static + Trigonometry + Float + ConstPI + Sinc,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    lanczos_sinc(x, 5f32.as_())
}

pub(crate) fn lanczos6<
    V: Copy + PartialEq + Div<Output = V> + 'static + Trigonometry + Float + ConstPI + Sinc,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    lanczos_sinc(x, 6f32.as_())
}

pub(crate) fn lanczos2<
    V: Copy + PartialEq + Div<Output = V> + 'static + Trigonometry + Float + ConstPI + Sinc,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    lanczos_sinc(x, 2f32.as_())
}

#[cfg(test)]
mod tests {
    use super::*;

    const JINC: [(&str, fn(f64) -> f64); 5] = [
        ("lanczos2_jinc", lanczos2_jinc),
        ("lanczos3_jinc", lanczos3_jinc),
        ("lanczos4_jinc", lanczos4_jinc),
        ("lanczos5_jinc", lanczos5_jinc),
        ("lanczos6_jinc", lanczos6_jinc),
    ];

    /// `jinc(0) == 1`, so the centre tap must be 1 — not 0.
    #[test]
    fn centre_tap_is_unity() {
        for (name, kernel) in JINC {
            let w = kernel(0.0);
            assert!((w - 1.0).abs() < 1e-9, "{name}(0) = {w}, expected 1");
        }
        // The sinc siblings already behave; keep them pinned alongside.
        for (name, kernel) in [
            ("lanczos2", lanczos2 as fn(f64) -> f64),
            ("lanczos3", lanczos3),
        ] {
            let w = kernel(0.0);
            assert!((w - 1.0).abs() < 1e-9, "{name}(0) = {w}, expected 1");
        }
    }

    /// The pre-fix `x == 0` early-out made the kernel jump 1.0 -> 0.0 at a
    /// single point. Nothing may discontinuously drop at the centre.
    #[test]
    fn is_continuous_at_the_centre() {
        for (name, kernel) in JINC {
            for eps in [1e-12, 1e-9, 1e-6] {
                let (at_zero, near_zero) = (kernel(0.0), kernel(eps));
                assert!(
                    (at_zero - near_zero).abs() < 1e-5,
                    "{name}: w(0) = {at_zero} but w({eps}) = {near_zero}"
                );
            }
        }
    }

    #[test]
    fn centre_is_the_maximum() {
        for (name, kernel) in JINC {
            let peak = kernel(0.0);
            for i in 0..=6000 {
                let x = i as f64 / 1000.0;
                assert!(
                    kernel(x) <= peak + 1e-12,
                    "{name}: w({x}) exceeds centre tap"
                );
            }
        }
    }

    /// At a 1:1 ratio `weights.rs` produces `dx == 0` for the centre tap, so
    /// after normalization that tap must dominate — this is the case the
    /// pre-fix kernel turned into "reconstruct a pixel from its neighbours only".
    #[test]
    fn identity_ratio_keeps_the_centre_pixel() {
        for (name, kernel) in JINC {
            let raw: Vec<f64> = (-3..=3).map(|k| kernel((k as f64).abs())).collect();
            let sum: f64 = raw.iter().sum();
            let centre = raw[3] / sum;
            assert!(
                centre > 0.5,
                "{name}: centre tap holds only {centre} of the weight at 1:1"
            );
        }
    }

    #[test]
    fn is_symmetric_within_the_support() {
        for (name, kernel) in JINC {
            for i in 0..=6000 {
                let x = i as f64 / 1000.0;
                assert!(
                    (kernel(x) - kernel(-x)).abs() < 1e-12,
                    "{name}: not symmetric at {x}"
                );
            }
        }
    }
}
