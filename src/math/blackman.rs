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
use crate::math::sinc::{Sinc, Trigonometry};
use crate::sinc::sinc;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::ops::{Add, Mul};

#[inline(always)]
pub(crate) fn blackman_window<
    V: Copy
        + ConstPI
        + 'static
        + Mul<Output = V>
        + Float
        + Trigonometry
        + MulAdd<V, Output = V>
        + Add<V, Output = V>,
>(
    d: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    mla(
        0.07684867f32.as_(),
        (2f32.as_() * d).f_cospi(),
        mla(0.49656062f32.as_(), d.f_cospi(), 0.4265907f32.as_()),
    )
}

pub(crate) fn blackman<
    V: Copy
        + ConstPI
        + 'static
        + Mul<Output = V>
        + Trigonometry
        + Float
        + Sinc
        + MulAdd<V, Output = V>
        + Add<V, Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    let x = x.abs();
    if x < 2.0f32.as_() {
        sinc(x) * blackman_window(x / 2f32.as_())
    } else {
        0f32.as_()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Textbook Blackman over an index `t` in [0, 1], peaking at `t = 0.5`.
    fn window_index(t: f64) -> f64 {
        const A0: f64 = 7938.0 / 18608.0;
        const A1: f64 = 9240.0 / 18608.0;
        const A2: f64 = 1430.0 / 18608.0;
        let tau = 2.0 * std::f64::consts::PI * t;
        A0 - A1 * tau.cos() + A2 * (2.0 * tau).cos()
    }

    #[test]
    fn window_peaks_at_the_centre() {
        assert!(
            (blackman_window(0.0f64) - 1.0).abs() < 1e-7,
            "w(0) = {}",
            blackman_window(0.0f64)
        );
        // Monotone falloff away from the centre over the support.
        let mut prev = f64::INFINITY;
        for i in 0..=1000 {
            let d = i as f64 / 1000.0;
            let w = blackman_window(d);
            assert!(w <= prev + 1e-9, "window not decreasing at d = {d}");
            prev = w;
        }
    }

    /// The distance form must be the textbook window, just re-parameterized.
    #[test]
    fn window_matches_textbook_index_form() {
        for i in 0..=1000 {
            let d = i as f64 / 1000.0;
            let got = blackman_window(d);
            let want = window_index(0.5 + d / 2.0);
            assert!((got - want).abs() < 1e-7, "d = {d}: got {got}, want {want}");
        }
    }

    #[test]
    fn kernel_is_an_interpolating_windowed_sinc() {
        // Centre tap is unity...
        assert!(
            (blackman(0.0f64) - 1.0).abs() < 1e-7,
            "w(0) = {}",
            blackman(0.0f64)
        );
        // ...and the kernel vanishes at every other integer, so resampling on
        // an aligned grid is a no-op.
        for k in 1..=3 {
            let v = blackman(k as f64);
            assert!(v.abs() < 1e-7, "blackman({k}) = {v}");
        }
        assert_eq!(blackman(2.5f64), 0.0);
    }

    #[test]
    fn kernel_is_symmetric_and_peaks_at_zero() {
        let peak = blackman(0.0f64);
        for i in 0..=2000 {
            let x = i as f64 / 1000.0;
            assert!(
                blackman(x) <= peak + 1e-9,
                "value exceeds centre tap at {x}"
            );
            assert!(
                (blackman(x) - blackman(-x)).abs() < 1e-12,
                "not symmetric at {x}"
            );
        }
    }
}
