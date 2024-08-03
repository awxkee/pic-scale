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
use num_traits::AsPrimitive;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[inline(always)]
pub fn bc_spline<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>,
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
        return ((12f32.as_() - 9f32.as_() * b - 6f32.as_() * c) * tp
            + ((-18f32).as_() + 12f32.as_() * b + 6f32.as_() * c) * dp
            + (6f32.as_() - 2f32.as_() * b))
            * (1f32.as_() / 6f32.as_());
    } else if x < 2f32.as_() {
        return ((-b - 6f32.as_() * c) * tp
            + (6f32.as_() * b + 30f32.as_() * c) * dp
            + ((-12f32).as_() * b - 48f32.as_() * c) * x
            + (8f32.as_() * b + 24f32.as_() * c))
            * (1f32.as_() / 6f32.as_());
    }
    return 0f32.as_();
}

#[inline(always)]
pub fn hermite_spline<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    return bc_spline(x, 0f32.as_(), 0f32.as_());
}

#[inline(always)]
pub fn b_spline<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    return bc_spline(x, 1f32.as_(), 0f32.as_());
}

#[inline(always)]
pub fn mitchell_netravalli<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    return bc_spline(x, 1f32.as_() / 3f32.as_(), 1f32.as_() / 3f32.as_());
}

#[inline(always)]
pub fn catmull_rom<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    return bc_spline(x, 0f32.as_(), 0.5f32.as_());
}

#[inline(always)]
pub fn robidoux<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>
        + ConstSqrt2,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    return bc_spline(
        x,
        12f32.as_() / (19f32.as_() + 9f32.as_() * V::const_sqrt2()),
        13f32.as_() / (58f32.as_() + 216f32.as_() * V::const_sqrt2()),
    );
}

#[inline(always)]
pub fn robidoux_sharp<
    V: Copy
        + Add<Output = V>
        + Mul<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + 'static
        + PartialEq
        + PartialOrd
        + Neg<Output = V>
        + ConstSqrt2,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    return bc_spline(
        x,
        6f32.as_() / (13f32.as_() + 7f32.as_() * V::const_sqrt2()),
        7f32.as_() / (2f32.as_() + 12f32.as_() * V::const_sqrt2()),
    );
}
