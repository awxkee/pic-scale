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

use crate::math::mla;
use num_traits::{AsPrimitive, MulAdd, Signed};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub(crate) fn cubic_spline<
    V: Copy
        + Mul<Output = V>
        + Add<Output = V>
        + Neg<Output = V>
        + Sub<Output = V>
        + PartialOrd
        + PartialEq
        + Div<Output = V>
        + MulAdd<V, Output = V>
        + 'static,
>(
    d: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    let mut x = d;
    if x < 0f32.as_() {
        x = -x;
    }
    if x < 1f32.as_() {
        return mla(x * x, mla(3f32.as_(), x, -6f32.as_()), 4f32.as_()) * (1f32.as_() / 6f32.as_());
    } else if x < 2f32.as_() {
        return mla(x, mla(x, 6f32.as_() - x, (-12f32).as_()), 8f32.as_())
            * (1f32.as_() / 6f32.as_());
    }
    0f32.as_()
}

pub(crate) fn bicubic_spline<
    V: Copy
        + Mul<Output = V>
        + Sub<Output = V>
        + Add<Output = V>
        + 'static
        + Neg<Output = V>
        + Signed
        + PartialOrd
        + MulAdd<V, Output = V>,
>(
    d: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    let x = d;
    let a = (-0.5).as_();
    let modulo = x.abs();
    if modulo >= 2f32.as_() {
        return 0f32.as_();
    }
    let floatd = modulo * modulo;
    let triplet = floatd * modulo;
    if modulo <= 1f32.as_() {
        return mla(
            a + 2f32.as_(),
            triplet,
            mla(-(a + 3f32.as_()), floatd, 1f32.as_()),
        );
    }
    mla(
        a,
        triplet,
        mla(
            -5f32.as_(),
            a,
            mla(-4f32.as_(), a, mla(8f32.as_() * a, modulo, floatd)),
        ),
    )
}
