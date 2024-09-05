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

use num_traits::AsPrimitive;
use std::ops::{Add, Div, Mul, Sub};

#[inline(always)]
pub(crate) fn spline16<
    V: Copy
        + Div<Output = V>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + 'static
        + PartialOrd
        + Div<Output = V>,
>(
    x: V,
) -> V
where
    f64: AsPrimitive<V>,
    f32: AsPrimitive<V>,
{
    if x < 1.0.as_() {
        ((x - 9.0.as_() / 5.0.as_()) * x - 1.0.as_() / 5.0.as_()) * x + 1.0.as_()
    } else {
        (((-1.0).as_() / 3.0.as_() * (x - 1f32.as_()) + 4.0.as_() / 5.0.as_()) * (x - 1f32.as_())
            - 7.0.as_() / 15.0.as_())
            * (x - 1f32.as_())
    }
}

#[inline(always)]
pub(crate) fn spline36<
    V: Copy
        + Div<Output = V>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + 'static
        + PartialOrd
        + Div<Output = V>,
>(
    x: V,
) -> V
where
    f64: AsPrimitive<V>,
    f32: AsPrimitive<V>,
{
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

#[inline(always)]
pub(crate) fn spline64<
    V: Copy
        + Div<Output = V>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + 'static
        + PartialOrd
        + Div<Output = V>,
>(
    x: V,
) -> V
where
    f64: AsPrimitive<V>,
    f32: AsPrimitive<V>,
{
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
