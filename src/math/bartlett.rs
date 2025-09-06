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
    let l = 2.0f32.as_();
    let fac = (x / (l - 1.0f32.as_()) - 0.5f32.as_()).abs();
    mla(
        0.38f32.as_(),
        (2f32.as_() * fac).f_cospi(),
        mla(-0.4832.as_(), fac, 0.62f32.as_()),
    )
}
