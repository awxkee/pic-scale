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
use std::ops::{Add, Div, Mul, MulAssign, Sub};

#[inline(always)]
pub(crate) fn lagrange<
    V: Copy
        + PartialEq
        + PartialOrd
        + AsPrimitive<usize>
        + Mul<Output = V>
        + Add<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + MulAssign,
>(
    x: V,
    support: usize,
) -> V
where
    f32: AsPrimitive<V>,
    usize: AsPrimitive<V>,
{
    if x > support.as_() {
        return 0f32.as_();
    }
    let order = (2.0f32.as_() * support.as_()).as_();
    let n = (support.as_() + x).as_();
    let mut value = 1.0f32.as_();
    for i in 0..order {
        if i != n {
            value *= (n.as_() - i.as_() - x) / (n.as_() - i.as_());
        }
    }
    value
}

#[inline(always)]
pub(crate) fn lagrange2<
    V: Copy
        + PartialEq
        + PartialOrd
        + AsPrimitive<usize>
        + Mul<Output = V>
        + Add<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + MulAssign,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
    usize: AsPrimitive<V>,
{
    lagrange(x, 2)
}

#[inline(always)]
pub(crate) fn lagrange3<
    V: Copy
        + PartialEq
        + PartialOrd
        + AsPrimitive<usize>
        + Mul<Output = V>
        + Add<Output = V>
        + Sub<Output = V>
        + Div<Output = V>
        + MulAssign,
>(
    x: V,
) -> V
where
    f32: AsPrimitive<V>,
    usize: AsPrimitive<V>,
{
    lagrange(x, 3)
}
