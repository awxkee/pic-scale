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
use crate::sinc::sinc;
use num_traits::{AsPrimitive, Float};
use std::ops::Div;

#[inline(always)]
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
    if x == 0f32.as_() || x > 16.247661874700962f32.as_() {
        return 0f32.as_();
    }
    if x.abs() < a {
        let jinc = V::jinc();
        let d = V::const_pi() * x;
        return jinc(d) * jinc(d * scale_a);
    }
    0f32.as_()
}

#[inline(always)]
pub fn lanczos3_jinc<
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

#[inline(always)]
pub fn lanczos2_jinc<
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

#[inline(always)]
pub fn lanczos4_jinc<
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

#[inline(always)]
pub fn lanczos6_jinc<
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

#[inline(always)]
pub fn lanczos_sinc<V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI>(
    x: V,
    a: V,
) -> V
where
    f32: AsPrimitive<V>,
{
    let scale_a: V = 1f32.as_() / a;
    if x.abs() < a {
        let d = V::const_pi() * x;
        return sinc(d) * sinc(d * scale_a);
    }
    0f32.as_()
}

#[inline(always)]
pub fn lanczos3<V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI>(x: V) -> V
where
    f32: AsPrimitive<V>,
    f64: AsPrimitive<V>,
{
    lanczos_sinc(x, 3f32.as_())
}

#[inline(always)]
pub fn lanczos4<V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI>(x: V) -> V
where
    f32: AsPrimitive<V>,
{
    lanczos_sinc(x, 4f32.as_())
}

#[inline(always)]
pub fn lanczos6<V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI>(x: V) -> V
where
    f32: AsPrimitive<V>,
{
    lanczos_sinc(x, 6f32.as_())
}

#[inline(always)]
pub fn lanczos2<V: Copy + PartialEq + Div<Output = V> + 'static + Float + ConstPI>(x: V) -> V
where
    f32: AsPrimitive<V>,
{
    lanczos_sinc(x, 2f32.as_())
}
