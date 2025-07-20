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
use pxfm::{f_cosf, f_sinc, f_sincf, f_sincosf};
use std::ops::Div;

pub(crate) trait Trigonometry {
    fn f_cos(self) -> Self;
    fn f_sincos(self) -> (Self, Self)
    where
        Self: Sized;
}

impl Trigonometry for f32 {
    #[inline]
    fn f_cos(self) -> Self {
        f_cosf(self)
    }

    #[inline]
    fn f_sincos(self) -> (Self, Self) {
        f_sincosf(self)
    }
}

impl Trigonometry for f64 {
    #[inline]
    fn f_cos(self) -> Self {
        pxfm::f_cos(self)
    }

    #[inline]
    fn f_sincos(self) -> (Self, Self) {
        pxfm::f_sincos(self)
    }
}

pub(crate) trait Sinc {
    fn sinc(self) -> Self;
}

impl Sinc for f32 {
    #[inline]
    fn sinc(self) -> Self {
        f_sincf(self)
    }
}

impl Sinc for f64 {
    fn sinc(self) -> Self {
        f_sinc(self)
    }
}

#[inline]
pub(crate) fn sinc<V: Copy + PartialEq + Div<Output = V> + 'static + Sinc + Trigonometry>(x: V) -> V
where
    f32: AsPrimitive<V>,
{
    x.sinc()
}
