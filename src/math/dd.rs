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
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Copy, Clone)]
pub(crate) struct DoubleDouble {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

impl AsPrimitive<DoubleDouble> for f64 {
    #[inline]
    fn as_(self) -> DoubleDouble {
        DoubleDouble::from_f64(self)
    }
}

impl DoubleDouble {
    // Non FMA helper
    #[allow(dead_code)]
    #[inline]
    pub(crate) const fn split(a: f64) -> DoubleDouble {
        // CN = 2^N.
        const CN: f64 = (1 << 27) as f64;
        const C: f64 = CN + 1.0;
        let t1 = C * a;
        let t2 = a - t1;
        let r_hi = t1 + t2;
        let r_lo = a - r_hi;
        DoubleDouble::new(r_lo, r_hi)
    }

    // Non FMA helper
    #[allow(dead_code)]
    #[inline]
    fn from_exact_mult_impl_non_fma(asz: DoubleDouble, a: f64, b: f64) -> Self {
        let bs = DoubleDouble::split(b);

        let r_hi = a * b;
        let t1 = asz.hi * bs.hi - r_hi;
        let t2 = asz.hi * bs.lo + t1;
        let t3 = asz.lo * bs.hi + t2;
        let r_lo = asz.lo * bs.lo + t3;
        DoubleDouble::new(r_lo, r_hi)
    }

    #[inline]
    pub(crate) const fn from_f64(x: f64) -> Self {
        DoubleDouble { lo: 0., hi: x }
    }

    #[inline]
    pub(crate) const fn from_full_exact_add(a: f64, b: f64) -> DoubleDouble {
        let r_hi = a + b;
        let t1 = r_hi - a;
        let t2 = r_hi - t1;
        let t3 = b - t1;
        let t4 = a - t2;
        let r_lo = t3 + t4;
        DoubleDouble::new(r_lo, r_hi)
    }

    #[inline]
    pub(crate) const fn new(lo: f64, hi: f64) -> Self {
        DoubleDouble { lo, hi }
    }

    #[inline]
    pub(crate) fn full_dd_add(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
        let DoubleDouble { hi: sh, lo: sl } = DoubleDouble::from_full_exact_add(a.hi, b.hi);
        let DoubleDouble { hi: th, lo: tl } = DoubleDouble::from_full_exact_add(a.lo, b.lo);
        let c = sl + th;
        let v = DoubleDouble::from_exact_add(sh, c);
        let w = tl + v.lo;
        DoubleDouble::from_exact_add(v.hi, w)
    }

    // valid only for |a| > b
    #[inline]
    pub(crate) const fn from_exact_add(a: f64, b: f64) -> DoubleDouble {
        let r_hi = a + b;
        let t = r_hi - a;
        let r_lo = b - t;
        DoubleDouble::new(r_lo, r_hi)
    }

    #[inline]
    pub(crate) fn full_dd_sub(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
        DoubleDouble::full_dd_add(a, -b)
    }

    #[inline]
    pub(crate) fn from_exact_mult(a: f64, b: f64) -> Self {
        #[cfg(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            target_arch = "aarch64"
        ))]
        {
            let r_hi = a * b;
            let r_lo = f64::mul_add(a, b, -r_hi);
            DoubleDouble::new(r_lo, r_hi)
        }
        #[cfg(not(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            target_arch = "aarch64"
        )))]
        {
            let splat = DoubleDouble::split(a);
            DoubleDouble::from_exact_mult_impl_non_fma(splat, a, b)
        }
    }

    #[inline]
    pub(crate) fn quick_mult(a: DoubleDouble, b: DoubleDouble) -> Self {
        #[cfg(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            target_arch = "aarch64"
        ))]
        {
            let mut r = DoubleDouble::from_exact_mult(a.hi, b.hi);
            let t1 = f64::mul_add(a.hi, b.lo, r.lo);
            let t2 = f64::mul_add(a.lo, b.hi, t1);
            r.lo = t2;
            r
        }
        #[cfg(not(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            target_arch = "aarch64"
        )))]
        {
            let DoubleDouble { hi: ch, lo: cl1 } = DoubleDouble::from_exact_mult(a.hi, b.hi);
            let tl1 = a.hi * b.lo;
            let tl2 = a.lo * b.hi;
            let cl2 = tl1 + tl2;
            let cl3 = cl1 + cl2;
            DoubleDouble::new(cl3, ch)
        }
    }

    #[inline]
    pub(crate) fn div(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
        let q = 1.0 / b.hi;
        let r_hi = a.hi * q;
        #[cfg(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            target_arch = "aarch64"
        ))]
        {
            let e_hi = f64::mul_add(b.hi, -r_hi, a.hi);
            let e_lo = f64::mul_add(b.lo, -r_hi, a.lo);
            let r_lo = q * (e_hi + e_lo);
            DoubleDouble::new(r_lo, r_hi)
        }
        #[cfg(not(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            target_arch = "aarch64"
        )))]
        {
            let b_hi_r_hi = DoubleDouble::from_exact_mult(b.hi, -r_hi);
            let b_lo_r_hi = DoubleDouble::from_exact_mult(b.lo, -r_hi);
            let e_hi = (a.hi + b_hi_r_hi.hi) + b_hi_r_hi.lo;
            let e_lo = (a.lo + b_lo_r_hi.hi) + b_lo_r_hi.lo;
            let r_lo = q * (e_hi + e_lo);
            DoubleDouble::new(r_lo, r_hi)
        }
    }
}

impl Neg for DoubleDouble {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        DoubleDouble::new(-self.lo, -self.hi)
    }
}

impl Add<DoubleDouble> for DoubleDouble {
    type Output = DoubleDouble;
    #[inline]
    fn add(self, rhs: DoubleDouble) -> Self::Output {
        DoubleDouble::full_dd_add(self, rhs)
    }
}

impl AddAssign for DoubleDouble {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = DoubleDouble::full_dd_add(*self, rhs)
    }
}

impl Sub<DoubleDouble> for DoubleDouble {
    type Output = DoubleDouble;
    #[inline]
    fn sub(self, rhs: DoubleDouble) -> Self::Output {
        DoubleDouble::full_dd_sub(self, rhs)
    }
}

impl Mul<DoubleDouble> for DoubleDouble {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: DoubleDouble) -> Self::Output {
        let r = DoubleDouble::quick_mult(self, rhs);
        DoubleDouble::from_exact_add(r.hi, r.lo)
    }
}

impl Div<DoubleDouble> for DoubleDouble {
    type Output = Self;

    #[inline]
    fn div(self, rhs: DoubleDouble) -> Self::Output {
        let r = DoubleDouble::div(self, rhs);
        DoubleDouble::from_exact_add(r.hi, r.lo)
    }
}

impl AsPrimitive<f64> for DoubleDouble {
    #[inline]
    fn as_(self) -> f64 {
        self.lo + self.hi
    }
}

impl AsPrimitive<f32> for DoubleDouble {
    #[inline]
    fn as_(self) -> f32 {
        (self.lo + self.hi) as f32
    }
}

impl AsPrimitive<DoubleDouble> for f32 {
    #[inline]
    fn as_(self) -> DoubleDouble {
        DoubleDouble::from_f64(self as f64)
    }
}

impl PartialEq for DoubleDouble {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}

impl PartialOrd for DoubleDouble {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match self.hi.partial_cmp(&other.hi) {
            Some(Ordering::Less) => Some(Ordering::Less),
            Some(Ordering::Greater) => Some(Ordering::Greater),
            Some(Ordering::Equal) => self.lo.partial_cmp(&other.lo),
            None => None,
        }
    }
}
