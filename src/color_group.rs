/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#![allow(dead_code)]
use crate::saturate_narrow::SaturateNarrow;
use num_traits::{AsPrimitive, FromPrimitive, MulAdd, Num};
use std::ops::{Add, AddAssign, Mul, Shr, ShrAssign, Sub, SubAssign};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ColorGroup<const COMPS: usize, J: Copy> {
    pub r: J,
    pub g: J,
    pub b: J,
    pub a: J,
}

impl<const COMPS: usize, J> ColorGroup<COMPS, J>
where
    J: Copy + Default,
{
    #[inline]
    pub fn new() -> ColorGroup<COMPS, J> {
        ColorGroup {
            r: J::default(),
            g: J::default(),
            b: J::default(),
            a: J::default(),
        }
    }

    #[inline]
    pub fn from_components(r: J, g: J, b: J, a: J) -> ColorGroup<COMPS, J> {
        ColorGroup { r, g, b, a }
    }

    #[inline]
    pub fn dup(v: J) -> ColorGroup<COMPS, J> {
        ColorGroup {
            r: v,
            g: v,
            b: v,
            a: v,
        }
    }
}

impl<const COMPS: usize, J> ColorGroup<COMPS, J>
where
    J: Copy + Default + 'static,
{
    #[inline]
    pub fn from_slice<T>(store: &[T], offset: usize) -> ColorGroup<COMPS, J>
    where
        T: AsPrimitive<J>,
    {
        unsafe {
            if COMPS == 1 {
                ColorGroup {
                    r: (*store.get_unchecked(offset)).as_(),
                    g: J::default(),
                    b: J::default(),
                    a: J::default(),
                }
            } else if COMPS == 2 {
                ColorGroup {
                    r: (*store.get_unchecked(offset)).as_(),
                    g: (*store.get_unchecked(offset + 1)).as_(),
                    b: J::default(),
                    a: J::default(),
                }
            } else if COMPS == 3 {
                ColorGroup {
                    r: (*store.get_unchecked(offset)).as_(),
                    g: (*store.get_unchecked(offset + 1)).as_(),
                    b: (*store.get_unchecked(offset + 2)).as_(),
                    a: J::default(),
                }
            } else if COMPS == 4 {
                ColorGroup {
                    r: (*store.get_unchecked(offset)).as_(),
                    g: (*store.get_unchecked(offset + 1)).as_(),
                    b: (*store.get_unchecked(offset + 2)).as_(),
                    a: (*store.get_unchecked(offset + 3)).as_(),
                }
            } else {
                panic!("Not implemented.")
            }
        }
    }

    #[inline]
    pub fn from_ptr<T>(store: *const T, offset: usize) -> ColorGroup<COMPS, J>
    where
        T: AsPrimitive<J>,
    {
        unsafe {
            let l_ptr = store.add(offset);
            if COMPS == 1 {
                ColorGroup {
                    r: l_ptr.read_unaligned().as_(),
                    g: J::default(),
                    b: J::default(),
                    a: J::default(),
                }
            } else if COMPS == 2 {
                ColorGroup {
                    r: l_ptr.read_unaligned().as_(),
                    g: l_ptr.add(1).read_unaligned().as_(),
                    b: J::default(),
                    a: J::default(),
                }
            } else if COMPS == 3 {
                ColorGroup {
                    r: l_ptr.read_unaligned().as_(),
                    g: l_ptr.add(1).read_unaligned().as_(),
                    b: l_ptr.add(2).read_unaligned().as_(),
                    a: J::default(),
                }
            } else if COMPS == 4 {
                ColorGroup {
                    r: l_ptr.read_unaligned().as_(),
                    g: l_ptr.add(1).read_unaligned().as_(),
                    b: l_ptr.add(2).read_unaligned().as_(),
                    a: l_ptr.add(3).read_unaligned().as_(),
                }
            } else {
                panic!("Not implemented.")
            }
        }
    }

    #[inline]
    pub fn to_ptr(self, ptr: *mut J, offset: usize) {
        unsafe {
            let s_ptr = ptr.add(offset);
            s_ptr.write_unaligned(self.r);
            if COMPS > 1 {
                s_ptr.add(1).write_unaligned(self.g);
            }
            if COMPS > 2 {
                s_ptr.add(2).write_unaligned(self.b);
            }
            if COMPS == 4 {
                s_ptr.add(3).write_unaligned(self.a);
            }
        }
    }

    #[inline]
    pub fn as_ptr<V: Copy + 'static>(self, ptr: *mut V, offset: usize)
    where
        J: Copy + AsPrimitive<V>,
    {
        unsafe {
            let s_ptr = ptr.add(offset);
            s_ptr.write_unaligned(self.r.as_());
            if COMPS > 1 {
                s_ptr.add(1).write_unaligned(self.g.as_());
            }
            if COMPS > 2 {
                s_ptr.add(2).write_unaligned(self.b.as_());
            }
            if COMPS == 4 {
                s_ptr.add(3).write_unaligned(self.a.as_());
            }
        }
    }
}

impl<const COMPS: usize, J> ColorGroup<COMPS, J>
where
    J: Copy + Default + 'static + Num + Ord,
{
    #[inline]
    pub fn min_scalar(&self, other: J) -> ColorGroup<COMPS, J> {
        if COMPS == 1 {
            ColorGroup::from_components(self.r.min(other), J::default(), J::default(), J::default())
        } else if COMPS == 2 {
            ColorGroup::from_components(
                self.r.min(other),
                self.g.min(other),
                J::default(),
                J::default(),
            )
        } else if COMPS == 3 {
            ColorGroup::from_components(
                self.r.min(other),
                self.g.min(other),
                self.b.min(other),
                J::default(),
            )
        } else {
            ColorGroup::from_components(
                self.r.min(other),
                self.g.min(other),
                self.b.min(other),
                self.a.min(other),
            )
        }
    }

    #[inline]
    pub(crate) fn max_scalar(&self, other: J) -> ColorGroup<COMPS, J> {
        if COMPS == 1 {
            ColorGroup::from_components(self.r.max(other), J::default(), J::default(), J::default())
        } else if COMPS == 2 {
            ColorGroup::from_components(
                self.r.max(other),
                self.g.max(other),
                J::default(),
                J::default(),
            )
        } else if COMPS == 3 {
            ColorGroup::from_components(
                self.r.max(other),
                self.g.max(other),
                self.b.max(other),
                J::default(),
            )
        } else {
            ColorGroup::from_components(
                self.r.max(other),
                self.g.max(other),
                self.b.max(other),
                self.a.max(other),
            )
        }
    }
}

impl<const COMPS: usize, J> Mul<J> for ColorGroup<COMPS, J>
where
    J: Copy + Mul<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r * rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r * rhs, self.g * rhs, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a * rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> ColorGroup<COMPS, J>
where
    J: Copy + Default + 'static,
{
    #[inline]
    pub fn saturate_narrow<V>(&self) -> ColorGroup<COMPS, V>
    where
        V: Copy + Default,
        J: SaturateNarrow<V>,
    {
        if COMPS == 1 {
            ColorGroup::<COMPS, V>::from_components(
                self.r.saturate_narrow(),
                V::default(),
                V::default(),
                V::default(),
            )
        } else if COMPS == 2 {
            ColorGroup::<COMPS, V>::from_components(
                self.r.saturate_narrow(),
                self.g.saturate_narrow(),
                V::default(),
                V::default(),
            )
        } else if COMPS == 3 {
            ColorGroup::<COMPS, V>::from_components(
                self.r.saturate_narrow(),
                self.g.saturate_narrow(),
                self.b.saturate_narrow(),
                V::default(),
            )
        } else {
            ColorGroup::<COMPS, V>::from_components(
                self.r.saturate_narrow(),
                self.g.saturate_narrow(),
                self.b.saturate_narrow(),
                self.a.saturate_narrow(),
            )
        }
    }
}

impl<const COMPS: usize, J> Mul<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + Mul<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: ColorGroup<COMPS, J>) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r * rhs.r, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r * rhs.r, self.g * rhs.g, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r * rhs.r, self.g * rhs.g, self.b * rhs.b, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(
                self.r * rhs.r,
                self.g * rhs.g,
                self.b * rhs.b,
                self.a * rhs.b,
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Sub<J> for ColorGroup<COMPS, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r - rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r - rhs, self.g - rhs, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a - rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Sub<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: ColorGroup<COMPS, J>) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r - rhs.r, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r - rhs.r, self.g - rhs.g, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(
                self.r - rhs.r,
                self.g - rhs.g,
                self.b - rhs.b,
                self.a - rhs.a,
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Add<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + Add<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: ColorGroup<COMPS, J>) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r + rhs.r, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r + rhs.r, self.g + rhs.g, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(
                self.r + rhs.r,
                self.g + rhs.g,
                self.b + rhs.b,
                self.a + rhs.a,
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Add<J> for ColorGroup<COMPS, J>
where
    J: Copy + Add<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r + rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r + rhs, self.g + rhs, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r + rhs, self.g + rhs, self.b + rhs, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(self.r + rhs, self.g + rhs, self.b + rhs, self.a + rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> Shr<J> for ColorGroup<COMPS, J>
where
    J: Copy + Shr<J, Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn shr(self, rhs: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r >> rhs, self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(self.r >> rhs, self.g >> rhs, self.b, self.a)
        } else if COMPS == 3 {
            ColorGroup::from_components(self.r >> rhs, self.g >> rhs, self.b >> rhs, self.a)
        } else if COMPS == 4 {
            ColorGroup::from_components(self.r >> rhs, self.g >> rhs, self.b >> rhs, self.a >> rhs)
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> ShrAssign<J> for ColorGroup<COMPS, J>
where
    J: Copy + ShrAssign<J> + Default + 'static,
{
    #[inline]
    fn shr_assign(&mut self, rhs: J) {
        if COMPS == 1 {
            self.r >>= rhs;
        } else if COMPS == 2 {
            self.r >>= rhs;
            self.g >>= rhs;
        } else if COMPS == 3 {
            self.r >>= rhs;
            self.g >>= rhs;
            self.b >>= rhs;
        } else if COMPS == 4 {
            self.r >>= rhs;
            self.g >>= rhs;
            self.b >>= rhs;
            self.a >>= rhs;
        }
    }
}

impl<const COMPS: usize, J> MulAdd<ColorGroup<COMPS, J>, J> for ColorGroup<COMPS, J>
where
    J: Copy + MulAdd<J, Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline]
    fn mul_add(self, a: ColorGroup<COMPS, J>, b: J) -> Self::Output {
        if COMPS == 1 {
            ColorGroup::from_components(self.r.mul_add(b, a.r), self.g, self.b, self.a)
        } else if COMPS == 2 {
            ColorGroup::from_components(
                self.r.mul_add(b, a.r),
                self.g.mul_add(b, a.g),
                self.b,
                self.a,
            )
        } else if COMPS == 3 {
            ColorGroup::from_components(
                self.r.mul_add(b, a.r),
                self.g.mul_add(b, a.g),
                self.b.mul_add(b, a.b),
                self.a,
            )
        } else if COMPS == 4 {
            ColorGroup::from_components(
                self.r.mul_add(b, a.r),
                self.g.mul_add(b, a.g),
                self.b.mul_add(b, a.b),
                self.a.mul_add(b, a.a),
            )
        } else {
            panic!("Not implemented.");
        }
    }
}

impl<const COMPS: usize, J> AddAssign<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + AddAssign,
{
    #[inline]
    fn add_assign(&mut self, rhs: ColorGroup<COMPS, J>) {
        if COMPS == 1 {
            self.r += rhs.r;
        } else if COMPS == 2 {
            self.r += rhs.r;
            self.g += rhs.g;
        } else if COMPS == 3 {
            self.r += rhs.r;
            self.g += rhs.g;
            self.b += rhs.b;
        } else if COMPS == 4 {
            self.r += rhs.r;
            self.g += rhs.g;
            self.b += rhs.b;
            self.a += rhs.a;
        }
    }
}

impl<const COMPS: usize, J> SubAssign<ColorGroup<COMPS, J>> for ColorGroup<COMPS, J>
where
    J: Copy + SubAssign,
{
    #[inline]
    fn sub_assign(&mut self, rhs: ColorGroup<COMPS, J>) {
        if COMPS == 1 {
            self.r -= rhs.r;
        } else if COMPS == 2 {
            self.r -= rhs.r;
            self.g -= rhs.g;
        } else if COMPS == 3 {
            self.r -= rhs.r;
            self.g -= rhs.g;
            self.b -= rhs.b;
        } else if COMPS == 4 {
            self.r -= rhs.r;
            self.g -= rhs.g;
            self.b -= rhs.b;
            self.a -= rhs.a;
        }
    }
}

impl<const COMPS: usize, J> Default for ColorGroup<COMPS, J>
where
    J: Copy + FromPrimitive + Default,
{
    #[inline]
    fn default() -> Self {
        ColorGroup::new()
    }
}