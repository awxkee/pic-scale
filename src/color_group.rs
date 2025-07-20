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
#![forbid(unsafe_code)]
use crate::mlaf::mlaf;
use crate::saturate_narrow::SaturateNarrow;
use num_traits::{FromPrimitive, MulAdd, WrappingAdd, WrappingMul};
use std::ops::{Add, AddAssign, Mul, Shr, ShrAssign, Sub, SubAssign};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ColorGroup<const CN: usize, J: Copy> {
    pub(crate) r: J,
    pub(crate) g: J,
    pub(crate) b: J,
    pub(crate) a: J,
}

impl<const CN: usize, J> ColorGroup<CN, J>
where
    J: Copy + Default,
{
    #[inline(always)]
    pub(crate) fn new() -> ColorGroup<CN, J> {
        ColorGroup {
            r: J::default(),
            g: J::default(),
            b: J::default(),
            a: J::default(),
        }
    }

    #[inline(always)]
    pub(crate) fn from_components(r: J, g: J, b: J, a: J) -> ColorGroup<CN, J> {
        ColorGroup { r, g, b, a }
    }

    #[inline(always)]
    pub(crate) fn dup(v: J) -> ColorGroup<CN, J> {
        ColorGroup {
            r: v,
            g: v,
            b: v,
            a: v,
        }
    }
}

impl<const CN: usize, J> Mul<J> for ColorGroup<CN, J>
where
    J: Copy + Mul<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: J) -> Self::Output {
        if CN == 1 {
            ColorGroup::from_components(self.r * rhs, self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(self.r * rhs, self.g * rhs, self.b, self.a)
        } else if CN == 3 {
            ColorGroup::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a)
        } else if CN == 4 {
            ColorGroup::from_components(self.r * rhs, self.g * rhs, self.b * rhs, self.a * rhs)
        } else {
            unimplemented!();
        }
    }
}

impl<const CN: usize, J> ColorGroup<CN, J>
where
    J: Copy + WrappingMul<Output = J> + Default + 'static,
{
    #[inline(always)]
    pub(crate) fn trunc_mul(self, rhs: J) -> ColorGroup<CN, J> {
        if CN == 1 {
            ColorGroup::from_components(self.r.wrapping_mul(&rhs), self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(
                self.r.wrapping_mul(&rhs),
                self.g.wrapping_mul(&rhs),
                self.b,
                self.a,
            )
        } else if CN == 3 {
            ColorGroup::from_components(
                self.r.wrapping_mul(&rhs),
                self.g.wrapping_mul(&rhs),
                self.b.wrapping_mul(&rhs),
                self.a,
            )
        } else if CN == 4 {
            ColorGroup::from_components(
                self.r.wrapping_mul(&rhs),
                self.g.wrapping_mul(&rhs),
                self.b.wrapping_mul(&rhs),
                self.a.wrapping_mul(&rhs),
            )
        } else {
            unimplemented!();
        }
    }
}

impl<const CN: usize, J> ColorGroup<CN, J>
where
    J: Copy + WrappingAdd<Output = J> + Default + 'static,
{
    #[inline(always)]
    pub(crate) fn trunc_add(self, rhs: &ColorGroup<CN, J>) -> ColorGroup<CN, J> {
        if CN == 1 {
            ColorGroup::from_components(self.r.wrapping_add(&rhs.r), self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(
                self.r.wrapping_add(&rhs.r),
                self.g.wrapping_add(&rhs.g),
                self.b,
                self.a,
            )
        } else if CN == 3 {
            ColorGroup::from_components(
                self.r.wrapping_add(&rhs.r),
                self.g.wrapping_add(&rhs.g),
                self.b.wrapping_add(&rhs.b),
                self.a,
            )
        } else if CN == 4 {
            ColorGroup::from_components(
                self.r.wrapping_add(&rhs.r),
                self.g.wrapping_add(&rhs.g),
                self.b.wrapping_add(&rhs.b),
                self.a.wrapping_add(&rhs.a),
            )
        } else {
            unimplemented!();
        }
    }
}

impl<const CN: usize, J> ColorGroup<CN, J>
where
    J: Copy + Default + 'static,
{
    #[inline(always)]
    pub(crate) fn saturate_narrow<V>(&self, bit_depth: u32) -> ColorGroup<CN, V>
    where
        V: Copy + Default,
        J: SaturateNarrow<V>,
    {
        if CN == 1 {
            ColorGroup::<CN, V>::from_components(
                self.r.saturate_narrow(bit_depth),
                V::default(),
                V::default(),
                V::default(),
            )
        } else if CN == 2 {
            ColorGroup::<CN, V>::from_components(
                self.r.saturate_narrow(bit_depth),
                self.g.saturate_narrow(bit_depth),
                V::default(),
                V::default(),
            )
        } else if CN == 3 {
            ColorGroup::<CN, V>::from_components(
                self.r.saturate_narrow(bit_depth),
                self.g.saturate_narrow(bit_depth),
                self.b.saturate_narrow(bit_depth),
                V::default(),
            )
        } else {
            ColorGroup::<CN, V>::from_components(
                self.r.saturate_narrow(bit_depth),
                self.g.saturate_narrow(bit_depth),
                self.b.saturate_narrow(bit_depth),
                self.a.saturate_narrow(bit_depth),
            )
        }
    }
}

impl<const CN: usize, J> Mul<ColorGroup<CN, J>> for ColorGroup<CN, J>
where
    J: Copy + Mul<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: ColorGroup<CN, J>) -> Self::Output {
        if CN == 1 {
            ColorGroup::from_components(self.r * rhs.r, self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(self.r * rhs.r, self.g * rhs.g, self.b, self.a)
        } else if CN == 3 {
            ColorGroup::from_components(self.r * rhs.r, self.g * rhs.g, self.b * rhs.b, self.a)
        } else if CN == 4 {
            ColorGroup::from_components(
                self.r * rhs.r,
                self.g * rhs.g,
                self.b * rhs.b,
                self.a * rhs.b,
            )
        } else {
            unimplemented!();
        }
    }
}

impl ColorGroup<4, i32> {
    #[inline(always)]
    #[allow(clippy::manual_clamp)]
    pub(crate) fn saturate_ar30(&self) -> ColorGroup<4, i32> {
        ColorGroup::from_components(
            (self.r >> PRECISION).min(1023).max(0),
            (self.g >> PRECISION).min(1023).max(0),
            (self.b >> PRECISION).min(1023).max(0),
            (self.a >> PRECISION).min(3).max(0),
        )
    }

    #[inline(always)]
    pub(crate) fn to_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(self) -> u32 {
        let ar30_type: Rgb30 = AR30_TYPE.into();
        ar30_type.pack_w_a::<AR30_ORDER>(self.r, self.g, self.b, self.a)
    }
}

impl<const CN: usize, J> Sub<J> for ColorGroup<CN, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: J) -> Self::Output {
        if CN == 1 {
            ColorGroup::from_components(self.r - rhs, self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(self.r - rhs, self.g - rhs, self.b, self.a)
        } else if CN == 3 {
            ColorGroup::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a)
        } else if CN == 4 {
            ColorGroup::from_components(self.r - rhs, self.g - rhs, self.b - rhs, self.a - rhs)
        } else {
            unimplemented!();
        }
    }
}

impl<const CN: usize, J> Sub<ColorGroup<CN, J>> for ColorGroup<CN, J>
where
    J: Copy + Sub<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: ColorGroup<CN, J>) -> Self::Output {
        if CN == 1 {
            ColorGroup::from_components(self.r - rhs.r, self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(self.r - rhs.r, self.g - rhs.g, self.b, self.a)
        } else if CN == 3 {
            ColorGroup::from_components(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b, self.a)
        } else if CN == 4 {
            ColorGroup::from_components(
                self.r - rhs.r,
                self.g - rhs.g,
                self.b - rhs.b,
                self.a - rhs.a,
            )
        } else {
            unimplemented!("Not implemented.");
        }
    }
}

impl<const CN: usize, J> Add<ColorGroup<CN, J>> for ColorGroup<CN, J>
where
    J: Copy + Add<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: ColorGroup<CN, J>) -> Self::Output {
        if CN == 1 {
            ColorGroup::from_components(self.r + rhs.r, self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(self.r + rhs.r, self.g + rhs.g, self.b, self.a)
        } else if CN == 3 {
            ColorGroup::from_components(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b, self.a)
        } else if CN == 4 {
            ColorGroup::from_components(
                self.r + rhs.r,
                self.g + rhs.g,
                self.b + rhs.b,
                self.a + rhs.a,
            )
        } else {
            unimplemented!("Not implemented.");
        }
    }
}

impl<const CN: usize, J> Add<J> for ColorGroup<CN, J>
where
    J: Copy + Add<Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: J) -> Self::Output {
        if CN == 1 {
            ColorGroup::from_components(self.r + rhs, self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(self.r + rhs, self.g + rhs, self.b, self.a)
        } else if CN == 3 {
            ColorGroup::from_components(self.r + rhs, self.g + rhs, self.b + rhs, self.a)
        } else if CN == 4 {
            ColorGroup::from_components(self.r + rhs, self.g + rhs, self.b + rhs, self.a + rhs)
        } else {
            unimplemented!("Not implemented.");
        }
    }
}

impl<const CN: usize, J> Shr<J> for ColorGroup<CN, J>
where
    J: Copy + Shr<J, Output = J> + Default + 'static,
{
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: J) -> Self::Output {
        if CN == 1 {
            ColorGroup::from_components(self.r >> rhs, self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(self.r >> rhs, self.g >> rhs, self.b, self.a)
        } else if CN == 3 {
            ColorGroup::from_components(self.r >> rhs, self.g >> rhs, self.b >> rhs, self.a)
        } else if CN == 4 {
            ColorGroup::from_components(self.r >> rhs, self.g >> rhs, self.b >> rhs, self.a >> rhs)
        } else {
            unimplemented!("Not implemented.");
        }
    }
}

impl<const CN: usize, J> ShrAssign<J> for ColorGroup<CN, J>
where
    J: Copy + ShrAssign<J> + Default + 'static,
{
    #[inline(always)]
    fn shr_assign(&mut self, rhs: J) {
        if CN == 1 {
            self.r >>= rhs;
        } else if CN == 2 {
            self.r >>= rhs;
            self.g >>= rhs;
        } else if CN == 3 {
            self.r >>= rhs;
            self.g >>= rhs;
            self.b >>= rhs;
        } else if CN == 4 {
            self.r >>= rhs;
            self.g >>= rhs;
            self.b >>= rhs;
            self.a >>= rhs;
        }
    }
}

impl<const CN: usize, J> MulAdd<ColorGroup<CN, J>, J> for ColorGroup<CN, J>
where
    J: Copy + MulAdd<J, Output = J> + Default + 'static + Mul<J, Output = J> + Add<J, Output = J>,
{
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: ColorGroup<CN, J>, b: J) -> Self::Output {
        if CN == 1 {
            ColorGroup::from_components(mlaf(self.r, a.r, b), self.g, self.b, self.a)
        } else if CN == 2 {
            ColorGroup::from_components(mlaf(self.r, a.r, b), mlaf(self.g, a.g, b), self.b, self.a)
        } else if CN == 3 {
            ColorGroup::from_components(
                mlaf(self.r, a.r, b),
                mlaf(self.g, a.g, b),
                mlaf(self.b, a.b, b),
                self.a,
            )
        } else if CN == 4 {
            ColorGroup::from_components(
                mlaf(self.r, a.r, b),
                mlaf(self.g, a.g, b),
                mlaf(self.b, a.b, b),
                mlaf(self.a, a.a, b),
            )
        } else {
            unimplemented!();
        }
    }
}

impl<const CN: usize, J> AddAssign<ColorGroup<CN, J>> for ColorGroup<CN, J>
where
    J: Copy + AddAssign,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: ColorGroup<CN, J>) {
        if CN == 1 {
            self.r += rhs.r;
        } else if CN == 2 {
            self.r += rhs.r;
            self.g += rhs.g;
        } else if CN == 3 {
            self.r += rhs.r;
            self.g += rhs.g;
            self.b += rhs.b;
        } else if CN == 4 {
            self.r += rhs.r;
            self.g += rhs.g;
            self.b += rhs.b;
            self.a += rhs.a;
        }
    }
}

impl<const CN: usize, J> SubAssign<ColorGroup<CN, J>> for ColorGroup<CN, J>
where
    J: Copy + SubAssign,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: ColorGroup<CN, J>) {
        if CN == 1 {
            self.r -= rhs.r;
        } else if CN == 2 {
            self.r -= rhs.r;
            self.g -= rhs.g;
        } else if CN == 3 {
            self.r -= rhs.r;
            self.g -= rhs.g;
            self.b -= rhs.b;
        } else if CN == 4 {
            self.r -= rhs.r;
            self.g -= rhs.g;
            self.b -= rhs.b;
            self.a -= rhs.a;
        }
    }
}

impl<const CN: usize, J> Default for ColorGroup<CN, J>
where
    J: Copy + FromPrimitive + Default,
{
    #[inline(always)]
    fn default() -> Self {
        ColorGroup::new()
    }
}

macro_rules! ld_ar30 {
    ($store: expr, $ar_type: expr, $ar_order: ty) => {{
        let ar_type: crate::ar30::Rgb30 = $ar_type.into();
        let read_bits = u32::from_ne_bytes([$store[0], $store[1], $store[2], $store[3]]);
        let unpacked = ar_type.unpack::<$ar_order>(read_bits);
        ColorGroup::<4, i32> {
            r: unpacked.0 as i32,
            g: unpacked.1 as i32,
            b: unpacked.2 as i32,
            a: unpacked.3 as i32,
        }
    }};
}

pub(crate) use ld_ar30;

macro_rules! load_ar30_p {
    ($store: expr, $ar_type: expr, $ar_order: ty) => {{
        let ar_type: crate::ar30::Rgb30 = $ar_type.into();
        let read_bits = u32::from_ne_bytes([$store[0], $store[1], $store[2], $store[3]]);
        let unpacked = ar_type.unpack::<$ar_order>(read_bits);
        ColorGroup::<4, i32> {
            r: unpacked.0 as i32,
            g: unpacked.1 as i32,
            b: unpacked.2 as i32,
            a: unpacked.3 as i32,
        }
    }};
}

pub(crate) use load_ar30_p;

macro_rules! ld_ar30_with_offset {
    ($store: expr, $ar_type: expr, $ar_order: ty, $offset: expr) => {{
        let ar_type: crate::ar30::Rgb30 = $ar_type.into();
        let cn = $offset * 4;
        let read_bits =
            u32::from_ne_bytes([$store[cn], $store[cn + 1], $store[cn + 2], $store[cn + 3]]);
        let unpacked = ar_type.unpack::<$ar_order>(read_bits);
        ColorGroup::<4, i32> {
            r: unpacked.0 as i32,
            g: unpacked.1 as i32,
            b: unpacked.2 as i32,
            a: unpacked.3 as i32,
        }
    }};
}

pub(crate) use ld_ar30_with_offset;

macro_rules! ld_g {
    ($store: expr, $channels: expr, $vtype: ty) => {{
        if $channels == 1 {
            ColorGroup::<$channels, $vtype> {
                r: $store[0].as_(),
                g: 0.as_(),
                b: 0.as_(),
                a: 0.as_(),
            }
        } else if $channels == 2 {
            ColorGroup::<$channels, $vtype> {
                r: $store[0].as_(),
                g: $store[1].as_(),
                b: 0.as_(),
                a: 0.as_(),
            }
        } else if $channels == 3 {
            ColorGroup::<$channels, $vtype> {
                r: $store[0].as_(),
                g: $store[1].as_(),
                b: $store[2].as_(),
                a: 0.as_(),
            }
        } else if $channels == 4 {
            ColorGroup::<$channels, $vtype> {
                r: $store[0].as_(),
                g: $store[1].as_(),
                b: $store[2].as_(),
                a: $store[3].as_(),
            }
        } else {
            unimplemented!()
        }
    }};
}

pub(crate) use ld_g;

macro_rules! ldg_with_offset {
    ($store: expr, $channels: expr, $offset: expr, $vtype: ty) => {{
        if $channels == 1 {
            ColorGroup::<$channels, $vtype> {
                r: $store[$offset].as_(),
                g: 0.as_(),
                b: 0.as_(),
                a: 0.as_(),
            }
        } else if $channels == 2 {
            ColorGroup::<$channels, $vtype> {
                r: $store[$offset].as_(),
                g: $store[$offset + 1].as_(),
                b: 0.as_(),
                a: 0.as_(),
            }
        } else if $channels == 3 {
            ColorGroup::<$channels, $vtype> {
                r: $store[$offset].as_(),
                g: $store[$offset + 1].as_(),
                b: $store[$offset + 2].as_(),
                a: 0.as_(),
            }
        } else if $channels == 4 {
            ColorGroup::<$channels, $vtype> {
                r: $store[$offset].as_(),
                g: $store[$offset + 1].as_(),
                b: $store[$offset + 2].as_(),
                a: $store[$offset + 3].as_(),
            }
        } else {
            unimplemented!()
        }
    }};
}

pub(crate) use ldg_with_offset;

macro_rules! st_g {
    ($color_group: expr, $store: expr, $components: expr) => {{
        $store[0] = $color_group.r;
        if $components > 1 {
            $store[1] = $color_group.g;
        }
        if $components > 2 {
            $store[2] = $color_group.b;
        }
        if $components == 4 {
            $store[3] = $color_group.a;
        }
    }};
}

pub(crate) use st_g;

macro_rules! st_g_mixed {
    ($color_group: expr, $store: expr, $components: expr, $bit_depth: expr) => {{
        $store[0] = $color_group.r.to_mixed($bit_depth);
        if $components > 1 {
            $store[1] = $color_group.g.to_mixed($bit_depth);
        }
        if $components > 2 {
            $store[2] = $color_group.b.to_mixed($bit_depth);
        }
        if $components == 4 {
            $store[3] = $color_group.a.to_mixed($bit_depth);
        }
    }};
}

use crate::ar30::Rgb30;
use crate::support::PRECISION;
pub(crate) use st_g_mixed;
