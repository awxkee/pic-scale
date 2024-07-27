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
#[cfg(all(feature = "half"))]
use crate::alpha_handle_f16::{premultiply_alpha_rgba_f16, unpremultiply_alpha_rgba_f16};
use crate::alpha_handle_f32::{premultiply_alpha_rgba_f32, unpremultiply_alpha_rgba_f32};
use crate::alpha_handle_u16::{premultiply_alpha_rgba_u16, unpremultiply_alpha_rgba_u16};
use crate::alpha_handle_u8::{premultiply_alpha_rgba, unpremultiply_alpha_rgba};
use crate::ImageSize;
use num_traits::FromPrimitive;
use std::fmt::Debug;

#[derive(Debug)]
/// Holds an image
///
/// # Arguments
/// `N` - count of channels
///
/// # Examples
/// ImageStore<u8, 4> - represents RGBA
/// ImageStore<u8, 3> - represents RGB
/// ImageStore<f32, 3> - represents RGB in f32 and etc
pub struct ImageStore<'a, T, const N: usize>
where
    T: FromPrimitive + Clone + Copy + Debug,
{
    pub(crate) buffer: BufferStore<'a, T>,
    /// Channels in the image
    pub channels: usize,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// This is private field, currently used only for u16, will be automatically passed from upper func
    pub(crate) bit_depth: usize,
}

#[derive(Debug)]
pub(crate) enum BufferStore<'a, T: Copy + Debug> {
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

impl<'a, T: Copy + Debug> BufferStore<'a, T> {
    pub fn borrow(&self) -> &[T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }

    pub fn borrow_mut(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }
}

impl<T, const N: usize> ImageStore<'static, T, N>
where
    T: FromPrimitive + Clone + Copy + Debug + Default,
{
    pub fn new(
        slice_ref: Vec<T>,
        width: usize,
        height: usize,
    ) -> Result<ImageStore<'static, T, N>, String> {
        let expected_size = width * height * N;
        if slice_ref.len() != width * height * N {
            return Err(format!(
                "Image buffer len expected to be {} [w({})*h({})*channels({})] but received {}",
                expected_size,
                width,
                height,
                N,
                slice_ref.len()
            ));
        }
        Ok(ImageStore::<T, N> {
            buffer: BufferStore::Owned(slice_ref),
            channels: N,
            width,
            height,
            bit_depth: 0,
        })
    }

    pub fn alloc(width: usize, height: usize) -> ImageStore<'static, T, N> {
        let vc = vec![T::from_u32(0).unwrap_or_default(); width * N * height];
        ImageStore::<T, N> {
            buffer: BufferStore::Owned(vc),
            channels: N,
            width,
            height,
            bit_depth: 0,
        }
    }
}

impl<'a, T, const N: usize> ImageStore<'a, T, N>
where
    T: FromPrimitive + Clone + Copy + Debug,
{
    pub fn get_size(&self) -> ImageSize {
        ImageSize::new(self.width, self.height)
    }

    pub fn as_bytes(&self) -> &[T] {
        match &self.buffer {
            BufferStore::Borrowed(p) => p,
            BufferStore::Owned(v) => v,
        }
    }

    pub fn from_slice(
        slice_ref: &'a mut [T],
        width: usize,
        height: usize,
    ) -> Result<ImageStore<T, N>, String> {
        let expected_size = width * height * N;
        if slice_ref.len() != width * height * N {
            return Err(format!(
                "Image buffer len expected to be {} [w({})*h({})*channels({})] but received {}",
                expected_size,
                width,
                height,
                N,
                slice_ref.len()
            ));
        }
        Ok(ImageStore::<T, N> {
            buffer: BufferStore::Borrowed(slice_ref),
            channels: N,
            width,
            height,
            bit_depth: 0,
        })
    }
}

impl<'a> ImageStore<'a, u8, 4> {
    pub fn unpremultiply_alpha(&self, into: &mut ImageStore<u8, 4>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        unpremultiply_alpha_rgba(dst, src, self.width, self.height);
    }

    pub fn premultiply_alpha(&self, into: &mut ImageStore<u8, 4>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        premultiply_alpha_rgba(dst, src, self.width, self.height);
    }
}

impl<'a> ImageStore<'a, u16, 4> {
    pub fn unpremultiply_alpha(&self, into: &mut ImageStore<u16, 4>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        unpremultiply_alpha_rgba_u16(dst, src, self.width, self.height, self.bit_depth);
    }

    pub fn premultiply_alpha(&self, into: &mut ImageStore<u16, 4>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        premultiply_alpha_rgba_u16(dst, src, self.width, self.height, self.bit_depth);
    }
}

impl<'a> ImageStore<'a, f32, 4> {
    pub fn unpremultiply_alpha(&self, into: &mut ImageStore<f32, 4>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        unpremultiply_alpha_rgba_f32(dst, src, self.width, self.height);
    }

    pub fn premultiply_alpha(&self, into: &mut ImageStore<f32, 4>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        premultiply_alpha_rgba_f32(dst, src, self.width, self.height);
    }
}

#[cfg(all(feature = "half"))]
impl<'a> ImageStore<'a, half::f16, 4> {
    pub fn unpremultiply_alpha(&self, into: &mut ImageStore<half::f16, 4>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        unpremultiply_alpha_rgba_f16(dst, src, self.width, self.height);
    }

    pub fn premultiply_alpha(&self, into: &mut ImageStore<half::f16, 4>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        premultiply_alpha_rgba_f16(dst, src, self.width, self.height);
    }
}
