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
#[cfg(feature = "half")]
use crate::alpha_handle_f16::{premultiply_alpha_rgba_f16, unpremultiply_alpha_rgba_f16};
use crate::alpha_handle_f32::{premultiply_alpha_rgba_f32, unpremultiply_alpha_rgba_f32};
use crate::alpha_handle_u16::{premultiply_alpha_rgba_u16, unpremultiply_alpha_rgba_u16};
use crate::alpha_handle_u8::{premultiply_alpha_rgba, unpremultiply_alpha_rgba};
use crate::pic_scale_error::{PicScaleBufferMismatch, PicScaleError};
use crate::ImageSize;
use num_traits::FromPrimitive;
use rayon::ThreadPool;
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

impl<T: Copy + Debug> BufferStore<'_, T> {
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
    ) -> Result<ImageStore<'static, T, N>, PicScaleError> {
        let expected_size = width * height * N;
        if slice_ref.len() != width * height * N {
            return Err(PicScaleError::BufferMismatch(PicScaleBufferMismatch {
                expected: expected_size,
                width,
                height,
                channels: N,
                slice_len: slice_ref.len(),
            }));
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
    ) -> Result<ImageStore<'a, T, N>, PicScaleError> {
        let expected_size = width * height * N;
        if slice_ref.len() != width * height * N {
            return Err(PicScaleError::BufferMismatch(PicScaleBufferMismatch {
                expected: expected_size,
                width,
                height,
                channels: N,
                slice_len: slice_ref.len(),
            }));
        }
        Ok(ImageStore::<T, N> {
            buffer: BufferStore::Borrowed(slice_ref),
            channels: N,
            width,
            height,
            bit_depth: 0,
        })
    }

    pub fn copied<'b>(&self) -> ImageStore<'b, T, N> {
        ImageStore::<T, N> {
            buffer: BufferStore::Owned(self.buffer.borrow().to_vec()),
            channels: N,
            width: self.width,
            height: self.height,
            bit_depth: self.bit_depth,
        }
    }
}

impl ImageStore<'_, u8, 4> {
    pub fn unpremultiply_alpha(&mut self, pool: &Option<ThreadPool>) {
        let dst = self.buffer.borrow_mut();
        unpremultiply_alpha_rgba(dst, self.width, self.height, pool);
    }

    pub fn premultiply_alpha(&self, into: &mut ImageStore<'_, u8, 4>, pool: &Option<ThreadPool>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        premultiply_alpha_rgba(dst, src, self.width, self.height, pool);
    }
}

impl ImageStore<'_, u16, 4> {
    pub fn unpremultiply_alpha(&mut self, pool: &Option<ThreadPool>) {
        let in_place = self.buffer.borrow_mut();
        unpremultiply_alpha_rgba_u16(in_place, self.width, self.height, self.bit_depth, pool);
    }

    pub fn premultiply_alpha(&self, into: &mut ImageStore<'_, u16, 4>, pool: &Option<ThreadPool>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        premultiply_alpha_rgba_u16(dst, src, self.width, self.height, self.bit_depth, pool);
    }
}

impl ImageStore<'_, f32, 4> {
    pub fn unpremultiply_alpha(&mut self, pool: &Option<ThreadPool>) {
        let dst = self.buffer.borrow_mut();
        unpremultiply_alpha_rgba_f32(dst, self.width, self.height, pool);
    }

    pub fn premultiply_alpha(&self, into: &mut ImageStore<'_, f32, 4>, pool: &Option<ThreadPool>) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        premultiply_alpha_rgba_f32(dst, src, self.width, self.height, pool);
    }
}

#[cfg(feature = "half")]
impl<'a> ImageStore<'a, half::f16, 4> {
    pub fn unpremultiply_alpha(&mut self, pool: &Option<ThreadPool>) {
        let dst = self.buffer.borrow_mut();
        unpremultiply_alpha_rgba_f16(dst, self.width, self.height, pool);
    }

    pub fn premultiply_alpha(
        &self,
        into: &mut ImageStore<'_, half::f16, 4>,
        pool: &Option<ThreadPool>,
    ) {
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.borrow();
        premultiply_alpha_rgba_f16(dst, src, self.width, self.height, pool);
    }
}
