use crate::alpha_handle::{premultiply_alpha_rgba, unpremultiply_alpha_rgba};
use crate::ImageSize;
use num_traits::FromPrimitive;
use std::fmt::Debug;
/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[derive(Debug)]
pub struct ImageStore<'a, T, const N: usize>
where
    T: FromPrimitive + Clone + Copy + Debug,
{
    pub(crate) buffer: BufferStore<'a, T>,
    pub channels: usize,
    pub width: usize,
    pub height: usize,
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
    pub fn new(slice_ref: Vec<T>, width: usize, height: usize) -> ImageStore<'static, T, N> {
        ImageStore::<T, N> {
            buffer: BufferStore::Owned(slice_ref),
            channels: N,
            width,
            height,
        }
    }

    pub fn alloc(width: usize, height: usize) -> ImageStore<'static, T, N> {
        let vc = vec![T::from_u32(0).unwrap_or_default(); width * N * height];
        ImageStore::<T, N> {
            buffer: BufferStore::Owned(vc),
            channels: N,
            width,
            height,
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

    pub fn from_slice(slice_ref: &'a mut [T], width: usize, height: usize) -> ImageStore<T, N> {
        ImageStore::<T, N> {
            buffer: BufferStore::Borrowed(slice_ref),
            channels: N,
            width,
            height,
        }
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
