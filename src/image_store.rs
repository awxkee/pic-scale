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
use crate::alpha_check::has_non_constant_cap_alpha_rgba_f32;
#[cfg(feature = "nightly_f16")]
use crate::alpha_handle_f16::{premultiply_alpha_rgba_f16, unpremultiply_alpha_rgba_f16};
use crate::alpha_handle_f32::{premultiply_alpha_rgba_f32, unpremultiply_alpha_rgba_f32};
use crate::alpha_handle_u8::{premultiply_alpha_rgba, unpremultiply_alpha_rgba};
use crate::alpha_handle_u16::{premultiply_alpha_rgba_u16, unpremultiply_alpha_rgba_u16};
use crate::pic_scale_error::{PicScaleBufferMismatch, PicScaleError};
use crate::{ImageSize, WorkloadStrategy};
#[cfg(feature = "nightly_f16")]
use core::f16;
use std::fmt::Debug;

/// Holds an image
///
/// # Arguments
/// `N` - count of channels
///
/// # Examples
/// ImageStore<u8, 4> - represents RGBA
/// ImageStore<u8, 3> - represents RGB
/// ImageStore<f32, 3> - represents RGB in f32 and etc
#[derive(Debug, Clone)]
pub struct ImageStore<'a, T, const N: usize>
where
    T: Clone + Copy + Debug,
{
    pub buffer: std::borrow::Cow<'a, [T]>,
    /// Channels in the image
    pub channels: usize,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Image stride, if stride is zero then it considered to be `width * N`
    pub stride: usize,
    /// This is private field, currently used only for u16, will be automatically passed from upper func
    pub bit_depth: usize,
}

/// Holds an image
///
/// # Arguments
/// `N` - count of channels
///
/// # Examples
/// ImageStore<u8, 4> - represents RGBA
/// ImageStore<u8, 3> - represents RGB
/// ImageStore<f32, 3> - represents RGB in f32 and etc
#[derive(Debug)]
pub struct ImageStoreMut<'a, T, const N: usize>
where
    T: Clone + Copy + Debug,
{
    pub buffer: BufferStore<'a, T>,
    /// Channels in the image
    pub channels: usize,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Image stride, if stride is zero then it considered to be `width * N`
    pub stride: usize,
    /// Required for `u16` images
    pub bit_depth: usize,
}

pub(crate) trait CheckStoreDensity {
    fn should_have_bit_depth(&self) -> bool;
}

/// Structure for mutable target buffer
#[derive(Debug)]
pub enum BufferStore<'a, T: Copy + Debug> {
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

impl<T: Copy + Debug> BufferStore<'_, T> {
    #[allow(clippy::should_implement_trait)]
    /// Borrowing immutable slice
    pub fn borrow(&self) -> &[T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }

    #[allow(clippy::should_implement_trait)]
    /// Borrowing mutable slice
    pub fn borrow_mut(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }
}

impl<'a, T, const N: usize> ImageStore<'a, T, N>
where
    T: Clone + Copy + Debug + Default,
{
    /// Creates new store
    pub fn new(
        slice_ref: Vec<T>,
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
            buffer: std::borrow::Cow::Owned(slice_ref),
            channels: N,
            width,
            height,
            stride: width * N,
            bit_depth: 0,
        })
    }

    /// Borrows immutable slice as new image store
    pub fn borrow(
        slice_ref: &'a [T],
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
            buffer: std::borrow::Cow::Borrowed(slice_ref),
            channels: N,
            width,
            height,
            stride: width * N,
            bit_depth: 0,
        })
    }

    /// Allocates new owned image store
    pub fn alloc(width: usize, height: usize) -> ImageStore<'a, T, N> {
        let vc = vec![T::default(); width * N * height];
        ImageStore::<T, N> {
            buffer: std::borrow::Cow::Owned(vc),
            channels: N,
            width,
            height,
            stride: width * N,
            bit_depth: 0,
        }
    }
}

impl<const N: usize> CheckStoreDensity for ImageStoreMut<'_, u8, N> {
    fn should_have_bit_depth(&self) -> bool {
        false
    }
}

impl<const N: usize> CheckStoreDensity for ImageStoreMut<'_, f32, N> {
    fn should_have_bit_depth(&self) -> bool {
        false
    }
}

#[cfg(feature = "nightly_f16")]
impl<const N: usize> CheckStoreDensity for ImageStoreMut<'_, f16, N> {
    fn should_have_bit_depth(&self) -> bool {
        false
    }
}

impl<const N: usize> CheckStoreDensity for ImageStoreMut<'_, u16, N> {
    fn should_have_bit_depth(&self) -> bool {
        true
    }
}

impl<T, const N: usize> ImageStoreMut<'_, T, N>
where
    T: Clone + Copy + Debug + Default,
{
    pub(crate) fn validate(&self) -> Result<(), PicScaleError> {
        let expected_size = self.stride() * self.height;
        if self.buffer.borrow().len() != self.stride() * self.height {
            return Err(PicScaleError::BufferMismatch(PicScaleBufferMismatch {
                expected: expected_size,
                width: self.width,
                height: self.height,
                channels: N,
                slice_len: self.buffer.borrow().len(),
            }));
        }
        if self.stride < self.width * N {
            return Err(PicScaleError::InvalidStride(self.width * N, self.stride));
        }
        Ok(())
    }
}

impl<T, const N: usize> ImageStore<'_, T, N>
where
    T: Clone + Copy + Debug + Default,
{
    pub(crate) fn validate(&self) -> Result<(), PicScaleError> {
        let expected_size = self.stride() * self.height;
        if self.buffer.as_ref().len() != self.stride() * self.height {
            return Err(PicScaleError::BufferMismatch(PicScaleBufferMismatch {
                expected: expected_size,
                width: self.width,
                height: self.height,
                channels: N,
                slice_len: self.buffer.as_ref().len(),
            }));
        }
        if self.stride < self.width * N {
            return Err(PicScaleError::InvalidStride(self.width * N, self.stride));
        }
        Ok(())
    }
}

impl<'a, T, const N: usize> ImageStoreMut<'a, T, N>
where
    T: Clone + Copy + Debug + Default,
{
    /// Creates new mutable storage from vector
    ///
    /// Always sets bit-depth to `0`
    pub fn new(
        slice_ref: Vec<T>,
        width: usize,
        height: usize,
    ) -> Result<ImageStoreMut<'a, T, N>, PicScaleError> {
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
        Ok(ImageStoreMut::<T, N> {
            buffer: BufferStore::Owned(slice_ref),
            channels: N,
            width,
            height,
            stride: width * N,
            bit_depth: 0,
        })
    }

    /// Creates new mutable storage from slice
    ///
    /// Always sets bit-depth to `0`
    pub fn borrow(
        slice_ref: &'a mut [T],
        width: usize,
        height: usize,
    ) -> Result<ImageStoreMut<'a, T, N>, PicScaleError> {
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
        Ok(ImageStoreMut::<T, N> {
            buffer: BufferStore::Borrowed(slice_ref),
            channels: N,
            width,
            height,
            stride: width * N,
            bit_depth: 0,
        })
    }

    /// Allocates new mutable image storage
    ///
    /// Always sets bit depth to `0`
    pub fn alloc(width: usize, height: usize) -> ImageStoreMut<'a, T, N> {
        let vc = vec![T::default(); width * N * height];
        ImageStoreMut::<T, N> {
            buffer: BufferStore::Owned(vc),
            channels: N,
            width,
            height,
            stride: width * N,
            bit_depth: 0,
        }
    }

    /// Allocates new mutable image storage with required bit-depth
    pub fn alloc_with_depth(
        width: usize,
        height: usize,
        bit_depth: usize,
    ) -> ImageStoreMut<'a, T, N> {
        let vc = vec![T::default(); width * N * height];
        ImageStoreMut::<T, N> {
            buffer: BufferStore::Owned(vc),
            channels: N,
            width,
            height,
            stride: width * N,
            bit_depth,
        }
    }
}

impl<T, const N: usize> ImageStoreMut<'_, T, N>
where
    T: Clone + Copy + Debug,
{
    /// Returns safe stride
    ///
    /// If stride set to 0 then returns `width * N`
    #[inline]
    pub fn stride(&self) -> usize {
        if self.stride == 0 {
            return self.width * N;
        }
        self.stride
    }
}

impl<T, const N: usize> ImageStore<'_, T, N>
where
    T: Clone + Copy + Debug,
{
    /// Returns safe stride
    ///
    /// If stride set to 0 then returns `width * N`
    #[inline]
    pub fn stride(&self) -> usize {
        if self.stride == 0 {
            return self.width * N;
        }
        self.stride
    }
}

impl<'a, T, const N: usize> ImageStore<'a, T, N>
where
    T: Clone + Copy + Debug,
{
    /// Returns bounded image size
    pub fn get_size(&self) -> ImageSize {
        ImageSize::new(self.width, self.height)
    }

    /// Returns current image store as immutable slice
    pub fn as_bytes(&self) -> &[T] {
        match &self.buffer {
            std::borrow::Cow::Borrowed(br) => br,
            std::borrow::Cow::Owned(v) => v.as_ref(),
        }
    }

    /// Borrows immutable slice int oa new image store
    pub fn from_slice(
        slice_ref: &'a [T],
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
            buffer: std::borrow::Cow::Borrowed(slice_ref),
            channels: N,
            width,
            height,
            stride: width * N,
            bit_depth: 0,
        })
    }

    /// Deep copy immutable image store into a new immutable store
    pub fn copied<'b>(&self) -> ImageStore<'b, T, N> {
        ImageStore::<T, N> {
            buffer: std::borrow::Cow::Owned(self.buffer.as_ref().to_vec()),
            channels: N,
            width: self.width,
            height: self.height,
            stride: self.width * N,
            bit_depth: self.bit_depth,
        }
    }

    /// Deep copy immutable image into mutable
    pub fn copied_to_mut(&self, into: &mut ImageStoreMut<T, N>) {
        let into_stride = into.stride();
        for (src_row, dst_row) in self
            .buffer
            .as_ref()
            .chunks_exact(self.stride())
            .zip(into.buffer.borrow_mut().chunks_exact_mut(into_stride))
        {
            for (&src, dst) in src_row.iter().zip(dst_row.iter_mut()) {
                *dst = src;
            }
        }
    }
}

impl<'a, T, const N: usize> ImageStoreMut<'a, T, N>
where
    T: Clone + Copy + Debug,
{
    /// Returns bounded image size
    pub fn get_size(&self) -> ImageSize {
        ImageSize::new(self.width, self.height)
    }

    /// Returns current image as immutable slice
    pub fn as_bytes(&self) -> &[T] {
        match &self.buffer {
            BufferStore::Borrowed(p) => p,
            BufferStore::Owned(v) => v,
        }
    }

    /// Borrows mutable slice as new image store
    pub fn from_slice(
        slice_ref: &'a mut [T],
        width: usize,
        height: usize,
    ) -> Result<ImageStoreMut<'a, T, N>, PicScaleError> {
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
        Ok(ImageStoreMut::<T, N> {
            buffer: BufferStore::Borrowed(slice_ref),
            channels: N,
            width,
            height,
            stride: width * N,
            bit_depth: 0,
        })
    }

    /// Performs deep copy into a new mutable image
    pub fn copied<'b>(&self) -> ImageStoreMut<'b, T, N> {
        ImageStoreMut::<T, N> {
            buffer: BufferStore::Owned(self.buffer.borrow().to_vec()),
            channels: N,
            width: self.width,
            height: self.height,
            stride: self.width * N,
            bit_depth: self.bit_depth,
        }
    }

    /// Performs deep copy into a new immutable image
    pub fn to_immutable(&self) -> ImageStore<'_, T, N> {
        ImageStore::<T, N> {
            buffer: std::borrow::Cow::Owned(self.buffer.borrow().to_owned()),
            channels: N,
            width: self.width,
            height: self.height,
            stride: self.width * N,
            bit_depth: self.bit_depth,
        }
    }
}

pub(crate) trait AssociateAlpha<T: Clone + Copy + Debug, const N: usize> {
    fn premultiply_alpha(&self, into: &mut ImageStoreMut<'_, T, N>, pool: &novtb::ThreadPool);
    fn is_alpha_premultiplication_needed(&self) -> bool;
}

pub(crate) trait UnassociateAlpha<T: Clone + Copy + Debug, const N: usize> {
    fn unpremultiply_alpha(
        &mut self,
        pool: &novtb::ThreadPool,
        workload_strategy: WorkloadStrategy,
    );
}

impl AssociateAlpha<u8, 2> for ImageStore<'_, u8, 2> {
    fn premultiply_alpha(&self, into: &mut ImageStoreMut<'_, u8, 2>, pool: &novtb::ThreadPool) {
        let dst_stride = into.stride();
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.as_ref();
        use crate::alpha_handle_u8::premultiply_alpha_gray_alpha;
        premultiply_alpha_gray_alpha(
            dst,
            dst_stride,
            src,
            self.width,
            self.height,
            self.stride(),
            pool,
        );
    }

    fn is_alpha_premultiplication_needed(&self) -> bool {
        use crate::alpha_check::has_non_constant_cap_alpha_gray_alpha8;
        has_non_constant_cap_alpha_gray_alpha8(self.buffer.as_ref(), self.width, self.stride())
    }
}

impl AssociateAlpha<u16, 2> for ImageStore<'_, u16, 2> {
    fn premultiply_alpha(&self, into: &mut ImageStoreMut<'_, u16, 2>, pool: &novtb::ThreadPool) {
        let dst_stride = into.stride();
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.as_ref();
        use crate::alpha_handle_u16::premultiply_alpha_gray_alpha_u16;
        premultiply_alpha_gray_alpha_u16(
            dst,
            dst_stride,
            src,
            self.width,
            self.height,
            self.stride(),
            into.bit_depth,
            pool,
        );
    }

    fn is_alpha_premultiplication_needed(&self) -> bool {
        use crate::alpha_check::has_non_constant_cap_alpha_gray_alpha16;
        has_non_constant_cap_alpha_gray_alpha16(self.buffer.as_ref(), self.width, self.stride())
    }
}

impl AssociateAlpha<f32, 2> for ImageStore<'_, f32, 2> {
    fn premultiply_alpha(&self, into: &mut ImageStoreMut<'_, f32, 2>, pool: &novtb::ThreadPool) {
        let dst_stride = into.stride();
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.as_ref();
        use crate::alpha_handle_f32::premultiply_alpha_gray_alpha_f32;
        premultiply_alpha_gray_alpha_f32(
            dst,
            dst_stride,
            src,
            self.stride(),
            self.width,
            self.height,
            pool,
        );
    }

    fn is_alpha_premultiplication_needed(&self) -> bool {
        use crate::alpha_check::has_non_constant_cap_alpha_gray_alpha_f32;
        has_non_constant_cap_alpha_gray_alpha_f32(self.buffer.as_ref(), self.width, self.stride())
    }
}

impl AssociateAlpha<u8, 4> for ImageStore<'_, u8, 4> {
    fn premultiply_alpha(&self, into: &mut ImageStoreMut<'_, u8, 4>, pool: &novtb::ThreadPool) {
        let dst_stride = into.stride();
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.as_ref();
        premultiply_alpha_rgba(
            dst,
            dst_stride,
            src,
            self.width,
            self.height,
            self.stride(),
            pool,
        );
    }

    #[cfg(not(any(
        any(target_arch = "x86_64", target_arch = "x86"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    fn is_alpha_premultiplication_needed(&self) -> bool {
        use crate::alpha_check::has_non_constant_cap_alpha_rgba8;
        has_non_constant_cap_alpha_rgba8(self.buffer.as_ref(), self.width, self.stride())
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn is_alpha_premultiplication_needed(&self) -> bool {
        use crate::neon::neon_has_non_constant_cap_alpha_rgba8;
        neon_has_non_constant_cap_alpha_rgba8(self.buffer.as_ref(), self.width, self.stride())
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn is_alpha_premultiplication_needed(&self) -> bool {
        use crate::alpha_check::has_non_constant_cap_alpha_rgba8;
        #[cfg(feature = "sse")]
        use crate::sse::sse_has_non_constant_cap_alpha_rgba8;
        #[cfg(all(target_arch = "x86_64", feature = "nightly_avx512"))]
        if std::arch::is_x86_feature_detected!("avx512bw") {
            use crate::avx512::avx512_has_non_constant_cap_alpha_rgba8;
            return avx512_has_non_constant_cap_alpha_rgba8(
                self.buffer.as_ref(),
                self.width,
                self.stride(),
            );
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::avx_has_non_constant_cap_alpha_rgba8;
            return avx_has_non_constant_cap_alpha_rgba8(
                self.buffer.as_ref(),
                self.width,
                self.stride(),
            );
        }
        #[cfg(feature = "sse")]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return sse_has_non_constant_cap_alpha_rgba8(
                self.buffer.as_ref(),
                self.width,
                self.stride(),
            );
        }
        has_non_constant_cap_alpha_rgba8(self.buffer.as_ref(), self.width, self.stride())
    }
}

impl UnassociateAlpha<u8, 4> for ImageStoreMut<'_, u8, 4> {
    fn unpremultiply_alpha(
        &mut self,
        pool: &novtb::ThreadPool,
        workload_strategy: WorkloadStrategy,
    ) {
        let src_stride = self.stride();
        let dst = self.buffer.borrow_mut();
        unpremultiply_alpha_rgba(
            dst,
            self.width,
            self.height,
            src_stride,
            pool,
            workload_strategy,
        );
    }
}

impl UnassociateAlpha<u8, 2> for ImageStoreMut<'_, u8, 2> {
    fn unpremultiply_alpha(
        &mut self,
        pool: &novtb::ThreadPool,
        workload_strategy: WorkloadStrategy,
    ) {
        let src_stride = self.stride();
        let dst = self.buffer.borrow_mut();
        use crate::alpha_handle_u8::unpremultiply_alpha_gray_alpha;
        unpremultiply_alpha_gray_alpha(
            dst,
            self.width,
            self.height,
            src_stride,
            pool,
            workload_strategy,
        );
    }
}

impl UnassociateAlpha<f32, 2> for ImageStoreMut<'_, f32, 2> {
    fn unpremultiply_alpha(&mut self, pool: &novtb::ThreadPool, _: WorkloadStrategy) {
        let src_stride = self.stride();
        let dst = self.buffer.borrow_mut();
        use crate::alpha_handle_f32::unpremultiply_alpha_gray_alpha_f32;
        unpremultiply_alpha_gray_alpha_f32(dst, src_stride, self.width, self.height, pool);
    }
}

impl UnassociateAlpha<u16, 2> for ImageStoreMut<'_, u16, 2> {
    fn unpremultiply_alpha(&mut self, pool: &novtb::ThreadPool, _: WorkloadStrategy) {
        let src_stride = self.stride();
        let dst = self.buffer.borrow_mut();
        use crate::alpha_handle_u16::unpremultiply_alpha_gray_alpha_u16;
        unpremultiply_alpha_gray_alpha_u16(
            dst,
            src_stride,
            self.width,
            self.height,
            self.bit_depth,
            pool,
        );
    }
}

impl AssociateAlpha<u16, 4> for ImageStore<'_, u16, 4> {
    fn premultiply_alpha(&self, into: &mut ImageStoreMut<'_, u16, 4>, pool: &novtb::ThreadPool) {
        let dst_stride = into.stride();
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.as_ref();
        premultiply_alpha_rgba_u16(
            dst,
            dst_stride,
            src,
            self.width,
            self.height,
            self.stride(),
            into.bit_depth,
            pool,
        );
    }

    #[cfg(not(any(
        any(target_arch = "x86_64", target_arch = "x86"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    fn is_alpha_premultiplication_needed(&self) -> bool {
        use crate::alpha_check::has_non_constant_cap_alpha_rgba16;
        has_non_constant_cap_alpha_rgba16(self.buffer.as_ref(), self.width, self.stride())
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn is_alpha_premultiplication_needed(&self) -> bool {
        use crate::neon::neon_has_non_constant_cap_alpha_rgba16;
        neon_has_non_constant_cap_alpha_rgba16(self.buffer.as_ref(), self.width, self.stride())
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    fn is_alpha_premultiplication_needed(&self) -> bool {
        use crate::alpha_check::has_non_constant_cap_alpha_rgba16;
        #[cfg(feature = "sse")]
        use crate::sse::sse_has_non_constant_cap_alpha_rgba16;
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::avx_has_non_constant_cap_alpha_rgba16;
            return avx_has_non_constant_cap_alpha_rgba16(
                self.buffer.as_ref(),
                self.width,
                self.stride(),
            );
        }
        #[cfg(feature = "sse")]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return sse_has_non_constant_cap_alpha_rgba16(
                self.buffer.as_ref(),
                self.width,
                self.stride(),
            );
        }
        has_non_constant_cap_alpha_rgba16(self.buffer.as_ref(), self.width, self.stride())
    }
}

impl AssociateAlpha<f32, 4> for ImageStore<'_, f32, 4> {
    fn premultiply_alpha(&self, into: &mut ImageStoreMut<'_, f32, 4>, pool: &novtb::ThreadPool) {
        let src_stride = self.stride();
        let dst_stride = into.stride();
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.as_ref();
        premultiply_alpha_rgba_f32(
            dst,
            dst_stride,
            src,
            src_stride,
            self.width,
            self.height,
            pool,
        );
    }

    fn is_alpha_premultiplication_needed(&self) -> bool {
        has_non_constant_cap_alpha_rgba_f32(self.buffer.as_ref(), self.width, self.stride())
    }
}

#[cfg(feature = "nightly_f16")]
impl AssociateAlpha<f16, 4> for ImageStore<'_, f16, 4> {
    fn premultiply_alpha(&self, into: &mut ImageStoreMut<'_, f16, 4>, pool: &novtb::ThreadPool) {
        let src_stride = self.stride();
        let dst_stride = into.stride();
        let dst = into.buffer.borrow_mut();
        let src = self.buffer.as_ref();
        premultiply_alpha_rgba_f16(
            dst,
            dst_stride,
            src,
            src_stride,
            self.width,
            self.height,
            pool,
        );
    }

    fn is_alpha_premultiplication_needed(&self) -> bool {
        true
    }
}

impl UnassociateAlpha<u16, 4> for ImageStoreMut<'_, u16, 4> {
    fn unpremultiply_alpha(&mut self, pool: &novtb::ThreadPool, _: WorkloadStrategy) {
        let src_stride = self.stride();
        let in_place = self.buffer.borrow_mut();
        unpremultiply_alpha_rgba_u16(
            in_place,
            src_stride,
            self.width,
            self.height,
            self.bit_depth,
            pool,
        );
    }
}

impl UnassociateAlpha<f32, 4> for ImageStoreMut<'_, f32, 4> {
    fn unpremultiply_alpha(&mut self, pool: &novtb::ThreadPool, _: WorkloadStrategy) {
        let stride = self.stride();
        let dst = self.buffer.borrow_mut();
        unpremultiply_alpha_rgba_f32(dst, stride, self.width, self.height, pool);
    }
}

#[cfg(feature = "nightly_f16")]
impl UnassociateAlpha<f16, 4> for ImageStoreMut<'_, f16, 4> {
    fn unpremultiply_alpha(&mut self, pool: &novtb::ThreadPool, _: WorkloadStrategy) {
        let stride = self.stride();
        let dst = self.buffer.borrow_mut();
        unpremultiply_alpha_rgba_f16(dst, stride, self.width, self.height, pool);
    }
}

pub type Planar8ImageStore<'a> = ImageStore<'a, u8, 1>;
pub type Planar8ImageStoreMut<'a> = ImageStoreMut<'a, u8, 1>;
pub type CbCr8ImageStore<'a> = ImageStore<'a, u8, 2>;
pub type CbCr8ImageStoreMut<'a> = ImageStoreMut<'a, u8, 2>;
pub type GrayAlpha8ImageStore<'a> = ImageStore<'a, u8, 2>;
pub type GrayAlpha8ImageStoreMut<'a> = ImageStoreMut<'a, u8, 2>;
pub type Rgba8ImageStore<'a> = ImageStore<'a, u8, 4>;
pub type Rgba8ImageStoreMut<'a> = ImageStoreMut<'a, u8, 4>;
pub type Rgb8ImageStore<'a> = ImageStore<'a, u8, 3>;
pub type Rgb8ImageStoreMut<'a> = ImageStoreMut<'a, u8, 3>;

pub type Planar16ImageStore<'a> = ImageStore<'a, u16, 1>;
pub type Planar16ImageStoreMut<'a> = ImageStoreMut<'a, u16, 1>;
pub type CbCr16ImageStore<'a> = ImageStore<'a, u16, 2>;
pub type CbCr16ImageStoreMut<'a> = ImageStoreMut<'a, u16, 2>;
pub type GrayAlpha16ImageStore<'a> = ImageStore<'a, u16, 2>;
pub type GrayAlpha16ImageStoreMut<'a> = ImageStoreMut<'a, u16, 2>;
pub type Rgba16ImageStore<'a> = ImageStore<'a, u16, 4>;
pub type Rgba16ImageStoreMut<'a> = ImageStoreMut<'a, u16, 4>;
pub type Rgb16ImageStore<'a> = ImageStore<'a, u16, 3>;
pub type Rgb16ImageStoreMut<'a> = ImageStoreMut<'a, u16, 3>;

#[cfg(feature = "nightly_f16")]
pub type PlanarF16ImageStore<'a> = ImageStore<'a, f16, 1>;
#[cfg(feature = "nightly_f16")]
pub type PlanarF16ImageStoreMut<'a> = ImageStoreMut<'a, f16, 1>;
#[cfg(feature = "nightly_f16")]
pub type CbCrF16ImageStore<'a> = ImageStore<'a, f16, 2>;
#[cfg(feature = "nightly_f16")]
pub type CbCrF16ImageStoreMut<'a> = ImageStoreMut<'a, f16, 2>;
#[cfg(feature = "nightly_f16")]
pub type RgbaF16ImageStore<'a> = ImageStore<'a, f16, 4>;
#[cfg(feature = "nightly_f16")]
pub type RgbaF16ImageStoreMut<'a> = ImageStoreMut<'a, f16, 4>;
#[cfg(feature = "nightly_f16")]
pub type RgbF16ImageStore<'a> = ImageStore<'a, f16, 3>;
#[cfg(feature = "nightly_f16")]
pub type RgbF16ImageStoreMut<'a> = ImageStoreMut<'a, f16, 3>;

pub type PlanarF32ImageStore<'a> = ImageStore<'a, f32, 1>;
pub type PlanarF32ImageStoreMut<'a> = ImageStoreMut<'a, f32, 1>;
pub type CbCrF32ImageStore<'a> = ImageStore<'a, f32, 2>;
pub type CbCrF32ImageStoreMut<'a> = ImageStoreMut<'a, f32, 2>;
pub type GrayAlphaF32ImageStore<'a> = ImageStore<'a, f32, 2>;
pub type GrayAlphaF32ImageStoreMut<'a> = ImageStoreMut<'a, f32, 2>;
pub type RgbaF32ImageStore<'a> = ImageStore<'a, f32, 4>;
pub type RgbaF32ImageStoreMut<'a> = ImageStoreMut<'a, f32, 4>;
pub type RgbF32ImageStore<'a> = ImageStore<'a, f32, 3>;
pub type RgbF32ImageStoreMut<'a> = ImageStoreMut<'a, f32, 3>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_store_alpha_test_rgba8() {
        let image_size = 256usize;
        let mut image = vec![0u8; image_size * image_size * 4];
        image[3 + 150 * 4] = 75;
        let store = ImageStore::<u8, 4>::from_slice(&image, image_size, image_size).unwrap();
        let has_alpha = store.is_alpha_premultiplication_needed();
        assert_eq!(true, has_alpha);
    }

    #[test]
    fn check_alpha_not_exists_rgba8() {
        let image_size = 256usize;
        let image = vec![255u8; image_size * image_size * 4];
        let store = ImageStore::<u8, 4>::from_slice(&image, image_size, image_size).unwrap();
        let has_alpha = store.is_alpha_premultiplication_needed();
        assert_eq!(false, has_alpha);
    }
}
