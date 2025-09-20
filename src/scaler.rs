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
#![forbid(unsafe_code)]
use crate::ar30::{Ar30ByteOrder, Rgb30};
use crate::convolution::{ConvolutionOptions, HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::image_size::ImageSize;
use crate::image_store::{
    AssociateAlpha, CheckStoreDensity, ImageStore, ImageStoreMut, UnassociateAlpha,
};
use crate::math::WeightsGenerator;
use crate::nearest_sampler::resize_nearest;
use crate::pic_scale_error::{PicScaleError, try_vec};
use crate::resize_ar30::resize_ar30_impl;
use crate::support::check_image_size_overflow;
use crate::threading_policy::ThreadingPolicy;
use crate::{
    CbCr8ImageStore, CbCr16ImageStore, CbCrF32ImageStore, Planar8ImageStore, Planar16ImageStore,
    PlanarF32ImageStore, ResamplingFunction, Rgb8ImageStore, Rgb16ImageStore, RgbF32ImageStore,
    Rgba8ImageStore, Rgba16ImageStore, RgbaF32ImageStore,
};
use std::fmt::Debug;

#[derive(Debug, Copy, Clone)]
/// Represents base scaling structure
pub struct Scaler {
    pub(crate) function: ResamplingFunction,
    pub(crate) threading_policy: ThreadingPolicy,
    pub workload_strategy: WorkloadStrategy,
}

/// 8 bit-depth images scaling trait
pub trait Scaling {
    /// Sets threading policy
    ///
    /// Setting up threading policy, refer to [crate::ThreadingPolicy] for more info
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    /// use pic_scale::{ResamplingFunction, Scaler, Scaling, ThreadingPolicy};
    /// let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    /// scaler.set_threading_policy(ThreadingPolicy::Adaptive);
    /// ```
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy);

    /// Performs rescaling for planar image
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u8, 1>::alloc(50, 50);
    ///  scaler.resize_plane(&src_store, &mut dst_store).unwrap();
    /// ```
    fn resize_plane<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 1>,
        into: &mut ImageStoreMut<'a, u8, 1>,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for CbCr8 ( or 2 interleaved channels )
    ///
    /// Scales 2 interleaved channels as CbCr8, optionally it could handle LumaAlpha images also
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u8, 2>::alloc(50, 50);
    ///  scaler.resize_cbcr8(&src_store, &mut dst_store).unwrap();
    /// ```
    fn resize_cbcr8<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 2>,
        into: &mut ImageStoreMut<'a, u8, 2>,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for Gray Alpha ( or 2 interleaved channels with aloha )
    ///
    /// Scales 2 interleaved channels as Gray Alpha
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u8, 2>::alloc(50, 50);
    ///  scaler.resize_gray_alpha(&src_store, &mut dst_store, true).unwrap();
    /// ```
    fn resize_gray_alpha<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 2>,
        into: &mut ImageStoreMut<'a, u8, 2>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for RGB, channel order does not matter
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u8, 3>::alloc(50, 50);
    ///  scaler.resize_rgb(&src_store, &mut dst_store).unwrap();
    /// ```
    fn resize_rgb<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 3>,
        into: &mut ImageStoreMut<'a, u8, 3>,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for RGBA
    ///
    /// This method may premultiply and un associate alpha if required.
    /// Alpha position is always considered as last
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u8, 4>::alloc(50, 50);
    ///  scaler.resize_rgba(&src_store, &mut dst_store, false).unwrap();
    /// ```
    fn resize_rgba<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 4>,
        into: &mut ImageStoreMut<'a, u8, 4>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError>;
}

/// f32 images scaling trait
pub trait ScalingF32 {
    /// Performs rescaling planar f32 image
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ScalingF32};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<f32, 1>::alloc(50, 50);
    ///  scaler.resize_plane_f32(&src_store, &mut dst_store).unwrap();
    /// ```
    fn resize_plane_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 1>,
        into: &mut ImageStoreMut<'a, f32, 1>,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for CbCr f32 image
    ///
    /// Scales an interleaved CbCr f32. Also, could handle LumaAlpha images.
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ScalingF32};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<f32, 2>::alloc(50, 50);
    ///  scaler.resize_cbcr_f32(&src_store, &mut dst_store).unwrap();
    /// ```
    fn resize_cbcr_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 2>,
        into: &mut ImageStoreMut<'a, f32, 2>,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for Gray Alpha f32 image
    ///
    /// Scales an interleaved Gray and Alpha in f32.
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ScalingF32};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<f32, 2>::alloc(50, 50);
    ///  scaler.resize_gray_alpha_f32(&src_store, &mut dst_store, true).unwrap();
    /// ```
    fn resize_gray_alpha_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 2>,
        into: &mut ImageStoreMut<'a, f32, 2>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for RGB f32
    ///
    /// Scales an image RGB f32, channel order does not matter
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ScalingF32};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<f32, 3>::alloc(50, 50);
    ///  scaler.resize_rgb_f32(&src_store, &mut dst_store).unwrap();
    /// ```
    fn resize_rgb_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 3>,
        into: &mut ImageStoreMut<'a, f32, 3>,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for RGBA f32
    ///
    /// Scales an image RGBA f32, alpha expected to be at last position if
    /// alpha pre-multiplication is requested
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ScalingF32};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<f32, 4>::alloc(50, 50);
    ///  scaler.resize_rgba_f32(&src_store, &mut dst_store, false).unwrap();
    /// ```
    fn resize_rgba_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 4>,
        into: &mut ImageStoreMut<'a, f32, 4>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError>;
}

/// Defines execution hint about preferred strategy
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
pub enum WorkloadStrategy {
    /// Prefers quality to speed
    PreferQuality,
    /// Prefers speed to quality
    #[default]
    PreferSpeed,
}

/// 8+ bit-depth images scaling trait
pub trait ScalingU16 {
    /// Performs rescaling for Planar u16
    ///
    /// Scales planar high bit-depth image stored in `u16` type.
    /// To perform scaling image bit-depth should be set in target image,
    /// source image expects to have the same one.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    ///
    /// # Panic
    /// Method panics if bit-depth < 1 or bit-depth > 16
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ScalingU16};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u16, 1>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_plane_u16(&src_store, &mut dst_store).unwrap();
    /// ```
    fn resize_plane_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 1>,
        into: &mut ImageStoreMut<'a, u16, 1>,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for CbCr16
    ///
    /// Scales CbCr high bit-depth interleaved image in `u16` type, optionally it could handle LumaAlpha images also
    /// To perform scaling image bit-depth should be set in target image,
    /// source image expects to have the same one.
    /// Channel order does not matter.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    ///
    /// # Panics
    /// Method panics if bit-depth < 1 or bit-depth > 16
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ScalingU16};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u16, 2>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_cbcr_u16(&src_store, &mut dst_store).unwrap();
    /// ```
    ///
    fn resize_cbcr_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 2>,
        into: &mut ImageStoreMut<'a, u16, 2>,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for Gray Alpha high bit-depth ( or 2 interleaved channels with aloha )
    ///
    /// Scales 2 interleaved channels as Gray Alpha
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ScalingU16};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u16, 2>::alloc_with_depth(50, 50, 16);
    ///  scaler.resize_gray_alpha16(&src_store, &mut dst_store, true).unwrap();
    /// ```
    fn resize_gray_alpha16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 2>,
        into: &mut ImageStoreMut<'a, u16, 2>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for RGB
    ///
    /// Scales RGB high bit-depth image stored in `u16` type.
    /// To perform scaling image bit-depth should be set in target image,
    /// source image expects to have the same one.
    /// Channel order does not matter.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    ///
    /// # Panics
    /// Method panics if bit-depth < 1 or bit-depth > 16
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ScalingU16};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u16, 3>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_rgb_u16(&src_store, &mut dst_store).unwrap();
    /// ```
    ///
    fn resize_rgb_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 3>,
        into: &mut ImageStoreMut<'a, u16, 3>,
    ) -> Result<(), PicScaleError>;

    /// Performs rescaling for RGBA high bit-depth
    ///
    /// Scales RGB high bit-depth image stored in `u16` type.
    /// To perform scaling image bit-depth should be set in target image,
    /// source image expects to have the same one.
    /// If pre-multiplication is requested alpha should be at last, otherwise
    /// channel order does not matter.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    /// `premultiply_alpha` - flags is alpha is premultiplied
    ///
    /// # Panics
    /// Method panics if bit-depth < 1 or bit-depth > 16
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ScalingU16};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u16, 4>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_rgba_u16(&src_store, &mut dst_store, true).unwrap();
    /// ```
    ///
    fn resize_rgba_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 4>,
        into: &mut ImageStoreMut<'a, u16, 4>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError>;
}

impl Scaler {
    /// Creates new [Scaler] instance with corresponding filter
    ///
    /// Creates default [crate::Scaler] with corresponding filter and default [ThreadingPolicy::Single]
    ///
    pub fn new(filter: ResamplingFunction) -> Self {
        Scaler {
            function: filter,
            threading_policy: ThreadingPolicy::Single,
            workload_strategy: WorkloadStrategy::default(),
        }
    }

    /// Sets preferred workload strategy
    ///
    /// This is hint only, it may change something, or may not.
    pub fn set_workload_strategy(&mut self, workload_strategy: WorkloadStrategy) {
        self.workload_strategy = workload_strategy;
    }
}

impl Scaler {
    pub(crate) fn generic_resize<
        'a,
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<f32> + 'static,
        const N: usize,
    >(
        &self,
        store: &ImageStore<'a, T, N>,
        into: &mut ImageStoreMut<'a, T, N>,
    ) -> Result<(), PicScaleError>
    where
        ImageStore<'a, T, N>: VerticalConvolutionPass<T, N> + HorizontalConvolutionPass<T, N>,
        ImageStoreMut<'a, T, N>: CheckStoreDensity,
    {
        let new_size = into.get_size();
        into.validate()?;
        store.validate()?;
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err(PicScaleError::ZeroImageDimensions);
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err(PicScaleError::SourceImageIsTooLarge);
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err(PicScaleError::DestinationImageIsTooLarge);
        }

        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }

        if store.width == new_size.width && store.height == new_size.height {
            store.copied_to_mut(into);
            return Ok(());
        }

        let nova_thread_pool = self
            .threading_policy
            .get_nova_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == ResamplingFunction::Nearest {
            resize_nearest::<T, N>(
                store.buffer.as_ref(),
                store.width,
                store.height,
                into.buffer.borrow_mut(),
                new_size.width,
                new_size.height,
                &nova_thread_pool,
            );
            return Ok(());
        }

        let should_do_horizontal = store.width != new_size.width;
        let should_do_vertical = store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        if should_do_vertical && should_do_horizontal {
            let mut target_vertical = try_vec![T::default(); store.width * new_size.height * N];

            let mut new_image_vertical = ImageStoreMut::<T, N>::from_slice(
                &mut target_vertical,
                store.width,
                new_size.height,
            )?;
            new_image_vertical.bit_depth = into.bit_depth;
            let vertical_filters = T::make_weights(self.function, store.height, new_size.height)?;
            let options = ConvolutionOptions::new(self.workload_strategy);
            store.convolve_vertical(
                vertical_filters,
                &mut new_image_vertical,
                &nova_thread_pool,
                options,
            );

            let new_immutable_store = ImageStore::<T, N> {
                buffer: std::borrow::Cow::Owned(target_vertical),
                channels: N,
                width: store.width,
                height: new_size.height,
                stride: store.width * N,
                bit_depth: into.bit_depth,
            };
            let horizontal_filters = T::make_weights(self.function, store.width, new_size.width)?;
            let options = ConvolutionOptions::new(self.workload_strategy);
            new_immutable_store.convolve_horizontal(
                horizontal_filters,
                into,
                &nova_thread_pool,
                options,
            );
            Ok(())
        } else if should_do_vertical {
            let vertical_filters = T::make_weights(self.function, store.height, new_size.height)?;
            let options = ConvolutionOptions::new(self.workload_strategy);
            store.convolve_vertical(vertical_filters, into, &nova_thread_pool, options);
            Ok(())
        } else {
            assert!(should_do_horizontal);
            let horizontal_filters = T::make_weights(self.function, store.width, new_size.width)?;
            let options = ConvolutionOptions::new(self.workload_strategy);
            store.convolve_horizontal(horizontal_filters, into, &nova_thread_pool, options);
            Ok(())
        }
    }

    fn forward_resize_with_alpha<
        'a,
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<f32> + 'static,
        const N: usize,
    >(
        &self,
        store: &ImageStore<'a, T, N>,
        into: &mut ImageStoreMut<'a, T, N>,
        premultiply_alpha_requested: bool,
        nova_thread_pool: &novtb::ThreadPool,
    ) -> Result<(), PicScaleError>
    where
        ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, N> + HorizontalConvolutionPass<T, N> + AssociateAlpha<T, N>,
        ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        let new_size = into.get_size();
        let mut src_store: std::borrow::Cow<'_, ImageStore<'_, T, N>> =
            std::borrow::Cow::Borrowed(store);

        let mut has_alpha_premultiplied = true;

        if premultiply_alpha_requested {
            let is_alpha_premultiplication_reasonable =
                src_store.is_alpha_premultiplication_needed();
            if is_alpha_premultiplication_reasonable {
                let mut target_premultiplied =
                    try_vec![T::default(); src_store.width * src_store.height * N];
                let mut new_store = ImageStoreMut::<T, N>::from_slice(
                    &mut target_premultiplied,
                    src_store.width,
                    src_store.height,
                )?;
                new_store.bit_depth = into.bit_depth;
                src_store.premultiply_alpha(&mut new_store, nova_thread_pool);
                src_store = std::borrow::Cow::Owned(ImageStore::<T, N> {
                    buffer: std::borrow::Cow::Owned(target_premultiplied),
                    channels: N,
                    width: src_store.width,
                    height: src_store.height,
                    stride: src_store.width * N,
                    bit_depth: into.bit_depth,
                });
                has_alpha_premultiplied = true;
            }
        }

        let mut target_vertical = try_vec![T::default(); src_store.width * new_size.height * N];

        let mut new_image_vertical = ImageStoreMut::<T, N>::from_slice(
            &mut target_vertical,
            src_store.width,
            new_size.height,
        )?;
        new_image_vertical.bit_depth = into.bit_depth;
        let vertical_filters = T::make_weights(self.function, src_store.height, new_size.height)?;
        let options = ConvolutionOptions::new(self.workload_strategy);
        src_store.convolve_vertical(
            vertical_filters,
            &mut new_image_vertical,
            nova_thread_pool,
            options,
        );

        let new_immutable_store = ImageStore::<T, N> {
            buffer: std::borrow::Cow::Owned(target_vertical),
            channels: N,
            width: src_store.width,
            height: new_size.height,
            stride: src_store.width * N,
            bit_depth: into.bit_depth,
        };
        let horizontal_filters = T::make_weights(self.function, src_store.width, new_size.width)?;
        let options = ConvolutionOptions::new(self.workload_strategy);
        new_immutable_store.convolve_horizontal(
            horizontal_filters,
            into,
            nova_thread_pool,
            options,
        );

        if premultiply_alpha_requested && has_alpha_premultiplied {
            into.unpremultiply_alpha(nova_thread_pool, self.workload_strategy);
        }

        Ok(())
    }

    fn forward_resize_vertical_with_alpha<
        'a,
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<f32> + 'static,
        const N: usize,
    >(
        &self,
        store: &ImageStore<'a, T, N>,
        into: &mut ImageStoreMut<'a, T, N>,
        premultiply_alpha_requested: bool,
        nova_thread_pool: &novtb::ThreadPool,
    ) -> Result<(), PicScaleError>
    where
        ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, N> + HorizontalConvolutionPass<T, N> + AssociateAlpha<T, N>,
        ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        let new_size = into.get_size();
        let mut src_store = std::borrow::Cow::Borrowed(store);

        let mut has_alpha_premultiplied = true;

        if premultiply_alpha_requested {
            let is_alpha_premultiplication_reasonable =
                src_store.is_alpha_premultiplication_needed();
            if is_alpha_premultiplication_reasonable {
                let mut target_premultiplied =
                    try_vec![T::default(); src_store.width * src_store.height * N];
                let mut new_store = ImageStoreMut::<T, N>::from_slice(
                    &mut target_premultiplied,
                    src_store.width,
                    src_store.height,
                )?;
                new_store.bit_depth = into.bit_depth;
                src_store.premultiply_alpha(&mut new_store, nova_thread_pool);
                src_store = std::borrow::Cow::Owned(ImageStore::<T, N> {
                    buffer: std::borrow::Cow::Owned(target_premultiplied),
                    channels: N,
                    width: src_store.width,
                    height: src_store.height,
                    stride: src_store.width * N,
                    bit_depth: into.bit_depth,
                });
                has_alpha_premultiplied = true;
            }
        }

        let vertical_filters = T::make_weights(self.function, src_store.height, new_size.height)?;
        let options = ConvolutionOptions::new(self.workload_strategy);
        src_store.convolve_vertical(vertical_filters, into, nova_thread_pool, options);

        if premultiply_alpha_requested && has_alpha_premultiplied {
            into.unpremultiply_alpha(nova_thread_pool, self.workload_strategy);
        }

        Ok(())
    }

    fn forward_resize_horizontal_with_alpha<
        'a,
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<f32> + 'static,
        const N: usize,
    >(
        &self,
        store: &ImageStore<'a, T, N>,
        into: &mut ImageStoreMut<'a, T, N>,
        premultiply_alpha_requested: bool,
        nova_thread_pool: &novtb::ThreadPool,
    ) -> Result<(), PicScaleError>
    where
        ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, N> + HorizontalConvolutionPass<T, N> + AssociateAlpha<T, N>,
        ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        let new_size = into.get_size();
        let mut src_store = std::borrow::Cow::Borrowed(store);

        let mut has_alpha_premultiplied = true;

        if premultiply_alpha_requested {
            let is_alpha_premultiplication_reasonable =
                src_store.is_alpha_premultiplication_needed();
            if is_alpha_premultiplication_reasonable {
                let mut target_premultiplied =
                    try_vec![T::default(); src_store.width * src_store.height * N];
                let mut new_store = ImageStoreMut::<T, N>::from_slice(
                    &mut target_premultiplied,
                    src_store.width,
                    src_store.height,
                )?;
                new_store.bit_depth = into.bit_depth;
                src_store.premultiply_alpha(&mut new_store, nova_thread_pool);
                src_store = std::borrow::Cow::Owned(ImageStore::<T, N> {
                    buffer: std::borrow::Cow::Owned(target_premultiplied),
                    channels: N,
                    width: src_store.width,
                    height: src_store.height,
                    stride: src_store.width * N,
                    bit_depth: into.bit_depth,
                });
                has_alpha_premultiplied = true;
            }
        }

        let horizontal_filters = T::make_weights(self.function, src_store.width, new_size.width)?;
        let options = ConvolutionOptions::new(self.workload_strategy);
        src_store.convolve_horizontal(horizontal_filters, into, nova_thread_pool, options);

        if premultiply_alpha_requested && has_alpha_premultiplied {
            into.unpremultiply_alpha(nova_thread_pool, self.workload_strategy);
        }

        Ok(())
    }

    pub(crate) fn generic_resize_with_alpha<
        'a,
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<f32> + 'static,
        const N: usize,
    >(
        &self,
        store: &ImageStore<'a, T, N>,
        into: &mut ImageStoreMut<'a, T, N>,
        premultiply_alpha_requested: bool,
    ) -> Result<(), PicScaleError>
    where
        ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, N> + HorizontalConvolutionPass<T, N> + AssociateAlpha<T, N>,
        ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        let new_size = into.get_size();
        into.validate()?;
        store.validate()?;
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err(PicScaleError::ZeroImageDimensions);
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err(PicScaleError::SourceImageIsTooLarge);
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err(PicScaleError::DestinationImageIsTooLarge);
        }

        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }

        if store.width == new_size.width && store.height == new_size.height {
            store.copied_to_mut(into);
            return Ok(());
        }

        let nova_thread_pool = self
            .threading_policy
            .get_nova_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == ResamplingFunction::Nearest {
            resize_nearest::<T, N>(
                store.buffer.as_ref(),
                store.width,
                store.height,
                into.buffer.borrow_mut(),
                new_size.width,
                new_size.height,
                &nova_thread_pool,
            );
            return Ok(());
        }

        let should_do_horizontal = store.width != new_size.width;
        let should_do_vertical = store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        if should_do_vertical && should_do_horizontal {
            self.forward_resize_with_alpha(
                store,
                into,
                premultiply_alpha_requested,
                &nova_thread_pool,
            )
        } else if should_do_vertical {
            self.forward_resize_vertical_with_alpha(
                store,
                into,
                premultiply_alpha_requested,
                &nova_thread_pool,
            )
        } else {
            assert!(should_do_horizontal);
            self.forward_resize_horizontal_with_alpha(
                store,
                into,
                premultiply_alpha_requested,
                &nova_thread_pool,
            )
        }
    }
}

impl Scaling for Scaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.threading_policy = threading_policy;
    }

    fn resize_plane<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 1>,
        into: &mut ImageStoreMut<'a, u8, 1>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    fn resize_cbcr8<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 2>,
        into: &mut ImageStoreMut<'a, u8, 2>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    fn resize_gray_alpha<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 2>,
        into: &mut ImageStoreMut<'a, u8, 2>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError> {
        self.generic_resize_with_alpha(store, into, premultiply_alpha)
    }

    fn resize_rgb<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 3>,
        into: &mut ImageStoreMut<'a, u8, 3>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    fn resize_rgba<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 4>,
        into: &mut ImageStoreMut<'a, u8, 4>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError> {
        self.generic_resize_with_alpha(store, into, premultiply_alpha)
    }
}

impl ScalingF32 for Scaler {
    fn resize_plane_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 1>,
        into: &mut ImageStoreMut<'a, f32, 1>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    fn resize_cbcr_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 2>,
        into: &mut ImageStoreMut<'a, f32, 2>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    fn resize_gray_alpha_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 2>,
        into: &mut ImageStoreMut<'a, f32, 2>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError> {
        self.generic_resize_with_alpha(store, into, premultiply_alpha)
    }

    fn resize_rgb_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 3>,
        into: &mut ImageStoreMut<'a, f32, 3>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    fn resize_rgba_f32<'a>(
        &'a self,
        store: &ImageStore<'a, f32, 4>,
        into: &mut ImageStoreMut<'a, f32, 4>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError> {
        self.generic_resize_with_alpha(store, into, premultiply_alpha)
    }
}

impl ScalingU16 for Scaler {
    /// Performs rescaling for RGB
    ///
    /// Scales RGB high bit-depth image stored in `u16` type.
    /// To perform scaling image bit-depth should be set in target image,
    /// source image expects to have the same one.
    /// Channel order does not matter.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    ///
    /// # Panics
    /// Method panics if bit-depth < 1 or bit-depth > 16
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ScalingU16};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u16, 3>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_rgb_u16(&src_store, &mut dst_store).unwrap();
    /// ```
    ///
    fn resize_rgb_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 3>,
        into: &mut ImageStoreMut<'a, u16, 3>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    fn resize_cbcr_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 2>,
        into: &mut ImageStoreMut<'a, u16, 2>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    fn resize_gray_alpha16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 2>,
        into: &mut ImageStoreMut<'a, u16, 2>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError> {
        self.generic_resize_with_alpha(store, into, premultiply_alpha)
    }

    /// Resizes u16 image
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    /// `premultiply_alpha` - flags is alpha is premultiplied
    ///
    /// # Panics
    /// Method panics if bit -depth < 1 or bit-depth > 16
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ScalingU16};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u16, 4>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_rgba_u16(&src_store, &mut dst_store, true).unwrap();
    /// ```
    ///
    fn resize_rgba_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 4>,
        into: &mut ImageStoreMut<'a, u16, 4>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError> {
        self.generic_resize_with_alpha(store, into, premultiply_alpha)
    }

    /// Performs rescaling for Planar u16
    ///
    /// Scales planar high bit-depth image stored in `u16` type.
    /// To perform scaling image bit-depth should be set in target image,
    /// source image expects to have the same one.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    ///
    /// # Panic
    /// Method panics if bit-depth < 1 or bit-depth > 16
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ScalingU16};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<u16, 1>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_plane_u16(&src_store, &mut dst_store).unwrap();
    /// ```
    fn resize_plane_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 1>,
        into: &mut ImageStoreMut<'a, u16, 1>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }
}

impl Scaler {
    /// Resizes RGBA2101010 image
    ///
    /// This method ignores alpha scaling.
    ///
    /// # Arguments
    /// `src_image` - source AR30 image
    /// `dst_image` - destination AR30 image
    /// `new_size` - New image size
    ///
    pub fn resize_ar30(
        &self,
        src_image: &ImageStore<u8, 4>,
        dst_image: &mut ImageStoreMut<u8, 4>,
        order: Ar30ByteOrder,
    ) -> Result<(), PicScaleError> {
        src_image.validate()?;
        dst_image.validate()?;
        let dst_size = dst_image.get_size();
        let dst_stride = dst_image.stride();
        match order {
            Ar30ByteOrder::Host => {
                resize_ar30_impl::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Host as usize }>(
                    src_image.as_bytes(),
                    src_image.stride,
                    src_image.get_size(),
                    dst_image.buffer.borrow_mut(),
                    dst_stride,
                    dst_size,
                    self,
                )
            }
            Ar30ByteOrder::Network => {
                resize_ar30_impl::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Network as usize }>(
                    src_image.as_bytes(),
                    src_image.stride,
                    src_image.get_size(),
                    dst_image.buffer.borrow_mut(),
                    dst_stride,
                    dst_size,
                    self,
                )
            }
        }
    }

    /// Resizes RGBA1010102 image
    ///
    /// This method ignores alpha scaling.
    ///
    /// # Arguments
    /// `src_image` - source RA30 image
    /// `dst_image` - destination RA30 image
    ///
    pub fn resize_ra30(
        &self,
        src_image: &ImageStore<u8, 4>,
        dst_image: &mut ImageStoreMut<u8, 4>,
        order: Ar30ByteOrder,
    ) -> Result<(), PicScaleError> {
        src_image.validate()?;
        dst_image.validate()?;
        let dst_size = dst_image.get_size();
        let dst_stride = dst_image.stride();
        match order {
            Ar30ByteOrder::Host => {
                resize_ar30_impl::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Host as usize }>(
                    src_image.as_bytes(),
                    src_image.stride,
                    src_image.get_size(),
                    dst_image.buffer.borrow_mut(),
                    dst_stride,
                    dst_size,
                    self,
                )
            }
            Ar30ByteOrder::Network => {
                resize_ar30_impl::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Network as usize }>(
                    src_image.as_bytes(),
                    src_image.stride,
                    src_image.get_size(),
                    dst_image.buffer.borrow_mut(),
                    dst_stride,
                    dst_size,
                    self,
                )
            }
        }
    }
}

/// Declares default scaling options
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Default)]
pub struct ScalingOptions {
    pub resampling_function: ResamplingFunction,
    pub premultiply_alpha: bool,
    pub threading_policy: ThreadingPolicy,
}

/// Generic trait for [ImageStore] to implement abstract scaling.
pub trait ImageStoreScaling<'b, T, const N: usize>
where
    T: Clone + Copy + Debug,
{
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, T, N>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError>;
}

macro_rules! def_image_scaling_alpha {
    ($clazz: ident, $fx_type: ident, $cn: expr) => {
        impl<'b> ImageStoreScaling<'b, $fx_type, $cn> for $clazz<'b> {
            fn scale(
                &self,
                store: &mut ImageStoreMut<'b, $fx_type, $cn>,
                options: ScalingOptions,
            ) -> Result<(), PicScaleError> {
                let mut scaler = Scaler::new(options.resampling_function);
                scaler.set_threading_policy(options.threading_policy);
                scaler.generic_resize_with_alpha(self, store, options.premultiply_alpha)
            }
        }
    };
}

macro_rules! def_image_scaling {
    ($clazz: ident, $fx_type: ident, $cn: expr) => {
        impl<'b> ImageStoreScaling<'b, $fx_type, $cn> for $clazz<'b> {
            fn scale(
                &self,
                store: &mut ImageStoreMut<'b, $fx_type, $cn>,
                options: ScalingOptions,
            ) -> Result<(), PicScaleError> {
                let mut scaler = Scaler::new(options.resampling_function);
                scaler.set_threading_policy(options.threading_policy);
                scaler.generic_resize(self, store)
            }
        }
    };
}

def_image_scaling_alpha!(Rgba8ImageStore, u8, 4);
def_image_scaling!(Rgb8ImageStore, u8, 3);
def_image_scaling!(CbCr8ImageStore, u8, 2);
def_image_scaling!(Planar8ImageStore, u8, 1);
def_image_scaling!(Planar16ImageStore, u16, 1);
def_image_scaling!(CbCr16ImageStore, u16, 2);
def_image_scaling!(Rgb16ImageStore, u16, 3);
def_image_scaling_alpha!(Rgba16ImageStore, u16, 4);
def_image_scaling!(PlanarF32ImageStore, f32, 1);
def_image_scaling!(CbCrF32ImageStore, f32, 2);
def_image_scaling!(RgbF32ImageStore, f32, 3);
def_image_scaling_alpha!(RgbaF32ImageStore, f32, 4);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! check_rgba8 {
        ($dst: expr, $image_width: expr, $max: expr) => {
            {
                for (y, row) in $dst.chunks_exact($image_width * 4).enumerate() {
                    for (i, dst) in row.chunks_exact(4).enumerate() {
                        let diff0 = (dst[0] as i32 - 124).abs();
                        let diff1 = (dst[1] as i32 - 41).abs();
                        let diff2 = (dst[2] as i32 - 99).abs();
                        let diff3 = (dst[3] as i32 - 77).abs();
                        assert!(
                            diff0 < $max,
                            "Diff for channel 0 is expected < {}, but it was {diff0}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff1 < $max,
                            "Diff for channel 1 is expected < {}, but it was {diff1}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff2 < $max,
                            "Diff for channel 2 is expected < {}, but it was {diff2}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff3 < $max,
                            "Diff for channel 3 is expected < {}, but it was {diff3}, at (y: {y}, x: {i})",
                            $max
                        );
                    }
                }
            }
        };
    }

    macro_rules! check_rgb16 {
        ($dst: expr, $image_width: expr, $max: expr) => {
            {
                for (y, row) in $dst.chunks_exact($image_width * 3).enumerate() {
                    for (i, dst) in row.chunks_exact(3).enumerate() {
                        let diff0 = (dst[0] as i32 - 124).abs();
                        let diff1 = (dst[1] as i32 - 41).abs();
                        let diff2 = (dst[2] as i32 - 99).abs();
                        assert!(
                            diff0 < $max,
                            "Diff for channel 0 is expected < {}, but it was {diff0}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff1 < $max,
                            "Diff for channel 1 is expected < {}, but it was {diff1}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff2 < $max,
                            "Diff for channel 2 is expected < {}, but it was {diff2}, at (y: {y}, x: {i})",
                            $max
                        );
                    }
                }
            }
        };
    }

    #[test]
    fn check_rgba8_resizing_vertical() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        scaler
            .resize_rgba(&src_store, &mut target_store, false)
            .unwrap();
        let target_data = target_store.buffer.borrow();
        check_rgba8!(target_data, image_width, 34);
    }

    #[test]
    fn check_rgba8_resizing_both() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        image[3] = 78;
        let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width / 2, image_height / 2);
        scaler
            .resize_rgba(&src_store, &mut target_store, false)
            .unwrap();
        let target_data = target_store.buffer.borrow();
        check_rgba8!(target_data, image_width, 34);
    }

    #[test]
    fn check_rgba8_resizing_alpha() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        image[3] = 78;
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width / 2, image_height / 2);
        scaler
            .resize_rgba(&src_store, &mut target_store, true)
            .unwrap();
        let target_data = target_store.buffer.borrow();
        check_rgba8!(target_data, image_width, 126);
    }

    #[test]
    fn check_rgb8_resizing_vertical() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 3;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        scaler.resize_rgb(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 85);
    }

    #[test]
    fn check_rgb8_resizing_vertical_threading() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 3;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
        scaler.set_threading_policy(ThreadingPolicy::Adaptive);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        scaler.resize_rgb(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 85);
    }

    #[test]
    fn check_rgba10_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 4;
        let mut image = vec![0u16; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        image[3] = 78;
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 10);
        scaler
            .resize_rgba_u16(&src_store, &mut target_store, false)
            .unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 60);
    }

    #[test]
    fn check_rgb10_resizing_vertical() {
        let image_width = 8;
        let image_height = 4;
        const CN: usize = 3;
        let mut image = vec![0; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 10);
        scaler
            .resize_rgb_u16(&src_store, &mut target_store)
            .unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 85);
    }

    #[test]
    fn check_rgb10_resizing_vertical_adaptive() {
        let image_width = 8;
        let image_height = 4;
        const CN: usize = 3;
        let mut image = vec![0; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Adaptive);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 10);
        scaler
            .resize_rgb_u16(&src_store, &mut target_store)
            .unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 85);
    }

    #[test]
    fn check_rgb16_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 3;
        let mut image = vec![164; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 16);
        scaler
            .resize_rgb_u16(&src_store, &mut target_store)
            .unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 100);
    }

    #[test]
    fn check_rgba16_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 4;
        let mut image = vec![0u16; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 255;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 16);
        scaler
            .resize_rgba_u16(&src_store, &mut target_store, false)
            .unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 180);
    }

    #[test]
    fn check_rgba16_resizing_vertical_threading() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 4;
        let mut image = vec![0u16; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 255;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Adaptive);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 16);
        scaler
            .resize_rgba_u16(&src_store, &mut target_store, false)
            .unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 180);
    }

    #[test]
    fn check_rgba8_nearest_vertical() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Nearest);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        scaler
            .resize_rgba(&src_store, &mut target_store, false)
            .unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 80);
    }

    #[test]
    fn check_rgba8_nearest_vertical_threading() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Nearest);
        scaler.set_threading_policy(ThreadingPolicy::Adaptive);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        scaler
            .resize_rgba(&src_store, &mut target_store, false)
            .unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 80);
    }
}
