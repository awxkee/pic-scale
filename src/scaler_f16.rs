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

use crate::image_store::ImageStoreMut;
use crate::pic_scale_error::PicScaleError;
use crate::scaler::ScalingOptions;
use crate::{
    CbCrF16ImageStore, ImageStore, ImageStoreScaling, PlanarF16ImageStore, RgbF16ImageStore,
    RgbaF16ImageStore, Scaler, Scaling, ThreadingPolicy,
};
use core::f16;

/// Implements `f16` type support
#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl Scaler {
    /// Performs rescaling for RGBA f16
    ///
    /// Scales RGBA high bit-depth interleaved image in `f16` type.
    /// Channel order does not matter.
    /// To handle alpha pre-multiplication alpha channel expected to be at last position.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    /// `premultiply_alpha` - flag if it should handle alpha or not
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<f16, 4>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_rgba_f16(&src_store, &mut dst_store, false).unwrap();
    /// ```
    pub fn resize_rgba_f16<'a>(
        &'a self,
        store: &ImageStore<'a, f16, 4>,
        into: &mut ImageStoreMut<'a, f16, 4>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError> {
        self.generic_resize_with_alpha(store, into, premultiply_alpha)
    }

    /// Performs rescaling for RGB f16
    ///
    /// Scales RGB high bit-depth interleaved image in `f16` type.
    /// Channel order does not matter.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<f16, 3>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_rgb_f16(&src_store, &mut dst_store).unwrap();
    /// ```
    pub fn resize_rgb_f16<'a>(
        &'a self,
        store: &ImageStore<'a, f16, 3>,
        into: &mut ImageStoreMut<'a, f16, 3>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    /// Performs rescaling for CbCr f16
    ///
    /// Scales CbCr high bit-depth interleaved image in `f16` type, optionally it could handle LumaAlpha images also
    /// Channel order does not matter.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<f16, 2>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_cbcr_f16(&src_store, &mut dst_store).unwrap();
    /// ```
    pub fn resize_cbcr_f16<'a>(
        &'a self,
        store: &ImageStore<'a, f16, 2>,
        into: &mut ImageStoreMut<'a, f16, 2>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    /// Performs rescaling for Planar image f16
    ///
    /// Scales planar high bit-depth image in `f16` type, optionally it could handle LumaAlpha images also
    /// Channel order does not matter.
    ///
    /// # Arguments
    /// `store` - original image store
    /// `into` - target image store
    ///
    /// # Example
    ///
    /// #[no_build]
    /// ```rust
    ///  use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler};
    ///  let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
    ///  let src_store = ImageStore::alloc(100, 100);
    ///  let mut dst_store = ImageStoreMut::<f16, 1>::alloc_with_depth(50, 50, 10);
    ///  scaler.resize_plane_f16(&src_store, &mut dst_store).unwrap();
    /// ```
    ///
    pub fn resize_plane_f16<'a>(
        &'a self,
        store: &ImageStore<'a, f16, 1>,
        into: &mut ImageStoreMut<'a, f16, 1>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl<'b> ImageStoreScaling<'b, f16, 1> for PlanarF16ImageStore<'b> {
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, f16, 1>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError> {
        let mut scaler = Scaler::new(options.resampling_function);
        scaler.set_threading_policy(if options.use_multithreading {
            ThreadingPolicy::Adaptive
        } else {
            ThreadingPolicy::Single
        });
        scaler.generic_resize(self, store)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl<'b> ImageStoreScaling<'b, f16, 2> for CbCrF16ImageStore<'b> {
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, f16, 2>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError> {
        let mut scaler = Scaler::new(options.resampling_function);
        scaler.set_threading_policy(if options.use_multithreading {
            ThreadingPolicy::Adaptive
        } else {
            ThreadingPolicy::Single
        });
        scaler.generic_resize(self, store)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl<'b> ImageStoreScaling<'b, f16, 3> for RgbF16ImageStore<'b> {
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, f16, 3>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError> {
        let mut scaler = Scaler::new(options.resampling_function);
        scaler.set_threading_policy(if options.use_multithreading {
            ThreadingPolicy::Adaptive
        } else {
            ThreadingPolicy::Single
        });
        scaler.generic_resize(self, store)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl<'b> ImageStoreScaling<'b, f16, 4> for RgbaF16ImageStore<'b> {
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, f16, 4>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError> {
        let mut scaler = Scaler::new(options.resampling_function);
        scaler.set_threading_policy(if options.use_multithreading {
            ThreadingPolicy::Adaptive
        } else {
            ThreadingPolicy::Single
        });
        scaler.generic_resize_with_alpha(self, store, options.premultiply_alpha)
    }
}
