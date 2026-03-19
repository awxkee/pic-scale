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
use crate::image_store::ImageStoreMut;
use crate::plan::Resampling;
use crate::scaler::ScalingOptions;
use crate::validation::PicScaleError;
use crate::{
    CbCrF16ImageStore, ImageSize, ImageStoreScaling, PlanarF16ImageStore, RgbF16ImageStore,
    RgbaF16ImageStore, Scaler,
};
use core::f16;
use std::sync::Arc;

/// Implements `f16` type support
#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl Scaler {
    pub fn plan_planar_resampling_f16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<f16, 1>>, PicScaleError> {
        self.plan_generic_resize::<f16, f32, 1>(source_size, target_size, 8)
    }

    pub fn plan_cbcr_resampling_f16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<f16, 2>>, PicScaleError> {
        self.plan_generic_resize::<f16, f32, 2>(source_size, target_size, 8)
    }

    pub fn plan_rgb_resampling_f16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<f16, 3>>, PicScaleError> {
        self.plan_generic_resize::<f16, f32, 3>(source_size, target_size, 8)
    }

    pub fn plan_rgba_resampling_f16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<f16, 4>>, PicScaleError> {
        self.plan_generic_resize_with_alpha::<f16, f32, 4>(
            source_size,
            target_size,
            8,
            premultiply_alpha,
        )
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
        scaler.set_threading_policy(options.threading_policy);
        let plan = scaler.plan_generic_resize(self.get_size(), store.size(), store.bit_depth)?;
        plan.resample(self, store)
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
        scaler.set_threading_policy(options.threading_policy);
        let plan = scaler.plan_generic_resize(self.get_size(), store.size(), store.bit_depth)?;
        plan.resample(self, store)
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
        scaler.set_threading_policy(options.threading_policy);
        let plan = scaler.plan_generic_resize(self.get_size(), store.size(), store.bit_depth)?;
        plan.resample(self, store)
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
        scaler.set_threading_policy(options.threading_policy);
        let plan = scaler.plan_generic_resize_with_alpha(
            self.get_size(),
            store.size(),
            store.bit_depth,
            options.premultiply_alpha,
        )?;
        plan.resample(self, store)
    }
}
