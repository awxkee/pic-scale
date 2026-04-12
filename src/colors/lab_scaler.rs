/*
 * Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::colors::common_splitter::{SplitPlanInterceptor, Splitter};
use crate::plan::Resampling;
use crate::validation::PicScaleError;
use crate::{ImageSize, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy};
use colorutils_rs::{
    SRGB_TO_XYZ_D65, TransferFunction, XYZ_TO_SRGB_D65, lab_to_srgb, lab_with_alpha_to_rgba,
    rgb_to_lab, rgba_to_lab_with_alpha,
};
use std::sync::Arc;

#[derive(Debug, Copy, Clone)]
/// Converts image to *CIE LAB* components scales it and convert back
pub struct LabScaler {
    pub(crate) scaler: Scaler,
}

struct LabRgbSplitter {}

impl Splitter<u8, f32, 3> for LabRgbSplitter {
    fn split(
        &self,
        from: &ImageStore<'_, u8, 3>,
        into: &mut ImageStoreMut<'_, f32, 3>,
    ) -> Result<(), PicScaleError> {
        let mut dst_buffer = into.to_colorutils_buffer_mut();

        rgb_to_lab(
            &from.to_colorutils_buffer(),
            &mut dst_buffer,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .map_err(|x| PicScaleError::Generic(x.to_string()))
    }

    fn merge(
        &self,
        from: &ImageStore<'_, f32, 3>,
        into: &mut ImageStoreMut<'_, u8, 3>,
    ) -> Result<(), PicScaleError> {
        let mut dst_buffer = into.to_colorutils_buffer_mut();

        lab_to_srgb(&from.to_colorutils_buffer(), &mut dst_buffer)
            .map_err(|x| PicScaleError::Generic(x.to_string()))
    }
    fn bit_depth(&self) -> usize {
        8
    }
}

struct LabRgbaSplitter {}

impl Splitter<u8, f32, 4> for LabRgbaSplitter {
    fn split(
        &self,
        from: &ImageStore<'_, u8, 4>,
        into: &mut ImageStoreMut<'_, f32, 4>,
    ) -> Result<(), PicScaleError> {
        let mut dst_buffer = into.to_colorutils_buffer_mut();
        rgba_to_lab_with_alpha(
            &from.to_colorutils_buffer(),
            &mut dst_buffer,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        )
        .map_err(|x| PicScaleError::Generic(x.to_string()))
    }

    fn merge(
        &self,
        from: &ImageStore<'_, f32, 4>,
        into: &mut ImageStoreMut<'_, u8, 4>,
    ) -> Result<(), PicScaleError> {
        let mut dst_buffer = into.to_colorutils_buffer_mut();
        lab_with_alpha_to_rgba(
            &from.to_colorutils_buffer(),
            &mut dst_buffer,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        )
        .map_err(|x| PicScaleError::Generic(x.to_string()))
    }

    fn bit_depth(&self) -> usize {
        8
    }
}

impl LabScaler {
    pub fn new(filter: ResamplingFunction) -> Self {
        LabScaler {
            scaler: Scaler::new(filter),
        }
    }
}

impl LabScaler {
    pub fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) -> LabScaler {
        self.scaler.threading_policy = threading_policy;
        *self
    }

    pub fn plan_rgb_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 3>>, PicScaleError> {
        let intercept = self
            .scaler
            .plan_rgb_resampling_f32(source_size, target_size)?;
        let scratch_size = intercept.scratch_size();
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(LabRgbSplitter {}),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_rgba_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<u8, 4>>, PicScaleError> {
        let intercept =
            self.scaler
                .plan_rgba_resampling_f32(source_size, target_size, premultiply_alpha)?;
        let scratch_size = intercept.scratch_size();
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(LabRgbaSplitter {}),
            inner_scratch: scratch_size,
        }))
    }
}
