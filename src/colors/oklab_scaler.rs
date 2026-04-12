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
use colorutils_rs::{TransferFunction, oklab_to_rgb, oklab_to_rgba, rgb_to_oklab, rgba_to_oklab};
use std::sync::Arc;

use crate::colors::common_splitter::{SplitPlanInterceptor, Splitter};
use crate::plan::Resampling;
use crate::validation::PicScaleError;
use crate::{ImageSize, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy};

#[derive(Debug, Copy, Clone)]
/// Converts image to *Oklab* components scales it and convert back
pub struct OklabScaler {
    pub(crate) scaler: Scaler,
    pub(crate) transfer_function: TransferFunction,
}

impl OklabScaler {
    /// # Arguments
    /// - `transfer_function` - Transfer function to move into linear colorspace and back
    pub fn new(filter: ResamplingFunction, transfer_function: TransferFunction) -> Self {
        OklabScaler {
            scaler: Scaler::new(filter),
            transfer_function,
        }
    }
}

struct OklabRgbSplitter {
    transfer_function: TransferFunction,
}

impl Splitter<u8, f32, 3> for OklabRgbSplitter {
    fn split(
        &self,
        from: &ImageStore<'_, u8, 3>,
        into: &mut ImageStoreMut<'_, f32, 3>,
    ) -> Result<(), PicScaleError> {
        let mut dst_buffer = into.to_colorutils_buffer_mut();

        rgb_to_oklab(
            &from.to_colorutils_buffer(),
            &mut dst_buffer,
            self.transfer_function,
        )
        .map_err(|x| PicScaleError::Generic(x.to_string()))
    }

    fn merge(
        &self,
        from: &ImageStore<'_, f32, 3>,
        into: &mut ImageStoreMut<'_, u8, 3>,
    ) -> Result<(), PicScaleError> {
        let mut dst_buffer = into.to_colorutils_buffer_mut();
        oklab_to_rgb(
            &from.to_colorutils_buffer(),
            &mut dst_buffer,
            self.transfer_function,
        )
        .map_err(|x| PicScaleError::Generic(x.to_string()))
    }

    fn bit_depth(&self) -> usize {
        8
    }
}

struct OklabRgbaSplitter {
    transfer_function: TransferFunction,
}

impl Splitter<u8, f32, 4> for OklabRgbaSplitter {
    fn split(
        &self,
        from: &ImageStore<'_, u8, 4>,
        into: &mut ImageStoreMut<'_, f32, 4>,
    ) -> Result<(), PicScaleError> {
        let mut dst_buffer = into.to_colorutils_buffer_mut();

        rgba_to_oklab(
            &from.to_colorutils_buffer(),
            &mut dst_buffer,
            self.transfer_function,
        )
        .map_err(|x| PicScaleError::Generic(x.to_string()))
    }

    fn merge(
        &self,
        from: &ImageStore<'_, f32, 4>,
        into: &mut ImageStoreMut<'_, u8, 4>,
    ) -> Result<(), PicScaleError> {
        let mut dst_buffer = into.to_colorutils_buffer_mut();
        oklab_to_rgba(
            &from.to_colorutils_buffer(),
            &mut dst_buffer,
            self.transfer_function,
        )
        .map_err(|x| PicScaleError::Generic(x.to_string()))
    }

    fn bit_depth(&self) -> usize {
        8
    }
}

impl OklabScaler {
    pub fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) -> Self {
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
            splitter: Arc::new(OklabRgbSplitter {
                transfer_function: self.transfer_function,
            }),
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
            splitter: Arc::new(OklabRgbaSplitter {
                transfer_function: self.transfer_function,
            }),
            inner_scratch: scratch_size,
        }))
    }
}
