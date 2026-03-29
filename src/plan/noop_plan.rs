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
use crate::image_store::CheckStoreDensity;
use crate::validation::validate_sizes;
use crate::{ImageSize, ImageStore, ImageStoreMut, PicScaleError, ResamplingPlan};
use std::fmt::Debug;
use std::marker::PhantomData;

pub(crate) struct NoopPlan<T: Send + Sync, const N: usize> {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default, const N: usize> ResamplingPlan<T, N>
    for NoopPlan<T, N>
where
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity,
{
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError> {
        validate_sizes!(store, into, self.source_size, self.target_size);
        // validate size has copying
        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }
        Ok(())
    }

    fn resample_with_scratch(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        _scratch: &mut [T],
    ) -> Result<(), PicScaleError> {
        self.resample(store, into)
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![]
    }

    fn scratch_size(&self) -> usize {
        0
    }

    fn target_size(&self) -> ImageSize {
        self.target_size
    }

    fn source_size(&self) -> ImageSize {
        self.source_size
    }
}
