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
use crate::plan::Resampling;
use crate::support::check_image_size_overflow;
use crate::validation::try_vec;
use crate::{ImageSize, ImageStore, ImageStoreMut, PicScaleError, ResamplingPlan};
use std::fmt::Debug;
use std::sync::Arc;

pub(crate) trait Splitter<T, R, const N: usize>
where
    [T]: ToOwned<Owned = Vec<T>>,
    [R]: ToOwned<Owned = Vec<R>>,
{
    fn split(&self, from: &ImageStore<'_, T, N>, into: &mut ImageStoreMut<'_, R, N>);
    fn merge(&self, from: &ImageStore<'_, R, N>, into: &mut ImageStoreMut<'_, T, N>);
    fn bit_depth(&self) -> usize;
}

pub(crate) struct SplitPlanInterceptor<T, R, const N: usize> {
    pub(crate) intercept: Arc<Resampling<R, N>>,
    pub(crate) splitter: Arc<dyn Splitter<T, R, N> + Send + Sync>,
    pub(crate) inner_scratch: usize,
}

impl<T: Default + Clone + Copy + Debug, R: Default + Clone + Copy + Debug, const N: usize>
    ResamplingPlan<T, N> for SplitPlanInterceptor<T, R, N>
{
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError> {
        let mut scratch = self.alloc_scratch();
        self.resample_with_scratch(store, into, &mut scratch)
    }
    fn resample_with_scratch(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        _: &mut [T],
    ) -> Result<(), PicScaleError> {
        let new_size = into.size();
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

        if store.width == new_size.width && store.height == new_size.height {
            store.copied_to_mut(into);
            return Ok(());
        }

        let mut total_scratch =
            try_vec![R::default(); store.width * store.height * N + self.inner_scratch];
        let (scratch_target, scratch2) = total_scratch.split_at_mut(store.width * store.height * N);

        let mut intermediate_store =
            ImageStoreMut::<R, N>::from_slice(scratch_target, store.width, store.height)?;
        intermediate_store.bit_depth = self.splitter.bit_depth();

        self.splitter.split(store, &mut intermediate_store);

        let new_immutable_store = ImageStore::<R, N> {
            buffer: std::borrow::Cow::Borrowed(scratch_target),
            channels: N,
            width: store.width,
            height: store.height,
            stride: store.width * N,
            bit_depth: self.splitter.bit_depth(),
        };

        let mut scaled_im_store = ImageStoreMut::<R, N>::try_alloc_with_depth(
            into.width,
            into.height,
            self.splitter.bit_depth(),
        )?;

        self.intercept.resample_with_scratch(
            &new_immutable_store,
            &mut scaled_im_store,
            scratch2,
        )?;

        self.splitter.merge(&scaled_im_store.to_immutable(), into);
        Ok(())
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![]
    }

    fn scratch_size(&self) -> usize {
        self.inner_scratch
    }

    fn get_target_size(&self) -> ImageSize {
        self.intercept.get_target_size()
    }

    fn get_source_size(&self) -> ImageSize {
        self.intercept.get_source_size()
    }
}
