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
use crate::convolution::{ColumnFilter, RowFilter, TrampolineFilter};
use crate::image_store::CheckStoreDensity;
use crate::validation::{try_vec, validate_scratch, validate_sizes};
use crate::{ImageSize, ImageStore, ImageStoreMut, PicScaleError, ResamplingPlan, ThreadingPolicy};
use std::fmt::Debug;
use std::sync::Arc;

pub(crate) struct VerticalConvolvePlan<T: Send + Sync, const N: usize> {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) vertical_filter: Arc<dyn ColumnFilter<T, N> + Send + Sync>,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default, const N: usize> ResamplingPlan<T, N>
    for VerticalConvolvePlan<T, N>
where
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity,
{
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError> {
        validate_sizes!(store, into, self.source_size, self.target_size);
        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }
        self.vertical_filter.filter(store, into);
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

pub(crate) struct HorizontalConvolvePlan<T: Send + Sync, const N: usize> {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) horizontal_filter: Arc<dyn RowFilter<T, N> + Send + Sync>,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default, const N: usize> ResamplingPlan<T, N>
    for HorizontalConvolvePlan<T, N>
where
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity,
{
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError> {
        validate_sizes!(store, into, self.source_size, self.target_size);
        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }
        self.horizontal_filter.filter(store, into);
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

pub(crate) struct BothAxesConvolvePlan<T: Send + Sync, const N: usize> {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) horizontal_filter: Arc<dyn RowFilter<T, N> + Send + Sync>,
    pub(crate) vertical_filter: Arc<dyn ColumnFilter<T, N> + Send + Sync>,
    pub(crate) trampoline_filter: Arc<dyn TrampolineFilter<T, N> + Send + Sync>,
    pub(crate) threading_policy: ThreadingPolicy,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default, const N: usize> ResamplingPlan<T, N>
    for BothAxesConvolvePlan<T, N>
where
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity,
{
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.resample_with_scratch(store, into, &mut scratch)
    }

    fn resample_with_scratch(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        scratch: &mut [T],
    ) -> Result<(), PicScaleError> {
        validate_sizes!(store, into, self.source_size, self.target_size);
        let scratch = validate_scratch!(scratch, self.scratch_size());
        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }
        if self.threading_policy == ThreadingPolicy::Single {
            self.trampoline_filter.filter(store, into, scratch);
        } else {
            let mut new_image_vertical =
                ImageStoreMut::<T, N>::from_slice(scratch, store.width, self.target_size.height)?;
            new_image_vertical.bit_depth = into.bit_depth;
            self.vertical_filter.filter(store, &mut new_image_vertical);
            let new_immutable_store = new_image_vertical.to_immutable();
            self.horizontal_filter.filter(&new_immutable_store, into);
        }
        Ok(())
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![T::default(); self.scratch_size()]
    }

    fn scratch_size(&self) -> usize {
        if self.threading_policy == ThreadingPolicy::Single {
            self.trampoline_filter.scratch_size()
        } else {
            self.source_size.width * self.target_size.height * N
        }
    }

    fn target_size(&self) -> ImageSize {
        self.target_size
    }

    fn source_size(&self) -> ImageSize {
        self.source_size
    }
}
