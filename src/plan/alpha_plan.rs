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

use crate::convolution::Filtering;
use crate::image_store::{AssociateAlpha, CheckStoreDensity, UnassociateAlpha};
use crate::validation::{validate_scratch, validate_sizes};
use crate::{
    ImageSize, ImageStore, ImageStoreMut, PicScaleError, ResamplingPlan, ThreadingPolicy,
    WorkloadStrategy,
};
use std::fmt::Debug;
use std::sync::Arc;

pub(crate) struct AlphaConvolvePlan<T: Send + Sync, const N: usize> {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) threading_policy: ThreadingPolicy,
    pub(crate) horizontal_filter: Arc<dyn Filtering<T, N> + Send + Sync>,
    pub(crate) vertical_filter: Arc<dyn Filtering<T, N> + Send + Sync>,
    pub(crate) should_do_horizontal: bool,
    pub(crate) should_do_vertical: bool,
    pub(crate) workload_strategy: WorkloadStrategy,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default + 'static, const N: usize> ResamplingPlan<T, N>
    for AlphaConvolvePlan<T, N>
where
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
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
        scratch: &mut [T],
    ) -> Result<(), PicScaleError> {
        validate_sizes!(store, into, self.source_size, self.target_size);
        let scratch = validate_scratch!(scratch, self.scratch_size());

        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }

        let (alpha_scratch, rem) =
            scratch.split_at_mut(self.source_size.width * self.source_size.height * N);

        let pool = self.threading_policy.get_nova_pool(ImageSize::new(
            self.target_size.width,
            self.target_size.height,
        ));

        let mut src_store: std::borrow::Cow<'_, ImageStore<'_, T, N>> =
            std::borrow::Cow::Borrowed(store);

        let mut has_alpha_premultiplied = false;

        let is_alpha_premultiplication_reasonable = store.is_alpha_premultiplication_needed();
        if is_alpha_premultiplication_reasonable {
            let mut new_store = ImageStoreMut::<T, N>::from_slice(
                alpha_scratch,
                self.source_size.width,
                self.source_size.height,
            )?;
            new_store.bit_depth = into.bit_depth;
            src_store.premultiply_alpha(&mut new_store, &pool);
            src_store = std::borrow::Cow::Owned(ImageStore::<T, N> {
                buffer: std::borrow::Cow::Borrowed(alpha_scratch),
                channels: N,
                width: src_store.width,
                height: src_store.height,
                stride: src_store.width * N,
                bit_depth: into.bit_depth,
            });
            has_alpha_premultiplied = true;
        }

        if self.should_do_vertical && self.should_do_horizontal {
            let (scratch, _) =
                rem.split_at_mut(self.source_size.width * self.target_size.height * N);
            let mut new_image_vertical = ImageStoreMut::<T, N>::from_slice(
                scratch,
                src_store.width,
                self.target_size.height,
            )?;
            new_image_vertical.bit_depth = into.bit_depth;
            self.vertical_filter
                .filter(src_store.as_ref(), &mut new_image_vertical);
            let new_immutable_store = new_image_vertical.to_immutable();
            self.horizontal_filter.filter(&new_immutable_store, into);
        } else if self.should_do_vertical {
            self.vertical_filter.filter(src_store.as_ref(), into);
        } else {
            assert!(self.should_do_horizontal, "This should not happen.");
            self.horizontal_filter.filter(src_store.as_ref(), into);
        }

        if has_alpha_premultiplied {
            into.unpremultiply_alpha(&pool, self.workload_strategy);
        }

        Ok(())
    }

    fn alloc_scratch(&self) -> Vec<T> {
        if self.should_do_horizontal && self.should_do_vertical {
            vec![T::default(); self.scratch_size()]
        } else {
            vec![]
        }
    }

    fn scratch_size(&self) -> usize {
        let basic_size = if self.should_do_horizontal && self.should_do_vertical {
            self.source_size.width * self.target_size.height * N
        } else {
            0
        };
        // we don't know if we will need scratch for alpha yet, so always requesting it
        let alpha_scratch = self.source_size.width * self.source_size.height * N;
        basic_size + alpha_scratch
    }

    fn get_target_size(&self) -> ImageSize {
        self.target_size
    }

    fn get_source_size(&self) -> ImageSize {
        self.source_size
    }
}
