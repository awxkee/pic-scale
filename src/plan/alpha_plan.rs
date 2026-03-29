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
use crate::image_store::{AssociateAlpha, CheckStoreDensity, UnassociateAlpha};
use crate::validation::{try_vec, validate_scratch, validate_sizes};
use crate::{
    ImageSize, ImageStore, ImageStoreMut, PicScaleError, Resampling, ResamplingPlan,
    ThreadingPolicy, WorkloadStrategy,
};
use std::fmt::Debug;
use std::sync::Arc;

fn maybe_premultiply_alpha<'a, T, const N: usize>(
    store: &'a ImageStore<'a, T, N>,
    alpha_scratch: &'a mut [T],
    needs_alpha_forward: bool,
    needs_alpha_backward: bool,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) -> Result<(std::borrow::Cow<'a, ImageStore<'a, T, N>>, bool), PicScaleError>
where
    T: Copy + Send + Sync + Clone + Debug + Default + 'static,
    for<'b> ImageStore<'b, T, N>: AssociateAlpha<T, N>,
    for<'b> ImageStoreMut<'b, T, N>: CheckStoreDensity,
{
    if needs_alpha_forward && (!needs_alpha_backward || store.is_alpha_premultiplication_needed()) {
        let mut new_store =
            ImageStoreMut::<T, N>::from_slice(alpha_scratch, store.width, store.height)?;
        new_store.bit_depth = bit_depth;
        store.premultiply_alpha(&mut new_store, pool);
        let owned = ImageStore::<T, N> {
            buffer: std::borrow::Cow::Borrowed(alpha_scratch),
            channels: N,
            width: store.width,
            height: store.height,
            stride: store.width * N,
            bit_depth,
        };
        Ok((std::borrow::Cow::Owned(owned), true))
    } else {
        Ok((std::borrow::Cow::Borrowed(store), false))
    }
}

pub(crate) struct AlphaVerticalConvolvePlan<T: Send + Sync, const N: usize> {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) threading_policy: ThreadingPolicy,
    pub(crate) vertical_filter: Arc<dyn ColumnFilter<T, N> + Send + Sync>,
    pub(crate) workload_strategy: WorkloadStrategy,
    pub(crate) needs_alpha_forward: bool,
    pub(crate) needs_alpha_backward: bool,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default + 'static, const N: usize> ResamplingPlan<T, N>
    for AlphaVerticalConvolvePlan<T, N>
where
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
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
        let pool = self.threading_policy.get_nova_pool(self.target_size);
        let (src_store, has_alpha_premultiplied) = maybe_premultiply_alpha(
            store,
            scratch,
            self.needs_alpha_forward,
            self.needs_alpha_backward,
            into.bit_depth,
            &pool,
        )?;
        self.vertical_filter.filter(src_store.as_ref(), into);
        if has_alpha_premultiplied || (self.needs_alpha_backward && !self.needs_alpha_forward) {
            into.unpremultiply_alpha(&pool, self.workload_strategy);
        }
        Ok(())
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![T::default(); self.scratch_size()]
    }

    fn scratch_size(&self) -> usize {
        // alpha scratch only — no intermediate pixel buffer needed
        self.source_size.width * self.source_size.height * N
    }

    fn target_size(&self) -> ImageSize {
        self.target_size
    }
    fn source_size(&self) -> ImageSize {
        self.source_size
    }
}

pub(crate) struct AlphaHorizontalConvolvePlan<T: Send + Sync, const N: usize> {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) threading_policy: ThreadingPolicy,
    pub(crate) horizontal_filter: Arc<dyn RowFilter<T, N> + Send + Sync>,
    pub(crate) workload_strategy: WorkloadStrategy,
    pub(crate) needs_alpha_forward: bool,
    pub(crate) needs_alpha_backward: bool,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default + 'static, const N: usize> ResamplingPlan<T, N>
    for AlphaHorizontalConvolvePlan<T, N>
where
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
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
        let pool = self.threading_policy.get_nova_pool(self.target_size);
        let (src_store, has_alpha_premultiplied) = maybe_premultiply_alpha(
            store,
            scratch,
            self.needs_alpha_forward,
            self.needs_alpha_backward,
            into.bit_depth,
            &pool,
        )?;
        self.horizontal_filter.filter(src_store.as_ref(), into);
        if has_alpha_premultiplied || (self.needs_alpha_backward && !self.needs_alpha_forward) {
            into.unpremultiply_alpha(&pool, self.workload_strategy);
        }
        Ok(())
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![T::default(); self.scratch_size()]
    }

    fn scratch_size(&self) -> usize {
        self.source_size.width * self.source_size.height * N
    }

    fn target_size(&self) -> ImageSize {
        self.target_size
    }
    fn source_size(&self) -> ImageSize {
        self.source_size
    }
}

pub(crate) struct AlphaBothAxesConvolvePlan<T: Send + Sync, const N: usize> {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) threading_policy: ThreadingPolicy,
    pub(crate) horizontal_filter: Arc<dyn RowFilter<T, N> + Send + Sync>,
    pub(crate) vertical_filter: Arc<dyn ColumnFilter<T, N> + Send + Sync>,
    pub(crate) trampoline_filter: Arc<dyn TrampolineFilter<T, N> + Send + Sync>,
    pub(crate) workload_strategy: WorkloadStrategy,
    pub(crate) needs_alpha_forward: bool,
    pub(crate) needs_alpha_backward: bool,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default + 'static, const N: usize> ResamplingPlan<T, N>
    for AlphaBothAxesConvolvePlan<T, N>
where
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
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

        let alpha_size = self.source_size.width * self.source_size.height * N;
        let (alpha_scratch, filter_scratch) = scratch.split_at_mut(alpha_size);

        let pool = self.threading_policy.get_nova_pool(self.target_size);
        let (src_store, has_alpha_premultiplied) = maybe_premultiply_alpha(
            store,
            alpha_scratch,
            self.needs_alpha_forward,
            self.needs_alpha_backward,
            into.bit_depth,
            &pool,
        )?;

        if self.threading_policy == ThreadingPolicy::Single {
            self.trampoline_filter
                .filter(src_store.as_ref(), into, filter_scratch);
        } else {
            let intermediate_len = self.source_size.width * self.target_size.height * N;
            let (intermediate_scratch, _) = filter_scratch.split_at_mut(intermediate_len);
            let mut new_image_vertical = ImageStoreMut::<T, N>::from_slice(
                intermediate_scratch,
                src_store.width,
                self.target_size.height,
            )?;
            new_image_vertical.bit_depth = into.bit_depth;
            self.vertical_filter
                .filter(src_store.as_ref(), &mut new_image_vertical);
            let new_immutable_store = new_image_vertical.to_immutable();
            self.horizontal_filter.filter(&new_immutable_store, into);
        }

        if has_alpha_premultiplied || (self.needs_alpha_backward && !self.needs_alpha_forward) {
            into.unpremultiply_alpha(&pool, self.workload_strategy);
        }
        Ok(())
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![T::default(); self.scratch_size()]
    }

    fn scratch_size(&self) -> usize {
        let alpha_scratch = self.source_size.width * self.source_size.height * N;
        let filter_scratch = if self.threading_policy == ThreadingPolicy::Single {
            self.trampoline_filter.scratch_size()
        } else {
            self.source_size.width * self.target_size.height * N
        };
        alpha_scratch + filter_scratch
    }

    fn target_size(&self) -> ImageSize {
        self.target_size
    }
    fn source_size(&self) -> ImageSize {
        self.source_size
    }
}

pub(crate) fn make_alpha_plan<T, const N: usize>(
    source_size: ImageSize,
    destination_size: ImageSize,
    horizontal_plan: Option<Arc<dyn RowFilter<T, N> + Send + Sync>>,
    vertical_plan: Option<Arc<dyn ColumnFilter<T, N> + Send + Sync>>,
    trampoline_filter: Option<Arc<dyn TrampolineFilter<T, N> + Send + Sync>>,
    threading_policy: ThreadingPolicy,
    workload_strategy: WorkloadStrategy,
    needs_alpha_forward: bool,
    needs_alpha_backward: bool,
) -> Arc<Resampling<T, N>>
where
    T: Copy + Send + Sync + Clone + Debug + Default + 'static,
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
{
    let should_do_vertical = source_size.height != destination_size.height;
    let should_do_horizontal = source_size.width != destination_size.width;

    match (should_do_vertical, should_do_horizontal) {
        (true, true) => Arc::new(AlphaBothAxesConvolvePlan {
            source_size,
            target_size: destination_size,
            threading_policy,
            horizontal_filter: horizontal_plan.expect("Horizontal plan is expected"),
            vertical_filter: vertical_plan.expect("Vertical plan is expected"),
            trampoline_filter: trampoline_filter.expect("Trampoline filter plan is expected"),
            workload_strategy,
            needs_alpha_forward,
            needs_alpha_backward,
        }),
        (true, false) => Arc::new(AlphaVerticalConvolvePlan {
            source_size,
            target_size: destination_size,
            threading_policy,
            vertical_filter: vertical_plan.expect("Vertical plan is expected"),
            workload_strategy,
            needs_alpha_forward,
            needs_alpha_backward,
        }),
        (false, true) => Arc::new(AlphaHorizontalConvolvePlan {
            source_size,
            target_size: destination_size,
            threading_policy,
            horizontal_filter: horizontal_plan.expect("Horizontal plan is expected"),
            workload_strategy,
            needs_alpha_forward,
            needs_alpha_backward,
        }),
        (false, false) => Arc::new(AlphaVerticalConvolvePlan {
            source_size,
            target_size: destination_size,
            threading_policy,
            vertical_filter: vertical_plan.expect("Vertical plan is expected"),
            workload_strategy,
            needs_alpha_forward,
            needs_alpha_backward,
        }),
    }
}
