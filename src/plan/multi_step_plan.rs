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
use crate::validation::{try_vec, validate_scratch, validate_sizes};
use crate::{ImageSize, ImageStore, ImageStoreMut, PicScaleError, Resampling, ResamplingPlan};
use std::fmt::Debug;
use std::sync::Arc;

/// A resampling plan that chains multiple sub-plans to upscale in small
/// steps, giving Lanczos2 / Mitchell the dense source data they need to
/// produce smooth results even at very large magnification ratios.
pub(crate) struct MultiStepResamplePlan<T, const N: usize> {
    /// Ordered chain: [step_0_plan, step_1_plan, ..., final_plan].
    /// Each plan reads the output of the previous one.
    pub(crate) steps: Vec<Arc<Resampling<T, N>>>,
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    /// Precomputed at construction: max `scratch_size()` across all steps.
    kernel_scratch_size: usize,
    /// Precomputed at construction: size of each ping-pong pixel buffer.
    ping_pong_size: usize,
}

impl<T, const N: usize> MultiStepResamplePlan<T, N>
where
    T: Clone + Copy + Debug + Send + Sync + Default + 'static,
{
    pub(crate) fn new(
        steps: Vec<Arc<Resampling<T, N>>>,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Self {
        let kernel_scratch_size = steps.iter().map(|s| s.scratch_size()).max().unwrap_or(0);

        // Ping-pong buffers cover all steps except the last, which writes
        // directly into `destination`.
        let step_count = steps.len();
        let ping_pong_size = if step_count <= 1 {
            0
        } else {
            steps[..step_count - 1]
                .iter()
                .map(|s| {
                    let sz = s.target_size();
                    sz.width * sz.height * N
                })
                .max()
                .unwrap_or(0)
        };

        Self {
            steps,
            source_size,
            target_size,
            kernel_scratch_size,
            ping_pong_size,
        }
    }
}

impl<T, const N: usize> ResamplingPlan<T, N> for MultiStepResamplePlan<T, N>
where
    T: Clone + Copy + Debug + Send + Sync + Default + 'static,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity,
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn source_size(&self) -> ImageSize {
        self.source_size
    }

    fn target_size(&self) -> ImageSize {
        self.target_size
    }

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
        source: &ImageStore<'_, T, N>,
        destination: &mut ImageStoreMut<'_, T, N>,
        scratch: &mut [T],
    ) -> Result<(), PicScaleError> {
        validate_sizes!(source, destination, self.source_size, self.target_size);
        let scratch = validate_scratch!(scratch, self.scratch_size());
        if destination.should_have_bit_depth() && !(1..=16).contains(&destination.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(destination.bit_depth));
        }
        let step_count = self.steps.len();

        if step_count == 0 {
            return Err(PicScaleError::EmptyPlan);
        }

        // scratch layout:
        //   [ inner_scratch: kernel_scratch_size | A: ping_pong_size | B: ping_pong_size ]
        let (inner_scratch, pixel_scratch) = scratch.split_at_mut(self.kernel_scratch_size);

        if step_count == 1 {
            return self.steps[0].resample_with_scratch(source, destination, inner_scratch);
        }

        let first_target = self.steps[0].target_size();
        let first_len = first_target.width * first_target.height * N;
        {
            let (a, _) = pixel_scratch.split_at_mut(self.ping_pong_size);
            let mut dst_store = ImageStoreMut::from_slice(
                &mut a[..first_len],
                first_target.width,
                first_target.height,
            )?;
            dst_store.bit_depth = destination.bit_depth;
            self.steps[0].resample_with_scratch(source, &mut dst_store, inner_scratch)?;
        }

        let mut flipped = false;
        let mut current_size = first_target;

        for step_idx in 1..step_count - 1 {
            let step = &self.steps[step_idx];
            let next_size = step.target_size();
            let current_len = current_size.width * current_size.height * N;
            let next_len = next_size.width * next_size.height * N;

            let (a, b_tail) = pixel_scratch.split_at_mut(self.ping_pong_size);
            let b = &mut b_tail[..self.ping_pong_size];

            let (current_slice, next_slice): (&[T], &mut [T]) = if !flipped {
                (&a[..current_len], &mut b[..next_len])
            } else {
                (&b[..current_len], &mut a[..next_len])
            };

            let mut src_store =
                ImageStore::from_slice(current_slice, current_size.width, current_size.height)?;
            src_store.bit_depth = destination.bit_depth;
            let mut dst_store =
                ImageStoreMut::from_slice(next_slice, next_size.width, next_size.height)?;
            dst_store.bit_depth = destination.bit_depth;
            step.resample_with_scratch(&src_store, &mut dst_store, inner_scratch)?;

            current_size = next_size;
            flipped = !flipped;
        }

        let current_len = current_size.width * current_size.height * N;
        let (a, b_tail) = pixel_scratch.split_at_mut(self.ping_pong_size);
        let b = &b_tail[..self.ping_pong_size];
        let current_slice: &[T] = if !flipped {
            &a[..current_len]
        } else {
            &b[..current_len]
        };

        let src_store =
            ImageStore::from_slice(current_slice, current_size.width, current_size.height)?;
        self.steps[step_count - 1].resample_with_scratch(&src_store, destination, inner_scratch)?;

        Ok(())
    }

    fn scratch_size(&self) -> usize {
        self.kernel_scratch_size + 2 * self.ping_pong_size
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![T::default(); self.scratch_size()]
    }
}
