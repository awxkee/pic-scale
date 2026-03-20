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
use crate::{ImageSize, ImageStore, ImageStoreMut, PicScaleError};

/// A precomputed resampling plan for scaling images of pixel type `T` with `N` channels.
///
/// A plan is created once via methods like [`plan_rgba_resampling`] or [`plan_rgb_resampling`]
/// on a scaler, and can then be executed repeatedly against different image buffers of the
/// same dimensions without recomputing filter weights.
pub trait ResamplingPlan<T: Copy, const N: usize> {
    /// Resamples `store` into `into`, allocating any necessary scratch memory internally.
    ///
    /// This is the simplest way to execute a plan. If you are resampling many images in a
    /// tight loop and want to avoid repeated allocations, prefer [`resample_with_scratch`]
    /// with a buffer obtained from [`alloc_scratch`].
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError>;
    /// Resamples `store` into `into` using the caller-supplied `scratch` buffer.
    ///
    /// Avoids internal allocation on every call, which is useful when resampling many
    /// images of the same size. The scratch buffer must be at least [`scratch_size`] elements
    /// long; obtain a correctly sized buffer with [`alloc_scratch`].
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_rgb_resampling(source_size, target_size)?;
    /// let mut scratch = plan.alloc_scratch();
    ///
    /// for frame in frames {
    ///     plan.resample_with_scratch(&frame, &mut output, &mut scratch)?;
    /// }
    fn resample_with_scratch(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        scratch: &mut [T],
    ) -> Result<(), PicScaleError>;
    /// Allocates a scratch buffer of the correct size for use with [`resample_with_scratch`].
    ///
    /// The returned `Vec` is zero initialized and exactly [`scratch_size`] elements long.
    /// Reuse it across calls to avoid repeated allocation.
    fn alloc_scratch(&self) -> Vec<T>;
    /// Returns the number of `T` elements required in the scratch buffer.
    ///
    /// Pass a slice of at least this length to [`resample_with_scratch`].
    fn scratch_size(&self) -> usize;
    /// Returns the target (output) image dimensions this plan was built for.
    fn target_size(&self) -> ImageSize;
    /// Returns the source (input) image dimensions this plan was built for.
    fn source_size(&self) -> ImageSize;
}

/// Type alias for a thread-safe, dynamically dispatched [`ResamplingPlan`].
///
/// Returned by scaler planning methods (e.g. `plan_rgba_resampling`) and intended
/// to be stored as `Arc<Resampling<T, N>>` so the plan can be shared across threads
/// or held for the lifetime of a processing pipeline.
///
/// # Example
///
/// ```rust,no_run,ignore
/// let plan: Arc<Resampling<u8, 4>> =
///     scaler.plan_rgba_resampling(source_size, target_size, true)?;
/// ```
pub type Resampling<T, const N: usize> = dyn ResamplingPlan<T, N> + Send + Sync;
