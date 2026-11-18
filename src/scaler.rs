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
use crate::convolution::{ConvolutionOptions, HorizontalFilterPass, VerticalConvolutionPass};
use crate::factory::{Ar30ByteOrder, Rgb30};
use crate::image_size::ImageSize;
use crate::image_store::{CheckStoreDensity, ImageStore, ImageStoreMut};
use crate::math::WeightsGenerator;
use crate::plan::{
    AlphaPlanner, Ar30Destructuring, Ar30DestructuringImpl, Ar30Plan, BothAxesConvolvePlan,
    DefaultPlanner, HorizontalConvolvePlan, NoopPlan, ResampleNearestPlan, Resampling,
    TrampolineFiltering, VerticalConvolvePlan,
};
use crate::threading_policy::ThreadingPolicy;
use crate::validation::PicScaleError;
use crate::{
    CbCr8ImageStore, CbCr16ImageStore, CbCrF32ImageStore, Planar8ImageStore, Planar16ImageStore,
    PlanarF32ImageStore, ResamplingFunction, ResamplingPlan, Rgb8ImageStore, Rgb16ImageStore,
    RgbF32ImageStore, Rgba8ImageStore, Rgba16ImageStore, RgbaF32ImageStore,
};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug, Copy, Clone)]
/// Represents base scaling structure
pub struct Scaler {
    pub(crate) function: ResamplingFunction,
    pub(crate) threading_policy: ThreadingPolicy,
    pub workload_strategy: WorkloadStrategy,
    pub(crate) multi_step_upscaling: bool,
    pub(crate) supersampling: bool,
}

/// Defines execution hint about preferred strategy
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
pub enum WorkloadStrategy {
    /// Prefers quality to speed
    PreferQuality,
    /// Prefers speed to quality
    #[default]
    PreferSpeed,
}

impl Scaler {
    /// Creates new [Scaler] instance with corresponding filter
    ///
    /// Creates default [crate::Scaler] with corresponding filter and default [ThreadingPolicy::Single]
    ///
    pub fn new(filter: ResamplingFunction) -> Self {
        Scaler {
            function: filter,
            threading_policy: ThreadingPolicy::Single,
            workload_strategy: WorkloadStrategy::default(),
            multi_step_upscaling: false,
            supersampling: false,
        }
    }

    /// Sets preferred workload strategy
    ///
    /// This is hint only, it may change something, or may be not.
    pub fn set_workload_strategy(&mut self, workload_strategy: WorkloadStrategy) -> Self {
        self.workload_strategy = workload_strategy;
        *self
    }

    /// Enables multistep upscaling for large magnification ratios.
    ///
    /// When upscaling by a large factor (e.g. 10× or more), a single-pass filter
    /// does not have enough source samples to interpolate smoothly — the kernel
    /// spans so few real pixels that the result looks blocky or rings heavily.
    /// Multistep upscaling breaks the operation into a chain of smaller steps,
    /// each within the filter's optimal range, so every pass has enough source
    /// data to produce a smooth result.
    ///
    /// The number of steps and the intermediate sizes are chosen automatically
    /// based on the resampling function's support width.
    ///
    /// This has no effect on downscaling or on [`ResamplingFunction::Nearest`],
    /// which are always single-pass. For modest upscale ratios already within
    /// the filter's safe range the plan degenerates to a single step with no
    /// overhead.
    pub fn set_multi_step_upsampling(&mut self, value: bool) -> Self {
        self.multi_step_upscaling = value;
        *self
    }

    /// Enables a cheap pre-filter pass before large downscales to improve
    /// quality and performance.
    ///
    /// When downscaling by a large ratio (≥ 3×) the quality filter must
    /// average a very large number of source pixels per output pixel, which
    /// is slow and can produce aliasing. With supersampling enabled, a fast
    /// pre-filter first reduces the image to approximately twice the target
    /// size, then the quality filter performs a final clean 2× reduction.
    /// This keeps the quality filter in its optimal range while the cheap
    /// pre-filter handles the heavy lifting.
    ///
    /// The pre-filter is chosen automatically based on the downscale ratio:
    /// - **≥ 4×**: [`ResamplingFunction::Nearest`] — fastest, no blending.
    /// - **3–4×**: [`ResamplingFunction::Box`] (area average) — slightly
    ///   higher quality than nearest for non-integer ratios.
    ///
    /// Has no effect on upscaling or on [`ResamplingFunction::Nearest`],
    /// which is always single-pass.
    pub fn set_supersampling(&mut self, value: bool) -> Self {
        self.supersampling = value;
        *self
    }
}

impl Scaler {
    pub(crate) fn build_single_step_plan<
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        W,
        const N: usize,
    >(
        &self,
        source_size: ImageSize,
        destination_size: ImageSize,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity,
    {
        if self.function == ResamplingFunction::Nearest {
            return Ok(Arc::new(ResampleNearestPlan {
                source_size,
                target_size: destination_size,
                _threading_policy: self.threading_policy,
                _phantom_data: PhantomData,
            }));
        }

        let should_do_horizontal = source_size.width != destination_size.width;
        let should_do_vertical = source_size.height != destination_size.height;

        let options = ConvolutionOptions {
            workload_strategy: self.workload_strategy,
            bit_depth,
            src_size: source_size,
            dst_size: destination_size,
        };

        let vertical_plan = if should_do_vertical {
            let vertical_filters =
                T::make_weights(self.function, source_size.height, destination_size.height)?;
            Some(ImageStore::<T, N>::vertical_plan(
                vertical_filters,
                self.threading_policy,
                options,
            ))
        } else {
            None
        };

        let horizontal_plan = if should_do_horizontal {
            let horizontal_filters =
                T::make_weights(self.function, source_size.width, destination_size.width)?;
            Some(ImageStore::<T, N>::horizontal_plan(
                horizontal_filters,
                self.threading_policy,
                options,
            ))
        } else {
            None
        };

        match (should_do_vertical, should_do_horizontal) {
            (true, true) => {
                let v = vertical_plan.expect("Should have a vertical filter");
                let h = horizontal_plan.expect("Should have a horizontal filter");
                let trampoline = Arc::new(TrampolineFiltering {
                    horizontal_filter: h.clone(),
                    vertical_filter: v.clone(),
                    source_size,
                    target_size: destination_size,
                });
                Ok(Arc::new(BothAxesConvolvePlan {
                    source_size,
                    target_size: destination_size,
                    horizontal_filter: h,
                    vertical_filter: v,
                    trampoline_filter: trampoline,
                    streaming: self.threading_policy.is_single_worker(destination_size),
                }))
            }
            (true, false) => Ok(Arc::new(VerticalConvolvePlan {
                source_size,
                target_size: destination_size,
                vertical_filter: vertical_plan.expect("Should have a vertical filter"),
            })),
            (false, true) => Ok(Arc::new(HorizontalConvolvePlan {
                source_size,
                target_size: destination_size,
                horizontal_filter: horizontal_plan.expect("Should have a horizontal filter"),
            })),
            (false, false) => Ok(Arc::new(NoopPlan {
                source_size,
                target_size: destination_size,
                _phantom: PhantomData,
            })),
        }
    }

    /// Creates a resampling plan for a single-channel (planar/grayscale) `u8` image.
    ///
    /// The returned [`Arc<Resampling<u8, 1>>`] can be executed repeatedly against images
    /// of `source_size` to produce output of `target_size` without recomputing filter weights.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_planar_resampling(source_size, target_size)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_planar_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 1>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize(self, source_size, target_size, 8)
    }

    /// Creates a resampling plan for a two-channel grayscale + alpha (`GA`) `u8` image.
    ///
    /// When `premultiply_alpha` is `true` the alpha channel is pre-multiplied into the gray
    /// channel before resampling and un-multiplied afterward.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `premultiply_alpha` — Whether to premultiply alpha before resampling.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// // Resample with alpha-aware filtering to avoid dark fringing
    /// let plan = scaler.plan_gray_alpha_resampling(source_size, target_size, true)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_gray_alpha_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<u8, 2>>, PicScaleError> {
        if premultiply_alpha {
            AlphaPlanner::plan_generic_resize_with_alpha(
                self,
                source_size,
                target_size,
                8,
                premultiply_alpha,
            )
        } else {
            DefaultPlanner::plan_generic_resize(self, source_size, target_size, 8)
        }
    }

    /// Creates a resampling plan for a two-channel chroma (`CbCr`) `u8` image.
    ///
    /// Intended for the chroma planes of YCbCr images (e.g. the `Cb`/`Cr` planes in
    /// 4:2:0 or 4:2:2 video), where both channels are treated as independent signals
    /// with no alpha relationship. For the luma plane use [`plan_planar_resampling`].
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input chroma plane.
    /// - `target_size` — Desired dimensions of the output chroma plane.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_cbcr_resampling(source_size, target_size)?;
    /// plan.resample(&cbcr_store, &mut target_cbcr_store)?;
    /// ```
    pub fn plan_cbcr_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 2>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize(self, source_size, target_size, 8)
    }

    /// Creates a resampling plan for a three-channel RGB `u8` image.
    ///
    /// The returned [`Arc<Resampling<u8, 3>>`] encodes all filter weights for scaling
    /// from `source_size` to `target_size` and can be reused across many frames without
    /// recomputation.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_rgb_resampling(source_size, target_size)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_rgb_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 3>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize(self, source_size, target_size, 8)
    }

    /// Creates a resampling plan for a four-channel RGBA `u8` image.
    ///
    /// When `premultiply_alpha` is `true` the RGB channels are pre-multiplied by alpha
    /// before resampling and un-multiplied afterward.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `premultiply_alpha` — Whether to premultiply alpha before resampling.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// // Resample a sprite sheet with correct alpha blending
    /// let plan = scaler.plan_rgba_resampling(source_size, target_size, true)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_rgba_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<u8, 4>>, PicScaleError> {
        if premultiply_alpha {
            AlphaPlanner::plan_generic_resize_with_alpha(
                self,
                source_size,
                target_size,
                8,
                premultiply_alpha,
            )
        } else {
            DefaultPlanner::plan_generic_resize(self, source_size, target_size, 8)
        }
    }

    /// Creates a resampling plan for a single-channel (planar/grayscale) `u16` image.
    ///
    /// The 16-bit variant of [`plan_planar_resampling`], suitable for high-bit-depth
    /// grayscale content such as HDR images or luma planes from 10/12-bit video.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `bit_depth` — Effective bit depth of the pixel data (e.g. `10`, `12`, or `16`).
    ///   Must not exceed `16`.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_planar_resampling16(source_size, target_size, 12)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_planar_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 1>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize(self, source_size, target_size, bit_depth)
    }

    /// Creates a resampling plan for a single-channel (planar/grayscale) `i16` image.
    ///
    /// The 16-bit variant of [`plan_planar_resampling`], suitable for high-bit-depth
    /// grayscale content such as HDR images or luma planes from 10/12-bit video.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `bit_depth` — Effective bit depth of the pixel data (e.g. `10`, `12`, or `16`).
    ///   Must not exceed `16`.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_planar_resampling_s16(source_size, target_size, 12)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_planar_resampling_s16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<i16, 1>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize(self, source_size, target_size, bit_depth)
    }

    /// Creates a resampling plan for a two-channel chroma (`CbCr`) `u16` image.
    ///
    /// The 16-bit variant of [`plan_cbcr_resampling`], intended for high-bit-depth chroma
    /// planes of YCbCr content (e.g. 10-bit 4:2:0 or 4:2:2 video). Both channels are
    /// treated as independent signals with no alpha relationship.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input chroma plane.
    /// - `target_size` — Desired dimensions of the output chroma plane.
    /// - `bit_depth` — Effective bit depth of the pixel data (e.g. `10`, `12`, or `16`).
    ///   Must not exceed `16`.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_cbcr_resampling16(source_size, target_size, 10)?;
    /// plan.resample(&cbcr_store, &mut target_cbcr_store)?;
    /// ```
    pub fn plan_cbcr_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 2>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize(self, source_size, target_size, bit_depth)
    }

    /// Creates a resampling plan for a two-channel grayscale + alpha (`GA`) `u16` image.
    ///
    /// The 16-bit variant of [`plan_gray_alpha_resampling`]. When `premultiply_alpha` is
    /// `true` the gray channel is pre-multiplied by alpha before resampling and
    /// un-multiplied afterward.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `premultiply_alpha` — Whether to premultiply alpha before resampling.
    /// - `bit_depth` — Effective bit depth of the pixel data (e.g. `10`, `12`, or `16`).
    ///   Must not exceed `16`.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_gray_alpha_resampling16(source_size, target_size, true, 16)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_gray_alpha_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 2>>, PicScaleError> {
        if premultiply_alpha {
            AlphaPlanner::plan_generic_resize_with_alpha(
                self,
                source_size,
                target_size,
                bit_depth,
                premultiply_alpha,
            )
        } else {
            self.plan_cbcr_resampling16(source_size, target_size, bit_depth)
        }
    }

    /// Creates a resampling plan for a three-channel RGB `u16` image.
    ///
    /// The 16-bit variant of [`plan_rgb_resampling`], suitable for high-bit-depth color
    /// images such as 10/12-bit HDR or wide-gamut content. All three channels are
    /// resampled independently with no alpha relationship.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `bit_depth` — Effective bit depth of the pixel data (e.g. `10`, `12`, or `16`).
    ///   Must not exceed `16`.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_rgb_resampling16(source_size, target_size, 12)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_rgb_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 3>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize(self, source_size, target_size, bit_depth)
    }

    /// Creates a resampling plan for a four-channel RGBA `u16` image.
    ///
    /// The 16-bit variant of [`plan_rgba_resampling`]. When `premultiply_alpha` is `true`
    /// the RGB channels are pre-multiplied by alpha before resampling and un-multiplied
    /// afterward.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `premultiply_alpha` — Whether to premultiply alpha before resampling.
    /// - `bit_depth` — Effective bit depth of the pixel data (e.g. `10`, `12`, or `16`).
    ///   Must not exceed `16`.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_rgba_resampling16(source_size, target_size, true, 10)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_rgba_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 4>>, PicScaleError> {
        if premultiply_alpha {
            AlphaPlanner::plan_generic_resize_with_alpha(
                self,
                source_size,
                target_size,
                bit_depth,
                premultiply_alpha,
            )
        } else {
            DefaultPlanner::plan_generic_resize(self, source_size, target_size, bit_depth)
        }
    }

    /// Creates a resampling plan for a single-channel (planar/grayscale) `f32` image.
    ///
    /// The `f32` variant of [`plan_planar_resampling`], suitable for HDR or linear-light
    /// grayscale content where full floating-point precision is required.
    ///
    /// The internal accumulator precision is selected automatically based on the scaler's
    /// [`WorkloadStrategy`]:
    /// - [`PreferQuality`](WorkloadStrategy::PreferQuality) — accumulates in `f64` for
    ///   maximum numerical accuracy.
    /// - [`PreferSpeed`](WorkloadStrategy::PreferSpeed) — accumulates in `f32` for
    ///   faster throughput at a small precision cost.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_planar_resampling_f32(source_size, target_size)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_planar_resampling_f32(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<f32, 1>>, PicScaleError> {
        match self.workload_strategy {
            WorkloadStrategy::PreferQuality => DefaultPlanner::plan_generic_resize::<f32, f64, 1>(
                self,
                source_size,
                target_size,
                8,
            ),
            WorkloadStrategy::PreferSpeed => DefaultPlanner::plan_generic_resize::<f32, f32, 1>(
                self,
                source_size,
                target_size,
                8,
            ),
        }
    }

    /// Creates a resampling plan for a two-channel chroma (`CbCr`) `f32` image.
    ///
    /// The `f32` variant of [`plan_cbcr_resampling`], intended for floating-point chroma
    /// planes of YCbCr content. Both channels are treated as independent signals with no
    /// alpha relationship.
    ///
    /// The internal accumulator precision is selected automatically based on the scaler's
    /// [`WorkloadStrategy`]:
    /// - [`PreferQuality`](WorkloadStrategy::PreferQuality) — accumulates in `f64` for
    ///   maximum numerical accuracy.
    /// - [`PreferSpeed`](WorkloadStrategy::PreferSpeed) — accumulates in `f32` for
    ///   faster throughput at a small precision cost.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input chroma plane.
    /// - `target_size` — Desired dimensions of the output chroma plane.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_cbcr_resampling_f32(source_size, target_size)?;
    /// plan.resample(&cbcr_store, &mut target_cbcr_store)?;
    /// ```
    pub fn plan_cbcr_resampling_f32(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<dyn ResamplingPlan<f32, 2> + Send + Sync>, PicScaleError> {
        match self.workload_strategy {
            WorkloadStrategy::PreferQuality => DefaultPlanner::plan_generic_resize::<f32, f64, 2>(
                self,
                source_size,
                target_size,
                8,
            ),
            WorkloadStrategy::PreferSpeed => DefaultPlanner::plan_generic_resize::<f32, f32, 2>(
                self,
                source_size,
                target_size,
                8,
            ),
        }
    }

    /// Creates a resampling plan for a two-channel grayscale + alpha (`GA`) `f32` image.
    ///
    /// The `f32` variant of [`plan_gray_alpha_resampling`]. When `premultiply_alpha` is
    /// `true` the gray channel is pre-multiplied by alpha before resampling and
    /// un-multiplied afterward, preventing dark fringing around transparent edges.
    /// Set it to `false` if the image uses straight alpha or the channels should be
    /// filtered independently.
    ///
    /// The internal accumulator precision is selected automatically based on the scaler's
    /// [`WorkloadStrategy`]:
    /// - [`PreferQuality`](WorkloadStrategy::PreferQuality) — accumulates in `f64` for
    ///   maximum numerical accuracy.
    /// - [`PreferSpeed`](WorkloadStrategy::PreferSpeed) — accumulates in `f32` for
    ///   faster throughput at a small precision cost.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `premultiply_alpha` — Whether to premultiply alpha before resampling.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_gray_alpha_resampling_f32(source_size, target_size, true)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_gray_alpha_resampling_f32(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<f32, 2>>, PicScaleError> {
        if premultiply_alpha {
            match self.workload_strategy {
                WorkloadStrategy::PreferQuality => {
                    AlphaPlanner::plan_generic_resize_with_alpha::<f32, f64, 2>(
                        self,
                        source_size,
                        target_size,
                        8,
                        premultiply_alpha,
                    )
                }
                WorkloadStrategy::PreferSpeed => {
                    AlphaPlanner::plan_generic_resize_with_alpha::<f32, f32, 2>(
                        self,
                        source_size,
                        target_size,
                        8,
                        premultiply_alpha,
                    )
                }
            }
        } else {
            self.plan_cbcr_resampling_f32(source_size, target_size)
        }
    }

    /// Creates a resampling plan for a three-channel RGB `f32` image.
    ///
    /// The `f32` variant of [`plan_rgb_resampling`], suitable for HDR or linear-light
    /// color images where full floating-point precision is required. All three channels
    /// are resampled independently with no alpha relationship.
    ///
    /// The internal accumulator precision is selected automatically based on the scaler's
    /// [`WorkloadStrategy`]:
    /// - [`PreferQuality`](WorkloadStrategy::PreferQuality) — accumulates in `f64` for
    ///   maximum numerical accuracy.
    /// - [`PreferSpeed`](WorkloadStrategy::PreferSpeed) — accumulates in `f32` for
    ///   faster throughput at a small precision cost.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_rgb_resampling_f32(source_size, target_size)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_rgb_resampling_f32(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<f32, 3>>, PicScaleError> {
        match self.workload_strategy {
            WorkloadStrategy::PreferQuality => DefaultPlanner::plan_generic_resize::<f32, f64, 3>(
                self,
                source_size,
                target_size,
                8,
            ),
            WorkloadStrategy::PreferSpeed => DefaultPlanner::plan_generic_resize::<f32, f32, 3>(
                self,
                source_size,
                target_size,
                8,
            ),
        }
    }

    /// Creates a resampling plan for a four-channel RGBA `f32` image.
    ///
    /// The `f32` variant of [`plan_rgba_resampling`]. When `premultiply_alpha` is `true`
    /// the RGB channels are pre-multiplied by alpha before resampling and un-multiplied
    /// afterward, preventing dark halos around semi-transparent edges. Set it to `false`
    /// if the image uses straight alpha or the channels should be filtered independently.
    ///
    /// The internal accumulator precision is selected automatically based on the scaler's
    /// [`WorkloadStrategy`]:
    /// - [`PreferQuality`](WorkloadStrategy::PreferQuality) — accumulates in `f64` for
    ///   maximum numerical accuracy.
    /// - [`PreferSpeed`](WorkloadStrategy::PreferSpeed) — accumulates in `f32` for
    ///   faster throughput at a small precision cost.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `premultiply_alpha` — Whether to premultiply alpha before resampling.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_rgba_resampling_f32(source_size, target_size, true)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_rgba_resampling_f32(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<f32, 4>>, PicScaleError> {
        if premultiply_alpha {
            match self.workload_strategy {
                WorkloadStrategy::PreferQuality => {
                    AlphaPlanner::plan_generic_resize_with_alpha::<f32, f64, 4>(
                        self,
                        source_size,
                        target_size,
                        8,
                        premultiply_alpha,
                    )
                }
                WorkloadStrategy::PreferSpeed => {
                    AlphaPlanner::plan_generic_resize_with_alpha::<f32, f32, 4>(
                        self,
                        source_size,
                        target_size,
                        8,
                        premultiply_alpha,
                    )
                }
            }
        } else {
            match self.workload_strategy {
                WorkloadStrategy::PreferQuality => {
                    DefaultPlanner::plan_generic_resize::<f32, f64, 4>(
                        self,
                        source_size,
                        target_size,
                        8,
                    )
                }
                WorkloadStrategy::PreferSpeed => {
                    DefaultPlanner::plan_generic_resize::<f32, f32, 4>(
                        self,
                        source_size,
                        target_size,
                        8,
                    )
                }
            }
        }
    }

    pub fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) -> Self {
        self.threading_policy = threading_policy;
        *self
    }
}

impl Scaler {
    pub(crate) fn plan_resize_ar30<const AR30_ORDER: usize>(
        &self,
        ar30_type: Rgb30,
        source_size: ImageSize,
        destination_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 4>>, PicScaleError> {
        if self.function == ResamplingFunction::Nearest {
            return Ok(Arc::new(ResampleNearestPlan {
                source_size,
                target_size: destination_size,
                _threading_policy: self.threading_policy,
                _phantom_data: PhantomData,
            }));
        }
        let inner_plan = self.plan_rgb_resampling16(source_size, destination_size, 10)?;
        let mut _decomposer: Arc<dyn Ar30Destructuring + Send + Sync> =
            Arc::new(Ar30DestructuringImpl::<AR30_ORDER> { rgb30: ar30_type });
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::{
                    avx_column_handler_fixed_point_ar30, avx_convolve_horizontal_rgba_rows_4_ar30,
                    avx_convolve_horizontal_rgba_rows_ar30,
                };
                use crate::plan::{HorizontalFiltering, VerticalFiltering};
                let should_do_horizontal = source_size.width != destination_size.width;
                let should_do_vertical = source_size.height != destination_size.height;

                let vertical_plan = if should_do_vertical {
                    let vertical_filters = u8::make_weights(
                        self.function,
                        source_size.height,
                        destination_size.height,
                    )?;
                    Some(Arc::new(VerticalFiltering {
                        filter_row: match ar30_type {
                            Rgb30::Ar30 => {
                                avx_column_handler_fixed_point_ar30::<
                                    { Rgb30::Ar30 as usize },
                                    AR30_ORDER,
                                >
                            }
                            Rgb30::Ra30 => {
                                avx_column_handler_fixed_point_ar30::<
                                    { Rgb30::Ra30 as usize },
                                    AR30_ORDER,
                                >
                            }
                        },
                        filter_weights: vertical_filters
                            .numerical_approximation_i16::<{ crate::support::PRECISION }>(0),
                        threading_policy: self.threading_policy,
                    }))
                } else {
                    None
                };

                let horizontal_plan = if should_do_horizontal {
                    let horizontal_filters =
                        u8::make_weights(self.function, source_size.width, destination_size.width)?;
                    Some(Arc::new(HorizontalFiltering {
                        filter_row: match ar30_type {
                            Rgb30::Ar30 => {
                                avx_convolve_horizontal_rgba_rows_ar30::<
                                    { Rgb30::Ar30 as usize },
                                    AR30_ORDER,
                                >
                            }
                            Rgb30::Ra30 => {
                                avx_convolve_horizontal_rgba_rows_ar30::<
                                    { Rgb30::Ra30 as usize },
                                    AR30_ORDER,
                                >
                            }
                        },
                        filter_4_rows: Some(match ar30_type {
                            Rgb30::Ar30 => {
                                avx_convolve_horizontal_rgba_rows_4_ar30::<
                                    { Rgb30::Ar30 as usize },
                                    AR30_ORDER,
                                >
                            }
                            Rgb30::Ra30 => {
                                avx_convolve_horizontal_rgba_rows_4_ar30::<
                                    { Rgb30::Ra30 as usize },
                                    AR30_ORDER,
                                >
                            }
                        }),
                        _threading_policy: self.threading_policy,
                        filter_weights: horizontal_filters
                            .numerical_approximation_i16::<{ crate::support::PRECISION }>(0),
                    }))
                } else {
                    None
                };

                return Ok(match (should_do_vertical, should_do_horizontal) {
                    (true, true) => {
                        let v = vertical_plan.expect("Should have vertical plan");
                        let h = horizontal_plan.expect("Should have horizontal plan");
                        let trampoline = Arc::new(TrampolineFiltering {
                            horizontal_filter: h.clone(),
                            vertical_filter: v.clone(),
                            source_size,
                            target_size: destination_size,
                        });
                        Arc::new(BothAxesConvolvePlan {
                            source_size,
                            target_size: destination_size,
                            horizontal_filter: h,
                            vertical_filter: v,
                            trampoline_filter: trampoline,
                            streaming: self.threading_policy.is_single_worker(destination_size),
                        })
                    }
                    (true, false) => Arc::new(VerticalConvolvePlan {
                        source_size,
                        target_size: destination_size,
                        vertical_filter: vertical_plan.expect("Should have vertical plan"),
                    }),
                    (false, true) => Arc::new(HorizontalConvolvePlan {
                        source_size,
                        target_size: destination_size,
                        horizontal_filter: horizontal_plan.expect("Should have horizontal plan"),
                    }),
                    (false, false) => Arc::new(NoopPlan {
                        source_size,
                        target_size: destination_size,
                        _phantom: PhantomData,
                    }),
                });
            }
        }
        Ok(Arc::new(Ar30Plan {
            source_size,
            target_size: destination_size,
            inner_filter: inner_plan,
            decomposer: _decomposer,
        }))
    }

    /// Creates a resampling plan for an AR30 (`RGBA2101010`) packed 10-bit image.
    ///
    /// AR30 stores each pixel as a 32-bit word with 10 bits per RGB channel and a
    /// 2-bit alpha.
    ///
    /// # Alpha is not preserved
    ///
    /// Only the three color channels are resampled. The source alpha bits are
    /// ignored and every output pixel is written fully opaque (`a = 3`). Do not use
    /// this plan for images that carry meaningful transparency — unpack them to
    /// [`Rgba16ImageStore`] and resample that instead.
    ///
    /// The `order` argument controls the byte layout of the packed word:
    /// - [`Ar30ByteOrder::Host`] — native endianness of the current platform.
    /// - [`Ar30ByteOrder::Network`] — big-endian (network) byte order.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `order` — Byte order of the packed AR30 words.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_ar30_resampling(source_size, target_size, Ar30ByteOrder::Host)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_ar30_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        order: Ar30ByteOrder,
    ) -> Result<Arc<Resampling<u8, 4>>, PicScaleError> {
        match order {
            Ar30ByteOrder::Host => self.plan_resize_ar30::<{ Ar30ByteOrder::Host as usize }>(
                Rgb30::Ar30,
                source_size,
                target_size,
            ),
            Ar30ByteOrder::Network => self.plan_resize_ar30::<{ Ar30ByteOrder::Network as usize }>(
                Rgb30::Ar30,
                source_size,
                target_size,
            ),
        }
    }

    /// Creates a resampling plan for an RA30 (`RGBA1010102`) packed 10-bit image.
    ///
    /// RA30 stores each pixel as a 32-bit word with 10 bits per RGB channel and a
    /// 2-bit alpha in the least-significant position.
    ///
    /// # Alpha is not preserved
    ///
    /// Only the three color channels are resampled. The source alpha bits are
    /// ignored and every output pixel is written fully opaque (`a = 3`). Do not use
    /// this plan for images that carry meaningful transparency — unpack them to
    /// [`Rgba16ImageStore`] and resample that instead.
    ///
    /// The `order` argument controls the byte layout of the packed word:
    /// - [`Ar30ByteOrder::Host`] — native endianness of the current platform.
    /// - [`Ar30ByteOrder::Network`] — big-endian (network) byte order.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `order` — Byte order of the packed RA30 words.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.resize_ra30(source_size, target_size, Ar30ByteOrder::Host)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_ra30_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        order: Ar30ByteOrder,
    ) -> Result<Arc<Resampling<u8, 4>>, PicScaleError> {
        match order {
            Ar30ByteOrder::Host => self.plan_resize_ar30::<{ Ar30ByteOrder::Host as usize }>(
                Rgb30::Ra30,
                source_size,
                target_size,
            ),
            Ar30ByteOrder::Network => self.plan_resize_ar30::<{ Ar30ByteOrder::Network as usize }>(
                Rgb30::Ra30,
                source_size,
                target_size,
            ),
        }
    }
}

/// Declares default scaling options
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Default)]
pub struct ScalingOptions {
    pub resampling_function: ResamplingFunction,
    pub premultiply_alpha: bool,
    pub threading_policy: ThreadingPolicy,
}

/// Generic trait for [ImageStore] to implement abstract scaling.
pub trait ImageStoreScaling<'b, T, const N: usize>
where
    T: Clone + Copy + Debug,
{
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, T, N>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError>;
}

macro_rules! def_image_scaling_alpha {
    ($clazz: ident, $fx_type: ident, $cn: expr) => {
        impl<'b> ImageStoreScaling<'b, $fx_type, $cn> for $clazz<'b> {
            fn scale(
                &self,
                store: &mut ImageStoreMut<'b, $fx_type, $cn>,
                options: ScalingOptions,
            ) -> Result<(), PicScaleError> {
                let scaler = Scaler::new(options.resampling_function)
                    .set_threading_policy(options.threading_policy);
                let plan = AlphaPlanner::plan_generic_resize_with_alpha::<$fx_type, f32, $cn>(
                    &scaler,
                    self.size(),
                    store.size(),
                    store.bit_depth,
                    options.premultiply_alpha,
                )?;
                plan.resample(self, store)
            }
        }
    };
}

macro_rules! def_image_scaling {
    ($clazz: ident, $fx_type: ident, $cn: expr) => {
        impl<'b> ImageStoreScaling<'b, $fx_type, $cn> for $clazz<'b> {
            fn scale(
                &self,
                store: &mut ImageStoreMut<'b, $fx_type, $cn>,
                options: ScalingOptions,
            ) -> Result<(), PicScaleError> {
                let scaler = Scaler::new(options.resampling_function)
                    .set_threading_policy(options.threading_policy);
                let plan = DefaultPlanner::plan_generic_resize::<$fx_type, f32, $cn>(
                    &scaler,
                    self.size(),
                    store.size(),
                    store.bit_depth,
                )?;
                plan.resample(self, store)
            }
        }
    };
}

def_image_scaling_alpha!(Rgba8ImageStore, u8, 4);
def_image_scaling!(Rgb8ImageStore, u8, 3);
def_image_scaling!(CbCr8ImageStore, u8, 2);
def_image_scaling!(Planar8ImageStore, u8, 1);
def_image_scaling!(Planar16ImageStore, u16, 1);
def_image_scaling!(CbCr16ImageStore, u16, 2);
def_image_scaling!(Rgb16ImageStore, u16, 3);
def_image_scaling_alpha!(Rgba16ImageStore, u16, 4);
def_image_scaling!(PlanarF32ImageStore, f32, 1);
def_image_scaling!(CbCrF32ImageStore, f32, 2);
def_image_scaling!(RgbF32ImageStore, f32, 3);
def_image_scaling_alpha!(RgbaF32ImageStore, f32, 4);

#[cfg(test)]
mod tests {
    use super::*;

    fn check_public_single_tap<T, const N: usize>(
        depths: &[usize],
        has_alpha: bool,
        make_plan: impl Fn(&Scaler, ImageSize, ImageSize, bool, usize) -> Arc<Resampling<T, N>>,
    ) where
        T: Copy + Default + Debug + PartialEq + From<u8> + Send + Sync + 'static,
        u16: num_traits::AsPrimitive<T>,
    {
        use num_traits::AsPrimitive;
        for &depth in depths {
            let max = ((1u32 << depth) - 1) as u16;
            for level in [0, 1, max / 2, max / 2 + 1, max - 1, max] {
                for (from, to) in [
                    ((1, 1), (7, 5)),
                    ((1, 5), (7, 5)),
                    ((5, 1), (5, 7)),
                    ((2, 4), (5, 4)),
                    ((4, 2), (4, 5)),
                ] {
                    let mixed = from.0 == 2 || from.1 == 2;
                    for filter in [
                        ResamplingFunction::Bilinear,
                        ResamplingFunction::Lanczos3,
                        ResamplingFunction::MitchellNetravalli,
                    ] {
                        if mixed && filter != ResamplingFunction::Bilinear {
                            continue;
                        }
                        for strategy in [
                            WorkloadStrategy::PreferSpeed,
                            WorkloadStrategy::PreferQuality,
                        ] {
                            let scaler = Scaler::new(filter).set_workload_strategy(strategy);
                            for alpha in
                                [false, true]
                                    .into_iter()
                                    .take(if has_alpha { 2 } else { 1 })
                            {
                                for padding in [0, 5] {
                                    let source_stride = from.0 * N + padding;
                                    let source_len = (from.1 - 1) * source_stride + from.0 * N;
                                    let value = |far_edge: bool, channel: usize| -> T {
                                        if alpha && channel == N - 1 {
                                            max.as_()
                                        } else {
                                            let sample = if far_edge { max - level } else { level };
                                            sample.saturating_sub(channel as u16).as_()
                                        }
                                    };
                                    let mut pixels = vec![T::from(0xa5); source_len];
                                    for y in 0..from.1 {
                                        for x in 0..from.0 {
                                            let far =
                                                (from.0 == 2 && x == 1) || (from.1 == 2 && y == 1);
                                            for c in 0..N {
                                                pixels[y * source_stride + x * N + c] =
                                                    value(far, c);
                                            }
                                        }
                                    }
                                    let mut source =
                                        ImageStore::<T, N>::from_slice(&pixels, from.0, from.1)
                                            .unwrap();
                                    source.stride = source_stride;
                                    source.bit_depth = depth;
                                    let target_stride = to.0 * N + padding;
                                    let target_len = (to.1 - 1) * target_stride + to.0 * N;
                                    let mut output = vec![T::from(0xa5); target_len + 2];
                                    let mut target = ImageStoreMut::<T, N>::from_slice(
                                        &mut output[1..target_len + 1],
                                        to.0,
                                        to.1,
                                    )
                                    .unwrap();
                                    target.stride = target_stride;
                                    target.bit_depth = depth;
                                    let plan = make_plan(
                                        &scaler,
                                        source.size(),
                                        target.size(),
                                        alpha,
                                        depth,
                                    );
                                    plan.resample_with_scratch(
                                        &source,
                                        &mut target,
                                        &mut plan.alloc_scratch(),
                                    )
                                    .unwrap();
                                    for y in 0..to.1 {
                                        for x in 0..to.0 {
                                            // Interior Bilinear outputs blend two samples; only the clamped edges are single-tap.
                                            if (from.0 == 2 && x != 0 && x != to.0 - 1)
                                                || (from.1 == 2 && y != 0 && y != to.1 - 1)
                                            {
                                                continue;
                                            }
                                            let far = (from.0 == 2 && x == to.0 - 1)
                                                || (from.1 == 2 && y == to.1 - 1);
                                            for c in 0..N {
                                                assert_eq!(
                                                    output[1 + y * target_stride + x * N + c],
                                                    value(far, c),
                                                    "depth={depth}, N={N}, {from:?}->{to:?}, {filter:?}, {strategy:?}, alpha={alpha}, level={level}, ({x},{y},{c})"
                                                );
                                            }
                                        }
                                        let end = 1 + y * target_stride + to.0 * N;
                                        let next =
                                            (1 + (y + 1) * target_stride).min(target_len + 1);
                                        assert!(
                                            output[end..next].iter().all(|&v| v == T::from(0xa5))
                                        );
                                    }
                                    assert_eq!(output[0], T::from(0xa5));
                                    assert_eq!(output[target_len + 1], T::from(0xa5));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn public_u8_single_tap_preserves_samples() {
        check_public_single_tap::<u8, 1>(&[8], false, |s, a, b, _, _| {
            s.plan_planar_resampling(a, b).unwrap()
        });
        check_public_single_tap::<u8, 2>(&[8], true, |s, a, b, alpha, _| {
            s.plan_gray_alpha_resampling(a, b, alpha).unwrap()
        });
        check_public_single_tap::<u8, 3>(&[8], false, |s, a, b, _, _| {
            s.plan_rgb_resampling(a, b).unwrap()
        });
        check_public_single_tap::<u8, 4>(&[8], true, |s, a, b, alpha, _| {
            s.plan_rgba_resampling(a, b, alpha).unwrap()
        });
    }

    #[test]
    fn public_u16_single_tap_preserves_samples() {
        check_public_single_tap::<u16, 1>(&[13, 14, 15, 16], false, |s, a, b, _, d| {
            s.plan_planar_resampling16(a, b, d).unwrap()
        });
        check_public_single_tap::<u16, 3>(&[13, 14, 15, 16], false, |s, a, b, _, d| {
            s.plan_rgb_resampling16(a, b, d).unwrap()
        });
        check_public_single_tap::<u16, 4>(&[13, 14, 15, 16], true, |s, a, b, alpha, d| {
            s.plan_rgba_resampling16(a, b, alpha, d).unwrap()
        });
    }

    #[test]
    fn single_worker_rgba8_two_axis_plans_stream() {
        let source_size = ImageSize::new(13, 11);
        let target_size = ImageSize::new(7, 9);
        let source_stride = source_size.width * 4 + 5;
        let source_len = (source_size.height - 1) * source_stride + source_size.width * 4;
        let pixels: Vec<u8> = (0..source_len).map(|i| (19 * i + i / 3) as u8).collect();
        let mut source =
            ImageStore::from_slice(&pixels, source_size.width, source_size.height).unwrap();
        source.stride = source_stride;

        let target_stride = target_size.width * 4 + 7;
        let target_len = (target_size.height - 1) * target_stride + target_size.width * 4;
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        for premultiply in [false, true] {
            let alpha_scratch = if premultiply {
                source_size.width * source_size.height * 4
            } else {
                0
            };
            let streaming_scratch = alpha_scratch + source_size.width * 4 * 4;
            let full_scratch = alpha_scratch + source_size.width * target_size.height * 4;

            let single = scaler
                .set_threading_policy(ThreadingPolicy::Single)
                .plan_rgba_resampling(source_size, target_size, premultiply)
                .unwrap();
            let mut expected = vec![0xa5; target_len];
            let mut destination =
                ImageStoreMut::from_slice(&mut expected, target_size.width, target_size.height)
                    .unwrap();
            destination.stride = target_stride;
            single.resample(&source, &mut destination).unwrap();

            for policy in [
                ThreadingPolicy::Fixed(0),
                ThreadingPolicy::Fixed(1),
                ThreadingPolicy::Adaptive,
                ThreadingPolicy::Fixed(2),
            ] {
                let plan = scaler
                    .set_threading_policy(policy)
                    .plan_rgba_resampling(source_size, target_size, premultiply)
                    .unwrap();
                let materializes_intermediate = policy == ThreadingPolicy::Fixed(2)
                    && cfg!(all(feature = "threading", not(target_arch = "wasm32")));
                let required = if materializes_intermediate {
                    full_scratch
                } else {
                    streaming_scratch
                };
                assert_eq!(plan.scratch_size(), required, "{policy:?}");
                assert_eq!(plan.alloc_scratch().len(), required, "{policy:?}");
                if materializes_intermediate {
                    continue;
                }

                let mut actual = vec![0xa5; target_len];
                let mut destination =
                    ImageStoreMut::from_slice(&mut actual, target_size.width, target_size.height)
                        .unwrap();
                destination.stride = target_stride;
                let mut scratch = vec![0; full_scratch];
                plan.resample_with_scratch(&source, &mut destination, &mut scratch)
                    .unwrap();
                assert_eq!(actual, expected, "{policy:?}, premultiply={premultiply}");
            }
        }
    }

    macro_rules! check_rgba8 {
        ($dst: expr, $image_width: expr, $max: expr) => {
            {
                for (y, row) in $dst.chunks_exact($image_width * 4).enumerate() {
                    for (i, dst) in row.as_chunks::<4>().0.iter().enumerate() {
                        let diff0 = (dst[0] as i32 - 124).abs();
                        let diff1 = (dst[1] as i32 - 41).abs();
                        let diff2 = (dst[2] as i32 - 99).abs();
                        let diff3 = (dst[3] as i32 - 77).abs();
                        assert!(
                            diff0 < $max,
                            "Diff for channel 0 is expected < {}, but it was {diff0}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff1 < $max,
                            "Diff for channel 1 is expected < {}, but it was {diff1}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff2 < $max,
                            "Diff for channel 2 is expected < {}, but it was {diff2}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff3 < $max,
                            "Diff for channel 3 is expected < {}, but it was {diff3}, at (y: {y}, x: {i})",
                            $max
                        );
                    }
                }
            }
        };
    }

    macro_rules! check_rgb16 {
        ($dst: expr, $image_width: expr, $max: expr) => {
            {
                for (y, row) in $dst.chunks_exact($image_width * 3).enumerate() {
                    for (i, dst) in row.chunks_exact(3).enumerate() {
                        let diff0 = (dst[0] as i32 - 124).abs();
                        let diff1 = (dst[1] as i32 - 41).abs();
                        let diff2 = (dst[2] as i32 - 99).abs();
                        assert!(
                            diff0 < $max,
                            "Diff for channel 0 is expected < {}, but it was {diff0}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff1 < $max,
                            "Diff for channel 1 is expected < {}, but it was {diff1}, at (y: {y}, x: {i})",
                            $max
                        );
                        assert!(
                            diff2 < $max,
                            "Diff for channel 2 is expected < {}, but it was {diff2}, at (y: {y}, x: {i})",
                            $max
                        );
                    }
                }
            }
        };
    }

    #[test]
    fn check_rgba8_resizing_vertical() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<4>().0.iter_mut() {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        let scaler =
            Scaler::new(ResamplingFunction::Bilinear).set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling(src_store.size(), target_store.size(), false)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();
        check_rgba8!(target_data, image_width, 34);
    }

    #[test]
    fn check_rgba8_resizing_both() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<4>().0.iter_mut() {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        image[3] = 78;
        let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width / 2, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling(src_store.size(), target_store.size(), false)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();
        check_rgba8!(target_data, image_width, 34);
    }

    #[test]
    fn check_rgba8_resizing_alpha() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<4>().0.iter_mut() {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        image[3] = 78;
        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width / 2, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling(src_store.size(), target_store.size(), true)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();
        check_rgba8!(target_data, image_width, 160);
    }

    #[test]
    fn check_rgb8_resizing_vertical() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 3;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgb_resampling(src_store.size(), target_store.size())
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 85);
    }

    #[test]
    fn check_rgb8_resizing_vertical_threading() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 3;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
        scaler.set_threading_policy(ThreadingPolicy::Adaptive);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgb_resampling(src_store.size(), target_store.size())
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 85);
    }

    #[test]
    fn check_rgba10_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 4;
        let mut image = vec![0u16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<4>().0.iter_mut() {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        image[3] = 78;
        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 10);
        let planned = scaler
            .plan_rgba_resampling16(src_store.size(), target_store.size(), true, 10)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 60);
    }

    #[test]
    fn check_rgb10_resizing_vertical() {
        let image_width = 8;
        let image_height = 4;
        const CN: usize = 3;
        let mut image = vec![0; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 10);
        let planned = scaler
            .plan_rgb_resampling16(src_store.size(), target_store.size(), 10)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 85);
    }

    #[test]
    fn check_rgb10_resizing_vertical_adaptive() {
        let image_width = 8;
        let image_height = 4;
        const CN: usize = 3;
        let mut image = vec![0; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Adaptive);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 10);
        let planned = scaler
            .plan_rgb_resampling16(src_store.size(), target_store.size(), 10)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 85);
    }

    #[test]
    fn check_rgb16_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 3;
        let mut image = vec![164; image_height * image_width * CN];
        for dst in image.chunks_exact_mut(3) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
        }
        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 16);
        let planned = scaler
            .plan_rgb_resampling16(src_store.size(), target_store.size(), 16)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgb16!(target_data, image_width, 100);
    }

    #[test]
    fn check_rgba16_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 4;
        let mut image = vec![0u16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<4>().0.iter_mut() {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 255;
        }
        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 16);
        let planned = scaler
            .plan_rgba_resampling16(src_store.size(), target_store.size(), false, 16)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 180);
    }

    #[test]
    fn check_rgba16_resizing_vertical_threading() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 4;
        let mut image = vec![0u16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<4>().0.iter_mut() {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 255;
        }
        let scaler = Scaler::new(ResamplingFunction::Lanczos3)
            .set_threading_policy(ThreadingPolicy::Adaptive);
        let mut src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        src_store.bit_depth = 10;
        let mut target_store = ImageStoreMut::alloc_with_depth(image_width, image_height / 2, 16);
        let planned = scaler
            .plan_rgba_resampling16(src_store.size(), target_store.size(), false, 16)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 180);
    }

    #[test]
    fn check_rgba8_nearest_vertical() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<4>().0.iter_mut() {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Nearest);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling(src_store.size(), target_store.size(), false)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 80);
    }

    #[test]
    fn check_rgba8_nearest_vertical_threading() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0u8; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<4>().0.iter_mut() {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        let scaler = Scaler::new(ResamplingFunction::Nearest)
            .set_threading_policy(ThreadingPolicy::Adaptive);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling(src_store.size(), target_store.size(), false)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 80);
    }

    #[test]
    fn check_plane_s16_10bit_resizing_horizontal() {
        let image_width = 8;
        let image_height = 1;
        const CN: usize = 1;
        let mut image = vec![0i16; image_height * image_width * CN];
        for (i, px) in image.iter_mut().enumerate() {
            *px = (100 + i as i16 * 10).min(511);
        }
        image[0] = -200;

        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);

        let src_store =
            ImageStore::<i16, CN>::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store =
            ImageStoreMut::<i16, CN>::alloc_with_depth(image_width / 2, image_height, 10);

        let planned = scaler
            .plan_planar_resampling_s16(src_store.size(), target_store.size(), 10)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();

        let target_data = target_store.buffer.borrow();
        // All output pixels must stay within signed 10-bit bounds [-512, 511]
        for &px in target_data.iter() {
            assert!(
                px >= -512 && px <= 511,
                "pixel {px} out of 10-bit signed range"
            );
        }
    }

    #[test]
    fn check_plane_s16_10bit_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 1;
        let mut image = vec![0i16; image_height * image_width * CN];
        for px in image.iter_mut() {
            *px = 124;
        }
        image[0] = -200;

        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);

        let src_store =
            ImageStore::<i16, CN>::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store =
            ImageStoreMut::<i16, CN>::alloc_with_depth(image_width, image_height / 2, 10);

        let planned = scaler
            .plan_planar_resampling_s16(src_store.size(), target_store.size(), 10)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();

        let target_data = target_store.buffer.borrow();
        for &px in target_data.iter() {
            assert!(
                px >= -512 && px <= 511,
                "pixel {px} out of 10-bit signed range"
            );
        }
        for &px in target_data.iter().skip(1) {
            assert!(
                (px - 124).abs() < 30,
                "flat region drifted: got {px}, expected ~124"
            );
        }
    }

    #[test]
    fn check_plane_s16_16bit_resizing_horizontal() {
        let image_width = 8;
        let image_height = 1;
        const CN: usize = 1;
        let mut image = vec![0i16; image_height * image_width * CN];
        for (i, px) in image.iter_mut().enumerate() {
            *px = (1000 + i as i16 * 500).min(i16::MAX);
        }
        image[0] = i16::MIN;

        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);

        let src_store =
            ImageStore::<i16, CN>::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store =
            ImageStoreMut::<i16, CN>::alloc_with_depth(image_width / 2, image_height, 16);

        let planned = scaler
            .plan_planar_resampling_s16(src_store.size(), target_store.size(), 16)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();

        let target_data = target_store.buffer.borrow();
        for &px in target_data.iter() {
            assert!(
                px >= i16::MIN && px <= i16::MAX,
                "pixel {px} out of 16-bit signed range"
            );
        }
    }

    #[test]
    fn check_plane_s16_16bit_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 1;
        let mut image = vec![0i16; image_height * image_width * CN];
        for px in image.iter_mut() {
            *px = 5000;
        }

        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);

        let src_store =
            ImageStore::<i16, CN>::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store =
            ImageStoreMut::<i16, CN>::alloc_with_depth(image_width, image_height / 2, 16);

        let planned = scaler
            .plan_planar_resampling_s16(src_store.size(), target_store.size(), 16)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();

        let target_data = target_store.buffer.borrow();
        for &px in target_data.iter() {
            assert!(
                px >= i16::MIN && px <= i16::MAX,
                "pixel {px} out of 16-bit signed range"
            );
        }
        // Flat region check — skip pixel influenced by the outlier at [0]
        for &px in target_data.iter().skip(1) {
            assert!(
                (px as i32 - 5000).abs() < 500,
                "flat region drifted: got {px}, expected ~5000"
            );
        }
    }

    #[test]
    fn check_plane_s16_10bit_both_axes() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 1;
        let mut image = vec![0i16; image_height * image_width * CN];
        for px in image.iter_mut() {
            *px = 200;
        }
        image[0] = -300;

        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);

        let src_store =
            ImageStore::<i16, CN>::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store =
            ImageStoreMut::<i16, CN>::alloc_with_depth(image_width / 2, image_height / 2, 10);

        let planned = scaler
            .plan_planar_resampling_s16(src_store.size(), target_store.size(), 10)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();

        let target_data = target_store.buffer.borrow();
        for &px in target_data.iter() {
            assert!(
                px >= -512 && px <= 511,
                "pixel {px} out of 10-bit signed range"
            );
        }
    }

    #[test]
    fn check_plane_s16_16bit_both_axes() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 1;
        let mut image = vec![0i16; image_height * image_width * CN];
        for px in image.iter_mut() {
            *px = 10000;
        }
        image[0] = i16::MIN;

        let scaler =
            Scaler::new(ResamplingFunction::Lanczos3).set_threading_policy(ThreadingPolicy::Single);

        let src_store =
            ImageStore::<i16, CN>::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store =
            ImageStoreMut::<i16, CN>::alloc_with_depth(image_width / 2, image_height / 2, 16);

        let planned = scaler
            .plan_planar_resampling_s16(src_store.size(), target_store.size(), 16)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();

        let target_data = target_store.buffer.borrow();
        for &px in target_data.iter() {
            assert!(
                px >= i16::MIN && px <= i16::MAX,
                "pixel {px} out of 16-bit signed range"
            );
        }
    }
}
