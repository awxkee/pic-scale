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
use crate::ar30::{Ar30ByteOrder, Rgb30};
use crate::convolution::{ConvolutionOptions, HorizontalFilterPass, VerticalConvolutionPass};
use crate::filter_weights::{DefaultWeightsConverter, WeightsConverter};
use crate::image_size::ImageSize;
use crate::image_store::{
    AssociateAlpha, CheckStoreDensity, ImageStore, ImageStoreMut, UnassociateAlpha,
};
use crate::math::WeightsGenerator;
use crate::plan::{
    AlphaConvolvePlan, HorizontalFiltering, NonAlphaConvolvePlan, ResampleNearestPlan, Resampling,
    TrampolineFiltering, VerticalFiltering,
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
        }
    }

    /// Sets preferred workload strategy
    ///
    /// This is hint only, it may change something, or may not.
    pub fn set_workload_strategy(&mut self, workload_strategy: WorkloadStrategy) -> Self {
        self.workload_strategy = workload_strategy;
        *self
    }
}

impl Scaler {
    pub(crate) fn plan_generic_resize<
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
                threading_policy: self.threading_policy,
                _phantom_data: PhantomData,
            }));
        }
        let vertical_filters =
            T::make_weights(self.function, source_size.height, destination_size.height)?;
        let horizontal_filters =
            T::make_weights(self.function, source_size.width, destination_size.width)?;
        let options = ConvolutionOptions {
            workload_strategy: self.workload_strategy,
            bit_depth,
            src_size: source_size,
            dst_size: destination_size,
        };
        let vertical_plan =
            ImageStore::<T, N>::vertical_plan(vertical_filters, self.threading_policy, options);
        let horizontal_plan =
            ImageStore::<T, N>::horizontal_plan(horizontal_filters, self.threading_policy, options);

        let should_do_horizontal = source_size.width != destination_size.width;
        let should_do_vertical = source_size.height != destination_size.height;

        let trampoline_filter = Arc::new(TrampolineFiltering {
            horizontal_filter: horizontal_plan.clone(),
            vertical_filter: vertical_plan.clone(),
            source_size,
            target_size: destination_size,
        });

        Ok(Arc::new(NonAlphaConvolvePlan {
            source_size,
            target_size: destination_size,
            horizontal_filter: horizontal_plan,
            vertical_filter: vertical_plan,
            trampoline_filter,
            should_do_vertical,
            should_do_horizontal,
            threading_policy: self.threading_policy,
        }))
    }

    pub(crate) fn plan_generic_resize_with_alpha<
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        W,
        const N: usize,
    >(
        &self,
        source_size: ImageSize,
        destination_size: ImageSize,
        bit_depth: usize,
        needs_alpha_premultiplication: bool,
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N> + AssociateAlpha<T, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        if self.function == ResamplingFunction::Nearest {
            return Ok(Arc::new(ResampleNearestPlan {
                source_size,
                target_size: destination_size,
                threading_policy: self.threading_policy,
                _phantom_data: PhantomData,
            }));
        }
        if !needs_alpha_premultiplication {
            return self.plan_generic_resize(source_size, destination_size, bit_depth);
        }
        let vertical_filters =
            T::make_weights(self.function, source_size.height, destination_size.height)?;
        let horizontal_filters =
            T::make_weights(self.function, source_size.width, destination_size.width)?;
        let options = ConvolutionOptions {
            workload_strategy: self.workload_strategy,
            bit_depth,
            src_size: source_size,
            dst_size: destination_size,
        };
        let vertical_plan =
            ImageStore::<T, N>::vertical_plan(vertical_filters, self.threading_policy, options);
        let horizontal_plan =
            ImageStore::<T, N>::horizontal_plan(horizontal_filters, self.threading_policy, options);

        let should_do_horizontal = source_size.width != destination_size.width;
        let should_do_vertical = source_size.height != destination_size.height;

        let trampoline_filter = Arc::new(TrampolineFiltering {
            horizontal_filter: horizontal_plan.clone(),
            vertical_filter: vertical_plan.clone(),
            source_size,
            target_size: destination_size,
        });

        Ok(Arc::new(AlphaConvolvePlan {
            source_size,
            target_size: destination_size,
            threading_policy: self.threading_policy,
            horizontal_filter: horizontal_plan,
            vertical_filter: vertical_plan,
            trampoline_filter,
            should_do_vertical,
            should_do_horizontal,
            workload_strategy: self.workload_strategy,
        }))
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
        self.plan_generic_resize(source_size, target_size, 8)
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
            self.plan_generic_resize_with_alpha(source_size, target_size, 8, premultiply_alpha)
        } else {
            self.plan_generic_resize(source_size, target_size, 8)
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
        self.plan_generic_resize(source_size, target_size, 8)
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
        self.plan_generic_resize(source_size, target_size, 8)
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
            self.plan_generic_resize_with_alpha(source_size, target_size, 8, premultiply_alpha)
        } else {
            self.plan_generic_resize(source_size, target_size, 8)
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
        self.plan_generic_resize(source_size, target_size, bit_depth)
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
        self.plan_generic_resize(source_size, target_size, bit_depth)
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
            self.plan_generic_resize_with_alpha(
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
        self.plan_generic_resize(source_size, target_size, bit_depth)
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
            self.plan_generic_resize_with_alpha(
                source_size,
                target_size,
                bit_depth,
                premultiply_alpha,
            )
        } else {
            self.plan_generic_resize(source_size, target_size, bit_depth)
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
            WorkloadStrategy::PreferQuality => {
                self.plan_generic_resize::<f32, f64, 1>(source_size, target_size, 8)
            }
            WorkloadStrategy::PreferSpeed => {
                self.plan_generic_resize::<f32, f32, 1>(source_size, target_size, 8)
            }
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
            WorkloadStrategy::PreferQuality => {
                self.plan_generic_resize::<f32, f64, 2>(source_size, target_size, 8)
            }
            WorkloadStrategy::PreferSpeed => {
                self.plan_generic_resize::<f32, f32, 2>(source_size, target_size, 8)
            }
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
                WorkloadStrategy::PreferQuality => self
                    .plan_generic_resize_with_alpha::<f32, f64, 2>(
                        source_size,
                        target_size,
                        8,
                        premultiply_alpha,
                    ),
                WorkloadStrategy::PreferSpeed => self
                    .plan_generic_resize_with_alpha::<f32, f32, 2>(
                        source_size,
                        target_size,
                        8,
                        premultiply_alpha,
                    ),
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
            WorkloadStrategy::PreferQuality => {
                self.plan_generic_resize::<f32, f64, 3>(source_size, target_size, 8)
            }
            WorkloadStrategy::PreferSpeed => {
                self.plan_generic_resize::<f32, f32, 3>(source_size, target_size, 8)
            }
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
                WorkloadStrategy::PreferQuality => self
                    .plan_generic_resize_with_alpha::<f32, f64, 4>(
                        source_size,
                        target_size,
                        8,
                        premultiply_alpha,
                    ),
                WorkloadStrategy::PreferSpeed => self
                    .plan_generic_resize_with_alpha::<f32, f32, 4>(
                        source_size,
                        target_size,
                        8,
                        premultiply_alpha,
                    ),
            }
        } else {
            match self.workload_strategy {
                WorkloadStrategy::PreferQuality => {
                    self.plan_generic_resize::<f32, f64, 4>(source_size, target_size, 8)
                }
                WorkloadStrategy::PreferSpeed => {
                    self.plan_generic_resize::<f32, f32, 4>(source_size, target_size, 8)
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
    pub(crate) fn plan_resize_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
        &self,
        source_size: ImageSize,
        destination_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 4>>, PicScaleError> {
        if self.function == ResamplingFunction::Nearest {
            return Ok(Arc::new(ResampleNearestPlan {
                source_size,
                target_size: destination_size,
                threading_policy: self.threading_policy,
                _phantom_data: PhantomData,
            }));
        }
        let vertical_filters =
            u8::make_weights(self.function, source_size.height, destination_size.height)?;
        let horizontal_filters =
            u8::make_weights(self.function, source_size.width, destination_size.width)?;
        let options = ConvolutionOptions {
            workload_strategy: self.workload_strategy,
            bit_depth: 10,
            src_size: source_size,
            dst_size: destination_size,
        };
        let q_vert_filters = DefaultWeightsConverter::default().prepare_weights(&vertical_filters);
        use crate::dispatch_group_ar30::{
            get_horizontal_dispatch_ar30, get_horizontal_dispatch4_ar30,
            get_vertical_dispatcher_ar30,
        };
        let vertical_plan = Arc::new(VerticalFiltering {
            filter_weights: q_vert_filters,
            threading_policy: self.threading_policy,
            filter_row: get_vertical_dispatcher_ar30::<AR30_TYPE, AR30_ORDER>(options),
        });

        let hor_dispatch4 = get_horizontal_dispatch4_ar30::<AR30_TYPE, AR30_ORDER>(options);
        let hor_dispatch = get_horizontal_dispatch_ar30::<AR30_TYPE, AR30_ORDER>();
        let q_hor_filters = DefaultWeightsConverter::default().prepare_weights(&horizontal_filters);

        let horizontal_plan = Arc::new(HorizontalFiltering {
            filter_weights: q_hor_filters,
            threading_policy: self.threading_policy,
            filter_4_rows: Some(hor_dispatch4),
            filter_row: hor_dispatch,
        });

        let should_do_horizontal = source_size.width != destination_size.width;
        let should_do_vertical = source_size.height != destination_size.height;

        let trampoline_filter = Arc::new(TrampolineFiltering {
            horizontal_filter: horizontal_plan.clone(),
            vertical_filter: vertical_plan.clone(),
            source_size,
            target_size: destination_size,
        });

        Ok(Arc::new(NonAlphaConvolvePlan {
            source_size,
            target_size: destination_size,
            horizontal_filter: horizontal_plan,
            vertical_filter: vertical_plan,
            trampoline_filter,
            should_do_vertical,
            should_do_horizontal,
            threading_policy: self.threading_policy,
        }))
    }

    /// Creates a resampling plan for an AR30 (`RGBA2101010`) packed 10-bit image.
    ///
    /// AR30 stores each pixel as a 32-bit word with 10 bits per RGB channel and a
    /// 2-bit alpha.
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
            Ar30ByteOrder::Host => self
                .plan_resize_ar30::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Host as usize }>(
                    source_size,
                    target_size,
                ),
            Ar30ByteOrder::Network => self
                .plan_resize_ar30::<{ Rgb30::Ar30 as usize }, { Ar30ByteOrder::Network as usize }>(
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
    pub fn resize_ra30(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        order: Ar30ByteOrder,
    ) -> Result<Arc<Resampling<u8, 4>>, PicScaleError> {
        match order {
            Ar30ByteOrder::Host => self
                .plan_resize_ar30::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Host as usize }>(
                    source_size,
                    target_size,
                ),
            Ar30ByteOrder::Network => self
                .plan_resize_ar30::<{ Rgb30::Ra30 as usize }, { Ar30ByteOrder::Network as usize }>(
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
                let plan = scaler.plan_generic_resize_with_alpha::<$fx_type, f32, $cn>(
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
                let plan = scaler.plan_generic_resize::<$fx_type, f32, $cn>(
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

    macro_rules! check_rgba8 {
        ($dst: expr, $image_width: expr, $max: expr) => {
            {
                for (y, row) in $dst.chunks_exact($image_width * 4).enumerate() {
                    for (i, dst) in row.chunks_exact(4).enumerate() {
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
        for dst in image.chunks_exact_mut(4) {
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
        for dst in image.chunks_exact_mut(4) {
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
        for dst in image.chunks_exact_mut(4) {
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
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        image[3] = 78;
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
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
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
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
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
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
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 255;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
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
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 255;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Adaptive);
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
        for dst in image.chunks_exact_mut(4) {
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
        for dst in image.chunks_exact_mut(4) {
            dst[0] = 124;
            dst[1] = 41;
            dst[2] = 99;
            dst[3] = 77;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Nearest);
        scaler.set_threading_policy(ThreadingPolicy::Adaptive);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling(src_store.size(), target_store.size(), false)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        check_rgba8!(target_data, image_width, 80);
    }
}
