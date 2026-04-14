/*
 * Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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
use crate::convolution::{
    ConvolutionOptions, HorizontalFilterPass, TrampolineFilter, VerticalConvolutionPass,
};
use crate::image_store::{AssociateAlpha, CheckStoreDensity, UnassociateAlpha};
use crate::math::WeightsGenerator;
use crate::plan::default_planner::DefaultPlanner;
use crate::plan::superresolution::plan_intermediate_sizes;
use crate::plan::supersampling::{supersampling_intermediate_size, supersampling_prefilter};
use crate::plan::{
    MultiStepResamplePlan, NoopPlan, ResampleNearestPlan, TrampolineFiltering, make_alpha_plan,
};
use crate::{
    ImageSize, ImageStore, ImageStoreMut, PicScaleError, Resampling, ResamplingFunction, Scaler,
};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

pub(crate) struct AlphaPlanner {}

impl AlphaPlanner {
    /// Builds a multi-step upscaling plan, chaining intermediate resize steps.
    /// Alpha is associated before the first step and unassociated after the last.
    fn plan_multi_step_upscale_with_alpha<T, W, const N: usize>(
        scaler: &Scaler,
        source_size: ImageSize,
        destination_size: ImageSize,
        bit_depth: usize,
        needs_alpha_premultiplication: bool,
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N> + AssociateAlpha<T, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        let intermediates = plan_intermediate_sizes(source_size, destination_size, scaler.function);

        let all_sizes: Vec<ImageSize> = intermediates
            .iter()
            .copied()
            .chain(std::iter::once(destination_size))
            .collect();

        let last_idx = all_sizes.len() - 1;
        let mut steps: Vec<Arc<Resampling<T, N>>> = Vec::new();
        let mut prev_size = source_size;

        for (idx, &next_size) in all_sizes.iter().enumerate() {
            let is_first = idx == 0;
            let is_last = idx == last_idx;

            let sub_plan = Self::build_single_step_plan_with_alpha::<T, W, N>(
                scaler,
                prev_size,
                next_size,
                bit_depth,
                is_first && needs_alpha_premultiplication,
                is_last && needs_alpha_premultiplication,
            )?;

            steps.push(sub_plan);
            prev_size = next_size;
        }

        Ok(Arc::new(MultiStepResamplePlan::new(
            steps,
            source_size,
            destination_size,
        )))
    }

    /// Builds a two-step supersampling plan for large downscales:
    /// a cheap pre-filter pass to an intermediate size, then a quality pass
    /// to the final destination.
    fn plan_supersampling<T, W, const N: usize>(
        scaler: &Scaler,
        source_size: ImageSize,
        destination_size: ImageSize,
        bit_depth: usize,
        needs_alpha_premultiplication: bool,
        prefilter: ResamplingFunction,
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N> + AssociateAlpha<T, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        let intermediate = supersampling_intermediate_size(source_size, destination_size);

        // Nearest does pixel selection only — premultiplication has no effect.
        let prefilter_needs_alpha =
            needs_alpha_premultiplication && prefilter != ResamplingFunction::Nearest;

        let pre_scaler = Scaler {
            function: prefilter,
            threading_policy: scaler.threading_policy,
            workload_strategy: scaler.workload_strategy,
            multi_step_upscaling: false,
            supersampling: false,
        };

        let pre_plan = Self::build_pre_pass_plan::<T, W, N>(
            &pre_scaler,
            source_size,
            intermediate,
            bit_depth,
            prefilter_needs_alpha,
        )?;

        let quality_plan = Self::build_quality_pass_plan::<T, W, N>(
            &pre_scaler,
            intermediate,
            destination_size,
            bit_depth,
            needs_alpha_premultiplication,
            prefilter_needs_alpha,
        )?;

        Ok(Arc::new(MultiStepResamplePlan::new(
            vec![pre_plan, quality_plan],
            source_size,
            destination_size,
        )))
    }

    /// Builds the averaging pre-pass for supersampling.
    /// Associates alpha before blending if needed; keeps data premultiplied
    /// for the quality pass.
    fn build_pre_pass_plan<T, W, const N: usize>(
        pre_scaler: &Scaler,
        source_size: ImageSize,
        intermediate: ImageSize,
        bit_depth: usize,
        prefilter_needs_alpha: bool,
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N> + AssociateAlpha<T, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        if prefilter_needs_alpha {
            // Associate before averaging; leave premultiplied for quality pass.
            Self::build_single_step_plan_with_alpha::<T, W, N>(
                pre_scaler,
                source_size,
                intermediate,
                bit_depth,
                true,  // associate
                false, // keep premultiplied
            )
        } else {
            // Nearest pre-pass: plain resize, no alpha handling.
            pre_scaler.build_single_step_plan::<T, W, N>(source_size, intermediate, bit_depth)
        }
    }

    /// Builds the quality pass for supersampling.
    /// If the pre-pass already premultiplied, skip association and only
    /// unassociate at the end. Otherwise handle alpha as a normal single step.
    fn build_quality_pass_plan<T, W, const N: usize>(
        scaler: &Scaler,
        intermediate: ImageSize,
        destination_size: ImageSize,
        bit_depth: usize,
        needs_alpha_premultiplication: bool,
        prefilter_needs_alpha: bool,
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N> + AssociateAlpha<T, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        if prefilter_needs_alpha {
            // Data is already premultiplied — skip association, only unassociate.
            Self::build_single_step_plan_with_alpha::<T, W, N>(
                scaler,
                intermediate,
                destination_size,
                bit_depth,
                false, // already premultiplied
                true,  // unassociate
            )
        } else {
            // Nearest pre-pass left straight alpha — full associate+unassociate.
            Self::build_single_step_plan_with_alpha::<T, W, N>(
                scaler,
                intermediate,
                destination_size,
                bit_depth,
                needs_alpha_premultiplication,
                needs_alpha_premultiplication,
            )
        }
    }

    pub(crate) fn plan_generic_resize_with_alpha<
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        W,
        const N: usize,
    >(
        scaler: &Scaler,
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
        if scaler.function == ResamplingFunction::Nearest {
            return Ok(Arc::new(ResampleNearestPlan {
                source_size,
                target_size: destination_size,
                threading_policy: scaler.threading_policy,
                _phantom_data: PhantomData,
            }));
        }

        let is_upscale = destination_size.width >= source_size.width
            && destination_size.height >= source_size.height;

        if scaler.multi_step_upscaling && is_upscale {
            return Self::plan_multi_step_upscale_with_alpha::<T, W, N>(
                scaler,
                source_size,
                destination_size,
                bit_depth,
                needs_alpha_premultiplication,
            );
        }

        // ── Supersampling for large downscales ────────────────────────────────────
        let is_downscale = destination_size.width <= source_size.width
            && destination_size.height <= source_size.height;

        if scaler.supersampling && is_downscale {
            let ratio_w = source_size.width as f64 / destination_size.width as f64;
            let ratio_h = source_size.height as f64 / destination_size.height as f64;

            if let Some(prefilter) = supersampling_prefilter(ratio_w, ratio_h) {
                let intermediate = supersampling_intermediate_size(source_size, destination_size);

                if intermediate.width < source_size.width
                    || intermediate.height < source_size.height
                {
                    return Self::plan_supersampling::<T, W, N>(
                        scaler,
                        source_size,
                        destination_size,
                        bit_depth,
                        needs_alpha_premultiplication,
                        prefilter,
                    );
                }
            }
        }

        Self::build_single_step_plan_with_alpha::<T, W, N>(
            scaler,
            source_size,
            destination_size,
            bit_depth,
            needs_alpha_premultiplication,
            needs_alpha_premultiplication,
        )
    }

    pub(crate) fn build_single_step_plan_with_alpha<
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        W,
        const N: usize,
    >(
        scaler: &Scaler,
        source_size: ImageSize,
        destination_size: ImageSize,
        bit_depth: usize,
        needs_alpha_premultiplication_forward: bool,
        needs_alpha_premultiplication_backward: bool,
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N> + AssociateAlpha<T, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        if scaler.function == ResamplingFunction::Nearest {
            return Ok(Arc::new(ResampleNearestPlan {
                source_size,
                target_size: destination_size,
                threading_policy: scaler.threading_policy,
                _phantom_data: PhantomData,
            }));
        }
        if !needs_alpha_premultiplication_backward && !needs_alpha_premultiplication_forward {
            return DefaultPlanner::plan_generic_resize(
                scaler,
                source_size,
                destination_size,
                bit_depth,
            );
        }

        let should_do_horizontal = source_size.width != destination_size.width;
        let should_do_vertical = source_size.height != destination_size.height;

        if !should_do_vertical && !should_do_horizontal {
            return Ok(Arc::new(NoopPlan {
                source_size,
                target_size: destination_size,
                _phantom: PhantomData,
            }));
        }

        let options = ConvolutionOptions {
            workload_strategy: scaler.workload_strategy,
            bit_depth,
            src_size: source_size,
            dst_size: destination_size,
        };

        let vertical_plan = if should_do_vertical {
            let vertical_filters =
                T::make_weights(scaler.function, source_size.height, destination_size.height)?;
            Some(ImageStore::<T, N>::vertical_plan(
                vertical_filters,
                scaler.threading_policy,
                options,
            ))
        } else {
            None
        };

        let horizontal_plan = if should_do_horizontal {
            let horizontal_filters =
                T::make_weights(scaler.function, source_size.width, destination_size.width)?;
            Some(ImageStore::<T, N>::horizontal_plan(
                horizontal_filters,
                scaler.threading_policy,
                options,
            ))
        } else {
            None
        };

        let trampoline = if should_do_vertical && should_do_horizontal {
            Some(Arc::new(TrampolineFiltering {
                horizontal_filter: horizontal_plan.as_ref().unwrap().clone(),
                vertical_filter: vertical_plan.as_ref().unwrap().clone(),
                source_size,
                target_size: destination_size,
            })
                as Arc<dyn TrampolineFilter<T, N> + Send + Sync>)
        } else {
            None
        };

        Ok(make_alpha_plan(
            source_size,
            destination_size,
            horizontal_plan,
            vertical_plan,
            trampoline,
            scaler.threading_policy,
            scaler.workload_strategy,
            needs_alpha_premultiplication_forward,
            needs_alpha_premultiplication_backward,
        ))
    }
}
