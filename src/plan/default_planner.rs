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
use crate::convolution::{HorizontalFilterPass, VerticalConvolutionPass};
use crate::image_store::CheckStoreDensity;
use crate::math::WeightsGenerator;
use crate::plan::superresolution::plan_intermediate_sizes;
use crate::plan::supersampling::{supersampling_intermediate_size, supersampling_prefilter};
use crate::plan::{MultiStepResamplePlan, ResampleNearestPlan};
use crate::{
    ImageSize, ImageStore, ImageStoreMut, PicScaleError, Resampling, ResamplingFunction, Scaler,
};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

pub(crate) struct DefaultPlanner {}

impl DefaultPlanner {
    /// Builds a multi-step upscaling plan without alpha handling.
    fn plan_multi_step_upscale_simple<T, W, const N: usize>(
        scaler: &Scaler,
        source_size: ImageSize,
        destination_size: ImageSize,
        bit_depth: usize,
        intermediates: &[ImageSize],
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity,
    {
        let mut steps: Vec<Arc<Resampling<T, N>>> = Vec::new();
        let mut prev_size = source_size;

        for &next_size in intermediates
            .iter()
            .chain(std::iter::once(&destination_size))
        {
            let sub_plan =
                scaler.build_single_step_plan::<T, W, N>(prev_size, next_size, bit_depth)?;
            steps.push(sub_plan);
            prev_size = next_size;
        }

        Ok(Arc::new(MultiStepResamplePlan::new(
            steps,
            source_size,
            destination_size,
        )))
    }

    /// Builds a two-step supersampling plan without alpha handling.
    fn plan_supersampling_simple<T, W, const N: usize>(
        scaler: &Scaler,
        source_size: ImageSize,
        destination_size: ImageSize,
        bit_depth: usize,
        prefilter: ResamplingFunction,
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity,
    {
        let intermediate = supersampling_intermediate_size(source_size, destination_size);

        let pre_scaler = Scaler {
            function: prefilter,
            threading_policy: scaler.threading_policy,
            workload_strategy: scaler.workload_strategy,
            multi_step_upscaling: false,
            supersampling: false,
        };

        let pre_plan =
            pre_scaler.build_single_step_plan::<T, W, N>(source_size, intermediate, bit_depth)?;

        let quality_plan =
            scaler.build_single_step_plan::<T, W, N>(intermediate, destination_size, bit_depth)?;

        Ok(Arc::new(MultiStepResamplePlan::new(
            vec![pre_plan, quality_plan],
            source_size,
            destination_size,
        )))
    }

    /// Entry point — dispatches to the appropriate planning strategy.
    pub(crate) fn plan_generic_resize<T, W, const N: usize>(
        scaler: &Scaler,
        source_size: ImageSize,
        destination_size: ImageSize,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<T, N>>, PicScaleError>
    where
        T: Clone + Copy + Debug + Send + Sync + Default + WeightsGenerator<W> + 'static,
        for<'a> ImageStore<'a, T, N>:
            VerticalConvolutionPass<T, W, N> + HorizontalFilterPass<T, W, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity,
    {
        if scaler.function == ResamplingFunction::Nearest {
            return Ok(Arc::new(ResampleNearestPlan {
                source_size,
                target_size: destination_size,
                threading_policy: scaler.threading_policy,
                _phantom_data: PhantomData,
            }));
        }

        // ── Multi-step upscaling ──────────────────────────────────────────────
        let is_upscale = destination_size.width >= source_size.width
            && destination_size.height >= source_size.height;

        if scaler.multi_step_upscaling && is_upscale {
            let intermediates =
                plan_intermediate_sizes(source_size, destination_size, scaler.function);

            if !intermediates.is_empty() {
                return Self::plan_multi_step_upscale_simple::<T, W, N>(
                    scaler,
                    source_size,
                    destination_size,
                    bit_depth,
                    &intermediates,
                );
            }
        }

        // ── Supersampling for large downscales ────────────────────────────────
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
                    return Self::plan_supersampling_simple::<T, W, N>(
                        scaler,
                        source_size,
                        destination_size,
                        bit_depth,
                        prefilter,
                    );
                }
            }
        }

        scaler.build_single_step_plan::<T, W, N>(source_size, destination_size, bit_depth)
    }
}
