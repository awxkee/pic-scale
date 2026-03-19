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

use crate::filter_weights::FilterWeights;
use crate::image_store::ImageStoreMut;
use crate::scaler::WorkloadStrategy;
use crate::{ImageSize, ImageStore, ThreadingPolicy};
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct ConvolutionOptions {
    pub(crate) workload_strategy: WorkloadStrategy,
    pub(crate) bit_depth: usize,
    pub(crate) src_size: ImageSize,
    pub(crate) dst_size: ImageSize,
}

pub(crate) trait HorizontalFilterPass<T, W, const N: usize>
where
    T: Clone + Copy + Debug,
{
    fn horizontal_plan(
        filter_weights: FilterWeights<W>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn RowFilter<T, N> + Send + Sync>;
}

pub(crate) trait VerticalConvolutionPass<T, W, const N: usize>
where
    T: Clone + Copy + Debug,
{
    fn vertical_plan(
        filter_weights: FilterWeights<W>,
        threading_policy: ThreadingPolicy,
        options: ConvolutionOptions,
    ) -> Arc<dyn ColumnFilter<T, N> + Send + Sync>;
}

pub(crate) trait RowFilter<T, const N: usize>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn filter(&self, source: &ImageStore<'_, T, N>, destination: &mut ImageStoreMut<T, N>);
}

pub(crate) trait ColumnFilter<T, const N: usize>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn filter(&self, source: &ImageStore<'_, T, N>, destination: &mut ImageStoreMut<T, N>);
}
