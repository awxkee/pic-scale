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
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::{ImageSize, ImageStore, ImageStoreMut, ThreadingPolicy};
use novtb::{ParallelZonedIterator, TbSliceMut};

pub(crate) struct VerticalFiltering<T, F, const N: usize> {
    pub(crate) filter_row: fn(usize, &FilterBounds, &[T], &mut [T], usize, &[F], u32),
    pub(crate) filter_weights: FilterWeights<F>,
    pub(crate) threading_policy: ThreadingPolicy,
}

impl<T: Send + Sync, F: Send + Sync, const N: usize> Filtering<T, N> for VerticalFiltering<T, F, N>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn filter(&self, source: &ImageStore<'_, T, N>, destination: &mut ImageStoreMut<T, N>) {
        let pool = self
            .threading_policy
            .get_nova_pool(ImageSize::new(destination.width, destination.height));
        let src_stride = source.stride();
        let dst_stride = destination.stride();

        let dst_width = destination.width;

        let row_filter = self.filter_row;

        destination
            .buffer
            .borrow_mut()
            .tb_par_chunks_exact_mut(dst_stride)
            .for_each_enumerated(&pool, |y, row| {
                let bounds = self.filter_weights.bounds[y];
                let filter_offset = y * self.filter_weights.aligned_size;
                let weights = &self.filter_weights.weights[filter_offset..];
                let source_buffer = source.buffer.as_ref();
                row_filter(
                    dst_width,
                    &bounds,
                    source_buffer,
                    &mut row[..dst_width * N],
                    src_stride,
                    weights,
                    destination.bit_depth as u32,
                );
            });
    }
}
