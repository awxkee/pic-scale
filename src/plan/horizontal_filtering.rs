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
use crate::convolution::RowFilter;
use crate::filter_weights::FilterWeights;
use crate::{ImageSize, ImageStore, ImageStoreMut, ThreadingPolicy};
use novtb::{ParallelZonedIterator, TbSliceMut};

pub(crate) struct HorizontalFiltering<T, F, const N: usize> {
    pub(crate) filter_4_rows: Option<
        fn(
            src: &[T],
            src_stride: usize,
            dst: &mut [T],
            dst_stride: usize,
            filter_weights: &FilterWeights<F>,
            bit_depth: u32,
        ),
    >,
    pub(crate) filter_row:
        fn(src: &[T], dst: &mut [T], filter_weights: &FilterWeights<F>, bit_depth: u32),
    pub(crate) filter_weights: FilterWeights<F>,
    pub(crate) threading_policy: ThreadingPolicy,
}

impl<T: Send + Sync, F: Send + Sync, const N: usize> RowFilter<T, N>
    for HorizontalFiltering<T, F, N>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn filter(&self, source: &ImageStore<'_, T, N>, destination: &mut ImageStoreMut<T, N>) {
        let pool = self
            .threading_policy
            .get_nova_pool(ImageSize::new(destination.width, destination.height));

        let src_stride = source.stride();
        let dst_stride = destination.stride();

        let mut processed_4 = false;

        if let Some(dispatcher) = self.filter_4_rows {
            let src = source.buffer.as_ref();
            destination
                .buffer
                .borrow_mut()
                .tb_par_chunks_exact_mut(dst_stride * 4)
                .for_each_enumerated(&pool, |y, dst| {
                    let src = &src[y * src_stride * 4..(y + 1) * src_stride * 4];

                    dispatcher(
                        src,
                        src_stride,
                        dst,
                        dst_stride,
                        &self.filter_weights,
                        destination.bit_depth as u32,
                    );
                });
            processed_4 = true;
        }

        let left_src_rows = if processed_4 {
            source
                .buffer
                .as_ref()
                .chunks_exact(src_stride * 4)
                .remainder()
        } else {
            source.buffer.as_ref()
        };
        let left_dst_rows = if processed_4 {
            destination
                .buffer
                .borrow_mut()
                .chunks_exact_mut(dst_stride * 4)
                .into_remainder()
        } else {
            destination.buffer.borrow_mut()
        };

        let one_row_filter = self.filter_row;

        left_dst_rows
            .tb_par_chunks_exact_mut(dst_stride)
            .for_each_enumerated(&pool, |y, dst| {
                let src = &left_src_rows[y * src_stride..(y + 1) * src_stride];
                one_row_filter(src, dst, &self.filter_weights, destination.bit_depth as u32);
            });
    }
}
