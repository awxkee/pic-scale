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
        let dst_width = destination.width;

        let mut processed_4 = false;

        let dst_height = destination.height;
        let dst_bit_depth = destination.bit_depth as u32;
        let dst_buffer = destination.projected();
        let src = source.projected();

        if let Some(dispatcher) = self.filter_4_rows {
            dst_buffer
                .tb_par_chunks_mut(dst_stride * 4)
                .take(dst_height / 4)
                .zip(src.chunks(src_stride * 4).take(dst_height / 4))
                .for_each(&pool, |(dst, src)| {
                    dispatcher(
                        src,
                        src_stride,
                        dst,
                        dst_stride,
                        &self.filter_weights,
                        dst_bit_depth,
                    );
                });
            processed_4 = true;
        }

        let safe_4_chunks = (dst_height / 4) * 4;

        let already_processed_y = safe_4_chunks * 4;

        if already_processed_y < dst_height {
            let left_src = if processed_4 {
                &src[already_processed_y * src_stride..]
            } else {
                src
            };

            let max_length = dst_buffer.len();
            let left_dst = if processed_4 {
                let offset = (already_processed_y * dst_stride).min(max_length);
                &mut dst_buffer[offset..]
            } else {
                dst_buffer
            };

            let one_row_filter = self.filter_row;

            left_dst
                .tb_par_chunks_mut(dst_stride)
                .zip(left_src.chunks(src_stride))
                .for_each(&pool, |(dst, src)| {
                    one_row_filter(
                        src,
                        &mut dst[..dst_width * N],
                        &self.filter_weights,
                        dst_bit_depth,
                    );
                });
        }
    }

    fn can_do_4_rows(&self) -> bool {
        self.filter_4_rows.is_some()
    }

    fn run_on_4_rows(
        &self,
        src: &[T],
        src_stride: usize,
        dst: &mut [T],
        dst_stride: usize,
        bit_depth: u32,
    ) {
        if let Some(dispatcher) = self.filter_4_rows {
            dispatcher(
                src,
                src_stride,
                dst,
                dst_stride,
                &self.filter_weights,
                bit_depth,
            );
        } else {
            unreachable!(
                "4 rows filter was called where it's not implemented, this is an internal configuration error"
            );
        }
    }

    fn run_on_row(&self, src: &[T], dst: &mut [T], bit_depth: u32) {
        let one_row_filter = self.filter_row;
        one_row_filter(src, dst, &self.filter_weights, bit_depth);
    }
}
