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
use crate::convolution::{ColumnFilter, RowFilter, TrampolineFilter};
use crate::{ImageSize, ImageStore, ImageStoreMut};
use std::sync::Arc;

pub(crate) struct TrampolineFiltering<T, const N: usize> {
    pub(crate) horizontal_filter: Arc<dyn RowFilter<T, N> + Send + Sync>,
    pub(crate) vertical_filter: Arc<dyn ColumnFilter<T, N> + Send + Sync>,
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
}

impl<T: Send + Sync, const N: usize> TrampolineFilter<T, N> for TrampolineFiltering<T, N>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn filter(
        &self,
        source: &ImageStore<'_, T, N>,
        destination: &mut ImageStoreMut<T, N>,
        scratch: &mut [T],
    ) {
        let scratch_size = self.scratch_size();
        if scratch.len() < scratch_size {
            unreachable!(
                "Scratch size was smaller than required in trampoline filter, this is an internal configuration error"
            );
        }
        let scratch = &mut scratch[..scratch_size];

        let dst_stride = destination.stride();
        let dst_width = destination.width;

        let mut dst_target = destination.buffer.borrow_mut();

        let mut already_processed_y = 0usize;

        if self.horizontal_filter.can_do_4_rows() && self.target_size.height >= 4 {
            let dst = dst_target.chunks_exact_mut(dst_stride * 4);
            for (y, dst) in dst.enumerate() {
                let real_y = y * 4;
                for (i, scratch_row) in scratch.chunks_exact_mut(source.width * N).enumerate() {
                    self.vertical_filter.run_on_row(
                        source.buffer.as_ref(),
                        scratch_row,
                        source.width,
                        source.stride(),
                        real_y + i,
                        destination.bit_depth as u32,
                    );
                }
                self.horizontal_filter.run_on_4_rows(
                    scratch,
                    source.width * N,
                    dst,
                    dst_stride,
                    destination.bit_depth as u32,
                )
            }
            already_processed_y = dst_target.chunks_exact_mut(dst_stride * 4).len() * 4;
            dst_target = dst_target.chunks_exact_mut(dst_stride * 4).into_remainder();
        }

        let dst = dst_target.chunks_exact_mut(dst_stride);

        for (y, dst) in dst.enumerate() {
            let (scratch_row, _) = scratch.split_at_mut(source.width * N);
            self.vertical_filter.run_on_row(
                source.buffer.as_ref(),
                scratch_row,
                source.width,
                source.stride(),
                y + already_processed_y,
                destination.bit_depth as u32,
            );
            self.horizontal_filter.run_on_row(
                scratch_row,
                &mut dst[..dst_width * N],
                destination.bit_depth as u32,
            )
        }
    }

    fn scratch_size(&self) -> usize {
        if self.target_size.height >= 4 {
            self.source_size.width * 4.min(self.target_size.height) * N
        } else {
            self.source_size.width * N
        }
    }
}
