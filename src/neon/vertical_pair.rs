/*
 * Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *  contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use crate::convolution::{ColumnFilter, ColumnSource};
use crate::filter_weights::FilterBounds;
use crate::plan::VerticalFiltering;
use crate::{ImageStore, ImageStoreMut};

#[cfg(feature = "rdm")]
pub(super) const U16_PAIR_TAPS: usize = 12;
#[cfg(feature = "rdm")]
pub(super) const U16_PAIR_STEP: usize = 2;

type RowPair<T, F, const TAPS: usize> = unsafe fn(
    src: &[T],
    stride: usize,
    first: &mut [T],
    second: &mut [T],
    first_weights: &[F; TAPS],
    second_weights: &[F; TAPS],
    bit_depth: u32,
);

pub(crate) struct PairedVertical<
    T,
    F,
    const N: usize,
    const TAPS: usize,
    const ROW_STEP: usize,
    const TILE: usize,
> {
    pub(crate) filter: VerticalFiltering<T, F, N>,
    pub(crate) convolve_pair: RowPair<T, F, TAPS>,
}

#[cfg(feature = "rdm")]
pub(crate) type PairedU16Vertical<F, const N: usize> =
    PairedVertical<u16, F, N, U16_PAIR_TAPS, U16_PAIR_STEP, 16>;

impl<
    T: Clone + Send + Sync,
    F: Send + Sync,
    const N: usize,
    const TAPS: usize,
    const ROW_STEP: usize,
    const TILE: usize,
> ColumnFilter<T, N> for PairedVertical<T, F, N, TAPS, ROW_STEP, TILE>
{
    fn filter(&self, source: &ImageStore<'_, T, N>, destination: &mut ImageStoreMut<T, N>) {
        self.filter.filter(source, destination);
    }

    fn source_bounds(&self, row: usize) -> FilterBounds {
        self.filter.source_bounds(row)
    }

    fn run_on_4_rows(
        &self,
        src: &[T],
        dst: &mut [T],
        dst_width: usize,
        src_stride: usize,
        row: usize,
        bit_depth: u32,
    ) {
        self.run_on_band_rows(
            &ColumnSource {
                pixels: src,
                width: dst_width,
                stride: src_stride,
                first_row: 0,
            },
            dst,
            row,
            bit_depth,
        );
    }

    fn run_on_row(
        &self,
        src: &[T],
        dst: &mut [T],
        dst_width: usize,
        src_stride: usize,
        row: usize,
        bit_depth: u32,
    ) {
        self.filter
            .run_on_row(src, dst, dst_width, src_stride, row, bit_depth);
    }

    fn run_on_band_rows(
        &self,
        source: &ColumnSource<'_, T>,
        dst: &mut [T],
        row: usize,
        bit_depth: u32,
    ) {
        let row_len = source.width * N;
        if !row_len.is_multiple_of(TILE) {
            self.filter.run_on_band_rows(source, dst, row, bit_depth);
            return;
        }

        let mut pairs = dst.chunks_exact_mut(2 * row_len);
        let mut y = row;
        for pair in pairs.by_ref() {
            let (first, second) = pair.split_at_mut(row_len);
            let weights = &self.filter.filter_weights;
            let a = weights.bounds[y];
            let b = weights.bounds[y + 1];
            if a.size == TAPS && b.size == TAPS && b.start.checked_sub(a.start) == Some(ROW_STEP) {
                let input = &source.pixels[(a.start - source.first_row) * source.stride..];
                let first_weights = weights.weights[y * weights.aligned_size..]
                    .first_chunk::<TAPS>()
                    .unwrap();
                let second_weights = weights.weights[(y + 1) * weights.aligned_size..]
                    .first_chunk::<TAPS>()
                    .unwrap();
                // The factory selects the supported kernel. The validated row widths and
                // matching tap windows keep all vector loads and stores in bounds.
                unsafe {
                    (self.convolve_pair)(
                        input,
                        source.stride,
                        first,
                        second,
                        first_weights,
                        second_weights,
                        bit_depth,
                    );
                }
            } else {
                self.filter.run_on_band_row(source, first, y, bit_depth);
                self.filter
                    .run_on_band_row(source, second, y + 1, bit_depth);
            }
            y += 2;
        }
        self.filter
            .run_on_band_rows(source, pairs.into_remainder(), y, bit_depth);
    }

    fn run_on_band_row(
        &self,
        source: &ColumnSource<'_, T>,
        dst: &mut [T],
        row: usize,
        bit_depth: u32,
    ) {
        self.filter.run_on_band_row(source, dst, row, bit_depth);
    }
}
