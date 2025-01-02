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
use crate::filter_weights::{FilterBounds, FilterWeights, WeightsConverter};
use crate::image_store::ImageStoreMut;
use crate::support::PRECISION;
use crate::ImageStore;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;
use std::sync::Arc;

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_horizontal_dispatch_u8<const CHANNELS: usize>(
    image_store: &ImageStore<u8, CHANNELS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<u8, CHANNELS>,
    pool: &Option<ThreadPool>,
    dispatcher_4_rows: Option<fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i16>)>,
    dispatcher_1_row: fn(&[u8], &mut [u8], &FilterWeights<i16>),
    weights_converter: impl WeightsConverter,
) {
    let approx_weights = weights_converter.prepare_weights(&filter_weights);

    let src = image_store.buffer.as_ref();
    let dst = destination.buffer.borrow_mut();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        pool.install(|| {
            let mut rem = dst;
            let mut src_rem = src;
            if let Some(dispatcher_4) = dispatcher_4_rows {
                rem.par_chunks_exact_mut(dst_stride * 4)
                    .zip(src_rem.par_chunks_exact(src_stride * 4))
                    .for_each(|(dst, src)| {
                        dispatcher_4(src, src_stride, dst, dst_stride, &arc_weights);
                    });

                rem = rem.chunks_exact_mut(dst_stride * 4).into_remainder();
                src_rem = src_rem.chunks_exact(src_stride * 4).remainder();
            }

            rem.par_chunks_exact_mut(dst_stride)
                .zip(src_rem.par_chunks_exact(src_stride))
                .for_each(|(dst, src)| {
                    dispatcher_1_row(src, dst, &arc_weights);
                });
        });
    } else {
        let mut rem = dst;
        let mut src_rem = src;
        if let Some(dispatcher_4) = dispatcher_4_rows {
            rem.chunks_exact_mut(dst_stride * 4)
                .zip(src_rem.chunks_exact(src_stride * 4))
                .for_each(|(dst, src)| {
                    dispatcher_4(src, src_stride, dst, dst_stride, &approx_weights);
                });

            rem = rem.chunks_exact_mut(dst_stride * 4).into_remainder();
            src_rem = src_rem.chunks_exact(src_stride * 4).remainder();
        }

        rem.chunks_exact_mut(dst_stride)
            .zip(src_rem.chunks_exact(src_stride))
            .for_each(|(dst, src)| {
                dispatcher_1_row(src, dst, &approx_weights);
            });
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_vertical_dispatch_u8<'a, const COMPONENTS: usize>(
    image_store: &ImageStore<u8, COMPONENTS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<'a, u8, COMPONENTS>,
    pool: &Option<ThreadPool>,
    dispatcher: fn(usize, &FilterBounds, &[u8], &mut [u8], usize, &[i16]),
) {
    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;

    let dst_width = destination.width;

    if let Some(pool) = pool {
        pool.install(|| {
            let destination_image = destination.buffer.borrow_mut();
            let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
            destination_image
                .par_chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(y, row)| {
                    let bounds = filter_weights.bounds[y];
                    let filter_offset = y * filter_weights.aligned_size;
                    let weights = &approx.weights[filter_offset..];
                    let source_buffer = image_store.buffer.as_ref();
                    dispatcher(dst_width, &bounds, source_buffer, row, src_stride, weights);
                });
        });
    } else {
        let destination_image = destination.buffer.borrow_mut();
        let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
        destination_image
            .chunks_exact_mut(dst_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let bounds = filter_weights.bounds[y];
                let filter_offset = y * filter_weights.aligned_size;
                let weights = &approx.weights[filter_offset..];
                let source_buffer = image_store.buffer.as_ref();
                dispatcher(dst_width, &bounds, source_buffer, row, src_stride, weights);
            });
    }
}
