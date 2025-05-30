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

use crate::ImageStore;
use crate::filter_weights::{FilterBounds, FilterWeights, WeightsConverter};
use crate::image_store::ImageStoreMut;
use core::f16;
use rayon::ThreadPool;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_vertical_dispatch_f16<V: Copy + Send + Sync, const CN: usize>(
    image_store: &ImageStore<f16, CN>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<f16, CN>,
    pool: &Option<ThreadPool>,
    dispatcher: fn(usize, &FilterBounds, &[f16], &mut [f16], usize, &[V]),
    weights_converter: impl WeightsConverter<V>,
) {
    let src_stride = image_store.stride();
    let dst_stride = destination.stride();

    let c_weights = weights_converter.prepare_weights(&filter_weights).weights;

    let dst_width = destination.width;

    if let Some(pool) = pool {
        pool.install(|| {
            destination
                .buffer
                .borrow_mut()
                .par_chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(y, row)| {
                    let bounds = filter_weights.bounds[y];
                    let filter_offset = y * filter_weights.aligned_size;
                    let weights = &c_weights[filter_offset..];
                    let source_buffer = image_store.buffer.as_ref();
                    dispatcher(
                        dst_width,
                        &bounds,
                        source_buffer,
                        &mut row[..dst_width * CN],
                        src_stride,
                        weights,
                    );
                });
        });
    } else {
        destination
            .buffer
            .borrow_mut()
            .chunks_exact_mut(dst_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let bounds = filter_weights.bounds[y];
                let filter_offset = y * filter_weights.aligned_size;
                let weights = &c_weights[filter_offset..];
                let source_buffer = image_store.buffer.as_ref();
                dispatcher(
                    dst_width,
                    &bounds,
                    source_buffer,
                    &mut row[..dst_width * CN],
                    src_stride,
                    weights,
                );
            });
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_horizontal_dispatch_f16<V: Copy + Send + Sync, const CN: usize>(
    image_store: &ImageStore<f16, CN>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<f16, CN>,
    pool: &Option<ThreadPool>,
    dispatcher_4_rows: Option<
        fn(usize, usize, &FilterWeights<V>, &[f16], usize, &mut [f16], usize),
    >,
    dispatcher_row: fn(usize, usize, &FilterWeights<V>, &[f16], &mut [f16]),
    weights_converter: impl WeightsConverter<V>,
) {
    let src_stride = image_store.stride();
    let dst_stride = destination.stride();
    let dst_width = destination.width;
    let src_width = image_store.width;

    let c_weights = weights_converter.prepare_weights(&filter_weights);

    if let Some(pool) = pool {
        pool.install(|| {
            let mut processed_4 = false;

            if let Some(dispatcher) = dispatcher_4_rows {
                image_store
                    .buffer
                    .as_ref()
                    .par_chunks_exact(src_stride * 4)
                    .zip(
                        destination
                            .buffer
                            .borrow_mut()
                            .par_chunks_exact_mut(dst_stride * 4),
                    )
                    .for_each(|(src, dst)| {
                        dispatcher(
                            dst_width, src_width, &c_weights, src, src_stride, dst, dst_stride,
                        );
                    });
                processed_4 = true;
            }

            let left_src_rows = if processed_4 {
                image_store
                    .buffer
                    .as_ref()
                    .chunks_exact(src_stride * 4)
                    .remainder()
            } else {
                image_store.buffer.as_ref()
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

            left_src_rows
                .par_chunks_exact(src_stride)
                .zip(left_dst_rows.par_chunks_exact_mut(dst_stride))
                .for_each(|(src, dst)| {
                    dispatcher_row(dst_width, src_width, &c_weights, src, dst);
                });
        });
    } else {
        let mut processed_4 = false;
        if let Some(dispatcher) = dispatcher_4_rows {
            for (src, dst) in image_store
                .buffer
                .as_ref()
                .chunks_exact(src_stride * 4)
                .zip(
                    destination
                        .buffer
                        .borrow_mut()
                        .chunks_exact_mut(dst_stride * 4),
                )
            {
                dispatcher(
                    dst_width, src_width, &c_weights, src, src_stride, dst, dst_stride,
                );
            }
            processed_4 = true;
        }

        let left_src_rows = if processed_4 {
            image_store
                .buffer
                .as_ref()
                .chunks_exact(src_stride * 4)
                .remainder()
        } else {
            image_store.buffer.as_ref()
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
        for (src, dst) in left_src_rows
            .chunks_exact(src_stride)
            .zip(left_dst_rows.chunks_exact_mut(dst_stride))
        {
            dispatcher_row(dst_width, src_width, &c_weights, src, dst);
        }
    }
}
