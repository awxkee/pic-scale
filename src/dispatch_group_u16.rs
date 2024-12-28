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
use crate::handler_provider::{
    ColumnHandlerFixedPoint, ColumnHandlerFloatingPoint, RowHandlerFixedPoint,
    RowHandlerFloatingPoint,
};
use crate::image_store::ImageStoreMut;
use crate::support::PRECISION;
use crate::ImageStore;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_horizontal_dispatch_u16<const CHANNELS: usize>(
    image_store: &ImageStore<u16, CHANNELS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<u16, CHANNELS>,
    pool: &Option<ThreadPool>,
) {
    let src = image_store.buffer.as_ref();
    let dst = destination.buffer.borrow_mut();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;
    let bit_depth = image_store.bit_depth;

    if let Some(pool) = pool {
        pool.install(|| {
            if bit_depth > 12 {
                dst.par_chunks_exact_mut(dst_stride * 4)
                    .zip(src.par_chunks_exact(src_stride * 4))
                    .for_each(|(dst, src)| {
                        u16::handle_row_4::<CHANNELS>(
                            src,
                            src_stride,
                            dst,
                            dst_stride,
                            &filter_weights,
                            bit_depth as u32,
                        );
                    });

                let remainder = dst.chunks_exact_mut(dst_stride * 4).into_remainder();
                let src_remainder = src.chunks_exact(src_stride * 4).remainder();

                remainder
                    .par_chunks_exact_mut(dst_stride)
                    .zip(src_remainder.par_chunks_exact(src_stride))
                    .for_each(|(dst, src)| {
                        u16::handle_row::<CHANNELS>(src, dst, &filter_weights, bit_depth as u32);
                    });
            } else {
                let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
                dst.par_chunks_exact_mut(dst_stride * 4)
                    .zip(src.par_chunks_exact(src_stride * 4))
                    .for_each(|(dst, src)| {
                        u16::handle_fixed_row_4::<i32, CHANNELS>(
                            src,
                            src_stride,
                            dst,
                            dst_stride,
                            &approx,
                            bit_depth as u32,
                        );
                    });

                let remainder = dst.chunks_exact_mut(dst_stride * 4).into_remainder();
                let src_remainder = src.chunks_exact(src_stride * 4).remainder();

                remainder
                    .par_chunks_exact_mut(dst_stride)
                    .zip(src_remainder.par_chunks_exact(src_stride))
                    .for_each(|(dst, src)| {
                        u16::handle_fixed_row::<i32, CHANNELS>(src, dst, &approx, bit_depth as u32);
                    });
            }
        });
    } else if bit_depth > 12 {
        dst.chunks_exact_mut(dst_stride * 4)
            .zip(src.chunks_exact(src_stride * 4))
            .for_each(|(dst, src)| {
                u16::handle_row_4::<CHANNELS>(
                    src,
                    src_stride,
                    dst,
                    dst_stride,
                    &filter_weights,
                    bit_depth as u32,
                );
            });

        let remainder = dst.chunks_exact_mut(dst_stride * 4).into_remainder();
        let src_remainder = src.chunks_exact(src_stride * 4).remainder();

        remainder
            .chunks_exact_mut(dst_stride)
            .zip(src_remainder.chunks_exact(src_stride))
            .for_each(|(dst, src)| {
                u16::handle_row::<CHANNELS>(src, dst, &filter_weights, bit_depth as u32);
            });
    } else {
        let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
        dst.chunks_exact_mut(dst_stride * 4)
            .zip(src.chunks_exact(src_stride * 4))
            .for_each(|(dst, src)| {
                u16::handle_fixed_row_4::<i32, CHANNELS>(
                    src,
                    src_stride,
                    dst,
                    dst_stride,
                    &approx,
                    bit_depth as u32,
                );
            });

        let remainder = dst.chunks_exact_mut(dst_stride * 4).into_remainder();
        let src_remainder = src.chunks_exact(src_stride * 4).remainder();

        remainder
            .chunks_exact_mut(dst_stride)
            .zip(src_remainder.chunks_exact(src_stride))
            .for_each(|(dst, src)| {
                u16::handle_fixed_row::<i32, CHANNELS>(src, dst, &approx, bit_depth as u32);
            });
    }
}

pub(crate) fn convolve_vertical_dispatch_u16<const COMPONENTS: usize>(
    image_store: &ImageStore<u16, COMPONENTS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<'_, u16, COMPONENTS>,
    pool: &Option<ThreadPool>,
) {
    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;
    let bit_depth = image_store.bit_depth;

    let dst_width = destination.width;

    if let Some(pool) = pool {
        pool.install(|| {
            let destination_image = destination.buffer.borrow_mut();
            if bit_depth > 12 {
                destination_image
                    .par_chunks_exact_mut(dst_stride)
                    .enumerate()
                    .for_each(|(y, row)| {
                        let bounds = filter_weights.bounds[y];
                        let filter_offset = y * filter_weights.aligned_size;
                        let weights = &filter_weights.weights[filter_offset..];
                        let source_buffer = image_store.buffer.as_ref();
                        u16::handle_floating_column(
                            dst_width,
                            &bounds,
                            source_buffer,
                            row,
                            src_stride,
                            weights,
                            bit_depth as u32,
                        );
                    });
            } else {
                let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
                destination_image
                    .par_chunks_exact_mut(dst_stride)
                    .enumerate()
                    .for_each(|(y, row)| {
                        let bounds = filter_weights.bounds[y];
                        let filter_offset = y * filter_weights.aligned_size;
                        let weights = &approx.weights[filter_offset..];
                        let source_buffer = image_store.buffer.as_ref();
                        u16::handle_fixed_column::<i32, COMPONENTS>(
                            dst_width,
                            &bounds,
                            source_buffer,
                            row,
                            src_stride,
                            weights,
                            bit_depth as u32,
                        );
                    });
            }
        });
    } else if bit_depth > 12 {
        let destination_image = destination.buffer.borrow_mut();
        destination_image
            .chunks_exact_mut(dst_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let bounds = filter_weights.bounds[y];
                let filter_offset = y * filter_weights.aligned_size;
                let weights = &filter_weights.weights[filter_offset..];
                let source_buffer = image_store.buffer.as_ref();
                u16::handle_floating_column(
                    dst_width,
                    &bounds,
                    source_buffer,
                    row,
                    src_stride,
                    weights,
                    bit_depth as u32,
                );
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
                u16::handle_fixed_column::<i32, COMPONENTS>(
                    dst_width,
                    &bounds,
                    source_buffer,
                    row,
                    src_stride,
                    weights,
                    bit_depth as u32,
                );
            });
    }
}
