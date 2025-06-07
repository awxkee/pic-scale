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
use novtb::{ParallelZonedIterator, TbSliceMut};

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_horizontal_dispatch_u8<V: Send + Sync, const CN: usize>(
    image_store: &ImageStore<u8, CN>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<u8, CN>,
    pool: &novtb::ThreadPool,
    dispatcher_4_rows: Option<fn(&[u8], usize, &mut [u8], usize, &FilterWeights<V>)>,
    dispatcher_1_row: fn(&[u8], &mut [u8], &FilterWeights<V>),
    weights_converter: impl WeightsConverter<V>,
) {
    let approx_weights = weights_converter.prepare_weights(&filter_weights);

    let src = image_store.buffer.as_ref();

    let dst_stride = destination.stride();

    let dst = destination.buffer.borrow_mut();

    let src_stride = image_store.stride();

    let mut rem = dst;
    let mut src_rem = src;
    if let Some(dispatcher_4) = dispatcher_4_rows {
        rem.tb_par_chunks_exact_mut(dst_stride * 4)
            .for_each_enumerated(pool, |y, dst| {
                let src = &src_rem[y * src_stride * 4..(y + 1) * src_stride * 4];
                dispatcher_4(src, src_stride, dst, dst_stride, &approx_weights);
            });

        rem = rem.chunks_exact_mut(dst_stride * 4).into_remainder();
        src_rem = src_rem.chunks_exact(src_stride * 4).remainder();
    }

    rem.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| {
            let src = &src_rem[y * src_stride..(y + 1) * src_stride];
            dispatcher_1_row(src, dst, &approx_weights);
        });
}

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_vertical_dispatch_u8<'a, V: Copy + Send + Sync, const CN: usize>(
    image_store: &ImageStore<u8, CN>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<'a, u8, CN>,
    pool: &novtb::ThreadPool,
    dispatcher: fn(usize, &FilterBounds, &[u8], &mut [u8], usize, &[V]),
    weights_converter: impl WeightsConverter<V>,
) {
    let src_stride = image_store.stride();
    let dst_stride = destination.stride();

    let dst_width = destination.width;

    let approx = weights_converter.prepare_weights(&filter_weights);
    let process_row = |y: usize, row: &mut [u8]| {
        let bounds = filter_weights.bounds[y];
        let filter_offset = y * filter_weights.aligned_size;
        let weights = &approx.weights[filter_offset..];
        let source_buffer = image_store.buffer.as_ref();

        dispatcher(
            dst_width,
            &bounds,
            source_buffer,
            &mut row[..dst_width * CN],
            src_stride,
            weights,
        );
    };

    let destination_image = destination.buffer.borrow_mut();
    destination_image
        .tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, row| {
            process_row(y, row);
        });
}
