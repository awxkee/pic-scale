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

use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::support::PRECISION;
use crate::unsafe_slice::UnsafeSlice;
use crate::ImageStore;
use rayon::ThreadPool;
use std::sync::Arc;

pub(crate) fn convolve_horizontal_dispatch_u16<const CHANNELS: usize>(
    image_store: &ImageStore<u16, CHANNELS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<u16, CHANNELS>,
    pool: &Option<ThreadPool>,
    dispatcher_4_rows: Option<
        fn(usize, usize, &FilterWeights<i16>, *const u16, usize, *mut u16, usize, usize),
    >,
    dispatcher_1_row: fn(usize, usize, &FilterWeights<i16>, *const u16, *mut u16, usize),
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<PRECISION>(0);

    let mut unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
    let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;
    let dst_width = destination.width;
    let src_width = image_store.width;
    let bit_depth = image_store.bit_depth;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        pool.scope(|scope| {
            let mut yy = 0usize;
            if let Some(dispatcher) = dispatcher_4_rows {
                for y in (0..destination.height.saturating_sub(4)).step_by(4) {
                    let weights = arc_weights.clone();
                    scope.spawn(move |_| {
                        let unsafe_source_ptr_0 =
                            unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * y) };
                        let dst_ptr = unsafe_slice.mut_ptr();
                        let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                        dispatcher(
                            dst_width,
                            src_width,
                            &weights,
                            unsafe_source_ptr_0,
                            src_stride,
                            unsafe_destination_ptr_0,
                            dst_stride,
                            bit_depth,
                        );
                    });
                    yy = y;
                }
            }
            for y in yy..destination.height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let unsafe_source_ptr_0 =
                        unsafe { image_store.buffer.borrow().as_ptr().add(src_stride * y) };
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    dispatcher_1_row(
                        dst_width,
                        src_width,
                        &weights,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                        bit_depth,
                    );
                });
            }
        });
    } else {
        let mut yy = 0usize;
        if let Some(dispatcher) = dispatcher_4_rows {
            while yy + 4 < destination.height {
                dispatcher(
                    dst_width,
                    src_width,
                    &approx_weights,
                    unsafe_source_ptr_0,
                    src_stride,
                    unsafe_destination_ptr_0,
                    dst_stride,
                    bit_depth,
                );
                unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride * 4) };
                unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride * 4) };
                yy += 4;
            }
        }

        for _ in yy..destination.height {
            dispatcher_1_row(
                dst_width,
                src_width,
                &approx_weights,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                bit_depth,
            );
            unsafe_source_ptr_0 = unsafe { unsafe_source_ptr_0.add(src_stride) };
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}

pub(crate) fn convolve_vertical_dispatch_u16<'a, const COMPONENTS: usize>(
    image_store: &ImageStore<u16, COMPONENTS>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStore<'a, u16, COMPONENTS>,
    pool: &Option<ThreadPool>,
    dispatcher: fn(usize, &FilterBounds, *const u16, *mut u16, usize, *const i16, usize),
) {
    let approx_weights = filter_weights.numerical_approximation_i16::<PRECISION>(0);

    let src_stride = image_store.width * image_store.channels;
    let dst_stride = destination.width * image_store.channels;
    let bit_depth = image_store.bit_depth;

    let dst_width = destination.width;

    if let Some(pool) = pool {
        let arc_weights = Arc::new(approx_weights);
        let borrowed = destination.buffer.borrow_mut();
        let unsafe_slice = UnsafeSlice::new(borrowed);
        pool.scope(|scope| {
            for y in 0..destination.height {
                let weights = arc_weights.clone();
                scope.spawn(move |_| {
                    let bounds = unsafe { weights.bounds.get_unchecked(y) };
                    let weight_ptr =
                        unsafe { weights.weights.as_ptr().add(weights.aligned_size * y) };
                    let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
                    let dst_ptr = unsafe_slice.mut_ptr();
                    let unsafe_destination_ptr_0 = unsafe { dst_ptr.add(dst_stride * y) };
                    dispatcher(
                        dst_width,
                        bounds,
                        unsafe_source_ptr_0,
                        unsafe_destination_ptr_0,
                        src_stride,
                        weight_ptr,
                        bit_depth,
                    );
                });
            }
        });
    } else {
        let unsafe_source_ptr_0 = image_store.buffer.borrow().as_ptr();
        let mut unsafe_destination_ptr_0 = destination.buffer.borrow_mut().as_mut_ptr();
        let mut filter_offset = 0usize;
        for y in 0..destination.height {
            let bounds = unsafe { approx_weights.bounds.get_unchecked(y) };
            let weight_ptr = unsafe { approx_weights.weights.as_ptr().add(filter_offset) };
            dispatcher(
                dst_width,
                bounds,
                unsafe_source_ptr_0,
                unsafe_destination_ptr_0,
                src_stride,
                weight_ptr,
                bit_depth,
            );

            filter_offset += approx_weights.aligned_size;
            unsafe_destination_ptr_0 = unsafe { unsafe_destination_ptr_0.add(dst_stride) };
        }
    }
}