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
#![allow(clippy::type_complexity)]

use crate::ImageStore;
use crate::convolution::ConvolutionOptions;
use crate::filter_weights::{
    DefaultWeightsConverter, FilterBounds, FilterWeights, WeightsConverter,
};
use crate::handler_provider::{
    ColumnHandlerFixedPoint, ColumnHandlerFloatingPoint, RowHandlerFixedPoint,
    RowHandlerFloatingPoint,
};
use crate::image_store::ImageStoreMut;
use crate::support::PRECISION;
use novtb::{ParallelZonedIterator, TbSliceMut};

trait HorizontalHandlerRow {
    fn handle_row_4(
        &self,
        src: &[u16],
        src_stride: usize,
        dst: &mut [u16],
        dst_stride: usize,
        bit_depth: u32,
    );

    fn handle_row(&self, src: &[u16], dst: &mut [u16], bit_depth: u32);
}

struct HorizontalDefaultHandler {
    weights: FilterWeights<f32>,
    handle_row_4_impl: fn(&[u16], usize, &mut [u16], usize, &FilterWeights<f32>, u32),
    handle_row_impl: fn(&[u16], &mut [u16], &FilterWeights<f32>, u32),
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
struct HorizontalDefaultHandlerQ0_31 {
    weights: FilterWeights<i32>,
    handle_row_4_impl: fn(&[u16], usize, &mut [u16], usize, &FilterWeights<i32>, u32),
    handle_row_impl: fn(&[u16], &mut [u16], &FilterWeights<i32>, u32),
}

struct HorizontalDefaultHandlerQ0_15 {
    weights: FilterWeights<i16>,
    handle_row_4_impl: fn(&[u16], usize, &mut [u16], usize, &FilterWeights<i16>, u32),
    handle_row_impl: fn(&[u16], &mut [u16], &FilterWeights<i16>, u32),
}

macro_rules! make_handler {
    ($handler_name: ident) => {
        impl HorizontalHandlerRow for $handler_name {
            fn handle_row_4(
                &self,
                src: &[u16],
                src_stride: usize,
                dst: &mut [u16],
                dst_stride: usize,
                bit_depth: u32,
            ) {
                let w = self.handle_row_4_impl;
                w(src, src_stride, dst, dst_stride, &self.weights, bit_depth);
            }

            fn handle_row(&self, src: &[u16], dst: &mut [u16], bit_depth: u32) {
                let w = self.handle_row_impl;
                w(src, dst, &self.weights, bit_depth)
            }
        }
    };
}

make_handler!(HorizontalDefaultHandler);
#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
make_handler!(HorizontalDefaultHandlerQ0_31);
make_handler!(HorizontalDefaultHandlerQ0_15);

trait RowFactoryProducer {
    fn make_handler<const CN: usize>(
        weights: &FilterWeights<f32>,
        bit_depth: usize,
    ) -> Box<dyn HorizontalHandlerRow + Send + Sync>;
}

impl RowFactoryProducer for u16 {
    fn make_handler<const CN: usize>(
        weights: &FilterWeights<f32>,
        bit_depth: usize,
    ) -> Box<dyn HorizontalHandlerRow + Send + Sync> {
        if bit_depth < 12 {
            let approx = weights.numerical_approximation_i16::<PRECISION>(0);
            return Box::new(HorizontalDefaultHandlerQ0_15 {
                weights: approx,
                handle_row_4_impl: u16::handle_fixed_row_4::<i32, CN>,
                handle_row_impl: u16::handle_fixed_row::<i32, CN>,
            });
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
        {
            let has_rdm = std::arch::is_aarch64_feature_detected!("rdm");
            if has_rdm && CN == 4 {
                use crate::neon::{
                    convolve_horizontal_rgba_neon_rows_4_hb_u16,
                    convolve_horizontal_rgba_neon_u16_hb_row,
                };
                let approx_num = weights.numerical_approximation::<i32, 31>(0);
                return Box::new(HorizontalDefaultHandlerQ0_31 {
                    weights: approx_num,
                    handle_row_4_impl: convolve_horizontal_rgba_neon_rows_4_hb_u16,
                    handle_row_impl: convolve_horizontal_rgba_neon_u16_hb_row,
                });
            } else if has_rdm && CN == 3 {
                use crate::neon::{
                    convolve_horizontal_rgb_neon_rows_4_hb_u16,
                    convolve_horizontal_rgb_neon_u16_hb_row,
                };
                let approx_num = weights.numerical_approximation::<i32, 31>(0);
                return Box::new(HorizontalDefaultHandlerQ0_31 {
                    weights: approx_num,
                    handle_row_4_impl: convolve_horizontal_rgb_neon_rows_4_hb_u16,
                    handle_row_impl: convolve_horizontal_rgb_neon_u16_hb_row,
                });
            } else if has_rdm && CN == 1 {
                use crate::neon::{
                    convolve_horizontal_plane_neon_rows_4_hb_u16,
                    convolve_horizontal_plane_neon_u16_hb_row,
                };
                let approx_num = weights.numerical_approximation::<i32, 31>(0);

                return Box::new(HorizontalDefaultHandlerQ0_31 {
                    weights: approx_num,
                    handle_row_4_impl: convolve_horizontal_plane_neon_rows_4_hb_u16,
                    handle_row_impl: convolve_horizontal_plane_neon_u16_hb_row,
                });
            }
        }
        Box::new(HorizontalDefaultHandler {
            weights: weights.clone(),
            handle_row_4_impl: u16::handle_row_4::<CN>,
            handle_row_impl: u16::handle_row::<CN>,
        })
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_horizontal_dispatch_u16<const CN: usize>(
    image_store: &ImageStore<u16, CN>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<u16, CN>,
    pool: &novtb::ThreadPool,
) {
    let src = image_store.buffer.as_ref();
    let dst_stride = destination.stride();
    let dst = destination.buffer.borrow_mut();

    let src_stride = image_store.stride();
    let bit_depth = destination.bit_depth;

    let handler = u16::make_handler::<CN>(&filter_weights, bit_depth);

    dst.tb_par_chunks_exact_mut(dst_stride * 4)
        .for_each_enumerated(pool, |y, dst| {
            let src = &src[y * src_stride * 4..(y + 1) * src_stride * 4];
            handler.handle_row_4(src, src_stride, dst, dst_stride, bit_depth as u32);
        });

    let remainder = dst.chunks_exact_mut(dst_stride * 4).into_remainder();
    let src_remainder = src.chunks_exact(src_stride * 4).remainder();

    remainder
        .tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| {
            let src = &src_remainder[y * src_stride..(y + 1) * src_stride];
            handler.handle_row(src, dst, bit_depth as u32);
        });
}

pub(crate) fn convolve_vertical_dispatch_u16<const CN: usize>(
    image_store: &ImageStore<u16, CN>,
    filter_weights: FilterWeights<f32>,
    destination: &mut ImageStoreMut<'_, u16, CN>,
    pool: &novtb::ThreadPool,
    _options: ConvolutionOptions,
) {
    let src_stride = image_store.stride();
    let dst_stride = destination.stride();
    let bit_depth = destination.bit_depth;

    let dst_width = destination.width;

    let destination_image = destination.buffer.borrow_mut();
    if bit_depth > 12 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
        {
            if DefaultHighBitDepthHighHandlerNeon::is_available() {
                return execute_low_precision_row(
                    image_store,
                    &filter_weights,
                    src_stride,
                    dst_stride,
                    bit_depth,
                    dst_width,
                    destination.buffer.borrow_mut(),
                    DefaultHighBitDepthHighHandlerNeon::default(),
                    WeightsConverterQ0_31::default(),
                    pool,
                );
            }
        }
        destination_image
            .tb_par_chunks_exact_mut(dst_stride)
            .for_each_enumerated(pool, |y, row| {
                let bounds = filter_weights.bounds[y];
                let filter_offset = y * filter_weights.aligned_size;
                let weights = &filter_weights.weights[filter_offset..];
                let source_buffer = image_store.buffer.as_ref();
                u16::handle_floating_column(
                    dst_width,
                    &bounds,
                    source_buffer,
                    &mut row[..dst_width * CN],
                    src_stride,
                    weights,
                    bit_depth as u32,
                );
            });
    } else {
        execute_low_precision_row(
            image_store,
            &filter_weights,
            src_stride,
            dst_stride,
            bit_depth,
            dst_width,
            destination_image,
            DefaultHighBitDepthLowerHandler::default(),
            DefaultWeightsConverter::default(),
            pool,
        );
    }
}

trait HandleVertical<W, const CN: usize> {
    fn handle_fixed_column(
        &self,
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[W],
        bit_depth: u32,
    );
}

#[derive(Default)]
struct DefaultHighBitDepthLowerHandler {}

impl<const CN: usize> HandleVertical<i16, CN> for DefaultHighBitDepthLowerHandler {
    fn handle_fixed_column(
        &self,
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[i16],
        bit_depth: u32,
    ) {
        u16::handle_fixed_column::<i32, CN>(
            dst_width, bounds, src, dst, src_stride, weight, bit_depth,
        );
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
#[derive(Default)]
pub(crate) struct WeightsConverterQ0_31 {}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
impl WeightsConverter<i32> for WeightsConverterQ0_31 {
    fn prepare_weights(&self, weights: &FilterWeights<f32>) -> FilterWeights<i32> {
        weights.numerical_approximation::<i32, 31>(0)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
#[derive(Default)]
struct DefaultHighBitDepthHighHandlerNeon {}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
impl DefaultHighBitDepthHighHandlerNeon {
    fn is_available() -> bool {
        std::arch::is_aarch64_feature_detected!("rdm")
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
impl<const CN: usize> HandleVertical<i32, CN> for DefaultHighBitDepthHighHandlerNeon {
    fn handle_fixed_column(
        &self,
        dst_width: usize,
        bounds: &FilterBounds,
        src: &[u16],
        dst: &mut [u16],
        src_stride: usize,
        weight: &[i32],
        bit_depth: u32,
    ) {
        use crate::neon::convolve_column_hb_u16;
        convolve_column_hb_u16(dst_width, bounds, src, dst, src_stride, weight, bit_depth);
    }
}

#[inline]
fn execute_low_precision_row<W: Send + Sync, const CN: usize>(
    image_store: &ImageStore<u16, CN>,
    filter_weights: &FilterWeights<f32>,
    src_stride: usize,
    dst_stride: usize,
    bit_depth: usize,
    dst_width: usize,
    destination_image: &mut [u16],
    handler: impl HandleVertical<W, CN> + Sync + Send,
    weights: impl WeightsConverter<W>,
    pool: &novtb::ThreadPool,
) {
    let approx = weights.prepare_weights(filter_weights);
    destination_image
        .tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, row| {
            let bounds = filter_weights.bounds[y];
            let filter_offset = y * filter_weights.aligned_size;
            let weights = &approx.weights[filter_offset..];
            let source_buffer = image_store.buffer.as_ref();
            handler.handle_fixed_column(
                dst_width,
                &bounds,
                source_buffer,
                &mut row[..dst_width * CN],
                src_stride,
                weights,
                bit_depth as u32,
            );
        });
}
