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
use crate::floating_point_horizontal::{
    convolve_row_handler_floating_point, convolve_row_handler_floating_point_4,
};

pub(crate) fn convolve_horizontal_native_row_f32<const CHANNELS: usize>(
    _: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    dst: &mut [f32],
) {
    convolve_row_handler_floating_point::<f32, f32, f32, CHANNELS>(src, dst, filter_weights, 8)
}

pub(crate) fn convolve_horizontal_native_row_f32_f64<const CN: usize>(
    _: usize,
    _: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    convolve_row_handler_floating_point::<f32, f64, f64, CN>(src, dst, filter_weights, 8)
}

pub(crate) fn convolve_horizontal_rgba_4_row_f32<const CHANNELS: usize>(
    _: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    convolve_row_handler_floating_point_4::<f32, f32, f32, CHANNELS>(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
        8,
    )
}

pub(crate) fn convolve_horizontal_4_row_f32_f64<const CN: usize>(
    _: usize,
    _: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    convolve_row_handler_floating_point_4::<f32, f64, f64, CN>(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
        8,
    )
}
