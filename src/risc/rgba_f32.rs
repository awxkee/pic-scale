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
use crate::risc::xvsetvlmax_f32m1_k;
use std::arch::asm;

#[target_feature(enable = "v")]
unsafe fn convolve_horizontal_rgba_risc_row_one_f32_impl(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let bounds_start = bounds.start;
            let bounds_size = bounds.size;
            let local_filter_ptr = weights_ptr.add(filter_offset);
            asm!(include_str!("horiz_rgba_n1_f32.asm"),
                 in(reg) local_filter_ptr,
                 in(reg) bounds_start,
                 in(reg) bounds_size,
                 in(reg) unsafe_source_ptr_0,
                 in(reg) unsafe_destination_ptr_0,
                 in(reg) x);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgba_risc_row_one_f32(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
) {
    unsafe {
        convolve_horizontal_rgba_risc_row_one_f32_impl(
            dst_width,
            src_width,
            filter_weights,
            unsafe_source_ptr_0,
            unsafe_destination_ptr_0,
        );
    }
}

#[target_feature(enable = "v")]
unsafe fn convolve_horizontal_rgba_risc_rows_4_impl(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f32,
    dst_stride: usize,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        let real_src_stride = src_stride * 4;
        let real_dst_stride = dst_stride * 4;

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let bounds_start = bounds.start;
            let bounds_size = bounds.size;
            let local_filter_ptr = weights_ptr.add(filter_offset);
            asm!(include_str!("horiz_rgba_n4_f32.asm"),
                 in(reg) local_filter_ptr,
                 in(reg) bounds_start,
                 in(reg) bounds_size,
                 in(reg) unsafe_source_ptr_0,
                 in(reg) unsafe_destination_ptr_0,
                 in(reg) x,
                 in(reg) real_src_stride,
                 in(reg) real_dst_stride);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgba_risc_rows_4(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f32,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f32,
    dst_stride: usize,
) {
    unsafe {
        convolve_horizontal_rgba_risc_rows_4_impl(
            dst_width,
            src_width,
            filter_weights,
            unsafe_source_ptr_0,
            src_stride,
            unsafe_destination_ptr_0,
            dst_stride,
        );
    }
}