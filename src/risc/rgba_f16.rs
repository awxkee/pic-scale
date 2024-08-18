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
use half::f16;
use std::arch::asm;

#[target_feature(enable = "v,zfh")]
unsafe fn convolve_horizontal_rgba_risc_row_one_f16_impl(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    unsafe_destination_ptr_0: *mut f16,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let bounds_start = bounds.start;
            let bounds_size = bounds.size;
            let local_filter_ptr = weights_ptr.add(filter_offset);
            asm!(include_str!("horiz_rgba_n1_f16.asm"),
                 in(reg) local_filter_ptr,
                 in(reg) bounds_start,
                 in(reg) bounds_size,
                 in(reg) unsafe_source_ptr_0,
                 in(reg) unsafe_destination_ptr_0,
                 in(reg) x,
                 t1 = out(reg) _,
                 ft1 = out(freg) _,
                 t2 = out(reg) _,
                 ft2 = out(freg) _,
                 t4 = out(reg) _,
                 t5 = out(reg) _,
                 t6 = out(reg) _,
                 out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgba_risc_row_one_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    unsafe_destination_ptr_0: *mut f16,
) {
    unsafe {
        convolve_horizontal_rgba_risc_row_one_f16_impl(
            dst_width,
            src_width,
            filter_weights,
            unsafe_source_ptr_0,
            unsafe_destination_ptr_0,
        );
    }
}

#[target_feature(enable = "v,zfh")]
unsafe fn convolve_horizontal_rgba_risc_rows_4_impl_f16(
    dst_width: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f16,
    dst_stride: usize,
) {
    unsafe {
        let mut filter_offset = 0usize;
        let weights_ptr = filter_weights.weights.as_ptr();

        let real_src_stride = src_stride * std::mem::size_of::<u16>();
        let real_dst_stride = dst_stride * std::mem::size_of::<u16>();

        for x in 0..dst_width {
            let bounds = filter_weights.bounds.get_unchecked(x);
            let bounds_start = bounds.start;
            let bounds_size = bounds.size;
            let local_filter_ptr = weights_ptr.add(filter_offset);
            asm!(include_str!("horiz_rgba_n4_f16.asm"),
                 in(reg) local_filter_ptr,
                 in(reg) bounds_start,
                 in(reg) bounds_size,
                 in(reg) unsafe_source_ptr_0,
                 in(reg) unsafe_destination_ptr_0,
                 in(reg) x,
                 in(reg) real_src_stride,
                 in(reg) real_dst_stride,
                 t1 = out(reg) _,
                 ft1 = out(freg) _,
                 t2 = out(reg) _,
                 ft2 = out(freg) _,
                 t3 = out(reg) _,
                 t4 = out(reg) _,
                 t5 = out(reg) _,
                 t6 = out(reg) _,
                 out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                 out("v7") _, out("v8") _, out("v9") _);

            filter_offset += filter_weights.aligned_size;
        }
    }
}

pub fn convolve_horizontal_rgba_risc_rows_4_f16(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f32>,
    unsafe_source_ptr_0: *const f16,
    src_stride: usize,
    unsafe_destination_ptr_0: *mut f16,
    dst_stride: usize,
) {
    unsafe {
        convolve_horizontal_rgba_risc_rows_4_impl_f16(
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
