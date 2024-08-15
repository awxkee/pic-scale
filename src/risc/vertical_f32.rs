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
use crate::filter_weights::FilterBounds;
use crate::risc::xvsetvlmax_f32m1;
use std::arch::asm;

pub fn convolve_vertical_rgb_risc_row_f32<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    src_stride: usize,
    weight_ptr: *const f32,
) {
    let mut cx = 0usize;
    let dst_width = width * CHANNELS;

    let lane_length = unsafe { xvsetvlmax_f32m1() };
    let double_length = 2 * lane_length;
    let triple_length = 3 * lane_length;

    while cx + triple_length < dst_width {
        unsafe {
            let bnd_start = bounds.start;
            let bounds_size = bounds.size;
            asm!(include_str!("vert_n_m_3_f32.asm"),
                 in(reg) bnd_start,
                 in(reg) cx,
                 in(reg) unsafe_source_ptr_0,
                 in(reg) src_stride,
                 in(reg) unsafe_destination_ptr_0,
                 in(reg) weight_ptr,
                 in(reg) bounds_size);
        }
        cx += triple_length;
    }

    while cx + double_length < dst_width {
        unsafe {
            let bnd_start = bounds.start;
            let bounds_size = bounds.size;
            asm!(include_str!("vert_n_m_2_f32.asm"),
                 in(reg) bnd_start,
                 in(reg) cx,
                 in(reg) unsafe_source_ptr_0,
                 in(reg) src_stride,
                 in(reg) unsafe_destination_ptr_0,
                 in(reg) weight_ptr,
                 in(reg) bounds_size);
        }
        cx += double_length;
    }

    while cx + lane_length < dst_width {
        unsafe {
            let bnd_start = bounds.start;
            let bounds_size = bounds.size;
            asm!(include_str!("vert_n_f32.asm"),
                 in(reg) bnd_start,
                 in(reg) cx,
                 in(reg) unsafe_source_ptr_0,
                 in(reg) src_stride,
                 in(reg) unsafe_destination_ptr_0,
                 in(reg) weight_ptr,
                 in(reg) bounds_size);
        }
        cx += lane_length;
    }

    while cx < dst_width {
        unsafe {
            let bnd_start = bounds.start;
            let bounds_size = bounds.size;
            asm!(include_str!("vert_1_f32.asm"),
                 in(reg) bnd_start,
                 in(reg) cx,
                 in(reg) unsafe_source_ptr_0,
                 in(reg) src_stride,
                 in(reg) unsafe_destination_ptr_0,
                 in(reg) weight_ptr,
                 in(reg) bounds_size);
        }
        cx += 1;
    }
}
