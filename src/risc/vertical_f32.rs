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
    weight_ptr: &[f32],
) {
    unsafe {
        convolve_vertical_rgb_risc_row_f32_impl::<CHANNELS>(
            width,
            bounds,
            unsafe_source_ptr_0,
            unsafe_destination_ptr_0,
            src_stride,
            weight_ptr,
        );
    }
}

#[inline]
#[target_feature(enable = "v")]
unsafe fn convolve_vertical_rgb_risc_row_f32_impl<const CHANNELS: usize>(
    width: usize,
    bounds: &FilterBounds,
    unsafe_source_ptr_0: *const f32,
    unsafe_destination_ptr_0: *mut f32,
    src_stride: usize,
    weight_ptr: &[f32],
) {
    let mut cx = 0usize;
    let dst_width = width * CHANNELS;

    let lane_length = unsafe { xvsetvlmax_f32m1() };
    let double_length = 2 * lane_length;

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
                 in(reg) weight_ptr.as_ptr(),
                 in(reg) bounds_size,
                 t1 = out(reg) _,
                 ft1 = out(freg) _,
                 t2 = out(reg) _,
                 t3 = out(reg) _,
                 t4 = out(reg) _,
                 t5 = out(reg) _,
                 t6 = out(reg) _,
                 out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _);
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
                 in(reg) weight_ptr.as_ptr(),
                 in(reg) bounds_size,
                 t1 = out(reg) _,
                 ft1 = out(freg) _,
                 t2 = out(reg) _,
                 t3 = out(reg) _,
                 t4 = out(reg) _,
                 t5 = out(reg) _,
                 t6 = out(reg) _,
                 out("v1") _, out("v2") _, out("v3") _);
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
                 in(reg) weight_ptr.as_ptr(),
                 in(reg) bounds_size,
                 t1 = out(reg) _,
                 ft1 = out(freg) _,
                 t2 = out(reg) _,
                 ft2 = out(freg) _,
                 t3 = out(reg) _,
                 ft3 = out(freg) _,
                 t4 = out(reg) _,
                 t5 = out(reg) _,
                 t6 = out(reg) _);
        }
        cx += 1;
    }
}
