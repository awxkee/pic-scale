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

use crate::risc::xvsetvlmax_e8m1;
use crate::{premultiply_pixel, unpremultiply_pixel};
use std::arch::asm;

#[target_feature(enable = "v")]
unsafe fn risc_premultiply_alpha_rgba_u8_impl(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;
    let src_stride = 4 * width;

    let mut offset = _cy * src_stride;

    for _ in _cy..height {
        let mut _cx = 0usize;

        let iter_width = xvsetvlmax_e8m1();
        while _cx + iter_width < width {
            let px = _cx * 4;
            let src_ptr = src.as_ptr().add(offset + px);
            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            asm!(include_str!("premultiply_alpha_u8.asm"),
                     in(reg) src_ptr,
                     in(reg) dst_ptr,
                     t0 = out(reg) _,
                     t1 = out(reg) _,
                     t4 = out(reg) _,
                     out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                     out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                     out("v12") _, out("v13") _);
            _cx += iter_width;
        }

        for x in _cx..width {
            let px = x * 4;
            premultiply_pixel!(dst, src, offset + px);
        }

        offset += 4 * width;
    }
}

pub fn risc_premultiply_alpha_rgba_u8(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
    unsafe {
        risc_premultiply_alpha_rgba_u8_impl(dst, src, width, height);
    }
}

#[target_feature(enable = "v")]
unsafe fn risc_unpremultiply_alpha_rgba_u8_impl(
    dst: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
) {
    let mut _cy = 0usize;

    let mut offset = 0usize;
    offset += _cy * width * 4;

    let iter_width = xvsetvlmax_e8m1();

    for _ in _cy..height {
        let mut _cx = 0usize;

        while _cx + iter_width < width {
            let px = _cx * 4;
            let src_ptr = src.as_ptr().add(offset + px);
            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            asm!(include_str!("unpremultiply_alpha_u8.asm"),
                in(reg) src_ptr,
                in(reg) dst_ptr,
                in(reg) iter_width,
                t4 = out(reg) _,
                t5 = out(reg) _,
                out("v0") _,
                out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _);
            _cx += iter_width;
        }

        for x in _cx..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel!(dst, src, pixel_offset);
        }

        offset += 4 * width;
    }
}

pub fn risc_unpremultiply_alpha_rgba_u8(dst: &mut [u8], src: &[u8], width: usize, height: usize) {
    unsafe {
        risc_unpremultiply_alpha_rgba_u8_impl(dst, src, width, height);
    }
}