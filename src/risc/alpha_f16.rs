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

use crate::risc::xvsetvlmax_f16m1;
use crate::{premultiply_pixel_f16, unpremultiply_pixel_f16, ThreadingPolicy};
use half::f16;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;
use std::arch::asm;

#[target_feature(enable = "v")]
unsafe fn risc_premultiply_alpha_rgba_f16_row_impl(
    dst: &mut [f16],
    src: &[f16],
    width: usize,
    offset: usize,
) {
    let mut _cx = 0usize;

    unsafe {
        let iter_width = xvsetvlmax_f16m1();
        while _cx + iter_width < width {
            let px = _cx * 4;
            let src_ptr = src.as_ptr().add(offset + px);
            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            asm!(include_str!("premultiply_alpha_f16.asm"),
                     in(reg) src_ptr,
                     in(reg) dst_ptr,
                     t1 = out(reg) _,
                     out("v0") _, out("v1") _,
                     out("v2") _, out("v3") _);
            _cx += iter_width;
        }
    }

    for x in _cx..width {
        let px = x * 4;
        premultiply_pixel_f16!(dst, src, offset + px);
    }
}

#[target_feature(enable = "v")]
unsafe fn risc_premultiply_alpha_rgba_f16_impl(
    dst: &mut [f16],
    src: &[f16],
    width: usize,
    height: usize,
    threading_policy: ThreadingPolicy,
) {
    let allowed_threading = threading_policy.allowed_threading();

    if allowed_threading {
        src.par_chunks_exact(width * 4)
            .zip(dst.par_chunks_exact_mut(width * 4))
            .for_each(|(src, dst)| unsafe {
                risc_premultiply_alpha_rgba_f16_row_impl(dst, src, width, 0);
            });
    } else {
        let mut offset = 0usize;

        for _ in 0..height {
            risc_premultiply_alpha_rgba_f16_row_impl(dst, src, width, offset);

            offset += 4 * width;
        }
    }
}

pub fn risc_premultiply_alpha_rgba_f16(
    dst: &mut [f16],
    src: &[f16],
    width: usize,
    height: usize,
    threading_policy: ThreadingPolicy,
) {
    unsafe {
        risc_premultiply_alpha_rgba_f16_impl(dst, src, width, height, threading_policy);
    }
}

#[target_feature(enable = "v")]
unsafe fn risc_unpremultiply_alpha_rgba_f16_row_impl(
    dst: &mut [f16],
    src: &[f16],
    width: usize,
    offset: usize,
) {
    let mut _cx = 0usize;

    unsafe {
        let iter_width = xvsetvlmax_f16m1();
        while _cx + iter_width < width {
            let px = _cx * 4;
            let src_ptr = src.as_ptr().add(offset + px);
            let dst_ptr = dst.as_mut_ptr().add(offset + px);
            asm!(include_str!("unpremultiply_alpha_f16.asm"), in(reg) src_ptr,
                     in(reg) dst_ptr,
                     t1 = out(reg) _,
                     ft1 = out(freg) _,
                     a7 = out(reg) _,
                     out("v0") _,
                     out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                     out("v7") _, out("v8") _);
            _cx += iter_width;
        }
    }

    for x in _cx..width {
        let px = x * 4;
        let pixel_offset = offset + px;
        unpremultiply_pixel_f16!(dst, src, pixel_offset);
    }
}

#[target_feature(enable = "v")]
unsafe fn risc_unpremultiply_alpha_rgba_f16_impl(
    dst: &mut [f16],
    src: &[f16],
    width: usize,
    height: usize,
    threading_policy: ThreadingPolicy,
) {
    let allowed_threading = threading_policy.allowed_threading();

    if allowed_threading {
        src.par_chunks_exact(width * 4)
            .zip(dst.par_chunks_exact_mut(width * 4))
            .for_each(|(src, dst)| unsafe {
                risc_unpremultiply_alpha_rgba_f16_row_impl(dst, src, width, 0);
            });
    } else {
        let mut offset = 0usize;

        for _ in 0..height {
            unsafe {
                risc_unpremultiply_alpha_rgba_f16_row_impl(dst, src, width, offset);
            }
            offset += 4 * width;
        }
    }
}

pub fn risc_unpremultiply_alpha_rgba_f16(
    dst: &mut [f16],
    src: &[f16],
    width: usize,
    height: usize,
    threading_policy: ThreadingPolicy,
) {
    unsafe {
        risc_unpremultiply_alpha_rgba_f16_impl(dst, src, width, height, threading_policy);
    }
}
