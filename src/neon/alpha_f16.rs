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

use std::arch::aarch64::*;

use crate::alpha_handle_f16::{premultiply_pixel_f16_row, unpremultiply_pixel_f16_row};
use crate::neon::f16_utils::*;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;

unsafe fn neon_premultiply_alpha_rgba_row_f16(dst: &mut [half::f16], src: &[half::f16]) {
    let mut rem = dst;
    let mut src_rem = src;

    for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
        let src_ptr = src.as_ptr();
        let pixel = vld4q_u16(src_ptr as *const u16);

        let low_alpha = xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.3)));
        let low_r = vmulq_f32(
            xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.0))),
            low_alpha,
        );
        let low_g = vmulq_f32(
            xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.1))),
            low_alpha,
        );
        let low_b = vmulq_f32(
            xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.2))),
            low_alpha,
        );

        let high_alpha = xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.3)));
        let high_r = vmulq_f32(
            xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.0))),
            high_alpha,
        );
        let high_g = vmulq_f32(
            xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.1))),
            high_alpha,
        );
        let high_b = vmulq_f32(
            xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.2))),
            high_alpha,
        );
        let r_values =
            xreinterpretq_u16_f16(xcombine_f16(xvcvt_f16_f32(low_r), xvcvt_f16_f32(high_r)));
        let g_values =
            xreinterpretq_u16_f16(xcombine_f16(xvcvt_f16_f32(low_g), xvcvt_f16_f32(high_g)));
        let b_values =
            xreinterpretq_u16_f16(xcombine_f16(xvcvt_f16_f32(low_b), xvcvt_f16_f32(high_b)));

        let dst_ptr = dst.as_mut_ptr();
        let store_pixel = uint16x8x4_t(r_values, g_values, b_values, pixel.3);
        vst4q_u16(dst_ptr as *mut u16, store_pixel);
    }

    rem = rem.chunks_exact_mut(8 * 4).into_remainder();
    src_rem = src_rem.chunks_exact(8 * 4).remainder();

    premultiply_pixel_f16_row(rem, src_rem);
}

pub(crate) fn neon_premultiply_alpha_rgba_f16(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            dst.par_chunks_exact_mut(width * 4)
                .zip(src.par_chunks_exact(width * 4))
                .for_each(|(dst, src)| unsafe {
                    neon_premultiply_alpha_rgba_row_f16(dst, src);
                });
        });
    } else {
        dst.chunks_exact_mut(width * 4)
            .zip(src.chunks_exact(width * 4))
            .for_each(|(dst, src)| unsafe {
                neon_premultiply_alpha_rgba_row_f16(dst, src);
            });
    }
}

unsafe fn neon_unpremultiply_alpha_rgba_row_f16(in_place: &mut [half::f16]) {
    let mut rem = in_place;

    for dst in rem.chunks_exact_mut(8 * 4) {
        let src_ptr = dst.as_ptr();
        let pixel = vld4q_u16(src_ptr as *const u16);

        let low_alpha = xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.3)));
        let low_zero_mask = vceqzq_f32(low_alpha);
        let zeros = vdupq_n_f32(0.);

        let low_r = vbslq_f32(
            low_zero_mask,
            zeros,
            vdivq_f32(
                xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.0))),
                low_alpha,
            ),
        );
        let low_g = vbslq_f32(
            low_zero_mask,
            zeros,
            vdivq_f32(
                xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.1))),
                low_alpha,
            ),
        );
        let low_b = vbslq_f32(
            low_zero_mask,
            zeros,
            vdivq_f32(
                xvcvt_f32_f16(xreinterpret_f16_u16(vget_low_u16(pixel.2))),
                low_alpha,
            ),
        );

        let high_alpha = xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.3)));
        let high_zero_mask = vceqzq_f32(high_alpha);

        let high_r = vbslq_f32(
            high_zero_mask,
            zeros,
            vdivq_f32(
                xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.0))),
                high_alpha,
            ),
        );
        let high_g = vbslq_f32(
            high_zero_mask,
            zeros,
            vdivq_f32(
                xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.1))),
                high_alpha,
            ),
        );
        let high_b = vbslq_f32(
            high_zero_mask,
            zeros,
            vdivq_f32(
                xvcvt_f32_f16(xreinterpret_f16_u16(vget_high_u16(pixel.2))),
                high_alpha,
            ),
        );

        let r_values =
            xreinterpretq_u16_f16(xcombine_f16(xvcvt_f16_f32(low_r), xvcvt_f16_f32(high_r)));
        let g_values =
            xreinterpretq_u16_f16(xcombine_f16(xvcvt_f16_f32(low_g), xvcvt_f16_f32(high_g)));
        let b_values =
            xreinterpretq_u16_f16(xcombine_f16(xvcvt_f16_f32(low_b), xvcvt_f16_f32(high_b)));

        let dst_ptr = dst.as_mut_ptr();
        let store_pixel = uint16x8x4_t(r_values, g_values, b_values, pixel.3);
        vst4q_u16(dst_ptr as *mut u16, store_pixel);
    }

    rem = rem.chunks_exact_mut(8 * 4).into_remainder();

    unpremultiply_pixel_f16_row(rem);
}

pub(crate) fn neon_unpremultiply_alpha_rgba_f16(
    in_place: &mut [half::f16],
    stride: usize,
    width: usize,
    _: usize,
    pool: &Option<ThreadPool>,
) {
    if let Some(pool) = pool {
        pool.install(|| {
            in_place
                .par_chunks_exact_mut(stride)
                .for_each(|row| unsafe {
                    neon_unpremultiply_alpha_rgba_row_f16(&mut row[..width * 4]);
                });
        });
    } else {
        in_place.chunks_exact_mut(stride).for_each(|row| unsafe {
            neon_unpremultiply_alpha_rgba_row_f16(&mut row[..width * 4]);
        });
    }
}
