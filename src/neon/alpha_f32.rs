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

#[target_feature(enable = "neon")]
fn neon_premultiply_alpha_rgba_row_f32(dst: &mut [f32], src: &[f32]) {
    let mut rem = dst;
    let mut src_rem = src;

    for (dst, src) in rem
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(src_rem.as_chunks::<16>().0.iter())
    {
        let pixel0 = unsafe { vld1q_f32(src.as_ptr().cast()) };
        let pixel1 = unsafe { vld1q_f32(src[4..].as_ptr().cast()) };
        let pixel2 = unsafe { vld1q_f32(src[8..].as_ptr().cast()) };
        let pixel3 = unsafe { vld1q_f32(src[12..].as_ptr().cast()) };

        let mut new_px0 = vmulq_f32(pixel0, vdupq_laneq_f32::<3>(pixel0));
        new_px0 = vcopyq_laneq_f32::<3, 3>(new_px0, pixel0);

        let mut new_px1 = vmulq_f32(pixel1, vdupq_laneq_f32::<3>(pixel1));
        new_px1 = vcopyq_laneq_f32::<3, 3>(new_px1, pixel1);

        let mut new_px2 = vmulq_f32(pixel2, vdupq_laneq_f32::<3>(pixel2));
        new_px2 = vcopyq_laneq_f32::<3, 3>(new_px2, pixel2);

        let mut new_px3 = vmulq_f32(pixel3, vdupq_laneq_f32::<3>(pixel3));
        new_px3 = vcopyq_laneq_f32::<3, 3>(new_px3, pixel3);

        unsafe {
            vst1q_f32(dst.as_mut_ptr().cast(), new_px0);
            vst1q_f32(dst[4..].as_mut_ptr().cast(), new_px1);
            vst1q_f32(dst[8..].as_mut_ptr().cast(), new_px2);
            vst1q_f32(dst[12..].as_mut_ptr().cast(), new_px3);
        }
    }

    rem = rem.as_chunks_mut::<16>().1;
    src_rem = src.as_chunks::<16>().1;

    for (dst, src) in rem
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(src_rem.as_chunks::<4>().0.iter())
    {
        let pixel = unsafe { vld1q_f32(src.as_ptr().cast()) };
        let mut new_px = vmulq_f32(pixel, vdupq_laneq_f32::<3>(pixel));
        new_px = vcopyq_laneq_f32::<3, 3>(new_px, pixel);
        unsafe {
            vst1q_f32(dst.as_mut_ptr().cast(), new_px);
        }
    }
}

pub(crate) fn neon_premultiply_alpha_rgba_f32(dst: &mut [f32], src: &[f32]) {
    unsafe {
        neon_premultiply_alpha_rgba_row_f32(dst, src);
    }
}

#[target_feature(enable = "neon")]
fn neon_unpremultiply_alpha_rgba_f32_row(in_place: &mut [f32]) {
    let mut rem = in_place;

    for dst in rem.as_chunks_mut::<16>().0.iter_mut() {
        let pixel0 = unsafe { vld1q_f32(dst.as_ptr().cast()) };
        let pixel1 = unsafe { vld1q_f32(dst[4..].as_ptr().cast()) };
        let pixel2 = unsafe { vld1q_f32(dst[8..].as_ptr().cast()) };
        let pixel3 = unsafe { vld1q_f32(dst[12..].as_ptr().cast()) };

        let a_values0 = vdupq_laneq_f32::<3>(pixel0);
        let a_values1 = vdupq_laneq_f32::<3>(pixel1);
        let a_values2 = vdupq_laneq_f32::<3>(pixel2);
        let a_values3 = vdupq_laneq_f32::<3>(pixel3);

        let is_zero_mask0 = vceqzq_f32(a_values0);
        let is_zero_mask1 = vceqzq_f32(a_values1);
        let is_zero_mask2 = vceqzq_f32(a_values2);
        let is_zero_mask3 = vceqzq_f32(a_values3);

        let a_values_full = vreinterpretq_f32_f64(vzip1q_f64(
            vreinterpretq_f64_f32(vzip1q_f32(a_values0, a_values1)),
            vreinterpretq_f64_f32(vzip1q_f32(a_values2, a_values3)),
        ));

        let recip = vdivq_f32(vdupq_n_f32(1.), a_values_full);

        let mut new_px0 = vmulq_laneq_f32::<0>(pixel0, recip);
        new_px0 = vbslq_f32(is_zero_mask0, vdupq_n_f32(0.), new_px0);
        new_px0 = vcopyq_laneq_f32::<3, 3>(new_px0, pixel0);

        let mut new_px1 = vmulq_laneq_f32::<1>(pixel1, recip);
        new_px1 = vbslq_f32(is_zero_mask1, vdupq_n_f32(0.), new_px1);
        new_px1 = vcopyq_laneq_f32::<3, 3>(new_px1, pixel1);

        let mut new_px2 = vmulq_laneq_f32::<2>(pixel2, recip);
        new_px2 = vbslq_f32(is_zero_mask2, vdupq_n_f32(0.), new_px2);
        new_px2 = vcopyq_laneq_f32::<3, 3>(new_px2, pixel2);

        let mut new_px3 = vmulq_laneq_f32::<3>(pixel3, recip);
        new_px3 = vbslq_f32(is_zero_mask3, vdupq_n_f32(0.), new_px3);
        new_px3 = vcopyq_laneq_f32::<3, 3>(new_px3, pixel3);

        unsafe {
            vst1q_f32(dst.as_mut_ptr().cast(), new_px0);
            vst1q_f32(dst[4..].as_mut_ptr().cast(), new_px1);
            vst1q_f32(dst[8..].as_mut_ptr().cast(), new_px2);
            vst1q_f32(dst[12..].as_mut_ptr().cast(), new_px3);
        }
    }

    rem = rem.as_chunks_mut::<16>().1;

    for dst in rem.as_chunks_mut::<4>().0.iter_mut() {
        let pixel = unsafe { vld1q_f32(dst.as_ptr().cast()) };
        let a_values = vdupq_laneq_f32::<3>(pixel);
        let is_zero_mask = vceqzq_f32(a_values);
        let recip = vdivq_f32(vdupq_n_f32(1.), a_values);
        let mut new_px = vmulq_f32(pixel, recip);
        new_px = vbslq_f32(is_zero_mask, vdupq_n_f32(0.), new_px);
        new_px = vcopyq_laneq_f32::<3, 3>(new_px, pixel);
        unsafe {
            vst1q_f32(dst.as_mut_ptr().cast(), new_px);
        }
    }
}

pub(crate) fn neon_unpremultiply_alpha_rgba_f32(in_place: &mut [f32]) {
    unsafe { neon_unpremultiply_alpha_rgba_f32_row(in_place) }
}
