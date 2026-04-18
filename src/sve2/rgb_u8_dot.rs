/*
 * Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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
use std::arch::aarch64::*;

pub(crate) fn sve_convolve_horizontal_rgb_neon_rows_4_dot(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_neon_rows_4_impl(src, src_stride, dst, dst_stride, filter_weights);
    }
}

macro_rules! pack_s32x4_to_u8 {
    ($a:expr) => {{
        let n = svqshrunb_n_s32::<7>($a);
        let n_packed = svuzp1_u16(n, n);
        let b = svqxtnb_u16(n_packed);
        svuzp1_u8(b, b)
    }};
}

#[target_feature(enable = "sve2,sve,i8mm")]
fn convolve_horizontal_rgb_neon_rows_4_impl(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i8>,
) {
    static TBL0: [u8; 16] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255];
    let v_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), TBL0.as_ptr()) };
    let v_weights = svreinterpret_u8_u32(svdup_n_u32(u32::from_ne_bytes([0, 1, 2, 3])));

    // (r0 g0 b0 r1) (g2 b2 r3 g3) (b3 r4 g4 b4) (r5 g5 b5 r6)

    let rnd_const: i32 = 1 << 6;

    const CN: usize = 3;
    let init = svdup_n_s32(rnd_const);
    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.as_chunks_mut::<CN>().0;
    let iter_row1 = row1_ref.as_chunks_mut::<CN>().0;
    let iter_row2 = row2_ref.as_chunks_mut::<CN>().0;
    let iter_row3 = row3_ref.as_chunks_mut::<CN>().0;

    let pg4 = svwhilelt_b8_u32(0u32, 4u32);
    let pg3 = svwhilelt_b8_u32(0u32, 3u32);
    let pg12 = svwhilelt_b8_u32(0u32, 12u32);

    for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
        .iter_mut()
        .zip(iter_row1.iter_mut())
        .zip(iter_row2.iter_mut())
        .zip(iter_row3.iter_mut())
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;
        let mut store_0 = init;
        let mut store_1 = init;
        let mut store_2 = init;
        let mut store_3 = init;

        let src0 = src;
        let src1 = unsafe { src0.get_unchecked(src_stride..) };
        let src2 = unsafe { src1.get_unchecked(src_stride..) };
        let src3 = unsafe { src2.get_unchecked(src_stride..) };

        while jx + 4 <= bounds.size {
            let bounds_start = bounds.start + jx;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let vw = svtbl_s8(unsafe { svld1_s8(pg4, w_ptr.as_ptr()) }, v_weights);

            let rgb_pixel0 = unsafe { svld1_u8(pg12, src0.get_unchecked(bounds_start * CN)) };
            let rgb_pixel1 = unsafe { svld1_u8(pg12, src1.get_unchecked(bounds_start * CN)) };
            let rgb_pixel2 = unsafe { svld1_u8(pg12, src2.get_unchecked(bounds_start * CN)) };
            let rgb_pixel3 = unsafe { svld1_u8(pg12, src3.get_unchecked(bounds_start * CN)) };
            store_0 = svusdot_s32(store_0, svtbl_u8(rgb_pixel0, v_tbl), vw);
            store_1 = svusdot_s32(store_1, svtbl_u8(rgb_pixel1, v_tbl), vw);
            store_2 = svusdot_s32(store_2, svtbl_u8(rgb_pixel2, v_tbl), vw);
            store_3 = svusdot_s32(store_3, svtbl_u8(rgb_pixel3, v_tbl), vw);
            jx += 4;
        }

        if jx < bounds.size {
            let pq = svwhilelt_b8_u64(jx as u64, bounds.size as u64);
            let pqb = svwhilelt_b8_u64(0u64, (bounds.size - jx) as u64 * 3);

            let bounds_start = bounds.start + jx;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let vw = svtbl_s8(unsafe { svld1_s8(pq, w_ptr.as_ptr()) }, v_weights);

            let rgb_pixel0 = unsafe { svld1_u8(pqb, src0.get_unchecked(bounds_start * CN)) };
            let rgb_pixel1 = unsafe { svld1_u8(pqb, src1.get_unchecked(bounds_start * CN)) };
            let rgb_pixel2 = unsafe { svld1_u8(pqb, src2.get_unchecked(bounds_start * CN)) };
            let rgb_pixel3 = unsafe { svld1_u8(pqb, src3.get_unchecked(bounds_start * CN)) };
            store_0 = svusdot_s32(store_0, svtbl_u8(rgb_pixel0, v_tbl), vw);
            store_1 = svusdot_s32(store_1, svtbl_u8(rgb_pixel1, v_tbl), vw);
            store_2 = svusdot_s32(store_2, svtbl_u8(rgb_pixel2, v_tbl), vw);
            store_3 = svusdot_s32(store_3, svtbl_u8(rgb_pixel3, v_tbl), vw);
        }

        let v0 = pack_s32x4_to_u8!(store_0);
        let v1 = pack_s32x4_to_u8!(store_1);
        let v2 = pack_s32x4_to_u8!(store_2);
        let v3 = pack_s32x4_to_u8!(store_3);
        unsafe {
            svst1_u8(pg3, chunk0.as_mut_ptr(), v0);
            svst1_u8(pg3, chunk1.as_mut_ptr(), v1);
            svst1_u8(pg3, chunk2.as_mut_ptr(), v2);
            svst1_u8(pg3, chunk3.as_mut_ptr(), v3);
        }
    }
}

pub(crate) fn sve_convolve_horizontal_rgb_neon_row_one_dot(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
    _: u32,
) {
    unsafe {
        convolve_horizontal_rgb_neon_row_one_impl_dot(src, dst, filter_weights);
    }
}

#[target_feature(enable = "sve2,sve,i8mm")]
fn convolve_horizontal_rgb_neon_row_one_impl_dot(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i8>,
) {
    const CN: usize = 3;

    static TBL0: [u8; 16] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255];
    let v_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), TBL0.as_ptr()) };
    let v_weights = svreinterpret_u8_u32(svdup_n_u32(u32::from_ne_bytes([0, 1, 2, 3])));

    let rnd_const: i32 = 1 << 6;
    let init = svdup_n_s32(rnd_const);

    let pg4 = svwhilelt_b8_u32(0u32, 4u32);
    let pg3 = svwhilelt_b8_u32(0u32, 3u32);
    let pg12 = svwhilelt_b8_u32(0u32, 12u32);

    for ((dst, bounds), weights) in dst
        .as_chunks_mut::<CN>()
        .0
        .iter_mut()
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;
        let mut store_0 = init;

        while jx + 4 <= bounds.size {
            let bounds_start = bounds.start + jx;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let vw = svtbl_s8(unsafe { svld1_s8(pg4, w_ptr.as_ptr()) }, v_weights);

            let rgb_pixel0 = unsafe { svld1_u8(pg12, src.get_unchecked(bounds_start * CN)) };
            store_0 = svusdot_s32(store_0, svtbl_u8(rgb_pixel0, v_tbl), vw);
            jx += 4;
        }

        if jx < bounds.size {
            let pq = svwhilelt_b8_u64(jx as u64, bounds.size as u64);
            let pqb = svwhilelt_b8_u64(0u64, (bounds.size - jx) as u64 * 3);

            let bounds_start = bounds.start + jx;
            let w_ptr = unsafe { weights.get_unchecked(jx..) };
            let vw = svtbl_s8(unsafe { svld1_s8(pq, w_ptr.as_ptr()) }, v_weights);

            let rgb_pixel0 = unsafe { svld1_u8(pqb, src.get_unchecked(bounds_start * CN)) };
            store_0 = svusdot_s32(store_0, svtbl_u8(rgb_pixel0, v_tbl), vw);
        }

        let v0 = pack_s32x4_to_u8!(store_0);
        unsafe {
            svst1_u8(pg3, dst.as_mut_ptr(), v0);
        }
    }
}
