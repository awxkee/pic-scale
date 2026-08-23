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
use crate::neon::utils::{prefer_vfmaq_f32, xvld1q_f32_x2};
use crate::neon::utils::{xvld1q_f32_x4, xvst1q_f32_x2, xvst1q_f32_x4};
use std::arch::aarch64::*;

fn conv_vertical_part_neon_32_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);
        let mut store_1 = vdupq_n_f32(0.);
        let mut store_2 = vdupq_n_f32(0.);
        let mut store_3 = vdupq_n_f32(0.);

        let mut store_4 = vdupq_n_f32(0.);
        let mut store_5 = vdupq_n_f32(0.);
        let mut store_6 = vdupq_n_f32(0.);
        let mut store_7 = vdupq_n_f32(0.);

        let px = start_x;

        let mut j = 0usize;
        while j + 4 <= bounds.size {
            let py = start_y + j;
            let weights = vld1q_f32(filter.get_unchecked(j..).as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);

            let item_row_0 = xvld1q_f32_x4(src_ptr.as_ptr());
            let item_row_1 = xvld1q_f32_x4(src_ptr.get_unchecked(16..).as_ptr());

            store_0 = vfmaq_laneq_f32::<0>(store_0, item_row_0.0, weights);
            store_1 = vfmaq_laneq_f32::<0>(store_1, item_row_0.1, weights);
            store_2 = vfmaq_laneq_f32::<0>(store_2, item_row_0.2, weights);
            store_3 = vfmaq_laneq_f32::<0>(store_3, item_row_0.3, weights);

            store_4 = vfmaq_laneq_f32::<0>(store_4, item_row_1.0, weights);
            store_5 = vfmaq_laneq_f32::<0>(store_5, item_row_1.1, weights);
            store_6 = vfmaq_laneq_f32::<0>(store_6, item_row_1.2, weights);
            store_7 = vfmaq_laneq_f32::<0>(store_7, item_row_1.3, weights);

            let item_row_0 = xvld1q_f32_x4(src_ptr.get_unchecked(src_stride..).as_ptr());
            let item_row_1 = xvld1q_f32_x4(src_ptr.get_unchecked(src_stride + 16..).as_ptr());

            store_0 = vfmaq_laneq_f32::<1>(store_0, item_row_0.0, weights);
            store_1 = vfmaq_laneq_f32::<1>(store_1, item_row_0.1, weights);
            store_2 = vfmaq_laneq_f32::<1>(store_2, item_row_0.2, weights);
            store_3 = vfmaq_laneq_f32::<1>(store_3, item_row_0.3, weights);

            store_4 = vfmaq_laneq_f32::<1>(store_4, item_row_1.0, weights);
            store_5 = vfmaq_laneq_f32::<1>(store_5, item_row_1.1, weights);
            store_6 = vfmaq_laneq_f32::<1>(store_6, item_row_1.2, weights);
            store_7 = vfmaq_laneq_f32::<1>(store_7, item_row_1.3, weights);

            let item_row_0 = xvld1q_f32_x4(src_ptr.get_unchecked(src_stride * 2..).as_ptr());
            let item_row_1 = xvld1q_f32_x4(src_ptr.get_unchecked(src_stride * 2 + 16..).as_ptr());

            store_0 = vfmaq_laneq_f32::<2>(store_0, item_row_0.0, weights);
            store_1 = vfmaq_laneq_f32::<2>(store_1, item_row_0.1, weights);
            store_2 = vfmaq_laneq_f32::<2>(store_2, item_row_0.2, weights);
            store_3 = vfmaq_laneq_f32::<2>(store_3, item_row_0.3, weights);

            store_4 = vfmaq_laneq_f32::<2>(store_4, item_row_1.0, weights);
            store_5 = vfmaq_laneq_f32::<2>(store_5, item_row_1.1, weights);
            store_6 = vfmaq_laneq_f32::<2>(store_6, item_row_1.2, weights);
            store_7 = vfmaq_laneq_f32::<2>(store_7, item_row_1.3, weights);

            let item_row_0 = xvld1q_f32_x4(src_ptr.get_unchecked(src_stride * 3..).as_ptr());
            let item_row_1 = xvld1q_f32_x4(src_ptr.get_unchecked(src_stride * 3 + 16..).as_ptr());

            store_0 = vfmaq_laneq_f32::<3>(store_0, item_row_0.0, weights);
            store_1 = vfmaq_laneq_f32::<3>(store_1, item_row_0.1, weights);
            store_2 = vfmaq_laneq_f32::<3>(store_2, item_row_0.2, weights);
            store_3 = vfmaq_laneq_f32::<3>(store_3, item_row_0.3, weights);

            store_4 = vfmaq_laneq_f32::<3>(store_4, item_row_1.0, weights);
            store_5 = vfmaq_laneq_f32::<3>(store_5, item_row_1.1, weights);
            store_6 = vfmaq_laneq_f32::<3>(store_6, item_row_1.2, weights);
            store_7 = vfmaq_laneq_f32::<3>(store_7, item_row_1.3, weights);

            j += 4;
        }

        while j + 2 <= bounds.size {
            let py = start_y + j;
            let weights = vld1_f32(filter.get_unchecked(j..).as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);

            let item_row_0 = xvld1q_f32_x4(src_ptr.as_ptr());
            let item_row_1 = xvld1q_f32_x4(src_ptr.get_unchecked(16..).as_ptr());

            store_0 = vfmaq_lane_f32::<0>(store_0, item_row_0.0, weights);
            store_1 = vfmaq_lane_f32::<0>(store_1, item_row_0.1, weights);
            store_2 = vfmaq_lane_f32::<0>(store_2, item_row_0.2, weights);
            store_3 = vfmaq_lane_f32::<0>(store_3, item_row_0.3, weights);

            store_4 = vfmaq_lane_f32::<0>(store_4, item_row_1.0, weights);
            store_5 = vfmaq_lane_f32::<0>(store_5, item_row_1.1, weights);
            store_6 = vfmaq_lane_f32::<0>(store_6, item_row_1.2, weights);
            store_7 = vfmaq_lane_f32::<0>(store_7, item_row_1.3, weights);

            let item_row_0 = xvld1q_f32_x4(src_ptr.get_unchecked(src_stride..).as_ptr());
            let item_row_1 = xvld1q_f32_x4(src_ptr.get_unchecked(src_stride + 16..).as_ptr());

            store_0 = vfmaq_lane_f32::<1>(store_0, item_row_0.0, weights);
            store_1 = vfmaq_lane_f32::<1>(store_1, item_row_0.1, weights);
            store_2 = vfmaq_lane_f32::<1>(store_2, item_row_0.2, weights);
            store_3 = vfmaq_lane_f32::<1>(store_3, item_row_0.3, weights);

            store_4 = vfmaq_lane_f32::<1>(store_4, item_row_1.0, weights);
            store_5 = vfmaq_lane_f32::<1>(store_5, item_row_1.1, weights);
            store_6 = vfmaq_lane_f32::<1>(store_6, item_row_1.2, weights);
            store_7 = vfmaq_lane_f32::<1>(store_7, item_row_1.3, weights);
            j += 2;
        }

        for j in j..bounds.size {
            let py = start_y + j;
            let v_weight = vld1q_dup_f32(filter.get_unchecked(j..).as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = xvld1q_f32_x4(src_ptr);
            let item_row_1 = xvld1q_f32_x4(src_ptr.add(16));

            store_0 = prefer_vfmaq_f32(store_0, item_row_0.0, v_weight);
            store_1 = prefer_vfmaq_f32(store_1, item_row_0.1, v_weight);
            store_2 = prefer_vfmaq_f32(store_2, item_row_0.2, v_weight);
            store_3 = prefer_vfmaq_f32(store_3, item_row_0.3, v_weight);

            store_4 = prefer_vfmaq_f32(store_4, item_row_1.0, v_weight);
            store_5 = prefer_vfmaq_f32(store_5, item_row_1.1, v_weight);
            store_6 = prefer_vfmaq_f32(store_6, item_row_1.2, v_weight);
            store_7 = prefer_vfmaq_f32(store_7, item_row_1.3, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..);
        let f_set = float32x4x4_t(store_0, store_1, store_2, store_3);
        xvst1q_f32_x4(dst_ptr.as_mut_ptr(), f_set);

        let f_set_1 = float32x4x4_t(store_4, store_5, store_6, store_7);
        xvst1q_f32_x4(dst_ptr.get_unchecked_mut(16..).as_mut_ptr(), f_set_1);
    }
}

fn conv_vertical_part_neon_16_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);
        let mut store_1 = vdupq_n_f32(0.);
        let mut store_2 = vdupq_n_f32(0.);
        let mut store_3 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let v_weight = vld1q_dup_f32(filter.get_unchecked(j..).as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = xvld1q_f32_x4(src_ptr);

            store_0 = prefer_vfmaq_f32(store_0, item_row_0.0, v_weight);
            store_1 = prefer_vfmaq_f32(store_1, item_row_0.1, v_weight);
            store_2 = prefer_vfmaq_f32(store_2, item_row_0.2, v_weight);
            store_3 = prefer_vfmaq_f32(store_3, item_row_0.3, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..);
        let f_set = float32x4x4_t(store_0, store_1, store_2, store_3);
        xvst1q_f32_x4(dst_ptr.as_mut_ptr(), f_set);
    }
}

fn convolve_vertical_part_neon_8_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);
        let mut store_1 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1q_dup_f32(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);
            let item_row = xvld1q_f32_x2(src_ptr.as_ptr());

            store_0 = prefer_vfmaq_f32(store_0, item_row.0, v_weight);
            store_1 = prefer_vfmaq_f32(store_1, item_row.1, v_weight);
        }

        let item = float32x4x2_t(store_0, store_1);

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        xvst1q_f32_x2(dst_ptr, item);
    }
}

#[inline(always)]
fn convolve_vertical_part_neon_4_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1q_dup_f32(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);

            let item_row = xvld1q_f32_x2(src_ptr.as_ptr());

            store_0 = prefer_vfmaq_f32(store_0, item_row.0, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        vst1q_f32(dst_ptr, store_0);
    }
}

fn convolve_vertical_part_neon_1_f32(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = vdupq_n_f32(0.);

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = vld1q_dup_f32(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);
            let item_row = vld1q_dup_f32(src_ptr.as_ptr());

            store_0 = prefer_vfmaq_f32(store_0, item_row, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        vst1q_lane_f32::<0>(dst_ptr, store_0);
    }
}

pub(crate) fn convolve_vertical_rgb_neon_row_f32(
    _: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
    _: u32,
) {
    let mut cx = 0usize;
    let dst_width = dst.len();

    while cx + 32 <= dst_width {
        conv_vertical_part_neon_32_f32(bounds.start, cx, src, src_stride, dst, weight_ptr, bounds);

        cx += 32;
    }

    while cx + 16 <= dst_width {
        conv_vertical_part_neon_16_f32(bounds.start, cx, src, src_stride, dst, weight_ptr, bounds);

        cx += 16;
    }

    while cx + 8 <= dst_width {
        convolve_vertical_part_neon_8_f32(
            bounds.start,
            cx,
            src,
            src_stride,
            dst,
            weight_ptr,
            bounds,
        );

        cx += 8;
    }

    while cx + 4 <= dst_width {
        convolve_vertical_part_neon_4_f32(
            bounds.start,
            cx,
            src,
            src_stride,
            dst,
            weight_ptr,
            bounds,
        );

        cx += 4;
    }

    while cx < dst_width {
        convolve_vertical_part_neon_1_f32(
            bounds.start,
            cx,
            src,
            src_stride,
            dst,
            weight_ptr,
            bounds,
        );
        cx += 1;
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::ResamplingFunction;
    use crate::floating_point_vertical::column_handler_floating_point;
    use crate::test_utils::{XorShiftRng, assert_f32_slices_close, make_row_filter_weights_f32};

    // NEON has no runtime feature detection (it's baseline on aarch64) and
    // uses `vfmaq_laneq_f32`/`prefer_vfmaq_f32`, fusing multiply+add per lane
    // in the same accumulation order as the scalar loop, so this is expected
    // to be bit-exact (start from 0.0, per the task's guidance, before
    // reaching for a tolerance).
    const ATOL: f32 = 0.0;

    #[test]
    fn neon_vertical_matches_scalar_reference() {
        run_vertical_comparison(convolve_vertical_rgb_neon_row_f32, "NEON");
    }

    #[allow(clippy::type_complexity)]
    fn run_vertical_comparison(
        simd_fn: fn(usize, &FilterBounds, &[f32], &mut [f32], usize, &[f32], u32),
        label: &str,
    ) {
        // Row widths chosen to be multiples of 1 (plane), 3 (rgb) and 4 (rgba)
        // pixels, plus a couple of odd sizes so every SIMD tail-loop width
        // (32/16/8/4/1 lanes) gets exercised at least once.
        const ROW_WIDTHS: [usize; 7] = [1, 3, 4, 39, 99, 160, 41];

        for resampling in [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Nearest,
        ] {
            for (in_height, out_height) in [(64usize, 37usize), (37, 64), (5, 3)] {
                let filter_weights =
                    make_row_filter_weights_f32(resampling, in_height, out_height).unwrap();

                for &row_width in ROW_WIDTHS.iter() {
                    let src_stride = row_width;
                    let mut rng = XorShiftRng::new(
                        0xC0FFEE
                            ^ (in_height as u64) << 32
                            ^ (out_height as u64) << 16
                            ^ row_width as u64,
                    );
                    let src = rng.fill_f32_unit(src_stride * in_height);

                    for y in 0..out_height {
                        let bounds = filter_weights.bounds[y];
                        let filter_offset = y * filter_weights.aligned_size;
                        let weights = &filter_weights.weights[filter_offset..];

                        let mut dst_scalar = vec![0f32; row_width];
                        let mut dst_simd = vec![0f32; row_width];

                        column_handler_floating_point::<f32, f32, f32>(
                            0,
                            &bounds,
                            &src,
                            &mut dst_scalar,
                            src_stride,
                            weights,
                            8,
                        );
                        simd_fn(row_width, &bounds, &src, &mut dst_simd, src_stride, weights, 8);

                        assert_f32_slices_close(
                            &dst_simd,
                            &dst_scalar,
                            ATOL,
                            &format!(
                                "{resampling:?} {in_height}->{out_height} row {y} width {row_width}: {label}"
                            ),
                        );
                    }
                }
            }
        }
    }
}
