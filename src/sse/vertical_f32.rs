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
use crate::sse::_mm_prefer_fma_ps;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
fn convolve_vertical_part_sse_24_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();
        let mut store_1 = _mm_setzero_ps();
        let mut store_2 = _mm_setzero_ps();
        let mut store_3 = _mm_setzero_ps();
        let mut store_4 = _mm_setzero_ps();
        let mut store_5 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load1_ps(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = _mm_loadu_ps(src_ptr);
            let item_row_1 = _mm_loadu_ps(src_ptr.add(4));
            let item_row_2 = _mm_loadu_ps(src_ptr.add(8));
            let item_row_3 = _mm_loadu_ps(src_ptr.add(12));
            let item_row_4 = _mm_loadu_ps(src_ptr.add(16));
            let item_row_5 = _mm_loadu_ps(src_ptr.add(20));

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
            store_1 = _mm_prefer_fma_ps::<FMA>(store_1, item_row_1, v_weight);
            store_2 = _mm_prefer_fma_ps::<FMA>(store_2, item_row_2, v_weight);
            store_3 = _mm_prefer_fma_ps::<FMA>(store_3, item_row_3, v_weight);
            store_4 = _mm_prefer_fma_ps::<FMA>(store_4, item_row_4, v_weight);
            store_5 = _mm_prefer_fma_ps::<FMA>(store_5, item_row_5, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_ps(dst_ptr, store_0);
        _mm_storeu_ps(dst_ptr.add(4), store_1);
        _mm_storeu_ps(dst_ptr.add(8), store_2);
        _mm_storeu_ps(dst_ptr.add(12), store_3);
        _mm_storeu_ps(dst_ptr.add(16), store_4);
        _mm_storeu_ps(dst_ptr.add(20), store_5);
    }
}

#[inline(always)]
fn convolve_vertical_part_sse_16_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();
        let mut store_1 = _mm_setzero_ps();
        let mut store_2 = _mm_setzero_ps();
        let mut store_3 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load1_ps(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = _mm_loadu_ps(src_ptr);
            let item_row_1 = _mm_loadu_ps(src_ptr.add(4));
            let item_row_2 = _mm_loadu_ps(src_ptr.add(8));
            let item_row_3 = _mm_loadu_ps(src_ptr.add(12));

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
            store_1 = _mm_prefer_fma_ps::<FMA>(store_1, item_row_1, v_weight);
            store_2 = _mm_prefer_fma_ps::<FMA>(store_2, item_row_2, v_weight);
            store_3 = _mm_prefer_fma_ps::<FMA>(store_3, item_row_3, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_ps(dst_ptr, store_0);
        _mm_storeu_ps(dst_ptr.add(4), store_1);
        _mm_storeu_ps(dst_ptr.add(8), store_2);
        _mm_storeu_ps(dst_ptr.add(12), store_3);
    }
}

#[inline(always)]
fn convolve_vertical_part_sse_8_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();
        let mut store_1 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load1_ps(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = _mm_loadu_ps(src_ptr);
            let item_row_1 = _mm_loadu_ps(src_ptr.add(4));

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
            store_1 = _mm_prefer_fma_ps::<FMA>(store_1, item_row_1, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_ps(dst_ptr, store_0);
        _mm_storeu_ps(dst_ptr.add(4), store_1);
    }
}

#[inline(always)]
fn convolve_vertical_part_sse_4_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load1_ps(weight.as_ptr());

            let src_ptr = src.get_unchecked(src_stride * py + px..).as_ptr();

            let item_row_0 = _mm_loadu_ps(src_ptr);

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_storeu_ps(dst_ptr, store_0);
    }
}

#[inline(always)]
pub(crate) fn convolve_vertical_part_sse_f32<const FMA: bool>(
    start_y: usize,
    start_x: usize,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    filter: &[f32],
    bounds: &FilterBounds,
) {
    unsafe {
        let mut store_0 = _mm_setzero_ps();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = filter.get_unchecked(j..);
            let v_weight = _mm_load_ss(weight.as_ptr());
            let src_ptr = src.get_unchecked(src_stride * py + px..);

            let item_row_0 = _mm_load_ss(src_ptr.as_ptr());

            store_0 = _mm_prefer_fma_ps::<FMA>(store_0, item_row_0, v_weight);
        }

        let dst_ptr = dst.get_unchecked_mut(px..).as_mut_ptr();
        _mm_store_ss(dst_ptr, store_0);
    }
}

pub(crate) fn convolve_vertical_rgb_sse_row_f32(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
    _: u32,
) {
    unsafe {
        convolve_vertical_rgb_sse_row_f32_regular(width, bounds, src, dst, src_stride, weight_ptr);
    }
}

#[target_feature(enable = "sse4.1")]
/// This inlining is required to activate all features for runtime dispatch.
fn convolve_vertical_rgb_sse_row_f32_regular(
    width: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    convolve_vertical_rgb_sse_row_f32_impl::<false>(
        width, bounds, src, dst, src_stride, weight_ptr,
    );
}

#[inline(always)]
fn convolve_vertical_rgb_sse_row_f32_impl<const FMA: bool>(
    _: usize,
    bounds: &FilterBounds,
    src: &[f32],
    dst: &mut [f32],
    src_stride: usize,
    weight_ptr: &[f32],
) {
    let mut cx = 0usize;
    let dst_width = dst.len();

    while cx + 24 <= dst_width {
        convolve_vertical_part_sse_24_f32::<FMA>(
            bounds.start,
            cx,
            src,
            src_stride,
            dst,
            weight_ptr,
            bounds,
        );

        cx += 24;
    }

    while cx + 16 <= dst_width {
        convolve_vertical_part_sse_16_f32::<FMA>(
            bounds.start,
            cx,
            src,
            src_stride,
            dst,
            weight_ptr,
            bounds,
        );

        cx += 16;
    }

    while cx + 8 <= dst_width {
        convolve_vertical_part_sse_8_f32::<FMA>(
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
        convolve_vertical_part_sse_4_f32::<FMA>(
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
        convolve_vertical_part_sse_f32::<FMA>(
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

    // SSE only has a single dispatch path here (no runtime FMA selection - see
    // `convolve_vertical_rgb_sse_row_f32_regular` always instantiating
    // `FMA = false`), and each lane accumulates one weighted row at a time in
    // the same order as the scalar loop, so this backend is bit-exact.
    const ATOL: f32 = 0.0;

    #[test]
    fn sse_vertical_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        run_vertical_comparison(convolve_vertical_rgb_sse_row_f32, "SSE");
    }

    #[allow(clippy::type_complexity)]
    fn run_vertical_comparison(
        simd_fn: fn(usize, &FilterBounds, &[f32], &mut [f32], usize, &[f32], u32),
        label: &str,
    ) {
        // Row widths chosen to be multiples of 1 (plane), 3 (rgb) and 4 (rgba)
        // pixels, plus a couple of odd sizes so every SIMD tail-loop width
        // (24/16/8/4/1 lanes) gets exercised at least once.
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
