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

use crate::avx2::utils::{_mm_fma_pd, _mm_hsum_pd, _mm256_fma_pd};
use crate::filter_weights::FilterWeights;
use std::arch::x86_64::*;

pub(crate) fn convolve_hor_plane_avx_row_one_f32_f64<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        if FMA {
            convolve_hor_plane_avx_row_one_fma_f32_f64(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        } else {
            convolve_hor_plane_avx_row_one_regular_f32_f64(
                dst_width,
                src_width,
                filter_weights,
                src,
                dst,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_hor_plane_avx_row_one_regular_f32_f64(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        let unit = Row1ExecutorUnit::<false>::default();
        unit.pass(dst_width, src_width, filter_weights, src, dst);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_hor_plane_avx_row_one_fma_f32_f64(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    dst: &mut [f32],
) {
    unsafe {
        let unit = Row1ExecutorUnit::<true>::default();
        unit.pass(dst_width, src_width, filter_weights, src, dst);
    }
}

#[derive(Copy, Clone, Default)]
struct Row1ExecutorUnit<const FMA: bool> {}

impl<const FMA: bool> Row1ExecutorUnit<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        dst_width: usize,
        _: usize,
        filter_weights: &FilterWeights<f64>,
        src: &[f32],
        dst: &mut [f32],
    ) {
        unsafe {
            let mut filter_offset = 0usize;
            let weights_ptr = &filter_weights.weights;

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store = _mm256_setzero_pd();

                while jx + 4 < bounds.size {
                    let bounds_start = bounds.start + jx;
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let w0 = _mm256_loadu_pd(ptr.as_ptr());
                    let px0 = _mm_loadu_ps(src.get_unchecked(bounds_start..).as_ptr() as *const _);
                    store = _mm256_fma_pd::<FMA>(store, _mm256_cvtps_pd(px0), w0);
                    jx += 4;
                }

                let mut store = _mm_add_pd(
                    _mm256_castpd256_pd128(store),
                    _mm256_extractf128_pd::<1>(store),
                );

                while jx + 2 < bounds.size {
                    let bounds_start = bounds.start + jx;
                    let w = weights_ptr.get_unchecked(jx + filter_offset..);
                    let w0 = _mm_loadu_pd(w.as_ptr() as *const _);
                    let px0 = _mm_castsi128_ps(_mm_loadu_si64(
                        src.get_unchecked(bounds_start..).as_ptr() as *const _,
                    ));
                    store = _mm_fma_pd::<FMA>(store, _mm_cvtps_pd(px0), w0);
                    jx += 2;
                }

                while jx < bounds.size {
                    let bounds_start = bounds.start + jx;
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let weight0 = _mm_load_sd(ptr.as_ptr());
                    let px0 = _mm_load_ss(src.get_unchecked(bounds_start..).as_ptr());
                    store = _mm_fma_pd::<FMA>(store, _mm_cvtps_pd(px0), weight0);
                    jx += 1;
                }

                let px = x;
                let dest_ptr = dst.get_unchecked_mut(px);
                _mm_store_ss(dest_ptr, _mm_cvtpd_ps(_mm_hsum_pd(store)));

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}
pub(crate) fn convolve_hor_plane_avx_rows_4_f32_f64<const FMA: bool>(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        if FMA {
            convolve_horizontal_plane_avx_rows_4_fma_f32_f64(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        } else {
            convolve_horizontal_plane_avx_rows_4_regular_f32_f64(
                dst_width,
                src_width,
                filter_weights,
                src,
                src_stride,
                dst,
                dst_stride,
            );
        }
    }
}

#[target_feature(enable = "avx2")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_plane_avx_rows_4_regular_f32_f64(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let unit = Row4ExecutionUnit::<false>::default();
        unit.pass(
            dst_width,
            src_width,
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
/// This inlining is required to activate all features for runtime dispatch.
unsafe fn convolve_horizontal_plane_avx_rows_4_fma_f32_f64(
    dst_width: usize,
    src_width: usize,
    filter_weights: &FilterWeights<f64>,
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let unit = Row4ExecutionUnit::<true>::default();
        unit.pass(
            dst_width,
            src_width,
            filter_weights,
            src,
            src_stride,
            dst,
            dst_stride,
        );
    }
}

#[derive(Copy, Clone, Default)]
struct Row4ExecutionUnit<const FMA: bool> {}

impl<const FMA: bool> Row4ExecutionUnit<FMA> {
    #[inline(always)]
    unsafe fn pass(
        &self,
        dst_width: usize,
        _: usize,
        filter_weights: &FilterWeights<f64>,
        src: &[f32],
        src_stride: usize,
        dst: &mut [f32],
        dst_stride: usize,
    ) {
        unsafe {
            let mut filter_offset = 0usize;
            let weights_ptr = &filter_weights.weights;

            for x in 0..dst_width {
                let bounds = filter_weights.bounds.get_unchecked(x);
                let mut jx = 0usize;
                let mut store_0 = _mm256_setzero_pd();
                let mut store_1 = _mm256_setzero_pd();
                let mut store_2 = _mm256_setzero_pd();
                let mut store_3 = _mm256_setzero_pd();

                let src1 = src.get_unchecked(src_stride..);
                let src2 = src.get_unchecked(src_stride * 2..);
                let src3 = src.get_unchecked(src_stride * 3..);

                while jx + 4 < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);

                    let bounds_start = bounds.start + jx;
                    let w0 = _mm256_loadu_pd(ptr.as_ptr());

                    let px0 = _mm_loadu_ps(src.get_unchecked(bounds_start..).as_ptr() as *const _);
                    let px1 = _mm_loadu_ps(src1.get_unchecked(bounds_start..).as_ptr() as *const _);
                    let px2 = _mm_loadu_ps(src2.get_unchecked(bounds_start..).as_ptr() as *const _);
                    let px3 = _mm_loadu_ps(src3.get_unchecked(bounds_start..).as_ptr() as *const _);

                    store_0 = _mm256_fma_pd::<FMA>(store_0, _mm256_cvtps_pd(px0), w0);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, _mm256_cvtps_pd(px1), w0);
                    store_2 = _mm256_fma_pd::<FMA>(store_2, _mm256_cvtps_pd(px2), w0);
                    store_3 = _mm256_fma_pd::<FMA>(store_3, _mm256_cvtps_pd(px3), w0);

                    jx += 4;
                }

                let mut store_0 = _mm256_add_pd(
                    _mm256_castsi256_pd(_mm256_permute2x128_si256::<0x20>(
                        _mm256_castpd_si256(store_0),
                        _mm256_castpd_si256(store_1),
                    )),
                    _mm256_castsi256_pd(_mm256_permute2x128_si256::<0x31>(
                        _mm256_castpd_si256(store_0),
                        _mm256_castpd_si256(store_1),
                    )),
                );

                let mut store_1 = _mm256_add_pd(
                    _mm256_castsi256_pd(_mm256_permute2x128_si256::<0x20>(
                        _mm256_castpd_si256(store_2),
                        _mm256_castpd_si256(store_3),
                    )),
                    _mm256_castsi256_pd(_mm256_permute2x128_si256::<0x31>(
                        _mm256_castpd_si256(store_2),
                        _mm256_castpd_si256(store_3),
                    )),
                );

                while jx + 2 < bounds.size {
                    let w = weights_ptr.get_unchecked(jx + filter_offset..);
                    let bounds_start = bounds.start + jx;
                    let wh = _mm_loadu_pd(w.as_ptr() as *const _);

                    let w0 = _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(wh), wh);

                    let px0 = _mm_castsi128_ps(_mm_loadu_si64(
                        src.get_unchecked(bounds_start..).as_ptr() as *const _,
                    ));
                    let px1 = _mm_castsi128_ps(_mm_loadu_si64(
                        src1.get_unchecked(bounds_start..).as_ptr() as *const _,
                    ));
                    let px2 = _mm_castsi128_ps(_mm_loadu_si64(
                        src2.get_unchecked(bounds_start..).as_ptr() as *const _,
                    ));
                    let px3 = _mm_castsi128_ps(_mm_loadu_si64(
                        src3.get_unchecked(bounds_start..).as_ptr() as *const _,
                    ));

                    let px01 = _mm256_cvtps_pd(_mm_movelh_ps(px0, px1));
                    let px23 = _mm256_cvtps_pd(_mm_movelh_ps(px2, px3));

                    store_0 = _mm256_fma_pd::<FMA>(store_0, px01, w0);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, px23, w0);
                    jx += 2;
                }

                while jx < bounds.size {
                    let ptr = weights_ptr.get_unchecked(jx + filter_offset..);
                    let wh = _mm_load_sd(ptr.as_ptr());
                    let w0 = _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(wh), wh);

                    let bounds_start = bounds.start + jx;

                    let px0 = _mm_load_ss(src.get_unchecked(bounds_start..).as_ptr());
                    let px1 = _mm_load_ss(src1.get_unchecked(bounds_start..).as_ptr());
                    let px2 = _mm_load_ss(src2.get_unchecked(bounds_start..).as_ptr());
                    let px3 = _mm_load_ss(src3.get_unchecked(bounds_start..).as_ptr());

                    let px01 = _mm256_cvtps_pd(_mm_movelh_ps(px0, px1));
                    let px23 = _mm256_cvtps_pd(_mm_movelh_ps(px2, px3));

                    store_0 = _mm256_fma_pd::<FMA>(store_0, px01, w0);
                    store_1 = _mm256_fma_pd::<FMA>(store_1, px23, w0);

                    jx += 1;
                }

                let px = x;
                let dest_ptr = dst.get_unchecked_mut(px);
                _mm_store_ss(
                    dest_ptr,
                    _mm_cvtpd_ps(_mm_hsum_pd(_mm256_castpd256_pd128(store_0))),
                );

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride);
                _mm_store_ss(
                    dest_ptr,
                    _mm_cvtpd_ps(_mm_hsum_pd(_mm256_extractf128_pd::<1>(store_0))),
                );

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 2);
                _mm_store_ss(
                    dest_ptr,
                    _mm_cvtpd_ps(_mm_hsum_pd(_mm256_castpd256_pd128(store_1))),
                );

                let dest_ptr = dst.get_unchecked_mut(px + dst_stride * 3);
                _mm_store_ss(
                    dest_ptr,
                    _mm_cvtpd_ps(_mm_hsum_pd(_mm256_extractf128_pd::<1>(store_1))),
                );

                filter_offset += filter_weights.aligned_size;
            }
        }
    }
}
