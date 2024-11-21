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

use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::fixed_point_horizontal_ar30::{
    convolve_row_handler_fixed_point_4_ar30, convolve_row_handler_fixed_point_ar30,
};
use crate::fixed_point_vertical_ar30::column_handler_fixed_point_ar30;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    neon_column_handler_fixed_point_ar30, neon_convolve_horizontal_rgba_rows_4_ar30,
};
use crate::support::PRECISION;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_horizontal_dispatch_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    src: &[u32],
    src_stride: usize,
    filter_weights: FilterWeights<f32>,
    dst: &mut [u32],
    dst_stride: usize,
    pool: &Option<ThreadPool>,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    if let Some(pool) = pool {
        pool.install(|| {
            let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
            dst.par_chunks_exact_mut(dst_stride * 4)
                .zip(src.par_chunks_exact(src_stride * 4))
                .for_each(|(dst, src)| {
                    let mut _dispatch: fn(&[u32], usize, &mut [u32], usize, &FilterWeights<i16>) =
                        convolve_row_handler_fixed_point_4_ar30::<AR30_TYPE, AR30_ORDER>;
                    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                    if is_rdm_available {
                        _dispatch =
                            neon_convolve_horizontal_rgba_rows_4_ar30::<AR30_TYPE, AR30_ORDER>;
                    }
                    _dispatch(src, src_stride, dst, dst_stride, &approx);
                });

            let remainder = dst.chunks_exact_mut(dst_stride * 4).into_remainder();
            let src_remainder = src.chunks_exact(src_stride * 4).remainder();

            remainder
                .par_chunks_exact_mut(dst_stride)
                .zip(src_remainder.par_chunks_exact(src_stride))
                .for_each(|(dst, src)| {
                    convolve_row_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>(
                        src, dst, &approx,
                    );
                });
        });
    } else {
        let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
        dst.chunks_exact_mut(dst_stride * 4)
            .zip(src.chunks_exact(src_stride * 4))
            .for_each(|(dst, src)| {
                let mut _dispatch: fn(&[u32], usize, &mut [u32], usize, &FilterWeights<i16>) =
                    convolve_row_handler_fixed_point_4_ar30::<AR30_TYPE, AR30_ORDER>;
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                if is_rdm_available {
                    _dispatch = neon_convolve_horizontal_rgba_rows_4_ar30::<AR30_TYPE, AR30_ORDER>;
                }
                _dispatch(src, src_stride, dst, dst_stride, &approx);
            });

        let remainder = dst.chunks_exact_mut(dst_stride * 4).into_remainder();
        let src_remainder = src.chunks_exact(src_stride * 4).remainder();

        remainder
            .chunks_exact_mut(dst_stride)
            .zip(src_remainder.chunks_exact(src_stride))
            .for_each(|(dst, src)| {
                convolve_row_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>(src, dst, &approx);
            });
    }
}

pub(crate) fn convolve_vertical_dispatch_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    src: &[u32],
    src_stride: usize,
    filter_weights: FilterWeights<f32>,
    dst: &mut [u32],
    dst_stride: usize,
    pool: &Option<ThreadPool>,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    if let Some(pool) = pool {
        pool.install(|| {
            let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
            dst.par_chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(y, row)| {
                    let bounds = approx.bounds[y];
                    let filter_offset = y * approx.aligned_size;
                    let weights = &approx.weights[filter_offset..];
                    let mut _dispatch: fn(&FilterBounds, &[u32], &mut [u32], usize, &[i16]) =
                        column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
                    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                    if is_rdm_available {
                        _dispatch = neon_column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
                    }

                    _dispatch(&bounds, src, row, src_stride, weights);
                });
        });
    } else {
        let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
        dst.chunks_exact_mut(dst_stride)
            .enumerate()
            .for_each(|(y, row)| {
                let bounds = approx.bounds[y];
                let filter_offset = y * approx.aligned_size;
                let weights = &approx.weights[filter_offset..];

                let mut _dispatch: fn(&FilterBounds, &[u32], &mut [u32], usize, &[i16]) =
                    column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                if is_rdm_available {
                    _dispatch = neon_column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
                }

                _dispatch(&bounds, src, row, src_stride, weights);
            });
    }
}
