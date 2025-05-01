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

use crate::convolution::ConvolutionOptions;
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::fixed_point_horizontal_ar30::{
    convolve_row_handler_fixed_point_4_ar30, convolve_row_handler_fixed_point_ar30,
};
use crate::fixed_point_vertical_ar30::column_handler_fixed_point_ar30;
use crate::support::PRECISION;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use rayon::ThreadPool;

#[allow(clippy::type_complexity)]
pub(crate) fn convolve_horizontal_dispatch_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    src: &[u8],
    src_stride: usize,
    filter_weights: FilterWeights<f32>,
    dst: &mut [u8],
    dst_stride: usize,
    pool: &Option<ThreadPool>,
    _options: ConvolutionOptions,
) {
    let _dispatch4 = get_horizontal_dispatch4::<AR30_TYPE, AR30_ORDER>(_options);
    let _dispatch = get_horizontal_dispatch::<AR30_TYPE, AR30_ORDER>();
    if let Some(pool) = pool {
        pool.install(|| {
            let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
            dst.par_chunks_exact_mut(dst_stride * 4)
                .zip(src.par_chunks_exact(src_stride * 4))
                .for_each(|(dst, src)| {
                    _dispatch4(src, src_stride, dst, dst_stride, &approx);
                });

            let remainder = dst.chunks_exact_mut(dst_stride * 4).into_remainder();
            let src_remainder = src.chunks_exact(src_stride * 4).remainder();

            remainder
                .par_chunks_exact_mut(dst_stride)
                .zip(src_remainder.par_chunks_exact(src_stride))
                .for_each(|(dst, src)| {
                    _dispatch(src, dst, &approx);
                });
        });
    } else {
        let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
        dst.chunks_exact_mut(dst_stride * 4)
            .zip(src.chunks_exact(src_stride * 4))
            .for_each(|(dst, src)| {
                _dispatch4(src, src_stride, dst, dst_stride, &approx);
            });

        let remainder = dst.chunks_exact_mut(dst_stride * 4).into_remainder();
        let src_remainder = src.chunks_exact(src_stride * 4).remainder();

        remainder
            .chunks_exact_mut(dst_stride)
            .zip(src_remainder.chunks_exact(src_stride))
            .for_each(|(dst, src)| {
                _dispatch(src, dst, &approx);
            });
    }
}

fn get_horizontal_dispatch<const AR30_TYPE: usize, const AR30_ORDER: usize>(
) -> fn(&[u8], &mut [u8], &FilterWeights<i16>) {
    let mut _dispatch: fn(&[u8], &mut [u8], &FilterWeights<i16>) =
        convolve_row_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use crate::neon::neon_convolve_horizontal_rgba_rows_ar30;
        _dispatch = neon_convolve_horizontal_rgba_rows_ar30::<AR30_TYPE, AR30_ORDER>;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            use crate::sse::sse_convolve_horizontal_rgba_rows_ar30;
            _dispatch = sse_convolve_horizontal_rgba_rows_ar30::<AR30_TYPE, AR30_ORDER>;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::avx_convolve_horizontal_rgba_rows_ar30;
            _dispatch = avx_convolve_horizontal_rgba_rows_ar30::<AR30_TYPE, AR30_ORDER>;
        }
    }
    _dispatch
}

fn get_horizontal_dispatch4<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    _options: ConvolutionOptions,
) -> fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i16>) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    let mut _dispatch: fn(&[u8], usize, &mut [u8], usize, &FilterWeights<i16>) =
        convolve_row_handler_fixed_point_4_ar30::<AR30_TYPE, AR30_ORDER>;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        match _options.workload_strategy {
            crate::WorkloadStrategy::PreferSpeed =>
            {
                #[cfg(feature = "rdm")]
                if is_rdm_available {
                    use crate::neon::neon_convolve_horizontal_rgba_rows_4_ar30_rdm;
                    _dispatch =
                        neon_convolve_horizontal_rgba_rows_4_ar30_rdm::<AR30_TYPE, AR30_ORDER>;
                } else {
                    use crate::neon::neon_convolve_horizontal_rgba_rows_4_ar30;
                    _dispatch = neon_convolve_horizontal_rgba_rows_4_ar30::<AR30_TYPE, AR30_ORDER>;
                }
            }
            crate::WorkloadStrategy::PreferQuality => {
                use crate::neon::neon_convolve_horizontal_rgba_rows_4_ar30;
                _dispatch = neon_convolve_horizontal_rgba_rows_4_ar30::<AR30_TYPE, AR30_ORDER>;
            }
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            use crate::sse::sse_convolve_horizontal_rgba_rows_4_ar30;
            _dispatch = sse_convolve_horizontal_rgba_rows_4_ar30::<AR30_TYPE, AR30_ORDER>;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::avx_convolve_horizontal_rgba_rows_4_ar30;
            _dispatch = avx_convolve_horizontal_rgba_rows_4_ar30::<AR30_TYPE, AR30_ORDER>;
        }
    }
    _dispatch
}

pub(crate) fn convolve_vertical_dispatch_ar30<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    src: &[u8],
    src_stride: usize,
    filter_weights: FilterWeights<f32>,
    dst: &mut [u8],
    dst_stride: usize,
    pool: &Option<ThreadPool>,
    width: usize,
    _options: ConvolutionOptions,
) {
    let _dispatch = get_vertical_dispatcher::<AR30_TYPE, AR30_ORDER>(_options);

    if let Some(pool) = pool {
        pool.install(|| {
            let approx = filter_weights.numerical_approximation_i16::<PRECISION>(0);
            dst.par_chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(y, row)| {
                    let bounds = approx.bounds[y];
                    let filter_offset = y * approx.aligned_size;
                    let weights = &approx.weights[filter_offset..];
                    let row = &mut row[..4 * width];
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

                let row = &mut row[..4 * width];
                _dispatch(&bounds, src, row, src_stride, weights);
            });
    }
}

fn get_vertical_dispatcher<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    _options: ConvolutionOptions,
) -> fn(&FilterBounds, &[u8], &mut [u8], usize, &[i16]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "rdm"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    let mut _dispatch: fn(&FilterBounds, &[u8], &mut [u8], usize, &[i16]) =
        column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        match _options.workload_strategy {
            crate::WorkloadStrategy::PreferSpeed => {
                #[cfg(feature = "rdm")]
                if is_rdm_available {
                    use crate::neon::neon_column_handler_fixed_point_ar30_rdm;
                    _dispatch = neon_column_handler_fixed_point_ar30_rdm::<AR30_TYPE, AR30_ORDER>;
                } else {
                    use crate::neon::neon_column_handler_fixed_point_ar30;
                    _dispatch = neon_column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
                }
                #[cfg(not(feature = "rdm"))]
                {
                    use crate::neon::neon_column_handler_fixed_point_ar30;
                    _dispatch = neon_column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
                }
            }
            crate::WorkloadStrategy::PreferQuality => {
                use crate::neon::neon_column_handler_fixed_point_ar30;
                _dispatch = neon_column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
            }
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            use crate::sse::sse_column_handler_fixed_point_ar30;
            _dispatch = sse_column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::avx_column_handler_fixed_point_ar30;
            _dispatch = avx_column_handler_fixed_point_ar30::<AR30_TYPE, AR30_ORDER>;
        }
    }
    _dispatch
}
