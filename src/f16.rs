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

// RGBA

use crate::ImageStore;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::avx2::{
    convolve_horizontal_rgba_avx_row_one_f16, convolve_horizontal_rgba_avx_rows_4_f16,
    convolve_vertical_avx_row_f16,
};
use crate::convolution::{ConvolutionOptions, HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::dispatch_group_f16::{convolve_horizontal_dispatch_f16, convolve_vertical_dispatch_f16};
use crate::filter_weights::{FilterBounds, FilterWeights, PassthroughWeightsConverter};
use crate::floating_point_horizontal::{
    convolve_row_handler_floating_point, convolve_row_handler_floating_point_4,
};
use crate::floating_point_vertical::column_handler_floating_point;
use crate::image_store::ImageStoreMut;
#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::neon::{
    convolve_horizontal_rgb_neon_row_one_f16, convolve_horizontal_rgb_neon_rows_4_f16,
    convolve_horizontal_rgba_neon_row_one_f16, convolve_horizontal_rgba_neon_rows_4_f16,
    convolve_vertical_rgb_neon_row_f16,
};
#[cfg(all(target_arch = "aarch64", target_feature = "neon",))]
use crate::neon::{
    xconvolve_horizontal_rgb_neon_row_one_f16, xconvolve_horizontal_rgb_neon_rows_4_f16,
    xconvolve_horizontal_rgba_neon_row_one_f16, xconvolve_horizontal_rgba_neon_rows_4_f16,
    xconvolve_vertical_rgb_neon_row_f16,
};
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
use crate::sse::{
    convolve_horizontal_rgb_sse_row_one_f16, convolve_horizontal_rgb_sse_rows_4_f16,
    convolve_horizontal_rgba_sse_row_one_f16, convolve_horizontal_rgba_sse_rows_4_f16,
    convolve_vertical_sse_row_f16,
};
use core::{f16, f32};
use rayon::ThreadPool;

fn convolve_horizontal_rgba_4_row_f16<const CHANNELS: usize>(
    _: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    src_stride: usize,
    dst: &mut [f16],
    dst_stride: usize,
) {
    let transient_src = src.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let mut transient_dst = vec![0f32; dst.len()];
    convolve_row_handler_floating_point_4::<f32, f32, f32, CHANNELS>(
        &transient_src,
        src_stride,
        &mut transient_dst,
        dst_stride,
        filter_weights,
        8,
    );
    for (dst, src) in dst.iter_mut().zip(transient_dst.iter()) {
        *dst = *src as f16;
    }
}

fn convolve_horizontal_rgb_native_row_f16<const CHANNELS: usize>(
    _: usize,
    _: usize,
    filter_weights: &FilterWeights<f32>,
    src: &[f16],
    dst: &mut [f16],
) {
    let transient_src = src.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let mut transient_dst = vec![0f32; dst.len()];
    convolve_row_handler_floating_point::<f32, f32, f32, CHANNELS>(
        &transient_src,
        &mut transient_dst,
        filter_weights,
        8,
    );
    for (dst, src) in dst.iter_mut().zip(transient_dst.iter()) {
        *dst = *src as f16;
    }
}

impl HorizontalConvolutionPass<f16, 4> for ImageStore<'_, f16, 4> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f16, 4>,
        pool: &Option<ThreadPool>,
        _options: ConvolutionOptions,
    ) {
        #[allow(clippy::type_complexity)]
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, &[f16], usize, &mut [f16], usize),
        > = Some(convolve_horizontal_rgba_4_row_f16::<4>);
        #[allow(clippy::type_complexity)]
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, &[f16], &mut [f16]) =
            convolve_horizontal_rgb_native_row_f16::<4>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgba_neon_rows_4_f16);
            _dispatcher_row = convolve_horizontal_rgba_neon_row_one_f16;
            match _options.workload_strategy {
                crate::WorkloadStrategy::PreferSpeed => {
                    if std::arch::is_aarch64_feature_detected!("fp16") {
                        _dispatcher_4_rows = Some(xconvolve_horizontal_rgba_neon_rows_4_f16);
                        _dispatcher_row = xconvolve_horizontal_rgba_neon_row_one_f16;
                    }
                }
                crate::WorkloadStrategy::PreferQuality => {
                    if std::arch::is_aarch64_feature_detected!("fhm") {
                        use crate::filter_weights::WeightFloat16Converter;
                        use crate::neon::{
                            convolve_horizontal_rgba_neon_row_one_f16_fhm,
                            convolve_horizontal_rgba_neon_rows_4_f16_fhm,
                        };
                        return convolve_horizontal_dispatch_f16(
                            self,
                            filter_weights,
                            destination,
                            pool,
                            Some(convolve_horizontal_rgba_neon_rows_4_f16_fhm),
                            convolve_horizontal_rgba_neon_row_one_f16_fhm,
                            WeightFloat16Converter::default(),
                        );
                    }
                }
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_sse_rows_4_f16::<false, false>);
                _dispatcher_row = convolve_horizontal_rgba_sse_row_one_f16::<false, false>;
                if is_f16c_available {
                    _dispatcher_4_rows =
                        Some(convolve_horizontal_rgba_sse_rows_4_f16::<true, false>);
                    _dispatcher_row = convolve_horizontal_rgba_sse_row_one_f16::<true, false>;
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            let fma_available = std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx2") && is_f16c_available {
                _dispatcher_4_rows = Some(convolve_horizontal_rgba_avx_rows_4_f16::<false>);
                _dispatcher_row = convolve_horizontal_rgba_avx_row_one_f16::<false>;
                if fma_available {
                    _dispatcher_4_rows = Some(convolve_horizontal_rgba_avx_rows_4_f16::<true>);
                    _dispatcher_row = convolve_horizontal_rgba_avx_row_one_f16::<true>;
                }
            }
        }
        convolve_horizontal_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher_4_rows,
            _dispatcher_row,
            PassthroughWeightsConverter::default(),
        );
    }
}

fn convolve_vertical_rgb_native_row_f16(
    _: usize,
    bounds: &FilterBounds,
    src: &[f16],
    dst: &mut [f16],
    src_stride: usize,
    weight: &[f32],
) {
    let transient_src = src.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let mut transient_dst = vec![0f32; dst.len()];
    column_handler_floating_point::<f32, f32, f32>(
        bounds,
        &transient_src,
        &mut transient_dst,
        src_stride,
        weight,
        8,
    );
    for (dst, src) in dst.iter_mut().zip(transient_dst.iter()) {
        *dst = *src as f16;
    }
}

impl VerticalConvolutionPass<f16, 4> for ImageStore<'_, f16, 4> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f16, 4>,
        pool: &Option<ThreadPool>,
        _options: ConvolutionOptions,
    ) {
        #[allow(clippy::type_complexity)]
        let mut _dispatcher: fn(usize, &FilterBounds, &[f16], &mut [f16], usize, &[f32]) =
            convolve_vertical_rgb_native_row_f16;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f16;
            match _options.workload_strategy {
                crate::WorkloadStrategy::PreferQuality => {
                    use crate::filter_weights::WeightFloat16Converter;
                    use crate::neon::convolve_vertical_rgb_neon_row_f16_fhm;
                    if std::arch::is_aarch64_feature_detected!("fhm") {
                        return convolve_vertical_dispatch_f16(
                            self,
                            filter_weights,
                            destination,
                            pool,
                            convolve_vertical_rgb_neon_row_f16_fhm,
                            WeightFloat16Converter {},
                        );
                    }
                }
                crate::WorkloadStrategy::PreferSpeed => {
                    if std::arch::is_aarch64_feature_detected!("fp16") {
                        _dispatcher = xconvolve_vertical_rgb_neon_row_f16;
                    }
                }
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_sse_row_f16::<false, false>;
                if is_f16c_available {
                    _dispatcher = convolve_vertical_sse_row_f16::<true, false>;
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            let is_fma_available = std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx2") && is_f16c_available {
                _dispatcher = convolve_vertical_avx_row_f16::<false>;
                if is_fma_available {
                    _dispatcher = convolve_vertical_avx_row_f16::<true>;
                }
            }
        }
        convolve_vertical_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher,
            PassthroughWeightsConverter {},
        );
    }
}

impl HorizontalConvolutionPass<f16, 3> for ImageStore<'_, f16, 3> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f16, 3>,
        pool: &Option<ThreadPool>,
        _options: ConvolutionOptions,
    ) {
        #[allow(clippy::type_complexity)]
        let mut _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, &[f16], usize, &mut [f16], usize),
        > = Some(convolve_horizontal_rgba_4_row_f16::<3>);
        #[allow(clippy::type_complexity)]
        let mut _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, &[f16], &mut [f16]) =
            convolve_horizontal_rgb_native_row_f16::<3>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher_4_rows = Some(convolve_horizontal_rgb_neon_rows_4_f16);
            _dispatcher_row = convolve_horizontal_rgb_neon_row_one_f16;
            match _options.workload_strategy {
                crate::WorkloadStrategy::PreferQuality => {
                    if std::arch::is_aarch64_feature_detected!("fhm") {
                        use crate::filter_weights::WeightFloat16Converter;
                        use crate::neon::{
                            convolve_horizontal_rgb_neon_row_one_f16_fhm,
                            convolve_horizontal_rgb_neon_rows_4_f16_fhm,
                        };
                        return convolve_horizontal_dispatch_f16(
                            self,
                            filter_weights,
                            destination,
                            pool,
                            Some(convolve_horizontal_rgb_neon_rows_4_f16_fhm),
                            convolve_horizontal_rgb_neon_row_one_f16_fhm,
                            WeightFloat16Converter::default(),
                        );
                    }
                }
                crate::WorkloadStrategy::PreferSpeed => {
                    if std::arch::is_aarch64_feature_detected!("fp16")
                        && _options.workload_strategy == crate::WorkloadStrategy::PreferSpeed
                    {
                        _dispatcher_4_rows = Some(xconvolve_horizontal_rgb_neon_rows_4_f16);
                        _dispatcher_row = xconvolve_horizontal_rgb_neon_row_one_f16;
                    }
                }
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher_4_rows = Some(convolve_horizontal_rgb_sse_rows_4_f16::<false, false>);
                _dispatcher_row = convolve_horizontal_rgb_sse_row_one_f16::<false, false>;
                if is_x86_feature_detected!("f16c") {
                    if is_x86_feature_detected!("fma") {
                        _dispatcher_4_rows =
                            Some(convolve_horizontal_rgb_sse_rows_4_f16::<true, true>);
                        _dispatcher_row = convolve_horizontal_rgb_sse_row_one_f16::<true, true>;
                    } else {
                        _dispatcher_4_rows =
                            Some(convolve_horizontal_rgb_sse_rows_4_f16::<true, false>);
                        _dispatcher_row = convolve_horizontal_rgb_sse_row_one_f16::<true, false>;
                    }
                }
            }
        }
        convolve_horizontal_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher_4_rows,
            _dispatcher_row,
            PassthroughWeightsConverter::default(),
        );
    }
}

impl VerticalConvolutionPass<f16, 3> for ImageStore<'_, f16, 3> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f16, 3>,
        pool: &Option<ThreadPool>,
        _options: ConvolutionOptions,
    ) {
        #[allow(clippy::type_complexity)]
        let mut _dispatcher: fn(usize, &FilterBounds, &[f16], &mut [f16], usize, &[f32]) =
            convolve_vertical_rgb_native_row_f16;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f16;
            match _options.workload_strategy {
                crate::WorkloadStrategy::PreferQuality => {
                    use crate::filter_weights::WeightFloat16Converter;
                    use crate::neon::convolve_vertical_rgb_neon_row_f16_fhm;
                    if std::arch::is_aarch64_feature_detected!("fhm") {
                        return convolve_vertical_dispatch_f16(
                            self,
                            filter_weights,
                            destination,
                            pool,
                            convolve_vertical_rgb_neon_row_f16_fhm,
                            WeightFloat16Converter {},
                        );
                    }
                }
                crate::WorkloadStrategy::PreferSpeed => {
                    if std::arch::is_aarch64_feature_detected!("fp16") {
                        _dispatcher = xconvolve_vertical_rgb_neon_row_f16;
                    }
                }
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_sse_row_f16::<false, false>;
                if is_f16c_available {
                    _dispatcher = convolve_vertical_sse_row_f16::<true, false>;
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            let is_fma_available = std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx2") && is_f16c_available {
                _dispatcher = convolve_vertical_avx_row_f16::<false>;
                if is_fma_available {
                    _dispatcher = convolve_vertical_avx_row_f16::<true>;
                }
            }
        }
        convolve_vertical_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher,
            PassthroughWeightsConverter::default(),
        );
    }
}

impl HorizontalConvolutionPass<f16, 1> for ImageStore<'_, f16, 1> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f16, 1>,
        pool: &Option<ThreadPool>,
        _: ConvolutionOptions,
    ) {
        #[allow(clippy::type_complexity)]
        let _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, &[f16], usize, &mut [f16], usize),
        > = Some(convolve_horizontal_rgba_4_row_f16::<1>);
        let _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, &[f16], &mut [f16]) =
            convolve_horizontal_rgb_native_row_f16::<1>;
        convolve_horizontal_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher_4_rows,
            _dispatcher_row,
            PassthroughWeightsConverter::default(),
        );
    }
}

impl VerticalConvolutionPass<f16, 1> for ImageStore<'_, f16, 1> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f16, 1>,
        pool: &Option<ThreadPool>,
        _options: ConvolutionOptions,
    ) {
        #[allow(clippy::type_complexity)]
        let mut _dispatcher: fn(usize, &FilterBounds, &[f16], &mut [f16], usize, &[f32]) =
            convolve_vertical_rgb_native_row_f16;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f16;
            match _options.workload_strategy {
                crate::WorkloadStrategy::PreferQuality => {
                    use crate::filter_weights::WeightFloat16Converter;
                    use crate::neon::convolve_vertical_rgb_neon_row_f16_fhm;
                    if std::arch::is_aarch64_feature_detected!("fhm") {
                        return convolve_vertical_dispatch_f16(
                            self,
                            filter_weights,
                            destination,
                            pool,
                            convolve_vertical_rgb_neon_row_f16_fhm,
                            WeightFloat16Converter {},
                        );
                    }
                }
                crate::WorkloadStrategy::PreferSpeed => {
                    if std::arch::is_aarch64_feature_detected!("fp16") {
                        _dispatcher = xconvolve_vertical_rgb_neon_row_f16;
                    }
                }
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_sse_row_f16::<false, false>;
                if is_f16c_available {
                    _dispatcher = convolve_vertical_sse_row_f16::<true, false>;
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            let is_fma_available = std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx2") && is_f16c_available {
                _dispatcher = convolve_vertical_avx_row_f16::<false>;
                if is_fma_available {
                    _dispatcher = convolve_vertical_avx_row_f16::<true>;
                }
            }
        }
        convolve_vertical_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher,
            PassthroughWeightsConverter::default(),
        );
    }
}

impl HorizontalConvolutionPass<f16, 2> for ImageStore<'_, f16, 2> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f16, 2>,
        pool: &Option<ThreadPool>,
        _: ConvolutionOptions,
    ) {
        #[allow(clippy::type_complexity)]
        let _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<f32>, &[f16], usize, &mut [f16], usize),
        > = Some(convolve_horizontal_rgba_4_row_f16::<2>);
        let _dispatcher_row: fn(usize, usize, &FilterWeights<f32>, &[f16], &mut [f16]) =
            convolve_horizontal_rgb_native_row_f16::<2>;
        convolve_horizontal_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher_4_rows,
            _dispatcher_row,
            PassthroughWeightsConverter::default(),
        );
    }
}

impl VerticalConvolutionPass<f16, 2> for ImageStore<'_, f16, 2> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStoreMut<f16, 2>,
        pool: &Option<ThreadPool>,
        _options: ConvolutionOptions,
    ) {
        #[allow(clippy::type_complexity)]
        let mut _dispatcher: fn(usize, &FilterBounds, &[f16], &mut [f16], usize, &[f32]) =
            convolve_vertical_rgb_native_row_f16;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_f16;
            match _options.workload_strategy {
                crate::WorkloadStrategy::PreferQuality => {
                    use crate::filter_weights::WeightFloat16Converter;
                    use crate::neon::convolve_vertical_rgb_neon_row_f16_fhm;
                    if std::arch::is_aarch64_feature_detected!("fhm") {
                        return convolve_vertical_dispatch_f16(
                            self,
                            filter_weights,
                            destination,
                            pool,
                            convolve_vertical_rgb_neon_row_f16_fhm,
                            WeightFloat16Converter {},
                        );
                    }
                }
                crate::WorkloadStrategy::PreferSpeed => {
                    if std::arch::is_aarch64_feature_detected!("fp16") {
                        _dispatcher = xconvolve_vertical_rgb_neon_row_f16;
                    }
                }
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            if std::arch::is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_sse_row_f16::<false, false>;
                if is_f16c_available {
                    _dispatcher = convolve_vertical_sse_row_f16::<true, false>;
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let is_f16c_available = std::arch::is_x86_feature_detected!("f16c");
            let is_fma_available = std::arch::is_x86_feature_detected!("fma");
            if std::arch::is_x86_feature_detected!("avx2") && is_f16c_available {
                _dispatcher = convolve_vertical_avx_row_f16::<false>;
                if is_fma_available {
                    _dispatcher = convolve_vertical_avx_row_f16::<true>;
                }
            }
        }
        convolve_vertical_dispatch_f16(
            self,
            filter_weights,
            destination,
            pool,
            _dispatcher,
            PassthroughWeightsConverter::default(),
        );
    }
}
