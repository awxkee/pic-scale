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

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx2"
))]
use crate::avx2::convolve_vertical_rgb_avx_row_u16;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::convolve_naive_u16::{
    convolve_horizontal_rgba_native_4_row_u16, convolve_horizontal_rgba_native_row_u16,
    convolve_vertical_rgb_native_row_u16,
};
use crate::dispatch_group_u16::{convolve_horizontal_dispatch_u16, convolve_vertical_dispatch_u16};
use crate::filter_weights::{FilterBounds, FilterWeights};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::convolve_vertical_rgb_neon_row_u16;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
use crate::sse::convolve_vertical_rgb_sse_row_u16;
use crate::ImageStore;
use rayon::ThreadPool;

impl<'a> HorizontalConvolutionPass<u16, 1> for ImageStore<'a, u16, 1> {
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u16, 1>,
        _pool: &Option<ThreadPool>,
    ) {
        let _dispatcher_4_rows: Option<
            fn(usize, usize, &FilterWeights<i16>, *const u16, usize, *mut u16, usize, usize),
        > = Some(convolve_horizontal_rgba_native_4_row_u16::<1>);
        let _dispatcher_1_row: fn(usize, usize, &FilterWeights<i16>, *const u16, *mut u16, usize) =
            convolve_horizontal_rgba_native_row_u16::<1>;
        convolve_horizontal_dispatch_u16(
            self,
            filter_weights,
            destination,
            _pool,
            _dispatcher_4_rows,
            _dispatcher_1_row,
        );
    }
}

impl<'a> VerticalConvolutionPass<u16, 1> for ImageStore<'a, u16, 1> {
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<u16, 1>,
        pool: &Option<ThreadPool>,
    ) {
        let mut _dispatcher: fn(
            dst_width: usize,
            bounds: &FilterBounds,
            unsafe_source_ptr_0: *const u16,
            unsafe_destination_ptr_0: *mut u16,
            src_stride: usize,
            weight_ptr: *const i16,
            usize,
        ) = convolve_vertical_rgb_native_row_u16::<1>;
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            _dispatcher = convolve_vertical_rgb_neon_row_u16::<1>;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "sse4.1"
        ))]
        {
            if is_x86_feature_detected!("sse4.1") {
                _dispatcher = convolve_vertical_rgb_sse_row_u16::<1>;
            }
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "x86"),
            target_feature = "avx2"
        ))]
        {
            if is_x86_feature_detected!("avx2") {
                _dispatcher = convolve_vertical_rgb_avx_row_u16::<1>;
            }
        }
        convolve_vertical_dispatch_u16(self, filter_weights, destination, pool, _dispatcher);
    }
}
