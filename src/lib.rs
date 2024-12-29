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
#![deny(deprecated)]
// #![deny(unreachable_code, unused)]
#![allow(clippy::too_many_arguments)]
mod alpha_check;
#[cfg(feature = "half")]
mod alpha_handle_f16;
mod alpha_handle_f32;
mod alpha_handle_u16;
mod alpha_handle_u8;
mod ar30;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx2;
mod color_group;
#[cfg(feature = "colorspaces")]
mod colors;
mod convolution;
mod convolve_naive_f32;
mod cpu_features;
mod dispatch_group_ar30;
#[cfg(feature = "half")]
mod dispatch_group_f16;
mod dispatch_group_f32;
mod dispatch_group_u16;
mod dispatch_group_u8;
#[cfg(feature = "half")]
mod f16;
mod filter_weights;
mod fixed_point_horizontal;
mod fixed_point_horizontal_ar30;
mod fixed_point_vertical;
mod fixed_point_vertical_ar30;
mod floating_point_horizontal;
mod floating_point_vertical;
mod handler_provider;
mod image_size;
mod image_store;
mod math;
mod mixed_storage;
mod mlaf;
mod nearest_sampler;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod pic_scale_error;
mod plane_f32;
mod plane_u16;
mod plane_u8;
mod resize_ar30;
mod rgb_f32;
mod rgb_u16;
mod rgb_u8;
mod rgba_f32;
mod rgba_u16;
mod rgba_u8;
mod sampler;
mod saturate_narrow;
mod scaler;
#[cfg(feature = "half")]
mod scaler_f16;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod sse;
mod support;
mod threading_policy;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128",))]
mod wasm32;

pub use ar30::Ar30ByteOrder;
#[cfg(feature = "colorspaces")]
pub use colors::*;
#[cfg(feature = "colorspaces")]
pub use colorutils_rs::TransferFunction;
pub use image_size::ImageSize;
pub use image_store::{ImageStore, ImageStoreMut};
pub use math::*;
pub use sampler::*;
pub use scaler::Scaler;
pub use scaler::Scaling;
pub use scaler::ScalingF32;
pub use scaler::ScalingU16;
pub use threading_policy::*;
