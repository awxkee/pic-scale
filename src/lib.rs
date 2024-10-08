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
#![allow(clippy::too_many_arguments)]
#![cfg_attr(
    all(
        feature = "riscv",
        any(target_arch = "riscv64", target_arch = "riscv32")
    ),
    feature(riscv_target_feature)
)]
#![cfg_attr(
    all(
        feature = "riscv",
        any(target_arch = "riscv64", target_arch = "riscv32")
    ),
    feature(stdarch_riscv_feature_detection)
)]
#[cfg(feature = "half")]
mod alpha_handle_f16;
mod alpha_handle_f32;
mod alpha_handle_u16;
mod alpha_handle_u8;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx2;
mod color_group;
mod colors;
mod convolution;
mod convolve_naive_f32;
mod convolve_naive_u16;
mod convolve_naive_u8;
mod cpu_features;
#[cfg(feature = "half")]
mod dispatch_group_f16;
mod dispatch_group_f32;
mod dispatch_group_u16;
mod dispatch_group_u8;
#[cfg(feature = "half")]
mod f16;
mod filter_weights;
mod image_size;
mod image_store;
mod math;
mod nearest_sampler;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod plane_f32;
mod plane_u16;
mod plane_u8;
mod rgb_f32;
mod rgb_u16;
mod rgb_u8;
mod rgba_f32;
mod rgba_u16;
mod rgba_u8;
#[cfg(all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    feature = "riscv"
))]
mod risc;
mod sampler;
mod saturate_narrow;
mod scaler;
#[cfg(feature = "half")]
mod scaler_f16;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod sse;
mod support;
mod threading_policy;
mod unsafe_slice;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;

pub use colors::JzazbzScaler;
pub use colors::LChScaler;
pub use colors::LabScaler;
pub use colors::LinearApproxScaler;
pub use colors::LinearScaler;
pub use colors::OklabScaler;
pub use colors::SigmoidalScaler;
pub use colors::XYZScaler;
pub use colors::*;
pub use colorutils_rs::TransferFunction;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use cpu_features::{is_aarch_f16_supported, is_aarch_f16c_supported};
pub use image_size::ImageSize;
pub use image_store::ImageStore;
pub use math::*;
pub use sampler::*;
pub use scaler::Scaler;
pub use scaler::Scaling;
pub use scaler::ScalingF32;
pub use scaler::ScalingU16;
pub use threading_policy::*;
