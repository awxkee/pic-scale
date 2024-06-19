/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

mod acceleration_feature;
mod alpha_handle;
mod avx2_utils;
mod chunking;
mod colors;
mod convolution;
mod convolve_naive_f32;
mod convolve_naive_u8;
mod dispatch_group_f32;
mod dispatch_group_u8;
mod filter_weights;
mod image_size;
mod image_store;
mod math;
mod nearest_sampler;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod rgb_f32;
mod rgb_u8;
mod rgba_f32;
mod rgba_u8;
mod sampler;
mod scaler;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "sse4.1"
))]
mod sse;
mod support;
mod threading_policy;
mod unsafe_slice;

pub use colors::LChScaler;
pub use colors::LabScaler;
pub use colors::LinearApproxScaler;
pub use colors::LinearScaler;
pub use colors::SigmoidalScaler;
pub use colors::XYZScaler;
pub use colors::*;
pub use colorutils_rs::TransferFunction;
pub use image_size::ImageSize;
pub use image_store::ImageStore;
pub use math::*;
pub use sampler::*;
pub use scaler::Scaler;
pub use scaler::Scaling;
pub use threading_policy::*;
