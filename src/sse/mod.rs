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

#[cfg(feature = "half")]
mod alpha_f16;
mod alpha_f32;
mod alpha_u16;
mod alpha_u8;
mod plane_f32;
mod plane_u8;
mod rgb_f32;
mod rgb_u16;
mod rgb_u8;
#[cfg(feature = "half")]
mod rgba_f16;
mod rgba_f32;
mod rgba_u16;
mod rgba_u8;
mod routines;
mod u8_utils;
mod utils;
#[cfg(feature = "half")]
mod vertical_f16;
mod vertical_f32;
mod vertical_u16;
mod vertical_u8;
#[cfg(feature = "half")]
mod f16_utils;
#[cfg(feature = "half")]
mod rgb_f16;

#[cfg(feature = "half")]
pub use alpha_f16::{sse_premultiply_alpha_rgba_f16, sse_unpremultiply_alpha_rgba_f16};
pub use alpha_f32::sse_premultiply_alpha_rgba_f32;
pub use alpha_f32::sse_unpremultiply_alpha_rgba_f32;
pub use alpha_u16::{premultiply_alpha_sse_rgba_u16, unpremultiply_alpha_sse_rgba_u16};
pub use alpha_u8::sse_premultiply_alpha_rgba;
pub use alpha_u8::sse_unpremultiply_alpha_rgba;
pub use plane_f32::convolve_horizontal_plane_sse_row_one;
pub use plane_f32::convolve_horizontal_plane_sse_rows_4;
pub use plane_u8::{convolve_horizontal_plane_sse_row, convolve_horizontal_plane_sse_rows_4_u8};
pub use rgb_f32::*;
pub use rgb_u16::{convolve_horizontal_rgb_sse_row_u16, convolve_horizontal_rgb_sse_rows_4_u16};
pub use rgb_u8::*;
#[cfg(feature = "half")]
pub use rgba_f16::{
    convolve_horizontal_rgba_sse_row_one_f16, convolve_horizontal_rgba_sse_rows_4_f16,
};
pub use rgba_f32::*;
pub use rgba_u16::{convolve_horizontal_rgba_sse_row_u16, convolve_horizontal_rgba_sse_rows_4_u16};
pub use rgba_u8::{convolve_horizontal_rgba_sse_rows_4, convolve_horizontal_rgba_sse_rows_one};
pub use u8_utils::*;
pub use utils::*;
#[cfg(feature = "half")]
pub use vertical_f16::convolve_vertical_rgb_sse_row_f16;
pub use vertical_f32::convolve_vertical_rgb_sse_row_f32;
pub use vertical_u16::convolve_vertical_rgb_sse_row_u16;
pub use vertical_u8::convolve_vertical_sse_row;
#[cfg(feature = "half")]
pub use rgb_f16::{convolve_horizontal_rgb_sse_rows_4_f16, convolve_horizontal_rgb_sse_row_one_f16};

pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
