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
mod check_alpha;
#[cfg(feature = "half")]
mod f16_utils;
mod plane_f32;
mod plane_u8;
#[cfg(feature = "half")]
mod rgb_f16;
mod rgb_f32;
mod rgb_u8;
#[cfg(feature = "half")]
mod rgba_f16;
mod rgba_f32;
mod rgba_u16;
mod rgba_u16_lb;
mod rgba_u8;
mod rgba_u8_lb;
mod routines;
mod u8_utils;
mod utils;
#[cfg(feature = "half")]
mod vertical_f16;
mod vertical_f32;
mod vertical_u16;
mod vertical_u16_lb;
mod vertical_u8;
mod vertical_u8_lp;

#[cfg(feature = "half")]
pub(crate) use alpha_f16::{sse_premultiply_alpha_rgba_f16, sse_unpremultiply_alpha_rgba_f16};
pub(crate) use alpha_f32::sse_premultiply_alpha_rgba_f32;
pub(crate) use alpha_f32::sse_unpremultiply_alpha_rgba_f32;
pub(crate) use alpha_u16::{premultiply_alpha_sse_rgba_u16, unpremultiply_alpha_sse_rgba_u16};
pub(crate) use alpha_u8::{
    _mm_div_by_255_epi16, sse_premultiply_alpha_rgba, sse_unpremultiply_alpha_rgba,
    sse_unpremultiply_row,
};
pub(crate) use check_alpha::{
    sse_has_non_constant_cap_alpha_rgba16, sse_has_non_constant_cap_alpha_rgba8,
};
pub(crate) use plane_f32::convolve_horizontal_plane_sse_row_one;
pub(crate) use plane_f32::convolve_horizontal_plane_sse_rows_4;
pub(crate) use plane_u8::{
    convolve_horizontal_plane_sse_row, convolve_horizontal_plane_sse_rows_4_u8,
};
#[cfg(feature = "half")]
pub(crate) use rgb_f16::{
    convolve_horizontal_rgb_sse_row_one_f16, convolve_horizontal_rgb_sse_rows_4_f16,
};
pub(crate) use rgb_f32::{
    convolve_horizontal_rgb_sse_row_one_f32, convolve_horizontal_rgb_sse_rows_4_f32,
};
pub(crate) use rgb_u8::*;
#[cfg(feature = "half")]
pub(crate) use rgba_f16::{
    convolve_horizontal_rgba_sse_row_one_f16, convolve_horizontal_rgba_sse_rows_4_f16,
};
pub(crate) use rgba_f32::{
    convolve_horizontal_rgba_sse_row_one_f32, convolve_horizontal_rgba_sse_rows_4_f32,
};
pub(crate) use rgba_u16::{
    convolve_horizontal_rgba_sse_rows_4_u16, convolve_horizontal_rgba_sse_u16_row,
};
pub(crate) use rgba_u16_lb::{
    convolve_horizontal_rgba_sse_rows_4_lb_u8, convolve_horizontal_rgba_sse_u16_lb_row,
};
pub(crate) use rgba_u8::{
    convolve_horizontal_rgba_sse_rows_4, convolve_horizontal_rgba_sse_rows_one,
};
pub(crate) use rgba_u8_lb::{
    convolve_horizontal_rgba_sse_rows_4_lb, convolve_horizontal_rgba_sse_rows_one_lb,
};
pub(crate) use routines::{load_4_weights, load_4_weights_group_2_avx, load_8_weights_group_4_avx};
pub(crate) use u8_utils::*;
pub(crate) use utils::*;
#[cfg(feature = "half")]
pub(crate) use vertical_f16::convolve_vertical_sse_row_f16;
pub(crate) use vertical_f32::convolve_vertical_rgb_sse_row_f32;
pub(crate) use vertical_u16::convolve_column_sse_u16;
pub(crate) use vertical_u16_lb::convolve_column_lb_sse_u16;
pub(crate) use vertical_u8::convolve_vertical_sse_row;
pub(crate) use vertical_u8_lp::convolve_vertical_sse_row_lp;

pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
