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
#[cfg(feature = "half")]
mod rgba_f16;
mod rgba_f32;
mod rgba_u8_lb;
pub(crate) mod utils;
#[cfg(feature = "half")]
mod vertical_f16;
mod vertical_f32;
mod vertical_u16_lb;
mod vertical_u8;
mod vertical_u8_lp;

#[cfg(feature = "half")]
pub(crate) use alpha_f16::{avx_premultiply_alpha_rgba_f16, avx_unpremultiply_alpha_rgba_f16};
pub(crate) use alpha_f32::avx_premultiply_alpha_rgba_f32;
pub(crate) use alpha_f32::avx_unpremultiply_alpha_rgba_f32;
pub(crate) use alpha_u16::{avx_premultiply_alpha_rgba_u16, avx_unpremultiply_alpha_rgba_u16};
pub(crate) use alpha_u8::avx_premultiply_alpha_rgba;
pub(crate) use alpha_u8::avx_unpremultiply_alpha_rgba;
#[cfg(feature = "half")]
pub(crate) use rgba_f16::{
    convolve_horizontal_rgba_avx_row_one_f16, convolve_horizontal_rgba_avx_rows_4_f16,
};
pub(crate) use rgba_f32::{
    convolve_horizontal_rgba_avx_row_one_f32, convolve_horizontal_rgba_avx_rows_4_f32,
};
pub(crate) use rgba_u8_lb::{
    convolve_horizontal_rgba_avx_rows_4_lb, convolve_horizontal_rgba_avx_rows_one_lb,
};
#[cfg(feature = "half")]
pub(crate) use vertical_f16::convolve_vertical_avx_row_f16;
pub(crate) use vertical_f32::convolve_vertical_avx_row_f32;
pub(crate) use vertical_u16_lb::convolve_column_lb_avx2_u16;
pub(crate) use vertical_u8::convolve_vertical_avx_row;
pub(crate) use vertical_u8_lp::convolve_vertical_avx_row_lp;
