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
#[cfg(feature = "nightly_f16")]
mod alpha_f16;
#[cfg(feature = "nightly_f16")]
mod alpha_f16_full;
mod alpha_f32;
mod alpha_u16;
mod alpha_u8;
mod ar30;
#[cfg(feature = "rdm")]
mod cbcr8_rdm;
mod check_alpha;
#[cfg(feature = "nightly_f16")]
mod convolve_f16;
mod horizontal_ar30;
#[cfg(feature = "rdm")]
mod horizontal_ar30_rdm;
mod plane_f32;
mod plane_u16_hb;
mod plane_u8;
#[cfg(feature = "rdm")]
mod plane_u8_rdm;
#[cfg(feature = "nightly_f16")]
mod rgb_f16;
#[cfg(feature = "nightly_f16")]
mod rgb_f16_fhm;
#[cfg(feature = "nightly_f16")]
mod rgb_f16_full;
mod rgb_f32;
#[cfg(feature = "rdm")]
mod rgb_u16_hb;
mod rgb_u16_lb;
mod rgb_u8;
#[cfg(feature = "rdm")]
mod rgb_u8_sqrdml;
#[cfg(feature = "nightly_f16")]
mod rgba_f16;
#[cfg(feature = "nightly_f16")]
mod rgba_f16_fhm;
#[cfg(feature = "nightly_f16")]
mod rgba_f16_full;
mod rgba_f32;
#[cfg(feature = "rdm")]
mod rgba_u16_hb;
mod rgba_u16_lb;
mod rgba_u8;
#[cfg(feature = "rdm")]
mod rgba_u8_rdm;
mod utils;
mod vertical_ar30;
#[cfg(feature = "rdm")]
mod vertical_ar30_rdm;
#[cfg(feature = "nightly_f16")]
mod vertical_f16;
#[cfg(feature = "nightly_f16")]
mod vertical_f16_fhm;
#[cfg(feature = "nightly_f16")]
mod vertical_f16_full;
mod vertical_f32;
mod vertical_u16;
#[cfg(feature = "rdm")]
mod vertical_u16_hb;
mod vertical_u16_lb;
mod vertical_u8;
#[cfg(feature = "rdm")]
mod vertical_u8_rdm;
mod weights;

#[cfg(feature = "nightly_f16")]
pub(crate) use alpha_f16::{neon_premultiply_alpha_rgba_f16, neon_unpremultiply_alpha_rgba_f16};
#[cfg(feature = "nightly_f16")]
pub(crate) use alpha_f16_full::{
    neon_premultiply_alpha_rgba_f16_full, neon_unpremultiply_alpha_rgba_f16_full,
};
pub(crate) use alpha_f32::neon_premultiply_alpha_rgba_f32;
pub(crate) use alpha_f32::neon_unpremultiply_alpha_rgba_f32;
pub(crate) use alpha_u16::{neon_premultiply_alpha_rgba_u16, neon_unpremultiply_alpha_rgba_u16};
pub(crate) use alpha_u8::neon_premultiply_alpha_rgba;
pub(crate) use alpha_u8::neon_unpremultiply_alpha_rgba;
#[cfg(feature = "rdm")]
pub(crate) use cbcr8_rdm::{
    convolve_horizontal_cbcr_neon_rdm_row, convolve_horizontal_cbcr_neon_rows_rdm_4_u8,
};
pub(crate) use check_alpha::{
    neon_has_non_constant_cap_alpha_rgba16, neon_has_non_constant_cap_alpha_rgba8,
};
pub(crate) use horizontal_ar30::{
    neon_convolve_horizontal_rgba_rows_4_ar30, neon_convolve_horizontal_rgba_rows_ar30,
};
#[cfg(feature = "rdm")]
pub(crate) use horizontal_ar30_rdm::neon_convolve_horizontal_rgba_rows_4_ar30_rdm;
pub(crate) use plane_f32::convolve_horizontal_plane_neon_row_one;
pub(crate) use plane_f32::convolve_horizontal_plane_neon_rows_4;
pub(crate) use plane_u8::{
    convolve_horizontal_plane_neon_row, convolve_horizontal_plane_neon_row_q,
    convolve_horizontal_plane_neon_rows_4_u8, convolve_horizontal_plane_neon_rows_4_u8_q,
};
#[cfg(feature = "rdm")]
pub(crate) use plane_u8_rdm::{
    convolve_horizontal_plane_neon_rdm_row, convolve_horizontal_plane_neon_rows_rdm_4_u8,
};
#[cfg(feature = "nightly_f16")]
pub(crate) use rgb_f16::{
    convolve_horizontal_rgb_neon_row_one_f16, convolve_horizontal_rgb_neon_rows_4_f16,
};
#[cfg(feature = "nightly_f16")]
pub(crate) use rgb_f16_fhm::{
    convolve_horizontal_rgb_neon_row_one_f16_fhm, convolve_horizontal_rgb_neon_rows_4_f16_fhm,
};
#[cfg(feature = "nightly_f16")]
pub(crate) use rgb_f16_full::{
    xconvolve_horizontal_rgb_neon_row_one_f16, xconvolve_horizontal_rgb_neon_rows_4_f16,
};
pub(crate) use rgb_f32::{
    convolve_horizontal_rgb_neon_row_one_f32, convolve_horizontal_rgb_neon_rows_4_f32,
};
#[cfg(feature = "rdm")]
pub(crate) use rgb_u16_hb::{
    convolve_horizontal_rgb_neon_rows_4_hb_u16, convolve_horizontal_rgb_neon_u16_hb_row,
};
pub(crate) use rgb_u16_lb::{
    convolve_horizontal_rgb_neon_rows_4_lb_u16, convolve_horizontal_rgb_neon_u16_lb_row,
};
pub(crate) use rgb_u8::{
    convolve_horizontal_rgb_neon_row_one, convolve_horizontal_rgb_neon_row_one_q,
    convolve_horizontal_rgb_neon_rows_4, convolve_horizontal_rgb_neon_rows_4_q,
};
#[cfg(feature = "rdm")]
pub(crate) use rgb_u8_sqrdml::{
    convolve_horizontal_rgb_neon_rdm_row_one, convolve_horizontal_rgb_neon_rdm_rows_4,
};
#[cfg(feature = "nightly_f16")]
pub(crate) use rgba_f16::convolve_horizontal_rgba_neon_row_one_f16;
#[cfg(feature = "nightly_f16")]
pub(crate) use rgba_f16::convolve_horizontal_rgba_neon_rows_4_f16;
#[cfg(feature = "nightly_f16")]
pub(crate) use rgba_f16_fhm::{
    convolve_horizontal_rgba_neon_row_one_f16_fhm, convolve_horizontal_rgba_neon_rows_4_f16_fhm,
};
#[cfg(feature = "nightly_f16")]
pub(crate) use rgba_f16_full::{
    xconvolve_horizontal_rgba_neon_row_one_f16, xconvolve_horizontal_rgba_neon_rows_4_f16,
};
pub(crate) use rgba_f32::{
    convolve_horizontal_rgba_neon_row_one, convolve_horizontal_rgba_neon_rows_4,
};
#[cfg(feature = "rdm")]
pub(crate) use rgba_u16_hb::{
    convolve_horizontal_rgba_neon_rows_4_hb_u16, convolve_horizontal_rgba_neon_u16_hb_row,
};
pub(crate) use rgba_u16_lb::{
    convolve_horizontal_rgba_neon_rows_4_lb_u16, convolve_horizontal_rgba_neon_u16_lb_row,
};
pub(crate) use rgba_u8::{
    convolve_horizontal_rgba_neon_row, convolve_horizontal_rgba_neon_row_q,
    convolve_horizontal_rgba_neon_rows_4_u8, convolve_horizontal_rgba_neon_rows_4_u8_q,
};
#[cfg(feature = "rdm")]
pub(crate) use rgba_u8_rdm::{
    convolve_horizontal_rgba_neon_row_i16, convolve_horizontal_rgba_neon_rows_4_u8_i16,
};
pub(crate) use vertical_ar30::neon_column_handler_fixed_point_ar30;
#[cfg(feature = "rdm")]
pub(crate) use vertical_ar30_rdm::neon_column_handler_fixed_point_ar30_rdm;
#[cfg(feature = "nightly_f16")]
pub(crate) use vertical_f16::convolve_vertical_rgb_neon_row_f16;
#[cfg(feature = "nightly_f16")]
pub(crate) use vertical_f16_fhm::convolve_vertical_rgb_neon_row_f16_fhm;
#[cfg(feature = "nightly_f16")]
pub(crate) use vertical_f16_full::xconvolve_vertical_rgb_neon_row_f16;
pub(crate) use vertical_f32::convolve_vertical_rgb_neon_row_f32;
pub(crate) use vertical_u16::convolve_column_u16;
#[cfg(feature = "rdm")]
pub(crate) use vertical_u16_hb::convolve_column_hb_u16;
pub(crate) use vertical_u16_lb::convolve_column_lb_u16;
pub(crate) use vertical_u8::{
    convolve_vertical_neon_i32_precision, convolve_vertical_neon_i32_precision_d,
};
#[cfg(feature = "rdm")]
pub(crate) use vertical_u8_rdm::convolve_vertical_neon_i16_precision;
#[cfg(feature = "nightly_f16")]
pub(crate) use weights::convert_weights_to_f16_fhm;
