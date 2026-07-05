/*
 * Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::metadata::Orientation;
use fast_transpose::{
    FlipMode, FlopMode, flip_plane, flip_plane_with_alpha, flip_plane16, flip_plane16_with_alpha,
    flip_rgb, flip_rgb_f32, flip_rgb16, flip_rgba, flip_rgba_f32, flip_rgba16, flop_plane,
    flop_plane_with_alpha, flop_plane16, flop_plane16_with_alpha, flop_rgb, flop_rgb_f32,
    flop_rgb16, flop_rgba, flop_rgba_f32, flop_rgba16, rotate180_plane, rotate180_plane_with_alpha,
    rotate180_plane16, rotate180_plane16_with_alpha, rotate180_rgb, rotate180_rgb_f32,
    rotate180_rgb16, rotate180_rgba, rotate180_rgba_f32, rotate180_rgba16, transpose_plane,
    transpose_plane_with_alpha, transpose_plane16, transpose_plane16_with_alpha, transpose_rgb,
    transpose_rgb_f32, transpose_rgb16, transpose_rgba, transpose_rgba_f32, transpose_rgba16,
};
use image::GenericImageView;

enum Op {
    Flip,
    Flop,
    Rotate180,
    Transpose(FlipMode, FlopMode),
}

macro_rules! apply_op {
    // u8 plane
    ($op:expr, plane_u8, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0u8; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => flip_plane($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Flop => flop_plane($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Rotate180 => {
                rotate180_plane($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Transpose(flip, flop) => {
                transpose_plane($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h, flip, flop).unwrap()
            }
        }
        dst
    }};
    // u8 plane+alpha
    ($op:expr, plane_alpha_u8, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0u8; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => {
                flip_plane_with_alpha($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Flop => {
                flop_plane_with_alpha($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Rotate180 => {
                rotate180_plane_with_alpha($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Transpose(flip, flop) => transpose_plane_with_alpha(
                $buf,
                $w * $cn,
                &mut dst,
                $out_w * $cn,
                $w,
                $h,
                flip,
                flop,
            )
            .unwrap(),
        }
        dst
    }};
    // u8 rgb
    ($op:expr, rgb_u8, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0u8; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => flip_rgb($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Flop => flop_rgb($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Rotate180 => rotate180_rgb($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Transpose(flip, flop) => {
                transpose_rgb($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h, flip, flop).unwrap()
            }
        }
        dst
    }};
    // u8 rgba
    ($op:expr, rgba_u8, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0u8; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => flip_rgba($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Flop => flop_rgba($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Rotate180 => {
                rotate180_rgba($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Transpose(flip, flop) => {
                transpose_rgba($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h, flip, flop).unwrap()
            }
        }
        dst
    }};
    // u16 plane
    ($op:expr, plane_u16, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0u16; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => flip_plane16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Flop => flop_plane16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Rotate180 => {
                rotate180_plane16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Transpose(flip, flop) => {
                transpose_plane16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h, flip, flop)
                    .unwrap()
            }
        }
        dst
    }};
    // u16 plane+alpha
    ($op:expr, plane_alpha_u16, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0u16; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => {
                flip_plane16_with_alpha($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Flop => {
                flop_plane16_with_alpha($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Rotate180 => {
                rotate180_plane16_with_alpha($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h)
                    .unwrap()
            }
            Op::Transpose(flip, flop) => transpose_plane16_with_alpha(
                $buf,
                $w * $cn,
                &mut dst,
                $out_w * $cn,
                $w,
                $h,
                flip,
                flop,
            )
            .unwrap(),
        }
        dst
    }};
    // u16 rgb
    ($op:expr, rgb_u16, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0u16; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => flip_rgb16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Flop => flop_rgb16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Rotate180 => {
                rotate180_rgb16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Transpose(flip, flop) => {
                transpose_rgb16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h, flip, flop).unwrap()
            }
        }
        dst
    }};
    // u16 rgba
    ($op:expr, rgba_u16, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0u16; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => flip_rgba16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Flop => flop_rgba16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Rotate180 => {
                rotate180_rgba16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Transpose(flip, flop) => {
                transpose_rgba16($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h, flip, flop)
                    .unwrap()
            }
        }
        dst
    }};
    // f32 plane
    ($op:expr, plane_f32, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0f32; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => flip_plane_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Flop => flop_plane_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Rotate180 => {
                rotate180_plane_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Transpose(flip, flop) => {
                transpose_plane_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h, flip, flop)
                    .unwrap()
            }
        }
        dst
    }};
    // f32 plane+alpha
    ($op:expr, plane_alpha_f32, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0f32; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => {
                flip_plane_f32_with_alpha($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Flop => {
                flop_plane_f32_with_alpha($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Rotate180 => {
                rotate180_plane_f32_with_alpha($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h)
                    .unwrap()
            }
            Op::Transpose(flip, flop) => transpose_plane_f32_with_alpha(
                $buf,
                $w * $cn,
                &mut dst,
                $out_w * $cn,
                $w,
                $h,
                flip,
                flop,
            )
            .unwrap(),
        }
        dst
    }};
    // f32 rgb
    ($op:expr, rgb_f32, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0f32; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => flip_rgb_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Flop => flop_rgb_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Rotate180 => {
                rotate180_rgb_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Transpose(flip, flop) => {
                transpose_rgb_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h, flip, flop)
                    .unwrap()
            }
        }
        dst
    }};
    // f32 rgba
    ($op:expr, rgba_f32, $buf:expr, $w:expr, $h:expr, $cn:expr, $out_w:expr, $out_h:expr) => {{
        let mut dst = vec![0f32; $out_w * $out_h * $cn];
        match $op {
            Op::Flip => flip_rgba_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Flop => flop_rgba_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap(),
            Op::Rotate180 => {
                rotate180_rgba_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h).unwrap()
            }
            Op::Transpose(flip, flop) => {
                transpose_rgba_f32($buf, $w * $cn, &mut dst, $out_w * $cn, $w, $h, flip, flop)
                    .unwrap()
            }
        }
        dst
    }};
}

/// Apply EXIF orientation to a `DynamicImage`, consuming it and returning
/// the corrected image.  After this the orientation is baked into pixels,
/// so the EXIF tag should be reset to 1 on output.
pub(crate) fn apply_orientation(
    img: image::DynamicImage,
    orientation: Orientation,
) -> image::DynamicImage {
    let op = match orientation {
        Orientation::Normal => return img,
        Orientation::FlipH => Op::Flip,
        Orientation::FlipV => Op::Flop,
        Orientation::Rotate180 => Op::Rotate180,
        Orientation::Rotate90 => Op::Transpose(FlipMode::Flip, FlopMode::Flop),
        Orientation::Rotate270 => Op::Transpose(FlipMode::NoFlip, FlopMode::NoFlop),
        Orientation::Transpose => Op::Transpose(FlipMode::NoFlip, FlopMode::Flop),
        Orientation::Transverse => Op::Transpose(FlipMode::Flip, FlopMode::NoFlop),
    };
    let (width, height) = img.dimensions();
    let (w, h) = (width as usize, height as usize);
    let (out_w, out_h) = match orientation {
        Orientation::Transpose => (h, w),
        Orientation::Rotate90 => (h, w),
        Orientation::Rotate270 => (h, w),
        _ => (w, h),
    };

    use image::DynamicImage::*;
    match img {
        ImageLuma8(buf) => ImageLuma8(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, plane_u8, buf.as_raw(), w, h, 1, out_w, out_h),
            )
            .unwrap(),
        ),
        ImageLumaA8(buf) => ImageLumaA8(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, plane_alpha_u8, buf.as_raw(), w, h, 2, out_w, out_h),
            )
            .unwrap(),
        ),
        ImageRgb8(buf) => ImageRgb8(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, rgb_u8, buf.as_raw(), w, h, 3, out_w, out_h),
            )
            .unwrap(),
        ),
        ImageRgba8(buf) => ImageRgba8(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, rgba_u8, buf.as_raw(), w, h, 4, out_w, out_h),
            )
            .unwrap(),
        ),
        ImageLuma16(buf) => ImageLuma16(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, plane_u16, buf.as_raw(), w, h, 1, out_w, out_h),
            )
            .unwrap(),
        ),
        ImageLumaA16(buf) => ImageLumaA16(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, plane_alpha_u16, buf.as_raw(), w, h, 2, out_w, out_h),
            )
            .unwrap(),
        ),
        ImageRgb16(buf) => ImageRgb16(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, rgb_u16, buf.as_raw(), w, h, 3, out_w, out_h),
            )
            .unwrap(),
        ),
        ImageRgba16(buf) => ImageRgba16(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, rgba_u16, buf.as_raw(), w, h, 4, out_w, out_h),
            )
            .unwrap(),
        ),
        ImageRgb32F(buf) => ImageRgb32F(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, rgb_f32, buf.as_raw(), w, h, 3, out_w, out_h),
            )
            .unwrap(),
        ),
        ImageRgba32F(buf) => ImageRgba32F(
            image::ImageBuffer::from_raw(
                out_w as u32,
                out_h as u32,
                apply_op!(op, rgba_f32, buf.as_raw(), w, h, 4, out_w, out_h),
            )
            .unwrap(),
        ),
        // fallback for any future variants
        other => apply_orientation(ImageRgba8(other.to_rgba8()), orientation),
    }
}
