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
#![cfg(not(target_arch = "wasm32"))]
use crate::core::PicError;
use image::DynamicImage;

fn to_u16(bytes: &[u8]) -> std::borrow::Cow<'_, [u16]> {
    if bytes.len().is_multiple_of(2) && (bytes.as_ptr() as usize).is_multiple_of(align_of::<u16>())
    {
        return std::borrow::Cow::Borrowed(bytemuck::cast_slice(bytes));
    }
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn expand_to_16bit(buffer: &mut [u16], is12_bit: bool) {
    if is12_bit {
        for px in buffer.iter_mut() {
            *px = (*px << 4) | (*px >> 8);
        }
    } else {
        for px in buffer.iter_mut() {
            *px = (*px << 6) | (*px >> 4);
        }
    }
}

pub(crate) fn decode_heic(bytes: &[u8]) -> crate::core::Result<DynamicImage> {
    use libheif_rs::{Chroma, ColorSpace, HeifContext, LibHeif, MatrixCoefficients, RgbChroma};
    use yuv::{
        YuvPlanarImage, YuvPlanarImageWithAlpha, YuvRange, YuvStandardMatrix, i010_alpha_to_rgba10,
        i010_to_rgb10, i012_alpha_to_rgba12, i012_to_rgb12, i210_alpha_to_rgba10, i210_to_rgb10,
        i212_alpha_to_rgba12, i212_to_rgb12, i410_alpha_to_rgba10, i410_to_rgb10,
        i412_alpha_to_rgba12, i412_to_rgb12, icgc010_alpha_to_rgba10, icgc010_to_rgb10,
        icgc012_alpha_to_rgba12, icgc012_to_rgb12, icgc210_alpha_to_rgba10, icgc210_to_rgb10,
        icgc212_alpha_to_rgba12, icgc212_to_rgb12, icgc410_alpha_to_rgba10, icgc410_to_rgb10,
        icgc412_alpha_to_rgba12, icgc412_to_rgb12, ycgco420_alpha_to_rgba, ycgco420_to_rgb,
        ycgco422_alpha_to_rgba, ycgco422_to_rgb, ycgco444_alpha_to_rgba, ycgco444_to_rgb,
        yuv420_alpha_to_rgba, yuv420_to_rgb, yuv422_alpha_to_rgba, yuv422_to_rgb,
        yuv444_alpha_to_rgba, yuv444_to_rgb,
    };

    let lib = LibHeif::new();
    let ctx = HeifContext::read_from_bytes(bytes)
        .map_err(|e| PicError::Format(format!("libheif: {e}")))?;
    let handle = ctx
        .primary_image_handle()
        .map_err(|e| PicError::Format(format!("libheif handle: {e}")))?;

    let has_alpha = handle.has_alpha_channel();
    let w = handle.width();
    let h = handle.height();

    let mut matrix = YuvStandardMatrix::Bt601;
    let mut range = YuvRange::Limited;
    let mut is_ycgco = false;

    if let Some(nclx) = handle.color_profile_nclx() {
        matrix = match nclx.matrix_coefficients() {
            MatrixCoefficients::ITU_R_BT_709_5 => YuvStandardMatrix::Bt709,
            MatrixCoefficients::ITU_R_BT_2020_2_NonConstantLuminance => YuvStandardMatrix::Bt2020,
            MatrixCoefficients::ITU_R_BT_2020_2_ConstantLuminance => YuvStandardMatrix::Bt2020,
            MatrixCoefficients::YCgCo => {
                is_ycgco = true;
                YuvStandardMatrix::Bt601
            }
            _ => YuvStandardMatrix::Bt601,
        };
        range = if nclx.full_range_flag() == 1 {
            YuvRange::Full
        } else {
            YuvRange::Limited
        };
    }

    let rgb_stride = w * 3;
    let rgba_stride = w * 4;
    let bit_depth = handle.luma_bits_per_pixel() as u32;
    let high_bit = bit_depth > 8;

    for chroma in [Chroma::C420, Chroma::C422, Chroma::C444] {
        let Ok(plane) = lib.decode(&handle, ColorSpace::YCbCr(chroma), None) else {
            continue;
        };
        let planes = plane.planes();
        let (Some(y_p), Some(cb_p), Some(cr_p)) = (planes.y, planes.cb, planes.cr) else {
            continue;
        };

        // ── High bit-depth (10 / 12 bit) ─────────────────────────────────────
        if high_bit {
            let y16 = to_u16(y_p.data);
            let cb16 = to_u16(cb_p.data);
            let cr16 = to_u16(cr_p.data);
            let ys = (y_p.stride / 2) as u32;
            let us = (cb_p.stride / 2) as u32;
            let vs = (cr_p.stride / 2) as u32;
            let is_12 = bit_depth >= 12;

            if is_ycgco {
                if let Some(a_p) = planes.a {
                    let a16 = to_u16(a_p.data);
                    let yuva = YuvPlanarImageWithAlpha {
                        y_plane: &y16,
                        y_stride: ys,
                        u_plane: &cb16,
                        u_stride: us,
                        v_plane: &cr16,
                        v_stride: vs,
                        a_plane: &a16,
                        a_stride: (a_p.stride / 2) as u32,
                        width: w,
                        height: h,
                    };
                    let mut out = vec![0u16; (w * 4 * h) as usize];
                    match (chroma, is_12) {
                        (Chroma::C420, false) => {
                            icgc010_alpha_to_rgba10(&yuva, &mut out, w * 4, range)
                        }
                        (Chroma::C422, false) => {
                            icgc210_alpha_to_rgba10(&yuva, &mut out, w * 4, range)
                        }
                        (_, false) => icgc410_alpha_to_rgba10(&yuva, &mut out, w * 4, range),
                        (Chroma::C420, true) => {
                            icgc012_alpha_to_rgba12(&yuva, &mut out, w * 4, range)
                        }
                        (Chroma::C422, true) => {
                            icgc212_alpha_to_rgba12(&yuva, &mut out, w * 4, range)
                        }
                        (_, true) => icgc412_alpha_to_rgba12(&yuva, &mut out, w * 4, range),
                    }
                    .map_err(|e| PicError::Format(format!("icgc→rgba16: {e}")))?;
                    expand_to_16bit(&mut out, is_12);
                    return Ok(DynamicImage::ImageRgba16(
                        image::ImageBuffer::from_raw(w, h, out)
                            .ok_or_else(|| PicError::Format("HEIC RGBA16 mismatch".into()))?,
                    ));
                }
                let yuv = YuvPlanarImage {
                    y_plane: &y16,
                    y_stride: ys,
                    u_plane: &cb16,
                    u_stride: us,
                    v_plane: &cr16,
                    v_stride: vs,
                    width: w,
                    height: h,
                };
                let mut out = vec![0u16; (w * 3 * h) as usize];
                match (chroma, is_12) {
                    (Chroma::C420, false) => icgc010_to_rgb10(&yuv, &mut out, w * 3, range),
                    (Chroma::C422, false) => icgc210_to_rgb10(&yuv, &mut out, w * 3, range),
                    (_, false) => icgc410_to_rgb10(&yuv, &mut out, w * 3, range),
                    (Chroma::C420, true) => icgc012_to_rgb12(&yuv, &mut out, w * 3, range),
                    (Chroma::C422, true) => icgc212_to_rgb12(&yuv, &mut out, w * 3, range),
                    (_, true) => icgc412_to_rgb12(&yuv, &mut out, w * 3, range),
                }
                .map_err(|e| PicError::Format(format!("icgc→rgb16: {e}")))?;
                expand_to_16bit(&mut out, is_12);
                return Ok(DynamicImage::ImageRgb16(
                    image::ImageBuffer::from_raw(w, h, out)
                        .ok_or_else(|| PicError::Format("HEIC RGB16 mismatch".into()))?,
                ));
            }

            if let Some(a_p) = planes.a {
                let a16 = to_u16(a_p.data);
                let yuva = YuvPlanarImageWithAlpha {
                    y_plane: &y16,
                    y_stride: ys,
                    u_plane: &cb16,
                    u_stride: us,
                    v_plane: &cr16,
                    v_stride: vs,
                    a_plane: &a16,
                    a_stride: (a_p.stride / 2) as u32,
                    width: w,
                    height: h,
                };
                let mut out = vec![0u16; (w * 4 * h) as usize];
                match (chroma, is_12) {
                    (Chroma::C420, false) => {
                        i010_alpha_to_rgba10(&yuva, &mut out, w * 4, range, matrix)
                    }
                    (Chroma::C422, false) => {
                        i210_alpha_to_rgba10(&yuva, &mut out, w * 4, range, matrix)
                    }
                    (_, false) => i410_alpha_to_rgba10(&yuva, &mut out, w * 4, range, matrix),
                    (Chroma::C420, true) => {
                        i012_alpha_to_rgba12(&yuva, &mut out, w * 4, range, matrix)
                    }
                    (Chroma::C422, true) => {
                        i212_alpha_to_rgba12(&yuva, &mut out, w * 4, range, matrix)
                    }
                    (_, true) => i412_alpha_to_rgba12(&yuva, &mut out, w * 4, range, matrix),
                }
                .map_err(|e| PicError::Format(format!("yuv→rgba16: {e}")))?;
                expand_to_16bit(&mut out, is_12);
                return Ok(DynamicImage::ImageRgba16(
                    image::ImageBuffer::from_raw(w, h, out)
                        .ok_or_else(|| PicError::Format("HEIC RGBA16 mismatch".into()))?,
                ));
            }

            let yuv = YuvPlanarImage {
                y_plane: &y16,
                y_stride: ys,
                u_plane: &cb16,
                u_stride: us,
                v_plane: &cr16,
                v_stride: vs,
                width: w,
                height: h,
            };
            let mut out = vec![0u16; (w * 3 * h) as usize];
            match (chroma, is_12) {
                (Chroma::C420, false) => i010_to_rgb10(&yuv, &mut out, w * 3, range, matrix),
                (Chroma::C422, false) => i210_to_rgb10(&yuv, &mut out, w * 3, range, matrix),
                (_, false) => i410_to_rgb10(&yuv, &mut out, w * 3, range, matrix),
                (Chroma::C420, true) => i012_to_rgb12(&yuv, &mut out, w * 3, range, matrix),
                (Chroma::C422, true) => i212_to_rgb12(&yuv, &mut out, w * 3, range, matrix),
                (_, true) => i412_to_rgb12(&yuv, &mut out, w * 3, range, matrix),
            }
            .map_err(|e| PicError::Format(format!("yuv→rgb16: {e}")))?;
            expand_to_16bit(&mut out, is_12);
            return Ok(DynamicImage::ImageRgb16(
                image::ImageBuffer::from_raw(w, h, out)
                    .ok_or_else(|| PicError::Format("HEIC RGB16 mismatch".into()))?,
            ));
        }

        return if is_ycgco {
            if has_alpha && let Some(a_p) = planes.a {
                let yuv = YuvPlanarImageWithAlpha {
                    y_plane: y_p.data,
                    y_stride: y_p.stride as u32,
                    u_plane: cb_p.data,
                    u_stride: cb_p.stride as u32,
                    v_plane: cr_p.data,
                    v_stride: cr_p.stride as u32,
                    a_plane: a_p.data,
                    a_stride: a_p.stride as u32,
                    width: w,
                    height: h,
                };
                let mut rgba = vec![0u8; (rgba_stride * h) as usize];
                match chroma {
                    Chroma::C420 => ycgco420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    Chroma::C422 => ycgco422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    _ => ycgco444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                }
                .map_err(|e| PicError::Format(format!("ycgco→rgba: {e}")))?;
                Ok(DynamicImage::ImageRgba8(
                    image::RgbaImage::from_raw(w, h, rgba)
                        .ok_or_else(|| PicError::Format("HEIC RGBA mismatch".into()))?,
                ))
            } else {
                let yuv = YuvPlanarImage {
                    y_plane: y_p.data,
                    y_stride: y_p.stride as u32,
                    u_plane: cb_p.data,
                    u_stride: cb_p.stride as u32,
                    v_plane: cr_p.data,
                    v_stride: cr_p.stride as u32,
                    width: w,
                    height: h,
                };
                let mut rgb = vec![0u8; (rgb_stride * h) as usize];
                match chroma {
                    Chroma::C420 => ycgco420_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                    Chroma::C422 => ycgco422_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                    _ => ycgco444_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                }
                .map_err(|e| PicError::Format(format!("ycgco→rgb: {e}")))?;
                Ok(DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(w, h, rgb)
                        .ok_or_else(|| PicError::Format("HEIC RGB mismatch".into()))?,
                ))
            }
        } else if has_alpha && let Some(a_p) = planes.a {
            let yuv = YuvPlanarImageWithAlpha {
                y_plane: y_p.data,
                y_stride: y_p.stride as u32,
                u_plane: cb_p.data,
                u_stride: cb_p.stride as u32,
                v_plane: cr_p.data,
                v_stride: cr_p.stride as u32,
                a_plane: a_p.data,
                a_stride: a_p.stride as u32,
                width: w,
                height: h,
            };
            let mut rgba = vec![0u8; (rgba_stride * h) as usize];
            match chroma {
                Chroma::C420 => {
                    yuv420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false)
                }
                Chroma::C422 => {
                    yuv422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false)
                }
                _ => yuv444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
            }
            .map_err(|e| PicError::Format(format!("yuv→rgba: {e}")))?;
            Ok(DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(w, h, rgba)
                    .ok_or_else(|| PicError::Format("HEIC RGBA mismatch".into()))?,
            ))
        } else {
            let yuv = YuvPlanarImage {
                y_plane: y_p.data,
                y_stride: y_p.stride as u32,
                u_plane: cb_p.data,
                u_stride: cb_p.stride as u32,
                v_plane: cr_p.data,
                v_stride: cr_p.stride as u32,
                width: w,
                height: h,
            };
            let mut rgb = vec![0u8; (rgb_stride * h) as usize];
            match chroma {
                Chroma::C420 => yuv420_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
                Chroma::C422 => yuv422_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
                _ => yuv444_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
            }
            .map_err(|e| PicError::Format(format!("yuv→rgb: {e}")))?;
            Ok(DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(w, h, rgb)
                    .ok_or_else(|| PicError::Format("HEIC RGB mismatch".into()))?,
            ))
        };
    }

    let chroma = if has_alpha {
        RgbChroma::Rgba
    } else {
        RgbChroma::Rgb
    };
    let plane = lib
        .decode(&handle, ColorSpace::Rgb(chroma), None)
        .map_err(|e| PicError::Format(format!("libheif RGB fallback: {e}")))?;
    let il = plane
        .planes()
        .interleaved
        .ok_or_else(|| PicError::Format("libheif: no interleaved plane".into()))?;

    let channels = if has_alpha { 4usize } else { 3 };
    let packed = if il.stride == w as usize * channels {
        il.data.to_vec()
    } else {
        let mut out = Vec::with_capacity(w as usize * h as usize * channels);
        for row in 0..h as usize {
            out.extend_from_slice(
                &il.data[row * il.stride..row * il.stride + w as usize * channels],
            );
        }
        out
    };

    Ok(if has_alpha {
        DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(w, h, packed)
                .ok_or_else(|| PicError::Format("HEIC RGBA mismatch".into()))?,
        )
    } else {
        DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(w, h, packed)
                .ok_or_else(|| PicError::Format("HEIC RGB mismatch".into()))?,
        )
    })
}
