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

use crate::metadata::{Metadata, MetadataOptions, Orientation, apply_orientation, extract, inject};
use image::{DynamicImage, ImageFormat, ImageReader};

use pic_scale::{
    ImageSize, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy,
    WorkloadStrategy,
};
use std::io::Cursor;

#[cfg(not(target_arch = "wasm32"))]
fn to_u16<'a>(bytes: &'a [u8]) -> std::borrow::Cow<'a, [u16]> {
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

#[derive(Debug, thiserror::Error)]
pub enum PicError {
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("pic-scale error: {0:?}")]
    Scale(pic_scale::PicScaleError),
    #[error("unsupported format: {0}")]
    Format(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<pic_scale::PicScaleError> for PicError {
    fn from(e: pic_scale::PicScaleError) -> Self {
        PicError::Scale(e)
    }
}

pub type Result<T> = std::result::Result<T, PicError>;

/// Maps the string filter name (matching JS/Python API) to a ResamplingFunction.
pub fn parse_filter(name: &str) -> Result<ResamplingFunction> {
    match name.to_lowercase().as_str() {
        "nearest" => Ok(ResamplingFunction::Nearest),
        "bilinear" => Ok(ResamplingFunction::Bilinear),
        "bicubic" => Ok(ResamplingFunction::Bicubic),
        "lanczos" | "lanczos3" => Ok(ResamplingFunction::Lanczos3),
        "lanczos2" => Ok(ResamplingFunction::Lanczos2),
        "lanczos4" => Ok(ResamplingFunction::Lanczos4),
        "box" => Ok(ResamplingFunction::Box),
        "hamming" => Ok(ResamplingFunction::Hamming),
        "mitchell" => Ok(ResamplingFunction::MitchellNetravalli),
        "catmull_rom" | "catmullrom" => Ok(ResamplingFunction::CatmullRom),
        "gaussian" => Ok(ResamplingFunction::Gaussian),
        "hann" => Ok(ResamplingFunction::Hann),
        other => Err(PicError::Format(format!("Unknown filter '{other}'"))),
    }
}

pub struct EncodeOptions {
    pub format: String,
    pub quality: u8, // JPEG quality 1–100; HEIF/AVIF quality 1–100
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            format: "png".into(),
            quality: 85,
        }
    }
}

/// Returns true if the format string targets HEIF or AVIF output,
/// which is handled by libheif rather than the image crate.
pub fn is_heif_format(fmt: &str) -> bool {
    matches!(
        fmt.to_lowercase().as_str(),
        "heic" | "heif" | "avif" | "avifs"
    )
}

pub fn parse_format(fmt: &str) -> Result<ImageFormat> {
    match fmt.to_lowercase().as_str() {
        "jpg" | "jpeg" => Ok(ImageFormat::Jpeg),
        "png" => Ok(ImageFormat::Png),
        "webp" => Ok(ImageFormat::WebP),
        "tiff" | "tif" => Ok(ImageFormat::Tiff),
        "bmp" => Ok(ImageFormat::Bmp),
        "ico" => Ok(ImageFormat::Ico),
        "qoi" => Ok(ImageFormat::Qoi),
        // HEIF/AVIF are routed through libheif — not via ImageFormat
        "heic" | "heif" | "avif" | "avifs" => Err(PicError::Format(
            "Use encode_heif() for HEIC/HEIF/AVIF output".into(),
        )),
        other => Err(PicError::Format(format!("Unknown format '{other}'"))),
    }
}

/// Detect HEIC/HEIF by the ISOBMFF `ftyp` box at offset 4–12.
#[cfg(not(target_arch = "wasm32"))]
fn is_heic(bytes: &[u8]) -> bool {
    if bytes.len() < 12 {
        return false;
    }
    if &bytes[4..8] != b"ftyp" {
        return false;
    }
    matches!(
        &bytes[8..12],
        b"heic"
            | b"heix"
            | b"hevc"
            | b"hevx"
            | b"mif1"
            | b"msf1"
            | b"miaf"
            | b"MiHE"
            | b"avif"
            | b"avis"
    )
}

/// Encode a `DynamicImage` to HEIC, HEIF, or AVIF bytes using libheif.
///
/// `format`  — `"heic"` / `"heif"` uses HEVC compression (H.265).
///             `"avif"` / `"avifs"` uses AV1 compression.
/// `quality` — 1–100, maps to libheif's quality scale.
/// `icc`     — optional ICC profile bytes to embed in the output container.
/// `exif`    — optional raw EXIF block (starting with `"Exif\0\0"`) to embed.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn encode_heif(
    img: &DynamicImage,
    format: &str,
    quality: u8,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>> {
    use libheif_rs::{
        Channel, ColorSpace as HColorSpace, CompressionFormat, EncoderQuality, HeifContext,
        Image as HImage, LibHeif, RgbChroma as HRgbChroma,
    };

    let compression = match format.to_lowercase().as_str() {
        "avif" | "avifs" => CompressionFormat::Av1,
        _ => CompressionFormat::Hevc,
    };

    let lib = LibHeif::new();
    let mut encoder = lib
        .encoder_for_format(compression)
        .map_err(|e| PicError::Format(format!("libheif encoder: {e}")))?;
    encoder
        .set_quality(EncoderQuality::Lossy(quality))
        .map_err(|e| PicError::Format(format!("libheif set_quality: {e}")))?;

    // Convert DynamicImage to RGB8 or RGBA8 for libheif input.
    // libheif can handle other bit depths but HEIC/AVIF encode path
    // via libheif works best with 8-bit interleaved input.
    let (rgba_pixels, has_alpha, w, h) = match img {
        DynamicImage::ImageRgba8(i) => (i.as_raw().clone(), true, i.width(), i.height()),
        DynamicImage::ImageRgb8(i) => (i.as_raw().clone(), false, i.width(), i.height()),
        other => {
            if other.color().has_alpha() {
                let i = other.to_rgba8();
                (i.as_raw().clone(), true, i.width(), i.height())
            } else {
                let i = other.to_rgb8();
                (i.as_raw().clone(), false, i.width(), i.height())
            }
        }
    };

    let chroma = if has_alpha {
        HRgbChroma::Rgba
    } else {
        HRgbChroma::Rgb
    };
    let channels = if has_alpha { 4u32 } else { 3u32 };

    let mut heif_img = HImage::new(w, h, HColorSpace::Rgb(chroma))
        .map_err(|e| PicError::Format(format!("libheif new image: {e}")))?;

    // Create the interleaved plane and write pixel data
    let channel = Channel::Interleaved;
    heif_img
        .create_plane(channel, w, h, 8)
        .map_err(|e| PicError::Format(format!("libheif create_plane: {e}")))?;

    {
        let plane = heif_img
            .planes_mut()
            .interleaved
            .ok_or_else(|| PicError::Format("libheif: no interleaved plane".into()))?;
        let stride = plane.stride;
        let dst = plane.data;
        let src_stride = w as usize * channels as usize;

        if stride == src_stride {
            dst[..rgba_pixels.len()].copy_from_slice(&rgba_pixels);
        } else {
            // Pad rows to match libheif's stride
            for row in 0..h as usize {
                let src = &rgba_pixels[row * src_stride..(row + 1) * src_stride];
                dst[row * stride..row * stride + src_stride].copy_from_slice(src);
            }
        }
    }

    // Embed ICC profile if provided
    if let Some(icc_bytes) = icc {
        use libheif_rs::ColorProfileRaw;
        use libheif_rs::color_profile_types::R_ICC;
        heif_img
            .set_color_profile_raw(&ColorProfileRaw::new(R_ICC, icc_bytes.to_vec()))
            .map_err(|e| PicError::Format(format!("libheif set ICC: {e}")))?;
    }

    let mut ctx =
        HeifContext::new().map_err(|e| PicError::Format(format!("libheif context: {e}")))?;
    let handle = ctx
        .encode_image(&heif_img, &mut encoder, None)
        .map_err(|e| PicError::Format(format!("libheif encode: {e}")))?;

    // Embed EXIF block if provided
    if let Some(exif_bytes) = exif {
        // libheif expects raw EXIF starting after the "Exif\0\0" prefix
        let payload = if exif_bytes.starts_with(b"Exif\0\0") {
            &exif_bytes[6..]
        } else {
            exif_bytes
        };
        ctx.add_exif_metadata(&handle, payload)
            .map_err(|e| PicError::Format(format!("libheif add EXIF: {e}")))?;
    }

    let buf = ctx
        .write_to_bytes()
        .map_err(|e| PicError::Format(format!("libheif write: {e}")))?;
    Ok(buf)
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn decode_heic(bytes: &[u8]) -> Result<DynamicImage> {
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

        // ── 8-bit path ───────────────────────────────────────────────────────
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

    // ── Fallback: let libheif do YUV→RGB ─────────────────────────────────────
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

/// Decoded image held in memory. Wraps a `DynamicImage` and exposes
/// decode / resize / encode operations that work on both NAPI and WASM targets.
pub struct CoreImage {
    pub inner: DynamicImage,
    pub metadata: Option<Metadata>,
}

unsafe impl Send for CoreImage {}
unsafe impl Sync for CoreImage {}

impl CoreImage {
    // ── decode ────────────────────────────────────────────────────────────────

    /// Decode an image from raw bytes (any supported format).
    ///
    /// On native targets, HEIC/HEIF is handled by `libheif-rs` which registers
    /// itself as a decoder hook into `image`'s `ImageReader` via its `image`
    /// feature. YUV planar data is converted via the `yuv` crate's SIMD paths.
    /// On WASM, HEIC is not supported.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let meta = extract(bytes);

        // HEIC/HEIF: detect by ISOBMFF ftyp box and call libheif directly.
        // The image crate has no native HEIC support so we bypass ImageReader.
        #[cfg(not(target_arch = "wasm32"))]
        if is_heic(bytes) {
            let img = decode_heic(bytes)?;
            return Ok(CoreImage {
                inner: img,
                metadata: Some(meta),
            });
        }

        let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
        let img = reader.decode()?;
        Ok(CoreImage {
            inner: img,
            metadata: Some(meta),
        })
    }

    /// Decode an image from a file path (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn width(&self) -> u32 {
        self.inner.width()
    }
    pub fn height(&self) -> u32 {
        self.inner.height()
    }
    pub fn channels(&self) -> u8 {
        self.inner.color().channel_count()
    }

    // ── resize ────────────────────────────────────────────────────────────────

    /// Resize to `(dst_w, dst_h)` using the named filter.
    /// `premultiply_alpha` only affects images with an alpha channel.
    pub fn resize(
        &self,
        dst_w: u32,
        dst_h: u32,
        filter: &str,
        premultiply_alpha: bool,
        workers: usize,
    ) -> Result<CoreImage> {
        let func = parse_filter(filter)?;
        let threading = match workers {
            0 => ThreadingPolicy::Adaptive,
            1 => ThreadingPolicy::Single,
            n => ThreadingPolicy::Fixed(n),
        };
        let mut scaler = Scaler::new(func);
        scaler.set_threading_policy(threading);
        scaler.set_workload_strategy(WorkloadStrategy::PreferSpeed);

        let src_size = ImageSize::new(self.width() as usize, self.height() as usize);
        let dst_size = ImageSize::new(dst_w as usize, dst_h as usize);

        let resized = match &self.inner {
            // ── u8 variants ─────────────────────────────────────────────────
            DynamicImage::ImageLuma8(img) => {
                let store = ImageStore::<u8, 1>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst = ImageStoreMut::<u8, 1>::alloc(dst_w as usize, dst_h as usize);
                scaler
                    .plan_planar_resampling(src_size, dst_size)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageLuma8(
                    image::GrayImage::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec()).unwrap(),
                )
            }
            DynamicImage::ImageLumaA8(img) => {
                let store = ImageStore::<u8, 2>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst = ImageStoreMut::<u8, 2>::alloc(dst_w as usize, dst_h as usize);
                scaler
                    .plan_gray_alpha_resampling(src_size, dst_size, premultiply_alpha)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageLumaA8(
                    image::GrayAlphaImage::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec())
                        .unwrap(),
                )
            }
            DynamicImage::ImageRgb8(img) => {
                let store = ImageStore::<u8, 3>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst = ImageStoreMut::<u8, 3>::alloc(dst_w as usize, dst_h as usize);
                scaler
                    .plan_rgb_resampling(src_size, dst_size)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec()).unwrap(),
                )
            }
            DynamicImage::ImageRgba8(img) => {
                let store = ImageStore::<u8, 4>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst = ImageStoreMut::<u8, 4>::alloc(dst_w as usize, dst_h as usize);
                scaler
                    .plan_rgba_resampling(src_size, dst_size, premultiply_alpha)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageRgba8(
                    image::RgbaImage::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec()).unwrap(),
                )
            }
            // ── u16 variants ────────────────────────────────────────────────
            DynamicImage::ImageLuma16(img) => {
                let store = ImageStore::<u16, 1>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst =
                    ImageStoreMut::<u16, 1>::alloc_with_depth(dst_w as usize, dst_h as usize, 16);
                scaler
                    .plan_planar_resampling16(src_size, dst_size, 16)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageLuma16(
                    image::ImageBuffer::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec())
                        .unwrap(),
                )
            }
            DynamicImage::ImageLumaA16(img) => {
                let store = ImageStore::<u16, 2>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst =
                    ImageStoreMut::<u16, 2>::alloc_with_depth(dst_w as usize, dst_h as usize, 16);
                scaler
                    .plan_gray_alpha_resampling16(src_size, dst_size, premultiply_alpha, 16)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageLumaA16(
                    image::ImageBuffer::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec())
                        .unwrap(),
                )
            }
            DynamicImage::ImageRgb16(img) => {
                let store = ImageStore::<u16, 3>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst =
                    ImageStoreMut::<u16, 3>::alloc_with_depth(dst_w as usize, dst_h as usize, 16);
                scaler
                    .plan_rgb_resampling16(src_size, dst_size, 16)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageRgb16(
                    image::ImageBuffer::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec())
                        .unwrap(),
                )
            }
            DynamicImage::ImageRgba16(img) => {
                let store = ImageStore::<u16, 4>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst =
                    ImageStoreMut::<u16, 4>::alloc_with_depth(dst_w as usize, dst_h as usize, 16);
                scaler
                    .plan_rgba_resampling16(src_size, dst_size, premultiply_alpha, 16)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageRgba16(
                    image::ImageBuffer::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec())
                        .unwrap(),
                )
            }
            // ── f32 variants ─────────────────────────────────────────────────
            DynamicImage::ImageRgb32F(img) => {
                let store = ImageStore::<f32, 3>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst = ImageStoreMut::<f32, 3>::alloc(dst_w as usize, dst_h as usize);
                scaler
                    .plan_rgb_resampling_f32(src_size, dst_size)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageRgb32F(
                    image::ImageBuffer::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec())
                        .unwrap(),
                )
            }
            DynamicImage::ImageRgba32F(img) => {
                let store = ImageStore::<f32, 4>::from_slice(
                    img.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst = ImageStoreMut::<f32, 4>::alloc(dst_w as usize, dst_h as usize);
                scaler
                    .plan_rgba_resampling_f32(src_size, dst_size, premultiply_alpha)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageRgba32F(
                    image::ImageBuffer::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec())
                        .unwrap(),
                )
            }
            // fallback: convert to RGBA8 and resize
            other => {
                let rgba = other.to_rgba8();
                let store = ImageStore::<u8, 4>::from_slice(
                    rgba.as_raw(),
                    self.width() as usize,
                    self.height() as usize,
                )?;
                let mut dst = ImageStoreMut::<u8, 4>::alloc(dst_w as usize, dst_h as usize);
                scaler
                    .plan_rgba_resampling(src_size, dst_size, premultiply_alpha)?
                    .resample(&store, &mut dst)?;
                DynamicImage::ImageRgba8(
                    image::RgbaImage::from_raw(dst_w, dst_h, dst.buffer.borrow().to_vec()).unwrap(),
                )
            }
        };

        Ok(CoreImage {
            inner: resized,
            metadata: self.metadata.clone(),
        })
    }

    /// Encode the image to bytes in the given format (no metadata).
    #[allow(unused)]
    pub fn encode(&self, opts: &EncodeOptions) -> Result<Vec<u8>> {
        self.encode_with_metadata(opts, None, &MetadataOptions::strip())
    }

    /// Encode and optionally inject metadata from the source into the output.
    pub fn encode_with_metadata(
        &self,
        opts: &EncodeOptions,
        src_meta: Option<&Metadata>,
        meta_opts: &MetadataOptions,
    ) -> Result<Vec<u8>> {
        let fmt_lower = opts.format.to_lowercase();

        #[cfg(not(target_arch = "wasm32"))]
        if is_heif_format(&fmt_lower) {
            let icc = if meta_opts.icc {
                src_meta.and_then(|m| m.icc.as_deref())
            } else {
                None
            };
            let exif = if meta_opts.exif {
                src_meta.and_then(|m| m.exif.as_deref())
            } else {
                None
            };
            return encode_heif(&self.inner, &fmt_lower, opts.quality, icc, exif);
        }
        #[cfg(target_arch = "wasm32")]
        if is_heif_format(&fmt_lower) {
            return Err(PicError::Format(
                "HEIF/AVIF encoding is not supported in WASM builds".into(),
            ));
        }

        // ── All other formats via the image crate ────────────────────────────
        let fmt = parse_format(&opts.format)?;
        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);

        if fmt == ImageFormat::Jpeg {
            use image::codecs::jpeg::JpegEncoder;
            let encoder = JpegEncoder::new_with_quality(&mut cursor, opts.quality);
            self.inner.write_with_encoder(encoder)?;
        } else {
            self.inner.write_to(&mut cursor, fmt)?;
        }

        if let Some(meta) = src_meta {
            buf = inject(buf, meta, meta_opts)?;
        }
        Ok(buf)
    }

    /// Apply EXIF orientation then resize. Propagates metadata to the result.
    pub fn resize_with_metadata(
        &self,
        dst_w: u32,
        dst_h: u32,
        filter: &str,
        premultiply_alpha: bool,
        workers: usize,
        meta_opts: &MetadataOptions,
    ) -> Result<(CoreImage, Option<Metadata>)> {
        let orientation = self
            .metadata
            .as_ref()
            .map(|m| m.orientation)
            .unwrap_or(Orientation::Normal);
        let oriented = if meta_opts.auto_orient && !orientation.is_normal() {
            CoreImage {
                inner: apply_orientation(self.inner.clone(), orientation),
                metadata: self.metadata.clone(),
            }
        } else {
            CoreImage {
                inner: self.inner.clone(),
                metadata: self.metadata.clone(),
            }
        };

        // 2. Resize
        let resized = oriented.resize(dst_w, dst_h, filter, premultiply_alpha, workers)?;
        let meta = self.metadata.clone();
        Ok((resized, meta))
    }

    /// Save to a file path (native only). Injects ICC+EXIF into the output.
    /// Supports all formats including HEIC, HEIF, and AVIF via libheif.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&self, path: &str, quality: u8) -> Result<()> {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("png");
        let opts = EncodeOptions {
            format: ext.to_string(),
            quality,
        };
        let meta_opts = MetadataOptions {
            icc: true,
            exif: true,
            xmp: false,
            auto_orient: false,
        };
        let bytes = self.encode_with_metadata(&opts, self.metadata.as_ref(), &meta_opts)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ─── resize modes ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeMode {
    /// Stretch to exact (width, height) — ignores aspect ratio.
    Fill,
    /// Scale uniformly to fit inside the box; pad with `bg_color`.
    Fit,
    /// Scale uniformly to cover the box; crop excess from the centre.
    Cover,
    /// Scale to match target width; height adjusts proportionally.
    FitWidth,
    /// Scale to match target height; width adjusts proportionally.
    FitHeight,
}

impl ResizeMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "fill" => Ok(ResizeMode::Fill),
            "fit" => Ok(ResizeMode::Fit),
            "cover" => Ok(ResizeMode::Cover),
            "fit_width" | "fitwidth" => Ok(ResizeMode::FitWidth),
            "fit_height" | "fitheight" => Ok(ResizeMode::FitHeight),
            other => Err(PicError::Format(format!("Unknown resize mode '{other}'"))),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BgColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

fn compute_geometry(
    src_w: u32,
    src_h: u32,
    box_w: u32,
    box_h: u32,
    mode: ResizeMode,
) -> (u32, u32, u32, u32, u32, u32) {
    match mode {
        ResizeMode::Fill => (0, 0, src_w, src_h, box_w, box_h),
        ResizeMode::FitWidth => {
            let scale = box_w as f64 / src_w as f64;
            let h = (src_h as f64 * scale).round() as u32;
            (0, 0, src_w, src_h, box_w, h.max(1))
        }
        ResizeMode::FitHeight => {
            let scale = box_h as f64 / src_h as f64;
            let w = (src_w as f64 * scale).round() as u32;
            (0, 0, src_w, src_h, w.max(1), box_h)
        }
        ResizeMode::Fit => {
            let scale = (box_w as f64 / src_w as f64).min(box_h as f64 / src_h as f64);
            let w = (src_w as f64 * scale).round() as u32;
            let h = (src_h as f64 * scale).round() as u32;
            (0, 0, src_w, src_h, w.max(1), h.max(1))
        }
        ResizeMode::Cover => {
            let scale = (box_w as f64 / src_w as f64).max(box_h as f64 / src_h as f64);
            let crop_w = (box_w as f64 / scale).round() as u32;
            let crop_h = (box_h as f64 / scale).round() as u32;
            let crop_x = (src_w.saturating_sub(crop_w)) / 2;
            let crop_y = (src_h.saturating_sub(crop_h)) / 2;
            (
                crop_x,
                crop_y,
                crop_w.min(src_w),
                crop_h.min(src_h),
                box_w,
                box_h,
            )
        }
    }
}

impl CoreImage {
    #[allow(clippy::too_many_arguments)]
    pub fn resize_mode(
        &self,
        box_w: u32,
        box_h: u32,
        filter: &str,
        mode: ResizeMode,
        premultiply_alpha: bool,
        workers: usize,
        bg: BgColor,
    ) -> Result<CoreImage> {
        let src_w = self.width();
        let src_h = self.height();
        let (crop_x, crop_y, crop_w, crop_h, scale_w, scale_h) =
            compute_geometry(src_w, src_h, box_w, box_h, mode);

        let cropped = if crop_x != 0 || crop_y != 0 || crop_w != src_w || crop_h != src_h {
            CoreImage {
                inner: self.inner.crop_imm(crop_x, crop_y, crop_w, crop_h),
                metadata: self.metadata.clone(),
            }
        } else {
            CoreImage {
                inner: self.inner.clone(),
                metadata: self.metadata.clone(),
            }
        };

        let scaled = cropped.resize(scale_w, scale_h, filter, premultiply_alpha, workers)?;

        if mode == ResizeMode::Fit && (scale_w != box_w || scale_h != box_h) {
            let has_alpha = scaled.channels() == 2 || scaled.channels() == 4;
            let mut canvas = if has_alpha {
                DynamicImage::new_rgba8(box_w, box_h)
            } else {
                DynamicImage::new_rgb8(box_w, box_h)
            };
            if has_alpha {
                for pixel in canvas.as_mut_rgba8().unwrap().pixels_mut() {
                    *pixel = image::Rgba([bg.r, bg.g, bg.b, bg.a]);
                }
            } else {
                for pixel in canvas.as_mut_rgb8().unwrap().pixels_mut() {
                    *pixel = image::Rgb([bg.r, bg.g, bg.b]);
                }
            }
            let paste_x = (box_w.saturating_sub(scale_w)) / 2;
            let paste_y = (box_h.saturating_sub(scale_h)) / 2;
            image::imageops::overlay(&mut canvas, &scaled.inner, paste_x as i64, paste_y as i64);
            return Ok(CoreImage {
                inner: canvas,
                metadata: self.metadata.clone(),
            });
        }

        Ok(scaled)
    }
}
