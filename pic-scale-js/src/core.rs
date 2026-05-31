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

use crate::metadata::{Metadata, MetadataOptions, Orientation, extract, inject};
use image::{DynamicImage, ImageFormat, ImageReader};

use crate::jxl::{decode_jxl, encode_jxl, is_jxl};
use crate::transpose::apply_orientation;
use pic_scale::{
    ImageSize, ImageStore, ImageStoreMut, PicScaleError, ResamplingFunction, Scaler,
    ThreadingPolicy, WorkloadStrategy,
};
use std::io::Cursor;

#[derive(Debug, thiserror::Error)]
pub enum PicError {
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("pic-scale error: {0:?}")]
    Scale(PicScaleError),
    #[error("unsupported format: {0}")]
    Format(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<PicScaleError> for PicError {
    fn from(e: PicScaleError) -> Self {
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

pub fn is_jxl_format(fmt: &str) -> bool {
    matches!(fmt.to_lowercase().as_str(), "jxl" | "jpegxl")
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

/// Decoded image held in memory. Wraps a `DynamicImage` and exposes
/// decode / resize / encode operations that work on both NAPI and WASM targets.
pub struct CoreImage {
    pub inner: DynamicImage,
    pub metadata: Option<Metadata>,
}

unsafe impl Send for CoreImage {}
unsafe impl Sync for CoreImage {}

impl CoreImage {
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
            use crate::heif::decode_heic;
            let img = decode_heic(bytes)?;
            return Ok(CoreImage {
                inner: img,
                metadata: Some(meta),
            });
        }
        if is_jxl(bytes) {
            let img = decode_jxl(bytes)?;
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
        if dst_w == 0 || dst_w == 0 {
            return Err(PicError::Scale(PicScaleError::Generic(
                "Dst width and height cannot be 0".to_string(),
            )));
        }
        let area_ok = isize::try_from(dst_w)
            .ok()
            .zip(isize::try_from(dst_h).ok())
            .and_then(|(w, h)| w.checked_mul(h))
            .is_some();
        if !area_ok {
            return Err(PicError::Scale(PicScaleError::Generic(
                "Dst width or height overflows isize".to_string(),
            )));
        }
        let func = parse_filter(filter)?;
        let threading = match workers {
            0 => ThreadingPolicy::Adaptive,
            1 => ThreadingPolicy::Single,
            n => ThreadingPolicy::Fixed(n),
        };
        let scaler = Scaler::new(func)
            .set_threading_policy(threading)
            .set_workload_strategy(WorkloadStrategy::PreferSpeed);

        let src_size = ImageSize::new(self.width() as usize, self.height() as usize);
        let dst_size = ImageSize::new(dst_w as usize, dst_h as usize);

        let resized = match &self.inner {
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

        if is_jxl_format(&fmt_lower) {
            let icc = if meta_opts.icc {
                src_meta.and_then(|m| m.icc.as_deref())
            } else {
                None
            };
            return encode_jxl(&self.inner, opts.quality, icc);
        }

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
        if dst_w == 0 || dst_w == 0 {
            return Err(PicError::Scale(PicScaleError::Generic(
                "Dst width and height cannot be 0".to_string(),
            )));
        }
        let area_ok = isize::try_from(dst_w)
            .ok()
            .zip(isize::try_from(dst_h).ok())
            .and_then(|(w, h)| w.checked_mul(h))
            .is_some();
        if !area_ok {
            return Err(PicError::Scale(PicScaleError::Generic(
                "Dst width or height overflows isize".to_string(),
            )));
        }
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
