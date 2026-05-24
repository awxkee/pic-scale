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

use crate::core::PicError;
use img_parts::{ImageICC, jpeg::Jpeg, png::Png, webp::WebP};
use std::io::Cursor;

// ─── orientation ─────────────────────────────────────────────────────────────

/// EXIF orientation value (1–8).  Maps to the operations needed to make the
/// image "upright" — i.e. what must be done to the pixels before resizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Orientation {
    /// No rotation needed (EXIF 1, or no EXIF).
    #[default]
    Normal,
    /// Flip horizontal (EXIF 2).
    FlipH,
    /// Rotate 180° (EXIF 3).
    Rotate180,
    /// Flip vertical (EXIF 4).
    FlipV,
    /// Transpose: flip along top-left→bottom-right diagonal (EXIF 5).
    Transpose,
    /// Rotate 90° clockwise (EXIF 6).
    Rotate90,
    /// Transverse: flip along top-right→bottom-left diagonal (EXIF 7).
    Transverse,
    /// Rotate 270° clockwise / 90° counter-clockwise (EXIF 8).
    Rotate270,
}

impl Orientation {
    pub fn from_exif_value(v: u32) -> Self {
        match v {
            2 => Orientation::FlipH,
            3 => Orientation::Rotate180,
            4 => Orientation::FlipV,
            5 => Orientation::Transpose,
            6 => Orientation::Rotate90,
            7 => Orientation::Transverse,
            8 => Orientation::Rotate270,
            _ => Orientation::Normal,
        }
    }

    /// Reset tag value — after we've applied the orientation we write back `1`
    /// so downstream readers don't rotate again.
    pub fn is_normal(self) -> bool {
        self == Orientation::Normal
    }
}

/// Read the EXIF orientation from raw encoded image bytes.
/// Returns `Orientation::Normal` if no EXIF or no orientation tag is found.
pub fn read_orientation(bytes: &[u8]) -> Orientation {
    let mut cursor = Cursor::new(bytes);
    let reader = match exif::Reader::new().read_from_container(&mut cursor) {
        Ok(r) => r,
        Err(_) => return Orientation::Normal,
    };
    reader
        .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .map(Orientation::from_exif_value)
        .unwrap_or_default()
}

/// Apply EXIF orientation to a `DynamicImage`, consuming it and returning
/// the corrected image.  After this the orientation is baked into pixels,
/// so the EXIF tag should be reset to 1 on output.
pub fn apply_orientation(
    img: image::DynamicImage,
    orientation: Orientation,
) -> image::DynamicImage {
    use image::imageops;
    match orientation {
        Orientation::Normal => img,
        Orientation::FlipH => img.fliph(),
        Orientation::Rotate180 => img.rotate180(),
        Orientation::FlipV => img.flipv(),
        Orientation::Transpose => {
            // transpose = rotate90 + fliph
            image::DynamicImage::ImageRgba8(imageops::flip_horizontal(&img.rotate90()))
        }
        Orientation::Rotate90 => img.rotate90(),
        Orientation::Transverse => {
            // transverse = rotate270 + fliph
            image::DynamicImage::ImageRgba8(imageops::flip_horizontal(&img.rotate270()))
        }
        Orientation::Rotate270 => img.rotate270(),
    }
}

// ─── metadata options ─────────────────────────────────────────────────────────

/// Which metadata to carry from source to destination.
#[derive(Debug, Clone, Default)]
pub struct MetadataOptions {
    /// Copy ICC colour profile.  Default `true`.
    pub icc: bool,
    /// Copy EXIF block (excluding orientation tag which is reset to 1).
    pub exif: bool,
    /// Copy XMP block.
    pub xmp: bool,
    /// Auto-rotate based on EXIF orientation before resizing.  Default `true`.
    pub auto_orient: bool,
}

impl MetadataOptions {
    /// Carry everything — ICC, EXIF (orientation reset), XMP, auto-rotate.
    #[allow(unused)]
    pub fn all() -> Self {
        MetadataOptions {
            icc: true,
            exif: true,
            xmp: true,
            auto_orient: true,
        }
    }

    /// Strip all metadata — smallest output file.
    #[allow(unused)]
    pub fn strip() -> Self {
        MetadataOptions::default()
    }
}

// ─── metadata extraction ──────────────────────────────────────────────────────

/// Raw metadata blobs extracted from a source image.
#[derive(Debug, Default, Clone)]
pub struct Metadata {
    pub icc: Option<Vec<u8>>,
    pub exif: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    pub orientation: Orientation,
}

/// Detect format and extract metadata blobs from raw encoded bytes.
pub fn extract(src_bytes: &[u8]) -> Metadata {
    let orientation = read_orientation(src_bytes);
    let mut meta = Metadata {
        orientation,
        ..Default::default()
    };

    // Detect format by magic bytes
    if src_bytes.starts_with(b"\xff\xd8") {
        extract_jpeg(src_bytes, &mut meta);
    } else if src_bytes.starts_with(b"\x89PNG") {
        extract_png(src_bytes, &mut meta);
    } else if src_bytes.starts_with(b"RIFF") && src_bytes.get(8..12) == Some(b"WEBP") {
        extract_webp(src_bytes, &mut meta);
    }
    // TIFF / other formats: orientation already read via kamadak-exif above;
    // ICC/XMP not yet extracted for those formats.

    meta
}

fn extract_jpeg(bytes: &[u8], meta: &mut Metadata) {
    if let Ok(jpeg) = Jpeg::from_bytes(bytes.to_vec().into()) {
        meta.icc = jpeg.icc_profile().map(|b| b.to_vec());
        // EXIF segment: APP1 starting with "Exif\0\0"
        meta.exif = jpeg.segments().iter().find_map(|seg| {
            let d = seg.contents();
            if d.starts_with(b"Exif\0\0") {
                Some(d.to_vec())
            } else {
                None
            }
        });
        // XMP segment: APP1 starting with the XMP namespace URI
        meta.xmp = jpeg.segments().iter().find_map(|seg| {
            let d = seg.contents();
            if d.starts_with(b"http://ns.adobe.com/xap/1.0/\0") {
                Some(d.to_vec())
            } else {
                None
            }
        });
    }
}

fn extract_png(bytes: &[u8], meta: &mut Metadata) {
    if let Ok(png) = Png::from_bytes(bytes.to_vec().into()) {
        meta.icc = png.icc_profile().map(|b| b.to_vec());
    }
}

fn extract_webp(bytes: &[u8], meta: &mut Metadata) {
    if let Ok(webp) = WebP::from_bytes(bytes.to_vec().into()) {
        meta.icc = webp.icc_profile().map(|b| b.to_vec());
    }
}

// ─── metadata injection ───────────────────────────────────────────────────────

/// Inject metadata blobs into encoded destination bytes.
///
/// Call this AFTER encoding the resized image. The encoded bytes are parsed,
/// the selected metadata chunks are inserted, and new bytes are returned.
pub fn inject(
    dst_bytes: Vec<u8>,
    meta: &Metadata,
    opts: &MetadataOptions,
) -> Result<Vec<u8>, PicError> {
    if !opts.icc && !opts.exif && !opts.xmp {
        return Ok(dst_bytes);
    }

    // Detect destination format
    if dst_bytes.starts_with(b"\xff\xd8") {
        inject_jpeg(dst_bytes, meta, opts)
    } else if dst_bytes.starts_with(b"\x89PNG") {
        inject_png(dst_bytes, meta, opts)
    } else if dst_bytes.starts_with(b"RIFF") && dst_bytes.get(8..12) == Some(b"WEBP") {
        inject_webp(dst_bytes, meta, opts)
    } else {
        // Format not supported for metadata injection — return as-is
        Ok(dst_bytes)
    }
}

fn inject_jpeg(
    dst_bytes: Vec<u8>,
    meta: &Metadata,
    opts: &MetadataOptions,
) -> Result<Vec<u8>, PicError> {
    let mut jpeg =
        Jpeg::from_bytes(dst_bytes.into()).map_err(|e| PicError::Format(e.to_string()))?;

    if opts.icc
        && let Some(icc) = &meta.icc
    {
        jpeg.set_icc_profile(Some(icc.clone().into()));
    }
    // EXIF and XMP segments are inserted as raw APP1 segments.
    // img-parts handles the segment framing automatically.
    if opts.exif
        && let Some(exif) = &meta.exif
    {
        use img_parts::jpeg::{JpegSegment, markers};
        // Remove any existing EXIF APP1 first to avoid duplicates
        jpeg.segments_mut()
            .retain(|s| s.marker() != markers::APP1 || !s.contents().starts_with(b"Exif\0\0"));
        let seg = JpegSegment::new_with_contents(markers::APP1, exif.clone().into());
        // Insert after SOI (index 0)
        jpeg.segments_mut().insert(1, seg);
    }
    if opts.xmp
        && let Some(xmp) = &meta.xmp
    {
        use img_parts::jpeg::{JpegSegment, markers};
        jpeg.segments_mut().retain(|s| {
            s.marker() != markers::APP1
                || !s.contents().starts_with(b"http://ns.adobe.com/xap/1.0/\0")
        });
        let seg = JpegSegment::new_with_contents(markers::APP1, xmp.clone().into());
        jpeg.segments_mut().insert(1, seg);
    }

    Ok(jpeg.encoder().bytes().to_vec())
}

fn inject_png(
    dst_bytes: Vec<u8>,
    meta: &Metadata,
    opts: &MetadataOptions,
) -> Result<Vec<u8>, PicError> {
    let mut png = Png::from_bytes(dst_bytes.into()).map_err(|e| PicError::Format(e.to_string()))?;
    if opts.icc
        && let Some(icc) = &meta.icc
    {
        png.set_icc_profile(Some(icc.clone().into()));
    }
    Ok(png.encoder().bytes().to_vec())
}

fn inject_webp(
    dst_bytes: Vec<u8>,
    meta: &Metadata,
    opts: &MetadataOptions,
) -> Result<Vec<u8>, PicError> {
    let mut webp =
        WebP::from_bytes(dst_bytes.into()).map_err(|e| PicError::Format(e.to_string()))?;
    if opts.icc
        && let Some(icc) = &meta.icc
    {
        webp.set_icc_profile(Some(icc.clone().into()));
    }
    Ok(webp.encoder().bytes().to_vec())
}
