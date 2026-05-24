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

use crate::core::{BgColor, CoreImage, EncodeOptions, ResizeMode};
use crate::metadata::MetadataOptions;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

#[derive(Debug, Clone)]
#[napi(string_enum)]
pub enum Workers {
    Adaptive,
    Single,
}

fn map_err(e: crate::core::PicError) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}

// ─── option structs ───────────────────────────────────────────────────────────

#[napi(object)]
pub struct ResizeOptions {
    /// Resampling filter. Default `"lanczos"`.
    /// One of: nearest, bilinear, bicubic, lanczos, lanczos2, lanczos4,
    ///         box, hamming, mitchell, catmull_rom, gaussian, hann.
    pub filter: Option<String>,
    /// Resize mode. Default `"fill"`.
    /// - `"fill"`       — stretch to exact size
    /// - `"fit"`        — fit inside box, pad edges with bgColor
    /// - `"cover"`      — fill box, crop from centre
    /// - `"fit_width"`  — scale to width, height proportional
    /// - `"fit_height"` — scale to height, width proportional
    pub mode: Option<String>,
    /// Pre-multiply alpha before resampling (LA/RGBA). Default `true`.
    pub premultiply_alpha: Option<bool>,
    /// Thread count. `0` = adaptive. Default `1`.
    pub workers: Option<u32>,
    /// Background colour for `"fit"` padding as `[r, g, b, a]`. Default `[0,0,0,0]`.
    pub bg_color: Option<Vec<u8>>,
    /// Copy ICC colour profile to output. Default `true`.
    pub with_icc: Option<bool>,
    /// Copy EXIF block (orientation reset to 1). Default `true`.
    pub with_exif: Option<bool>,
    /// Copy XMP block. Default `false`.
    pub with_xmp: Option<bool>,
    /// Auto-rotate pixels per EXIF orientation before resize. Default `true`.
    pub auto_orient: Option<bool>,
}

#[napi(object)]
pub struct EncodeOpts {
    /// JPEG quality 1–100. Ignored for lossless formats. Default `85`.
    pub quality: Option<u32>,
}

// ─── Image class ──────────────────────────────────────────────────────────────

#[napi]
pub struct Image {
    pub(crate) core: Arc<CoreImage>,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

#[allow(clippy::new_without_default)]
#[napi]
impl Image {
    /// Open an image from a file path (async, non-blocking).
    #[napi(factory)]
    pub async fn open(path: String) -> Result<Image> {
        let img = tokio::task::spawn_blocking(move || CoreImage::open(&path))
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?
            .map_err(map_err)?;
        Ok(Image {
            core: Arc::new(img),
        })
    }

    /// Decode an image from a `Buffer` or `Uint8Array` (sync).
    #[napi(factory)]
    pub fn from_buffer(data: Buffer) -> Result<Image> {
        let img = CoreImage::from_bytes(&data).map_err(map_err)?;
        Ok(Image {
            core: Arc::new(img),
        })
    }

    /// Image width in pixels.
    #[napi(getter)]
    pub fn width(&self) -> u32 {
        self.core.width()
    }

    /// Image height in pixels.
    #[napi(getter)]
    pub fn height(&self) -> u32 {
        self.core.height()
    }

    /// Number of channels (1 = gray, 2 = gray+alpha, 3 = RGB, 4 = RGBA).
    #[napi(getter)]
    pub fn channels(&self) -> u8 {
        self.core.channels()
    }

    /// Resize to `(width, height)` — async, runs on Tokio thread pool.
    ///
    /// Auto-orients, resizes with the chosen filter and mode, and propagates
    /// ICC/EXIF metadata to the result for later injection via `toBuffer`/`save`.
    #[napi]
    pub async fn resize(
        &self,
        width: u32,
        height: u32,
        opts: Option<ResizeOptions>,
    ) -> Result<Image> {
        let opts = opts.unwrap_or(ResizeOptions {
            filter: None,
            mode: None,
            premultiply_alpha: None,
            workers: None,
            bg_color: None,
            with_icc: None,
            with_exif: None,
            with_xmp: None,
            auto_orient: None,
        });
        let filter = opts.filter.clone().unwrap_or_else(|| "lanczos".into());
        let mode = ResizeMode::parse(opts.mode.as_deref().unwrap_or("fill")).map_err(map_err)?;
        let premul = opts.premultiply_alpha.unwrap_or(true);
        let workers = opts.workers.unwrap_or(1) as usize;
        let bg = opts
            .bg_color
            .as_deref()
            .map(|c| BgColor {
                r: c.first().copied().unwrap_or(0),
                g: c.get(1).copied().unwrap_or(0),
                b: c.get(2).copied().unwrap_or(0),
                a: c.get(3).copied().unwrap_or(0),
            })
            .unwrap_or_default();
        let meta_opts = MetadataOptions {
            icc: opts.with_icc.unwrap_or(true),
            exif: opts.with_exif.unwrap_or(true),
            xmp: opts.with_xmp.unwrap_or(false),
            auto_orient: opts.auto_orient.unwrap_or(true),
        };
        let core = Arc::clone(&self.core);
        let resized = tokio::task::spawn_blocking(move || {
            let (oriented, _) =
                core.resize_with_metadata(width, height, &filter, premul, workers, &meta_opts)?;
            oriented.resize_mode(width, height, &filter, mode, premul, workers, bg)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
        .map_err(map_err)?;
        Ok(Image {
            core: Arc::new(resized),
        })
    }

    /// Resize sync — blocks the event loop. Use for scripts or worker threads.
    #[napi(js_name = "resizeSync")]
    pub fn resize_sync(
        &self,
        width: u32,
        height: u32,
        opts: Option<ResizeOptions>,
    ) -> Result<Image> {
        let opts = opts.unwrap_or(ResizeOptions {
            filter: None,
            mode: None,
            premultiply_alpha: None,
            workers: None,
            bg_color: None,
            with_icc: None,
            with_exif: None,
            with_xmp: None,
            auto_orient: None,
        });
        let filter = opts.filter.as_deref().unwrap_or("lanczos");
        let mode = ResizeMode::parse(opts.mode.as_deref().unwrap_or("fill")).map_err(map_err)?;
        let premul = opts.premultiply_alpha.unwrap_or(true);
        let workers = opts.workers.unwrap_or(1) as usize;
        let bg = opts
            .bg_color
            .as_deref()
            .map(|c| BgColor {
                r: c.first().copied().unwrap_or(0),
                g: c.get(1).copied().unwrap_or(0),
                b: c.get(2).copied().unwrap_or(0),
                a: c.get(3).copied().unwrap_or(0),
            })
            .unwrap_or_default();
        let meta_opts = MetadataOptions {
            icc: opts.with_icc.unwrap_or(true),
            exif: opts.with_exif.unwrap_or(true),
            xmp: opts.with_xmp.unwrap_or(false),
            auto_orient: opts.auto_orient.unwrap_or(true),
        };
        let (oriented, _) = self
            .core
            .resize_with_metadata(width, height, filter, premul, workers, &meta_opts)
            .map_err(map_err)?;
        let resized = oriented
            .resize_mode(width, height, filter, mode, premul, workers, bg)
            .map_err(map_err)?;
        Ok(Image {
            core: Arc::new(resized),
        })
    }

    /// Encode to a `Buffer` — async. ICC + EXIF injected automatically.
    #[napi(js_name = "toBuffer")]
    pub async fn to_buffer(
        &self,
        format: Option<String>,
        opts: Option<EncodeOpts>,
    ) -> Result<Buffer> {
        let fmt = format.unwrap_or_else(|| "png".into());
        let quality = opts.and_then(|o| o.quality).unwrap_or(85) as u8;
        let enc = EncodeOptions {
            format: fmt,
            quality,
        };
        let meta = self.core.metadata.clone();
        let core = Arc::clone(&self.core);
        let bytes = tokio::task::spawn_blocking(move || {
            let meta_opts = MetadataOptions {
                icc: true,
                exif: true,
                xmp: false,
                auto_orient: false,
            };
            core.encode_with_metadata(&enc, meta.as_ref(), &meta_opts)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
        .map_err(map_err)?;
        Ok(Buffer::from(bytes))
    }

    /// Encode to a `Buffer` — sync variant.
    #[napi(js_name = "toBufferSync")]
    pub fn to_buffer_sync(
        &self,
        format: Option<String>,
        opts: Option<EncodeOpts>,
    ) -> Result<Buffer> {
        let fmt = format.unwrap_or_else(|| "png".into());
        let quality = opts.and_then(|o| o.quality).unwrap_or(85) as u8;
        let enc = EncodeOptions {
            format: fmt,
            quality,
        };
        let meta_opts = MetadataOptions {
            icc: true,
            exif: true,
            xmp: false,
            auto_orient: false,
        };
        let bytes = self
            .core
            .encode_with_metadata(&enc, self.core.metadata.as_ref(), &meta_opts)
            .map_err(map_err)?;
        Ok(Buffer::from(bytes))
    }

    /// Save to a file — async, format inferred from extension. Injects ICC+EXIF.
    #[napi]
    pub async fn save(&self, path: String, opts: Option<EncodeOpts>) -> Result<()> {
        let quality = opts.and_then(|o| o.quality).unwrap_or(85) as u8;
        let core = Arc::clone(&self.core);
        tokio::task::spawn_blocking(move || core.save(&path, quality))
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?
            .map_err(map_err)
    }

    /// Clone this image.
    #[napi]
    pub fn clone(&self) -> Image {
        Image {
            core: Arc::new(CoreImage {
                inner: self.core.inner.clone(),
                metadata: self.core.metadata.clone(),
            }),
        }
    }

    #[napi]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        format!(
            "Image({}x{}, {}ch)",
            self.core.width(),
            self.core.height(),
            self.core.channels()
        )
    }
}
