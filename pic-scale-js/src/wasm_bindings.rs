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
//
// All methods are synchronous. WASM is single-threaded — there is no event
// loop to block and no spawn_blocking equivalent. CPU-heavy work (resize,
// encode) runs inline. For non-blocking behaviour in the browser, call from
// a Web Worker:
//
//   // worker.js
//   import init, { Image } from '@radzivon-bartoshyk/pic-scale/wasm'
//   await init()
//   self.onmessage = async ({ data }) => {
//     const img   = Image.fromBytes(data.bytes)
//     const small = img.resize(data.w, data.h)
//     const out   = small.toBytes('jpeg', 85)
//     self.postMessage(out, [out.buffer])   // transfer, zero-copy
//   }
//
// Usage (main thread):
//   import init, { Image } from '@radzivon-bartoshyk/pic-scale/wasm'
//   await init()
//   const bytes = new Uint8Array(await file.arrayBuffer())
//   const img   = Image.fromBytes(bytes)
//   const small = img.resize(800, 600)
//   const out   = small.toBytes('jpeg', 85)   // Uint8Array

use crate::core::{CoreImage, EncodeOptions};
use wasm_bindgen::prelude::*;

fn map_err(e: crate::core::PicError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen(typescript_custom_section)]
const WORKERS_TYPE: &str = r#"
export type Workers = 'adaptive' | 'single';
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Workers")]
    pub type Workers;
}

/// Decoded image held in WASM linear memory.
///
/// All I/O is bytes-in / bytes-out — there is no filesystem on WASM.
/// Memory is freed automatically when the JS object is garbage-collected.
#[wasm_bindgen]
pub struct Image {
    core: CoreImage,
}

#[wasm_bindgen]
impl Image {
    /// Decode an image from a `Uint8Array` (any supported format).
    ///
    /// Supported formats: JPEG, PNG, WebP, TIFF, BMP, ICO, QOI.
    /// ICC profile, EXIF orientation, and XMP are extracted automatically
    /// and stored for later injection via `toBytes`.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(data: &[u8]) -> Result<Image, JsValue> {
        let img = CoreImage::from_bytes(data).map_err(map_err)?;
        Ok(Image { core: img })
    }

    /// Image width in pixels.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.core.width()
    }

    /// Image height in pixels.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.core.height()
    }

    /// Number of channels (1 = gray, 2 = gray+alpha, 3 = RGB, 4 = RGBA).
    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u8 {
        self.core.channels()
    }

    /// Resize to `(width, height)`.
    ///
    /// Parameters
    /// ----------
    /// width, height     — target dimensions in pixels
    /// filter            — resampling filter (default `"lanczos"`)
    ///                     one of: nearest, bilinear, bicubic, lanczos, lanczos2,
    ///                             lanczos4, box, hamming, mitchell, catmull_rom,
    ///                             gaussian, hann
    /// mode              — resize mode (default `"fill"`):
    ///                     `"fill"` stretch · `"fit"` letterbox · `"cover"` crop centre
    ///                     `"fit_width"` · `"fit_height"`
    /// premultiply_alpha — pre-multiply alpha for RGBA/LA (default true)
    /// workers           — thread count; 0 = adaptive (default 1)
    /// bg_color          — `[r, g, b, a]` padding for `"fit"` mode (default transparent)
    /// auto_orient       — rotate pixels per EXIF orientation before resize (default true)
    ///
    /// Returns a new `Image` — the original is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn resize(
        &self,
        width: u32,
        height: u32,
        filter: Option<String>,
        mode: Option<String>,
        premultiply_alpha: Option<bool>,
        workers: Option<Workers>,
        bg_color: Option<Vec<u8>>,
        auto_orient: Option<bool>,
    ) -> Result<Image, JsValue> {
        let filter = filter.as_deref().unwrap_or("lanczos");
        let mode_str = mode.as_deref().unwrap_or("fill");
        let mode = crate::core::ResizeMode::parse(mode_str).map_err(map_err)?;
        let premultiply = premultiply_alpha.unwrap_or(true);
        let bg = bg_color
            .as_deref()
            .map(|c| crate::core::BgColor {
                r: c.first().copied().unwrap_or(0),
                g: c.get(1).copied().unwrap_or(0),
                b: c.get(2).copied().unwrap_or(0),
                a: c.get(3).copied().unwrap_or(0),
            })
            .unwrap_or_default();
        let meta_opts = crate::metadata::MetadataOptions {
            icc: true,
            exif: true,
            xmp: false,
            auto_orient: auto_orient.unwrap_or(true),
        };

        // 1. Apply EXIF orientation + resize to (width, height) via the
        //    mode-unaware path — resize_with_metadata always does Fill.
        // 2. Then apply the crop/pad mode on the oriented+scaled result
        //    so the final output matches exactly (width, height).

        let workers: usize = workers
            .map(|w| {
                let v: JsValue = w.into();
                if let Some(n) = v.as_f64() {
                    return n as usize;
                }
                match v.as_string().as_deref() {
                    Some("adaptive") => 0,
                    Some("single") => 1,
                    _ => 1,
                }
            })
            .unwrap_or(1);

        let (oriented_and_scaled, _) = self
            .core
            .resize_with_metadata(width, height, filter, premultiply, workers, &meta_opts)
            .map_err(map_err)?;

        let mut resized = if mode != crate::core::ResizeMode::Fill {
            // resize_mode on the already-sized image just handles crop/pad;
            // the pixel scaling was already done above.
            let meta_clone = oriented_and_scaled.metadata.clone();
            let mut r = oriented_and_scaled
                .resize_mode(width, height, filter, mode, premultiply, workers, bg)
                .map_err(map_err)?;
            r.metadata = meta_clone;
            r
        } else {
            oriented_and_scaled
        };

        // Reset orientation tag — baked into pixels now
        if meta_opts.auto_orient
            && let Some(ref mut m) = resized.metadata
        {
            m.orientation = crate::metadata::Orientation::Normal;
        }
        Ok(Image { core: resized })
    }

    /// Encode the image and return the bytes as a `Uint8Array`.
    ///
    /// `format`    — `"jpeg"`, `"png"`, `"webp"`, `"tiff"`, `"bmp"`, `"qoi"`, `"heic"`, `"heif"`, `"avif"`. Default `"png"`.
    /// `quality`   — JPEG quality 1–100 (default 85). Ignored for lossless formats.
    /// `with_icc`  — embed ICC colour profile (default true).
    /// `with_exif` — embed EXIF block, orientation reset to 1 (default true).
    /// `with_xmp`  — embed XMP block (default false).
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(
        &self,
        format: Option<String>,
        quality: Option<u8>,
        with_icc: Option<bool>,
        with_exif: Option<bool>,
        with_xmp: Option<bool>,
    ) -> std::result::Result<Vec<u8>, JsValue> {
        let opts = EncodeOptions {
            format: format.unwrap_or_else(|| "png".into()),
            quality: quality.unwrap_or(85),
        };
        let meta_opts = crate::metadata::MetadataOptions {
            icc: with_icc.unwrap_or(true),
            exif: with_exif.unwrap_or(true),
            xmp: with_xmp.unwrap_or(false),
            auto_orient: false, // already applied at resize time
        };
        self.core
            .encode_with_metadata(&opts, self.core.metadata.as_ref(), &meta_opts)
            .map_err(map_err)
    }

    /// String representation — useful for debugging.
    #[wasm_bindgen(js_name = toString)]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        format!(
            "Image({}×{}, {} ch)",
            self.core.width(),
            self.core.height(),
            self.core.channels()
        )
    }
}
