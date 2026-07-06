/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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
use image::DynamicImage;
use maroontree::{
    BitDepth, EncodeConfig, EncodeError, PlanarImage, encode_gray_alpha8, encode_gray_alpha10,
    encode_gray8, encode_gray10, encode_rgb8, encode_rgb10, encode_rgba8, encode_rgba8_with_alpha,
    encode_rgba10, encode_rgba10_with_alpha,
};

#[derive(Clone, Debug)]
pub struct AvifOptions {
    pub config: EncodeConfig,
    pub preserve_alpha: bool,
}

impl AvifOptions {
    #[inline]
    pub fn new(config: EncodeConfig) -> Self {
        Self {
            config,
            preserve_alpha: true,
        }
    }

    #[inline]
    pub fn without_alpha(mut self) -> Self {
        self.preserve_alpha = false;
        self
    }

    #[inline]
    fn into_config(self) -> EncodeConfig {
        self.config
    }
}

/// Convenience methods for encoding `image::DynamicImage` as AV1-in-AVIF.
///
/// This trait is deliberately kept out of the crate root so the image-rs
/// dependency can stay optional.
pub trait DynamicImageAvifExt {
    /// Encode with explicit options.
    fn encode_av1_avif_with_options(&self, options: AvifOptions) -> Result<Vec<u8>, EncodeError>;
}

impl DynamicImageAvifExt for DynamicImage {
    fn encode_av1_avif_with_options(&self, options: AvifOptions) -> Result<Vec<u8>, EncodeError> {
        let preserve_alpha = options.preserve_alpha;
        let cfg = options.into_config();

        match self {
            DynamicImage::ImageLuma8(im) => {
                let img = PlanarImage::from_luma(
                    im.width() as usize,
                    im.height() as usize,
                    BitDepth::Eight,
                    im.as_raw(),
                )?;
                encode_gray8(&img, &cfg)
            }
            DynamicImage::ImageLumaA8(im) => {
                let w = im.width() as usize;
                let h = im.height() as usize;
                if preserve_alpha {
                    let img = PlanarImage::from_interleaved_gray_alpha(
                        w,
                        h,
                        BitDepth::Eight,
                        im.as_raw(),
                    )?;
                    encode_gray_alpha8(&img, &cfg)
                } else {
                    let luma = take_channel_u8(im.as_raw(), 2, 0);
                    let img = PlanarImage::from_luma(w, h, BitDepth::Eight, &luma)?;
                    encode_gray8(&img, &cfg)
                }
            }
            DynamicImage::ImageRgb8(im) => {
                let img = PlanarImage::from_interleaved_rgb(
                    im.width() as usize,
                    im.height() as usize,
                    BitDepth::Eight,
                    im.as_raw(),
                )?;
                encode_rgb8(&img, &cfg)
            }
            DynamicImage::ImageRgba8(im) => {
                let img = PlanarImage::from_interleaved_rgba(
                    im.width() as usize,
                    im.height() as usize,
                    BitDepth::Eight,
                    im.as_raw(),
                )?;
                if preserve_alpha {
                    encode_rgba8_with_alpha(&img, &cfg)
                } else {
                    encode_rgba8(&img, &cfg)
                }
            }
            DynamicImage::ImageLuma16(im) => {
                let luma = u16_to_10_vec(im.as_raw());
                let img = PlanarImage::from_luma(
                    im.width() as usize,
                    im.height() as usize,
                    BitDepth::Ten,
                    &luma,
                )?;
                encode_gray10(&img, &cfg)
            }
            DynamicImage::ImageLumaA16(im) => {
                let w = im.width() as usize;
                let h = im.height() as usize;
                if preserve_alpha {
                    let gray_alpha = u16_to_10_vec(im.as_raw());
                    let img =
                        PlanarImage::from_interleaved_gray_alpha(w, h, BitDepth::Ten, &gray_alpha)?;
                    encode_gray_alpha10(&img, &cfg)
                } else {
                    let luma = take_channel_u16_to_10(im.as_raw(), 2, 0);
                    let img = PlanarImage::from_luma(w, h, BitDepth::Ten, &luma)?;
                    encode_gray10(&img, &cfg)
                }
            }
            DynamicImage::ImageRgb16(im) => {
                let rgb = u16_to_10_vec(im.as_raw());
                let img = PlanarImage::from_interleaved_rgb(
                    im.width() as usize,
                    im.height() as usize,
                    BitDepth::Ten,
                    &rgb,
                )?;
                encode_rgb10(&img, &cfg)
            }
            DynamicImage::ImageRgba16(im) => {
                let rgba = u16_to_10_vec(im.as_raw());
                let img = PlanarImage::from_interleaved_rgba(
                    im.width() as usize,
                    im.height() as usize,
                    BitDepth::Ten,
                    &rgba,
                )?;
                if preserve_alpha {
                    encode_rgba10_with_alpha(&img, &cfg)
                } else {
                    encode_rgba10(&img, &cfg)
                }
            }
            DynamicImage::ImageRgb32F(im) => {
                let rgb = f32_to_10_vec(im.as_raw());
                let img = PlanarImage::from_interleaved_rgb(
                    im.width() as usize,
                    im.height() as usize,
                    BitDepth::Ten,
                    &rgb,
                )?;
                encode_rgb10(&img, &cfg)
            }
            DynamicImage::ImageRgba32F(im) => {
                let rgba = f32_to_10_vec(im.as_raw());
                let img = PlanarImage::from_interleaved_rgba(
                    im.width() as usize,
                    im.height() as usize,
                    BitDepth::Ten,
                    &rgba,
                )?;
                if preserve_alpha {
                    encode_rgba10_with_alpha(&img, &cfg)
                } else {
                    encode_rgba10(&img, &cfg)
                }
            }
            _ => {
                let rgba = self.to_rgba8();
                let img = PlanarImage::from_interleaved_rgba(
                    rgba.width() as usize,
                    rgba.height() as usize,
                    BitDepth::Eight,
                    rgba.as_raw(),
                )?;
                if preserve_alpha {
                    encode_rgba8_with_alpha(&img, &cfg)
                } else {
                    encode_rgba8(&img, &cfg)
                }
            }
        }
    }
}

#[inline]
fn scale_u16_to_10(v: u16) -> u16 {
    // Full-range 16-bit DynamicImage sample -> full-range 10-bit AV1 sample.
    // Uses rounded integer rescaling: round(v * 1023 / 65535).
    (((v as u32 * 1023) + 32767) / 65535) as u16
}

#[inline]
fn scale_f32_to_10(v: f32) -> u16 {
    let v = if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    };
    (v * 1023.0 + 0.5) as u16
}

#[inline]
fn u16_to_10_vec(src: &[u16]) -> Vec<u16> {
    src.iter().map(|&v| scale_u16_to_10(v)).collect()
}

#[inline]
fn f32_to_10_vec(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&v| scale_f32_to_10(v)).collect()
}

#[inline]
fn take_channel_u8(src: &[u8], channels: usize, channel: usize) -> Vec<u8> {
    debug_assert!(channel < channels);
    src.chunks_exact(channels).map(|px| px[channel]).collect()
}

#[inline]
fn take_channel_u16_to_10(src: &[u16], channels: usize, channel: usize) -> Vec<u16> {
    debug_assert!(channel < channels);
    src.chunks_exact(channels)
        .map(|px| scale_u16_to_10(px[channel]))
        .collect()
}
