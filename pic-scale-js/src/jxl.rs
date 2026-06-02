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
use image::DynamicImage;
use jixel::{
    ColorSpace, EncodeConfig, FlMeta, encode_fast_lossless, encode_fast_lossless_u16, encode_image,
    encode_image_12bit, encode_image_f32, encode_image_gray, encode_image_gray_12bit,
    encode_image_gray_alpha, encode_image_gray_alpha_12bit, encode_image_with_alpha,
    encode_image_with_alpha_12bit, encode_image_with_alpha_f32,
};
use jxl::api::{JxlColorType, JxlDataFormat};
use jxl::headers::extra_channels::ExtraChannel;

pub(crate) fn is_jxl(bytes: &[u8]) -> bool {
    if bytes.len() < 2 {
        return false;
    }
    if bytes[0] == 0xFF && bytes[1] == 0x0A {
        return true;
    }
    bytes.len() >= 12 && &bytes[4..8] == b"JXL " && &bytes[8..12] == b"\x0D\x0A\x87\x0A"
}

pub(crate) fn decode_jxl(bytes: &[u8]) -> Result<DynamicImage, PicError> {
    use jxl::api::{
        Endianness, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult,
    };

    let mut input = bytes;

    let mut decoder_with_image_info = match JxlDecoder::new(JxlDecoderOptions::default())
        .process(&mut input)
        .map_err(|x| PicError::Format(format!("jxl {x}")))?
    {
        ProcessingResult::Complete { result: d } => d,
        ProcessingResult::NeedsMoreInput { .. } => {
            return Err(PicError::Format("jxl: truncated before basic_info".into()));
        }
    };

    let info = decoder_with_image_info.basic_info();
    let (w, h) = info.size;
    let bits = info.bit_depth.bits_per_sample();
    let has_alpha = info
        .extra_channels
        .iter()
        .any(|ec| ec.ec_type == ExtraChannel::Alpha);
    let is_gray = matches!(
        decoder_with_image_info.current_pixel_format().color_type,
        JxlColorType::Grayscale | JxlColorType::GrayscaleAlpha
    );

    let color_type = match (is_gray, has_alpha) {
        (false, false) => JxlColorType::Rgb,
        (false, true) => JxlColorType::Rgba,
        (true, false) => JxlColorType::Grayscale,
        (true, true) => JxlColorType::GrayscaleAlpha,
    };

    let color_data_format = Some(match bits {
        0..=8 => JxlDataFormat::U8 {
            bit_depth: bits as u8,
        },
        9..=16 => JxlDataFormat::U16 {
            bit_depth: 16,
            endianness: Endianness::LittleEndian,
        },
        _ => JxlDataFormat::F32 {
            endianness: Endianness::LittleEndian,
        },
    });

    decoder_with_image_info.set_pixel_format(JxlPixelFormat {
        color_type,
        color_data_format,
        extra_channel_format: vec![None; info.extra_channels.len()],
    });

    let decoder_with_frame_info = match decoder_with_image_info
        .process(&mut input)
        .map_err(|x| PicError::Format(format!("jxl {x}")))?
    {
        ProcessingResult::Complete { result: d } => d,
        ProcessingResult::NeedsMoreInput { .. } => {
            return Err(PicError::Format("jxl: truncated before frame info".into()));
        }
    };

    macro_rules! decode_pixels {
        (u8, $channels:expr, $label:expr, $img:ident) => {{
            let stride = w * $channels;
            let mut buf = vec![0u8; h * stride];
            let mut out = [JxlOutputBuffer::new(buf.as_mut_slice(), h, stride)];
            decoder_with_frame_info
                .process(&mut input, &mut out)
                .map_err(|x| PicError::Format(format!("jxl {x}")))?;
            DynamicImage::$img(
                image::ImageBuffer::from_raw(w as u32, h as u32, buf)
                    .ok_or_else(|| PicError::Format(format!("jxl {} buffer mismatch", $label)))?,
            )
        }};
        ($T:ty, $channels:expr, $label:expr, $img:ident) => {{
            let stride_bytes = w * $channels * size_of::<$T>();
            let mut buf = vec![0 as $T; w * h * $channels];
            let mut out = [JxlOutputBuffer::new(
                bytemuck::cast_slice_mut(&mut buf),
                h,
                stride_bytes,
            )];
            decoder_with_frame_info
                .process(&mut input, &mut out)
                .map_err(|x| PicError::Format(format!("jxl {x}")))?;
            DynamicImage::$img(
                image::ImageBuffer::from_raw(w as u32, h as u32, buf)
                    .ok_or_else(|| PicError::Format(format!("jxl {} buffer mismatch", $label)))?,
            )
        }};
    }

    let image = match (is_gray, has_alpha, bits) {
        (false, false, 0..=8) => decode_pixels!(u8, 3, "RGB8", ImageRgb8),
        (false, true, 0..=8) => decode_pixels!(u8, 4, "RGBA8", ImageRgba8),
        (true, false, 0..=8) => decode_pixels!(u8, 1, "Luma8", ImageLuma8),
        (true, true, 0..=8) => decode_pixels!(u8, 2, "LumaA8", ImageLumaA8),
        (false, false, 9..=16) => decode_pixels!(u16, 3, "RGB16", ImageRgb16),
        (false, true, 9..=16) => decode_pixels!(u16, 4, "RGBA16", ImageRgba16),
        (true, false, 9..=16) => decode_pixels!(u16, 1, "Luma16", ImageLuma16),
        (true, true, 9..=16) => decode_pixels!(u16, 2, "LumaA16", ImageLumaA16),
        (false, false, _) => decode_pixels!(f32, 3, "Rgb32F", ImageRgb32F),
        (false, true, _) => decode_pixels!(f32, 4, "Rgba32F", ImageRgba32F),
        // DynamicImage has no Luma32F/LumaA32F — expand gray to RGB(A)
        (true, false, _) => {
            let stride_bytes = w * size_of::<f32>();
            let mut buf = vec![0f32; w * h];
            let mut out = [JxlOutputBuffer::new(
                bytemuck::cast_slice_mut(&mut buf),
                h,
                stride_bytes,
            )];
            decoder_with_frame_info
                .process(&mut input, &mut out)
                .map_err(|x| PicError::Format(format!("jxl {x}")))?;
            let rgb: Vec<f32> = buf.iter().flat_map(|&v| [v, v, v]).collect();
            DynamicImage::ImageRgb32F(
                image::ImageBuffer::from_raw(w as u32, h as u32, rgb).ok_or_else(|| {
                    PicError::Format("jxl Gray to RGB buffer mismatch".to_string())
                })?,
            )
        }
        (true, true, _) => {
            let stride_bytes = w * 2 * size_of::<f32>();
            let mut buf = vec![0f32; w * h * 2];
            let mut out = [JxlOutputBuffer::new(
                bytemuck::cast_slice_mut(&mut buf),
                h,
                stride_bytes,
            )];
            decoder_with_frame_info
                .process(&mut input, &mut out)
                .map_err(|x| PicError::Format(format!("jxl {x}")))?;
            let rgba: Vec<f32> = buf
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&[g, a]| [g, g, g, a])
                .collect();
            DynamicImage::ImageRgba32F(
                image::ImageBuffer::from_raw(w as u32, h as u32, rgba).ok_or_else(|| {
                    PicError::Format("jxl GrayA to RGBA buffer mismatch".to_string())
                })?,
            )
        }
    };

    Ok(image)
}

pub(crate) fn encode_jxl(
    img: &DynamicImage,
    quality: u8,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> crate::core::Result<Vec<u8>> {
    let width = img.width() as usize;
    let height = img.height() as usize;
    let lossless = quality == 100;
    let has_alpha = img.color().has_alpha();

    if lossless
        && !matches!(img, DynamicImage::ImageRgb32F(_))
        && !matches!(img, DynamicImage::ImageRgba32F(_))
    {
        let mut meta = FlMeta::default();
        if let Some(exif) = exif {
            meta.exif = Some(exif.to_vec());
        }
        let result = match img {
            DynamicImage::ImageRgba16(img16) => encode_fast_lossless_u16(
                &img16.iter().map(|x| x >> 4).collect::<Vec<_>>(),
                width,
                height,
                ColorSpace::Rgb,
                true,
                12,
                &meta,
            ),
            DynamicImage::ImageRgb16(img16) => encode_fast_lossless_u16(
                &img16.iter().map(|x| x >> 4).collect::<Vec<_>>(),
                width,
                height,
                ColorSpace::Rgb,
                false,
                12,
                &meta,
            ),
            DynamicImage::ImageLuma8(luma8) => {
                encode_fast_lossless(luma8, width, height, ColorSpace::Gray, false, &meta)
            }
            DynamicImage::ImageLumaA8(luma8) => {
                encode_fast_lossless(luma8, width, height, ColorSpace::Gray, true, &meta)
            }
            DynamicImage::ImageLuma16(luma16) => encode_fast_lossless_u16(
                &luma16.iter().map(|x| x >> 4).collect::<Vec<_>>(),
                width,
                height,
                ColorSpace::Gray,
                false,
                12,
                &meta,
            ),
            DynamicImage::ImageLumaA16(luma16) => encode_fast_lossless_u16(
                &luma16.iter().map(|x| x >> 4).collect::<Vec<_>>(),
                width,
                height,
                ColorSpace::Gray,
                true,
                12,
                &meta,
            ),
            _ if has_alpha => {
                let rgba = img.to_rgba8();
                encode_fast_lossless(rgba.as_raw(), width, height, ColorSpace::Rgb, true, &meta)
            }
            _ => {
                let rgb = img.to_rgb8();
                encode_fast_lossless(rgb.as_raw(), width, height, ColorSpace::Rgb, false, &meta)
            }
        };
        return result.map_err(|x| PicError::Format(x.to_string()));
    }

    let mut config = EncodeConfig::default()
        .with_quality(quality as f32)
        .with_lossless(lossless);

    if let Some(icc_data) = icc {
        config = config.with_icc_profile(icc_data.to_vec());
    }
    if let Some(exif_data) = exif {
        config = config.with_exif(exif_data.to_vec());
    }

    let result = match img {
        DynamicImage::ImageRgba16(img16) => encode_image_with_alpha_12bit(
            &img16.iter().map(|x| x >> 4).collect::<Vec<_>>(),
            width,
            height,
            &config,
        ),
        DynamicImage::ImageRgb16(img16) => encode_image_12bit(
            &img16.iter().map(|x| x >> 4).collect::<Vec<_>>(),
            width,
            height,
            &config,
        ),
        DynamicImage::ImageLuma8(luma8) => encode_image_gray(luma8, width, height, &config),
        DynamicImage::ImageLumaA8(luma8) => encode_image_gray_alpha(luma8, width, height, &config),
        DynamicImage::ImageLuma16(luma16) => encode_image_gray_12bit(
            &luma16.iter().map(|x| x >> 4).collect::<Vec<_>>(),
            width,
            height,
            &config,
        ),
        DynamicImage::ImageLumaA16(luma16) => encode_image_gray_alpha_12bit(
            &luma16.iter().map(|x| x >> 4).collect::<Vec<_>>(),
            width,
            height,
            &config,
        ),
        DynamicImage::ImageRgb32F(img32) => encode_image_f32(img32, width, height, &config),
        DynamicImage::ImageRgba32F(img32) => {
            encode_image_with_alpha_f32(img32, width, height, &config)
        }
        DynamicImage::ImageRgb8(rgb) => encode_image(rgb, width, height, &config),
        DynamicImage::ImageRgba8(rgba) => encode_image_with_alpha(rgba, width, height, &config),
        _ if has_alpha => {
            let rgba = img.to_rgba8();
            encode_image_with_alpha(rgba.as_raw(), width, height, &config)
        }
        _ => {
            let rgb = img.to_rgb8();
            encode_image(rgb.as_raw(), width, height, &config)
        }
    };

    result.map_err(|x| PicError::Format(x.to_string()))
}
