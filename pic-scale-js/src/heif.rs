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
use crate::transpose::apply_orientation;
use hpvca::{ChromaFormat, EncodeConfig};
use hpvcd::Orientation;
use image::{DynamicImage, Luma};
use std::num::NonZeroUsize;

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

fn to_core_orientation(orientation: hpvcd::Orientation) -> crate::metadata::Orientation {
    match orientation {
        Orientation::Normal => crate::metadata::Orientation::Normal,
        Orientation::FlipH => crate::metadata::Orientation::FlipH,
        Orientation::Rotate180 => crate::metadata::Orientation::Rotate180,
        Orientation::FlipV => crate::metadata::Orientation::FlipV,
        Orientation::Transpose => crate::metadata::Orientation::Transpose,
        Orientation::Rotate90 => crate::metadata::Orientation::Rotate90,
        Orientation::Transverse => crate::metadata::Orientation::Transverse,
        Orientation::Rotate270 => crate::metadata::Orientation::Rotate270,
    }
}

pub(crate) fn decode_heic(bytes: &[u8]) -> crate::core::Result<DynamicImage> {
    use hpvcd::{ChromaFormat, MatrixCoefficients};
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

    let dec = hpvcd::decode_heic_yuv(bytes).map_err(|e| PicError::Format(format!("hpvcd: {e}")))?;

    let w = dec.width;
    let h = dec.height;
    let bit_depth = dec.bit_depth.bits() as u32;
    let high_bit = bit_depth > 8;
    let is_12 = bit_depth >= 12;

    // Chroma subsampling factors → tight plane strides.
    let (sub_w, sub_h) = match dec.chroma {
        ChromaFormat::Yuv420 => (2u32, 2u32),
        ChromaFormat::Yuv422 => (2, 1),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => (1, 1),
    };
    let y_stride = w;
    let c_stride = w.div_ceil(sub_w);
    let _ch = h.div_ceil(sub_h);

    let mut matrix = YuvStandardMatrix::Bt601;
    let mut range = YuvRange::Limited;
    let mut is_ycgco = false;
    if let Some(enc) = &dec.color.cicp {
        matrix = match enc.matrix {
            MatrixCoefficients::Smpte170m => YuvStandardMatrix::Bt601,
            MatrixCoefficients::Bt709 => YuvStandardMatrix::Bt709,
            MatrixCoefficients::Bt2020Ncl | MatrixCoefficients::Bt2020Cl => {
                YuvStandardMatrix::Bt2020
            }
            MatrixCoefficients::YCgCo
            | MatrixCoefficients::YCgCoRe
            | MatrixCoefficients::YCgCoRo => {
                is_ycgco = true;
                YuvStandardMatrix::Bt601
            }
            _ => YuvStandardMatrix::Bt601,
        };
        range = if enc.full_range {
            YuvRange::Full
        } else {
            YuvRange::Limited
        };
    }

    if dec.chroma == ChromaFormat::Monochrome {
        return finish_monochrome(&dec, w, h, high_bit, is_12);
    }

    let rgb_stride = w * 3;
    let rgba_stride = w * 4;
    let has_alpha = dec.alpha.is_some();

    let img = if high_bit {
        let y16 = match dec.y.as_u16() {
            None => {
                return Err(PicError::Format(
                    "HEIC RGBA10/RGBA12 Luma is not actually 10/12".to_string(),
                ));
            }
            Some(v) => v,
        };
        let cb16 = match dec.cb.as_u16() {
            None => {
                return Err(PicError::Format(
                    "HEIC RGBA10/RGBA12 Cb is not actually 10/12".to_string(),
                ));
            }
            Some(v) => v,
        };
        let cr16 = match dec.cr.as_u16() {
            None => {
                return Err(PicError::Format(
                    "HEIC RGBA10/RGBA12 Cr is not actually 10/12".to_string(),
                ));
            }
            Some(v) => v,
        };

        if is_ycgco {
            if let Some(a) = dec.alpha.as_ref().and_then(|a| a.as_u16()) {
                let yuva = YuvPlanarImageWithAlpha {
                    y_plane: y16,
                    y_stride,
                    u_plane: cb16,
                    u_stride: c_stride,
                    v_plane: cr16,
                    v_stride: c_stride,
                    a_plane: a,
                    a_stride: y_stride,
                    width: w,
                    height: h,
                };
                let mut out = vec![0u16; (w * 4 * h) as usize];
                match (sub_w, sub_h, is_12) {
                    (2, 2, false) => icgc010_alpha_to_rgba10(&yuva, &mut out, w * 4, range),
                    (2, 1, false) => icgc210_alpha_to_rgba10(&yuva, &mut out, w * 4, range),
                    (_, _, false) => icgc410_alpha_to_rgba10(&yuva, &mut out, w * 4, range),
                    (2, 2, true) => icgc012_alpha_to_rgba12(&yuva, &mut out, w * 4, range),
                    (2, 1, true) => icgc212_alpha_to_rgba12(&yuva, &mut out, w * 4, range),
                    (_, _, true) => icgc412_alpha_to_rgba12(&yuva, &mut out, w * 4, range),
                }
                .map_err(|e| PicError::Format(format!("icgc→rgba16: {e}")))?;
                expand_to_16bit(&mut out, is_12);
                rgba16_image(w, h, out)?
            } else {
                let yuv = YuvPlanarImage {
                    y_plane: y16,
                    y_stride,
                    u_plane: cb16,
                    u_stride: c_stride,
                    v_plane: cr16,
                    v_stride: c_stride,
                    width: w,
                    height: h,
                };
                let mut out = vec![0u16; (w * 3 * h) as usize];
                match (sub_w, sub_h, is_12) {
                    (2, 2, false) => icgc010_to_rgb10(&yuv, &mut out, w * 3, range),
                    (2, 1, false) => icgc210_to_rgb10(&yuv, &mut out, w * 3, range),
                    (_, _, false) => icgc410_to_rgb10(&yuv, &mut out, w * 3, range),
                    (2, 2, true) => icgc012_to_rgb12(&yuv, &mut out, w * 3, range),
                    (2, 1, true) => icgc212_to_rgb12(&yuv, &mut out, w * 3, range),
                    (_, _, true) => icgc412_to_rgb12(&yuv, &mut out, w * 3, range),
                }
                .map_err(|e| PicError::Format(format!("icgc→rgb16: {e}")))?;
                expand_to_16bit(&mut out, is_12);
                rgb16_image(w, h, out)?
            }
        } else if let Some(a) = dec.alpha.as_ref().and_then(|a| a.as_u16()) {
            let yuva = YuvPlanarImageWithAlpha {
                y_plane: y16,
                y_stride,
                u_plane: cb16,
                u_stride: c_stride,
                v_plane: cr16,
                v_stride: c_stride,
                a_plane: a,
                a_stride: y_stride,
                width: w,
                height: h,
            };
            let mut out = vec![0u16; (w * 4 * h) as usize];
            match (sub_w, sub_h, is_12) {
                (2, 2, false) => i010_alpha_to_rgba10(&yuva, &mut out, w * 4, range, matrix),
                (2, 1, false) => i210_alpha_to_rgba10(&yuva, &mut out, w * 4, range, matrix),
                (_, _, false) => i410_alpha_to_rgba10(&yuva, &mut out, w * 4, range, matrix),
                (2, 2, true) => i012_alpha_to_rgba12(&yuva, &mut out, w * 4, range, matrix),
                (2, 1, true) => i212_alpha_to_rgba12(&yuva, &mut out, w * 4, range, matrix),
                (_, _, true) => i412_alpha_to_rgba12(&yuva, &mut out, w * 4, range, matrix),
            }
            .map_err(|e| PicError::Format(format!("yuv→rgba16: {e}")))?;
            expand_to_16bit(&mut out, is_12);
            rgba16_image(w, h, out)?
        } else {
            let yuv = YuvPlanarImage {
                y_plane: y16,
                y_stride,
                u_plane: cb16,
                u_stride: c_stride,
                v_plane: cr16,
                v_stride: c_stride,
                width: w,
                height: h,
            };
            let mut out = vec![0u16; (w * 3 * h) as usize];
            match (sub_w, sub_h, is_12) {
                (2, 2, false) => i010_to_rgb10(&yuv, &mut out, w * 3, range, matrix),
                (2, 1, false) => i210_to_rgb10(&yuv, &mut out, w * 3, range, matrix),
                (_, _, false) => i410_to_rgb10(&yuv, &mut out, w * 3, range, matrix),
                (2, 2, true) => i012_to_rgb12(&yuv, &mut out, w * 3, range, matrix),
                (2, 1, true) => i212_to_rgb12(&yuv, &mut out, w * 3, range, matrix),
                (_, _, true) => i412_to_rgb12(&yuv, &mut out, w * 3, range, matrix),
            }
            .map_err(|e| PicError::Format(format!("yuv→rgb16: {e}")))?;
            expand_to_16bit(&mut out, is_12);
            rgb16_image(w, h, out)?
        }
    } else {
        let y8 = match dec.y.as_u8() {
            None => {
                return Err(PicError::Format(
                    "HEIC RGBA8 Luma is not actually 8".to_string(),
                ));
            }
            Some(v) => v,
        };
        let cb8 = match dec.cb.as_u8() {
            None => {
                return Err(PicError::Format(
                    "HEIC RGBA8 Cb is not actually 8".to_string(),
                ));
            }
            Some(v) => v,
        };
        let cr8 = match dec.cr.as_u8() {
            None => {
                return Err(PicError::Format(
                    "HEIC RGBA8 Cr is not actually 8".to_string(),
                ));
            }
            Some(v) => v,
        };

        if is_ycgco {
            if let Some(a) = dec.alpha.as_ref().and_then(|a| a.as_u8()) {
                let yuv = YuvPlanarImageWithAlpha {
                    y_plane: y8,
                    y_stride,
                    u_plane: cb8,
                    u_stride: c_stride,
                    v_plane: cr8,
                    v_stride: c_stride,
                    a_plane: a,
                    a_stride: y_stride,
                    width: w,
                    height: h,
                };
                let mut rgba = vec![0u8; (rgba_stride * h) as usize];
                match (sub_w, sub_h) {
                    (2, 2) => ycgco420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    (2, 1) => ycgco422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    _ => ycgco444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                }
                .map_err(|e| PicError::Format(format!("ycgco→rgba: {e}")))?;
                rgba8_image(w, h, rgba)?
            } else {
                let yuv = YuvPlanarImage {
                    y_plane: y8,
                    y_stride,
                    u_plane: cb8,
                    u_stride: c_stride,
                    v_plane: cr8,
                    v_stride: c_stride,
                    width: w,
                    height: h,
                };
                let mut rgb = vec![0u8; (rgb_stride * h) as usize];
                match (sub_w, sub_h) {
                    (2, 2) => ycgco420_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                    (2, 1) => ycgco422_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                    _ => ycgco444_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                }
                .map_err(|e| PicError::Format(format!("ycgco→rgb: {e}")))?;
                rgb8_image(w, h, rgb)?
            }
        } else if let Some(a) = dec.alpha.as_ref().and_then(|a| a.as_u8()) {
            let yuv = YuvPlanarImageWithAlpha {
                y_plane: y8,
                y_stride,
                u_plane: cb8,
                u_stride: c_stride,
                v_plane: cr8,
                v_stride: c_stride,
                a_plane: a,
                a_stride: y_stride,
                width: w,
                height: h,
            };
            let mut rgba = vec![0u8; (rgba_stride * h) as usize];
            match (sub_w, sub_h) {
                (2, 2) => yuv420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
                (2, 1) => yuv422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
                _ => yuv444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
            }
            .map_err(|e| PicError::Format(format!("yuv→rgba: {e}")))?;
            rgba8_image(w, h, rgba)?
        } else {
            let yuv = YuvPlanarImage {
                y_plane: y8,
                y_stride,
                u_plane: cb8,
                u_stride: c_stride,
                v_plane: cr8,
                v_stride: c_stride,
                width: w,
                height: h,
            };
            let mut rgb = vec![0u8; (rgb_stride * h) as usize];
            match (sub_w, sub_h) {
                (2, 2) => yuv420_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
                (2, 1) => yuv422_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
                _ => yuv444_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
            }
            .map_err(|e| PicError::Format(format!("yuv→rgb: {e}")))?;
            rgb8_image(w, h, rgb)?
        }
    };

    let _ = has_alpha; // (kept for parity/readability)
    Ok(apply_orientation(img, to_core_orientation(dec.orientation)))
}

fn rgb8_image(w: u32, h: u32, buf: Vec<u8>) -> crate::core::Result<DynamicImage> {
    Ok(DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(w, h, buf)
            .ok_or_else(|| PicError::Format("HEIC RGB mismatch".into()))?,
    ))
}
fn rgba8_image(w: u32, h: u32, buf: Vec<u8>) -> crate::core::Result<DynamicImage> {
    Ok(DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(w, h, buf)
            .ok_or_else(|| PicError::Format("HEIC RGBA mismatch".into()))?,
    ))
}
fn rgb16_image(w: u32, h: u32, buf: Vec<u16>) -> crate::core::Result<DynamicImage> {
    Ok(DynamicImage::ImageRgb16(
        image::ImageBuffer::from_raw(w, h, buf)
            .ok_or_else(|| PicError::Format("HEIC RGB16 mismatch".into()))?,
    ))
}
fn rgba16_image(w: u32, h: u32, buf: Vec<u16>) -> crate::core::Result<DynamicImage> {
    Ok(DynamicImage::ImageRgba16(
        image::ImageBuffer::from_raw(w, h, buf)
            .ok_or_else(|| PicError::Format("HEIC RGBA16 mismatch".into()))?,
    ))
}

/// 4:0:0 monochrome: replicate the luma plane across R, G, B.
fn finish_monochrome(
    dec: &hpvcd::DecodedYuv,
    w: u32,
    h: u32,
    high_bit: bool,
    is_12: bool,
) -> crate::core::Result<DynamicImage> {
    let img = if high_bit {
        let mut y = dec.y.as_u16().expect("high-bit luma is u16").to_vec();
        expand_to_16bit(&mut y, is_12);
        DynamicImage::ImageLuma16(
            image::ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(w, h, y.to_vec())
                .ok_or_else(|| PicError::Format("HEIC RGB mismatch".into()))?,
        )
    } else {
        let y = dec.y.as_u8().expect("8-bit luma is u8");
        if let Some(alpha) = dec.alpha.as_ref() {
            let y = match dec.y.as_u8() {
                None => {
                    return Err(PicError::Format(
                        "HEIC RGBA8 luma 8 is not actually 8".into(),
                    ));
                }
                Some(v) => v,
            };
            let alpha = match alpha.as_u8() {
                None => {
                    return Err(PicError::Format(
                        "HEIC RGBA8 alpha 8 is not actually 8".into(),
                    ));
                }
                Some(v) => v,
            };
            let mut new_img = vec![0u8; dec.width as usize * dec.height as usize * 2];
            for ((dst, src), alpha) in new_img
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(y.iter())
                .zip(alpha.iter())
            {
                dst[0] = *src;
                dst[1] = *alpha;
            }
            DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(w, h, y.to_vec())
                    .ok_or_else(|| PicError::Format("HEIC RGB mismatch".into()))?,
            )
        } else {
            DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(w, h, y.to_vec())
                    .ok_or_else(|| PicError::Format("HEIC RGB mismatch".into()))?,
            )
        }
    };
    Ok(apply_orientation(img, to_core_orientation(dec.orientation)))
}

pub(crate) fn encode_heic(
    img: &DynamicImage,
    quality: u8,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> crate::core::Result<Vec<u8>> {
    let mut base_cfg = EncodeConfig::default()
        .with_quality(quality)
        .with_chroma(ChromaFormat::Yuv420)
        .with_threads(
            std::thread::available_parallelism()
                .unwrap_or(NonZeroUsize::new(1).unwrap())
                .get(),
        );
    if let Some(icc_data) = icc {
        base_cfg = base_cfg.with_icc_profile(icc_data.to_vec());
    }
    if let Some(exif_data) = exif {
        base_cfg = base_cfg.with_exif(exif_data.to_vec());
    }

    let cfg_rgb_8 = base_cfg.clone();
    let cfg_gray_8 = base_cfg.clone().with_chroma(ChromaFormat::Monochrome);
    let cfg_gray_10 = base_cfg.clone().with_chroma(ChromaFormat::Monochrome);

    let w = img.width();
    let h = img.height();

    let map_err = |e: hpvca::EncodeError| PicError::HeicEncoder(e.to_string());

    match img {
        DynamicImage::ImageLuma8(luma) => {
            hpvca::encode_gray(luma, w, h, &cfg_gray_8).map_err(map_err)
        }

        DynamicImage::ImageLumaA8(luma_alpha) => {
            hpvca::encode_gray_alpha_with_alpha(luma_alpha, w, h, &cfg_gray_8).map_err(map_err)
        }

        DynamicImage::ImageRgb8(rgb) => hpvca::encode_rgb(rgb, w, h, &cfg_rgb_8).map_err(map_err),

        DynamicImage::ImageRgba8(rgba) => {
            hpvca::encode_rgba_with_alpha(rgba, w, h, &cfg_rgb_8).map_err(map_err)
        }

        DynamicImage::ImageLuma16(luma) => hpvca::encode_gray10(
            &luma.iter().map(|&x| x >> 6).collect::<Vec<_>>(),
            w,
            h,
            &cfg_gray_10,
        )
        .map_err(map_err),

        DynamicImage::ImageLumaA16(luma_alpha) => hpvca::encode_gray_alpha10_with_alpha(
            &luma_alpha.iter().map(|&x| x >> 6).collect::<Vec<_>>(),
            w,
            h,
            &cfg_gray_10,
        )
        .map_err(map_err),

        DynamicImage::ImageRgb16(rgb) => hpvca::encode_rgb10(
            &rgb.iter().map(|&x| x >> 6).collect::<Vec<_>>(),
            w,
            h,
            &cfg_rgb_8,
        )
        .map_err(map_err),

        DynamicImage::ImageRgba16(rgba) => hpvca::encode_rgba10_with_alpha(
            &rgba.iter().map(|&x| x >> 6).collect::<Vec<_>>(),
            w,
            h,
            &cfg_rgb_8,
        )
        .map_err(map_err),

        DynamicImage::ImageLuma32F(luma) => hpvca::encode_gray10(
            &luma
                .iter()
                .map(|&x| (x * 1023.).round().clamp(0., 1023.) as u16)
                .collect::<Vec<_>>(),
            w,
            h,
            &cfg_gray_10,
        )
        .map_err(map_err),

        DynamicImage::ImageLumaA32F(luma_alpha) => hpvca::encode_gray_alpha10_with_alpha(
            &luma_alpha
                .iter()
                .map(|&x| (x * 1023.).round().clamp(0., 1023.) as u16)
                .collect::<Vec<_>>(),
            w,
            h,
            &cfg_gray_10,
        )
        .map_err(map_err),

        DynamicImage::ImageRgb32F(rgb) => hpvca::encode_rgb10(
            &rgb.iter()
                .map(|&x| (x * 1023.).round().clamp(0., 1023.) as u16)
                .collect::<Vec<_>>(),
            w,
            h,
            &cfg_rgb_8,
        )
        .map_err(map_err),

        DynamicImage::ImageRgba32F(rgba) => hpvca::encode_rgba10_with_alpha(
            &rgba
                .iter()
                .map(|&x| (x * 1023.).round().clamp(0., 1023.) as u16)
                .collect::<Vec<_>>(),
            w,
            h,
            &cfg_rgb_8,
        )
        .map_err(map_err),
        _ => {
            let rgba8 = img.to_rgba8();
            hpvca::encode_rgba(&rgba8, w, h, &cfg_rgb_8).map_err(map_err)
        }
    }
}
