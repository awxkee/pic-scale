/*
 * Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::DynamicImage;
use pic_scale::{ImageSize, ResamplingFunction, Rgba8ImageStore, Rgba8ImageStoreMut, Scaler};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    PicScale,
    PicScaleSafe,
    Fir,
    Accelerate,
}

#[derive(Clone, Debug)]
pub struct RunResult {
    pub test_case: String,
    pub backend: Backend,
    pub point_time: f64,
    pub filter: String,
}

pub(crate) const PREHEATING: usize = 10;

pub(crate) fn geometric_mean(v: &[f64]) -> f64 {
    v.iter()
        .fold(1.0, |acc, &x| acc * x.powf(1.0 / v.len() as f64))
}

pub(crate) fn runner_rgba_ps(
    test_case: String,
    image: &DynamicImage,
    target_size: ImageSize,
    iterations: usize,
    filter: String,
    function: ResamplingFunction,
) -> RunResult {
    assert!(matches!(image, DynamicImage::ImageRgba8(_)));

    let source_image = Rgba8ImageStore::from_slice(
        image.as_bytes(),
        image.width() as usize,
        image.height() as usize,
    )
    .unwrap();
    let mut dest_image = Rgba8ImageStoreMut::alloc(target_size.width, target_size.height);

    let scaling = Scaler::new(function);
    let plan = scaling
        .plan_rgba_resampling(
            ImageSize::new(image.width() as usize, image.height() as usize),
            target_size,
            false,
        )
        .unwrap();

    let mut scratch = plan.alloc_scratch();
    // preheat
    for _ in 0..PREHEATING {
        plan.resample_with_scratch(&source_image, &mut dest_image, &mut scratch)
            .unwrap();
    }

    let mut time = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        plan.resample_with_scratch(&source_image, &mut dest_image, &mut scratch)
            .unwrap();
        time.push(start.elapsed().as_nanos() as f64);
    }
    RunResult {
        test_case,
        backend: Backend::PicScale,
        filter,
        point_time: geometric_mean(&time),
    }
}

pub(crate) fn runner_rgba_pss(
    test_case: String,
    image: &DynamicImage,
    target_size: ImageSize,
    iterations: usize,
    filter: String,
    function: pic_scale_safe::ResamplingFunction,
) -> RunResult {
    assert!(matches!(image, DynamicImage::ImageRgba8(_)));

    let src = &image.as_bytes();

    // preheat
    for _ in 0..PREHEATING {
        pic_scale_safe::resize_rgba8(
            &src,
            pic_scale_safe::ImageSize::new(image.width() as usize, image.height() as usize),
            pic_scale_safe::ImageSize::new(target_size.width, target_size.height),
            function,
        )
        .unwrap();
    }

    let mut time = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        pic_scale_safe::resize_rgba8(
            &src,
            pic_scale_safe::ImageSize::new(image.width() as usize, image.height() as usize),
            pic_scale_safe::ImageSize::new(target_size.width, target_size.height),
            function,
        )
        .unwrap();
        time.push(start.elapsed().as_nanos() as f64);
    }
    RunResult {
        test_case,
        backend: Backend::PicScaleSafe,
        filter,
        point_time: geometric_mean(&time),
    }
}

pub(crate) fn runner_rgba_fir(
    test_case: String,
    image: &DynamicImage,
    target_size: ImageSize,
    iterations: usize,
    filter: String,
    filter_type: FilterType,
) -> RunResult {
    assert!(matches!(image, DynamicImage::ImageRgba8(_)));

    let pixel_type: PixelType = PixelType::U8x4;

    let src_image = Image::from_vec_u8(
        image.width(),
        image.height(),
        image.as_bytes().to_vec(),
        pixel_type,
    )
    .unwrap();

    let mut dst_image = Image::new(
        target_size.width as u32,
        target_size.width as u32,
        pixel_type,
    );

    let mut resizer = Resizer::new();
    // preheat
    for _ in 0..PREHEATING {
        resizer
            .resize(
                &src_image,
                &mut dst_image,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(filter_type))
                    .use_alpha(false),
            )
            .unwrap();
    }

    let mut time = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        resizer
            .resize(
                &src_image,
                &mut dst_image,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(filter_type))
                    .use_alpha(false),
            )
            .unwrap();
        time.push(start.elapsed().as_nanos() as f64);
    }
    RunResult {
        test_case,
        backend: Backend::Fir,
        filter,
        point_time: geometric_mean(&time),
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub(crate) fn runner_rgba_accelerate(
    test_case: String,
    image: &DynamicImage,
    target_size: ImageSize,
    iterations: usize,
    filter: String,
) -> RunResult {
    assert!(matches!(image, DynamicImage::ImageRgba8(_)));

    let source_image = Rgba8ImageStore::from_slice(
        image.as_bytes(),
        image.width() as usize,
        image.height() as usize,
    )
    .unwrap();
    let mut dest_image = Rgba8ImageStoreMut::alloc(target_size.width, target_size.height);
    use accelerate::{kvImageDoNotTile, vImage_Buffer, vImageScale_ARGB8888};

    let src_buffer = vImage_Buffer {
        data: source_image.buffer.as_ptr() as *mut libc::c_void,
        width: source_image.width,
        height: source_image.height,
        row_bytes: source_image.stride(),
    };

    let target_stride = dest_image.stride();
    let target_ptr = dest_image.buffer.borrow_mut().as_mut_ptr() as *mut libc::c_void;

    let mut dst_buffer = vImage_Buffer {
        data: target_ptr,
        width: dest_image.width,
        height: dest_image.height,
        row_bytes: target_stride,
    };

    // preheat
    for _ in 0..PREHEATING {
        unsafe {
            _ = vImageScale_ARGB8888(
                &src_buffer,
                &mut dst_buffer,
                std::ptr::null_mut(),
                kvImageDoNotTile,
            );
        }
    }

    let mut time = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        unsafe {
            _ = vImageScale_ARGB8888(
                &src_buffer,
                &mut dst_buffer,
                std::ptr::null_mut(),
                kvImageDoNotTile,
            );
        }
        time.push(start.elapsed().as_nanos() as f64);
    }
    RunResult {
        test_case,
        backend: Backend::Accelerate,
        filter,
        point_time: geometric_mean(&time),
    }
}
