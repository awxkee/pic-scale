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
use crate::runner::{Backend, PREHEATING, RunResult, geometric_mean};
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, EncodableLayout};
use pic_scale::{ImageSize, ResamplingFunction, RgbaF32ImageStore, RgbaF32ImageStoreMut, Scaler};
use std::time::Instant;

pub(crate) fn runner_rgba_ps_f32(
    test_case: String,
    image: &DynamicImage,
    target_size: ImageSize,
    iterations: usize,
    filter: String,
    function: ResamplingFunction,
) -> RunResult {
    assert!(matches!(image, DynamicImage::ImageRgba32F(_)));

    let v = image.to_rgba32f().to_vec();
    let source_image =
        RgbaF32ImageStore::from_slice(&v, image.width() as usize, image.height() as usize).unwrap();
    let mut dest_image = RgbaF32ImageStoreMut::alloc(target_size.width, target_size.height);

    let scaling = Scaler::new(function);
    let plan = scaling
        .plan_rgba_resampling_f32(
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

pub(crate) fn runner_rgba_pss_f32(
    test_case: String,
    image: &DynamicImage,
    target_size: ImageSize,
    iterations: usize,
    filter: String,
    function: pic_scale_safe::ResamplingFunction,
) -> RunResult {
    assert!(matches!(image, DynamicImage::ImageRgba32F(_)));

    let v = image.to_rgba32f().to_vec();

    // preheat
    for _ in 0..PREHEATING {
        pic_scale_safe::resize_rgba_f32(
            &v,
            pic_scale_safe::ImageSize::new(image.width() as usize, image.height() as usize),
            pic_scale_safe::ImageSize::new(target_size.width, target_size.height),
            function,
        )
        .unwrap();
    }

    let mut time = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        pic_scale_safe::resize_rgba_f32(
            &v,
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

pub(crate) fn runner_rgba_fir_f32(
    test_case: String,
    image: &DynamicImage,
    target_size: ImageSize,
    iterations: usize,
    filter: String,
    filter_type: FilterType,
) -> RunResult {
    assert!(matches!(image, DynamicImage::ImageRgba32F(_)));

    let f32_img = image.to_rgba32f();
    let v = f32_img.as_bytes();

    let pixel_type: PixelType = PixelType::F32x4;

    let src_image =
        Image::from_vec_u8(image.width(), image.height(), v.to_vec(), pixel_type).unwrap();

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
pub(crate) fn runner_rgba_accelerate_f32(
    test_case: String,
    image: &DynamicImage,
    target_size: ImageSize,
    iterations: usize,
    filter: String,
) -> RunResult {
    assert!(matches!(image, DynamicImage::ImageRgba32F(_)));

    let v = image.to_rgba32f().to_vec();
    let source_image =
        RgbaF32ImageStore::from_slice(&v, image.width() as usize, image.height() as usize).unwrap();
    let mut dest_image = RgbaF32ImageStoreMut::alloc(target_size.width, target_size.height);

    use accelerate::{kvImageDoNotTile, vImage_Buffer, vImageScale_ARGBFFFF};

    let src_buffer = vImage_Buffer {
        data: source_image.buffer.as_ptr() as *mut libc::c_void,
        width: source_image.width,
        height: source_image.height,
        row_bytes: source_image.stride() * size_of::<f32>(),
    };

    let target_stride = dest_image.stride();
    let target_ptr = dest_image.buffer.borrow_mut().as_mut_ptr() as *mut libc::c_void;

    let mut dst_buffer = vImage_Buffer {
        data: target_ptr,
        width: dest_image.width,
        height: dest_image.height,
        row_bytes: target_stride * size_of::<f32>(),
    };

    // preheat
    for _ in 0..PREHEATING {
        unsafe {
            _ = vImageScale_ARGBFFFF(
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
            _ = vImageScale_ARGBFFFF(
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
