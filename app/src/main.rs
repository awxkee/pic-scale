use std::time::Instant;

use fast_image_resize::{CpuExtensions, IntoImageView, PixelType, ResizeAlg, ResizeOptions, Resizer};
use fast_image_resize::FilterType::Lanczos3;
use fast_image_resize::images::Image;
use image::{EncodableLayout, GenericImageView};
use image::io::Reader as ImageReader;

use image_scale::{ImageSize, ImageStore, ResamplingFunction, Scaler, ThreadingPolicy};

fn main() {
    // test_fast_image();

    let img = ImageReader::open("./assets/asset_middle.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    println!("dimensions {:?}", img.dimensions());

    println!("{:?}", img.color());

    let mut bytes = Vec::from(img.as_bytes());

    let start_time = Instant::now();

    let mut scaler = Scaler::new(ResamplingFunction::Gaussian);
    scaler.set_threading_policy(ThreadingPolicy::Single);
    let store = ImageStore::<u8, 3>::from_slice(&mut bytes, dimensions.0 as usize, dimensions.1 as usize);
    let resized = scaler.resize_rgb(ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2), store);

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Scaler: {:.2?}", elapsed_time);

    if resized.channels == 4 {
        image::save_buffer(
            "converted.png",
            resized.as_bytes(),
            resized.width as u32,
            resized.height as u32,
            image::ExtendedColorType::Rgba8,
        )
            .unwrap();
    } else {
        image::save_buffer(
            "converted.jpg",
            resized.as_bytes(),
            resized.width as u32,
            resized.height as u32,
            image::ExtendedColorType::Rgb8,
        )
            .unwrap();
    }
}

fn test_fast_image() {
    let img = ImageReader::open("./assets/asset_5.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();

    let mut vc = Vec::from(img.as_bytes());

    let start_time = Instant::now();

    let pixel_type: PixelType = PixelType::U8x4;

    let src_image = Image::from_slice_u8(
        dimensions.0,
        dimensions.1,
        &mut vc,
        pixel_type,
    ).unwrap();

    let mut dst_image = Image::new(
        dimensions.0 / 2,
        dimensions.1 / 2,
        pixel_type,
    );

    let mut resizer = Resizer::new();
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Neon);
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Sse4_1);
    }
    resizer.resize(
        &src_image,
        &mut dst_image,
        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(Lanczos3)),
    ).unwrap();

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Fast image resize: {:.2?}", elapsed_time);

    if pixel_type == PixelType::U8x3 {
        image::save_buffer(
            "fast_image.jpg",
            dst_image.buffer(),
            dst_image.width() as u32,
            dst_image.height() as u32,
            image::ExtendedColorType::Rgb8,
        )
            .unwrap();
    } else {
        image::save_buffer(
            "fast_image.png",
            dst_image.buffer(),
            dst_image.width() as u32,
            dst_image.height() as u32,
            image::ExtendedColorType::Rgba8,
        )
            .unwrap();
    }
}