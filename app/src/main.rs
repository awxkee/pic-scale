mod merge;
mod split;

use std::time::Instant;

use fast_image_resize::images::Image;
use fast_image_resize::FilterType::Lanczos3;
use fast_image_resize::{
    CpuExtensions, IntoImageView, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::{EncodableLayout, GenericImageView, ImageReader};
use pic_scale::{ImageSize, ImageStore, ResamplingFunction, Scaler, Scaling, ScalingF32, ScalingU16, ThreadingPolicy};

fn main() {
    // test_fast_image();
    let img = ImageReader::open("./assets/test_1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let transient = img.to_rgba8();
    let mut bytes = Vec::from(transient.as_bytes());

    let mut scaler = Scaler::new(ResamplingFunction::MitchellNetravalli);
    scaler.set_threading_policy(ThreadingPolicy::Single);

    // let mut choke: Vec<f32> = bytes
    //     .iter()
    //     .map(|&x| x as f32 * (1. / 255.))
    //     .collect();

    let start_time = Instant::now();
    let store =
        ImageStore::<u8, 4>::from_slice(&mut bytes, dimensions.0 as usize, dimensions.1 as usize)
            .unwrap();
    let resized = scaler.resize_rgba(
        ImageSize::new(74, 74),
        store,
        true,
    );

    let dst: Vec<u8> = Vec::from(resized.as_bytes());
    // println!("f1 {}, f2 {}, f3 {}, f4 {}", dst[0], dst[1], dst[2], dst[3]);
    // let dst: Vec<u8> = resized
    //     .as_bytes()
    //     .iter()
    //     .map(|&x| (x * 255f32) as u8)
    //     .collect();

    // let mut r_chan = vec![0u8; dimensions.0 as usize * dimensions.1 as usize];
    // let mut g_chan = vec![0u8; dimensions.0 as usize * dimensions.1 as usize];
    // let mut b_chan = vec![0u8; dimensions.0 as usize * dimensions.1 as usize];
    // split_channels_3(
    //     &bytes,
    //     dimensions.0 as usize,
    //     dimensions.1 as usize,
    //     &mut r_chan,
    //     &mut g_chan,
    //     &mut b_chan,
    // );
    //
    // let store =
    //     ImageStore::<u8, 1>::from_slice(&mut r_chan, dimensions.0 as usize, dimensions.1 as usize);
    // let resized = scaler.resize_plane(
    //     ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
    //     store,
    // );
    //
    // let store1 =
    //     ImageStore::<u8, 1>::from_slice(&mut g_chan, dimensions.0 as usize, dimensions.1 as usize);
    // let resized1 = scaler.resize_plane(
    //     ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
    //     store1,
    // );
    //
    // let store2 =
    //     ImageStore::<u8, 1>::from_slice(&mut b_chan, dimensions.0 as usize, dimensions.1 as usize);
    // let resized2 = scaler.resize_plane(
    //     ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
    //     store2,
    // );
    //
    // let mut dst = vec![0u8; resized.width * resized.height * 3];
    // merge_channels_3(
    //     &mut dst,
    //     resized.width,
    //     resized.height,
    //     &resized.as_bytes(),
    //     &resized1.as_bytes(),
    //     &resized2.as_bytes(),
    // );

    //

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Scaler: {:.2?}", elapsed_time);

    // let dst: Vec<u8> = resized
    //     .as_bytes()
    //     .iter()
    //     .map(|&x| (x * 255f32) as u8)
    //     .collect();

    // let dst: Vec<u8> = resized.as_bytes().iter().map(|&x| (x >> 2) as u8).collect();
    //
    // let dst = resized.as_bytes();

    if resized.channels == 4 {
        image::save_buffer(
            "converted.png",
            &dst,
            resized.width as u32,
            resized.height as u32,
            image::ColorType::Rgba8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "converted.jpg",
            &dst,
            resized.width as u32,
            resized.height as u32,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }

    // for i in 0..37 {
    //     let mut scaler = Scaler::new(i.into());
    //     scaler.set_threading_policy(ThreadingPolicy::Adaptive);
    //     let store =
    //         ImageStore::<u8, 4>::from_slice(&mut bytes, dimensions.0 as usize, dimensions.1 as usize);
    //     let resized = scaler.resize_rgba(
    //         ImageSize::new(dimensions.0 as usize / 3, dimensions.1 as usize / 3),
    //         store,
    //         true,
    //     );
    //
    //     let elapsed_time = start_time.elapsed();
    //     // Print the elapsed time in milliseconds
    //     println!("Scaler: {:.2?}", elapsed_time);
    //
    //     if resized.channels == 4 {
    //         image::save_buffer(
    //             format!("converted_{}.png", i),
    //             resized.as_bytes(),
    //             resized.width as u32,
    //             resized.height as u32,
    //             image::ExtendedColorType::Rgba8,
    //         )
    //             .unwrap();
    //     } else {
    //         image::save_buffer(
    //             format!("converted_{}.jpg", i),
    //             resized.as_bytes(),
    //             resized.width as u32,
    //             resized.height as u32,
    //             image::ExtendedColorType::Rgb8,
    //         )
    //             .unwrap();
    //     }
    // }
}

fn u16_to_u8(u16_buffer: &[u16]) -> &[u8] {
    let len = u16_buffer.len() * 2;
    unsafe { std::slice::from_raw_parts(u16_buffer.as_ptr() as *const u8, len) }
}

fn u8_to_u16(u8_buffer: &[u8]) -> &[u16] {
    let len = u8_buffer.len() / 2;
    unsafe { std::slice::from_raw_parts(u8_buffer.as_ptr() as *const u16, len) }
}

fn test_fast_image() {
    let img = ImageReader::open("./assets/asset_5.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();

    let mut vc = Vec::from(img.as_bytes());

    // let mut converted_bytes: Vec<u16> = vc.iter().map(|&x| (x as u16) << 2).collect();

    // let mut chokidar = Vec::from(u16_to_u8(&converted_bytes));

    let start_time = Instant::now();

    let pixel_type: PixelType = PixelType::U8x4;

    let src_image = Image::from_vec_u8(dimensions.0, dimensions.1, vc, pixel_type).unwrap();

    let mut dst_image = Image::new(dimensions.0 / 2, dimensions.1 / 2, pixel_type);

    let mut resizer = Resizer::new();
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Neon);
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Sse4_1);
    }
    resizer
        .resize(
            &src_image,
            &mut dst_image,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(Lanczos3))
                .use_alpha(false),
        )
        .unwrap();

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Fast image resize: {:.2?}", elapsed_time);

    let converted_16 = dst_image.buffer(); // Vec::from(u8_to_u16(dst_image.buffer()));

    let dst: Vec<u8> = converted_16.iter().map(|&x| (x >> 2) as u8).collect();

    if pixel_type == PixelType::U8x3 || pixel_type == PixelType::U16x3 {
        image::save_buffer(
            "fast_image.jpg",
            &dst,
            dst_image.width() as u32,
            dst_image.height() as u32,
            image::ColorType::Rgb8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "fast_image.png",
            &dst,
            dst_image.width() as u32,
            dst_image.height() as u32,
            image::ColorType::Rgba8,
        )
        .unwrap();
    }
}
