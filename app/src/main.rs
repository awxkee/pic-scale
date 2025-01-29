#![feature(avx512_target_feature)]
#![feature(f16)]
mod merge;
mod split;

use std::time::Instant;

use core::f16;
use fast_image_resize::images::Image;
use fast_image_resize::{
    CpuExtensions, FilterType, IntoImageView, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::{EncodableLayout, GenericImageView, ImageReader};
use pic_scale::{
    Ar30ByteOrder, ImageSize, ImageStore, ImageStoreMut, ImageStoreScaling, ResamplingFunction,
    RgbF16ImageStore, RgbF16ImageStoreMut, Rgba16ImageStore, Rgba16ImageStoreMut,
    RgbaF16ImageStore, RgbaF16ImageStoreMut, Scaler, Scaling, ScalingU16, ThreadingPolicy,
    WorkloadStrategy,
};

fn resize_plane(
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    sampler: ResamplingFunction,
) {
    if src_width == 0
        || src_width > 2000
        || src_height == 0
        || src_height > 2000
        || dst_width == 0
        || dst_width > 512
        || dst_height == 0
        || dst_height > 512
    {
        return;
    }

    let mut src_data = vec![15u8; src_width * src_height * 1];

    let store = ImageStore::<u8, 1>::from_slice(&mut src_data, src_width, src_height).unwrap();
    let mut dst_store = ImageStoreMut::<u8, 1>::alloc(src_width / 2, src_height / 2);
    let scaler = Scaler::new(sampler);
    _ = scaler.resize_plane(&store, &mut dst_store).unwrap();
}

fn main() {
    // test_fast_image();
    let img = ImageReader::open("./assets/asset_4.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let transient = img.to_rgba8();
    let mut bytes = Vec::from(transient.as_bytes());

    let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    scaler.set_threading_policy(ThreadingPolicy::Single);
    scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);

    // resize_plane(378, 257, 257, 257, ResamplingFunction::Bilinear);

    let mut choke: Vec<u16> = bytes.iter().map(|&x| (x as u16) << 2).collect();

    // let rgb_feature16 = transient
    //     .iter()
    //     .map(|&x| (x as f32 / 255f32) as f16)
    //     .collect::<Vec<_>>();

    //
    let mut store =
        Rgba16ImageStore::from_slice(&choke, dimensions.0 as usize, dimensions.1 as usize).unwrap();
    store.bit_depth = 10;

    // let dst_size = ImageSize::new(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
    // // let mut resized_ar = vec![0u32; dst_size.width * dst_size.height];
    // // scaler
    // //     .resize_ra30(
    // //         &ar30_src,
    // //         ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
    // //         &mut resized_ar,
    // //         dst_size,
    // //         Ar30ByteOrder::Host,
    // //     )
    // //     .unwrap();
    //
    let mut dst_store = Rgba16ImageStoreMut::alloc_with_depth(
        dimensions.0 as usize / 2,
        dimensions.1 as usize / 2,
        10,
    );
    //
    // // for i in 0..25 {
    // let start_time = Instant::now();
    scaler
        .resize_rgba_u16(&store, &mut dst_store, false)
        .unwrap();
    //
    // let elapsed_time = start_time.elapsed();
    // // Print the elapsed time in milliseconds
    // println!("Scaler: {:.2?}", elapsed_time);
    //
    // // #[cfg(target_os = "macos")]
    // // {
    // //     use accelerate::{kvImageDoNotTile, vImageScale_ARGB8888, vImage_Buffer};
    // //     let src_buffer = vImage_Buffer {
    // //         data: store.buffer.as_ptr() as *mut libc::c_void,
    // //         height: store.height,
    // //         width: store.width,
    // //         row_bytes: store.stride(),
    // //     };
    // //
    // //     let mut dst_buffer = vImage_Buffer {
    // //         data: dst_store.buffer.borrow_mut().as_mut_ptr() as *mut libc::c_void,
    // //         height: dst_store.height,
    // //         width: dst_store.width,
    // //         row_bytes: dst_store.stride(),
    // //     };
    // //
    // //     let start_time = Instant::now();
    // //     let result = unsafe {
    // //         vImageScale_ARGB8888(&src_buffer, &mut dst_buffer, std::ptr::null_mut(), kvImageDoNotTile)
    // //     };
    // //     if result != 0 {
    // //         panic!("Can' resize by accelerate");
    // //     }
    // //
    // //     let elapsed_time = start_time.elapsed();
    // //     // Print the elapsed time in milliseconds
    // //     println!("Accelerate: {:.2?}", elapsed_time);
    // // }
    //
    // // let dst: Vec<u8> = resized
    // //     .as_bytes()
    // //     .iter()
    // //     .map(|&x| (x * 255f32) as u8)
    // //     .collect();
    //
    let dst: Vec<u8> = dst_store
        .as_bytes()
        .iter()
        .map(|&x| (x >> 2) as u8)
        .collect();

    // let dst = dst_store
    //     .as_bytes()
    //     .iter()
    //     .map(|&x| (x as f32 * 255.).round() as u8)
    //     .collect::<Vec<_>>();
    // let dst = resized;
    // image::save_buffer(
    //     "converted.png",
    //     &dst,
    //     dst_size.width as u32,
    //     dst_size.height as u32,
    //     image::ColorType::Rgba8,
    // )
    // .unwrap();

    if dst_store.channels == 4 {
        image::save_buffer(
            "converted.png",
            &dst,
            dst_store.width as u32,
            dst_store.height as u32,
            image::ColorType::Rgba8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "converted.png",
            &dst,
            dst_store.width as u32,
            dst_store.height as u32,
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
    let img = ImageReader::open("./assets/nasa-4928x3279-rgba.png")
        .unwrap()
        .decode()
        .unwrap();
    let img = img.to_rgba8();
    let dimensions = img.dimensions();

    let mut vc = Vec::from(img.as_bytes());

    // let mut converted_bytes: Vec<u16> = vc.iter().map(|&x| (x as u16) << 8).collect();
    //
    // let mut chokidar = Vec::from(u16_to_u8(&converted_bytes));

    let start_time = Instant::now();

    let pixel_type: PixelType = PixelType::U8x4;

    let src_image = Image::from_vec_u8(dimensions.0, dimensions.1, vc, pixel_type).unwrap();

    let mut dst_image = Image::new(dimensions.0, dimensions.1 / 2, pixel_type);

    let mut resizer = Resizer::new();
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Neon);
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Avx2);
    }
    resizer
        .resize(
            &src_image,
            &mut dst_image,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                .use_alpha(true),
        )
        .unwrap();

    let elapsed_time = start_time.elapsed();
    // Print the elapsed time in milliseconds
    println!("Fast image resize: {:.2?}", elapsed_time);

    let vegi = dst_image.buffer().to_vec();

    // let converted_16 = Vec::from(u8_to_u16(dst_image.buffer()));
    //
    // let dst: Vec<u8> = converted_16
    //     .iter()
    //     .map(|&x| (x >> 8) as u8)
    //     .collect::<Vec<_>>();

    // if pixel_type == PixelType::U8x3 || pixel_type == PixelType::U16x3 {
    //     image::save_buffer(
    //         "fast_image.jpg",
    //         &vegi,
    //         dst_image.width(),
    //         dst_image.height(),
    //         image::ColorType::Rgb8,
    //     )
    //     .unwrap();
    // } else {
    //     image::save_buffer(
    //         "fast_image.png",
    //         &vegi,
    //         dst_image.width(),
    //         dst_image.height(),
    //         image::ColorType::Rgba8,
    //     )
    //     .unwrap();
    // }
}
