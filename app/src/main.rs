#![feature(f16)]
// mod acc;
mod fuzzer_explore;
mod merge;
mod split;

use std::time::Instant;

use crate::fuzzer_explore::resize_rgba;
use core::f16;
use fast_image_resize::images::Image;
use fast_image_resize::{
    CpuExtensions, FilterType, IntoImageView, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::{EncodableLayout, GenericImageView, ImageReader};
use pic_scale::{
    BufferStore, CbCr16ImageStore, CbCr16ImageStoreMut, ImageSize, ImageStore, ImageStoreMut,
    ImageStoreScaling, JzazbzScaler, LChScaler, LabScaler, LuvScaler, Planar16ImageStore,
    Planar16ImageStoreMut, ResamplingFunction, Rgb16ImageStore, Rgb16ImageStoreMut, Rgb8ImageStore,
    Rgb8ImageStoreMut, RgbF32ImageStore, RgbF32ImageStoreMut, Rgba16ImageStore,
    Rgba16ImageStoreMut, Rgba8ImageStore, Rgba8ImageStoreMut, RgbaF32ImageStore,
    RgbaF32ImageStoreMut, Scaler, SigmoidalScaler, ThreadingPolicy, TransferFunction,
    WorkloadStrategy, XYZScaler,
};
use rand::RngExt;

fn resize_rgba_fuzz(
    data: u8,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    sampler: ResamplingFunction,
    workload_strategy: WorkloadStrategy,
    threading_policy: ThreadingPolicy,
    supersampling: bool,
    multi_stage_upsampling: bool,
) {
    if src_width == 0
        || src_width > 2500
        || src_height == 0
        || src_height > 2500
        || dst_width == 0
        || dst_width > 512
        || dst_height == 0
        || dst_height > 512
    {
        return;
    }

    let src_stride = (src_width + rand::rng().random_range(0..100)) * 4;

    let mut src_data = vec![data; src_stride * src_height];
    src_data[0] = 255;
    src_data[3] = 17;

    let src_valid_size = src_stride * (src_height - 1) + src_width * 4;

    let store = ImageStore {
        buffer: std::borrow::Cow::Borrowed(&src_data[..src_valid_size]),
        channels: 4,
        width: src_width,
        height: src_height,
        stride: src_stride,
        bit_depth: 8,
    };

    let dst_stride = (dst_width + rand::rng().random_range(0..100)) * 4;
    let mut dst_data_full = vec![0u8; dst_stride * dst_height];

    let dst_valid_size = dst_stride * (dst_height - 1) + dst_width * 4;

    let mut target = ImageStoreMut {
        buffer: BufferStore::Borrowed(&mut dst_data_full[..dst_valid_size]),
        channels: 4,
        width: dst_width,
        height: dst_height,
        stride: dst_stride,
        bit_depth: 8,
    };

    let scaler = Scaler::new(sampler)
        .set_workload_strategy(workload_strategy)
        .set_threading_policy(threading_policy)
        .set_supersampling(supersampling)
        .set_multi_step_upsampling(multi_stage_upsampling);
    let planned = scaler
        .plan_rgba_resampling(store.size(), target.size(), false)
        .unwrap();
    planned.resample(&store, &mut target).unwrap();
    let store = ImageStore::<u8, 4>::borrow(&src_data, src_width, src_height).unwrap();
    let planned = scaler
        .plan_rgba_resampling(store.size(), target.size(), true)
        .unwrap();
    planned.resample(&store, &mut target).unwrap();
}

fn main() {
    // resize_rgba_fuzz(
    //     17,
    //     1023,
    //     5,
    //     6,
    //     5,
    //     ResamplingFunction::Bilinear,
    //     WorkloadStrategy::PreferSpeed,
    //     ThreadingPolicy::Single,
    //     false,
    //     true,
    // );
    // resize_rgba(0, 143, 35, 143, 35, ResamplingFunction::Bilinear, false);
    // resize_rgba(0, 1, 256, 79, 256, ResamplingFunction::Bilinear, false);
    #[allow(overflowing_literals)]
    // test_fast_image();
    let img = ImageReader::open("./assets/sample_fhd.jpg")
        .unwrap()
        .decode()
        .unwrap();
    // img.save("top_right.tga").unwrap();
    let dimensions = img.dimensions();
    let transient = img.to_rgb8();
    let mut bytes = transient
        .to_vec()
        .iter()
        .map(|&x| x as f32 / 255.)
        .collect::<Vec<_>>();

    // img.resize_exact(dimensions.0 as u32 / 4, dimensions.1 as u32 / 4, image::imageops::FilterType::Lanczos3).save("resized.png").unwrap();

    let mut scaler = Scaler::new(ResamplingFunction::Bilinear)
        .set_threading_policy(ThreadingPolicy::Single)
        .set_supersampling(false);
    // scaler.set_workload_strategy(WorkloadStrategy::PreferSpeed);

    let mut store =
        RgbF32ImageStore::from_slice(&bytes, dimensions.0 as usize, dimensions.1 as usize).unwrap();
    store.bit_depth = 10;

    let mut t_size = ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2);
    // t_size.height += 1;
    let resizing_plan = scaler
        .plan_rgb_resampling_f32(
            ImageSize::new(dimensions.0 as usize, dimensions.1 as usize),
            t_size,
        )
        .unwrap();
    let mut dst_store = RgbF32ImageStoreMut::alloc_with_depth(
        dimensions.0 as usize / 2,
        dimensions.1 as usize / 2,
        10,
    );
    resizing_plan.resample(&store, &mut dst_store).unwrap();
    // scaler.resize_rgba(&store, &mut dst_store, true).unwrap();
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
    // let dst: Vec<u8> = dst_store
    //     .as_bytes()
    //     .iter()
    //     .map(|&x| (x >> 4) as u8)
    //     .collect();

    let dst = dst_store
        .as_bytes()
        .iter()
        // .map(|&x| x)
        // .map(|&x| (((x) >> 2) as u8).min(255))
        .map(|&x| (x as f32 * 255.).round() as u8)
        .collect::<Vec<_>>();

    if dst_store.channels == 4 {
        image::save_buffer(
            "converted1.png",
            &dst,
            dst_store.width as u32,
            dst_store.height as u32,
            image::ColorType::Rgba8,
        )
        .unwrap();
    } else if dst_store.channels == 2 {
        image::save_buffer(
            "../../converted_del.png",
            &dst,
            dst_store.width as u32,
            dst_store.height as u32,
            image::ColorType::La8,
        )
        .unwrap();
    } else if dst_store.channels == 1 {
        image::save_buffer(
            "converted_s.png",
            &dst,
            dst_store.width as u32,
            dst_store.height as u32,
            image::ColorType::L8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "converted1.png",
            &dst,
            dst_store.width as u32,
            dst_store.height as u32,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
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
    let img = ImageReader::open("./assets/test_alpha.JPG")
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

    let mut dst_image = Image::new(3240, 2160, pixel_type);

    let mut resizer = Resizer::new();
    #[cfg(all(target_arch = "aarch64"))]
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
                .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom))
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
