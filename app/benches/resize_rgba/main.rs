use criterion::{criterion_group, criterion_main, Criterion};
use fast_image_resize::images::Image;
use fast_image_resize::FilterType::Lanczos3;
use fast_image_resize::{CpuExtensions, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{GenericImageView, ImageReader};
use pic_scale::{
    ImageSize, ImageStore, ResamplingFunction, Scaler, Scaling, ScalingF32, ThreadingPolicy,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/nasa-4928x3279-rgba.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let src_bytes = img.as_bytes();
    c.bench_function("Pic scale RGBA with alpha: Lanczos 3", |b| {
        let mut copied: Vec<u8> = Vec::from(src_bytes);
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Adaptive);
            let store = ImageStore::<u8, 4>::from_slice(
                &mut copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            _ = scaler.resize_rgba(
                ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
                store,
                true,
            );
        })
    });

    let f32_image: Vec<f32> = src_bytes.iter().map(|&x| x as f32 / 255f32).collect();

    c.bench_function("Pic scale RGBA with alpha f32: Lanczos 3", |b| {
        let mut copied: Vec<f32> = Vec::from(f32_image.clone());
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Adaptive);
            let store = ImageStore::<f32, 4>::from_slice(
                &mut copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            _ = scaler.resize_rgba_f32(
                ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
                store,
                false,
            );
        })
    });

    c.bench_function("Fast image resize RGBA with alpha: Lanczos 3", |b| {
        b.iter(|| {
            let mut vc = Vec::from(img.as_bytes());
            let pixel_type: PixelType = PixelType::U8x4;
            let src_image =
                Image::from_slice_u8(dimensions.0, dimensions.1, &mut vc, pixel_type).unwrap();
            let mut dst_image = Image::new(dimensions.0 / 2, dimensions.1 / 2, pixel_type);

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
                        .resize_alg(ResizeAlg::Convolution(Lanczos3))
                        .use_alpha(true),
                )
                .unwrap();
        })
    });

    c.bench_function("Pic scale RGBA without alpha: Lanczos 3", |b| {
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Adaptive);
            let mut copied: Vec<u8> = Vec::from(src_bytes);
            let store = ImageStore::<u8, 4>::from_slice(
                &mut copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            _ = scaler.resize_rgba(
                ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
                store,
                false,
            );
        })
    });

    c.bench_function("Fast image resize RGBA without alpha: Lanczos 3", |b| {
        b.iter(|| {
            let mut vc = Vec::from(img.as_bytes());
            let pixel_type: PixelType = PixelType::U8x4;
            let src_image =
                Image::from_slice_u8(dimensions.0, dimensions.1, &mut vc, pixel_type).unwrap();
            let mut dst_image = Image::new(dimensions.0 / 2, dimensions.1 / 2, pixel_type);

            let mut resizer = Resizer::new();
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            unsafe {
                resizer.set_cpu_extensions(CpuExtensions::None);
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
                        .resize_alg(ResizeAlg::Convolution(Lanczos3))
                        .use_alpha(false),
                )
                .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
