use criterion::{criterion_group, criterion_main, Criterion};
use fast_image_resize::images::Image;
use fast_image_resize::FilterType::Lanczos3;
use fast_image_resize::{CpuExtensions, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::io::Reader as ImageReader;
use image::GenericImageView;
use pic_scale::{ImageSize, ImageStore, ResamplingFunction, Scaler, Scaling, ThreadingPolicy};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/asset_alpha_rgba.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let src_bytes = img.as_bytes();
    c.bench_function("Pic scale RGBA with alpha: Lanczos 3", |b| {
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let mut copied: Vec<u8> = src_bytes.iter().map(|&x| x).collect();
            let store = ImageStore::<u8, 4>::from_slice(
                &mut copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            );
            _ = scaler.resize_rgba(
                ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
                store,
                true,
            );
        })
    });

    let f32_image: Vec<f32> = src_bytes.iter().map(|&x| x as f32 / 255f32).collect();

    c.bench_function("Pic scale RGBA with alpha f32: Lanczos 3", |b| {
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let mut copied: Vec<f32> = f32_image.iter().map(|&x| x).collect();
            let store = ImageStore::<f32, 4>::from_slice(
                &mut copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            );
            _ = scaler.resize_rgba_f32(
                ImageSize::new(dimensions.0 as usize / 2, dimensions.1 as usize / 2),
                store,
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
                resizer.set_cpu_extensions(CpuExtensions::Sse4_1);
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
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let mut copied: Vec<u8> = src_bytes.iter().map(|&x| x).collect();
            let store = ImageStore::<u8, 4>::from_slice(
                &mut copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            );
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
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
