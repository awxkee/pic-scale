use criterion::{criterion_group, criterion_main, Criterion};
use fast_image_resize::images::Image;
use fast_image_resize::FilterType::Lanczos3;
use fast_image_resize::{CpuExtensions, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{EncodableLayout, GenericImageView, ImageReader};
use pic_scale::{
    ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ScalingF32, ScalingU16,
    ThreadingPolicy, WorkloadStrategy,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/nasa-4928x3279-rgba.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let binding = img.to_rgb8();
    let src_bytes = binding.as_bytes();

    c.bench_function("Pic scale RGB: Lanczos 3", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        let store =
            ImageStore::<u8, 3>::from_slice(&copied, dimensions.0 as usize, dimensions.1 as usize)
                .unwrap();
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferSpeed);
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            scaler.resize_rgb(&store, &mut target).unwrap();
        })
    });

    c.bench_function("Pic scale RGB: Lanczos 3/Quality", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        let store =
            ImageStore::<u8, 3>::from_slice(&copied, dimensions.0 as usize, dimensions.1 as usize)
                .unwrap();
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            scaler.resize_rgb(&store, &mut target).unwrap();
        })
    });

    let f32_image: Vec<f32> = src_bytes.iter().map(|&x| x as f32 / 255f32).collect();

    c.bench_function("Pic scale RGB f32: Lanczos 3", |b| {
        let mut copied: Vec<f32> = Vec::from(f32_image.clone());
        let store = ImageStore::<f32, 3>::from_slice(
            &mut copied,
            dimensions.0 as usize,
            dimensions.1 as usize,
        )
        .unwrap();
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            scaler.resize_rgb_f32(&store, &mut target).unwrap();
        })
    });

    c.bench_function("Fast image resize RGB: Lanczos 3", |b| {
        let mut vc = Vec::from(img.as_bytes());
        let pixel_type: PixelType = PixelType::U8x3;

        let src_image =
            Image::from_slice_u8(dimensions.0, dimensions.1, &mut vc, pixel_type).unwrap();
        b.iter(|| {
            let mut dst_image = Image::new(dimensions.0 / 4, dimensions.1 / 4, pixel_type);

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
                        .use_alpha(false),
                )
                .unwrap();
        })
    });

    c.bench_function("Pic scale RGB10 without alpha: Lanczos 3", |b| {
        let mut copied: Vec<u16> = Vec::from(
            src_bytes
                .iter()
                .map(|&x| ((x as u16) << 2) | ((x as u16) >> 6))
                .collect::<Vec<_>>(),
        );
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let store = ImageStore::<u16, 3>::from_slice(
                &mut copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target = ImageStoreMut::alloc_with_depth(
                dimensions.0 as usize / 4,
                dimensions.1 as usize / 4,
                10,
            );
            _ = scaler.resize_rgb_u16(&store, &mut target);
        })
    });

    c.bench_function("Fast image resize RGB10: Lanczos 3", |b| {
        let mut copied: Vec<u8> = Vec::from(
            src_bytes
                .iter()
                .map(|&x| (((x as u16) << 2) | ((x as u16) >> 6)).to_ne_bytes())
                .flat_map(|x| x)
                .collect::<Vec<_>>(),
        );
        let pixel_type: PixelType = PixelType::U16x3;

        let src_image =
            Image::from_slice_u8(dimensions.0, dimensions.1, &mut copied, pixel_type).unwrap();
        b.iter(|| {
            let mut dst_image = Image::new(dimensions.0 / 4, dimensions.1 / 4, pixel_type);

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
                        .use_alpha(false),
                )
                .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
