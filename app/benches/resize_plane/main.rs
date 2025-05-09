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
    let binding = img.to_luma8();
    let binding16 = img.to_luma16();

    c.bench_function("Pic scale Plane16: Lanczos 3", |b| {
        let copied: Vec<u16> = binding16.as_raw().to_vec();
        let store =
            ImageStore::<u16, 1>::from_slice(&copied, dimensions.0 as usize, dimensions.1 as usize)
                .unwrap();
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);
            let mut target = ImageStoreMut::alloc_with_depth(
                dimensions.0 as usize / 4,
                dimensions.1 as usize / 4,
                16,
            );
            scaler.resize_plane_u16(&store, &mut target).unwrap();
        })
    });

    c.bench_function("Fast image resize Plane16: Lanczos 3", |b| {
        let jz = binding16
            .iter()
            .flat_map(|x| [x.to_ne_bytes()[0], x.to_ne_bytes()[1]])
            .collect::<Vec<u8>>();
        let mut vc = Vec::from(jz);
        let pixel_type: PixelType = PixelType::U16;

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

    c.bench_function("Pic scale Plane10: Lanczos 3", |b| {
        let copied: Vec<u16> = binding16
            .as_raw()
            .iter()
            .map(|&x| x >> 6)
            .collect::<Vec<_>>();
        let store =
            ImageStore::<u16, 1>::from_slice(&copied, dimensions.0 as usize, dimensions.1 as usize)
                .unwrap();
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);
            let mut target = ImageStoreMut::alloc_with_depth(
                dimensions.0 as usize / 4,
                dimensions.1 as usize / 4,
                10,
            );
            scaler.resize_plane_u16(&store, &mut target).unwrap();
        })
    });

    c.bench_function("Pic scale Plane8(Quality): Lanczos 3", |b| {
        let copied: Vec<u8> = binding.as_raw().to_vec();
        let store =
            ImageStore::<u8, 1>::from_slice(&copied, dimensions.0 as usize, dimensions.1 as usize)
                .unwrap();
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);
            let mut target = ImageStoreMut::alloc_with_depth(
                dimensions.0 as usize / 4,
                dimensions.1 as usize / 4,
                16,
            );
            scaler.resize_plane(&store, &mut target).unwrap();
        })
    });

    c.bench_function("Fast image resize Plane8: Lanczos 3", |b| {
        let mut vc = binding.as_raw().to_vec();
        let pixel_type: PixelType = PixelType::U8;

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

    let binding_f32 = img.to_luma32f();

    c.bench_function("Pic scale Plane32f(Speed): Lanczos 3", |b| {
        let copied: Vec<f32> = binding_f32.as_raw().to_vec();
        let store =
            ImageStore::<f32, 1>::from_slice(&copied, dimensions.0 as usize, dimensions.1 as usize)
                .unwrap();
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferSpeed);
            let mut target = ImageStoreMut::alloc_with_depth(
                dimensions.0 as usize / 4,
                dimensions.1 as usize / 4,
                16,
            );
            scaler.resize_plane_f32(&store, &mut target).unwrap();
        })
    });

    c.bench_function("Pic scale Plane32f(Quality): Lanczos 3", |b| {
        let copied: Vec<f32> = binding_f32.as_raw().to_vec();
        let store =
            ImageStore::<f32, 1>::from_slice(&copied, dimensions.0 as usize, dimensions.1 as usize)
                .unwrap();
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);
            let mut target = ImageStoreMut::alloc_with_depth(
                dimensions.0 as usize / 4,
                dimensions.1 as usize / 4,
                16,
            );
            scaler.resize_plane_f32(&store, &mut target).unwrap();
        })
    });

    c.bench_function("Fast image resize Plane F32: Lanczos 3", |b| {
        let packed_f32 = binding_f32
            .iter()
            .flat_map(|&x| x.to_ne_bytes())
            .collect::<Vec<u8>>();
        let mut vc = packed_f32.to_vec();
        let pixel_type: PixelType = PixelType::F32;

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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
