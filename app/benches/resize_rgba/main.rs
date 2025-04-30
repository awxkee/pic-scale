#![feature(f16)]
use criterion::{criterion_group, criterion_main, Criterion};
use fast_image_resize::images::Image;
use fast_image_resize::FilterType::Lanczos3;
use fast_image_resize::{CpuExtensions, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{GenericImageView, ImageReader};
use pic_scale::{
    Ar30ByteOrder, ImageStore, ImageStoreMut, LinearApproxScaler, ResamplingFunction,
    Rgba8ImageStore, Rgba8ImageStoreMut, Scaler, Scaling, ScalingF32, ScalingU16, ThreadingPolicy,
    WorkloadStrategy,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/nasa-4928x3279-rgba.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let src_bytes = img.as_bytes();

    c.bench_function("Pic scale RGBA with alpha: Lanczos 3", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);
            let store = ImageStore::<u8, 4>::from_slice(
                &copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            _ = scaler.resize_rgba(&store, &mut target, true);
        })
    });

    c.bench_function(
        "Pic scale RGBA with alpha + Linearization: Lanczos 3",
        |b| {
            let copied: Vec<u8> = Vec::from(src_bytes);
            b.iter(|| {
                let mut scaler = LinearApproxScaler::new(ResamplingFunction::Lanczos3);
                scaler.set_threading_policy(ThreadingPolicy::Single);
                let store = ImageStore::<u8, 4>::from_slice(
                    &copied,
                    dimensions.0 as usize,
                    dimensions.1 as usize,
                )
                .unwrap();
                let mut target =
                    ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
                _ = scaler.resize_rgba(&store, &mut target, true);
            })
        },
    );

    c.bench_function("Pic scale RGBA with alpha(Speed): Lanczos 3", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferSpeed);
            let store = ImageStore::<u8, 4>::from_slice(
                &copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            _ = scaler.resize_rgba(&store, &mut target, true);
        })
    });

    c.bench_function("Pic scale RGBA with alpha: Bilinear", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);
            let store = ImageStore::<u8, 4>::from_slice(
                &copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            _ = scaler.resize_rgba(&store, &mut target, true);
        })
    });

    c.bench_function("Pic scale RGBA with alpha(Speed): Bilinear", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Bilinear);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferSpeed);
            let store = ImageStore::<u8, 4>::from_slice(
                &copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            _ = scaler.resize_rgba(&store, &mut target, true);
        })
    });

    let f32_image: Vec<f32> = src_bytes.iter().map(|&x| x as f32 / 255f32).collect();

    c.bench_function("Pic scale RGBA with alpha f32: Lanczos 3", |b| {
        let copied: Vec<f32> = Vec::from(f32_image.clone());
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let store = ImageStore::<f32, 4>::from_slice(
                &copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            _ = scaler.resize_rgba_f32(&store, &mut target, false);
        })
    });

    c.bench_function("Fast image resize RGBAf32 w/o alpha: Lanczos 3", |b| {
        let packed_f32 = f32_image
            .iter()
            .flat_map(|&x| x.to_ne_bytes())
            .collect::<Vec<u8>>();
        let mut vc = Vec::from(packed_f32);
        b.iter(|| {
            let pixel_type: PixelType = PixelType::F32x4;
            let src_image =
                Image::from_slice_u8(dimensions.0, dimensions.1, &mut vc, pixel_type).unwrap();
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

    c.bench_function("Fast image resize RGBA8 with alpha: Lanczos 3", |b| {
        let mut vc = Vec::from(img.as_bytes());
        b.iter(|| {
            let pixel_type: PixelType = PixelType::U8x4;
            let src_image =
                Image::from_slice_u8(dimensions.0, dimensions.1, &mut vc, pixel_type).unwrap();
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
                        .use_alpha(true),
                )
                .unwrap();
        })
    });

    c.bench_function("Fast image resize RGBA8 with alpha: Bilinear", |b| {
        let mut vc = Vec::from(img.as_bytes());
        b.iter(|| {
            let pixel_type: PixelType = PixelType::U8x4;
            let src_image =
                Image::from_slice_u8(dimensions.0, dimensions.1, &mut vc, pixel_type).unwrap();
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
                        .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
                        .use_alpha(true),
                )
                .unwrap();
        })
    });

    c.bench_function("Pic scale RGBA without alpha: Lanczos 3", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let store = ImageStore::<u8, 4>::from_slice(
                &copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            _ = scaler.resize_rgba(&store, &mut target, false);
        })
    });

    c.bench_function("Pic scale RGBA without alpha: Lanczos 3/Quality", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);
            let store = ImageStore::<u8, 4>::from_slice(
                &copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            _ = scaler.resize_rgba(&store, &mut target, false);
        })
    });

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    c.bench_function("Apple Accelerate RGBA: Lanczos 3", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        use accelerate::{kvImageDoNotTile, vImageScale_ARGB8888, vImage_Buffer};
        b.iter(|| {
            let mut target =
                ImageStoreMut::<u8, 4>::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);

            let src_buffer = vImage_Buffer {
                data: copied.as_ptr() as *mut libc::c_void,
                width: dimensions.0 as usize,
                height: dimensions.1 as usize,
                row_bytes: dimensions.0 as usize * 4,
            };

            let target_stride = target.stride();
            let target_ptr = target.buffer.borrow_mut().as_mut_ptr() as *mut libc::c_void;

            let mut dst_buffer = vImage_Buffer {
                data: target_ptr,
                width: target.width,
                height: target.height,
                row_bytes: target_stride,
            };

            let result = unsafe {
                vImageScale_ARGB8888(
                    &src_buffer,
                    &mut dst_buffer,
                    std::ptr::null_mut(),
                    kvImageDoNotTile,
                )
            };
            if result != 0 {
                panic!("Can't resize by accelerate");
            }
        })
    });

    c.bench_function("Fast image resize RGBA without alpha: Lanczos 3", |b| {
        let mut vc = Vec::from(img.as_bytes());
        b.iter(|| {
            let pixel_type: PixelType = PixelType::U8x4;
            let src_image =
                Image::from_slice_u8(dimensions.0, dimensions.1, &mut vc, pixel_type).unwrap();
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

    c.bench_function("Pic scale RGBA10 with alpha: Lanczos 3", |b| {
        let mut copied: Vec<u16> = Vec::from(
            src_bytes
                .iter()
                .map(|&x| ((x as u16) << 2) | ((x as u16) >> 6))
                .collect::<Vec<_>>(),
        );
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let store = ImageStore::<u16, 4>::from_slice(
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
            _ = scaler.resize_rgba_u16(&store, &mut target, true);
        })
    });

    c.bench_function("Fir RGBA10 with alpha: Lanczos 3", |b| {
        let mut copied: Vec<u8> = Vec::from(
            src_bytes
                .iter()
                .map(|&x| (((x as u16) << 2) | ((x as u16) >> 6)).to_ne_bytes())
                .flat_map(|x| x)
                .collect::<Vec<_>>(),
        );
        b.iter(|| {
            let pixel_type: PixelType = PixelType::U16x4;
            let src_image =
                Image::from_slice_u8(dimensions.0, dimensions.1, &mut copied, pixel_type).unwrap();
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
                        .use_alpha(true),
                )
                .unwrap();
        })
    });

    c.bench_function("Pic scale RGBA16 without alpha: Lanczos 3", |b| {
        let mut copied: Vec<u16> = Vec::from(
            src_bytes
                .iter()
                .map(|&x| u16::from_ne_bytes([x, x]))
                .collect::<Vec<_>>(),
        );
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let store = ImageStore::<u16, 4>::from_slice(
                &mut copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target = ImageStoreMut::alloc_with_depth(
                dimensions.0 as usize / 4,
                dimensions.1 as usize / 4,
                16,
            );
            _ = scaler.resize_rgba_u16(&store, &mut target, false);
        })
    });

    c.bench_function("Pic scale RGBA10 without alpha: Lanczos 3", |b| {
        let mut copied: Vec<u16> = Vec::from(
            src_bytes
                .iter()
                .map(|&x| ((x as u16) << 2) | ((x as u16) >> 6))
                .collect::<Vec<_>>(),
        );
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            let store = ImageStore::<u16, 4>::from_slice(
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
            _ = scaler.resize_rgba_u16(&store, &mut target, false);
        })
    });

    c.bench_function("Fir RGBA10 without alpha: Lanczos 3", |b| {
        let mut copied: Vec<u8> = Vec::from(
            src_bytes
                .iter()
                .map(|&x| (((x as u16) << 2) | ((x as u16) >> 6)).to_ne_bytes())
                .flat_map(|x| x)
                .collect::<Vec<_>>(),
        );
        b.iter(|| {
            let pixel_type: PixelType = PixelType::U16x4;
            let src_image =
                Image::from_slice_u8(dimensions.0, dimensions.1, &mut copied, pixel_type).unwrap();
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

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    c.bench_function("Apple Accelerate RGBA10: Lanczos 3", |b| {
        let copied: Vec<u16> = Vec::from(
            src_bytes
                .iter()
                .map(|&x| ((x as u16) << 2) | ((x as u16) >> 6))
                .collect::<Vec<_>>(),
        );
        use accelerate::{kvImageDoNotTile, vImageScale_ARGB16U, vImage_Buffer};
        b.iter(|| {
            let mut target = ImageStoreMut::<u16, 4>::alloc(
                dimensions.0 as usize / 4,
                dimensions.1 as usize / 4,
            );

            let src_buffer = vImage_Buffer {
                data: copied.as_ptr() as *mut libc::c_void,
                width: dimensions.0 as usize,
                height: dimensions.1 as usize,
                row_bytes: dimensions.0 as usize * 4 * std::mem::size_of::<u16>(),
            };

            let target_stride = target.stride();
            let target_ptr = target.buffer.borrow_mut().as_mut_ptr() as *mut libc::c_void;

            let mut dst_buffer = vImage_Buffer {
                data: target_ptr,
                width: target.width,
                height: target.height,
                row_bytes: target_stride * std::mem::size_of::<u16>(),
            };

            let result = unsafe {
                vImageScale_ARGB16U(
                    &src_buffer,
                    &mut dst_buffer,
                    std::ptr::null_mut(),
                    kvImageDoNotTile,
                )
            };
            if result != 0 {
                panic!("Can't resize by accelerate");
            }
        })
    });

    use core::f16;

    c.bench_function("Pic scale RGBA F16 without alpha: Lanczos 3/Quality", |b| {
        let copied: Vec<f16> = vec![0.; src_bytes.len()];
        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);
            let store = ImageStore::<f16, 4>::from_slice(
                &copied,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
            let mut target =
                ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);
            scaler.resize_rgba_f16(&store, &mut target, false).unwrap();
        })
    });

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    c.bench_function("Apple Accelerate RGBAF16: Lanczos 3", |b| {
        let copied: Vec<f16> = vec![0.; src_bytes.len()];
        use accelerate::{kvImageDoNotTile, vImageScale_ARGB16F, vImage_Buffer};
        b.iter(|| {
            let mut target = ImageStoreMut::<f16, 4>::alloc(
                dimensions.0 as usize / 4,
                dimensions.1 as usize / 4,
            );

            let src_buffer = vImage_Buffer {
                data: copied.as_ptr() as *mut libc::c_void,
                width: dimensions.0 as usize,
                height: dimensions.1 as usize,
                row_bytes: dimensions.0 as usize * 4 * size_of::<f16>(),
            };

            let target_stride = target.stride();
            let target_ptr = target.buffer.borrow_mut().as_mut_ptr() as *mut libc::c_void;

            let mut dst_buffer = vImage_Buffer {
                data: target_ptr,
                width: target.width,
                height: target.height,
                row_bytes: target_stride * size_of::<f16>(),
            };

            let result = unsafe {
                vImageScale_ARGB16F(
                    &src_buffer,
                    &mut dst_buffer,
                    std::ptr::null_mut(),
                    kvImageDoNotTile,
                )
            };
            if result != 0 {
                panic!("Can't resize by accelerate");
            }
        })
    });

    c.bench_function("Pic scale RGBA1010102(N): Lanczos 3/Speed", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);

        let src_image =
            Rgba8ImageStore::borrow(&copied, dimensions.0 as usize, dimensions.1 as usize).unwrap();
        let mut dst_ar30 =
            Rgba8ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);

        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferSpeed);

            scaler
                .resize_ar30(&src_image, &mut dst_ar30, Ar30ByteOrder::Network)
                .unwrap();
        })
    });

    c.bench_function("Pic scale RGBA1010102(N): Lanczos 3/Quality", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);

        let src_image =
            Rgba8ImageStore::borrow(&copied, dimensions.0 as usize, dimensions.1 as usize).unwrap();
        let mut dst_ar30 =
            Rgba8ImageStoreMut::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);

        b.iter(|| {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_threading_policy(ThreadingPolicy::Single);
            scaler.set_workload_strategy(WorkloadStrategy::PreferQuality);

            scaler
                .resize_ar30(&src_image, &mut dst_ar30, Ar30ByteOrder::Network)
                .unwrap();
        })
    });

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    c.bench_function("Apple Accelerate RGBX1010102(N): Lanczos 3", |b| {
        let copied: Vec<u8> = Vec::from(src_bytes);
        use accelerate::{kvImageDoNotTile, vImageScale_XRGB2101010W, vImage_Buffer};

        let mut target =
            ImageStoreMut::<u8, 4>::alloc(dimensions.0 as usize / 4, dimensions.1 as usize / 4);

        b.iter(|| {
            let src_buffer = vImage_Buffer {
                data: copied.as_ptr() as *mut libc::c_void,
                width: dimensions.0 as usize,
                height: dimensions.1 as usize,
                row_bytes: dimensions.0 as usize * 4,
            };

            let target_stride = target.stride();
            let target_ptr = target.buffer.borrow_mut().as_mut_ptr() as *mut libc::c_void;

            let mut dst_buffer = vImage_Buffer {
                data: target_ptr,
                width: target.width,
                height: target.height,
                row_bytes: target_stride,
            };

            let result = unsafe {
                vImageScale_XRGB2101010W(
                    &src_buffer,
                    &mut dst_buffer,
                    std::ptr::null_mut(),
                    kvImageDoNotTile,
                )
            };
            if result != 0 {
                panic!("Can't resize by accelerate");
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
