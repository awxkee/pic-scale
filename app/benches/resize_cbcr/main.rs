use criterion::{criterion_group, criterion_main, Criterion};
use fast_image_resize::images::Image;
use fast_image_resize::FilterType::Lanczos3;
use fast_image_resize::{PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{GenericImageView, ImageReader};
use pic_scale::{
    ImageSize, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy,
    WorkloadStrategy,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/nasa-4928x3279-rgba.png")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let binding = img.to_luma_alpha16();
    let src_bytes = binding.as_raw().to_vec();

    c.bench_function("Pic scale CbCr10: Lanczos 3", |b| {
        let copied: Vec<u16> = src_bytes.to_vec();
        let mut store =
            ImageStore::<u16, 2>::from_slice(&copied, dimensions.0 as usize, dimensions.1 as usize)
                .unwrap();
        store.bit_depth = 10;
        let scaler = Scaler::new(ResamplingFunction::Lanczos3)
            .set_threading_policy(ThreadingPolicy::Single)
            .set_workload_strategy(WorkloadStrategy::PreferSpeed);
        let resampler = scaler
            .plan_cbcr_resampling16(
                store.size(),
                ImageSize::new(dimensions.0 as usize / 4, dimensions.1 as usize / 4),
                10,
            )
            .unwrap();
        let mut scratch = resampler.alloc_scratch();
        let mut target = ImageStoreMut::alloc_with_depth(
            dimensions.0 as usize / 4,
            dimensions.1 as usize / 4,
            10,
        );
        b.iter(|| {
            resampler
                .resample_with_scratch(&store, &mut target, &mut scratch)
                .unwrap();
        })
    });

    c.bench_function("Pic scale CbCr16: Lanczos 3", |b| {
        let copied: Vec<u16> = src_bytes.to_vec();
        let mut store =
            ImageStore::<u16, 2>::from_slice(&copied, dimensions.0 as usize, dimensions.1 as usize)
                .unwrap();
        store.bit_depth = 16;
        let scaler = Scaler::new(ResamplingFunction::Lanczos3)
            .set_threading_policy(ThreadingPolicy::Single)
            .set_workload_strategy(WorkloadStrategy::PreferSpeed);
        let resampler = scaler
            .plan_cbcr_resampling16(
                store.size(),
                ImageSize::new(dimensions.0 as usize / 4, dimensions.1 as usize / 4),
                16,
            )
            .unwrap();
        let mut scratch = resampler.alloc_scratch();
        let mut target = ImageStoreMut::alloc_with_depth(
            dimensions.0 as usize / 4,
            dimensions.1 as usize / 4,
            16,
        );
        b.iter(|| {
            resampler
                .resample_with_scratch(&store, &mut target, &mut scratch)
                .unwrap();
        })
    });

    c.bench_function("Fast image resize RGB: Lanczos 3", |b| {
        let mut copied: Vec<u8> = Vec::from(
            src_bytes
                .iter()
                .map(|&x| (((x as u16) << 2) | ((x as u16) >> 6)).to_ne_bytes())
                .flat_map(|x| x)
                .collect::<Vec<_>>(),
        );
        let pixel_type: PixelType = PixelType::U16x2;

        let src_image =
            Image::from_slice_u8(dimensions.0, dimensions.1, &mut copied, pixel_type).unwrap();
        let mut dst_image = Image::new(dimensions.0 / 4, dimensions.1 / 4, pixel_type);

        b.iter(|| {
            let mut resizer = Resizer::new();
            #[cfg(all(target_arch = "aarch64"))]
            unsafe {
                use fast_image_resize::CpuExtensions;
                resizer.set_cpu_extensions(CpuExtensions::Neon);
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            unsafe {
                use fast_image_resize::CpuExtensions;
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
