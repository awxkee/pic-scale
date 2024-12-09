#![no_main]

use libfuzzer_sys::fuzz_target;
use pic_scale::{ImageSize, ImageStore, ResamplingFunction, Scaler, ScalingF32};

fuzz_target!(|data: (u16, u16, u16, u16)| {
    resize_rgba(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        data.3 as usize,
        ResamplingFunction::Bilinear,
    )
});

fn resize_rgba(
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

    let mut src_data = vec![0f32; src_width * src_height * 4];

    let store = ImageStore::<f32, 4>::from_slice(&mut src_data, src_width, src_height).unwrap();
    let scaler = Scaler::new(sampler);
    _ = scaler
        .resize_rgba_f32(ImageSize::new(dst_width, dst_height), store, false)
        .unwrap();
    let store = ImageStore::<f32, 4>::from_slice(&mut src_data, src_width, src_height).unwrap();
    _ = scaler
        .resize_rgba_f32(ImageSize::new(dst_width, dst_height), store, true)
        .unwrap();
}
