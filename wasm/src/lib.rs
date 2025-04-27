extern crate wee_alloc;
use image::{DynamicImage, EncodableLayout, GenericImageView, ImageBuffer, ImageReader};
use js_sys::Uint8Array;
use pic_scale::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ThreadingPolicy};
use std::io::Cursor;
use std::panic;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

// Use `wee_alloc` as the global allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

pub fn set_panic_hook() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen]
pub fn process(image: Uint8Array) -> Uint8Array {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let arr = image.to_vec();
    let cursor = Cursor::new(arr);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let transient = img.to_rgba8();
    let mut bytes = Vec::from(transient.as_bytes());

    let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    scaler.set_threading_policy(ThreadingPolicy::Single);

    let store =
        ImageStore::<u8, 4>::from_slice(&mut bytes, dimensions.0 as usize, dimensions.1 as usize)
            .unwrap();

    let mut target_store =
        ImageStoreMut::<u8, 4>::alloc(dimensions.0 as usize / 3, dimensions.1 as usize / 3);

    scaler.resize_rgba(&store, &mut target_store, true).unwrap();

    let dst: Vec<u8> = target_store.buffer.borrow().to_vec();

    let img = ImageBuffer::from_raw(target_store.width as u32, target_store.height as u32, dst)
        .map(DynamicImage::ImageRgba8)
        .expect("Failed to create image from raw data");

    let mut bytes: Vec<u8> = Vec::new();

    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("Successfully write");

    let fixed_slice: &[u8] = &bytes;
    Uint8Array::from(fixed_slice)
}
