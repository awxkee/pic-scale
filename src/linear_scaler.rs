use std::time::Instant;
use colorutils_rs::{linear_to_rgb, linear_to_rgba, rgb_to_linear, rgba_to_linear, TransferFunction};
use crate::{ImageSize, ImageStore, ResamplingFunction, Scaler, ThreadingPolicy};

#[derive(Copy, Clone)]
pub struct LinearScaler {
    pub(crate) scaler: Scaler,
}

impl<'a> LinearScaler {
    pub fn new(filter: ResamplingFunction) -> Self {
        LinearScaler {
            scaler: Scaler::new(filter),
        }
    }

    pub fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.scaler.threading_policy = threading_policy;
    }

    pub fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3> {
        const COMPONENTS: usize = 3;
        let mut lab_f32: Vec<f32> = vec![];
        lab_f32.resize(store.width * COMPONENTS * store.height, 0f32);
        let mut lab_store = ImageStore::<f32, COMPONENTS>::new(lab_f32, store.width, store.height);
        let lab_stride =
            lab_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        let start_time = Instant::now();
        rgb_to_linear(
            &store.buffer.borrow(),
            store.width as u32 * COMPONENTS as u32,
            &mut lab_store.buffer.borrow_mut(),
            lab_stride,
            lab_store.width as u32,
            lab_store.height as u32,
            TransferFunction::Srgb,
        );
        let elapsed_time = start_time.elapsed();
        // Print the elapsed time in milliseconds
        println!("rgb_to_linear: {:.2?}", elapsed_time);
        let new_store = self.scaler.resize_rgb_f32(new_size, lab_store);
        let mut new_u8_store = vec![];
        new_u8_store.resize(new_size.width * COMPONENTS * new_size.height, 0u8);
        let mut new_u8_store =
            ImageStore::<u8, COMPONENTS>::new(new_u8_store, new_size.width, new_size.height);
        let new_lab_stride =
            new_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        let start_time = Instant::now();
        linear_to_rgb(
            &new_store.buffer.borrow(),
            new_lab_stride,
            &mut new_u8_store.buffer.borrow_mut(),
            new_u8_store.width as u32 * COMPONENTS as u32,
            new_store.width as u32,
            new_store.height as u32,
            TransferFunction::Srgb,
        );
        let elapsed_time = start_time.elapsed();
        // Print the elapsed time in milliseconds
        println!("linear_to_rgb: {:.2?}", elapsed_time);
        return new_u8_store;
    }

    pub fn resize_rgba(&self, new_size: ImageSize, store: ImageStore<u8, 4>) -> ImageStore<u8, 4> {
        const COMPONENTS: usize = 4;
        let mut lab_f32: Vec<f32> = vec![];
        lab_f32.resize(store.width * COMPONENTS * store.height, 0f32);
        let mut lab_store = ImageStore::<f32, COMPONENTS>::new(lab_f32, store.width, store.height);
        let lab_stride =
            lab_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        rgba_to_linear(
            &store.buffer.borrow(),
            store.width as u32 * COMPONENTS as u32,
            &mut lab_store.buffer.borrow_mut(),
            lab_stride,
            lab_store.width as u32,
            lab_store.height as u32,
            TransferFunction::Srgb,
        );
        let new_store = self.scaler.resize_rgba_f32(new_size, lab_store);
        let mut new_u8_store = vec![];
        new_u8_store.resize(new_size.width * COMPONENTS * new_size.height, 0u8);
        let mut new_u8_store =
            ImageStore::<u8, COMPONENTS>::new(new_u8_store, new_size.width, new_size.height);
        let new_lab_stride =
            new_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        linear_to_rgba(
            &new_store.buffer.borrow(),
            new_lab_stride,
            &mut new_u8_store.buffer.borrow_mut(),
            new_u8_store.width as u32 * COMPONENTS as u32,
            new_store.width as u32,
            new_store.height as u32,
            TransferFunction::Srgb,
        );
        return new_u8_store;
    }
}
