use colorutils_rs::{lab_to_srgb, lab_with_alpha_to_rgba, rgb_to_lab, rgba_to_lab_with_alpha};

use crate::{ImageSize, ImageStore, ResamplingFunction, Scaler};

#[derive(Copy, Clone)]
pub struct LabScaler {
    pub(crate) scaler: Scaler,
}

impl<'a> LabScaler {
    pub fn new(filter: ResamplingFunction) -> Self {
        LabScaler {
            scaler: Scaler::new(filter),
        }
    }

    pub fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3> {
        const COMPONENTS: usize = 3;
        let mut lab_f32: Vec<f32> = vec![];
        lab_f32.resize(store.width * COMPONENTS * store.height, 0f32);
        let mut lab_store = ImageStore::<f32, COMPONENTS>::new(lab_f32, store.width, store.height);
        let lab_stride =
            lab_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        rgb_to_lab(
            &store.buffer,
            store.width as u32 * COMPONENTS as u32,
            &mut lab_store.buffer,
            lab_stride,
            lab_store.width as u32,
            lab_store.height as u32,
        );
        let new_store = self.scaler.resize_rgb_f32(new_size, lab_store);
        let mut new_u8_store = vec![];
        new_u8_store.resize(new_size.width * COMPONENTS * new_size.height, 0u8);
        let mut new_u8_store =
            ImageStore::<u8, COMPONENTS>::new(new_u8_store, new_size.width, new_size.height);
        let new_lab_stride =
            new_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        lab_to_srgb(
            &new_store.buffer,
            new_lab_stride,
            &mut new_u8_store.buffer,
            new_u8_store.width as u32 * COMPONENTS as u32,
            new_store.width as u32,
            new_store.height as u32,
        );
        return new_u8_store;
    }

    fn rgba_to_laba(store: ImageStore<u8, 4>) -> ImageStore<f32, 4> {
        let mut lab_image = vec![];
        lab_image.resize(store.width * 4 * store.height, 0f32);
        let lab_stride = store.width as u32 * 4u32 * std::mem::size_of::<f32>() as u32;
        rgba_to_lab_with_alpha(&store.buffer,
                               store.width as u32 * 4u32,
                               &mut lab_image,
                               lab_stride,
                               store.width as u32,
                               store.height as u32);
        let new_store = ImageStore::<f32, 4>::new(lab_image, store.width, store.height);
        return new_store;
    }

    fn laba_to_srgba(store: ImageStore<f32, 4>) -> ImageStore<u8, 4> {
        let mut rgba = vec![];
        rgba.resize(store.width * 4 * store.height, 0u8);
        lab_with_alpha_to_rgba(
            &store.buffer,
            store.width as u32 * 4u32 * std::mem::size_of::<f32>() as u32,
            &mut rgba,
            store.width as u32 * 4u32,
            store.width as u32,
            store.height as u32,
        );
        let new_store = ImageStore::<u8, 4>::new(rgba, store.width, store.height);
        return new_store;
    }

    pub fn resize_rgba(&self, new_size: ImageSize, store: ImageStore<u8, 4>) -> ImageStore<u8, 4> {
        let lab_store = Self::rgba_to_laba(store);
        let new_store = self.scaler.resize_rgba_f32(new_size, lab_store);
        let rgba_store = Self::laba_to_srgba(new_store);
        return rgba_store;
    }
}
