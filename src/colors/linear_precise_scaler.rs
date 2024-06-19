/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::scaler::Scaling;
use crate::{ImageSize, ImageStore, ResamplingFunction, Scaler, ThreadingPolicy};
use colorutils_rs::{
    linear_to_rgb, linear_to_rgba, rgb_to_linear, rgba_to_linear, TransferFunction,
};

#[derive(Debug, Copy, Clone)]
/// Converts image to linear f32 components scales it and convert back. This is more precise and slower than scaling in linear u8
/// If you want just scale faster use *LinearApproxScaler*
pub struct LinearScaler {
    pub(crate) scaler: Scaler,
    pub(crate) transfer_function: TransferFunction,
}

impl LinearScaler {
    /// Creates new instance with sRGB transfer function
    pub fn new(filter: ResamplingFunction) -> Self {
        LinearScaler {
            scaler: Scaler::new(filter),
            transfer_function: TransferFunction::Srgb,
        }
    }

    /// Creates new instance with requested transfer function
    pub fn new_with_transfer(
        filter: ResamplingFunction,
        transfer_function: TransferFunction,
    ) -> Self {
        LinearScaler {
            scaler: Scaler::new(filter),
            transfer_function,
        }
    }

    fn rgba_to_linear(&self, store: ImageStore<u8, 4>) -> ImageStore<f32, 4> {
        let mut new_store = ImageStore::<f32, 4>::alloc(store.width, store.height);
        let lab_stride = store.width as u32 * 4u32 * std::mem::size_of::<f32>() as u32;
        rgba_to_linear(
            &store.buffer.borrow(),
            store.width as u32 * 4u32,
            &mut new_store.buffer.borrow_mut(),
            lab_stride,
            store.width as u32,
            store.height as u32,
            self.transfer_function,
        );
        return new_store;
    }

    fn linear_to_rgba(&self, store: ImageStore<f32, 4>) -> ImageStore<u8, 4> {
        let mut new_store = ImageStore::<u8, 4>::alloc(store.width, store.height);
        linear_to_rgba(
            &store.buffer.borrow(),
            store.width as u32 * 4u32 * std::mem::size_of::<f32>() as u32,
            &mut new_store.buffer.borrow_mut(),
            store.width as u32 * 4u32,
            store.width as u32,
            store.height as u32,
            self.transfer_function,
        );
        return new_store;
    }
}

impl Scaling for LinearScaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.scaler.threading_policy = threading_policy;
    }

    fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3> {
        const COMPONENTS: usize = 3;
        let mut lab_store = ImageStore::<f32, COMPONENTS>::alloc(store.width, store.height);
        let lab_stride =
            lab_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        rgb_to_linear(
            &store.buffer.borrow(),
            store.width as u32 * COMPONENTS as u32,
            &mut lab_store.buffer.borrow_mut(),
            lab_stride,
            lab_store.width as u32,
            lab_store.height as u32,
            self.transfer_function,
        );
        let new_store = self.scaler.resize_rgb_f32(new_size, lab_store);
        let mut new_u8_store = ImageStore::<u8, COMPONENTS>::alloc(new_size.width, new_size.height);
        let new_lab_stride =
            new_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        linear_to_rgb(
            &new_store.buffer.borrow(),
            new_lab_stride,
            &mut new_u8_store.buffer.borrow_mut(),
            new_u8_store.width as u32 * COMPONENTS as u32,
            new_store.width as u32,
            new_store.height as u32,
            self.transfer_function,
        );
        return new_u8_store;
    }

    fn resize_rgb_f32(&self, new_size: ImageSize, store: ImageStore<f32, 3>) -> ImageStore<f32, 3> {
        self.scaler.resize_rgb_f32(new_size, store)
    }

    fn resize_rgba(
        &self,
        new_size: ImageSize,
        store: ImageStore<u8, 4>,
        is_alpha_premultiplied: bool,
    ) -> ImageStore<u8, 4> {
        let mut src_store = store;
        if is_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(src_store.width, src_store.height);
            src_store.unpremultiply_alpha(&mut premultiplied_store);
            src_store = premultiplied_store;
        }
        let lab_store = self.rgba_to_linear(src_store);
        let new_store = self.scaler.resize_rgba_f32(new_size, lab_store);
        let rgba_store = self.linear_to_rgba(new_store);
        if is_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(rgba_store.width, rgba_store.height);
            rgba_store.premultiply_alpha(&mut premultiplied_store);
            return premultiplied_store;
        }
        return rgba_store;
    }

    fn resize_rgba_f32(
        &self,
        new_size: ImageSize,
        store: ImageStore<f32, 4>,
    ) -> ImageStore<f32, 4> {
        self.scaler.resize_rgba_f32(new_size, store)
    }
}
