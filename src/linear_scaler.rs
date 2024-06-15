/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use colorutils_rs::{
    linear_u8_to_rgb, linear_u8_to_rgba, rgb_to_linear_u8, rgba_to_linear_u8, TransferFunction,
};

use crate::scaler::Scaling;
use crate::{ImageSize, ImageStore, ResamplingFunction, Scaler, ThreadingPolicy};

#[derive(Debug, Copy, Clone)]
/// Linearize image, scale and then convert it back
pub struct LinearScaler {
    pub(crate) scaler: Scaler,
    pub(crate) transfer_function: TransferFunction,
}

impl LinearScaler {
    pub fn new(filter: ResamplingFunction) -> Self {
        LinearScaler {
            scaler: Scaler::new(filter),
            transfer_function: TransferFunction::Srgb,
        }
    }
}

impl<'a> Scaling for LinearScaler {

    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.scaler.threading_policy = threading_policy;
    }

    fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3> {
        const CHANNELS: usize = 3;
        let mut linear_store = ImageStore::<u8, CHANNELS>::alloc(store.width, store.height);
        rgb_to_linear_u8(
            store.buffer.borrow(),
            store.width as u32 * CHANNELS as u32,
            linear_store.buffer.borrow_mut(),
            linear_store.width as u32 * CHANNELS as u32,
            linear_store.width as u32,
            linear_store.height as u32,
            self.transfer_function,
        );
        let new_store = self.scaler.resize_rgb(new_size, linear_store);
        let mut gamma_store = ImageStore::<u8, CHANNELS>::alloc(new_store.width, new_store.height);
        let src = new_store.buffer.borrow();
        let gamma_buffer = gamma_store.buffer.borrow_mut();
        linear_u8_to_rgb(
            src,
            new_store.width as u32 * CHANNELS as u32,
            gamma_buffer,
            gamma_store.width as u32 * CHANNELS as u32,
            gamma_store.width as u32,
            gamma_store.height as u32,
            self.transfer_function,
        );
        gamma_store
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
        const CHANNELS: usize = 4;
        let mut src_store = store;
        if is_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(src_store.width, src_store.height);
            src_store.unpremultiply_alpha(&mut premultiplied_store);
            src_store = premultiplied_store;
        }
        let mut linear_store = ImageStore::<u8, CHANNELS>::alloc(src_store.width, src_store.height);
        rgba_to_linear_u8(
            src_store.buffer.borrow(),
            src_store.width as u32 * CHANNELS as u32,
            linear_store.buffer.borrow_mut(),
            linear_store.width as u32 * CHANNELS as u32,
            linear_store.width as u32,
            linear_store.height as u32,
            self.transfer_function,
        );
        let new_store = self.scaler.resize_rgba(new_size, linear_store, false);
        let mut gamma_store = ImageStore::<u8, CHANNELS>::alloc(new_store.width, new_store.height);
        let src = new_store.buffer.borrow();
        let gamma_buffer = gamma_store.buffer.borrow_mut();
        linear_u8_to_rgba(
            src,
            new_store.width as u32 * CHANNELS as u32,
            gamma_buffer,
            gamma_store.width as u32 * CHANNELS as u32,
            gamma_store.width as u32,
            gamma_store.height as u32,
            self.transfer_function,
        );
        if is_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(gamma_store.width, gamma_store.height);
            gamma_store.premultiply_alpha(&mut premultiplied_store);
            return premultiplied_store;
        }
        gamma_store
    }

    fn resize_rgba_f32(&self, new_size: ImageSize, store: ImageStore<f32, 4>) -> ImageStore<f32, 4> {
        self.scaler.resize_rgba_f32(new_size, store)
    }
}
