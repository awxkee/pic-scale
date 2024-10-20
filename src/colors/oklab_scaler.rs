/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use colorutils_rs::{oklab_to_rgb, oklab_to_rgba, rgb_to_oklab, rgba_to_oklab, TransferFunction};

use crate::scaler::ScalingF32;
use crate::{ImageSize, ImageStore, ResamplingFunction, Scaler, Scaling, ThreadingPolicy};

#[derive(Debug, Copy, Clone)]
/// Converts image to *Oklab* components scales it and convert back
pub struct OklabScaler {
    pub(crate) scaler: Scaler,
    pub(crate) transfer_function: TransferFunction,
}

impl OklabScaler {
    /// # Arguments
    /// - `transfer_function` - Transfer function to move into linear colorspace and back
    pub fn new(filter: ResamplingFunction, transfer_function: TransferFunction) -> Self {
        OklabScaler {
            scaler: Scaler::new(filter),
            transfer_function,
        }
    }

    fn rgba_to_laba(&self, store: ImageStore<u8, 4>) -> ImageStore<f32, 4> {
        let mut new_store = ImageStore::<f32, 4>::alloc(store.width, store.height);
        let lab_stride = store.width as u32 * 4u32 * std::mem::size_of::<f32>() as u32;
        rgba_to_oklab(
            store.buffer.borrow(),
            store.width as u32 * 4u32,
            new_store.buffer.borrow_mut(),
            lab_stride,
            store.width as u32,
            store.height as u32,
            self.transfer_function,
        );
        new_store
    }

    fn laba_to_srgba(&self, store: ImageStore<f32, 4>) -> ImageStore<u8, 4> {
        let mut new_store = ImageStore::<u8, 4>::alloc(store.width, store.height);
        oklab_to_rgba(
            store.buffer.borrow(),
            store.width as u32 * 4u32 * std::mem::size_of::<f32>() as u32,
            new_store.buffer.borrow_mut(),
            store.width as u32 * 4u32,
            store.width as u32,
            store.height as u32,
            self.transfer_function,
        );
        new_store
    }
}

impl Scaling for OklabScaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.scaler.threading_policy = threading_policy;
    }

    fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3> {
        const COMPONENTS: usize = 3;
        let mut lab_store = ImageStore::<f32, COMPONENTS>::alloc(store.width, store.height);
        let lab_stride =
            lab_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        rgb_to_oklab(
            store.buffer.borrow(),
            store.width as u32 * COMPONENTS as u32,
            lab_store.buffer.borrow_mut(),
            lab_stride,
            lab_store.width as u32,
            lab_store.height as u32,
            self.transfer_function,
        );
        let new_store = self.scaler.resize_rgb_f32(new_size, lab_store);
        let mut new_u8_store = ImageStore::<u8, COMPONENTS>::alloc(new_size.width, new_size.height);
        let new_lab_stride =
            new_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        oklab_to_rgb(
            new_store.buffer.borrow(),
            new_lab_stride,
            new_u8_store.buffer.borrow_mut(),
            new_u8_store.width as u32 * COMPONENTS as u32,
            new_store.width as u32,
            new_store.height as u32,
            self.transfer_function,
        );
        new_u8_store
    }

    fn resize_rgba(
        &self,
        new_size: ImageSize,
        store: ImageStore<u8, 4>,
        premultiply_alpha: bool,
    ) -> ImageStore<u8, 4> {
        let mut src_store = store;

        let pool = self
            .scaler
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if premultiply_alpha {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(src_store.width, src_store.height);
            src_store.premultiply_alpha(&mut premultiplied_store, &pool);
            src_store = premultiplied_store;
        }
        let lab_store = self.rgba_to_laba(src_store);
        let new_store = self
            .scaler
            .resize_rgba_f32_impl(new_size, lab_store, false, &pool);
        let rgba_store = self.laba_to_srgba(new_store);
        if premultiply_alpha {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(rgba_store.width, rgba_store.height);
            rgba_store.unpremultiply_alpha(&mut premultiplied_store, &pool);
            return premultiplied_store;
        }
        rgba_store
    }
}
