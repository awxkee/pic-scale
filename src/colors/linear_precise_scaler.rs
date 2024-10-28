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

use crate::alpha_check::has_non_constant_cap_alpha_rgba8;
use crate::scaler::{Scaling, ScalingF32};
use crate::support::check_image_size_overflow;
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

    fn linear_to_rgba(&self, store: ImageStore<f32, 4>) -> ImageStore<u8, 4> {
        let mut new_store = ImageStore::<u8, 4>::alloc(store.width, store.height);
        linear_to_rgba(
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

impl Scaling for LinearScaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.scaler.threading_policy = threading_policy;
    }

    fn resize_rgb(
        &self,
        new_size: ImageSize,
        store: ImageStore<u8, 3>,
    ) -> Result<ImageStore<u8, 3>, String> {
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err("One of image dimensions is 0, this should not happen".to_string());
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err("Input image larger than memory capabilities".to_string());
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err("Destination image larger than memory capabilities".to_string());
        }

        if store.width == new_size.width && store.height == new_size.height {
            return Ok(store.copied());
        }

        const COMPONENTS: usize = 3;
        let mut lab_store = ImageStore::<f32, COMPONENTS>::alloc(store.width, store.height);
        let lab_stride =
            lab_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        rgb_to_linear(
            store.buffer.borrow(),
            store.width as u32 * COMPONENTS as u32,
            lab_store.buffer.borrow_mut(),
            lab_stride,
            lab_store.width as u32,
            lab_store.height as u32,
            self.transfer_function,
        );
        let new_store = self.scaler.resize_rgb_f32(new_size, lab_store)?;
        let mut new_u8_store = ImageStore::<u8, COMPONENTS>::alloc(new_size.width, new_size.height);
        let new_lab_stride =
            new_store.width as u32 * COMPONENTS as u32 * std::mem::size_of::<f32>() as u32;
        linear_to_rgb(
            new_store.buffer.borrow(),
            new_lab_stride,
            new_u8_store.buffer.borrow_mut(),
            new_u8_store.width as u32 * COMPONENTS as u32,
            new_store.width as u32,
            new_store.height as u32,
            self.transfer_function,
        );
        Ok(new_u8_store)
    }

    fn resize_rgba(
        &self,
        new_size: ImageSize,
        store: ImageStore<u8, 4>,
        premultiply_alpha: bool,
    ) -> Result<ImageStore<u8, 4>, String> {
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err("One of image dimensions is 0, this should not happen".to_string());
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err("Input image larger than memory capabilities".to_string());
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err("Destination image larger than memory capabilities".to_string());
        }

        if store.width == new_size.width && store.height == new_size.height {
            return Ok(store.copied());
        }

        let mut src_store = store;

        let pool = self
            .scaler
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        let mut has_alpha_premultiplied = false;

        if premultiply_alpha {
            let is_alpha_premultiplication_reasonable =
                has_non_constant_cap_alpha_rgba8(src_store.buffer.borrow(), src_store.width);
            if is_alpha_premultiplication_reasonable {
                let mut premultiplied_store =
                    ImageStore::<u8, 4>::alloc(src_store.width, src_store.height);
                src_store.premultiply_alpha(&mut premultiplied_store, &pool);
                src_store = premultiplied_store;
                has_alpha_premultiplied = true;
            }
        }
        let lab_store = self.rgba_to_linear(src_store);
        let new_store = self
            .scaler
            .resize_rgba_f32_impl(new_size, lab_store, false, &pool)?;
        let rgba_store = self.linear_to_rgba(new_store);
        if premultiply_alpha && has_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(rgba_store.width, rgba_store.height);
            rgba_store.unpremultiply_alpha(&mut premultiplied_store, &pool);
            return Ok(premultiplied_store);
        }
        Ok(rgba_store)
    }
}
