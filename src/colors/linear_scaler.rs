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

use colorutils_rs::{
    linear_u8_to_rgb, linear_u8_to_rgba, rgb_to_linear_u8, rgba_to_linear_u8, TransferFunction,
};

use crate::scaler::Scaling;
use crate::{ImageSize, ImageStore, ResamplingFunction, Scaler, ThreadingPolicy};

#[derive(Debug, Copy, Clone)]
/// Linearize image into u8, scale and then convert it back. It's much faster than scale in f32, however involves some precision loss
pub struct LinearApproxScaler {
    pub(crate) scaler: Scaler,
    pub(crate) transfer_function: TransferFunction,
}

impl LinearApproxScaler {
    /// Creates new instance with sRGB transfer function
    pub fn new(filter: ResamplingFunction) -> Self {
        LinearApproxScaler {
            scaler: Scaler::new(filter),
            transfer_function: TransferFunction::Srgb,
        }
    }

    /// Creates new instance with provided transfer function
    pub fn new_with_transfer(
        filter: ResamplingFunction,
        transfer_function: TransferFunction,
    ) -> Self {
        LinearApproxScaler {
            scaler: Scaler::new(filter),
            transfer_function,
        }
    }
}

impl Scaling for LinearApproxScaler {
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
            src_store.unpremultiply_alpha(&mut premultiplied_store, self.scaler.threading_policy);
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
            gamma_store.premultiply_alpha(&mut premultiplied_store, self.scaler.threading_policy);
            return premultiplied_store;
        }
        gamma_store
    }
}
