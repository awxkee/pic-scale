/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use colorutils_rs::{Luv, Rgb};

use crate::{ImageSize, ImageStore, ResamplingFunction, Scaler, Scaling, ThreadingPolicy};

#[derive(Debug, Copy, Clone)]
pub struct LuvScaler {
    pub(crate) scaler: Scaler,
}

impl LuvScaler {
    pub fn new(filter: ResamplingFunction) -> Self {
        LuvScaler {
            scaler: Scaler::new(filter),
        }
    }

    fn rgbx_to_luv<const CHANNELS: usize>(
        &self,
        store: ImageStore<u8, CHANNELS>,
    ) -> ImageStore<f32, CHANNELS> {
        let mut new_store = ImageStore::<f32, CHANNELS>::alloc(store.width, store.height);
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;
        let src_buffer = store.buffer.borrow();
        let dst_buffer = new_store.buffer.borrow_mut();
        for _ in 0..store.height {
            for x in 0..store.width {
                let px = x * CHANNELS;
                let r = *unsafe { src_buffer.get_unchecked(src_offset + px) };
                let g = *unsafe { src_buffer.get_unchecked(src_offset + px + 1) };
                let b = *unsafe { src_buffer.get_unchecked(src_offset + px + 2) };

                let rgb = Rgb::new(r, g, b);
                let luv = rgb.to_luv();
                unsafe {
                    *dst_buffer.get_unchecked_mut(dst_offset + px) = luv.l;
                    *dst_buffer.get_unchecked_mut(dst_offset + px + 1) = luv.u;
                    *dst_buffer.get_unchecked_mut(dst_offset + px + 2) = luv.v;
                }
                if CHANNELS == 4 {
                    let a = *unsafe { src_buffer.get_unchecked(src_offset + px + 3) };
                    let a_f = a as f32 * (1f32 / 255f32);
                    unsafe {
                        *dst_buffer.get_unchecked_mut(dst_offset + px + 3) = a_f;
                    }
                }
            }

            src_offset += store.width * CHANNELS;
            dst_offset += new_store.width * CHANNELS;
        }
        new_store
    }

    fn luv_to_rgbx<const CHANNELS: usize>(
        &self,
        store: ImageStore<f32, CHANNELS>,
    ) -> ImageStore<u8, CHANNELS> {
        let mut new_store = ImageStore::<u8, CHANNELS>::alloc(store.width, store.height);
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;
        let src_buffer = store.buffer.borrow();
        let dst_buffer = new_store.buffer.borrow_mut();
        for _ in 0..store.height {
            for x in 0..store.width {
                let px = x * CHANNELS;
                let l = *unsafe { src_buffer.get_unchecked(src_offset + px) };
                let u = *unsafe { src_buffer.get_unchecked(src_offset + px + 1) };
                let v = *unsafe { src_buffer.get_unchecked(src_offset + px + 2) };

                let luv = Luv::new(l, u, v);
                let rgb = luv.to_rgb();
                unsafe {
                    *dst_buffer.get_unchecked_mut(dst_offset + px) = rgb.r;
                    *dst_buffer.get_unchecked_mut(dst_offset + px + 1) = rgb.g;
                    *dst_buffer.get_unchecked_mut(dst_offset + px + 2) = rgb.b;
                }
                if CHANNELS == 4 {
                    let a = *unsafe { src_buffer.get_unchecked(src_offset + px + 3) };
                    let a_f = a * 255f32;
                    unsafe {
                        *dst_buffer.get_unchecked_mut(dst_offset + px + 3) = a_f as u8;
                    }
                }
            }

            src_offset += store.width * CHANNELS;
            dst_offset += new_store.width * CHANNELS;
        }
        new_store
    }
}

impl Scaling for LuvScaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.scaler.set_threading_policy(threading_policy)
    }

    fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3> {
        let luv_image = self.rgbx_to_luv(store);
        let new_store = self.scaler.resize_rgb_f32(new_size, luv_image);
        let unorm_image = self.luv_to_rgbx(new_store);
        unorm_image
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
        let luv_image = self.rgbx_to_luv(src_store);
        let new_store = self.scaler.resize_rgba_f32(new_size, luv_image);
        let unorm_image = self.luv_to_rgbx(new_store);
        if is_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(unorm_image.width, unorm_image.height);
            unorm_image.premultiply_alpha(&mut premultiplied_store);
            return premultiplied_store;
        }
        return unorm_image;
    }

    fn resize_rgba_f32(
        &self,
        new_size: ImageSize,
        store: ImageStore<f32, 4>,
    ) -> ImageStore<f32, 4> {
        self.scaler.resize_rgba_f32(new_size, store)
    }
}
