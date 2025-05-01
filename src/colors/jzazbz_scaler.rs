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
    jzazbz_to_rgb, jzazbz_to_rgba, rgb_to_jzazbz, rgba_to_jzazbz, TransferFunction,
};

use crate::pic_scale_error::PicScaleError;
use crate::scaler::ScalingF32;
use crate::support::check_image_size_overflow;
use crate::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ThreadingPolicy};

#[derive(Debug, Copy, Clone)]
/// Converts image to *Jzazbz* components scales it and convert back
pub struct JzazbzScaler {
    pub(crate) scaler: Scaler,
    pub(crate) transfer_function: TransferFunction,
    pub(crate) display_luminance: f32,
}

impl JzazbzScaler {
    /// # Arguments
    /// - `transfer_function` - Transfer function to move into linear colorspace and back
    /// - `display_luminance` - Display luminance for *Jzazbz*
    pub fn new(
        filter: ResamplingFunction,
        display_luminance: f32,
        transfer_function: TransferFunction,
    ) -> Self {
        JzazbzScaler {
            scaler: Scaler::new(filter),
            transfer_function,
            display_luminance,
        }
    }

    fn rgba_to_laba<'a>(&self, store: &ImageStore<'a, u8, 4>) -> ImageStore<'a, f32, 4> {
        let mut source_slice = vec![f32::default(); 4 * store.width * store.height];
        let lab_stride = store.width as u32 * 4u32 * size_of::<f32>() as u32;
        rgba_to_jzazbz(
            store.buffer.as_ref(),
            store.width as u32 * 4u32,
            &mut source_slice,
            lab_stride,
            store.width as u32,
            store.height as u32,
            self.display_luminance,
            self.transfer_function,
        );
        let new_store = ImageStore::<f32, 4> {
            buffer: std::borrow::Cow::Owned(source_slice),
            channels: 4,
            width: store.width,
            height: store.height,
            stride: store.width * 4,
            bit_depth: store.bit_depth,
        };
        new_store
    }

    fn laba_to_srgba<'a>(
        &self,
        store: &ImageStoreMut<'a, f32, 4>,
        into: &mut ImageStoreMut<'a, u8, 4>,
    ) {
        jzazbz_to_rgba(
            store.buffer.borrow(),
            store.width as u32 * 4u32 * size_of::<f32>() as u32,
            into.buffer.borrow_mut(),
            store.width as u32 * 4u32,
            store.width as u32,
            store.height as u32,
            self.display_luminance,
            self.transfer_function,
        );
    }
}

impl Scaling for JzazbzScaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.scaler.threading_policy = threading_policy;
    }

    fn resize_plane<'a>(
        &'a self,
        _: &ImageStore<'a, u8, 1>,
        _: &mut ImageStoreMut<'a, u8, 1>,
    ) -> Result<(), PicScaleError> {
        unimplemented!()
    }

    fn resize_cbcr8<'a>(
        &'a self,
        _: &ImageStore<'a, u8, 2>,
        _: &mut ImageStoreMut<'a, u8, 2>,
    ) -> Result<(), PicScaleError> {
        unimplemented!()
    }

    fn resize_rgb<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 3>,
        into: &mut ImageStoreMut<'a, u8, 3>,
    ) -> Result<(), PicScaleError> {
        let new_size = into.get_size();
        into.validate()?;
        store.validate()?;
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err(PicScaleError::ZeroImageDimensions);
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err(PicScaleError::SourceImageIsTooLarge);
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err(PicScaleError::DestinationImageIsTooLarge);
        }

        if store.width == new_size.width && store.height == new_size.height {
            store.copied_to_mut(into);
            return Ok(());
        }

        const CN: usize = 3;

        let mut target = vec![f32::default(); store.width * store.height * CN];

        let mut lab_store =
            ImageStoreMut::<f32, CN>::from_slice(&mut target, store.width, store.height)?;
        lab_store.bit_depth = into.bit_depth;

        let lab_stride = lab_store.width as u32 * CN as u32 * size_of::<f32>() as u32;
        rgb_to_jzazbz(
            store.buffer.as_ref(),
            store.width as u32 * CN as u32,
            lab_store.buffer.borrow_mut(),
            lab_stride,
            lab_store.width as u32,
            lab_store.height as u32,
            self.display_luminance,
            self.transfer_function,
        );

        let new_immutable_store = ImageStore::<f32, CN> {
            buffer: std::borrow::Cow::Owned(target),
            channels: CN,
            width: store.width,
            height: store.height,
            stride: store.width * CN,
            bit_depth: into.bit_depth,
        };

        let mut new_store = ImageStoreMut::<f32, CN>::alloc(into.width, into.height);
        self.scaler
            .resize_rgb_f32(&new_immutable_store, &mut new_store)?;

        let new_lab_stride = new_store.width as u32 * CN as u32 * size_of::<f32>() as u32;

        jzazbz_to_rgb(
            new_store.buffer.borrow(),
            new_lab_stride,
            into.buffer.borrow_mut(),
            into.width as u32 * CN as u32,
            new_store.width as u32,
            new_store.height as u32,
            self.display_luminance,
            self.transfer_function,
        );
        Ok(())
    }

    fn resize_rgba<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 4>,
        into: &mut ImageStoreMut<'a, u8, 4>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError> {
        let new_size = into.get_size();
        into.validate()?;
        store.validate()?;
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err(PicScaleError::ZeroImageDimensions);
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err(PicScaleError::SourceImageIsTooLarge);
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err(PicScaleError::DestinationImageIsTooLarge);
        }

        if store.width == new_size.width && store.height == new_size.height {
            store.copied_to_mut(into);
            return Ok(());
        }

        let lab_store = self.rgba_to_laba(store);
        let mut new_target_store = ImageStoreMut::alloc(new_size.width, new_size.height);

        self.scaler
            .resize_rgba_f32(&lab_store, &mut new_target_store, premultiply_alpha)?;
        self.laba_to_srgba(&new_target_store, into);
        Ok(())
    }
}
