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
    SRGB_TO_XYZ_D65, TransferFunction, XYZ_TO_SRGB_D65, rgba_to_xyz_with_alpha, srgb_to_xyz,
    xyz_to_srgb, xyz_with_alpha_to_rgba,
};

use crate::pic_scale_error::PicScaleError;
use crate::scaler::{Scaling, ScalingF32};
use crate::support::check_image_size_overflow;
use crate::{ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy};

#[derive(Debug, Copy, Clone)]
/// Converts image to CIE XYZ components scales it and convert back
pub struct XYZScaler {
    pub(crate) scaler: Scaler,
}

impl XYZScaler {
    pub fn new(filter: ResamplingFunction) -> Self {
        XYZScaler {
            scaler: Scaler::new(filter),
        }
    }
}

impl Scaling for XYZScaler {
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

    fn resize_gray_alpha<'a>(
        &'a self,
        _: &ImageStore<'a, u8, 2>,
        _: &mut ImageStoreMut<'a, u8, 2>,
        _: bool,
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

        let mut target_vertical = vec![f32::default(); store.width * store.height * CN];

        let mut lab_store =
            ImageStoreMut::<f32, CN>::from_slice(&mut target_vertical, store.width, store.height)?;
        lab_store.bit_depth = into.bit_depth;

        let lab_stride = lab_store.width as u32 * CN as u32 * size_of::<f32>() as u32;

        srgb_to_xyz(
            store.buffer.as_ref(),
            store.width as u32 * CN as u32,
            lab_store.buffer.borrow_mut(),
            lab_stride,
            lab_store.width as u32,
            lab_store.height as u32,
        );

        let new_immutable_store = ImageStore::<f32, CN> {
            buffer: std::borrow::Cow::Owned(target_vertical),
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
        xyz_to_srgb(
            new_store.buffer.borrow(),
            new_lab_stride,
            into.buffer.borrow_mut(),
            into.width as u32 * CN as u32,
            new_store.width as u32,
            new_store.height as u32,
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

        const CN: usize = 4;

        let mut target_vertical = vec![f32::default(); store.width * store.height * CN];

        let mut lab_store =
            ImageStoreMut::<f32, CN>::from_slice(&mut target_vertical, store.width, store.height)?;
        lab_store.bit_depth = into.bit_depth;

        let lab_stride = lab_store.width as u32 * CN as u32 * size_of::<f32>() as u32;

        rgba_to_xyz_with_alpha(
            store.buffer.as_ref(),
            store.width as u32 * CN as u32,
            lab_store.buffer.borrow_mut(),
            lab_stride,
            lab_store.width as u32,
            lab_store.height as u32,
            &SRGB_TO_XYZ_D65,
            TransferFunction::Srgb,
        );

        let new_immutable_store = ImageStore::<f32, CN> {
            buffer: std::borrow::Cow::Owned(target_vertical),
            channels: CN,
            width: store.width,
            height: store.height,
            stride: store.width * CN,
            bit_depth: into.bit_depth,
        };

        let mut new_store = ImageStoreMut::<f32, CN>::alloc(into.width, into.height);

        self.scaler
            .resize_rgba_f32(&new_immutable_store, &mut new_store, premultiply_alpha)?;
        let new_lab_stride = new_store.width as u32 * CN as u32 * size_of::<f32>() as u32;
        xyz_with_alpha_to_rgba(
            new_store.buffer.borrow(),
            new_lab_stride,
            into.buffer.borrow_mut(),
            into.width as u32 * CN as u32,
            new_store.width as u32,
            new_store.height as u32,
            &XYZ_TO_SRGB_D65,
            TransferFunction::Srgb,
        );
        Ok(())
    }
}
