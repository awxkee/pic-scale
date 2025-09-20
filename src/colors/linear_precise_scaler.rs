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

use crate::pic_scale_error::{PicScaleError, try_vec};
use crate::scaler::{Scaling, ScalingF32};
use crate::support::check_image_size_overflow;
use crate::{
    ImageStore, ImageStoreMut, ImageStoreScaling, ResamplingFunction, Scaler, ScalingOptions,
    ScalingU16, ThreadingPolicy,
};
use colorutils_rs::TransferFunction;

#[derive(Debug, Copy, Clone)]
/// Converts image to linear f32 components scales it and convert back.
///
/// This is more precise and slower than scaling in linear fixed point.
///
/// This is an expensive method, and its precision might not be required,
/// consider to use [LinearApproxScaler] instead
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
}

struct Linearization {
    linearization: Box<[f32; 256]>,
    gamma: Box<[u8; 65536]>,
}

struct Linearization16 {
    linearization: Box<[f32; 65536]>,
    gamma: Box<[u16; 262144]>,
}

fn make_linearization16(
    transfer_function: TransferFunction,
    bit_depth: usize,
) -> Result<Linearization16, PicScaleError> {
    if bit_depth < 8 {
        return Err(PicScaleError::UnsupportedBitDepth(bit_depth));
    }
    let mut linearizing = Box::new([0f32; 65536]);
    let max_lin_depth = (1u32 << bit_depth) - 1;
    let keep_max = 1u32 << bit_depth;
    let mut gamma = Box::new([0u16; 262144]);

    for (i, dst) in linearizing.iter_mut().take(keep_max as usize).enumerate() {
        *dst = transfer_function.linearize(i as f32 / max_lin_depth as f32);
    }

    for (i, dst) in gamma.iter_mut().enumerate() {
        *dst = (transfer_function.gamma(i as f32 / 262143.) * max_lin_depth as f32)
            .round()
            .min(max_lin_depth as f32) as u16;
    }

    Ok(Linearization16 {
        linearization: linearizing,
        gamma,
    })
}

fn make_linearization(transfer_function: TransferFunction) -> Linearization {
    let mut linearizing = Box::new([0f32; 256]);
    let mut gamma = Box::new([0u8; 65536]);

    for (i, dst) in linearizing.iter_mut().enumerate() {
        *dst = transfer_function.linearize(i as f32 / 255.);
    }

    for (i, dst) in gamma.iter_mut().enumerate() {
        *dst = (transfer_function.gamma(i as f32 / 65535.) * 255.)
            .round()
            .min(255.) as u8;
    }

    Linearization {
        linearization: linearizing,
        gamma,
    }
}

fn resize_typical8<'a, const CN: usize>(
    resampling_function: ResamplingFunction,
    transfer_function: TransferFunction,
    threading_policy: ThreadingPolicy,
    store: &ImageStore<'a, u8, CN>,
    into: &mut ImageStoreMut<'a, u8, CN>,
) -> Result<(), PicScaleError>
where
    ImageStore<'a, f32, CN>: ImageStoreScaling<'a, f32, CN>,
{
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

    let mut target_vertical = try_vec![f32::default(); store.width * store.height * CN];

    let mut linear_store =
        ImageStoreMut::<f32, CN>::from_slice(&mut target_vertical, store.width, store.height)?;

    let linearization = make_linearization(transfer_function);

    for (&src, dst) in store
        .as_bytes()
        .iter()
        .zip(linear_store.buffer.borrow_mut())
    {
        *dst = linearization.linearization[src as usize];
    }

    let new_immutable_store = ImageStore::<f32, CN> {
        buffer: std::borrow::Cow::Owned(target_vertical),
        channels: CN,
        width: store.width,
        height: store.height,
        stride: store.width * CN,
        bit_depth: 12,
    };

    let mut new_store =
        ImageStoreMut::<f32, CN>::try_alloc_with_depth(into.width, into.height, 12)?;

    new_immutable_store.scale(
        &mut new_store,
        ScalingOptions {
            resampling_function,
            threading_policy,
            ..Default::default()
        },
    )?;

    for (&src, dst) in new_store.as_bytes().iter().zip(into.buffer.borrow_mut()) {
        let v = (src * 65535.).round().min(65535.).max(0.) as u16;
        *dst = linearization.gamma[v as usize];
    }

    Ok(())
}

fn resize_typical16<'a, const CN: usize>(
    resampling_function: ResamplingFunction,
    transfer_function: TransferFunction,
    threading_policy: ThreadingPolicy,
    store: &ImageStore<'a, u16, CN>,
    into: &mut ImageStoreMut<'a, u16, CN>,
) -> Result<(), PicScaleError>
where
    ImageStore<'a, f32, CN>: ImageStoreScaling<'a, f32, CN>,
{
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

    let mut target_vertical = try_vec![f32::default(); store.width * store.height * CN];

    let mut linear_store =
        ImageStoreMut::<f32, CN>::from_slice(&mut target_vertical, store.width, store.height)?;

    let linearization = make_linearization16(transfer_function, into.bit_depth)?;

    for (&src, dst) in store
        .as_bytes()
        .iter()
        .zip(linear_store.buffer.borrow_mut())
    {
        *dst = linearization.linearization[src as usize];
    }

    let new_immutable_store = ImageStore::<f32, CN> {
        buffer: std::borrow::Cow::Owned(target_vertical),
        channels: CN,
        width: store.width,
        height: store.height,
        stride: store.width * CN,
        bit_depth: 16,
    };

    let mut new_store = ImageStoreMut::<f32, CN>::try_alloc(into.width, into.height)?;

    new_immutable_store.scale(
        &mut new_store,
        ScalingOptions {
            resampling_function,
            threading_policy,
            ..Default::default()
        },
    )?;

    for (&src, dst) in new_store.as_bytes().iter().zip(into.buffer.borrow_mut()) {
        let v = ((src * 262143.).round().max(0.) as u32).min(262143);
        *dst = linearization.gamma[v as usize];
    }

    Ok(())
}

impl Scaling for LinearScaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.scaler.threading_policy = threading_policy;
    }

    fn resize_plane<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 1>,
        into: &mut ImageStoreMut<'a, u8, 1>,
    ) -> Result<(), PicScaleError> {
        resize_typical8(
            self.scaler.function,
            self.transfer_function,
            self.scaler.threading_policy,
            store,
            into,
        )
    }

    fn resize_cbcr8<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 2>,
        into: &mut ImageStoreMut<'a, u8, 2>,
    ) -> Result<(), PicScaleError> {
        resize_typical8(
            self.scaler.function,
            self.transfer_function,
            self.scaler.threading_policy,
            store,
            into,
        )
    }

    fn resize_gray_alpha<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 2>,
        into: &mut ImageStoreMut<'a, u8, 2>,
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

        const CN: usize = 2;

        let mut target_vertical = try_vec![f32::default(); store.width * store.height * CN];

        let mut linear_store =
            ImageStoreMut::<f32, CN>::from_slice(&mut target_vertical, store.width, store.height)?;

        let linearization = make_linearization(self.transfer_function);

        for (src, dst) in store
            .as_bytes()
            .chunks_exact(2)
            .zip(linear_store.buffer.borrow_mut().chunks_exact_mut(2))
        {
            dst[0] = linearization.linearization[src[0] as usize];
            dst[1] = src[1] as f32 / 255.;
        }

        let new_immutable_store = ImageStore::<f32, CN> {
            buffer: std::borrow::Cow::Owned(target_vertical),
            channels: CN,
            width: store.width,
            height: store.height,
            stride: store.width * CN,
            bit_depth: 12,
        };

        let mut new_store = ImageStoreMut::<f32, CN>::try_alloc(into.width, into.height)?;

        self.scaler.resize_gray_alpha_f32(
            &new_immutable_store,
            &mut new_store,
            premultiply_alpha,
        )?;

        for (src, dst) in new_store
            .as_bytes()
            .chunks_exact(2)
            .zip(into.buffer.borrow_mut().chunks_exact_mut(2))
        {
            let v0 = (src[0] * 65535.).round().min(65535.).max(0.) as u16;

            dst[0] = linearization.gamma[v0 as usize];
            dst[1] = (src[1] * 255.).round().min(255.).max(0.) as u8;
        }

        Ok(())
    }

    fn resize_rgb<'a>(
        &'a self,
        store: &ImageStore<'a, u8, 3>,
        into: &mut ImageStoreMut<'a, u8, 3>,
    ) -> Result<(), PicScaleError> {
        resize_typical8(
            self.scaler.function,
            self.transfer_function,
            self.scaler.threading_policy,
            store,
            into,
        )
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

        let mut target_vertical = try_vec![f32::default(); store.width * store.height * CN];

        let mut linear_store =
            ImageStoreMut::<f32, CN>::from_slice(&mut target_vertical, store.width, store.height)?;

        let linearization = make_linearization(self.transfer_function);

        for (src, dst) in store
            .as_bytes()
            .chunks_exact(4)
            .zip(linear_store.buffer.borrow_mut().chunks_exact_mut(4))
        {
            dst[0] = linearization.linearization[src[0] as usize];
            dst[1] = linearization.linearization[src[1] as usize];
            dst[2] = linearization.linearization[src[2] as usize];
            dst[3] = src[3] as f32 / 255.;
        }

        let new_immutable_store = ImageStore::<f32, CN> {
            buffer: std::borrow::Cow::Owned(target_vertical),
            channels: CN,
            width: store.width,
            height: store.height,
            stride: store.width * CN,
            bit_depth: 12,
        };

        let mut new_store = ImageStoreMut::<f32, CN>::try_alloc(into.width, into.height)?;

        self.scaler
            .resize_rgba_f32(&new_immutable_store, &mut new_store, premultiply_alpha)?;

        for (src, dst) in new_store
            .as_bytes()
            .chunks_exact(4)
            .zip(into.buffer.borrow_mut().chunks_exact_mut(4))
        {
            let v0 = (src[0] * 65535.).round().min(65535.).max(0.) as u16;
            let v1 = (src[1] * 65535.).round().min(65535.).max(0.) as u16;
            let v2 = (src[2] * 65535.).round().min(65535.).max(0.) as u16;

            dst[0] = linearization.gamma[v0 as usize];
            dst[1] = linearization.gamma[v1 as usize];
            dst[2] = linearization.gamma[v2 as usize];
            dst[3] = (src[3] * 255.).round().min(255.).max(0.) as u8;
        }

        Ok(())
    }
}

impl ScalingU16 for LinearScaler {
    fn resize_plane_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 1>,
        into: &mut ImageStoreMut<'a, u16, 1>,
    ) -> Result<(), PicScaleError> {
        resize_typical16(
            self.scaler.function,
            self.transfer_function,
            self.scaler.threading_policy,
            store,
            into,
        )
    }

    fn resize_cbcr_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 2>,
        into: &mut ImageStoreMut<'a, u16, 2>,
    ) -> Result<(), PicScaleError> {
        resize_typical16(
            self.scaler.function,
            self.transfer_function,
            self.scaler.threading_policy,
            store,
            into,
        )
    }

    fn resize_gray_alpha16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 2>,
        into: &mut ImageStoreMut<'a, u16, 2>,
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

        const CN: usize = 2;

        let mut target_vertical = try_vec![f32::default(); store.width * store.height * CN];

        let mut linear_store =
            ImageStoreMut::<f32, CN>::from_slice(&mut target_vertical, store.width, store.height)?;

        let linearization = make_linearization16(self.transfer_function, into.bit_depth)?;

        let max_bit_depth_value = ((1u32 << into.bit_depth) - 1) as f32;

        let v_recip = 1. / max_bit_depth_value;

        for (src, dst) in store
            .as_bytes()
            .chunks_exact(2)
            .zip(linear_store.buffer.borrow_mut().chunks_exact_mut(2))
        {
            dst[0] = linearization.linearization[src[0] as usize];
            dst[1] = src[1] as f32 * v_recip;
        }

        let new_immutable_store = ImageStore::<f32, CN> {
            buffer: std::borrow::Cow::Owned(target_vertical),
            channels: CN,
            width: store.width,
            height: store.height,
            stride: store.width * CN,
            bit_depth: 16,
        };

        let mut new_store = ImageStoreMut::<f32, CN>::try_alloc(into.width, into.height)?;

        self.scaler.resize_gray_alpha_f32(
            &new_immutable_store,
            &mut new_store,
            premultiply_alpha,
        )?;

        for (src, dst) in new_store
            .as_bytes()
            .chunks_exact(2)
            .zip(into.buffer.borrow_mut().chunks_exact_mut(2))
        {
            let v0 = ((src[0] * 262143.).round().max(0.) as u32).min(262143);

            dst[0] = linearization.gamma[v0 as usize];
            dst[1] = (src[1] * max_bit_depth_value)
                .round()
                .min(max_bit_depth_value)
                .max(0.) as u16;
        }

        Ok(())
    }

    fn resize_rgb_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 3>,
        into: &mut ImageStoreMut<'a, u16, 3>,
    ) -> Result<(), PicScaleError> {
        resize_typical16(
            self.scaler.function,
            self.transfer_function,
            self.scaler.threading_policy,
            store,
            into,
        )
    }

    fn resize_rgba_u16<'a>(
        &'a self,
        store: &ImageStore<'a, u16, 4>,
        into: &mut ImageStoreMut<'a, u16, 4>,
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

        let mut target_vertical = try_vec![f32::default(); store.width * store.height * CN];

        let mut linear_store =
            ImageStoreMut::<f32, CN>::from_slice(&mut target_vertical, store.width, store.height)?;

        let linearization = make_linearization16(self.transfer_function, into.bit_depth)?;

        let max_bit_depth_value = ((1u32 << into.bit_depth) - 1) as f32;

        let v_recip = 1. / max_bit_depth_value;

        for (src, dst) in store
            .as_bytes()
            .chunks_exact(4)
            .zip(linear_store.buffer.borrow_mut().chunks_exact_mut(4))
        {
            dst[0] = linearization.linearization[src[0] as usize];
            dst[1] = linearization.linearization[src[1] as usize];
            dst[2] = linearization.linearization[src[2] as usize];
            dst[3] = src[3] as f32 * v_recip;
        }

        let new_immutable_store = ImageStore::<f32, CN> {
            buffer: std::borrow::Cow::Owned(target_vertical),
            channels: CN,
            width: store.width,
            height: store.height,
            stride: store.width * CN,
            bit_depth: 16,
        };

        let mut new_store = ImageStoreMut::<f32, CN>::try_alloc(into.width, into.height)?;

        self.scaler
            .resize_rgba_f32(&new_immutable_store, &mut new_store, premultiply_alpha)?;

        for (src, dst) in new_store
            .as_bytes()
            .chunks_exact(4)
            .zip(into.buffer.borrow_mut().chunks_exact_mut(4))
        {
            let v0 = ((src[0] * 262143.).round().max(0.) as u32).min(262143);
            let v1 = ((src[1] * 262143.).round().max(0.) as u32).min(262143);
            let v2 = ((src[2] * 262143.).round().max(0.) as u32).min(262143);

            dst[0] = linearization.gamma[v0 as usize];
            dst[1] = linearization.gamma[v1 as usize];
            dst[2] = linearization.gamma[v2 as usize];
            dst[3] = (src[3] * max_bit_depth_value)
                .round()
                .min(max_bit_depth_value)
                .max(0.) as u16;
        }

        Ok(())
    }
}
