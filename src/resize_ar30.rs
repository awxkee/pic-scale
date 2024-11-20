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
use crate::dispatch_group_ar30::{
    convolve_horizontal_dispatch_ar30, convolve_vertical_dispatch_ar30,
};
use crate::nearest_sampler::resize_nearest;
use crate::pic_scale_error::PicScaleError;
use crate::support::check_image_size_overflow;
use crate::{ImageSize, ResamplingFunction, Scaler};

pub(crate) fn resize_ar30_impl<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    src: &[u32],
    src_size: ImageSize,
    dst: &mut [u32],
    dst_size: ImageSize,
    scaler: &Scaler,
) -> Result<(), PicScaleError> {
    if src_size.width == 0 || src_size.height == 0 || dst_size.width == 0 || dst_size.height == 0 {
        return Err(PicScaleError::ZeroImageDimensions);
    }

    if check_image_size_overflow(src_size.width, src_size.height, 1) {
        return Err(PicScaleError::SourceImageIsTooLarge);
    }

    if check_image_size_overflow(dst_size.width, dst_size.height, 1) {
        return Err(PicScaleError::DestinationImageIsTooLarge);
    }

    if src_size.width == dst_size.width && src_size.height == dst_size.height {
        for (src, dst) in src.iter().zip(dst.iter_mut()) {
            *dst = *src;
        }
        return Ok(());
    }

    let pool = scaler
        .threading_policy
        .get_pool(ImageSize::new(dst_size.width, dst_size.height));

    if scaler.function == ResamplingFunction::Nearest {
        resize_nearest::<u32, 1>(
            src,
            src_size.width,
            src_size.height,
            dst,
            dst_size.width,
            dst_size.height,
            &pool,
        );
        return Ok(());
    }

    let should_do_horizontal = src_size.width != dst_size.width;
    let should_do_vertical = src_size.height != dst_size.height;
    assert!(should_do_horizontal || should_do_vertical);

    let working_store = if should_do_vertical {
        let mut target = vec![0u32; src_size.width * dst_size.height];

        let vertical_filters = scaler.generate_weights(src_size.height, dst_size.height);
        convolve_vertical_dispatch_ar30::<AR30_TYPE, AR30_ORDER>(
            src,
            src_size.width,
            vertical_filters,
            &mut target,
            src_size.width,
            &pool,
        );

        std::borrow::Cow::Owned(target)
    } else {
        std::borrow::Cow::Borrowed(src)
    };

    if should_do_horizontal {
        let horizontal_filters = scaler.generate_weights(src_size.width, dst_size.width);
        convolve_horizontal_dispatch_ar30::<AR30_TYPE, AR30_ORDER>(
            working_store.as_ref(),
            src_size.width,
            horizontal_filters,
            dst,
            dst_size.width,
            &pool,
        );
    }

    Ok(())
}
