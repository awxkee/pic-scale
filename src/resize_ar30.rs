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
use crate::convolution::ConvolutionOptions;
use crate::dispatch_group_ar30::{
    convolve_horizontal_dispatch_ar30, convolve_vertical_dispatch_ar30,
};
use crate::nearest_sampler::resize_nearest;
use crate::pic_scale_error::PicScaleError;
use crate::support::check_image_size_overflow;
use crate::{ImageSize, PicScaleBufferMismatch, ResamplingFunction, Scaler};

pub(crate) fn resize_ar30_impl<const AR30_TYPE: usize, const AR30_ORDER: usize>(
    src: &[u8],
    src_stride: usize,
    src_size: ImageSize,
    dst: &mut [u8],
    dst_stride: usize,
    dst_size: ImageSize,
    scaler: &Scaler,
) -> Result<(), PicScaleError> {
    if src_size.width == 0 || src_size.height == 0 || dst_size.width == 0 || dst_size.height == 0 {
        return Err(PicScaleError::ZeroImageDimensions);
    }

    if check_image_size_overflow(src_size.width, src_size.height, 4) {
        return Err(PicScaleError::SourceImageIsTooLarge);
    }

    if check_image_size_overflow(dst_size.width, dst_size.height, 4) {
        return Err(PicScaleError::DestinationImageIsTooLarge);
    }

    if src.len() != src_stride * src_size.height {
        return Err(PicScaleError::BufferMismatch(PicScaleBufferMismatch {
            expected: src_stride * src_size.height,
            width: src_size.width,
            height: src_size.height,
            channels: 4,
            slice_len: src.len(),
        }));
    }
    if src_stride < src_size.width * 4 {
        return Err(PicScaleError::InvalidStride(src_size.width * 4, src_stride));
    }

    if dst.len() != dst_stride * dst_size.height {
        return Err(PicScaleError::BufferMismatch(PicScaleBufferMismatch {
            expected: dst_stride * dst_size.height,
            width: dst_size.width,
            height: dst_size.height,
            channels: 4,
            slice_len: dst.len(),
        }));
    }
    if dst_stride < dst_size.width * 4 {
        return Err(PicScaleError::InvalidStride(dst_size.width * 4, dst_stride));
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
        resize_nearest::<u8, 4>(
            src,
            src_stride,
            src_size.height,
            dst,
            dst_stride,
            dst_size.height,
            &pool,
        );
        return Ok(());
    }

    let should_do_horizontal = src_size.width != dst_size.width;
    let should_do_vertical = src_size.height != dst_size.height;
    assert!(should_do_horizontal || should_do_vertical);

    let options = ConvolutionOptions::new(scaler.workload_strategy);

    if should_do_vertical && !should_do_horizontal {
        let vertical_filters = scaler.generate_weights(src_size.height, dst_size.height);
        convolve_vertical_dispatch_ar30::<AR30_TYPE, AR30_ORDER>(
            src,
            src_stride,
            vertical_filters,
            dst,
            src_stride,
            &pool,
            src_size.width,
            options,
        );
        return Ok(());
    } else if should_do_horizontal && should_do_vertical {
        let mut target = vec![0u8; src_size.width * dst_size.height * 4];

        let vertical_filters = scaler.generate_weights(src_size.height, dst_size.height);
        convolve_vertical_dispatch_ar30::<AR30_TYPE, AR30_ORDER>(
            src,
            src_stride,
            vertical_filters,
            &mut target,
            src_size.width * 4,
            &pool,
            src_size.width,
            options,
        );

        let horizontal_filters = scaler.generate_weights(src_size.width, dst_size.width);
        convolve_horizontal_dispatch_ar30::<AR30_TYPE, AR30_ORDER>(
            &target,
            src_size.width * 4,
            horizontal_filters,
            dst,
            dst_stride,
            &pool,
            options,
        );
    } else {
        let horizontal_filters = scaler.generate_weights(src_size.width, dst_size.width);
        convolve_horizontal_dispatch_ar30::<AR30_TYPE, AR30_ORDER>(
            src,
            src_stride,
            horizontal_filters,
            dst,
            dst_stride,
            &pool,
            options,
        );
    }

    Ok(())
}
