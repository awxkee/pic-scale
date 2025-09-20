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
use std::error::Error;
use std::fmt::Display;

/// Buffer mismatch error description
#[derive(Copy, Clone, Debug)]
pub struct PicScaleBufferMismatch {
    pub expected: usize,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub slice_len: usize,
}

/// Error enumeration type
#[derive(Debug)]
pub enum PicScaleError {
    ZeroImageDimensions,
    SourceImageIsTooLarge,
    DestinationImageIsTooLarge,
    BufferMismatch(PicScaleBufferMismatch),
    InvalidStride(usize, usize),
    UnsupportedBitDepth(usize),
    UnknownResizingFilter,
    OutOfMemory(usize),
}

impl PicScaleError {
    /// Returns error as int code
    #[inline]
    pub fn code(&self) -> usize {
        match self {
            PicScaleError::ZeroImageDimensions => 1,
            PicScaleError::SourceImageIsTooLarge => 2,
            PicScaleError::DestinationImageIsTooLarge => 3,
            PicScaleError::BufferMismatch(_) => 4,
            PicScaleError::InvalidStride(_, _) => 5,
            PicScaleError::UnsupportedBitDepth(_) => 6,
            PicScaleError::UnknownResizingFilter => 7,
            PicScaleError::OutOfMemory(_) => 8,
        }
    }
}

impl Display for PicScaleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PicScaleError::InvalidStride(min_stride, real_stride) => f.write_fmt(format_args!(
                "Stride must be at least {min_stride}, but received {real_stride}",
            )),
            PicScaleError::ZeroImageDimensions => {
                f.write_str("One of image dimensions is 0, this should not happen")
            }
            PicScaleError::SourceImageIsTooLarge => {
                f.write_str("Input image larger than memory capabilities")
            }
            PicScaleError::DestinationImageIsTooLarge => {
                f.write_str("Destination image larger than memory capabilities")
            }
            PicScaleError::BufferMismatch(buffer_mismatch) => f.write_fmt(format_args!(
                "Image buffer len expected to be {} [w({})*h({})*channels({})] but received {}",
                buffer_mismatch.expected,
                buffer_mismatch.width,
                buffer_mismatch.height,
                buffer_mismatch.channels,
                buffer_mismatch.slice_len,
            )),
            PicScaleError::UnsupportedBitDepth(depth) => {
                f.write_fmt(format_args!("Bit-depth must be in [1, 16] but got {depth}",))
            }
            PicScaleError::UnknownResizingFilter => {
                f.write_str("Unknown resizing filter was requested")
            }
            PicScaleError::OutOfMemory(capacity) => f.write_fmt(format_args!(
                "There is no enough memory to allocate {capacity} bytes"
            )),
        }
    }
}

impl Error for PicScaleError {}

macro_rules! try_vec {
    () => {
        Vec::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut v = Vec::new();
        v.try_reserve_exact($n)
            .map_err(|_| crate::pic_scale_error::PicScaleError::OutOfMemory($n))?;
        v.resize($n, $elem);
        v
    }};
}

pub(crate) use try_vec;
