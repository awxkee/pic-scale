/*
 * Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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
use crate::{ImageStore, ImageStoreMut};
use std::fmt::Debug;

impl<F: Clone + Debug + Copy, const CN: usize> ImageStore<'_, F, CN>
where
    [F]: ToOwned<Owned = Vec<F>>,
{
    pub(crate) fn to_colorutils_buffer(&self) -> colorutils_rs::ImageBuffer<'_, F> {
        colorutils_rs::ImageBuffer {
            data: std::borrow::Cow::Borrowed(self.as_bytes()),
            width: self.width as u32,
            height: self.height as u32,
            stride: self.stride as u32,
            channels: self.channels as u32,
        }
    }
}

impl<F: Copy + Debug, const CN: usize> ImageStoreMut<'_, F, CN>
where
    [F]: ToOwned<Owned = Vec<F>>,
{
    pub(crate) fn to_colorutils_buffer_mut(&mut self) -> colorutils_rs::ImageBufferMut<'_, F> {
        let dst_width = self.width;
        let dst_height = self.height;
        let dst_stride = self.stride;
        let dst_channels = self.channels;
        colorutils_rs::ImageBufferMut {
            data: colorutils_rs::BufferStore::Borrowed(self.buffer.borrow_mut()),
            width: dst_width as u32,
            height: dst_height as u32,
            stride: dst_stride as u32,
            channels: dst_channels as u32,
        }
    }
}
