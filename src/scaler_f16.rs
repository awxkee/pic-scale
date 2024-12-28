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

use crate::image_store::ImageStoreMut;
use crate::pic_scale_error::PicScaleError;
use crate::{ImageStore, Scaler};
use half::f16;

// f16
impl Scaler {
    /// Resize f16 RGBA image
    pub fn resize_rgba_f16<'a>(
        &'a self,
        store: &ImageStore<'a, f16, 4>,
        into: &mut ImageStoreMut<'a, f16, 4>,
        premultiply_alpha: bool,
    ) -> Result<(), PicScaleError> {
        self.generic_resize_with_alpha(store, into, premultiply_alpha)
    }

    /// Resize f16 RGB image
    pub fn resize_rgb_f16<'a>(
        &'a self,
        store: &ImageStore<'a, f16, 3>,
        into: &mut ImageStoreMut<'a, f16, 3>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }

    /// Resize f16 plane
    pub fn resize_plane_f16<'a>(
        &'a self,
        store: &ImageStore<'a, f16, 1>,
        into: &mut ImageStoreMut<'a, f16, 1>,
    ) -> Result<(), PicScaleError> {
        self.generic_resize(store, into)
    }
}
