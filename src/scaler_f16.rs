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

use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::nearest_sampler::resize_nearest;
use crate::support::check_image_size_overflow;
use crate::ResamplingFunction::Nearest;
use crate::{ImageSize, ImageStore, Scaler};
use half::f16;

// f16
impl Scaler {
    /// Resize f16 RGBA image
    pub fn resize_rgba_f16<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, f16, 4>,
        premultiply_alpha: bool,
    ) -> Result<ImageStore<'a, f16, 4>, String> {
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err("One of image dimensions is 0, this should not happen".to_string());
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err("Input image larger than memory capabilities".to_string());
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err("Destination image larger than memory capabilities".to_string());
        }

        if store.width == new_size.width && store.height == new_size.height {
            return Ok(store.copied());
        }

        let mut src_store = store;

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
            let mut allocated_store: Vec<f16> =
                vec![f16::from_f32(0.); new_size.width * 4 * new_size.height];
            resize_nearest::<f16, 4>(
                &src_store.buffer.borrow(),
                src_store.width,
                src_store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                &pool,
            );
            let new_image =
                ImageStore::<f16, 4>::new(allocated_store, new_size.width, new_size.height)?;

            return Ok(new_image);
        }

        let should_do_horizontal = src_store.width != new_size.width;
        let should_do_vertical = src_store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        if premultiply_alpha {
            let mut premultiplied_store =
                ImageStore::<f16, 4>::alloc(src_store.width, src_store.height);
            src_store.premultiply_alpha(&mut premultiplied_store, &pool);
            src_store = premultiplied_store;
        }

        if should_do_vertical {
            let allocated_store_vertical: Vec<f16> =
                vec![f16::from_f32(0.); src_store.width * 4 * new_size.height];
            let mut new_image_vertical = ImageStore::<f16, 4>::new(
                allocated_store_vertical,
                src_store.width,
                new_size.height,
            )?;
            let vertical_filters =
                self.generate_weights(src_store.height, new_image_vertical.height);
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
            src_store = new_image_vertical;
        }

        assert_eq!(src_store.height, new_size.height);

        if should_do_horizontal {
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            let allocated_store_horizontal: Vec<f16> =
                vec![f16::from_f32(0.); new_size.width * 4 * new_size.height];
            let mut new_image_horizontal = ImageStore::<f16, 4>::new(
                allocated_store_horizontal,
                new_size.width,
                new_size.height,
            )?;
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }

        assert_eq!(src_store.width, new_size.width);

        if premultiply_alpha {
            let mut premultiplied_store =
                ImageStore::<f16, 4>::alloc(src_store.width, src_store.height);
            src_store.unpremultiply_alpha(&mut premultiplied_store, &pool);
            return Ok(premultiplied_store);
        }

        Ok(src_store)
    }

    /// Resize f16 RGB image
    pub fn resize_rgb_f16<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, f16, 3>,
    ) -> Result<ImageStore<'a, f16, 3>, String> {
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err("One of image dimensions is 0, this should not happen".to_string());
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err("Input image larger than memory capabilities".to_string());
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err("Destination image larger than memory capabilities".to_string());
        }

        if store.width == new_size.width && store.height == new_size.height {
            return Ok(store.copied());
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
            let mut allocated_store: Vec<f16> =
                vec![f16::from_f32(0.); new_size.width * 3 * new_size.height];
            resize_nearest::<f16, 3>(
                &store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                &pool,
            );
            let new_image =
                ImageStore::<f16, 3>::new(allocated_store, new_size.width, new_size.height)?;
            return Ok(new_image);
        }

        let mut src_store = store;

        let should_do_horizontal = src_store.width != new_size.width;
        let should_do_vertical = src_store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        if should_do_vertical {
            let allocated_store_vertical: Vec<f16> =
                vec![f16::from_f32(0.); src_store.width * 3 * new_size.height];
            let mut new_image_vertical = ImageStore::<f16, 3>::new(
                allocated_store_vertical,
                src_store.width,
                new_size.height,
            )?;
            let vertical_filters =
                self.generate_weights(src_store.height, new_image_vertical.height);
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
            src_store = new_image_vertical;
        }

        if should_do_horizontal {
            let allocated_store_horizontal: Vec<f16> =
                vec![f16::from_f32(0.); new_size.width * 3 * new_size.height];
            let mut new_image_horizontal = ImageStore::<f16, 3>::new(
                allocated_store_horizontal,
                new_size.width,
                new_size.height,
            )?;
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }

        Ok(src_store)
    }

    /// Resize f16 plane
    pub fn resize_plane_f16<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, f16, 1>,
    ) -> Result<ImageStore<'a, f16, 1>, String> {
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err("One of image dimensions is 0, this should not happen".to_string());
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err("Input image larger than memory capabilities".to_string());
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err("Destination image larger than memory capabilities".to_string());
        }

        if store.width == new_size.width && store.height == new_size.height {
            return Ok(store.copied());
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
            let mut allocated_store: Vec<f16> =
                vec![f16::from_f32(0.); new_size.width * new_size.height];
            resize_nearest::<f16, 1>(
                &store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                &pool,
            );
            let new_image =
                ImageStore::<f16, 1>::new(allocated_store, new_size.width, new_size.height)?;
            return Ok(new_image);
        }

        let mut src_store = store;

        let should_do_horizontal = src_store.width != new_size.width;
        let should_do_vertical = src_store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        if should_do_vertical {
            let allocated_store_vertical: Vec<f16> =
                vec![f16::from_f32(0.); src_store.width * new_size.height];
            let mut new_image_vertical = ImageStore::<f16, 1>::new(
                allocated_store_vertical,
                src_store.width,
                new_size.height,
            )?;
            let vertical_filters =
                self.generate_weights(src_store.height, new_image_vertical.height);
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
            src_store = new_image_vertical;
        }

        if should_do_horizontal {
            let allocated_store_horizontal: Vec<f16> =
                vec![f16::from_f32(0.); new_size.width * new_size.height];
            let mut new_image_horizontal = ImageStore::<f16, 1>::new(
                allocated_store_horizontal,
                new_size.width,
                new_size.height,
            )?;
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }

        Ok(src_store)
    }
}
