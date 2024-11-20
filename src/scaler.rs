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
use crate::alpha_check::{
    has_non_constant_cap_alpha_rgba16, has_non_constant_cap_alpha_rgba8,
    has_non_constant_cap_alpha_rgba_f32,
};
use crate::ar30::{Ar30ByteOrder, Rgb30};
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_size::ImageSize;
use crate::image_store::ImageStore;
use crate::nearest_sampler::resize_nearest;
use crate::pic_scale_error::PicScaleError;
use crate::resize_ar30::resize_ar30_impl;
use crate::support::check_image_size_overflow;
use crate::threading_policy::ThreadingPolicy;
use crate::{ConstPI, ConstSqrt2, Jinc, ResamplingFunction};
use num_traits::{AsPrimitive, Float, Signed};
use rayon::ThreadPool;
use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign, Neg};

#[derive(Debug, Copy, Clone)]
/// Represents base scaling structure
pub struct Scaler {
    pub(crate) function: ResamplingFunction,
    pub(crate) threading_policy: ThreadingPolicy,
}

pub trait Scaling {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy);

    /// Performs rescaling for RGB, channel order does not matter
    fn resize_rgb<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, u8, 3>,
    ) -> Result<ImageStore<'a, u8, 3>, PicScaleError>;

    /// Performs rescaling for RGBA, for pre-multiplying alpha, converting to LUV or LAB alpha must be last channel
    fn resize_rgba<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, u8, 4>,
        premultiply_alpha: bool,
    ) -> Result<ImageStore<'a, u8, 4>, PicScaleError>;
}

pub trait ScalingF32 {
    /// Performs rescaling for RGB f32, channel order does not matter
    fn resize_rgb_f32<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, f32, 3>,
    ) -> Result<ImageStore<'a, f32, 3>, PicScaleError>;

    /// Performs rescaling for RGBA f32, alpha expected to be last
    fn resize_rgba_f32<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, f32, 4>,
        premultiply_alpha: bool,
    ) -> Result<ImageStore<'a, f32, 4>, PicScaleError>;
}

pub trait ScalingU16 {
    /// Performs rescaling for Planar u16, channel order does not matter
    ///
    /// # Arguments
    /// `new_size` - New image size
    /// `store` - original image store
    /// `bit_depth` - image bit-depth, this is required for u16 image
    ///
    /// # Panics
    /// Panic if bit-depth < 1 or bit-depth > 16
    fn resize_plane_u16<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, u16, 1>,
        bit_depth: usize,
    ) -> Result<ImageStore<'a, u16, 1>, PicScaleError>;

    /// Performs rescaling for RGB, channel order does not matter
    ///
    /// # Arguments
    /// `new_size` - New image size
    /// `store` - original image store
    /// `bit_depth` - image bit-depth, this is required for u16 image
    ///
    /// # Panics
    /// Panic if bit-depth < 1 or bit-depth > 16
    fn resize_rgb_u16<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, u16, 3>,
        bit_depth: usize,
    ) -> Result<ImageStore<'a, u16, 3>, PicScaleError>;

    /// Performs rescaling for RGBA, for pre-multiplying alpha should be last
    ///
    /// # Arguments
    /// `new_size` - New image size
    /// `store` - original image store
    /// `bit_depth` - image bit-depth, this is required for u16 image
    /// `premultiply_alpha` - flags is alpha is premultiplied
    ///
    /// # Panics
    /// Panic if bit-depth < 1 or bit-depth > 16
    fn resize_rgba_u16<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, u16, 4>,
        bit_depth: usize,
        premultiply_alpha: bool,
    ) -> Result<ImageStore<'a, u16, 4>, PicScaleError>;
}

impl Scaler {
    /// Creates new Scaler instance with corresponding filter
    pub fn new(filter: ResamplingFunction) -> Self {
        Scaler {
            function: filter,
            threading_policy: ThreadingPolicy::Single,
        }
    }

    pub(crate) fn generate_weights<T>(&self, in_size: usize, out_size: usize) -> FilterWeights<T>
    where
        T: Copy
            + Neg
            + Signed
            + Float
            + 'static
            + ConstPI
            + MulAssign<T>
            + AddAssign<T>
            + AsPrimitive<f64>
            + AsPrimitive<usize>
            + Jinc<T>
            + ConstSqrt2
            + Default
            + AsPrimitive<i32>,
        f32: AsPrimitive<T>,
        f64: AsPrimitive<T>,
        i64: AsPrimitive<T>,
        i32: AsPrimitive<T>,
        usize: AsPrimitive<T>,
    {
        let resampling_filter = self.function.get_resampling_filter();
        let scale = in_size.as_() / out_size.as_();
        let is_resizable_kernel = resampling_filter.is_resizable_kernel;
        let filter_scale_cutoff = match is_resizable_kernel {
            true => scale.max(1f32.as_()),
            false => 1f32.as_(),
        };
        let filter_base_size = resampling_filter.min_kernel_size;
        let resampling_function = resampling_filter.kernel;

        let is_area = resampling_filter.is_area && scale < 1.as_();

        let mut bounds: Vec<FilterBounds> = vec![FilterBounds::new(0, 0); out_size];

        if !is_area {
            let window_func = resampling_filter.window;
            let base_size: usize = (filter_base_size.as_() * filter_scale_cutoff).round().as_();
            let kernel_size = base_size;
            let filter_radius = base_size.as_() / 2.as_();
            let filter_scale = 1f32.as_() / filter_scale_cutoff;
            let mut weights: Vec<T> = vec![T::default(); kernel_size * out_size];
            let mut local_filters = vec![T::default(); kernel_size];
            let mut filter_position = 0usize;
            let blur_scale = match window_func {
                None => 1f32.as_(),
                Some(window) => {
                    if window.blur.as_() > 0f32.as_() {
                        1f32.as_() / window.blur.as_()
                    } else {
                        0f32.as_()
                    }
                }
            };
            for (i, bound) in bounds.iter_mut().enumerate() {
                let center_x = ((i.as_() + 0.5.as_()) * scale).min(in_size.as_());
                let mut weights_sum: T = 0f32.as_();
                let mut local_filter_iteration = 0usize;

                let start: usize = (center_x - filter_radius).floor().max(0f32.as_()).as_();
                let end: usize = (center_x + filter_radius)
                    .ceil()
                    .min(in_size.as_())
                    .min(start.as_() + kernel_size.as_())
                    .as_();

                let center = center_x - 0.5.as_();

                for k in start..end {
                    let dx = k.as_() - center;
                    let weight;
                    if let Some(resampling_window) = window_func {
                        let mut x = dx.abs();
                        x = if resampling_window.blur.as_() > 0f32.as_() {
                            x * blur_scale
                        } else {
                            x
                        };
                        x = if x <= resampling_window.taper.as_() {
                            0f32.as_()
                        } else {
                            (x - resampling_window.taper.as_())
                                / (1f32.as_() - resampling_window.taper.as_())
                        };
                        let window_producer = resampling_window.window;
                        let x_kernel_scaled = x * filter_scale;
                        let window = if x < resampling_window.window_size.as_() {
                            window_producer(x_kernel_scaled * resampling_window.window_size.as_())
                        } else {
                            0f32.as_()
                        };
                        weight = window * resampling_function(x_kernel_scaled);
                    } else {
                        let dx = dx.abs();
                        weight = resampling_function(dx * filter_scale);
                    }
                    weights_sum += weight;
                    unsafe {
                        *local_filters.get_unchecked_mut(local_filter_iteration) = weight;
                    }
                    local_filter_iteration += 1;
                }

                let alpha: T = 0.7f32.as_();
                if resampling_filter.is_ewa && !local_filters.is_empty() {
                    weights_sum = unsafe { *local_filters.get_unchecked(0) };
                    for j in 1..local_filter_iteration {
                        let new_weight = alpha * unsafe { *local_filters.get_unchecked(j) }
                            + (1f32.as_() - alpha) * unsafe { *local_filters.get_unchecked(j - 1) };
                        unsafe {
                            *local_filters.get_unchecked_mut(j) = new_weight;
                        }
                        weights_sum += new_weight;
                    }
                }

                let size = end - start;

                *bound = FilterBounds::new(start, size);

                if weights_sum != 0f32.as_() {
                    let recpeq = 1f32.as_() / weights_sum;

                    for (dst, src) in weights
                        .iter_mut()
                        .skip(filter_position)
                        .take(size)
                        .zip(local_filters.iter().take(size))
                    {
                        *dst = *src * recpeq;
                    }
                }

                filter_position += kernel_size;
            }

            FilterWeights::<T>::new(
                weights,
                kernel_size,
                kernel_size,
                out_size,
                filter_radius.as_(),
                bounds,
            )
        } else {
            // Simulating INTER_AREA from OpenCV, for up scaling here,
            // this is necessary because weight computation is different
            // from any other func
            let inv_scale: T = 1.as_() / scale;
            let kernel_size = 2;
            let filter_radius: T = 1.as_();
            let mut weights: Vec<T> = vec![T::default(); kernel_size * out_size];
            let mut local_filters = vec![T::default(); kernel_size];
            let mut filter_position = 0usize;

            for (i, bound) in bounds.iter_mut().enumerate() {
                let mut weights_sum: T = 0f32.as_();

                let sx: T = (i.as_() * scale).floor();
                let fx = (i as i64 + 1).as_() - (sx + 1.as_()) * inv_scale;
                let dx = if fx <= 0.as_() {
                    0.as_()
                } else {
                    fx - fx.floor()
                };
                let dx = dx.abs();
                let weight0 = 1.as_() - dx;
                let weight1: T = dx;
                local_filters[0] = weight0;
                local_filters[1] = weight1;

                let start: usize = sx.floor().max(0f32.as_()).as_();
                let end: usize = (sx + kernel_size.as_())
                    .ceil()
                    .min(in_size.as_())
                    .min(start.as_() + kernel_size.as_())
                    .as_();

                let size = end - start;

                weights_sum += weight0;
                if size > 1 {
                    weights_sum += weight1;
                }
                *bound = FilterBounds::new(start, size);

                if weights_sum != 0f32.as_() {
                    let recpeq = 1f32.as_() / weights_sum;

                    for (dst, src) in weights
                        .iter_mut()
                        .skip(filter_position)
                        .take(size)
                        .zip(local_filters.iter().take(size))
                    {
                        *dst = *src * recpeq;
                    }
                } else {
                    weights[filter_position] = 1.as_();
                }

                filter_position += kernel_size;
            }

            FilterWeights::new(
                weights,
                kernel_size,
                kernel_size,
                out_size,
                filter_radius.as_(),
                bounds,
            )
        }
    }
}

impl Scaler {
    pub(crate) fn resize_rgba_impl<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, u8, 4>,
        premultiply_alpha: bool,
        pool: &Option<ThreadPool>,
    ) -> Result<ImageStore<'a, u8, 4>, PicScaleError> {
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
            return Ok(store.copied());
        }

        let should_do_horizontal = store.width != new_size.width;
        let should_do_vertical = store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        let mut src_store = store;

        if self.function == ResamplingFunction::Nearest {
            let mut new_image = ImageStore::<u8, 4>::alloc(new_size.width, new_size.height);
            resize_nearest::<u8, 4>(
                src_store.buffer.borrow(),
                src_store.width,
                src_store.height,
                new_image.buffer.borrow_mut(),
                new_size.width,
                new_size.height,
                pool,
            );
            return Ok(new_image);
        }

        let mut has_alpha_premultiplied = false;

        if premultiply_alpha {
            let is_alpha_premultiplication_reasonable =
                has_non_constant_cap_alpha_rgba8(src_store.buffer.borrow(), src_store.width);
            if is_alpha_premultiplication_reasonable {
                let mut new_store = ImageStore::<u8, 4>::alloc(src_store.width, src_store.height);
                src_store.premultiply_alpha(&mut new_store, pool);
                src_store = new_store;
                has_alpha_premultiplied = true;
            }
        }

        if should_do_vertical {
            let mut new_image_vertical =
                ImageStore::<u8, 4>::alloc(src_store.width, new_size.height);
            let vertical_filters =
                self.generate_weights(src_store.height, new_image_vertical.height);
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, pool);
            src_store = new_image_vertical;
        }

        assert_eq!(src_store.height, new_size.height);

        if should_do_horizontal {
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            let mut new_image_horizontal =
                ImageStore::<u8, 4>::alloc(new_size.width, new_size.height);
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, pool);
            src_store = new_image_horizontal;
        }

        assert_eq!(src_store.width, new_size.width);

        if premultiply_alpha && has_alpha_premultiplied {
            src_store.unpremultiply_alpha(pool);
        }

        Ok(src_store)
    }
}

impl Scaling for Scaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.threading_policy = threading_policy;
    }

    fn resize_rgb<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, u8, 3>,
    ) -> Result<ImageStore<'a, u8, 3>, PicScaleError> {
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
            return Ok(store.copied());
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == ResamplingFunction::Nearest {
            let mut allocated_store: Vec<u8> = vec![0u8; new_size.width * 3 * new_size.height];
            resize_nearest::<u8, 3>(
                store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                &pool,
            );
            return ImageStore::<u8, 3>::new(allocated_store, new_size.width, new_size.height);
        }

        let should_do_horizontal = store.width != new_size.width;
        let should_do_vertical = store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        let mut src_store = store;

        if should_do_vertical {
            let vertical_filters = self.generate_weights(src_store.height, new_size.height);
            let mut new_image_vertical =
                ImageStore::<u8, 3>::alloc(src_store.width, new_size.height);
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
            src_store = new_image_vertical;
        }

        assert_eq!(src_store.height, new_size.height);

        if should_do_horizontal {
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            let mut new_image_horizontal =
                ImageStore::<u8, 3>::alloc(new_size.width, new_size.height);
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }

        assert_eq!(src_store.width, new_size.width);

        Ok(src_store)
    }

    fn resize_rgba<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, u8, 4>,
        premultiply_alpha: bool,
    ) -> Result<ImageStore<'a, u8, 4>, PicScaleError> {
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));
        self.resize_rgba_impl(new_size, store, premultiply_alpha, &pool)
    }
}

impl Scaler {
    pub(crate) fn resize_rgba_f32_impl<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, f32, 4>,
        premultiply_alpha: bool,
        pool: &Option<ThreadPool>,
    ) -> Result<ImageStore<'a, f32, 4>, PicScaleError> {
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
            return Ok(store.copied());
        }

        let mut src_store = store;

        if self.function == ResamplingFunction::Nearest {
            let mut allocated_store: Vec<f32> = vec![0f32; new_size.width * 4 * new_size.height];
            resize_nearest::<f32, 4>(
                src_store.buffer.borrow(),
                src_store.width,
                src_store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                pool,
            );
            let new_image = ImageStore::new(allocated_store, new_size.width, new_size.height)?;
            return Ok(new_image);
        }

        let should_do_horizontal = src_store.width != new_size.width;
        let should_do_vertical = src_store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        let mut has_alpha_premultiplied = false;

        if premultiply_alpha {
            let is_alpha_premultiplication_reasonable =
                has_non_constant_cap_alpha_rgba_f32(src_store.buffer.borrow(), src_store.width);
            if is_alpha_premultiplication_reasonable {
                let mut new_store = ImageStore::<f32, 4>::alloc(src_store.width, new_size.height);
                src_store.premultiply_alpha(&mut new_store, pool);
                src_store = new_store;
                has_alpha_premultiplied = true;
            }
        }

        if should_do_vertical {
            let allocated_store_vertical: Vec<f32> =
                vec![0f32; src_store.width * 4 * new_size.height];
            let mut new_image_vertical = ImageStore::<f32, 4>::new(
                allocated_store_vertical,
                src_store.width,
                new_size.height,
            )?;
            let vertical_filters =
                self.generate_weights(src_store.height, new_image_vertical.height);
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, pool);
            src_store = new_image_vertical;
        }

        assert_eq!(src_store.height, new_size.height);

        if should_do_horizontal {
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            let allocated_store_horizontal: Vec<f32> =
                vec![0f32; new_size.width * 4 * new_size.height];
            let mut new_image_horizontal = ImageStore::<f32, 4>::new(
                allocated_store_horizontal,
                new_size.width,
                new_size.height,
            )?;
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, pool);
            src_store = new_image_horizontal;
        }

        assert_eq!(src_store.width, new_size.width);

        if premultiply_alpha && has_alpha_premultiplied {
            src_store.unpremultiply_alpha(pool);
        }

        Ok(src_store)
    }
}

impl ScalingF32 for Scaler {
    fn resize_rgb_f32<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, f32, 3>,
    ) -> Result<ImageStore<'a, f32, 3>, PicScaleError> {
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
            return Ok(store.copied());
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == ResamplingFunction::Nearest {
            let mut allocated_store: Vec<f32> = vec![0f32; new_size.width * 3 * new_size.height];
            resize_nearest::<f32, 3>(
                store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                &pool,
            );
            let new_image =
                ImageStore::<f32, 3>::new(allocated_store, new_size.width, new_size.height);
            return new_image;
        }

        let mut src_store = store;

        let should_do_horizontal = src_store.width != new_size.width;
        let should_do_vertical = src_store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        if should_do_vertical {
            let allocated_store_vertical: Vec<f32> =
                vec![0f32; src_store.width * 3 * new_size.height];
            let mut new_image_vertical = ImageStore::<f32, 3>::new(
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
            let allocated_store_horizontal: Vec<f32> =
                vec![0f32; new_size.width * 3 * new_size.height];
            let mut new_image_horizontal = ImageStore::<f32, 3>::new(
                allocated_store_horizontal,
                new_size.width,
                new_size.height,
            )?;
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }

        assert_eq!(src_store.width, new_size.width);

        Ok(src_store)
    }

    fn resize_rgba_f32<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, f32, 4>,
        premultiply_alpha: bool,
    ) -> Result<ImageStore<'a, f32, 4>, PicScaleError> {
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));
        self.resize_rgba_f32_impl(new_size, store, premultiply_alpha, &pool)
    }
}

impl Scaler {
    /// Performs rescaling for f32 plane
    pub fn resize_plane_f32<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, f32, 1>,
    ) -> Result<ImageStore<'a, f32, 1>, PicScaleError> {
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
            return Ok(store.copied());
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == ResamplingFunction::Nearest {
            let mut allocated_store: Vec<f32> = vec![0f32; new_size.width * new_size.height];
            resize_nearest::<f32, 1>(
                store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                &pool,
            );
            let new_image =
                ImageStore::<f32, 1>::new(allocated_store, new_size.width, new_size.height)?;
            return Ok(new_image);
        }

        let mut src_store = store;

        let should_do_horizontal = src_store.width != new_size.width;
        let should_do_vertical = src_store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        if should_do_vertical {
            let allocated_store_vertical: Vec<f32> = vec![0f32; src_store.width * new_size.height];
            let mut new_image_vertical = ImageStore::<f32, 1>::new(
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
            let allocated_store_horizontal: Vec<f32> = vec![0f32; new_size.width * new_size.height];
            let mut new_image_horizontal = ImageStore::<f32, 1>::new(
                allocated_store_horizontal,
                new_size.width,
                new_size.height,
            )?;
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }

        assert_eq!(src_store.width, new_size.width);

        Ok(src_store)
    }
}

impl Scaler {
    /// Performs rescaling for u8 plane
    pub fn resize_plane<'a>(
        &'a self,
        new_size: ImageSize,
        store: ImageStore<'a, u8, 1>,
    ) -> Result<ImageStore<'a, u8, 1>, PicScaleError> {
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
            return Ok(store.copied());
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == ResamplingFunction::Nearest {
            let mut allocated_store: Vec<u8> = vec![0u8; new_size.width * new_size.height];
            resize_nearest::<u8, 1>(
                store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                &pool,
            );
            let new_image =
                ImageStore::<u8, 1>::new(allocated_store, new_size.width, new_size.height)?;
            return Ok(new_image);
        }

        let should_do_horizontal = store.width != new_size.width;
        let should_do_vertical = store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        let mut src_store = store;

        if should_do_vertical {
            let vertical_filters = self.generate_weights(src_store.height, new_size.height);
            let mut new_image_vertical =
                ImageStore::<u8, 1>::alloc(src_store.width, new_size.height);
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
            src_store = new_image_vertical;
        }

        assert_eq!(src_store.height, new_size.height);

        if should_do_horizontal {
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            let mut new_image_horizontal =
                ImageStore::<u8, 1>::alloc(new_size.width, new_size.height);
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }

        assert_eq!(src_store.width, new_size.width);

        Ok(src_store)
    }
}

impl ScalingU16 for Scaler {
    fn resize_rgb_u16<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, u16, 3>,
        bit_depth: usize,
    ) -> Result<ImageStore<'a, u16, 3>, PicScaleError> {
        if store.width == 0 || store.height == 0 || new_size.width == 0 || new_size.height == 0 {
            return Err(PicScaleError::ZeroImageDimensions);
        }

        if check_image_size_overflow(store.width, store.height, store.channels) {
            return Err(PicScaleError::SourceImageIsTooLarge);
        }

        if check_image_size_overflow(new_size.width, new_size.height, store.channels) {
            return Err(PicScaleError::DestinationImageIsTooLarge);
        }

        if !(1..=16).contains(&bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(bit_depth));
        }

        let should_do_horizontal = store.width != new_size.width;
        let should_do_vertical = store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == ResamplingFunction::Nearest {
            let mut allocated_store: Vec<u16> = vec![0u16; new_size.width * 3 * new_size.height];
            resize_nearest::<u16, 3>(
                store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                &pool,
            );
            let mut new_image =
                ImageStore::<u16, 3>::new(allocated_store, new_size.width, new_size.height)?;
            new_image.bit_depth = bit_depth;
            return Ok(new_image);
        }

        let mut src_store = store;
        src_store.bit_depth = bit_depth;

        if should_do_vertical {
            let vertical_filters = self.generate_weights(src_store.height, new_size.height);
            let mut new_image_vertical =
                ImageStore::<u16, 3>::alloc(src_store.width, new_size.height);
            new_image_vertical.bit_depth = bit_depth;
            src_store.bit_depth = bit_depth;
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
            src_store = new_image_vertical;
        }

        assert_eq!(src_store.height, new_size.height);

        if should_do_horizontal {
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            let mut new_image_horizontal =
                ImageStore::<u16, 3>::alloc(new_size.width, new_size.height);
            new_image_horizontal.bit_depth = bit_depth;
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }

        assert_eq!(src_store.width, new_size.width);

        Ok(src_store)
    }

    /// Resizes u16 image
    ///
    /// # Arguments
    /// `new_size` - New image size
    /// `store` - original image store
    /// `bit_depth` - image bit depth, this is required for u16 image
    /// `premultiply_alpha` - flags is alpha is premultiplied
    ///
    /// # Panics
    /// Panic if bit depth < 1 or bit depth > 16
    fn resize_rgba_u16<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, u16, 4>,
        bit_depth: usize,
        premultiply_alpha: bool,
    ) -> Result<ImageStore<'a, u16, 4>, PicScaleError> {
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
            return Ok(store.copied());
        }

        let should_do_horizontal = store.width != new_size.width;
        let should_do_vertical = store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        if !(1..=16).contains(&bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(bit_depth));
        }

        let mut src_store = store;
        src_store.bit_depth = bit_depth;

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == ResamplingFunction::Nearest {
            let mut new_image = ImageStore::<u16, 4>::alloc(new_size.width, new_size.height);
            resize_nearest::<u16, 4>(
                src_store.buffer.borrow(),
                src_store.width,
                src_store.height,
                new_image.buffer.borrow_mut(),
                new_size.width,
                new_size.height,
                &pool,
            );
            new_image.bit_depth = bit_depth;
            return Ok(new_image);
        }

        let mut has_alpha_premultiplied = false;

        if premultiply_alpha {
            let is_alpha_premultiplication_reasonable =
                has_non_constant_cap_alpha_rgba16(src_store.buffer.borrow(), src_store.width);
            if is_alpha_premultiplication_reasonable {
                let mut new_store = ImageStore::<u16, 4>::alloc(src_store.width, src_store.height);
                new_store.bit_depth = src_store.bit_depth;
                src_store.premultiply_alpha(&mut new_store, &pool);
                src_store = new_store;
                has_alpha_premultiplied = true;
            }
        }

        if should_do_vertical {
            let mut new_image_vertical =
                ImageStore::<u16, 4>::alloc(src_store.width, new_size.height);
            let vertical_filters =
                self.generate_weights(src_store.height, new_image_vertical.height);
            src_store.bit_depth = bit_depth;
            new_image_vertical.bit_depth = bit_depth;
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
            src_store = new_image_vertical;
        }

        assert_eq!(src_store.height, new_size.height);

        if should_do_horizontal {
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            let mut new_image_horizontal =
                ImageStore::<u16, 4>::alloc(new_size.width, new_size.height);
            new_image_horizontal.bit_depth = bit_depth;
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }

        assert_eq!(src_store.width, new_size.width);

        if premultiply_alpha && has_alpha_premultiplied {
            src_store.unpremultiply_alpha(&pool);
            return Ok(src_store);
        }
        Ok(src_store)
    }

    /// Performs rescaling for u16 plane
    fn resize_plane_u16<'a>(
        &self,
        new_size: ImageSize,
        store: ImageStore<'a, u16, 1>,
        bit_depth: usize,
    ) -> Result<ImageStore<'a, u16, 1>, PicScaleError> {
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
            return Ok(store.copied());
        }

        if !(1..=16).contains(&bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(bit_depth));
        }

        let should_do_horizontal = store.width != new_size.width;
        let should_do_vertical = store.height != new_size.height;
        assert!(should_do_horizontal || should_do_vertical);

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == ResamplingFunction::Nearest {
            let mut allocated_store: Vec<u16> = vec![0u16; new_size.width * new_size.height];
            resize_nearest::<u16, 1>(
                store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
                &pool,
            );
            let mut new_image =
                ImageStore::<u16, 1>::new(allocated_store, new_size.width, new_size.height)?;
            new_image.bit_depth = bit_depth;
            return Ok(new_image);
        }

        let mut src_store = store;
        src_store.bit_depth = bit_depth;

        if should_do_vertical {
            let vertical_filters = self.generate_weights(src_store.height, new_size.height);
            let mut new_image_vertical =
                ImageStore::<u16, 1>::alloc(src_store.width, new_size.height);
            new_image_vertical.bit_depth = bit_depth;
            src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
            src_store = new_image_vertical;
        }

        assert_eq!(src_store.height, new_size.height);

        if should_do_horizontal {
            let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
            let mut new_image_horizontal =
                ImageStore::<u16, 1>::alloc(new_size.width, new_size.height);
            new_image_horizontal.bit_depth = bit_depth;
            src_store.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, &pool);
            src_store = new_image_horizontal;
        }
        assert_eq!(src_store.width, new_size.width);

        Ok(src_store)
    }
}

impl Scaler {
    /// Resizes RGBA2101010 image
    ///
    /// # Arguments
    /// `src` - source slice
    /// `src_size` - Source Image size
    /// `dst` - destination slice
    /// `new_size` - New image size
    ///
    pub fn resize_ar30(
        &self,
        src: &[u32],
        src_size: ImageSize,
        dst: &mut [u32],
        new_size: ImageSize,
        order: Ar30ByteOrder,
    ) -> Result<(), PicScaleError> {
        match order {
            Ar30ByteOrder::Host => resize_ar30_impl::<
                { Rgb30::Ar30 as usize },
                { Ar30ByteOrder::Host as usize },
            >(src, src_size, dst, new_size, self),
            Ar30ByteOrder::Network => resize_ar30_impl::<
                { Rgb30::Ar30 as usize },
                { Ar30ByteOrder::Network as usize },
            >(src, src_size, dst, new_size, self),
        }
    }

    /// Resizes RGBA1010102 image
    ///
    /// # Arguments
    /// `src` - source slice
    /// `src_size` - Source Image size
    /// `dst` - destination slice
    /// `new_size` - New image size
    ///
    pub fn resize_ra30(
        &self,
        src: &[u32],
        src_size: ImageSize,
        dst: &mut [u32],
        new_size: ImageSize,
        order: Ar30ByteOrder,
    ) -> Result<(), PicScaleError> {
        match order {
            Ar30ByteOrder::Host => resize_ar30_impl::<
                { Rgb30::Ra30 as usize },
                { Ar30ByteOrder::Host as usize },
            >(src, src_size, dst, new_size, self),
            Ar30ByteOrder::Network => resize_ar30_impl::<
                { Rgb30::Ra30 as usize },
                { Ar30ByteOrder::Network as usize },
            >(src, src_size, dst, new_size, self),
        }
    }
}
