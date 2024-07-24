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
use std::fmt::Debug;

use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_size::ImageSize;
use crate::image_store::ImageStore;
use crate::nearest_sampler::resize_nearest;
use crate::threading_policy::ThreadingPolicy;
use crate::ResamplingFunction::Nearest;
use crate::{ResamplingFilter, ResamplingFunction};

#[derive(Debug, Copy, Clone)]
/// Represents base scaling structure
pub struct Scaler {
    pub(crate) resampling_filter: ResamplingFilter,
    pub(crate) function: ResamplingFunction,
    pub(crate) threading_policy: ThreadingPolicy,
}

pub trait Scaling {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy);

    /// Performs rescaling for RGB, channel order does not matter
    fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3>;

    /// Performs rescaling for RGBA, for pre-multiplying alpha, converting to LUV or LAB alpha must be last channel
    fn resize_rgba(
        &self,
        new_size: ImageSize,
        store: ImageStore<u8, 4>,
        is_alpha_premultiplied: bool,
    ) -> ImageStore<u8, 4>;
}

pub trait ScalingF32 {
    /// Performs rescaling for RGB f32, channel order does not matter
    fn resize_rgb_f32(&self, new_size: ImageSize, store: ImageStore<f32, 3>) -> ImageStore<f32, 3>;
    /// Performs rescaling for RGBA f32
    fn resize_rgba_f32(
        &self,
        new_size: ImageSize,
        store: ImageStore<f32, 4>,
        is_alpha_premultiplied: bool,
    ) -> ImageStore<f32, 4>;
}

impl Scaler {
    /// Creates new Scaler instance with corresponding filter
    pub fn new(filter: ResamplingFunction) -> Self {
        Scaler {
            resampling_filter: filter.get_resampling_filter(),
            function: filter,
            threading_policy: ThreadingPolicy::Single,
        }
    }

    pub(crate) fn generate_weights(&self, in_size: usize, out_size: usize) -> FilterWeights<f32> {
        let scale = in_size as f32 / out_size as f32;
        let is_resizable_kernel = self.resampling_filter.is_resizable_kernel;
        let filter_scale_cutoff = match is_resizable_kernel {
            true => scale.max(1f32),
            false => 1f32,
        };
        let filter_base_size = self.resampling_filter.min_kernel_size;
        let resampling_function = self.resampling_filter.kernel;
        let window_func = self.resampling_filter.window;
        let base_size = (filter_base_size * filter_scale_cutoff).round() as usize;
        // Kernel size must be always odd
        let kernel_size = base_size * 2 + 1usize;
        let filter_radius = base_size as f32;
        let filter_scale = 1f32 / filter_scale_cutoff;
        let mut weights: Vec<f32> = vec![0f32; kernel_size * out_size];
        let mut local_filters = vec![0f32; kernel_size];
        let mut filter_position = 0usize;
        let blur_scale = match window_func {
            None => 1f32,
            Some(window) => {
                if window.blur > 0f32 {
                    1f32 / window.blur
                } else {
                    0f32
                }
            }
        };

        let mut bounds: Vec<FilterBounds> = vec![FilterBounds::new(0, 0); out_size];
        for i in 0..out_size {
            let center_x = ((i as f32 + 0.5f32) * scale).min(in_size as f32);
            let mut weights_sum: f32 = 0f32;
            let mut local_filter_iteration = 0usize;

            let start = (center_x - filter_radius).floor().max(0f32) as usize;
            let end = ((center_x + filter_radius).ceil().min(in_size as f32) as usize)
                .min(start + kernel_size);

            let center = center_x - 0.5f32;

            for k in start..end {
                let dx = k as f32 - center;
                let weight;
                if let Some(resampling_window) = window_func {
                    let mut x = dx.abs();
                    x = if resampling_window.blur > 0f32 {
                        x * blur_scale
                    } else {
                        x
                    };
                    x = if x <= resampling_window.taper {
                        0f32
                    } else {
                        (x - resampling_window.taper) / (1f32 - resampling_window.taper)
                    };
                    let window_producer = resampling_window.window;
                    let x_kernel_scaled = x * filter_scale;
                    let window = if x < resampling_window.window_size {
                        window_producer(x_kernel_scaled * resampling_window.window_size)
                    } else {
                        0f32
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

            const ALPHA: f32 = 0.7f32;
            if self.resampling_filter.is_ewa && !local_filters.is_empty() {
                weights_sum = unsafe { *local_filters.get_unchecked(0) };
                for j in 1..local_filter_iteration {
                    let new_weight = ALPHA * unsafe { *local_filters.get_unchecked(j) }
                        + (1f32 - ALPHA) * unsafe { *local_filters.get_unchecked(j - 1) };
                    unsafe {
                        *local_filters.get_unchecked_mut(j) = new_weight;
                    }
                    weights_sum += new_weight;
                }
            }

            let size = end - start;

            unsafe {
                *bounds.get_unchecked_mut(i) = FilterBounds::new(start, size);
            }

            if weights_sum != 0f32 {
                let recpeq = 1f32 / weights_sum;
                for i in 0..size {
                    unsafe {
                        *weights.get_unchecked_mut(filter_position + i) =
                            *local_filters.get_unchecked(i) * recpeq;
                    }
                }
            }

            filter_position += kernel_size;
        }

        return FilterWeights::<f32>::new(
            weights,
            kernel_size,
            kernel_size,
            out_size,
            filter_radius as i32,
            bounds,
        );
    }
}

impl Scaling for Scaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.threading_policy = threading_policy;
    }

    fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3> {
        if self.function == Nearest {
            let mut allocated_store: Vec<u8> = vec![0u8; new_size.width * 3 * new_size.height];
            resize_nearest::<u8, 3>(
                &store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
            );
            let new_image =
                ImageStore::<u8, 3>::new(allocated_store, new_size.width, new_size.height);
            return new_image;
        }
        let vertical_filters = self.generate_weights(store.height, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        let mut new_image_vertical = ImageStore::<u8, 3>::alloc(store.width, new_size.height);
        store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
        let mut new_image_horizontal = ImageStore::<u8, 3>::alloc(new_size.width, new_size.height);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            &pool,
        );
        new_image_horizontal
    }

    fn resize_rgba(
        &self,
        new_size: ImageSize,
        store: ImageStore<u8, 4>,
        is_alpha_premultiplied: bool,
    ) -> ImageStore<u8, 4> {
        let mut src_store = store;
        if is_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(src_store.width, src_store.height);
            src_store.unpremultiply_alpha(&mut premultiplied_store);
            src_store = premultiplied_store;
        }
        if self.function == Nearest {
            let mut new_image = ImageStore::<u8, 4>::alloc(new_size.width, new_size.height);
            resize_nearest::<u8, 4>(
                &src_store.buffer.borrow(),
                src_store.width,
                src_store.height,
                &mut new_image.buffer.borrow_mut(),
                new_size.width,
                new_size.height,
            );
            if is_alpha_premultiplied {
                let mut premultiplied_store =
                    ImageStore::<u8, 4>::alloc(new_image.width, new_image.height);
                new_image.premultiply_alpha(&mut premultiplied_store);
                return premultiplied_store;
            }
            return new_image;
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        let mut new_image_vertical = ImageStore::<u8, 4>::alloc(src_store.width, new_size.height);
        let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
        let vertical_filters = self.generate_weights(src_store.height, new_image_vertical.height);
        src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);

        let mut new_image_horizontal = ImageStore::<u8, 4>::alloc(new_size.width, new_size.height);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            &pool,
        );
        if is_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(new_image_horizontal.width, new_image_horizontal.height);
            new_image_horizontal.premultiply_alpha(&mut premultiplied_store);
            return premultiplied_store;
        }
        new_image_horizontal
    }
}

impl ScalingF32 for Scaler {
    fn resize_rgb_f32(&self, new_size: ImageSize, store: ImageStore<f32, 3>) -> ImageStore<f32, 3> {
        if self.function == Nearest {
            let mut allocated_store: Vec<f32> = vec![0f32; new_size.width * 4 * new_size.height];
            resize_nearest::<f32, 3>(
                &store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
            );
            let new_image =
                ImageStore::<f32, 3>::new(allocated_store, new_size.width, new_size.height);
            return new_image;
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        let allocated_store_vertical: Vec<f32> = vec![0f32; store.width * 3 * new_size.height];
        let mut new_image_vertical =
            ImageStore::<f32, 3>::new(allocated_store_vertical, store.width, new_size.height);
        let vertical_filters = self.generate_weights(store.height, new_image_vertical.height);
        store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);

        let allocated_store_horizontal: Vec<f32> = vec![0f32; new_size.width * 3 * new_size.height];
        let mut new_image_horizontal =
            ImageStore::<f32, 3>::new(allocated_store_horizontal, new_size.width, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            &pool,
        );
        new_image_horizontal
    }

    fn resize_rgba_f32(
        &self,
        new_size: ImageSize,
        store: ImageStore<f32, 4>,
        is_alpha_premultiplied: bool,
    ) -> ImageStore<f32, 4> {
        let mut src_store = store;
        if is_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<f32, 4>::alloc(src_store.width, src_store.height);
            src_store.unpremultiply_alpha(&mut premultiplied_store);
            src_store = premultiplied_store;
        }

        if self.function == Nearest {
            let mut allocated_store: Vec<f32> = vec![0f32; new_size.width * 4 * new_size.height];
            resize_nearest::<f32, 4>(
                &src_store.buffer.borrow(),
                src_store.width,
                src_store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
            );
            let new_image =
                ImageStore::<f32, 4>::new(allocated_store, new_size.width, new_size.height);

            if is_alpha_premultiplied {
                let mut premultiplied_store =
                    ImageStore::<f32, 4>::alloc(new_image.width, new_image.height);
                new_image.premultiply_alpha(&mut premultiplied_store);
                return premultiplied_store;
            }

            return new_image;
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        let allocated_store_vertical: Vec<f32> = vec![0f32; src_store.width * 4 * new_size.height];
        let mut new_image_vertical =
            ImageStore::<f32, 4>::new(allocated_store_vertical, src_store.width, new_size.height);
        let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
        let vertical_filters = self.generate_weights(src_store.height, new_image_vertical.height);
        src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);

        let allocated_store_horizontal: Vec<f32> = vec![0f32; new_size.width * 4 * new_size.height];
        let mut new_image_horizontal =
            ImageStore::<f32, 4>::new(allocated_store_horizontal, new_size.width, new_size.height);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            &pool,
        );

        if is_alpha_premultiplied {
            let mut premultiplied_store = ImageStore::<f32, 4>::alloc(
                new_image_horizontal.width,
                new_image_horizontal.height,
            );
            new_image_horizontal.premultiply_alpha(&mut premultiplied_store);
            return premultiplied_store;
        }

        new_image_horizontal
    }
}

impl Scaler {
    /// Performs rescaling for f32 plane
    pub fn resize_plane_f32(
        &self,
        new_size: ImageSize,
        store: ImageStore<f32, 1>,
    ) -> ImageStore<f32, 1> {
        if self.function == Nearest {
            let mut allocated_store: Vec<f32> = vec![0f32; new_size.width * 1 * new_size.height];
            resize_nearest::<f32, 1>(
                &store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
            );
            let new_image =
                ImageStore::<f32, 1>::new(allocated_store, new_size.width, new_size.height);
            return new_image;
        }

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        let allocated_store_vertical: Vec<f32> = vec![0f32; store.width * new_size.height];
        let mut new_image_vertical =
            ImageStore::<f32, 1>::new(allocated_store_vertical, store.width, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);
        let vertical_filters = self.generate_weights(store.height, new_image_vertical.height);
        store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);

        let allocated_store_horizontal: Vec<f32> = vec![0f32; new_size.width * new_size.height];
        let mut new_image_horizontal =
            ImageStore::<f32, 1>::new(allocated_store_horizontal, new_size.width, new_size.height);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            &pool,
        );
        new_image_horizontal
    }
}

impl Scaler {
    /// Performs rescaling for u8 plane
    pub fn resize_plane(&self, new_size: ImageSize, store: ImageStore<u8, 1>) -> ImageStore<u8, 1> {
        if self.function == Nearest {
            let mut allocated_store: Vec<u8> = vec![0u8; new_size.width * new_size.height];
            resize_nearest::<u8, 1>(
                &store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
            );
            let new_image =
                ImageStore::<u8, 1>::new(allocated_store, new_size.width, new_size.height);
            return new_image;
        }
        let vertical_filters = self.generate_weights(store.height, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        let mut new_image_vertical = ImageStore::<u8, 1>::alloc(store.width, new_size.height);
        store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
        let mut new_image_horizontal = ImageStore::<u8, 1>::alloc(new_size.width, new_size.height);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            &pool,
        );
        new_image_horizontal
    }
}
