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
use crate::alpha_check::has_non_constant_cap_alpha;
use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_size::ImageSize;
use crate::image_store::ImageStore;
use crate::nearest_sampler::resize_nearest;
use crate::threading_policy::ThreadingPolicy;
use crate::ResamplingFunction::Nearest;
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
    fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3>;

    /// Performs rescaling for RGBA, for pre-multiplying alpha, converting to LUV or LAB alpha must be last channel
    fn resize_rgba(
        &self,
        new_size: ImageSize,
        store: ImageStore<u8, 4>,
        premultiply_alpha: bool,
    ) -> ImageStore<u8, 4>;
}

pub trait ScalingF32 {
    /// Performs rescaling for RGB f32, channel order does not matter
    fn resize_rgb_f32(&self, new_size: ImageSize, store: ImageStore<f32, 3>) -> ImageStore<f32, 3>;

    /// Performs rescaling for RGBA f32, alpha expected to be last
    fn resize_rgba_f32(
        &self,
        new_size: ImageSize,
        store: ImageStore<f32, 4>,
        premultiply_alpha: bool,
    ) -> ImageStore<f32, 4>;
}

pub trait ScalingU16 {
    /// Performs rescaling for Planar u16, channel order does not matter
    ///
    /// # Arguments
    /// `new_size` - New image size
    /// `store` - original image store
    /// `bit_depth` - image bit depth, this is required for u16 image
    ///
    /// # Panics
    /// Panic if bit depth < 1 or bit depth > 16
    fn resize_plane_u16(
        &self,
        new_size: ImageSize,
        store: ImageStore<u16, 1>,
        bit_depth: usize,
    ) -> ImageStore<u16, 1>;

    /// Performs rescaling for RGB, channel order does not matter
    ///
    /// # Arguments
    /// `new_size` - New image size
    /// `store` - original image store
    /// `bit_depth` - image bit depth, this is required for u16 image
    ///
    /// # Panics
    /// Panic if bit depth < 1 or bit depth > 16
    fn resize_rgb_u16(
        &self,
        new_size: ImageSize,
        store: ImageStore<u16, 3>,
        bit_depth: usize,
    ) -> ImageStore<u16, 3>;

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
    fn resize_rgba_u16(
        &self,
        new_size: ImageSize,
        store: ImageStore<u16, 4>,
        bit_depth: usize,
        premultiply_alpha: bool,
    ) -> ImageStore<u16, 4>;
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
        let window_func = resampling_filter.window;
        let base_size: usize = (filter_base_size.as_() * filter_scale_cutoff).round().as_();
        // Kernel size must be always odd
        let kernel_size = base_size * 2 + 1usize;
        let filter_radius = base_size.as_();
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

        let mut bounds: Vec<FilterBounds> = vec![FilterBounds::new(0, 0); out_size];
        for i in 0..out_size {
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

            unsafe {
                *bounds.get_unchecked_mut(i) = FilterBounds::new(start, size);
            }

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
    }
}

impl Scaler {
    pub(crate) fn resize_rgba_impl(
        &self,
        new_size: ImageSize,
        store: ImageStore<u8, 4>,
        premultiply_alpha: bool,
        pool: &Option<ThreadPool>,
    ) -> ImageStore<u8, 4> {
        let mut src_store = store;
        if self.function == Nearest {
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
            return new_image;
        }

        let mut has_alpha_premultiplied = false;

        if premultiply_alpha {
            let is_alpha_premultiplication_reasonable = has_non_constant_cap_alpha::<u8, 3, 4>(
                src_store.buffer.borrow(),
                src_store.width,
                8,
            );
            if is_alpha_premultiplication_reasonable {
                let mut premultiplied_store =
                    ImageStore::<u8, 4>::alloc(src_store.width, src_store.height);
                src_store.premultiply_alpha(&mut premultiplied_store, pool);
                src_store = premultiplied_store;
                has_alpha_premultiplied = true;
            }
        }

        let mut new_image_vertical = ImageStore::<u8, 4>::alloc(src_store.width, new_size.height);
        let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
        let vertical_filters = self.generate_weights(src_store.height, new_image_vertical.height);
        src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, pool);

        let mut new_image_horizontal = ImageStore::<u8, 4>::alloc(new_size.width, new_size.height);
        new_image_vertical.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, pool);
        if premultiply_alpha && has_alpha_premultiplied {
            let mut premultiplied_store =
                ImageStore::<u8, 4>::alloc(new_image_horizontal.width, new_image_horizontal.height);
            new_image_horizontal.unpremultiply_alpha(&mut premultiplied_store, pool);
            return premultiplied_store;
        }
        new_image_horizontal
    }
}

impl Scaling for Scaler {
    fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.threading_policy = threading_policy;
    }

    fn resize_rgb(&self, new_size: ImageSize, store: ImageStore<u8, 3>) -> ImageStore<u8, 3> {
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
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
            let new_image =
                ImageStore::<u8, 3>::new(allocated_store, new_size.width, new_size.height);
            return new_image.unwrap();
        }
        let vertical_filters = self.generate_weights(store.height, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);

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
        premultiply_alpha: bool,
    ) -> ImageStore<u8, 4> {
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));
        self.resize_rgba_impl(new_size, store, premultiply_alpha, &pool)
    }
}

impl Scaler {
    pub(crate) fn resize_rgba_f32_impl(
        &self,
        new_size: ImageSize,
        store: ImageStore<f32, 4>,
        premultiply_alpha: bool,
        pool: &Option<ThreadPool>,
    ) -> ImageStore<f32, 4> {
        let mut src_store = store;
        if self.function == Nearest {
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
            let new_image =
                ImageStore::new(allocated_store, new_size.width, new_size.height).unwrap();
            return new_image;
        }

        if premultiply_alpha {
            let mut premultiplied_store =
                ImageStore::<f32, 4>::alloc(src_store.width, src_store.height);
            src_store.premultiply_alpha(&mut premultiplied_store, pool);
            src_store = premultiplied_store;
        }

        let allocated_store_vertical: Vec<f32> = vec![0f32; src_store.width * 4 * new_size.height];
        let mut new_image_vertical =
            ImageStore::<f32, 4>::new(allocated_store_vertical, src_store.width, new_size.height)
                .unwrap();
        let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
        let vertical_filters = self.generate_weights(src_store.height, new_image_vertical.height);
        src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, pool);

        let allocated_store_horizontal: Vec<f32> = vec![0f32; new_size.width * 4 * new_size.height];
        let mut new_image_horizontal =
            ImageStore::<f32, 4>::new(allocated_store_horizontal, new_size.width, new_size.height)
                .unwrap();
        new_image_vertical.convolve_horizontal(horizontal_filters, &mut new_image_horizontal, pool);

        if premultiply_alpha {
            let mut premultiplied_store = ImageStore::<f32, 4>::alloc(
                new_image_horizontal.width,
                new_image_horizontal.height,
            );
            new_image_horizontal.unpremultiply_alpha(&mut premultiplied_store, pool);
            return premultiplied_store;
        }

        new_image_horizontal
    }
}

impl ScalingF32 for Scaler {
    fn resize_rgb_f32(&self, new_size: ImageSize, store: ImageStore<f32, 3>) -> ImageStore<f32, 3> {
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
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
            return new_image.unwrap();
        }

        let allocated_store_vertical: Vec<f32> = vec![0f32; store.width * 3 * new_size.height];
        let mut new_image_vertical =
            ImageStore::<f32, 3>::new(allocated_store_vertical, store.width, new_size.height)
                .unwrap();
        let vertical_filters = self.generate_weights(store.height, new_image_vertical.height);
        store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);

        let allocated_store_horizontal: Vec<f32> = vec![0f32; new_size.width * 3 * new_size.height];
        let mut new_image_horizontal =
            ImageStore::<f32, 3>::new(allocated_store_horizontal, new_size.width, new_size.height)
                .unwrap();
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
        premultiply_alpha: bool,
    ) -> ImageStore<f32, 4> {
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));
        self.resize_rgba_f32_impl(new_size, store, premultiply_alpha, &pool)
    }
}

impl Scaler {
    /// Performs rescaling for f32 plane
    pub fn resize_plane_f32(
        &self,
        new_size: ImageSize,
        store: ImageStore<f32, 1>,
    ) -> ImageStore<f32, 1> {
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
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
                ImageStore::<f32, 1>::new(allocated_store, new_size.width, new_size.height)
                    .unwrap();
            return new_image;
        }

        let allocated_store_vertical: Vec<f32> = vec![0f32; store.width * new_size.height];
        let mut new_image_vertical =
            ImageStore::<f32, 1>::new(allocated_store_vertical, store.width, new_size.height)
                .unwrap();
        let horizontal_filters = self.generate_weights(store.width, new_size.width);
        let vertical_filters = self.generate_weights(store.height, new_image_vertical.height);
        store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);

        let allocated_store_horizontal: Vec<f32> = vec![0f32; new_size.width * new_size.height];
        let mut new_image_horizontal =
            ImageStore::<f32, 1>::new(allocated_store_horizontal, new_size.width, new_size.height)
                .unwrap();
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
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
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
                ImageStore::<u8, 1>::new(allocated_store, new_size.width, new_size.height).unwrap();
            return new_image;
        }
        let vertical_filters = self.generate_weights(store.height, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);

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

impl ScalingU16 for Scaler {
    fn resize_rgb_u16(
        &self,
        new_size: ImageSize,
        store: ImageStore<u16, 3>,
        bit_depth: usize,
    ) -> ImageStore<u16, 3> {
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
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
                ImageStore::<u16, 3>::new(allocated_store, new_size.width, new_size.height)
                    .unwrap();
            new_image.bit_depth = bit_depth;
            return new_image;
        }

        if !(1..=16).contains(&bit_depth) {
            panic!("Bit depth must be in [1, 16] but got {}", bit_depth);
        }

        let vertical_filters = self.generate_weights(store.height, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);

        let mut copied_store = store;

        let mut new_image_vertical =
            ImageStore::<u16, 3>::alloc(copied_store.width, new_size.height);
        new_image_vertical.bit_depth = bit_depth;
        copied_store.bit_depth = bit_depth;
        copied_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
        let mut new_image_horizontal = ImageStore::<u16, 3>::alloc(new_size.width, new_size.height);
        new_image_horizontal.bit_depth = bit_depth;
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            &pool,
        );
        new_image_horizontal
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
    fn resize_rgba_u16(
        &self,
        new_size: ImageSize,
        store: ImageStore<u16, 4>,
        bit_depth: usize,
        premultiply_alpha: bool,
    ) -> ImageStore<u16, 4> {
        let mut src_store = store;

        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
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
            return new_image;
        }

        let mut has_alpha_premultiplied = false;

        if premultiply_alpha {
            let is_alpha_premultiplication_reasonable = has_non_constant_cap_alpha::<u16, 3, 4>(
                src_store.buffer.borrow(),
                src_store.width,
                bit_depth as u32,
            );
            if is_alpha_premultiplication_reasonable {
                let mut premultiplied_store =
                    ImageStore::<u16, 4>::alloc(src_store.width, src_store.height);
                src_store.bit_depth = bit_depth;
                premultiplied_store.bit_depth = bit_depth;
                src_store.premultiply_alpha(&mut premultiplied_store, &pool);
                src_store = premultiplied_store;
                has_alpha_premultiplied = true;
            }
        }

        if !(1..=16).contains(&bit_depth) {
            panic!("Bit depth must be in [1, 16] but got {}", bit_depth);
        }

        let mut new_image_vertical = ImageStore::<u16, 4>::alloc(src_store.width, new_size.height);
        let horizontal_filters = self.generate_weights(src_store.width, new_size.width);
        let vertical_filters = self.generate_weights(src_store.height, new_image_vertical.height);
        src_store.bit_depth = bit_depth;
        new_image_vertical.bit_depth = bit_depth;
        src_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);

        let mut new_image_horizontal = ImageStore::<u16, 4>::alloc(new_size.width, new_size.height);
        new_image_horizontal.bit_depth = bit_depth;
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            &pool,
        );

        if premultiply_alpha & has_alpha_premultiplied {
            let mut premultiplied_store = ImageStore::<u16, 4>::alloc(
                new_image_horizontal.width,
                new_image_horizontal.height,
            );
            premultiplied_store.bit_depth = bit_depth;
            new_image_horizontal.unpremultiply_alpha(&mut premultiplied_store, &pool);
            return premultiplied_store;
        }
        new_image_horizontal
    }

    /// Performs rescaling for u16 plane
    fn resize_plane_u16(
        &self,
        new_size: ImageSize,
        store: ImageStore<u16, 1>,
        bit_depth: usize,
    ) -> ImageStore<u16, 1> {
        let pool = self
            .threading_policy
            .get_pool(ImageSize::new(new_size.width, new_size.height));

        if self.function == Nearest {
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
                ImageStore::<u16, 1>::new(allocated_store, new_size.width, new_size.height)
                    .unwrap();
            new_image.bit_depth = bit_depth;
            return new_image;
        }

        if !(1..=16).contains(&bit_depth) {
            panic!("Bit depth must be in [1, 16] but got {}", bit_depth);
        }

        let vertical_filters = self.generate_weights(store.height, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);

        let mut copied_store = store;
        copied_store.bit_depth = bit_depth;

        let mut new_image_vertical =
            ImageStore::<u16, 1>::alloc(copied_store.width, new_size.height);
        new_image_vertical.bit_depth = bit_depth;
        copied_store.convolve_vertical(vertical_filters, &mut new_image_vertical, &pool);
        let mut new_image_horizontal = ImageStore::<u16, 1>::alloc(new_size.width, new_size.height);
        new_image_horizontal.bit_depth = bit_depth;
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            &pool,
        );
        new_image_horizontal
    }
}
