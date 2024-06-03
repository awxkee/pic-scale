use crate::convolution::{HorizontalConvolutionPass, VerticalConvolutionPass};
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_size::ImageSize;
use crate::image_store::ImageStore;
use crate::nearest_sampler::resize_nearest;
use crate::threading_policy::ThreadingPolicy;
use crate::ResamplingFunction::Nearest;
use crate::{ResamplingFilter, ResamplingFunction};
use std::time::Instant;

#[derive(Copy, Clone)]
pub struct Scaler {
    pub(crate) resampling_filter: ResamplingFilter,
    pub(crate) function: ResamplingFunction,
    pub(crate) threading_policy: ThreadingPolicy,
}

impl<'a> Scaler {
    pub fn new(filter: ResamplingFunction) -> Self {
        Scaler {
            resampling_filter: filter.get_resampling_filter(),
            function: filter,
            threading_policy: ThreadingPolicy::Single,
        }
    }

    pub fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) {
        self.threading_policy = threading_policy;
    }

    pub(crate) fn generate_weights(&self, in_size: usize, out_size: usize) -> FilterWeights<f32> {
        let scale = (in_size as f32 / out_size as f32).max(1f32);
        let filter_base_size = self.resampling_filter.min_kernel_size as f32;
        let resampling_function = self.resampling_filter.function;
        // Kernel size must be always odd
        let base_size = (filter_base_size * scale).round() as usize;
        let kernel_size = base_size * 2 + 1usize;
        let filter_radius = base_size as i32;
        let filter_scale = 1f32 / scale;
        let mut weights: Vec<f32> = vec![];
        weights.resize(kernel_size * out_size, 0f32);
        let mut local_filters = vec![];
        local_filters.resize(kernel_size, 0f32);
        let mut filter_position = 0usize;

        let mut bounds: Vec<FilterBounds> = vec![];
        bounds.resize(out_size, FilterBounds::new(0, 0));
        for i in 0..out_size {
            let center_x = ((i as f32 + 0.5f32) * scale).min(in_size as f32);
            let mut weights_sum: f32 = 0f32;
            let mut local_filter_iteration = 0usize;

            let start = (center_x - filter_radius as f32).floor().max(0f32) as usize;
            let end = ((center_x + filter_radius as f32).ceil().min(in_size as f32) as usize)
                .min(start + kernel_size);

            let center = center_x - 0.5f32;

            for k in start..end {
                let dx = k as f32 - center;
                let weight = resampling_function(dx * filter_scale);
                weights_sum += weight;
                local_filters[local_filter_iteration] = weight;
                local_filter_iteration += 1;
            }

            let size = end - start;

            bounds[i] = FilterBounds::new(start, size);

            if weights_sum != 0f32 {
                let recpeq = 1f32 / weights_sum;
                for i in 0..size {
                    weights[filter_position + i] = local_filters[i] * recpeq;
                }
            }

            filter_position += kernel_size;
        }

        return FilterWeights::<f32>::new(
            weights,
            kernel_size,
            kernel_size,
            out_size,
            filter_radius,
            bounds,
        );
    }

    pub fn resize_rgb(
        &self,
        new_size: ImageSize,
        store: ImageStore<'static, u8, 3>,
    ) -> ImageStore<u8, 3> {
        if self.function == Nearest {
            let mut allocated_store: Vec<u8> = vec![];
            allocated_store.resize(new_size.width * 3 * new_size.height, 0u8);
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

        let start_time_filters = Instant::now();
        let vertical_filters = self.generate_weights(store.height, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);
        let elapsed_filters = start_time_filters.elapsed();

        let start_time = Instant::now();
        let mut new_image_vertical = ImageStore::<u8, 3>::alloc(store.width, new_size.height);
        store.convolve_vertical(
            vertical_filters,
            &mut new_image_vertical,
            self.threading_policy,
        );
        let elapsed_vertical = start_time.elapsed();

        let start_time = Instant::now();
        let mut new_image_horizontal = ImageStore::<u8, 3>::alloc(new_size.width, new_size.height);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            self.threading_policy,
        );
        let elapsed_time = start_time.elapsed();
        println!("Vertical: {:.2?}", elapsed_vertical);
        println!(
            "Horizontal: {:.2?}, Filters {:.2?}",
            elapsed_time, elapsed_filters
        );
        new_image_horizontal
    }

    pub fn resize_rgb_f32(
        &self,
        new_size: ImageSize,
        store: ImageStore<'static, f32, 3>,
    ) -> ImageStore<f32, 3> {
        if self.function == Nearest {
            let mut allocated_store: Vec<f32> = vec![];
            allocated_store.resize(new_size.width * 4 * new_size.height, 0f32);
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
        let mut allocated_store_vertical: Vec<f32> = vec![];
        allocated_store_vertical.resize(store.width * 3 * new_size.height, 0f32);
        let mut new_image_vertical =
            ImageStore::<f32, 3>::new(allocated_store_vertical, store.width, new_size.height);
        let vertical_filters = self.generate_weights(store.height, new_image_vertical.height);
        store.convolve_vertical(
            vertical_filters,
            &mut new_image_vertical,
            self.threading_policy,
        );

        let mut allocated_store_horizontal: Vec<f32> = vec![];
        allocated_store_horizontal.resize(new_size.width * 3 * new_size.height, 0f32);
        let mut new_image_horizontal =
            ImageStore::<f32, 3>::new(allocated_store_horizontal, new_size.width, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            self.threading_policy,
        );
        new_image_horizontal
    }

    pub fn resize_rgba_f32(
        &self,
        new_size: ImageSize,
        store: ImageStore<'static, f32, 4>,
    ) -> ImageStore<f32, 4> {
        if self.function == Nearest {
            let mut allocated_store: Vec<f32> = vec![];
            allocated_store.resize(new_size.width * 4 * new_size.height, 0f32);
            resize_nearest::<f32, 4>(
                &store.buffer.borrow(),
                store.width,
                store.height,
                &mut allocated_store,
                new_size.width,
                new_size.height,
            );
            let new_image =
                ImageStore::<f32, 4>::new(allocated_store, new_size.width, new_size.height);
            return new_image;
        }
        let mut allocated_store_vertical: Vec<f32> = vec![];
        allocated_store_vertical.resize(store.width * 4 * new_size.height, 0f32);
        let mut new_image_vertical =
            ImageStore::<f32, 4>::new(allocated_store_vertical, store.width, new_size.height);
        let vertical_filters = self.generate_weights(store.height, new_image_vertical.height);
        store.convolve_vertical(
            vertical_filters,
            &mut new_image_vertical,
            self.threading_policy,
        );

        let mut allocated_store_horizontal: Vec<f32> = vec![];
        allocated_store_horizontal.resize(new_size.width * 4 * new_size.height, 0f32);
        let mut new_image_horizontal =
            ImageStore::<f32, 4>::new(allocated_store_horizontal, new_size.width, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            self.threading_policy,
        );
        new_image_horizontal
    }

    pub fn resize_rgba(
        &self,
        new_size: ImageSize,
        store: ImageStore<'static, u8, 4>,
    ) -> ImageStore<u8, 4> {
        if self.function == Nearest {
            let mut new_image = ImageStore::<u8, 4>::alloc(new_size.width, new_size.height);
            resize_nearest::<u8, 4>(
                &store.buffer.borrow(),
                store.width,
                store.height,
                &mut new_image.buffer.borrow_mut(),
                new_size.width,
                new_size.height,
            );
            let new_image = ImageStore::<u8, 4>::alloc(new_size.width, new_size.height);
            return new_image;
        }
        let mut new_image_vertical = ImageStore::<u8, 4>::alloc(store.width, new_size.height);
        let vertical_filters = self.generate_weights(store.height, new_image_vertical.height);
        store.convolve_vertical(
            vertical_filters,
            &mut new_image_vertical,
            self.threading_policy,
        );

        let mut new_image_horizontal = ImageStore::<u8, 4>::alloc(new_size.width, new_size.height);
        let horizontal_filters = self.generate_weights(store.width, new_size.width);
        new_image_vertical.convolve_horizontal(
            horizontal_filters,
            &mut new_image_horizontal,
            self.threading_policy,
        );

        new_image_horizontal
    }
}
