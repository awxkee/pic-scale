use std::fmt::Debug;

use num_traits::FromPrimitive;
use rayon::ThreadPool;

use crate::filter_weights::FilterWeights;
use crate::ImageStore;

pub trait HorizontalConvolutionPass<T, const N: usize>
where
    T: FromPrimitive,
    T: Clone,
    T: Copy,
    T: Debug,
{
    fn convolve_horizontal(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<T, N>,
        pool: &Option<ThreadPool>,
    );
}

pub trait VerticalConvolutionPass<T, const N: usize>
where
    T: FromPrimitive,
    T: Clone,
    T: Copy,
    T: Debug,
{
    fn convolve_vertical(
        &self,
        filter_weights: FilterWeights<f32>,
        destination: &mut ImageStore<T, N>,
        pool: &Option<ThreadPool>,
    );
}
