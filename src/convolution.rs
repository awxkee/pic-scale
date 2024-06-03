use crate::filter_weights::FilterWeights;
use crate::threading_policy::ThreadingPolicy;
use crate::ImageStore;
use num_traits::FromPrimitive;
use std::fmt::Debug;

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
        threading_policy: ThreadingPolicy,
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
        threading_policy: ThreadingPolicy,
    );
}
