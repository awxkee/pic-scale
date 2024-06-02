use num_traits::FromPrimitive;

#[derive(Clone)]
pub struct ImageStore<T, const N: usize> where T: FromPrimitive, T: Clone, T: Copy {
    pub buffer: Vec<T>,
    pub channels: usize,
    pub width: usize,
    pub height: usize,
}

impl<'a, T, const N: usize> ImageStore<T, N> where T: FromPrimitive, T: Clone, T: Copy {
    pub fn new(slice_ref: Vec<T>, width: usize, height: usize) -> ImageStore<T, N> {
        ImageStore::<T, N> { buffer: slice_ref, channels: N, width, height }
    }

}