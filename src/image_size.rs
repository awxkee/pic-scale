#[derive(Copy, Clone)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

impl ImageSize {
    pub fn new(width: usize, height: usize) -> ImageSize {
        ImageSize { width, height }
    }
}