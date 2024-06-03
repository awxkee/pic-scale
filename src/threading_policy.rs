use crate::ImageSize;

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum ThreadingPolicy {
    Single,
    Fixed(usize),
    Adaptive,
}

impl ThreadingPolicy {
    pub fn get_threads_count(&self, for_size: ImageSize) -> usize {
        match self {
            ThreadingPolicy::Single => 1,
            ThreadingPolicy::Fixed(thread_count) => (*thread_count).max(1),
            ThreadingPolicy::Adaptive => {
                let box_size = 256 * 256;
                let new_box_size = for_size.height * for_size.width;
                return (new_box_size / box_size).max(1).min(16);
            }
        }
    }
}
