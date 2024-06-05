/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use rayon::ThreadPool;
use crate::ImageSize;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum ThreadingPolicy {
    Single,
    Fixed(usize),
    Adaptive,
}

impl<'a> ThreadingPolicy {
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

impl<'a> ThreadingPolicy {
    pub fn get_pool(&self, for_size: ImageSize) -> Option<ThreadPool> {
        if *self == ThreadingPolicy::Single {
            return None;
        }
        let threads_count = self.get_threads_count(for_size);
        let shared_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads_count)
            .use_current_thread()
            .build()
            .unwrap();
        return Some(shared_pool);
    }
}