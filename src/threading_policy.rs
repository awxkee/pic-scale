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
#![forbid(unsafe_code)]
use rayon::ThreadPool;
#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZeroUsize;
#[cfg(not(target_arch = "wasm32"))]
use std::thread::available_parallelism;

use crate::ImageSize;

#[repr(C)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
/// Declares thread policy usage
pub enum ThreadingPolicy {
    /// Will use only one current thread
    #[default]
    Single,
    /// Spawn provided threads count, will not work for wasm - fallback to Single
    Fixed(usize),
    /// Computes adaptive thread count between 1...`available parallelism`
    /// for given image bounds, will not work for wasm - fallback to Single
    Adaptive,
}

impl ThreadingPolicy {
    #[cfg(not(target_arch = "wasm32"))]
    /// Returns the number of threads to use for the given image dimensions under the
    /// selected policy variant.
    pub fn thread_count(&self, for_size: ImageSize) -> usize {
        match self {
            ThreadingPolicy::Single => 1,
            ThreadingPolicy::Fixed(thread_count) => (*thread_count).max(1),
            ThreadingPolicy::Adaptive => (for_size.width * for_size.height / (256 * 256))
                .clamp(1, Self::available_parallelism()),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn available_parallelism() -> usize {
        available_parallelism()
            .unwrap_or_else(|_| NonZeroUsize::new(1).unwrap())
            .get()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn thread_count(&self, _: ImageSize) -> usize {
        1
    }
}

impl ThreadingPolicy {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_pool(&self, for_size: ImageSize) -> Option<ThreadPool> {
        if *self == ThreadingPolicy::Single {
            return None;
        }
        let thread_count = self.thread_count(for_size);
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .ok()
    }

    pub(crate) fn get_nova_pool(&self, for_size: ImageSize) -> novtb::ThreadPool {
        if *self == ThreadingPolicy::Single {
            return novtb::ThreadPool::new(1);
        }
        let thread_count = self.thread_count(for_size);
        novtb::ThreadPool::new(thread_count)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn get_pool(&self, _: ImageSize) -> Option<ThreadPool> {
        None
    }
}
