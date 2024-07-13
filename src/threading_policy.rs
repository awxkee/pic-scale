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
        return match rayon::ThreadPoolBuilder::new()
            .num_threads(threads_count)
            .build()
        {
            Ok(pool) => Some(pool),
            Err(_) => None,
        };
    }
}
