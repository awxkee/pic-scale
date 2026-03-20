/*
 * Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::validation::validate_sizes;
use crate::{ImageSize, ImageStore, ImageStoreMut, PicScaleError, ResamplingPlan, ThreadingPolicy};
use novtb::{ParallelZonedIterator, TbSliceMut};
use std::fmt::Debug;

pub(crate) struct ResampleNearestPlan<T, const N: usize> {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) threading_policy: ThreadingPolicy,
    pub(crate) _phantom_data: std::marker::PhantomData<T>,
}

impl<T: Copy + Send + Sync + Clone + Debug, const N: usize> ResamplingPlan<T, N>
    for ResampleNearestPlan<T, N>
{
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError> {
        self.resample_with_scratch(store, into, &mut [])
    }

    fn resample_with_scratch(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        _: &mut [T],
    ) -> Result<(), PicScaleError> {
        validate_sizes!(store, into, self.source_size, self.target_size);
        const SCALE: i32 = 32;

        let k_x: u64 = ((self.source_size.width as u64) << SCALE) / self.target_size.width as u64;
        let k_y: u64 = ((self.source_size.height as u64) << SCALE) / self.target_size.height as u64;
        let k_x_half: u64 = k_x >> 1;
        let k_y_half: u64 = k_y >> 1;

        let dst_stride = into.stride();
        let src_stride = store.stride();

        let dst = into.buffer.borrow_mut();

        let pool = self.threading_policy.get_nova_pool(ImageSize::new(
            self.target_size.width,
            self.target_size.height,
        ));

        let src = store.buffer.as_ref();

        dst.tb_par_chunks_exact_mut(dst_stride)
            .for_each_enumerated(&pool, |y, dst_chunk| {
                let src_y = ((y as u64 * k_y + k_y_half) >> SCALE) as usize;
                let src_offset_y = src_y * src_stride;

                let mut src_x_fixed = k_x_half;
                for dst in dst_chunk.as_chunks_mut::<N>().0.iter_mut() {
                    let src_x = (src_x_fixed >> SCALE) as usize;

                    let src_px = src_x * N;
                    let offset = src_offset_y + src_px;

                    let src_slice = &src[offset..(offset + N)];

                    for (src, dst) in src_slice.iter().zip(dst.iter_mut()) {
                        *dst = *src;
                    }
                    src_x_fixed += k_x;
                }
            });
        Ok(())
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![]
    }

    fn scratch_size(&self) -> usize {
        0
    }

    fn target_size(&self) -> ImageSize {
        self.target_size
    }

    fn source_size(&self) -> ImageSize {
        self.source_size
    }
}
