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
use crate::factory::Rgb30;
use crate::image_store::CheckStoreDensity;
use crate::validation::{validate_scratch, validate_sizes};
use crate::{BufferStore, ImageSize, ImageStore, ImageStoreMut, PicScaleError, ResamplingPlan};
use std::sync::Arc;

pub(crate) trait Ar30Destructuring {
    fn decompose(&self, store: &ImageStore<'_, u8, 4>, into: &mut ImageStoreMut<u16, 3>);
    fn compose(&self, store: &ImageStore<u16, 3>, into: &mut ImageStoreMut<'_, u8, 4>);
}

pub(crate) struct Ar30DestructuringImpl<const BYTES: usize> {
    pub(crate) rgb30: Rgb30,
}

impl<const BYTES: usize> Ar30Destructuring for Ar30DestructuringImpl<BYTES> {
    fn decompose(&self, store: &ImageStore<'_, u8, 4>, into: &mut ImageStoreMut<u16, 3>) {
        match self.rgb30 {
            Rgb30::Ar30 => {
                for (dst, src) in into
                    .buffer
                    .borrow_mut()
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(store.buffer.as_chunks::<4>().0.iter())
                {
                    let unpacked = Rgb30::Ar30
                        .unpack::<BYTES>(u32::from_ne_bytes([src[0], src[1], src[2], src[3]]));
                    dst[0] = unpacked.0 as u16;
                    dst[1] = unpacked.1 as u16;
                    dst[2] = unpacked.2 as u16;
                }
            }
            Rgb30::Ra30 => {
                for (dst, src) in into
                    .buffer
                    .borrow_mut()
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(store.buffer.as_chunks::<4>().0.iter())
                {
                    let unpacked = Rgb30::Ra30
                        .unpack::<BYTES>(u32::from_ne_bytes([src[0], src[1], src[2], src[3]]));
                    dst[0] = unpacked.0 as u16;
                    dst[1] = unpacked.1 as u16;
                    dst[2] = unpacked.2 as u16;
                }
            }
        }
    }

    fn compose(&self, store: &ImageStore<u16, 3>, into: &mut ImageStoreMut<'_, u8, 4>) {
        match self.rgb30 {
            Rgb30::Ar30 => {
                for (dst, src) in into
                    .buffer
                    .borrow_mut()
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(store.buffer.as_chunks::<3>().0.iter())
                {
                    let packed = Rgb30::Ar30.pack_w_a::<BYTES>(
                        src[0] as i32,
                        src[1] as i32,
                        src[2] as i32,
                        3,
                    );
                    let target_bytes = packed.to_ne_bytes();
                    dst[0] = target_bytes[0];
                    dst[1] = target_bytes[1];
                    dst[2] = target_bytes[2];
                    dst[3] = target_bytes[3];
                }
            }
            Rgb30::Ra30 => {
                for (dst, src) in into
                    .buffer
                    .borrow_mut()
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(store.buffer.as_chunks::<3>().0.iter())
                {
                    let packed = Rgb30::Ra30.pack_w_a::<BYTES>(
                        src[0] as i32,
                        src[1] as i32,
                        src[2] as i32,
                        3,
                    );
                    let target_bytes = packed.to_ne_bytes();
                    dst[0] = target_bytes[0];
                    dst[1] = target_bytes[1];
                    dst[2] = target_bytes[2];
                    dst[3] = target_bytes[3];
                }
            }
        }
    }
}

struct ScratchLayout {
    inner_filter_size: usize,
    src_rgb16_size: usize,
    tgt_rgb16_size: usize,
}

impl ScratchLayout {
    fn new(plan: &Ar30Plan) -> Self {
        Self {
            inner_filter_size: plan.inner_filter.scratch_size() * size_of::<u16>() + 2,
            src_rgb16_size: plan.source_size.height * plan.source_size.width * 3 * size_of::<u16>()
                + 2,
            tgt_rgb16_size: plan.target_size.height * plan.target_size.width * 3 * size_of::<u16>()
                + 2,
        }
    }

    fn total(&self) -> usize {
        self.inner_filter_size + self.src_rgb16_size + self.tgt_rgb16_size
    }
}

pub(crate) struct Ar30Plan {
    pub(crate) source_size: ImageSize,
    pub(crate) target_size: ImageSize,
    pub(crate) inner_filter: Arc<dyn ResamplingPlan<u16, 3> + Send + Sync>,
    pub(crate) decomposer: Arc<dyn Ar30Destructuring + Send + Sync>,
}

impl ResamplingPlan<u8, 4> for Ar30Plan {
    fn resample(
        &self,
        store: &ImageStore<'_, u8, 4>,
        into: &mut ImageStoreMut<'_, u8, 4>,
    ) -> Result<(), PicScaleError> {
        let mut scratch = self.alloc_scratch();
        self.resample_with_scratch(store, into, &mut scratch)
    }

    fn resample_with_scratch(
        &self,
        store: &ImageStore<'_, u8, 4>,
        into: &mut ImageStoreMut<'_, u8, 4>,
        scratch: &mut [u8],
    ) -> Result<(), PicScaleError> {
        validate_sizes!(store, into, self.source_size, self.target_size);
        let scratch = validate_scratch!(scratch, self.scratch_size());
        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }
        let scratch_rgb16_image_size = self.source_size.height * self.source_size.width * 3;
        let scratch_target_rgb16_image_size = self.target_size.height * self.target_size.width * 3;
        let scratch_layout = ScratchLayout::new(self);
        let (inner_filter_scratch, tail0) = scratch.split_at_mut(scratch_layout.inner_filter_size);
        let (ar30_image_scratch_u8, tail1) = tail0.split_at_mut(scratch_layout.src_rgb16_size);
        let (ar30_target_scratch_u8, _) = tail1.split_at_mut(scratch_layout.tgt_rgb16_size);
        let (ar30_image_scratch, _) =
            mangle_slice_u8_as_u16(ar30_image_scratch_u8).split_at_mut(scratch_rgb16_image_size);
        let (inner_filter_scratch, _) = mangle_slice_u8_as_u16(inner_filter_scratch)
            .split_at_mut(self.inner_filter.scratch_size());
        let (ar30_target_scratch, _) = mangle_slice_u8_as_u16(ar30_target_scratch_u8)
            .split_at_mut(scratch_target_rgb16_image_size);
        let mut ar30_image = ImageStoreMut::<u16, 3> {
            buffer: BufferStore::Borrowed(ar30_image_scratch),
            channels: 3,
            width: store.width,
            height: store.height,
            stride: store.width * 3,
            bit_depth: 10,
        };
        self.decomposer.decompose(store, &mut ar30_image);
        let fixed_ar30 = ar30_image.to_immutable();
        let mut ar30_image_target = ImageStoreMut::<u16, 3> {
            buffer: BufferStore::Borrowed(ar30_target_scratch),
            channels: 3,
            width: into.width,
            height: into.height,
            stride: into.width * 3,
            bit_depth: 10,
        };
        self.inner_filter.resample_with_scratch(
            &fixed_ar30,
            &mut ar30_image_target,
            inner_filter_scratch,
        )?;
        let fixed_target = ar30_image_target.to_immutable();
        self.decomposer.compose(&fixed_target, into);
        Ok(())
    }

    fn alloc_scratch(&self) -> Vec<u8> {
        vec![u8::default(); self.scratch_size()]
    }

    fn scratch_size(&self) -> usize {
        let layout = ScratchLayout::new(self);
        layout.total()
    }

    fn target_size(&self) -> ImageSize {
        self.target_size
    }

    fn source_size(&self) -> ImageSize {
        self.source_size
    }
}

fn mangle_slice_u8_as_u16(slice: &mut [u8]) -> &mut [u16] {
    // we ensured in `scratch_size` that we'll have enough room to do so
    let (_, middle, _) = unsafe { slice.align_to_mut::<u16>() };
    middle
}
