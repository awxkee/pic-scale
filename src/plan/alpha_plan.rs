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

use crate::convolution::{ColumnFilter, ColumnSource, RowFilter, TrampolineFilter};
use crate::image_store::{AssociateAlpha, CheckStoreDensity, UnassociateAlpha};
use crate::plan::TrampolineFiltering;
use crate::threading_policy::ScalingPool;
use crate::validation::{try_vec, validate_scratch, validate_sizes};
use crate::{
    ImageSize, ImageStore, ImageStoreMut, PicScaleError, Resampling, ResamplingPlan,
    ThreadingPolicy, WorkloadStrategy,
};
use std::any::TypeId;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

struct AlphaPlan {
    source_size: ImageSize,
    target_size: ImageSize,
    threading_policy: ThreadingPolicy,
    workload_strategy: WorkloadStrategy,
    needs_alpha_forward: bool,
    needs_alpha_backward: bool,
}

impl AlphaPlan {
    fn alpha_scratch_size<const N: usize>(&self) -> usize {
        self.source_size.width * self.source_size.height * N
    }

    fn resample<T, const N: usize>(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        scratch: Option<&mut [T]>,
        filter_scratch_size: usize,
        filter: impl FnOnce(
            &ImageStore<'_, T, N>,
            &mut ImageStoreMut<'_, T, N>,
            &mut [T],
        ) -> Result<(), PicScaleError>,
    ) -> Result<(), PicScaleError>
    where
        T: Copy + Send + Sync + Clone + Debug + Default + 'static,
        for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
        for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
    {
        validate_sizes!(store, into, self.source_size, self.target_size);
        let alpha_scratch_size = self.alpha_scratch_size::<N>();
        let scratch = match scratch {
            Some(scratch) => Some(validate_scratch!(
                scratch,
                alpha_scratch_size + filter_scratch_size
            )),
            None => None,
        };
        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }

        let premultiply = self.needs_alpha_forward
            && (!self.needs_alpha_backward || store.is_alpha_premultiplication_needed());
        let mut allocated_scratch;
        let (scratch, alpha_size) = match scratch {
            // Caller-provided scratch retains its full size and layout for any source image.
            Some(scratch) => (scratch, alpha_scratch_size),
            None => {
                let alpha_size = if premultiply { alpha_scratch_size } else { 0 };
                allocated_scratch = try_vec![T::default(); alpha_size + filter_scratch_size];
                (allocated_scratch.as_mut_slice(), alpha_size)
            }
        };
        let (alpha_scratch, filter_scratch) = scratch.split_at_mut(alpha_size);

        let pool = self.threading_policy.get_nova_pool(self.target_size);
        let premultiplied;
        let source = if premultiply {
            premultiplied = premultiply_alpha(store, alpha_scratch, into.bit_depth, &pool)?;
            &premultiplied
        } else {
            store
        };
        filter(source, into, filter_scratch)?;

        if premultiply || (self.needs_alpha_backward && !self.needs_alpha_forward) {
            into.unpremultiply_alpha(&pool, self.workload_strategy);
        }
        Ok(())
    }
}

fn premultiply_alpha<'a, T, const N: usize>(
    store: &ImageStore<'_, T, N>,
    alpha_scratch: &'a mut [T],
    bit_depth: usize,
    pool: &ScalingPool,
) -> Result<ImageStore<'a, T, N>, PicScaleError>
where
    T: Copy + Send + Sync + Clone + Debug + Default + 'static,
    for<'b> ImageStore<'b, T, N>: AssociateAlpha<T, N>,
{
    let mut new_store =
        ImageStoreMut::<T, N>::from_slice(alpha_scratch, store.width, store.height)?;
    new_store.bit_depth = bit_depth;
    store.premultiply_alpha(&mut new_store, pool);
    Ok(ImageStore {
        buffer: std::borrow::Cow::Borrowed(alpha_scratch),
        channels: N,
        width: store.width,
        height: store.height,
        stride: store.width * N,
        bit_depth,
    })
}

struct AlphaVerticalConvolvePlan<T: Send + Sync, const N: usize> {
    alpha: AlphaPlan,
    vertical_filter: Arc<dyn ColumnFilter<T, N> + Send + Sync>,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default + 'static, const N: usize> ResamplingPlan<T, N>
    for AlphaVerticalConvolvePlan<T, N>
where
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
{
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError> {
        self.alpha
            .resample(store, into, None, 0, |source, into, _| {
                self.vertical_filter.filter(source, into);
                Ok(())
            })
    }

    fn resample_with_scratch(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        scratch: &mut [T],
    ) -> Result<(), PicScaleError> {
        self.alpha
            .resample(store, into, Some(scratch), 0, |source, into, _| {
                self.vertical_filter.filter(source, into);
                Ok(())
            })
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![T::default(); self.scratch_size()]
    }

    fn scratch_size(&self) -> usize {
        self.alpha.alpha_scratch_size::<N>()
    }

    fn target_size(&self) -> ImageSize {
        self.alpha.target_size
    }

    fn source_size(&self) -> ImageSize {
        self.alpha.source_size
    }
}

struct AlphaHorizontalConvolvePlan<T: Send + Sync, const N: usize> {
    alpha: AlphaPlan,
    horizontal_filter: Arc<dyn RowFilter<T, N> + Send + Sync>,
}

impl<T: Copy + Send + Sync + Clone + Debug + Default + 'static, const N: usize> ResamplingPlan<T, N>
    for AlphaHorizontalConvolvePlan<T, N>
where
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
{
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError> {
        self.alpha
            .resample(store, into, None, 0, |source, into, _| {
                self.horizontal_filter.filter(source, into);
                Ok(())
            })
    }

    fn resample_with_scratch(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        scratch: &mut [T],
    ) -> Result<(), PicScaleError> {
        self.alpha
            .resample(store, into, Some(scratch), 0, |source, into, _| {
                self.horizontal_filter.filter(source, into);
                Ok(())
            })
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![T::default(); self.scratch_size()]
    }

    fn scratch_size(&self) -> usize {
        self.alpha.alpha_scratch_size::<N>()
    }

    fn target_size(&self) -> ImageSize {
        self.alpha.target_size
    }

    fn source_size(&self) -> ImageSize {
        self.alpha.source_size
    }
}

struct AlphaBothAxesConvolvePlan<T: Send + Sync, const N: usize> {
    alpha: AlphaPlan,
    horizontal_filter: Arc<dyn RowFilter<T, N> + Send + Sync>,
    vertical_filter: Arc<dyn ColumnFilter<T, N> + Send + Sync>,
    trampoline_filter: Arc<TrampolineFiltering<T, N>>,
    streaming: bool,
    band_scratch_size: Option<usize>,
}

// A multiple of four keeps the trampoline's batched horizontal pass active.
const ALPHA_BAND_ROWS: usize = 16;

impl<T: Send + Sync + Clone, const N: usize> AlphaBothAxesConvolvePlan<T, N> {
    fn filter_scratch_size(&self) -> usize {
        if self.streaming {
            self.trampoline_filter.scratch_size()
        } else {
            self.alpha.source_size.width * self.alpha.target_size.height * N
        }
    }

    fn filter(
        &self,
        source: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        scratch: &mut [T],
    ) -> Result<(), PicScaleError> {
        if self.streaming {
            self.trampoline_filter.filter(source, into, scratch);
        } else {
            let mut new_image_vertical = ImageStoreMut::<T, N>::from_slice(
                scratch,
                source.width,
                self.alpha.target_size.height,
            )?;
            new_image_vertical.bit_depth = into.bit_depth;
            self.vertical_filter.filter(source, &mut new_image_vertical);
            let new_immutable_store = new_image_vertical.to_immutable();
            self.horizontal_filter.filter(&new_immutable_store, into);
        }
        Ok(())
    }

    fn source_rows(&self, mut output_rows: Range<usize>) -> Range<usize> {
        let first = self.vertical_filter.source_bounds(output_rows.start);
        output_rows.start += 1;
        output_rows.fold(first.start..first.start + first.size, |rows, output| {
            let bounds = self.vertical_filter.source_bounds(output);
            rows.start.min(bounds.start)..rows.end.max(bounds.start + bounds.size)
        })
    }

    fn bands(&self) -> impl Iterator<Item = Range<usize>> {
        let height = self.alpha.target_size.height;
        (0..height)
            .step_by(ALPHA_BAND_ROWS)
            .map(move |start| start..(start + ALPHA_BAND_ROWS).min(height))
    }
}

impl<T: Copy + Send + Sync + Debug + Default + 'static, const N: usize>
    AlphaBothAxesConvolvePlan<T, N>
where
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
{
    fn plan_band_scratch_size(&self) -> Option<usize> {
        if N == 4
            && (TypeId::of::<T>() == TypeId::of::<u8>()
                || TypeId::of::<T>() == TypeId::of::<u16>()
                || TypeId::of::<T>() == TypeId::of::<f32>())
            && self.streaming
            && self.alpha.needs_alpha_forward
            && self.alpha.needs_alpha_backward
            && self.alpha.target_size.width < self.alpha.source_size.width
            && self.alpha.target_size.height < self.alpha.source_size.height
            && self.alpha.target_size.height > ALPHA_BAND_ROWS
        {
            let band_height = self
                .bands()
                .map(|rows| self.source_rows(rows).len())
                .max()
                .unwrap_or(0);
            return Some(band_height * self.alpha.source_size.width * N);
        }
        None
    }

    fn resample_in_bands(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        scratch: Option<&mut [T]>,
        band_size: usize,
    ) -> Result<(), PicScaleError> {
        validate_sizes!(store, into, self.alpha.source_size, self.alpha.target_size);

        let filter_size = self.filter_scratch_size();
        let scratch = match scratch {
            Some(scratch) => Some(validate_scratch!(scratch, band_size + filter_size)),
            None => None,
        };
        if into.should_have_bit_depth() && !(1..=16).contains(&into.bit_depth) {
            return Err(PicScaleError::UnsupportedBitDepth(into.bit_depth));
        }
        let premultiply = store.is_alpha_premultiplication_needed();
        let mut allocated_scratch;
        let (scratch, alpha_size) = match scratch {
            Some(scratch) => (scratch, band_size),
            None => {
                let alpha_size = if premultiply { band_size } else { 0 };
                allocated_scratch = try_vec![T::default(); alpha_size + filter_size];
                (allocated_scratch.as_mut_slice(), alpha_size)
            }
        };
        let (alpha, intermediate) = scratch.split_at_mut(alpha_size);
        if !premultiply {
            return self.filter(store, into, intermediate);
        }

        let row_len = store.width * N;
        let pool = self
            .alpha
            .threading_policy
            .get_nova_pool(self.alpha.target_size);
        let mut previous: Range<usize> = 0..0;

        for rows in self.bands() {
            let band = self.source_rows(rows.clone());
            let mut next = band.start;
            if previous.contains(&band.start) {
                // Keep rows in the overlap so each source row is premultiplied once.
                next = previous.end.min(band.end);
                if band.start != previous.start {
                    let kept = (band.start - previous.start)..(next - previous.start);
                    alpha.copy_within(kept.start * row_len..kept.end * row_len, 0);
                }
            }
            if next < band.end {
                let input = store.crop(0, next, store.width, band.end - next)?;
                let start = (next - band.start) * row_len;
                let end = band.len() * row_len;
                premultiply_alpha(&input, &mut alpha[start..end], into.bit_depth, &pool)?;
            }

            let source = ColumnSource {
                pixels: &alpha[..band.len() * row_len],
                width: store.width,
                stride: row_len,
                first_row: band.start,
            };
            self.trampoline_filter
                .filter_rows(&source, into, intermediate, rows);
            previous = band;
        }

        into.unpremultiply_alpha(&pool, self.alpha.workload_strategy);
        Ok(())
    }
}

impl<T: Copy + Send + Sync + Clone + Debug + Default + 'static, const N: usize> ResamplingPlan<T, N>
    for AlphaBothAxesConvolvePlan<T, N>
where
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
{
    fn resample(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
    ) -> Result<(), PicScaleError> {
        if let Some(band_size) = self.band_scratch_size {
            return self.resample_in_bands(store, into, None, band_size);
        }
        self.alpha.resample(
            store,
            into,
            None,
            self.filter_scratch_size(),
            |source, into, scratch| self.filter(source, into, scratch),
        )
    }

    fn resample_with_scratch(
        &self,
        store: &ImageStore<'_, T, N>,
        into: &mut ImageStoreMut<'_, T, N>,
        scratch: &mut [T],
    ) -> Result<(), PicScaleError> {
        if let Some(band_size) = self.band_scratch_size {
            return self.resample_in_bands(store, into, Some(scratch), band_size);
        }
        self.alpha.resample(
            store,
            into,
            Some(scratch),
            self.filter_scratch_size(),
            |source, into, scratch| self.filter(source, into, scratch),
        )
    }

    fn alloc_scratch(&self) -> Vec<T> {
        vec![T::default(); self.scratch_size()]
    }

    fn scratch_size(&self) -> usize {
        self.band_scratch_size
            .unwrap_or_else(|| self.alpha.alpha_scratch_size::<N>())
            + self.filter_scratch_size()
    }

    fn target_size(&self) -> ImageSize {
        self.alpha.target_size
    }

    fn source_size(&self) -> ImageSize {
        self.alpha.source_size
    }
}

pub(crate) fn make_alpha_plan<T, const N: usize>(
    source_size: ImageSize,
    destination_size: ImageSize,
    horizontal_plan: Option<Arc<dyn RowFilter<T, N> + Send + Sync>>,
    vertical_plan: Option<Arc<dyn ColumnFilter<T, N> + Send + Sync>>,
    trampoline_filter: Option<Arc<TrampolineFiltering<T, N>>>,
    threading_policy: ThreadingPolicy,
    workload_strategy: WorkloadStrategy,
    needs_alpha_forward: bool,
    needs_alpha_backward: bool,
) -> Arc<Resampling<T, N>>
where
    T: Copy + Send + Sync + Clone + Debug + Default + 'static,
    for<'a> ImageStore<'a, T, N>: AssociateAlpha<T, N>,
    for<'a> ImageStoreMut<'a, T, N>: CheckStoreDensity + UnassociateAlpha<T, N>,
{
    let should_do_vertical = source_size.height != destination_size.height;
    let should_do_horizontal = source_size.width != destination_size.width;

    let alpha = AlphaPlan {
        source_size,
        target_size: destination_size,
        threading_policy,
        workload_strategy,
        needs_alpha_forward,
        needs_alpha_backward,
    };

    match (should_do_vertical, should_do_horizontal) {
        (true, true) => {
            let mut plan = AlphaBothAxesConvolvePlan {
                alpha,
                horizontal_filter: horizontal_plan.expect("Horizontal plan is expected"),
                vertical_filter: vertical_plan.expect("Vertical plan is expected"),
                trampoline_filter: trampoline_filter.expect("Trampoline filter plan is expected"),
                streaming: threading_policy.is_single_worker(destination_size),
                band_scratch_size: None,
            };
            plan.band_scratch_size = plan.plan_band_scratch_size();
            Arc::new(plan)
        }
        (true, false) => Arc::new(AlphaVerticalConvolvePlan {
            alpha,
            vertical_filter: vertical_plan.expect("Vertical plan is expected"),
        }),
        (false, true) => Arc::new(AlphaHorizontalConvolvePlan {
            alpha,
            horizontal_filter: horizontal_plan.expect("Horizontal plan is expected"),
        }),
        (false, false) => Arc::new(AlphaVerticalConvolvePlan {
            alpha,
            vertical_filter: vertical_plan.expect("Vertical plan is expected"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ResamplingFunction, Scaler};

    #[test]
    fn banded_rgba8_reuses_scratch_with_padded_rows() {
        for (source_size, target_size, strategy) in [
            (
                ImageSize::new(64, 70),
                ImageSize::new(32, 35),
                WorkloadStrategy::PreferSpeed,
            ),
            (
                ImageSize::new(67, 193),
                ImageSize::new(19, 33),
                WorkloadStrategy::PreferQuality,
            ),
        ] {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_workload_strategy(strategy);
            let plan = scaler
                .plan_rgba_resampling(source_size, target_size, true)
                .unwrap();
            let reference_plan = scaler
                .plan_rgba_resampling(source_size, target_size, false)
                .unwrap();
            let pool = ThreadingPolicy::Single.get_nova_pool(target_size);
            let full_alpha_size = source_size.width * source_size.height * 4;
            let required = plan.scratch_size();
            assert_eq!(plan.alloc_scratch().len(), required);
            assert!(required < full_alpha_size + reference_plan.scratch_size());

            let guard = 0x5a;
            let mut scratch = vec![guard; required + 9];
            let src_stride = source_size.width * 4 + 3;
            let len = (source_size.height - 1) * src_stride + source_size.width * 4;
            for (frame, varied_alpha) in [true, false, true].into_iter().enumerate() {
                let mut pixels: Vec<u8> = (0..len)
                    .map(|i| (i * 37 + i / 17 + frame * 29) as u8)
                    .collect();
                if !varied_alpha {
                    for row in pixels.chunks_mut(src_stride) {
                        for pixel in row[..source_size.width * 4].chunks_exact_mut(4) {
                            pixel[3] = 255;
                        }
                    }
                }
                let mut source =
                    ImageStore::from_slice(&pixels, source_size.width, source_size.height).unwrap();
                source.stride = src_stride;

                let stride = target_size.width * 4 + 7;
                let len = (target_size.height - 1) * stride + target_size.width * 4;
                let mut expected = vec![0xa5; len];
                let mut actual = vec![0xa5; len];
                let mut expected_dst =
                    ImageStoreMut::from_slice(&mut expected, target_size.width, target_size.height)
                        .unwrap();
                let mut actual_dst =
                    ImageStoreMut::from_slice(&mut actual, target_size.width, target_size.height)
                        .unwrap();
                expected_dst.stride = stride;
                actual_dst.stride = stride;

                if varied_alpha {
                    let mut full_alpha = vec![0; full_alpha_size];
                    let input = premultiply_alpha(&source, &mut full_alpha, 0, &pool).unwrap();
                    reference_plan.resample(&input, &mut expected_dst).unwrap();
                    expected_dst.unpremultiply_alpha(&pool, strategy);
                } else {
                    reference_plan.resample(&source, &mut expected_dst).unwrap();
                }

                if frame == 0 {
                    plan.resample(&source, &mut actual_dst).unwrap();
                    assert_eq!(actual_dst.projected(), expected_dst.projected());
                    actual_dst.projected().fill(0xa5);
                }
                assert!(matches!(
                    plan.resample_with_scratch(&source, &mut actual_dst, &mut scratch[..required - 1]),
                    Err(PicScaleError::InvalidScratchSize { expected, size })
                        if expected == required && size == required - 1
                ));
                let end = if frame == 0 {
                    required + 1
                } else {
                    scratch.len()
                };
                plan.resample_with_scratch(&source, &mut actual_dst, &mut scratch[1..end])
                    .unwrap();
                assert_eq!(scratch[0], guard);
                assert_eq!(scratch[required + 1..], [guard; 8]);
                assert_eq!(
                    actual, expected,
                    "{source_size:?} → {target_size:?}, frame {frame}"
                );
            }
        }
    }

    #[test]
    fn banded_rgba_u16_reuses_scratch_at_10_and_16_bits() {
        let source_size = ImageSize::new(64, 70);
        let target_size = ImageSize::new(32, 35);
        let src_stride = source_size.width * 4 + 3;
        let src_len = (source_size.height - 1) * src_stride + source_size.width * 4;
        let dst_stride = target_size.width * 4 + 7;
        let dst_len = (target_size.height - 1) * dst_stride + target_size.width * 4;
        let pool = ThreadingPolicy::Single.get_nova_pool(target_size);
        let scaler = Scaler::new(ResamplingFunction::Lanczos3);

        for bit_depth in [10, 16] {
            let max_value = ((1u32 << bit_depth) - 1) as u16;
            let pixels: Vec<u16> = (0..src_len)
                .map(|i| (i * 29819 + i / 17) as u16 & max_value)
                .collect();
            let mut source =
                ImageStore::from_slice(&pixels, source_size.width, source_size.height).unwrap();
            source.stride = src_stride;
            source.bit_depth = bit_depth;
            let mut expected = vec![0xa5; dst_len];
            let mut actual = vec![0xa5; dst_len];
            let mut expected_dst =
                ImageStoreMut::from_slice(&mut expected, target_size.width, target_size.height)
                    .unwrap();
            let mut actual_dst =
                ImageStoreMut::from_slice(&mut actual, target_size.width, target_size.height)
                    .unwrap();
            expected_dst.stride = dst_stride;
            expected_dst.bit_depth = bit_depth;
            actual_dst.stride = dst_stride;
            actual_dst.bit_depth = bit_depth;

            let reference = scaler
                .plan_rgba_resampling16(source_size, target_size, false, bit_depth)
                .unwrap();
            let mut full_alpha = vec![0; source_size.width * source_size.height * 4];
            let input = premultiply_alpha(&source, &mut full_alpha, bit_depth, &pool).unwrap();
            reference.resample(&input, &mut expected_dst).unwrap();
            expected_dst.unpremultiply_alpha(&pool, WorkloadStrategy::PreferSpeed);

            let plan = scaler
                .plan_rgba_resampling16(source_size, target_size, true, bit_depth)
                .unwrap();
            let mut scratch = plan.alloc_scratch();
            assert_eq!(scratch.len(), 46 * source_size.width * 4);
            actual_dst.bit_depth = 17;
            assert!(matches!(
                plan.resample_with_scratch(&source, &mut actual_dst, &mut scratch),
                Err(PicScaleError::UnsupportedBitDepth(17))
            ));
            actual_dst.bit_depth = bit_depth;
            plan.resample_with_scratch(&source, &mut actual_dst, &mut scratch)
                .unwrap();
            assert_eq!(actual, expected, "{bit_depth}-bit");
        }
    }

    #[test]
    fn banded_rgba_f32_preserves_bits_with_padded_rows() {
        let source_size = ImageSize::new(65, 73);
        let target_size = ImageSize::new(31, 27);
        let src_stride = source_size.width * 4 + 5;
        let src_len = (source_size.height - 1) * src_stride + source_size.width * 4;
        let dst_stride = target_size.width * 4 + 11;
        let dst_len = (target_size.height - 1) * dst_stride + target_size.width * 4;
        let samples = [
            0.0f32,
            -0.0,
            f32::from_bits(1),
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::from_bits(0x7fc0_002a),
            f32::from_bits(0xffc0_0013),
            f32::from_bits(0x7f80_0009),
            -4.125,
            0.3,
            1.0,
            4.25,
        ];
        let pixels: Vec<f32> = (0..src_len)
            .map(|i| samples[(i * 3 + i / 7) % samples.len()])
            .collect();
        let mut source =
            ImageStore::from_slice(&pixels, source_size.width, source_size.height).unwrap();
        source.stride = src_stride;
        let pool = ThreadingPolicy::Single.get_nova_pool(target_size);
        let full_alpha_size = source_size.width * source_size.height * 4;
        let mut full_alpha = vec![0.0; full_alpha_size];
        let input = premultiply_alpha(&source, &mut full_alpha, 0, &pool).unwrap();
        let guard = -32.25;

        for strategy in [
            WorkloadStrategy::PreferSpeed,
            WorkloadStrategy::PreferQuality,
        ] {
            let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
            scaler.set_workload_strategy(strategy);
            let reference = scaler
                .plan_rgba_resampling_f32(source_size, target_size, false)
                .unwrap();
            let plan = scaler
                .plan_rgba_resampling_f32(source_size, target_size, true)
                .unwrap();
            let mut scratch = plan.alloc_scratch();
            assert!(scratch.len() < full_alpha_size + reference.scratch_size());

            let mut expected = vec![guard; dst_len];
            let mut expected_dst =
                ImageStoreMut::from_slice(&mut expected, target_size.width, target_size.height)
                    .unwrap();
            expected_dst.stride = dst_stride;
            reference.resample(&input, &mut expected_dst).unwrap();
            expected_dst.unpremultiply_alpha(&pool, strategy);

            for caller_scratch in [true, false] {
                let mut actual = vec![guard; dst_len];
                let mut actual_dst =
                    ImageStoreMut::from_slice(&mut actual, target_size.width, target_size.height)
                        .unwrap();
                actual_dst.stride = dst_stride;
                if caller_scratch {
                    plan.resample_with_scratch(&source, &mut actual_dst, &mut scratch)
                        .unwrap();
                } else {
                    plan.resample(&source, &mut actual_dst).unwrap();
                }
                for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{strategy:?}, index {i}"
                    );
                }
            }
        }
    }
}
