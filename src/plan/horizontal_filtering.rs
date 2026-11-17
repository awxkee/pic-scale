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
use crate::convolution::RowFilter;
use crate::filter_weights::FilterWeights;
use crate::{ImageStore, ImageStoreMut, ThreadingPolicy};
#[cfg(feature = "threading")]
use novtb::{ParallelZonedIterator, TbSliceMut};

pub(crate) struct HorizontalFiltering<T, F, const N: usize> {
    pub(crate) filter_4_rows: Option<
        fn(
            src: &[T],
            src_stride: usize,
            dst: &mut [T],
            dst_stride: usize,
            filter_weights: &FilterWeights<F>,
            bit_depth: u32,
        ),
    >,
    pub(crate) filter_row:
        fn(src: &[T], dst: &mut [T], filter_weights: &FilterWeights<F>, bit_depth: u32),
    pub(crate) filter_weights: FilterWeights<F>,
    pub(crate) _threading_policy: ThreadingPolicy,
}

impl<T: Send + Sync, F: Send + Sync, const N: usize> RowFilter<T, N>
    for HorizontalFiltering<T, F, N>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn filter(&self, source: &ImageStore<'_, T, N>, destination: &mut ImageStoreMut<T, N>) {
        #[cfg(feature = "threading")]
        use crate::ImageSize;
        #[cfg(feature = "threading")]
        let pool = self
            ._threading_policy
            .get_nova_pool(ImageSize::new(destination.width, destination.height));

        let src_stride = source.stride();
        let dst_stride = destination.stride();
        let dst_width = destination.width;

        let dst_height = destination.height;
        let dst_bit_depth = destination.bit_depth as u32;
        let dst_buffer = destination.projected();
        let src = source.projected();

        let mut already_processed_y = 0;
        if let Some(dispatcher) = self.filter_4_rows
            && dst_height >= 4
        {
            let groups = dst_height / 4;
            let src_group_stride = src_stride * 4;
            let dst_group_stride = dst_stride * 4;
            let filter_group = |i, dst: &mut [T]| {
                let src_group = &src[i * src_group_stride..];
                dispatcher(
                    &src_group[..src_group.len().min(src_group_stride)],
                    src_stride,
                    dst,
                    dst_stride,
                    &self.filter_weights,
                    dst_bit_depth,
                );
            };

            // Projected images omit padding after the last active row.
            #[cfg(feature = "threading")]
            dst_buffer
                .tb_par_chunks_mut(dst_group_stride)
                .take(groups)
                .for_each_enumerated(&pool, filter_group);
            #[cfg(not(feature = "threading"))]
            dst_buffer
                .chunks_mut(dst_group_stride)
                .take(groups)
                .enumerate()
                .for_each(|(i, dst)| filter_group(i, dst));
            already_processed_y = groups * 4;
        }

        if already_processed_y < dst_height {
            let left_src = &src[already_processed_y * src_stride..];
            let left_dst = &mut dst_buffer[already_processed_y * dst_stride..];
            let one_row_filter = self.filter_row;
            let filter_row = |i, dst: &mut [T]| {
                let src_row = &left_src[i * src_stride..];
                one_row_filter(
                    &src_row[..src_row.len().min(src_stride)],
                    &mut dst[..dst_width * N],
                    &self.filter_weights,
                    dst_bit_depth,
                );
            };

            #[cfg(feature = "threading")]
            left_dst
                .tb_par_chunks_mut(dst_stride)
                .for_each_enumerated(&pool, filter_row);
            #[cfg(not(feature = "threading"))]
            left_dst
                .chunks_mut(dst_stride)
                .enumerate()
                .for_each(|(i, dst)| filter_row(i, dst));
        }
    }

    fn can_do_4_rows(&self) -> bool {
        self.filter_4_rows.is_some()
    }

    fn run_on_4_rows(
        &self,
        src: &[T],
        src_stride: usize,
        dst: &mut [T],
        dst_stride: usize,
        bit_depth: u32,
    ) {
        if let Some(dispatcher) = self.filter_4_rows {
            dispatcher(
                src,
                src_stride,
                dst,
                dst_stride,
                &self.filter_weights,
                bit_depth,
            );
        } else {
            unreachable!(
                "4 rows filter was called where it's not implemented, this is an internal configuration error"
            );
        }
    }

    fn run_on_row(&self, src: &[T], dst: &mut [T], bit_depth: u32) {
        let one_row_filter = self.filter_row;
        one_row_filter(src, dst, &self.filter_weights, bit_depth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ResamplingFunction, Scaler};

    #[test]
    fn horizontal_filter_keeps_final_padded_rows() {
        const SRC_WIDTH: usize = 13;
        const DST_WIDTH: usize = 7;
        const SRC_STRIDE: usize = SRC_WIDTH * 4 + 5;
        const DST_STRIDE: usize = DST_WIDTH * 4 + 7;
        const PAD: u8 = 0xa5;

        for dst_height in [4, 5, 8, 9] {
            for src_height in [dst_height, dst_height + 3] {
                for opaque in [false, true] {
                    let mut pixels: Vec<u8> = (0..(src_height - 1) * SRC_STRIDE + SRC_WIDTH * 4)
                        .map(|i| (i * 19 + i / 3) as u8)
                        .collect();
                    if opaque {
                        for row in pixels.chunks_mut(SRC_STRIDE) {
                            for pixel in row[..SRC_WIDTH * 4].chunks_exact_mut(4) {
                                pixel[3] = 255;
                            }
                        }
                    }
                    let mut src = ImageStore::from_slice(&pixels, SRC_WIDTH, src_height).unwrap();
                    src.stride = SRC_STRIDE;

                    let [single, parallel] = [ThreadingPolicy::Single, ThreadingPolicy::Fixed(2)]
                        .map(|policy| {
                            let mut out = vec![PAD; (dst_height - 1) * DST_STRIDE + DST_WIDTH * 4];
                            let mut dst =
                                ImageStoreMut::from_slice(&mut out, DST_WIDTH, dst_height).unwrap();
                            dst.stride = DST_STRIDE;
                            let plan = Scaler::new(ResamplingFunction::Lanczos3)
                                .set_threading_policy(policy)
                                .plan_rgba_resampling(src.size(), dst.size(), true)
                                .unwrap();
                            plan.resample(&src, &mut dst).unwrap();
                            out
                        });
                    assert_eq!(
                        parallel, single,
                        "{src_height} → {dst_height}, opaque={opaque}"
                    );
                    for row in parallel.chunks(DST_STRIDE) {
                        assert!(row[..DST_WIDTH * 4].iter().any(|&v| v != PAD));
                        assert!(row[DST_WIDTH * 4..].iter().all(|&v| v == PAD));
                    }
                }
            }
        }
    }

    #[test]
    fn horizontal_filter_without_four_row_dispatcher_filters_every_row() {
        let mut src = ImageStore::<u8, 1>::from_slice(&[1, 0, 2, 0, 3, 0, 4, 0, 5], 1, 5).unwrap();
        src.stride = 2;
        let mut data = [9; 9];
        let mut dst = ImageStoreMut::from_slice(&mut data, 1, 5).unwrap();
        dst.stride = 2;
        let filter = HorizontalFiltering {
            filter_4_rows: None,
            filter_row: |src, dst, _, _| dst[0] = src[0],
            filter_weights: FilterWeights::<()>::new(vec![], 0, 0, 0, 0, vec![]),
            _threading_policy: ThreadingPolicy::Fixed(2),
        };
        filter.filter(&src, &mut dst);
        assert_eq!(data, [1, 9, 2, 9, 3, 9, 4, 9, 5]);
    }
}
