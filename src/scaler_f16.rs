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
use crate::image_store::ImageStoreMut;
use crate::plan::{AlphaPlanner, DefaultPlanner, Resampling};
use crate::scaler::ScalingOptions;
use crate::validation::PicScaleError;
use crate::{
    CbCrF16ImageStore, ImageSize, ImageStoreScaling, PlanarF16ImageStore, RgbF16ImageStore,
    RgbaF16ImageStore, Scaler,
};
use core::f16;
use std::sync::Arc;

/// Implements `f16` type support
#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl Scaler {
    /// Creates a resampling plan for a single-channel (planar/grayscale) `f16` image.
    ///
    /// The `f16` variant of [`plan_planar_resampling`], suitable for half-precision
    /// grayscale content such as HDR render targets or compressed texture data.
    /// Filter weights are accumulated in `f32` to avoid precision loss during convolution.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_planar_resampling_f16(source_size, target_size)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_planar_resampling_f16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<f16, 1>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize::<f16, f32, 1>(self, source_size, target_size, 8)
    }

    /// Creates a resampling plan for a two-channel chroma (`CbCr`) `f16` image.
    ///
    /// The `f16` variant of [`plan_cbcr_resampling`], intended for half-precision chroma
    /// planes of YCbCr content. Both channels are treated as independent signals with no
    /// alpha relationship. Filter weights are accumulated in `f32` to avoid precision
    /// loss during convolution.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input chroma plane.
    /// - `target_size` — Desired dimensions of the output chroma plane.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_cbcr_resampling_f16(source_size, target_size)?;
    /// plan.resample(&cbcr_store, &mut target_cbcr_store)?;
    /// ```
    pub fn plan_cbcr_resampling_f16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<f16, 2>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize::<f16, f32, 2>(self, source_size, target_size, 8)
    }

    /// Creates a resampling plan for a three-channel RGB `f16` image.
    ///
    /// The `f16` variant of [`plan_rgb_resampling`], suitable for half-precision color
    /// images such as HDR render targets or OpenEXR content. All three channels are
    /// resampled independently with no alpha relationship. Filter weights are accumulated
    /// in `f32` to avoid precision loss during convolution.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_rgb_resampling_f16(source_size, target_size)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_rgb_resampling_f16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<f16, 3>>, PicScaleError> {
        DefaultPlanner::plan_generic_resize::<f16, f32, 3>(self, source_size, target_size, 8)
    }

    /// Creates a resampling plan for a four-channel RGBA `f16` image.
    ///
    /// The `f16` variant of [`plan_rgba_resampling`]. Alpha premultiplication is always
    /// applied — RGB channels are pre-multiplied by alpha before resampling and
    /// un-multiplied afterward — regardless of the `premultiply_alpha` flag.
    ///
    /// # Arguments
    ///
    /// - `source_size` — Dimensions of the input image.
    /// - `target_size` — Desired dimensions of the output image.
    /// - `premultiply_alpha` — Whether to premultiply alpha before resampling.
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let plan = scaler.plan_rgba_resampling_f16(source_size, target_size, true)?;
    /// plan.resample(&store, &mut target_store)?;
    /// ```
    pub fn plan_rgba_resampling_f16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<f16, 4>>, PicScaleError> {
        AlphaPlanner::plan_generic_resize_with_alpha::<f16, f32, 4>(
            self,
            source_size,
            target_size,
            8,
            premultiply_alpha,
        )
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl<'b> ImageStoreScaling<'b, f16, 1> for PlanarF16ImageStore<'b> {
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, f16, 1>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError> {
        let scaler =
            Scaler::new(options.resampling_function).set_threading_policy(options.threading_policy);
        let plan = DefaultPlanner::plan_generic_resize(
            &scaler,
            self.size(),
            store.size(),
            store.bit_depth,
        )?;
        plan.resample(self, store)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl<'b> ImageStoreScaling<'b, f16, 2> for CbCrF16ImageStore<'b> {
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, f16, 2>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError> {
        let scaler =
            Scaler::new(options.resampling_function).set_threading_policy(options.threading_policy);
        let plan = DefaultPlanner::plan_generic_resize(
            &scaler,
            self.size(),
            store.size(),
            store.bit_depth,
        )?;
        plan.resample(self, store)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl<'b> ImageStoreScaling<'b, f16, 3> for RgbF16ImageStore<'b> {
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, f16, 3>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError> {
        let scaler =
            Scaler::new(options.resampling_function).set_threading_policy(options.threading_policy);
        let plan = DefaultPlanner::plan_generic_resize(
            &scaler,
            self.size(),
            store.size(),
            store.bit_depth,
        )?;
        plan.resample(self, store)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nightly_f16")))]
impl<'b> ImageStoreScaling<'b, f16, 4> for RgbaF16ImageStore<'b> {
    fn scale(
        &self,
        store: &mut ImageStoreMut<'b, f16, 4>,
        options: ScalingOptions,
    ) -> Result<(), PicScaleError> {
        let scaler =
            Scaler::new(options.resampling_function).set_threading_policy(options.threading_policy);
        let plan = AlphaPlanner::plan_generic_resize_with_alpha(
            &scaler,
            self.size(),
            store.size(),
            store.bit_depth,
            options.premultiply_alpha,
        )?;
        plan.resample(self, store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageStore, ResamplingFunction, ThreadingPolicy};
    use core::f16;

    #[test]
    fn check_rgba_f16_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 4;
        let mut image = vec![0_f16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<CN>().0.iter_mut() {
            dst[0] = (124f32 / 255f32) as f16;
            dst[1] = (41f32 / 255f32) as f16;
            dst[2] = (99f32 / 255f32) as f16;
            dst[3] = 1f32 as f16;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::<f16, 4>::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling_f16(src_store.size(), target_store.size(), false)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        for dst in target_data.chunks_exact(CN) {
            assert!(
                (dst[0] as f32 * 255f32 - 124f32).abs() < 3f32,
                "R channel mismatch: {}",
                dst[0] as f32 * 255f32
            );
            assert!(
                (dst[1] as f32 * 255f32 - 41f32).abs() < 3f32,
                "G channel mismatch: {}",
                dst[1] as f32 * 255f32
            );
            assert!(
                (dst[2] as f32 * 255f32 - 99f32).abs() < 3f32,
                "B channel mismatch: {}",
                dst[2] as f32 * 255f32
            );
            assert!(
                (dst[3] as f32 - 1f32).abs() < 0.01f32,
                "A channel mismatch: {}",
                dst[3] as f32
            );
        }
    }

    #[test]
    fn check_rgba_f16_resizing_vertical_threading() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 4;
        let mut image = vec![0_f16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<CN>().0.iter_mut() {
            dst[0] = (124f32 / 255f32) as f16;
            dst[1] = (41f32 / 255f32) as f16;
            dst[2] = (99f32 / 255f32) as f16;
            dst[3] = 1f32 as f16;
        }
        let scaler = Scaler::new(ResamplingFunction::Lanczos3)
            .set_threading_policy(ThreadingPolicy::Adaptive);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::<f16, 4>::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling_f16(src_store.size(), target_store.size(), false)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        for dst in target_data.chunks_exact(CN) {
            assert!(
                (dst[0] as f32 * 255f32 - 124f32).abs() < 3f32,
                "R channel mismatch: {}",
                dst[0] as f32 * 255f32
            );
            assert!(
                (dst[1] as f32 * 255f32 - 41f32).abs() < 3f32,
                "G channel mismatch: {}",
                dst[1] as f32 * 255f32
            );
            assert!(
                (dst[2] as f32 * 255f32 - 99f32).abs() < 3f32,
                "B channel mismatch: {}",
                dst[2] as f32 * 255f32
            );
            assert!(
                (dst[3] as f32 - 1f32).abs() < 0.01f32,
                "A channel mismatch: {}",
                dst[3] as f32
            );
        }
    }

    #[test]
    fn check_rgba_f16_nearest_vertical() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0_f16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<CN>().0.iter_mut() {
            dst[0] = (124f32 / 255f32) as f16;
            dst[1] = (41f32 / 255f32) as f16;
            dst[2] = (99f32 / 255f32) as f16;
            dst[3] = (77f32 / 255f32) as f16;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Nearest);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::<f16, 4>::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling_f16(src_store.size(), target_store.size(), false)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        for dst in target_data.chunks_exact(CN) {
            assert!(
                (dst[0] as f32 * 255f32 - 124f32).abs() < 3f32,
                "R channel mismatch: {}",
                dst[0] as f32 * 255f32
            );
            assert!(
                (dst[1] as f32 * 255f32 - 41f32).abs() < 3f32,
                "G channel mismatch: {}",
                dst[1] as f32 * 255f32
            );
            assert!(
                (dst[2] as f32 * 255f32 - 99f32).abs() < 3f32,
                "B channel mismatch: {}",
                dst[2] as f32 * 255f32
            );
            assert!(
                (dst[3] as f32 * 255f32 - 77f32).abs() < 3f32,
                "A channel mismatch: {}",
                dst[3] as f32 * 255f32
            );
        }
    }

    #[test]
    fn check_rgba_f16_nearest_vertical_threading() {
        let image_width = 255;
        let image_height = 512;
        const CN: usize = 4;
        let mut image = vec![0_f16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<CN>().0.iter_mut() {
            dst[0] = (124f32 / 255f32) as f16;
            dst[1] = (41f32 / 255f32) as f16;
            dst[2] = (99f32 / 255f32) as f16;
            dst[3] = (77f32 / 255f32) as f16;
        }
        let scaler = Scaler::new(ResamplingFunction::Nearest)
            .set_threading_policy(ThreadingPolicy::Adaptive);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::<f16, 4>::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgba_resampling_f16(src_store.size(), target_store.size(), false)
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        for dst in target_data.chunks_exact(CN) {
            assert!(
                (dst[0] as f32 * 255f32 - 124f32).abs() < 3f32,
                "R channel mismatch: {}",
                dst[0] as f32 * 255f32
            );
            assert!(
                (dst[1] as f32 * 255f32 - 41f32).abs() < 3f32,
                "G channel mismatch: {}",
                dst[1] as f32 * 255f32
            );
            assert!(
                (dst[2] as f32 * 255f32 - 99f32).abs() < 3f32,
                "B channel mismatch: {}",
                dst[2] as f32 * 255f32
            );
            assert!(
                (dst[3] as f32 * 255f32 - 77f32).abs() < 3f32,
                "A channel mismatch: {}",
                dst[3] as f32 * 255f32
            );
        }
    }

    #[test]
    fn check_rgb_f16_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 3;
        let mut image = vec![0_f16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<CN>().0.iter_mut() {
            dst[0] = (124f32 / 255f32) as f16;
            dst[1] = (41f32 / 255f32) as f16;
            dst[2] = (99f32 / 255f32) as f16;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::<f16, 3>::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_rgb_resampling_f16(src_store.size(), target_store.size())
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        for dst in target_data.as_chunks::<CN>().0.iter() {
            assert!(
                (dst[0] as f32 * 255f32 - 124f32).abs() < 3f32,
                "R channel mismatch: {}",
                dst[0] as f32 * 255f32
            );
            assert!(
                (dst[1] as f32 * 255f32 - 41f32).abs() < 3f32,
                "G channel mismatch: {}",
                dst[1] as f32 * 255f32
            );
            assert!(
                (dst[2] as f32 * 255f32 - 99f32).abs() < 3f32,
                "B channel mismatch: {}",
                dst[2] as f32 * 255f32
            );
        }
    }

    #[test]
    fn check_cbcr_f16_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 2;
        let mut image = vec![0_f16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<CN>().0.iter_mut() {
            dst[0] = (124f32 / 255f32) as f16;
            dst[1] = (41f32 / 255f32) as f16;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::<f16, 2>::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_cbcr_resampling_f16(src_store.size(), target_store.size())
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        for dst in target_data.as_chunks::<CN>().0.iter() {
            assert!(
                (dst[0] as f32 * 255f32 - 124f32).abs() < 3f32,
                "R channel mismatch: {}",
                dst[0] as f32 * 255f32
            );
            assert!(
                (dst[1] as f32 * 255f32 - 41f32).abs() < 3f32,
                "G channel mismatch: {}",
                dst[1] as f32 * 255f32
            );
        }
    }

    #[test]
    fn check_planar_f16_resizing_vertical() {
        let image_width = 8;
        let image_height = 8;
        const CN: usize = 1;
        let mut image = vec![0_f16; image_height * image_width * CN];
        for dst in image.as_chunks_mut::<CN>().0.iter_mut() {
            dst[0] = (124f32 / 255f32) as f16;
        }
        let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
        scaler.set_threading_policy(ThreadingPolicy::Single);
        let src_store = ImageStore::from_slice(&image, image_width, image_height).unwrap();
        let mut target_store = ImageStoreMut::<f16, 1>::alloc(image_width, image_height / 2);
        let planned = scaler
            .plan_planar_resampling_f16(src_store.size(), target_store.size())
            .unwrap();
        planned.resample(&src_store, &mut target_store).unwrap();
        let target_data = target_store.buffer.borrow();

        for dst in target_data.as_chunks::<CN>().0.iter() {
            assert!(
                (dst[0] as f32 * 255f32 - 124f32).abs() < 3f32,
                "R channel mismatch: {}",
                dst[0] as f32 * 255f32
            );
        }
    }
}
