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
use crate::colors::common_splitter::{SplitPlanInterceptor, Splitter};
use crate::mixed_storage::CpuRound;
use crate::plan::Resampling;
use crate::validation::PicScaleError;
use crate::{ImageSize, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy};
use colorutils_rs::TransferFunction;
use std::sync::Arc;

#[derive(Debug, Copy, Clone)]
/// Converts image to linear f32 components scales it and convert back.
///
/// This is more precise and slower than scaling in linear fixed point.
///
/// This is an expensive method, and its precision might not be required,
/// consider to use [LinearApproxScaler] instead
pub struct LinearScaler {
    pub(crate) scaler: Scaler,
    pub(crate) transfer_function: TransferFunction,
}

impl LinearScaler {
    /// Creates new instance with sRGB transfer function
    pub fn new(filter: ResamplingFunction) -> Self {
        LinearScaler {
            scaler: Scaler::new(filter),
            transfer_function: TransferFunction::Srgb,
        }
    }

    /// Creates new instance with requested transfer function
    pub fn new_with_transfer(
        filter: ResamplingFunction,
        transfer_function: TransferFunction,
    ) -> Self {
        LinearScaler {
            scaler: Scaler::new(filter),
            transfer_function,
        }
    }
}

struct Common8BitSplitter<const N: usize> {
    linearization: Box<[f32; 256]>,
    gamma: Box<[u8; 65536]>,
    has_alpha: bool,
}

impl<const N: usize> Splitter<u8, f32, N> for Common8BitSplitter<N> {
    fn split(&self, from: &ImageStore<'_, u8, N>, into: &mut ImageStoreMut<'_, f32, N>) {
        const S: f32 = 1. / 255.;
        if N == 4 {
            for (src, dst) in from
                .as_bytes()
                .chunks_exact(4)
                .zip(into.buffer.borrow_mut().chunks_exact_mut(4))
            {
                dst[0] = self.linearization[src[0] as usize];
                dst[1] = self.linearization[src[1] as usize];
                dst[2] = self.linearization[src[2] as usize];
                dst[3] = src[3] as f32 * S;
            }
        } else if N == 2 && self.has_alpha {
            for (src, dst) in from
                .as_bytes()
                .chunks_exact(2)
                .zip(into.buffer.borrow_mut().chunks_exact_mut(2))
            {
                dst[0] = self.linearization[src[0] as usize];
                dst[1] = src[1] as f32 * S;
            }
        } else {
            for (&src, dst) in from.as_bytes().iter().zip(into.buffer.borrow_mut()) {
                *dst = self.linearization[src as usize];
            }
        }
    }

    fn merge(&self, from: &ImageStore<'_, f32, N>, into: &mut ImageStoreMut<'_, u8, N>) {
        if N == 4 {
            for (src, dst) in from
                .as_bytes()
                .chunks_exact(4)
                .zip(into.buffer.borrow_mut().chunks_exact_mut(4))
            {
                let v0 = (src[0] * 65535.).cpu_round() as u16;
                let v1 = (src[1] * 65535.).cpu_round() as u16;
                let v2 = (src[2] * 65535.).cpu_round() as u16;

                dst[0] = self.gamma[v0 as usize];
                dst[1] = self.gamma[v1 as usize];
                dst[2] = self.gamma[v2 as usize];
                dst[3] = (src[3] * 255.).cpu_round() as u8;
            }
        } else if N == 2 && self.has_alpha {
            for (src, dst) in from
                .as_bytes()
                .chunks_exact(2)
                .zip(into.buffer.borrow_mut().chunks_exact_mut(2))
            {
                let v0 = (src[0] * 65535.).cpu_round() as u16;

                dst[0] = self.gamma[v0 as usize];
                dst[1] = (src[1] * 255.).cpu_round() as u8;
            }
        } else {
            for (&src, dst) in from.as_bytes().iter().zip(into.buffer.borrow_mut()) {
                let v = (src * 65535.).cpu_round() as u16;
                *dst = self.gamma[v as usize];
            }
        }
    }
}

struct Common16BitSplitter<const N: usize> {
    linearization: Box<[f32; 65536]>,
    gamma: Box<[u16; 262144]>,
    has_alpha: bool,
    bit_depth: usize,
}

impl<const N: usize> Splitter<u16, f32, N> for Common16BitSplitter<N> {
    fn split(&self, from: &ImageStore<'_, u16, N>, into: &mut ImageStoreMut<'_, f32, N>) {
        if N == 4 {
            let max_bit_depth_value = ((1u32 << self.bit_depth) - 1) as f32;

            let v_recip = 1. / max_bit_depth_value;

            for (src, dst) in from
                .as_bytes()
                .chunks_exact(4)
                .zip(into.buffer.borrow_mut().chunks_exact_mut(4))
            {
                dst[0] = self.linearization[src[0] as usize];
                dst[1] = self.linearization[src[1] as usize];
                dst[2] = self.linearization[src[2] as usize];
                dst[3] = src[3] as f32 * v_recip;
            }
        } else if N == 2 && self.has_alpha {
            let max_bit_depth_value = ((1u32 << self.bit_depth) - 1) as f32;
            let v_recip = 1. / max_bit_depth_value;

            for (src, dst) in from
                .as_bytes()
                .chunks_exact(2)
                .zip(into.buffer.borrow_mut().chunks_exact_mut(2))
            {
                dst[0] = self.linearization[src[0] as usize];
                dst[1] = src[1] as f32 * v_recip;
            }
        } else {
            for (&src, dst) in from.as_bytes().iter().zip(into.buffer.borrow_mut()) {
                *dst = self.linearization[src as usize];
            }
        }
    }

    fn merge(&self, from: &ImageStore<'_, f32, N>, into: &mut ImageStoreMut<'_, u16, N>) {
        if N == 4 {
            let max_bit_depth_value = ((1u32 << self.bit_depth) - 1) as f32;
            for (src, dst) in from
                .as_bytes()
                .chunks_exact(4)
                .zip(into.buffer.borrow_mut().chunks_exact_mut(4))
            {
                let v0 = ((src[0] * 262143.).cpu_round().max(0.) as u32).min(262143);
                let v1 = ((src[1] * 262143.).cpu_round().max(0.) as u32).min(262143);
                let v2 = ((src[2] * 262143.).cpu_round().max(0.) as u32).min(262143);

                dst[0] = self.gamma[v0 as usize];
                dst[1] = self.gamma[v1 as usize];
                dst[2] = self.gamma[v2 as usize];
                dst[3] = (src[3] * max_bit_depth_value)
                    .cpu_round()
                    .min(max_bit_depth_value)
                    .max(0.) as u16;
            }
        } else if N == 2 && self.has_alpha {
            let max_bit_depth_value = ((1u32 << self.bit_depth) - 1) as f32;
            for (src, dst) in from
                .as_bytes()
                .chunks_exact(2)
                .zip(into.buffer.borrow_mut().chunks_exact_mut(2))
            {
                let v0 = ((src[0] * 262143.).cpu_round().max(0.) as u32).min(262143);

                dst[0] = self.gamma[v0 as usize];
                dst[1] = (src[1] * max_bit_depth_value)
                    .cpu_round()
                    .min(max_bit_depth_value)
                    .max(0.) as u16;
            }
        } else {
            for (&src, dst) in from.as_bytes().iter().zip(into.buffer.borrow_mut()) {
                let v = ((src * 262143.).cpu_round().max(0.) as u32).min(262143);
                *dst = self.gamma[v as usize];
            }
        }
    }
}

struct Linearization {
    linearization: Box<[f32; 256]>,
    gamma: Box<[u8; 65536]>,
}

struct Linearization16 {
    linearization: Box<[f32; 65536]>,
    gamma: Box<[u16; 262144]>,
}

fn make_linearization16(
    transfer_function: TransferFunction,
    bit_depth: usize,
) -> Result<Linearization16, PicScaleError> {
    if bit_depth < 8 {
        return Err(PicScaleError::UnsupportedBitDepth(bit_depth));
    }
    let mut linearizing = Box::new([0f32; 65536]);
    let max_lin_depth = (1u32 << bit_depth) - 1;
    let keep_max = 1u32 << bit_depth;
    let mut gamma = Box::new([0u16; 262144]);

    let rcp_bit_depth = 1. / max_lin_depth as f32;

    for (i, dst) in linearizing.iter_mut().take(keep_max as usize).enumerate() {
        *dst = transfer_function.linearize(i as f32 * rcp_bit_depth);
    }

    const RCP_LUT_CAP: f32 = 1. / 262143.;

    for (i, dst) in gamma.iter_mut().enumerate() {
        *dst = (transfer_function.gamma(i as f32 * RCP_LUT_CAP) * max_lin_depth as f32)
            .cpu_round()
            .min(max_lin_depth as f32) as u16;
    }

    Ok(Linearization16 {
        linearization: linearizing,
        gamma,
    })
}

fn make_linearization(transfer_function: TransferFunction) -> Linearization {
    let mut linearizing = Box::new([0f32; 256]);
    let mut gamma = Box::new([0u8; 65536]);

    const S: f32 = 1. / 255.;

    for (i, dst) in linearizing.iter_mut().enumerate() {
        *dst = transfer_function.linearize(i as f32 * S);
    }

    const RCP_LUT_CAP: f32 = 1. / 65535.;

    for (i, dst) in gamma.iter_mut().enumerate() {
        *dst = (transfer_function.gamma(i as f32 * RCP_LUT_CAP) * 255.)
            .cpu_round()
            .min(255.) as u8;
    }

    Linearization {
        linearization: linearizing,
        gamma,
    }
}

impl LinearScaler {
    pub fn set_threading_policy(&mut self, threading_policy: ThreadingPolicy) -> Self {
        self.scaler.threading_policy = threading_policy;
        *self
    }

    pub fn plan_planar_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 1>>, PicScaleError> {
        let intercept = self
            .scaler
            .plan_planar_resampling_f32(source_size, target_size)?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization(self.transfer_function);
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common8BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: false,
            }),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_cbcr_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 2>>, PicScaleError> {
        let intercept = self
            .scaler
            .plan_cbcr_resampling_f32(source_size, target_size)?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization(self.transfer_function);
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common8BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: false,
            }),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_gray_alpha_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<u8, 2>>, PicScaleError> {
        let intercept = self.scaler.plan_gray_alpha_resampling_f32(
            source_size,
            target_size,
            premultiply_alpha,
        )?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization(self.transfer_function);
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common8BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: true,
            }),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_rgb_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 3>>, PicScaleError> {
        let intercept = self
            .scaler
            .plan_rgb_resampling_f32(source_size, target_size)?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization(self.transfer_function);
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common8BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: false,
            }),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_rgba_resampling(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<u8, 4>>, PicScaleError> {
        let intercept =
            self.scaler
                .plan_rgba_resampling_f32(source_size, target_size, premultiply_alpha)?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization(self.transfer_function);
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common8BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: true,
            }),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_planar_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 1>>, PicScaleError> {
        let intercept = self
            .scaler
            .plan_planar_resampling_f32(source_size, target_size)?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization16(self.transfer_function, bit_depth)?;
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common16BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: false,
                bit_depth,
            }),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_gray_alpha_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 2>>, PicScaleError> {
        let intercept = self.scaler.plan_gray_alpha_resampling_f32(
            source_size,
            target_size,
            premultiply_alpha,
        )?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization16(self.transfer_function, bit_depth)?;
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common16BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: true,
                bit_depth,
            }),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_cbcr_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 2>>, PicScaleError> {
        let intercept = self
            .scaler
            .plan_cbcr_resampling_f32(source_size, target_size)?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization16(self.transfer_function, bit_depth)?;
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common16BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: false,
                bit_depth,
            }),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_rgb_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 3>>, PicScaleError> {
        let intercept = self
            .scaler
            .plan_rgb_resampling_f32(source_size, target_size)?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization16(self.transfer_function, bit_depth)?;
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common16BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: false,
                bit_depth,
            }),
            inner_scratch: scratch_size,
        }))
    }

    pub fn plan_rgba_resampling16(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
        bit_depth: usize,
    ) -> Result<Arc<Resampling<u16, 4>>, PicScaleError> {
        let intercept =
            self.scaler
                .plan_rgba_resampling_f32(source_size, target_size, premultiply_alpha)?;
        let scratch_size = intercept.scratch_size();
        let lin = make_linearization16(self.transfer_function, bit_depth)?;
        Ok(Arc::new(SplitPlanInterceptor {
            intercept,
            splitter: Arc::new(Common16BitSplitter {
                linearization: lin.linearization,
                gamma: lin.gamma,
                has_alpha: true,
                bit_depth,
            }),
            inner_scratch: scratch_size,
        }))
    }
}
