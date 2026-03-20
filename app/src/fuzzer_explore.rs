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
use pic_scale::{
    ImageSize, ImageStore, ImageStoreMut, JzazbzScaler, LChScaler, LabScaler, LinearApproxScaler,
    LinearScaler, LuvScaler, OklabScaler, PicScaleError, Resampling, ResamplingFunction,
    SigmoidalScaler, TransferFunction, XYZScaler,
};
use std::sync::Arc;

enum Scaler {
    Jzazbz(JzazbzScaler),
    Lab(LabScaler),
    LCh(LChScaler),
    Linear(LinearScaler),
    LinearApprox(LinearApproxScaler),
    Luv(LuvScaler),
    Oklab(OklabScaler),
    Sigmoidal(SigmoidalScaler),
    XYZ(XYZScaler),
}

impl Scaler {
    fn plan_rgba(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
        premultiply_alpha: bool,
    ) -> Result<Arc<Resampling<u8, 4>>, PicScaleError> {
        match self {
            Scaler::Jzazbz(s) => {
                s.plan_rgba_resampling(source_size, target_size, premultiply_alpha)
            }
            Scaler::Lab(s) => s.plan_rgba_resampling(source_size, target_size, premultiply_alpha),
            Scaler::LCh(s) => s.plan_rgba_resampling(source_size, target_size, premultiply_alpha),
            Scaler::Linear(s) => {
                s.plan_rgba_resampling(source_size, target_size, premultiply_alpha)
            }
            Scaler::LinearApprox(s) => {
                s.plan_rgba_resampling(source_size, target_size, premultiply_alpha)
            }
            Scaler::Luv(s) => s.plan_rgba_resampling(source_size, target_size, premultiply_alpha),
            Scaler::Oklab(s) => s.plan_rgba_resampling(source_size, target_size, premultiply_alpha),
            Scaler::Sigmoidal(s) => {
                s.plan_rgba_resampling(source_size, target_size, premultiply_alpha)
            }
            Scaler::XYZ(s) => s.plan_rgba_resampling(source_size, target_size, premultiply_alpha),
        }
    }

    fn plan_rgb(
        &self,
        source_size: ImageSize,
        target_size: ImageSize,
    ) -> Result<Arc<Resampling<u8, 3>>, PicScaleError> {
        match self {
            Scaler::Jzazbz(s) => s.plan_rgb_resampling(source_size, target_size),
            Scaler::Lab(s) => s.plan_rgb_resampling(source_size, target_size),
            Scaler::LCh(s) => s.plan_rgb_resampling(source_size, target_size),
            Scaler::Linear(s) => s.plan_rgb_resampling(source_size, target_size),
            Scaler::LinearApprox(s) => s.plan_rgb_resampling(source_size, target_size),
            Scaler::Luv(s) => s.plan_rgb_resampling(source_size, target_size),
            Scaler::Oklab(s) => s.plan_rgb_resampling(source_size, target_size),
            Scaler::Sigmoidal(s) => s.plan_rgb_resampling(source_size, target_size),
            Scaler::XYZ(s) => s.plan_rgb_resampling(source_size, target_size),
        }
    }
}

pub(crate) fn resize_rgba(
    data: u8,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    sampler: ResamplingFunction,
    mul_alpha: bool,
) {
    if src_width == 0
        || src_width > 2000
        || src_height == 0
        || src_height > 2000
        || dst_width == 0
        || dst_width > 512
        || dst_height == 0
        || dst_height > 512
    {
        return;
    }

    let scalers = vec![
        Scaler::Jzazbz(JzazbzScaler::new(sampler, 203f32, TransferFunction::Srgb)),
        Scaler::Lab(LabScaler::new(sampler)),
        Scaler::LCh(LChScaler::new(sampler)),
        Scaler::Linear(LinearScaler::new(sampler)),
        Scaler::LinearApprox(LinearApproxScaler::new(sampler)),
        Scaler::Luv(LuvScaler::new(sampler)),
        Scaler::Oklab(OklabScaler::new(sampler, TransferFunction::Srgb)),
        Scaler::Sigmoidal(SigmoidalScaler::new(sampler)),
        Scaler::XYZ(XYZScaler::new(sampler)),
    ];

    let source_size = ImageSize::new(src_width, src_height);
    let target_size = ImageSize::new(dst_width, dst_height);

    for scaler in &scalers {
        if mul_alpha {
            let plan = scaler.plan_rgba(source_size, target_size, true).unwrap();
            let mut src_data_rgba = vec![data; src_width * src_height * 4];
            src_data_rgba[3] = 18;
            let store =
                ImageStore::<u8, 4>::from_slice(&mut src_data_rgba, src_width, src_height).unwrap();
            let mut target_store = ImageStoreMut::alloc(dst_width, dst_height);
            plan.resample(&store, &mut target_store).unwrap();
        } else {
            let plan = scaler.plan_rgb(source_size, target_size).unwrap();
            let mut src_data_rgb = vec![data; src_width * src_height * 3];
            let store =
                ImageStore::<u8, 3>::from_slice(&mut src_data_rgb, src_width, src_height).unwrap();
            let mut target_store = ImageStoreMut::alloc(dst_width, dst_height);
            plan.resample(&store, &mut target_store).unwrap();
        }
    }
}
