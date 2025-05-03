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

#![no_main]

use libfuzzer_sys::fuzz_target;
use pic_scale::{
    ImageStore, ImageStoreMut, JzazbzScaler, LChScaler, LabScaler, LinearApproxScaler,
    LinearScaler, LuvScaler, OklabScaler, ResamplingFunction, Scaling, SigmoidalScaler,
    TransferFunction, XYZScaler,
};

fuzz_target!(|data: (u16, u16, u16, u16, u8)| {
    resize_rgba(
        data.4,
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        data.3 as usize,
        ResamplingFunction::Bilinear,
    )
});

fn resize_rgba(
    data: u8,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    sampler: ResamplingFunction,
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

    let scalers: Vec<Box<dyn Scaling>> = vec![
        Box::new(JzazbzScaler::new(sampler, 203f32, TransferFunction::Srgb)),
        Box::new(LabScaler::new(sampler)),
        Box::new(LChScaler::new(sampler)),
        Box::new(LinearScaler::new(sampler)),
        Box::new(LinearApproxScaler::new(sampler)),
        Box::new(LuvScaler::new(sampler)),
        Box::new(OklabScaler::new(sampler, TransferFunction::Srgb)),
        Box::new(SigmoidalScaler::new(sampler)),
        Box::new(XYZScaler::new(sampler)),
    ];

    for scaler in scalers {
        let mut src_data_rgb = vec![data; src_width * src_height * 3];
        let store =
            ImageStore::<u8, 3>::from_slice(&mut src_data_rgb, src_width, src_height).unwrap();
        let mut target_store = ImageStoreMut::alloc(dst_width, dst_height);
        scaler.resize_rgb(&store, &mut target_store).unwrap();

        let mut src_data_rgba = vec![data; src_width * src_height * 4];
        src_data_rgba[3] = 18;
        let store_rgba =
            ImageStore::<u8, 4>::from_slice(&mut src_data_rgba, src_width, src_height).unwrap();
        let mut target_store_rgba = ImageStoreMut::alloc(dst_width, dst_height);
        scaler
            .resize_rgba(&store_rgba, &mut target_store_rgba, false)
            .unwrap();

        scaler
            .resize_rgba(&store_rgba, &mut target_store_rgba, true)
            .unwrap();
    }
}
