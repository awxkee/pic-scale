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

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pic_scale::{
    ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ScalingF32, ThreadingPolicy,
    WorkloadStrategy,
};

#[derive(Clone, Debug, Arbitrary)]
pub struct SrcImage {
    pub src_width: u16,
    pub src_height: u16,
    pub dst_width: u16,
    pub dst_height: u16,
    pub value: u16,
    pub use_quality: bool,
    pub threading: bool,
}

fuzz_target!(|data: SrcImage| {
    resize_plane(
        data.src_width as usize,
        data.src_height as usize,
        data.dst_width as usize,
        data.dst_height as usize,
        ResamplingFunction::Bilinear,
        data.value as f32 / 65535.,
        data.use_quality,
        if data.threading {
            ThreadingPolicy::Adaptive
        } else {
            ThreadingPolicy::Single
        },
    )
});

fn resize_plane(
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    sampler: ResamplingFunction,
    value: f32,
    use_quality: bool,
    threading_policy: ThreadingPolicy,
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

    let mut src_data = vec![value; src_width * src_height];

    let store = ImageStore::<f32, 1>::from_slice(&mut src_data, src_width, src_height).unwrap();
    let mut target = ImageStoreMut::alloc(dst_width, dst_height);

    let mut scaler = Scaler::new(sampler);
    scaler.set_workload_strategy(if use_quality {
        WorkloadStrategy::PreferQuality
    } else {
        WorkloadStrategy::PreferSpeed
    });
    scaler.set_threading_policy(threading_policy);
    scaler.resize_plane_f32(&store, &mut target).unwrap();
}
