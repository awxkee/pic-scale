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
    Ar30ByteOrder, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, Scaling, ScalingU16,
    ThreadingPolicy, WorkloadStrategy,
};

#[derive(Clone, Debug, Arbitrary)]
pub struct SrcImage {
    pub src_width: u16,
    pub src_height: u16,
    pub dst_width: u16,
    pub dst_height: u16,
    pub value: u16,
    pub premultiply_alpha: bool,
    pub use_quality: bool,
    pub threading: bool,
}

fuzz_target!(|data: SrcImage| {
    let strategy = if data.use_quality {
        WorkloadStrategy::PreferQuality
    } else {
        WorkloadStrategy::PreferSpeed
    };
    resize_rgba(
        data.value,
        data.src_width as usize,
        data.src_height as usize,
        data.dst_width as usize,
        data.dst_height as usize,
        ResamplingFunction::Lanczos3,
        data.premultiply_alpha,
        strategy,
        if data.threading {
            ThreadingPolicy::Adaptive
        } else {
            ThreadingPolicy::Single
        },
    )
});

fn resize_rgba(
    data: u16,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    sampler: ResamplingFunction,
    premultiply_alpha: bool,
    workload_strategy: WorkloadStrategy,
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

    let mut src_data = vec![data; src_width * src_height * 4];
    src_data[0] = 255;
    src_data[3] = 17;

    let store = ImageStore::<u16, 4>::borrow(&src_data, src_width, src_height).unwrap();

    let mut target = ImageStoreMut::alloc_with_depth(dst_width, dst_height, 10);

    let mut scaler = Scaler::new(sampler);
    scaler.set_workload_strategy(workload_strategy);
    scaler.set_threading_policy(threading_policy);
    scaler
        .resize_rgba_u16(&store, &mut target, premultiply_alpha)
        .unwrap();

    let mut target = ImageStoreMut::alloc_with_depth(dst_width, dst_height, 16);

    let store = ImageStore::<u16, 4>::borrow(&src_data, src_width, src_height).unwrap();
    scaler
        .resize_rgba_u16(&store, &mut target, premultiply_alpha)
        .unwrap();

    let src_data2 = vec![data.min(255) as u8; src_width * src_height * 4];
    let store_ar30 = ImageStore::<u8, 4>::borrow(&src_data2, src_width, src_height).unwrap();
    let mut target_ar30 = ImageStoreMut::alloc_with_depth(dst_width, dst_height, 10);
    _ = scaler.resize_ar30(&store_ar30, &mut target_ar30, Ar30ByteOrder::Host);
}
