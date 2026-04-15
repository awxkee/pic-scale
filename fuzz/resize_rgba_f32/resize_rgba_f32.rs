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
    BufferStore, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy,
    WorkloadStrategy,
};
use rand::RngExt;

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
    resize_rgba(
        data.value as f32 / 65535.,
        data.src_width as usize,
        data.src_height as usize,
        data.dst_width as usize,
        data.dst_height as usize,
        ResamplingFunction::Bilinear,
        data.premultiply_alpha,
        data.use_quality,
        if data.threading {
            ThreadingPolicy::Adaptive
        } else {
            ThreadingPolicy::Single
        },
    )
});

fn resize_rgba(
    data: f32,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    sampler: ResamplingFunction,
    premultiply_alpha: bool,
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

    let src_stride = (src_width + rand::rng().random_range(0..100)) * 4;

    let mut src_data = vec![data; src_stride * src_height];
    src_data[0] = 0.32f32;
    src_data[3] = 0.6543432f32;

    let src_valid_size = src_stride * (src_height - 1) + src_width * 4;

    let store = ImageStore {
        buffer: std::borrow::Cow::Borrowed(&src_data[..src_valid_size]),
        channels: 4,
        width: src_width,
        height: src_height,
        stride: src_stride,
        bit_depth: 8,
    };

    let dst_stride = (dst_width + rand::rng().random_range(0..100)) * 4;
    let mut dst_data_full = vec![0f32; dst_stride * dst_height];

    let dst_valid_size = dst_stride * (dst_height - 1) + dst_width * 4;

    let mut target = ImageStoreMut {
        buffer: BufferStore::Borrowed(&mut dst_data_full[..dst_valid_size]),
        channels: 4,
        width: dst_width,
        height: dst_height,
        stride: dst_stride,
        bit_depth: 8,
    };

    let scaler = Scaler::new(sampler)
        .set_workload_strategy(if use_quality {
            WorkloadStrategy::PreferQuality
        } else {
            WorkloadStrategy::PreferSpeed
        })
        .set_threading_policy(threading_policy);
    let planned = scaler
        .plan_rgba_resampling_f32(store.size(), target.size(), premultiply_alpha)
        .unwrap();
    planned.resample(&store, &mut target).unwrap();
}
