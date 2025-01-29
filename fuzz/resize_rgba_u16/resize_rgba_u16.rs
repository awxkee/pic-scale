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
    Ar30ByteOrder, ImageSize, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ScalingU16,
    WorkloadStrategy,
};

fuzz_target!(|data: (u16, u16, u16, u16, bool, bool)| {
    let strategy = if data.5 {
        WorkloadStrategy::PreferQuality
    } else {
        WorkloadStrategy::PreferSpeed
    };
    resize_rgba(
        data.0 as usize,
        data.1 as usize,
        data.2 as usize,
        data.3 as usize,
        ResamplingFunction::Lanczos3,
        data.4,
        strategy,
    )
});

fn resize_rgba(
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    sampler: ResamplingFunction,
    premultiply_alpha: bool,
    workload_strategy: WorkloadStrategy,
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

    let store = ImageStore::<u16, 4>::alloc(src_width, src_height);
    let mut target = ImageStoreMut::alloc_with_depth(dst_width, dst_height, 10);

    let mut scaler = Scaler::new(sampler);
    scaler.set_workload_strategy(workload_strategy);
    scaler
        .resize_rgba_u16(&store, &mut target, premultiply_alpha)
        .unwrap();

    let mut target = ImageStoreMut::alloc_with_depth(dst_width, dst_height, 16);

    let store = ImageStore::<u16, 4>::alloc(src_width, src_height);
    scaler
        .resize_rgba_u16(&store, &mut target, premultiply_alpha)
        .unwrap();

    let src_data_ar30 = vec![1u8; src_width * src_height * 4];
    let mut dst_data_ar30 = vec![1u8; dst_width * dst_height * 4];
    _ = scaler.resize_ar30(
        &src_data_ar30,
        src_width * 4,
        ImageSize::new(src_width, src_height),
        &mut dst_data_ar30,
        dst_height * 4,
        ImageSize::new(dst_width, dst_height),
        Ar30ByteOrder::Host,
    );
}
