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

#[cfg(feature = "nightly_f16")]
use std::arch::aarch64::*;

#[cfg(feature = "nightly_f16")]
#[target_feature(enable = "fp16")]
fn convert_weights_to_f16_impl<J: Default + Clone>(weights: &[f32]) -> Vec<J> {
    unsafe {
        let mut new_weights = vec![J::default(); weights.len()];

        for (dst, src) in new_weights
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(weights.as_chunks::<8>().0.iter())
        {
            let j = vld1q_f32_x2(src.as_ptr());
            let cvt0 = vcvt_f16_f32(j.0);
            let cvt1 = vcvt_f16_f32(j.1);
            vst1q_u16(
                dst.as_mut_ptr() as *mut u16,
                vreinterpretq_u16_f16(vcombine_f16(cvt0, cvt1)),
            );
        }

        let dst = new_weights.as_chunks_mut::<8>().1;
        let src = weights.as_chunks::<8>().1;

        for (dst, src) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(src.as_chunks::<4>().0.iter())
        {
            let j = vld1q_f32(src.as_ptr());
            let cvt = vcvt_f16_f32(j);
            vst1_u16(dst.as_mut_ptr() as *mut u16, vreinterpret_u16_f16(cvt));
        }

        let dst = dst.as_chunks_mut::<4>().1;
        let src = src.as_chunks::<4>().1;

        for (dst, src) in dst.as_chunks_mut::<1>().0.iter_mut().zip(src.iter()) {
            let j = vcvt_f16_f32(vld1q_lane_f32::<0>(src, vdupq_n_f32(0.)));
            vst1_lane_u16::<0>(dst.as_mut_ptr() as *mut u16, vreinterpret_u16_f16(j));
        }

        new_weights
    }
}

#[cfg(feature = "nightly_f16")]
use core::f16;

#[cfg(feature = "nightly_f16")]
pub(crate) fn convert_weights_to_f16_fhm(weights: &[f32]) -> Vec<f16> {
    unsafe { convert_weights_to_f16_impl(weights) }
}
