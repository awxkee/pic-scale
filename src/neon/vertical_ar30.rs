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
use crate::filter_weights::FilterBounds;
use crate::neon::ar30::{vunzip_3_ar30, vzip_4_ar30};
use crate::neon::utils::{xvld1q_u32_x2, xvst1q_u32_x2};
use std::arch::aarch64::*;

pub(crate) fn neon_column_handler_fixed_point_ar30<
    const AR30_TYPE: usize,
    const AR30_ORDER: usize,
>(
    bounds: &FilterBounds,
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    weight: &[i16],
) {
    unsafe {
        let unit = ExecutionUnit::<AR30_TYPE, AR30_ORDER>::default();
        unit.pass(bounds, src, dst, src_stride, weight);
    }
}

#[derive(Copy, Clone, Default)]
struct ExecutionUnit<const AR30_TYPE: usize, const AR30_ORDER: usize> {}

impl<const AR30_TYPE: usize, const AR30_ORDER: usize> ExecutionUnit<AR30_TYPE, AR30_ORDER> {
    unsafe fn pass(
        &self,
        bounds: &FilterBounds,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        weight: &[i16],
    ) {
        unsafe {
            let mut cx = 0usize;

            let total_width = dst.len() / 4;

            const PREC: i32 = 16;
            const RND_CONST: i32 = (1 << (PREC - 1)) - 1;

            while cx + 8 < total_width {
                let v_max = vdupq_n_u16(1023);
                let filter = weight;
                let v_start_px = cx * 4;

                let mut v0 = vdupq_n_s32(RND_CONST);
                let mut v1 = vdupq_n_s32(RND_CONST);
                let mut v2 = vdupq_n_s32(RND_CONST);
                let mut v3 = vdupq_n_s32(RND_CONST);
                let mut v4 = vdupq_n_s32(RND_CONST);
                let mut v5 = vdupq_n_s32(RND_CONST);

                for (j, &k_weight) in filter.iter().take(bounds.size).enumerate() {
                    let py = bounds.start + j;
                    let weight = vdupq_n_s16(k_weight);
                    let offset = src_stride * py + v_start_px;
                    let src_ptr = src.get_unchecked(offset..(offset + 8 * 4));

                    let ps = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                        src_ptr.as_ptr() as *const _,
                    ));
                    v0 = vqdmlal_s16(v0, vget_low_s16(ps.0), vget_low_s16(weight));
                    v1 = vqdmlal_high_s16(v1, ps.0, weight);
                    v2 = vqdmlal_s16(v2, vget_low_s16(ps.1), vget_low_s16(weight));
                    v3 = vqdmlal_high_s16(v3, ps.1, weight);
                    v4 = vqdmlal_s16(v4, vget_low_s16(ps.2), vget_low_s16(weight));
                    v5 = vqdmlal_high_s16(v5, ps.2, weight);
                }

                let v0 = vqshrun_n_s32::<PREC>(v0);
                let v1 = vqshrun_n_s32::<PREC>(v1);
                let v2 = vqshrun_n_s32::<PREC>(v2);
                let v3 = vqshrun_n_s32::<PREC>(v3);
                let v4 = vqshrun_n_s32::<PREC>(v4);
                let v5 = vqshrun_n_s32::<PREC>(v5);

                let r_v = vminq_u16(vcombine_u16(v0, v1), v_max);
                let g_v = vminq_u16(vcombine_u16(v2, v3), v_max);
                let b_v = vminq_u16(vcombine_u16(v4, v5), v_max);

                let v_dst = dst.get_unchecked_mut(v_start_px..(v_start_px + 8 * 4));

                let vals = vzip_4_ar30::<AR30_TYPE, AR30_ORDER>(int16x8x4_t(
                    vreinterpretq_s16_u16(r_v),
                    vreinterpretq_s16_u16(g_v),
                    vreinterpretq_s16_u16(b_v),
                    vdupq_n_s16(3),
                ));
                xvst1q_u32_x2(v_dst.as_mut_ptr() as *mut _, vals);

                cx += 8;
            }

            if cx < total_width {
                let diff = total_width - cx;

                let mut src_transient: [u8; 4 * 8] = [0; 4 * 8];
                let mut dst_transient: [u8; 4 * 8] = [0; 4 * 8];

                let v_max = vdupq_n_u16(1023);
                let filter = weight;
                let v_start_px = cx * 4;

                let mut v0 = vdupq_n_s32(RND_CONST);
                let mut v1 = vdupq_n_s32(RND_CONST);
                let mut v2 = vdupq_n_s32(RND_CONST);
                let mut v3 = vdupq_n_s32(RND_CONST);
                let mut v4 = vdupq_n_s32(RND_CONST);
                let mut v5 = vdupq_n_s32(RND_CONST);

                for (j, &k_weight) in filter.iter().take(bounds.size).enumerate() {
                    let py = bounds.start + j;
                    let weight = vdupq_n_s16(k_weight);
                    let offset = src_stride * py + v_start_px;
                    let src_ptr = src.get_unchecked(offset..(offset + diff * 4));

                    std::ptr::copy_nonoverlapping(
                        src_ptr.as_ptr(),
                        src_transient.as_mut_ptr(),
                        diff * 4,
                    );

                    let ps = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                        src_transient.as_ptr() as *const _,
                    ));
                    v0 = vqdmlal_s16(v0, vget_low_s16(ps.0), vget_low_s16(weight));
                    v1 = vqdmlal_high_s16(v1, ps.0, weight);
                    v2 = vqdmlal_s16(v2, vget_low_s16(ps.1), vget_low_s16(weight));
                    v3 = vqdmlal_high_s16(v3, ps.1, weight);
                    v4 = vqdmlal_s16(v4, vget_low_s16(ps.2), vget_low_s16(weight));
                    v5 = vqdmlal_high_s16(v5, ps.2, weight);
                }

                let v0 = vqshrun_n_s32::<PREC>(v0);
                let v1 = vqshrun_n_s32::<PREC>(v1);
                let v2 = vqshrun_n_s32::<PREC>(v2);
                let v3 = vqshrun_n_s32::<PREC>(v3);
                let v4 = vqshrun_n_s32::<PREC>(v4);
                let v5 = vqshrun_n_s32::<PREC>(v5);

                let r_v = vminq_u16(vcombine_u16(v0, v1), v_max);
                let g_v = vminq_u16(vcombine_u16(v2, v3), v_max);
                let b_v = vminq_u16(vcombine_u16(v4, v5), v_max);

                let vals = vzip_4_ar30::<AR30_TYPE, AR30_ORDER>(int16x8x4_t(
                    vreinterpretq_s16_u16(r_v),
                    vreinterpretq_s16_u16(g_v),
                    vreinterpretq_s16_u16(b_v),
                    vdupq_n_s16(3),
                ));
                xvst1q_u32_x2(dst_transient.as_mut_ptr() as *mut _, vals);

                let v_dst = dst.get_unchecked_mut(v_start_px..(v_start_px + diff * 4));
                std::ptr::copy_nonoverlapping(dst_transient.as_ptr(), v_dst.as_mut_ptr(), diff * 4);
            }
        }
    }
}
