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
use crate::fixed_point_vertical_ar30::convolve_column_handler_fip_db_ar30;
use crate::neon::ar30::{vunzip_3_ar30, vzip_4_ar30};
use crate::neon::utils::{xvld1q_u32_x2, xvst1q_u32_x2};
use std::arch::aarch64::{
    int16x8x4_t, vdupq_n_s16, vmaxq_s16, vminq_s16, vqrdmlahq_s16, vqrdmulhq_s16, vrshrq_n_s16,
    vshlq_n_s16,
};

pub(crate) fn neon_column_handler_fixed_point_ar30_rdm<
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
    #[target_feature(enable = "rdm")]
    unsafe fn pass(
        &self,
        bounds: &FilterBounds,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        weight: &[i16],
    ) {
        let mut cx = 0usize;

        let total_width = dst.len() / 4;

        const PREC: i32 = 5;
        const BACK: i32 = 5;

        let bounds_size = bounds.size;

        while cx + 8 < total_width {
            unsafe {
                let v_max = vdupq_n_s16(1023);
                let zeros = vdupq_n_s16(0);
                let filter = weight;
                let v_start_px = cx * 4;

                let py = bounds.start;
                let weight = vdupq_n_s16(filter[0]);
                let offset = src_stride * py + v_start_px;
                let src_ptr = src.get_unchecked(offset..(offset + 8));

                let ps = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                    src_ptr.as_ptr() as *const _
                ));
                let mut v0 = vqrdmulhq_s16(vshlq_n_s16::<PREC>(ps.0), weight);
                let mut v1 = vqrdmulhq_s16(vshlq_n_s16::<PREC>(ps.1), weight);
                let mut v2 = vqrdmulhq_s16(vshlq_n_s16::<PREC>(ps.2), weight);

                if bounds_size == 2 {
                    let weights = filter.get_unchecked(0..2);
                    let py = bounds.start;
                    let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_start_px)..);

                    let v_weight1 = vdupq_n_s16(weights[1]);

                    let ps1 = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                        src_ptr1.as_ptr() as *const _,
                    ));
                    v0 = vqrdmlahq_s16(v0, vshlq_n_s16::<PREC>(ps1.0), v_weight1);
                    v1 = vqrdmlahq_s16(v1, vshlq_n_s16::<PREC>(ps1.1), v_weight1);
                    v2 = vqrdmlahq_s16(v2, vshlq_n_s16::<PREC>(ps1.2), v_weight1);
                } else if bounds_size == 3 {
                    let weights = filter.get_unchecked(0..3);
                    let py = bounds.start;
                    let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_start_px)..);
                    let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_start_px)..);

                    let v_weight1 = vdupq_n_s16(weights[1]);
                    let v_weight2 = vdupq_n_s16(weights[2]);

                    let ps1 = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                        src_ptr1.as_ptr() as *const _,
                    ));
                    v0 = vqrdmlahq_s16(v0, vshlq_n_s16::<PREC>(ps1.0), v_weight1);
                    v1 = vqrdmlahq_s16(v1, vshlq_n_s16::<PREC>(ps1.1), v_weight1);
                    v2 = vqrdmlahq_s16(v2, vshlq_n_s16::<PREC>(ps1.2), v_weight1);
                    let ps2 = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                        src_ptr2.as_ptr() as *const _,
                    ));
                    v0 = vqrdmlahq_s16(v0, vshlq_n_s16::<PREC>(ps2.0), v_weight2);
                    v1 = vqrdmlahq_s16(v1, vshlq_n_s16::<PREC>(ps2.1), v_weight2);
                    v2 = vqrdmlahq_s16(v2, vshlq_n_s16::<PREC>(ps2.2), v_weight2);
                } else if bounds_size == 4 {
                    let weights = filter.get_unchecked(0..4);
                    let py = bounds.start;
                    let src_ptr1 = src.get_unchecked((src_stride * (py + 1) + v_start_px)..);
                    let src_ptr2 = src.get_unchecked((src_stride * (py + 2) + v_start_px)..);
                    let src_ptr3 = src.get_unchecked((src_stride * (py + 3) + v_start_px)..);

                    let v_weight1 = vdupq_n_s16(weights[1]);
                    let v_weight2 = vdupq_n_s16(weights[2]);
                    let v_weight3 = vdupq_n_s16(weights[3]);

                    let ps1 = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                        src_ptr1.as_ptr() as *const _,
                    ));
                    v0 = vqrdmlahq_s16(v0, vshlq_n_s16::<PREC>(ps1.0), v_weight1);
                    v1 = vqrdmlahq_s16(v1, vshlq_n_s16::<PREC>(ps1.1), v_weight1);
                    v2 = vqrdmlahq_s16(v2, vshlq_n_s16::<PREC>(ps1.2), v_weight1);
                    let ps2 = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                        src_ptr2.as_ptr() as *const _,
                    ));
                    v0 = vqrdmlahq_s16(v0, vshlq_n_s16::<PREC>(ps2.0), v_weight2);
                    v1 = vqrdmlahq_s16(v1, vshlq_n_s16::<PREC>(ps2.1), v_weight2);
                    v2 = vqrdmlahq_s16(v2, vshlq_n_s16::<PREC>(ps2.2), v_weight2);
                    let ps3 = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                        src_ptr3.as_ptr() as *const _,
                    ));
                    v0 = vqrdmlahq_s16(v0, vshlq_n_s16::<PREC>(ps3.0), v_weight3);
                    v1 = vqrdmlahq_s16(v1, vshlq_n_s16::<PREC>(ps3.1), v_weight3);
                    v2 = vqrdmlahq_s16(v2, vshlq_n_s16::<PREC>(ps3.2), v_weight3);
                } else {
                    for (j, &k_weight) in filter.iter().take(bounds.size).skip(1).enumerate() {
                        // Adding 1 is necessary because skip do not incrementing value on values that skipped
                        let py = bounds.start + j + 1;
                        let weight = vdupq_n_s16(k_weight);
                        let offset = src_stride * py + v_start_px;
                        let src_ptr = src.get_unchecked(offset..(offset + 8 * 4));

                        let ps = vunzip_3_ar30::<AR30_TYPE, AR30_ORDER>(xvld1q_u32_x2(
                            src_ptr.as_ptr() as *const _,
                        ));
                        v0 = vqrdmlahq_s16(v0, vshlq_n_s16::<PREC>(ps.0), weight);
                        v1 = vqrdmlahq_s16(v1, vshlq_n_s16::<PREC>(ps.1), weight);
                        v2 = vqrdmlahq_s16(v2, vshlq_n_s16::<PREC>(ps.2), weight);
                    }
                }

                let v_dst = dst.get_unchecked_mut(v_start_px..(v_start_px + 8 * 4));

                v0 = vrshrq_n_s16::<BACK>(v0);
                v1 = vrshrq_n_s16::<BACK>(v1);
                v2 = vrshrq_n_s16::<BACK>(v2);

                v0 = vmaxq_s16(vminq_s16(v0, v_max), zeros);
                v1 = vmaxq_s16(vminq_s16(v1, v_max), zeros);
                v2 = vmaxq_s16(vminq_s16(v2, v_max), zeros);

                let vals =
                    vzip_4_ar30::<AR30_TYPE, AR30_ORDER>(int16x8x4_t(v0, v1, v2, vdupq_n_s16(3)));
                xvst1q_u32_x2(v_dst.as_mut_ptr() as *mut _, vals);
            }

            cx += 8;
        }

        while cx + 4 < total_width {
            convolve_column_handler_fip_db_ar30::<AR30_TYPE, AR30_ORDER, 4>(
                src, src_stride, dst, weight, bounds, cx,
            );

            cx += 4;
        }

        while cx < total_width {
            convolve_column_handler_fip_db_ar30::<AR30_TYPE, AR30_ORDER, 1>(
                src, src_stride, dst, weight, bounds, cx,
            );

            cx += 1;
        }
    }
}
