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
use crate::alpha_handle_f32::{premultiply_rgba_f32_row, unpremultiply_rgba_f32_row};
use crate::sse::{sse_deinterleave_rgba_ps, sse_interleave_rgba_ps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
fn sse_unpremultiply_row_f32(x: __m128, a: __m128) -> __m128 {
    unsafe {
        let is_zero_mask = _mm_cmpeq_ps(a, _mm_setzero_ps());
        let rs = _mm_div_ps(x, a);
        _mm_blendv_ps(rs, _mm_setzero_ps(), is_zero_mask)
    }
}

pub(crate) fn sse_unpremultiply_alpha_rgba_f32(in_place: &mut [f32]) {
    unsafe {
        sse_unpremultiply_alpha_rgba_f32_row_impl(in_place);
    }
}

#[target_feature(enable = "sse4.1")]
fn sse_unpremultiply_alpha_rgba_f32_row_impl(in_place: &mut [f32]) {
    unsafe {
        for dst in in_place.as_chunks_mut::<16>().0.iter_mut() {
            let rgba0 = _mm_loadu_ps(dst.as_ptr());
            let rgba1 = _mm_loadu_ps(dst[4..].as_ptr());
            let rgba2 = _mm_loadu_ps(dst[8..].as_ptr());
            let rgba3 = _mm_loadu_ps(dst[12..].as_ptr());

            let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba_ps(rgba0, rgba1, rgba2, rgba3);

            let rrr = sse_unpremultiply_row_f32(rrr, aaa);
            let ggg = sse_unpremultiply_row_f32(ggg, aaa);
            let bbb = sse_unpremultiply_row_f32(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba_ps(rrr, ggg, bbb, aaa);

            _mm_storeu_ps(dst.as_mut_ptr(), rgba0);
            _mm_storeu_ps(dst[4..].as_mut_ptr(), rgba1);
            _mm_storeu_ps(dst[8..].as_mut_ptr(), rgba2);
            _mm_storeu_ps(dst[12..].as_mut_ptr(), rgba3);
        }

        let rem = in_place.as_chunks_mut::<16>().1;

        unpremultiply_rgba_f32_row(rem);
    }
}

pub(crate) fn sse_premultiply_alpha_rgba_f32(dst: &mut [f32], src: &[f32]) {
    unsafe {
        sse_premultiply_alpha_rgba_f32_row_impl(dst, src);
    }
}

#[target_feature(enable = "sse4.1")]
fn sse_premultiply_alpha_rgba_f32_row_impl(dst: &mut [f32], src: &[f32]) {
    unsafe {
        let mut rem = dst;
        let mut src_rem = src;

        for (dst, src) in rem
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(src_rem.as_chunks::<16>().0.iter())
        {
            let src_ptr = src.as_ptr();
            let rgba0 = _mm_loadu_ps(src_ptr);
            let rgba1 = _mm_loadu_ps(src_ptr.add(4));
            let rgba2 = _mm_loadu_ps(src_ptr.add(8));
            let rgba3 = _mm_loadu_ps(src_ptr.add(12));
            let (rrr, ggg, bbb, aaa) = sse_deinterleave_rgba_ps(rgba0, rgba1, rgba2, rgba3);

            let rrr = _mm_mul_ps(rrr, aaa);
            let ggg = _mm_mul_ps(ggg, aaa);
            let bbb = _mm_mul_ps(bbb, aaa);

            let (rgba0, rgba1, rgba2, rgba3) = sse_interleave_rgba_ps(rrr, ggg, bbb, aaa);

            let dst_ptr = dst.as_mut_ptr();
            _mm_storeu_ps(dst_ptr, rgba0);
            _mm_storeu_ps(dst_ptr.add(4), rgba1);
            _mm_storeu_ps(dst_ptr.add(8), rgba2);
            _mm_storeu_ps(dst_ptr.add(12), rgba3);
        }

        rem = rem.as_chunks_mut::<16>().1;
        src_rem = src_rem.as_chunks::<16>().1;

        premultiply_rgba_f32_row(rem, src_rem);
    }
}

#[cfg(test)]
mod backend_comparison_tests {
    use super::*;
    use crate::test_utils::{XorShiftRng, assert_f32_slices_close};

    // Premultiply is a plain per-lane multiply with no reassociation, so it
    // is bit-exact against the scalar reference. Unpremultiply is not: the
    // scalar reference computes `1. / a` and then multiplies, while SSE does
    // a direct `_mm_div_ps(x, a)` - mathematically the same but rounded
    // differently, so it can be off by a few ULPs. Since alpha is kept away
    // from 0 (see `make_rgba`) that relative error stays small in absolute
    // terms too.
    const ATOL: f32 = 1e-4;

    /// Builds an RGBA f32 buffer in `[0, 1)` that mixes alpha == 0, alpha == 1
    /// and fully random pixels so both the vectorized bulk path and the
    /// scalar tail remainder exercise every branch the
    /// premultiply/unpremultiply math takes on alpha. Random alpha is kept
    /// away from 0 (but still far from 1) so that unpremultiplying does not
    /// blow up the relative rounding error between the direct SSE division
    /// and the scalar reciprocal-then-multiply into a huge absolute one.
    fn make_rgba(rng: &mut XorShiftRng, pixels: usize) -> Vec<f32> {
        let mut buf = rng.fill_f32_unit(pixels * 4);
        for chunk in buf.as_chunks_mut::<4>().0 {
            chunk[3] = 0.05 + 0.95 * chunk[3];
        }
        if let Some(chunk) = buf.first_chunk_mut::<4>() {
            chunk[3] = 0.;
        }
        if pixels > 1 {
            buf[4 + 3] = 1.;
        }
        buf
    }

    #[test]
    fn sse_premultiply_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        for pixels in [1usize, 4, 17, 256] {
            let mut rng = XorShiftRng::new(0xA11CE ^ pixels as u64);
            let src = make_rgba(&mut rng, pixels);

            let mut dst_scalar = vec![0f32; pixels * 4];
            let mut dst_sse = vec![0f32; pixels * 4];
            premultiply_rgba_f32_row(&mut dst_scalar, &src);
            sse_premultiply_alpha_rgba_f32(&mut dst_sse, &src);

            assert_f32_slices_close(
                &dst_sse,
                &dst_scalar,
                ATOL,
                &format!("{pixels} pixels: SSE premultiply"),
            );
        }
    }

    #[test]
    fn sse_unpremultiply_matches_scalar_reference() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        for pixels in [1usize, 4, 17, 256] {
            let mut rng = XorShiftRng::new(0xB0BA ^ pixels as u64);
            let src = make_rgba(&mut rng, pixels);

            let mut scalar_buf = src.clone();
            let mut sse_buf = src;
            unpremultiply_rgba_f32_row(&mut scalar_buf);
            sse_unpremultiply_alpha_rgba_f32(&mut sse_buf);

            assert_f32_slices_close(
                &sse_buf,
                &scalar_buf,
                ATOL,
                &format!("{pixels} pixels: SSE unpremultiply"),
            );
        }
    }
}
