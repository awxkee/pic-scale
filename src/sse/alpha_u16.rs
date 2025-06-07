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

use crate::sse::alpha_u8::_mm_select_si128;
use crate::sse::{sse_deinterleave_rgba_epi16, sse_interleave_rgba_epi16};
use novtb::{ParallelZonedIterator, TbSliceMut};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_unpremultiply_row_u16(
    x: __m128i,
    is_zero_mask: __m128i,
    a_lo_f: __m128,
    a_hi_f: __m128,
) -> __m128i {
    unsafe {
        let zeros = _mm_setzero_si128();
        let lo = _mm_unpacklo_epi16(x, zeros);
        let hi = _mm_unpackhi_epi16(x, zeros);

        let new_lo = _mm_cvtps_epi32(_mm_add_ps(
            _mm_set1_ps(0.5f32),
            _mm_mul_ps(_mm_cvtepi32_ps(lo), a_lo_f),
        ));
        let new_hi = _mm_cvtps_epi32(_mm_add_ps(
            _mm_set1_ps(0.5f32),
            _mm_mul_ps(_mm_cvtepi32_ps(hi), a_hi_f),
        ));

        let pixel = _mm_packs_epi32(new_lo, new_hi);
        _mm_select_si128(is_zero_mask, x, pixel)
    }
}

/// Exact division by 1023 with rounding to nearest
#[inline(always)]
pub(crate) unsafe fn _mm_div_by_1023_epi32(v: __m128i) -> __m128i {
    unsafe {
        const DIVIDING_BY: i32 = 10;
        let addition = _mm_set1_epi32(1 << (DIVIDING_BY - 1));
        let v = _mm_add_epi32(v, addition);
        _mm_srli_epi32::<DIVIDING_BY>(_mm_add_epi32(v, _mm_srli_epi32::<DIVIDING_BY>(v)))
    }
}

/// Exact division by 4095 with rounding to nearest
#[inline(always)]
pub(crate) unsafe fn _mm_div_by_4095_epi32(v: __m128i) -> __m128i {
    unsafe {
        const DIVIDING_BY: i32 = 12;
        let addition = _mm_set1_epi32(1 << (DIVIDING_BY - 1));
        let v = _mm_add_epi32(v, addition);
        _mm_srli_epi32::<DIVIDING_BY>(_mm_add_epi32(v, _mm_srli_epi32::<DIVIDING_BY>(v)))
    }
}

/// Exact division by 65535 with rounding to nearest
#[inline(always)]
pub(crate) unsafe fn _mm_div_by_65535_epi32(v: __m128i) -> __m128i {
    unsafe {
        const DIVIDING_BY: i32 = 16;
        let addition = _mm_set1_epi32(1 << (DIVIDING_BY - 1));
        let v = _mm_add_epi32(v, addition);
        _mm_srli_epi32::<DIVIDING_BY>(_mm_add_epi32(v, _mm_srli_epi32::<DIVIDING_BY>(v)))
    }
}

#[inline(always)]
unsafe fn _mm_div_by<const BIT_DEPTH: usize>(v: __m128i) -> __m128i {
    unsafe {
        if BIT_DEPTH == 10 {
            _mm_div_by_1023_epi32(v)
        } else if BIT_DEPTH == 12 {
            _mm_div_by_4095_epi32(v)
        } else {
            _mm_div_by_65535_epi32(v)
        }
    }
}

pub(crate) fn unpremultiply_alpha_sse_rgba_u16(
    in_place: &mut [u16],
    stride: usize,
    width: usize,
    height: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    unsafe {
        unpremultiply_alpha_sse_rgba_u16_impl(in_place, stride, width, height, bit_depth, pool);
    }
}

trait DisassociateAlpha {
    unsafe fn disassociate(&self, in_place: &mut [u16], bit_depth: usize);
}

#[derive(Default)]
struct DisassociateAlphaDefault {}

impl DisassociateAlphaDefault {
    #[inline(always)]
    unsafe fn disassociate_chunk(
        &self,
        in_place: &mut [u16],
        v_max_colors: __m128,
        bit_depth: usize,
    ) {
        unsafe {
            let src_ptr = in_place.as_ptr();

            let max_colors = (1u32 << bit_depth) - 1;
            let v_max_test = _mm_set1_epi16(max_colors as i16);

            let row0 = _mm_loadu_si128(src_ptr as *const __m128i);
            let row1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
            let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
            let row3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
            let (rrrr, gggg, bbbb, aaaa) = sse_deinterleave_rgba_epi16(row0, row1, row2, row3);

            let is_zero_mask = _mm_cmpeq_epi16(aaaa, _mm_setzero_si128());
            let a_lo_f = _mm_mul_ps(
                _mm_rcp_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(
                    aaaa,
                    _mm_setzero_si128(),
                ))),
                v_max_colors,
            );
            let a_hi_f = _mm_mul_ps(
                _mm_rcp_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(
                    aaaa,
                    _mm_setzero_si128(),
                ))),
                v_max_colors,
            );

            let mut new_rrrr = sse_unpremultiply_row_u16(rrrr, is_zero_mask, a_lo_f, a_hi_f);
            let mut new_gggg = sse_unpremultiply_row_u16(gggg, is_zero_mask, a_lo_f, a_hi_f);
            let mut new_bbbb = sse_unpremultiply_row_u16(bbbb, is_zero_mask, a_lo_f, a_hi_f);

            new_rrrr = _mm_min_epu16(new_rrrr, v_max_test);
            new_gggg = _mm_min_epu16(new_gggg, v_max_test);
            new_bbbb = _mm_min_epu16(new_bbbb, v_max_test);

            let (rgba0, rgba1, rgba2, rgba3) =
                sse_interleave_rgba_epi16(new_rrrr, new_gggg, new_bbbb, aaaa);

            let dst_ptr = in_place.as_mut_ptr();
            _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, rgba1);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba2);
            _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, rgba3);
        }
    }
}

impl DisassociateAlpha for DisassociateAlphaDefault {
    #[target_feature(enable = "sse4.1")]
    unsafe fn disassociate(&self, in_place: &mut [u16], bit_depth: usize) {
        unsafe {
            let max_colors = (1 << bit_depth) - 1;

            let v_max_colors = _mm_set1_ps(max_colors as f32);

            let mut rem = in_place;

            for dst in rem.chunks_exact_mut(8 * 4) {
                self.disassociate_chunk(dst, v_max_colors, bit_depth);
            }

            rem = rem.chunks_exact_mut(8 * 4).into_remainder();

            if !rem.is_empty() {
                assert!(rem.len() < 8 * 4);
                let mut buffer: [u16; 8 * 4] = [0u16; 8 * 4];

                std::ptr::copy_nonoverlapping(rem.as_ptr(), buffer.as_mut_ptr(), rem.len());

                self.disassociate_chunk(&mut buffer, v_max_colors, bit_depth);

                std::ptr::copy_nonoverlapping(buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn unpremultiply_alpha_sse_rgba_u16_row_impl(
    in_place: &mut [u16],
    bit_depth: usize,
    executor: impl DisassociateAlpha,
) {
    unsafe {
        executor.disassociate(in_place, bit_depth);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn unpremultiply_alpha_sse_rgba_u16_impl(
    in_place: &mut [u16],
    stride: usize,
    width: usize,
    _: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    in_place
        .tb_par_chunks_exact_mut(stride)
        .for_each(pool, |row| unsafe {
            unpremultiply_alpha_sse_rgba_u16_row_impl(
                &mut row[..width * 4],
                bit_depth,
                DisassociateAlphaDefault::default(),
            );
        });
}

#[inline(always)]
unsafe fn sse_premultiply_row_u16(
    x: __m128i,
    a_lo_f: __m128,
    a_hi_f: __m128,
    v_max_colors_scale: __m128,
) -> __m128i {
    unsafe {
        let zeros = _mm_setzero_si128();
        let lo = _mm_unpacklo_epi16(x, zeros);
        let hi = _mm_unpackhi_epi16(x, zeros);

        let new_lo = _mm_cvtps_epi32(_mm_mul_ps(
            _mm_mul_ps(_mm_cvtepi32_ps(lo), v_max_colors_scale),
            a_lo_f,
        ));
        let new_hi = _mm_cvtps_epi32(_mm_mul_ps(
            _mm_mul_ps(_mm_cvtepi32_ps(hi), v_max_colors_scale),
            a_hi_f,
        ));

        _mm_packs_epi32(new_lo, new_hi)
    }
}

pub(crate) fn premultiply_alpha_sse_rgba_u16(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    width: usize,
    _: usize,
    src_stride: usize,
    bit_depth: usize,
    pool: &novtb::ThreadPool,
) {
    dst.tb_par_chunks_exact_mut(dst_stride)
        .for_each_enumerated(pool, |y, dst| unsafe {
            let src = &src[y * src_stride..(y + 1) * src_stride];
            premultiply_alpha_sse_rgba_u16_row_impl(
                &mut dst[..width * 4],
                &src[..width * 4],
                bit_depth,
            );
        });
}

trait Sse41PremultiplyExecutor {
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize);
}

#[derive(Default)]
struct Sse41PremultiplyExecutorDefault<const BIT_DEPTH: usize> {}

impl<const BIT_DEPTH: usize> Sse41PremultiplyExecutorDefault<BIT_DEPTH> {
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn premultiply_chunk(&self, dst: &mut [u16], src: &[u16]) {
        unsafe {
            let zeros = _mm_setzero_si128();
            let src_ptr = src.as_ptr();
            let row0 = _mm_loadu_si128(src_ptr as *const __m128i);
            let row1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
            let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
            let row3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
            let (rrrr, gggg, bbbb, aaaa) = sse_deinterleave_rgba_epi16(row0, row1, row2, row3);

            let a_lo_f = _mm_unpacklo_epi16(aaaa, zeros);
            let a_hi_f = _mm_unpackhi_epi16(aaaa, zeros);

            let new_rrrr = _mm_packus_epi32(
                _mm_div_by::<BIT_DEPTH>(_mm_madd_epi16(_mm_unpacklo_epi16(rrrr, zeros), a_lo_f)),
                _mm_div_by::<BIT_DEPTH>(_mm_madd_epi16(_mm_unpackhi_epi16(rrrr, zeros), a_hi_f)),
            );
            let new_gggg = _mm_packus_epi32(
                _mm_div_by::<BIT_DEPTH>(_mm_madd_epi16(_mm_unpacklo_epi16(gggg, zeros), a_lo_f)),
                _mm_div_by::<BIT_DEPTH>(_mm_madd_epi16(_mm_unpackhi_epi16(gggg, zeros), a_hi_f)),
            );
            let new_bbbb = _mm_packus_epi32(
                _mm_div_by::<BIT_DEPTH>(_mm_madd_epi16(_mm_unpacklo_epi16(bbbb, zeros), a_lo_f)),
                _mm_div_by::<BIT_DEPTH>(_mm_madd_epi16(_mm_unpackhi_epi16(bbbb, zeros), a_hi_f)),
            );

            let (rgba0, rgba1, rgba2, rgba3) =
                sse_interleave_rgba_epi16(new_rrrr, new_gggg, new_bbbb, aaaa);

            let dst_ptr = dst.as_mut_ptr();
            _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, rgba1);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba2);
            _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, rgba3);
        }
    }
}

impl<const BIT_DEPTH: usize> Sse41PremultiplyExecutor
    for Sse41PremultiplyExecutorDefault<BIT_DEPTH>
{
    #[target_feature(enable = "sse4.1")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], _: usize) {
        unsafe {
            let mut rem = dst;
            let mut src_rem = src;

            for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
                self.premultiply_chunk(dst, src);
            }

            rem = rem.chunks_exact_mut(8 * 4).into_remainder();
            src_rem = src_rem.chunks_exact(8 * 4).remainder();

            if !rem.is_empty() {
                assert!(src_rem.len() < 8 * 4);
                assert!(rem.len() < 8 * 4);
                assert_eq!(src_rem.len(), rem.len());

                let mut buffer: [u16; 8 * 4] = [0u16; 8 * 4];
                let mut dst_buffer: [u16; 8 * 4] = [0u16; 8 * 4];

                std::ptr::copy_nonoverlapping(src_rem.as_ptr(), buffer.as_mut_ptr(), src_rem.len());

                self.premultiply_chunk(&mut dst_buffer, &buffer);

                std::ptr::copy_nonoverlapping(dst_buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }
}

#[derive(Default)]
struct Sse41PremultiplyExecutorAny {}

impl Sse41PremultiplyExecutorAny {
    #[inline(always)]
    unsafe fn premultiply_chunk(&self, dst: &mut [u16], src: &[u16], scale: __m128) {
        unsafe {
            let src_ptr = src.as_ptr();
            let row0 = _mm_loadu_si128(src_ptr as *const __m128i);
            let row1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
            let row2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
            let row3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);
            let (rrrr, gggg, bbbb, aaaa) = sse_deinterleave_rgba_epi16(row0, row1, row2, row3);

            let a_lo_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(aaaa, _mm_setzero_si128()));
            let a_hi_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(aaaa, _mm_setzero_si128()));

            let new_rrrr = sse_premultiply_row_u16(rrrr, a_lo_f, a_hi_f, scale);
            let new_gggg = sse_premultiply_row_u16(gggg, a_lo_f, a_hi_f, scale);
            let new_bbbb = sse_premultiply_row_u16(bbbb, a_lo_f, a_hi_f, scale);

            let (rgba0, rgba1, rgba2, rgba3) =
                sse_interleave_rgba_epi16(new_rrrr, new_gggg, new_bbbb, aaaa);

            let dst_ptr = dst.as_mut_ptr();
            _mm_storeu_si128(dst_ptr as *mut __m128i, rgba0);
            _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, rgba1);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, rgba2);
            _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, rgba3);
        }
    }
}

impl Sse41PremultiplyExecutor for Sse41PremultiplyExecutorAny {
    #[target_feature(enable = "sse4.1")]
    unsafe fn premultiply(&self, dst: &mut [u16], src: &[u16], bit_depth: usize) {
        unsafe {
            let max_colors = (1 << bit_depth) - 1;

            let mut rem = dst;
            let mut src_rem = src;

            let v_max_colors_scale =
                _mm_div_ps(_mm_set1_ps(1.), _mm_cvtepi32_ps(_mm_set1_epi32(max_colors)));
            for (dst, src) in rem.chunks_exact_mut(8 * 4).zip(src_rem.chunks_exact(8 * 4)) {
                self.premultiply_chunk(dst, src, v_max_colors_scale);
            }

            rem = rem.chunks_exact_mut(8 * 4).into_remainder();
            src_rem = src_rem.chunks_exact(8 * 4).remainder();

            if !rem.is_empty() {
                assert!(src_rem.len() < 8 * 4);
                assert!(rem.len() < 8 * 4);
                assert_eq!(src_rem.len(), rem.len());

                let mut buffer: [u16; 8 * 4] = [0u16; 8 * 4];
                let mut dst_buffer: [u16; 8 * 4] = [0u16; 8 * 4];

                std::ptr::copy_nonoverlapping(src_rem.as_ptr(), buffer.as_mut_ptr(), src_rem.len());

                self.premultiply_chunk(&mut dst_buffer, &buffer, v_max_colors_scale);

                std::ptr::copy_nonoverlapping(dst_buffer.as_ptr(), rem.as_mut_ptr(), rem.len());
            }
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn pma_sse41_rgba16_dispatch(
    dst: &mut [u16],
    src: &[u16],
    bit_depth: usize,
    executor: impl Sse41PremultiplyExecutor,
) {
    unsafe {
        executor.premultiply(dst, src, bit_depth);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn premultiply_alpha_sse_rgba_u16_row_impl(dst: &mut [u16], src: &[u16], bit_depth: usize) {
    unsafe {
        if bit_depth == 10 {
            pma_sse41_rgba16_dispatch(
                dst,
                src,
                bit_depth,
                Sse41PremultiplyExecutorDefault::<10>::default(),
            )
        } else if bit_depth == 12 {
            pma_sse41_rgba16_dispatch(
                dst,
                src,
                bit_depth,
                Sse41PremultiplyExecutorDefault::<12>::default(),
            )
        } else {
            pma_sse41_rgba16_dispatch(dst, src, bit_depth, Sse41PremultiplyExecutorAny::default())
        }
    }
}
