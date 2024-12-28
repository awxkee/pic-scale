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
use crate::filter_weights::FilterWeights;
use crate::sse::_mm_prefer_fma_ps;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
unsafe fn conv_horiz_rgba_1_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    store: __m128,
) -> __m128 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);
    let rgba_pixel = _mm_loadu_si64(src_ptr.as_ptr() as *const u8);
    _mm_prefer_fma_ps::<FMA>(
        store,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel, _mm_setzero_si128())),
        w0,
    )
}

#[inline]
unsafe fn conv_horiz_rgba_2_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    w1: __m128,
    store: __m128,
) -> __m128 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);

    let acc = _mm_prefer_fma_ps::<FMA>(
        store,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel, _mm_setzero_si128())),
        w0,
    );
    _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel, _mm_setzero_si128())),
        w1,
    )
}

#[inline]
unsafe fn conv_horiz_rgba_4_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    w0: __m128,
    w1: __m128,
    w2: __m128,
    w3: __m128,
    store: __m128,
) -> __m128 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let rgba_pixel0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
    let rgba_pixel1 = _mm_loadu_si128(src_ptr.get_unchecked(8..).as_ptr() as *const __m128i);

    let acc = _mm_prefer_fma_ps::<FMA>(
        store,
        _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgba_pixel1, _mm_setzero_si128())),
        w3,
    );
    let acc = _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel1, _mm_setzero_si128())),
        w2,
    );
    let acc = _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgba_pixel0, _mm_setzero_si128())),
        w1,
    );
    _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgba_pixel0, _mm_setzero_si128())),
        w0,
    )
}

#[inline(always)]
unsafe fn conv_horiz_rgba_8_u16<const FMA: bool>(
    start_x: usize,
    src: &[u16],
    set1: (__m128, __m128, __m128, __m128),
    set2: (__m128, __m128, __m128, __m128),
    store: __m128,
) -> __m128 {
    const COMPONENTS: usize = 4;
    let src_ptr = src.get_unchecked((start_x * COMPONENTS)..);

    let zeros = _mm_setzero_si128();

    let rgba_pixel0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
    let rgba_pixel1 = _mm_loadu_si128(src_ptr.get_unchecked(8..).as_ptr() as *const __m128i);
    let rgba_pixel2 = _mm_loadu_si128(src_ptr.get_unchecked(16..).as_ptr() as *const __m128i);
    let rgba_pixel3 = _mm_loadu_si128(src_ptr.get_unchecked(24..).as_ptr() as *const __m128i);

    let mut acc = _mm_prefer_fma_ps::<FMA>(
        store,
        _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgba_pixel1, zeros)),
        set1.3,
    );
    acc = _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel1, zeros)),
        set1.2,
    );
    acc = _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgba_pixel0, zeros)),
        set1.1,
    );
    acc = _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel0, zeros)),
        set1.0,
    );

    acc = _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgba_pixel3, zeros)),
        set2.3,
    );
    acc = _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel3, zeros)),
        set2.2,
    );
    acc = _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpackhi_epi16(rgba_pixel2, zeros)),
        set2.1,
    );
    acc = _mm_prefer_fma_ps::<FMA>(
        acc,
        _mm_cvtepi32_ps(_mm_unpacklo_epi16(rgba_pixel2, zeros)),
        set2.0,
    );
    acc
}

pub(crate) fn convolve_horizontal_rgba_sse_rows_4_u16(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        if std::arch::is_x86_feature_detected!("fma") {
            convolve_horizontal_rgba_sse_rows_4_u16_fma(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        } else {
            convolve_horizontal_rgba_sse_rows_4_u16_def(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
                bit_depth,
            );
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgba_sse_rows_4_u16_def(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    convolve_horizontal_rgba_sse_rows_4_u16_impl::<false>(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
        bit_depth,
    );
}

#[target_feature(enable = "sse4.1", enable = "fma")]
unsafe fn convolve_horizontal_rgba_sse_rows_4_u16_fma(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    convolve_horizontal_rgba_sse_rows_4_u16_impl::<true>(
        src,
        src_stride,
        dst,
        dst_stride,
        filter_weights,
        bit_depth,
    );
}

#[inline(always)]
unsafe fn convolve_horizontal_rgba_sse_rows_4_u16_impl<const FMA: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    const CHANNELS: usize = 4;

    let v_max_colors = _mm_set1_epi32((1 << bit_depth) - 1);

    let (row0_ref, rest) = dst.split_at_mut(dst_stride);
    let (row1_ref, rest) = rest.split_at_mut(dst_stride);
    let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

    let iter_row0 = row0_ref.chunks_exact_mut(CHANNELS);
    let iter_row1 = row1_ref.chunks_exact_mut(CHANNELS);
    let iter_row2 = row2_ref.chunks_exact_mut(CHANNELS);
    let iter_row3 = row3_ref.chunks_exact_mut(CHANNELS);

    for (((((chunk0, chunk1), chunk2), chunk3), &bounds), weights) in iter_row0
        .zip(iter_row1)
        .zip(iter_row2)
        .zip(iter_row3)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let mut jx = 0usize;
        let mut store_0 = _mm_setzero_ps();
        let mut store_1 = _mm_setzero_ps();
        let mut store_2 = _mm_setzero_ps();
        let mut store_3 = _mm_setzero_ps();

        let bounds_size = bounds.size;

        let src0 = src;
        let src1 = src0.get_unchecked(src_stride..);
        let src2 = src1.get_unchecked(src_stride..);
        let src3 = src2.get_unchecked(src_stride..);

        while jx + 8 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 8));
            let w0 = _mm_set1_ps(w_ptr[0]);
            let w1 = _mm_set1_ps(w_ptr[1]);
            let w2 = _mm_set1_ps(w_ptr[2]);
            let w3 = _mm_set1_ps(w_ptr[3]);
            let w4 = _mm_set1_ps(w_ptr[4]);
            let w5 = _mm_set1_ps(w_ptr[5]);
            let w6 = _mm_set1_ps(w_ptr[6]);
            let w7 = _mm_set1_ps(w_ptr[7]);
            let set1 = (w0, w1, w2, w3);
            let set2 = (w4, w5, w6, w7);
            store_0 = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src0, set1, set2, store_0);
            store_1 = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src1, set1, set2, store_1);
            store_2 = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src2, set1, set2, store_2);
            store_3 = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src3, set1, set2, store_3);
            jx += 8;
        }

        while jx + 4 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let w0 = _mm_set1_ps(w_ptr[0]);
            let w1 = _mm_set1_ps(w_ptr[1]);
            let w2 = _mm_set1_ps(w_ptr[2]);
            let w3 = _mm_set1_ps(w_ptr[3]);
            store_0 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src0, w0, w1, w2, w3, store_0);
            store_1 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src1, w0, w1, w2, w3, store_1);
            store_2 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src2, w0, w1, w2, w3, store_2);
            store_3 = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src3, w0, w1, w2, w3, store_3);
            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let w0 = _mm_set1_ps(w_ptr[0]);
            let w1 = _mm_set1_ps(w_ptr[1]);
            store_0 = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src0, w0, w1, store_0);
            store_1 = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src1, w0, w1, store_1);
            store_2 = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src2, w0, w1, store_2);
            store_3 = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src3, w0, w1, store_3);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let bounds_start = bounds.start + jx;
            let w0 = _mm_set1_ps(w_ptr[0]);
            store_0 = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src0, w0, store_0);
            store_1 = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src1, w0, store_1);
            store_2 = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src2, w0, store_2);
            store_3 = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src3, w0, store_3);
            jx += 1;
        }

        let v_st0 = _mm_min_epi32(
            _mm_cvtps_epi32(_mm_max_ps(store_0, _mm_setzero_ps())),
            v_max_colors,
        );
        let v_st1 = _mm_min_epi32(
            _mm_cvtps_epi32(_mm_max_ps(store_1, _mm_setzero_ps())),
            v_max_colors,
        );
        let v_st2 = _mm_min_epi32(
            _mm_cvtps_epi32(_mm_max_ps(store_2, _mm_setzero_ps())),
            v_max_colors,
        );
        let v_st3 = _mm_min_epi32(
            _mm_cvtps_epi32(_mm_max_ps(store_3, _mm_setzero_ps())),
            v_max_colors,
        );

        let store_16_0 = _mm_packus_epi32(v_st0, v_st0);
        let store_16_1 = _mm_packus_epi32(v_st1, v_st1);
        let store_16_2 = _mm_packus_epi32(v_st2, v_st2);
        let store_16_3 = _mm_packus_epi32(v_st3, v_st3);

        _mm_storeu_si64(chunk0.as_mut_ptr() as *mut u8, store_16_0);
        _mm_storeu_si64(chunk1.as_mut_ptr() as *mut u8, store_16_1);
        _mm_storeu_si64(chunk2.as_mut_ptr() as *mut u8, store_16_2);
        _mm_storeu_si64(chunk3.as_mut_ptr() as *mut u8, store_16_3);
    }
}

pub(crate) fn convolve_horizontal_rgba_sse_u16_row(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    unsafe {
        if std::arch::is_x86_feature_detected!("fma") {
            convolve_horizontal_rgba_sse_u16_row_fma(src, dst, filter_weights, bit_depth);
        } else {
            convolve_horizontal_rgba_sse_u16_row_def(src, dst, filter_weights, bit_depth);
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn convolve_horizontal_rgba_sse_u16_row_def(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    convolve_horizontal_rgba_sse_u16_row_impl::<false>(src, dst, filter_weights, bit_depth);
}

#[target_feature(enable = "sse4.1", enable = "fma")]
unsafe fn convolve_horizontal_rgba_sse_u16_row_fma(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    convolve_horizontal_rgba_sse_u16_row_impl::<true>(src, dst, filter_weights, bit_depth);
}

#[inline(always)]
unsafe fn convolve_horizontal_rgba_sse_u16_row_impl<const FMA: bool>(
    src: &[u16],
    dst: &mut [u16],
    filter_weights: &FilterWeights<f32>,
    bit_depth: u32,
) {
    const CHANNELS: usize = 4;

    let v_max_colors = _mm_set1_epi32((1 << bit_depth) - 1);

    for ((dst, bounds), weights) in dst
        .chunks_exact_mut(CHANNELS)
        .zip(filter_weights.bounds.iter())
        .zip(
            filter_weights
                .weights
                .chunks_exact(filter_weights.aligned_size),
        )
    {
        let bounds_size = bounds.size;
        let mut jx = 0usize;
        let mut store = _mm_setzero_ps();

        while jx + 8 < bounds_size {
            let bounds_start = bounds.start + jx;
            let w_ptr = weights.get_unchecked(jx..(jx + 8));
            let w0 = _mm_set1_ps(w_ptr[0]);
            let w1 = _mm_set1_ps(w_ptr[1]);
            let w2 = _mm_set1_ps(w_ptr[2]);
            let w3 = _mm_set1_ps(w_ptr[3]);
            let w4 = _mm_set1_ps(w_ptr[4]);
            let w5 = _mm_set1_ps(w_ptr[5]);
            let w6 = _mm_set1_ps(w_ptr[6]);
            let w7 = _mm_set1_ps(w_ptr[7]);
            let set1 = (w0, w1, w2, w3);
            let set2 = (w4, w5, w6, w7);
            store = conv_horiz_rgba_8_u16::<FMA>(bounds_start, src, set1, set2, store);
            jx += 8;
        }

        while jx + 4 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 4));
            let w0 = _mm_set1_ps(w_ptr[0]);
            let w1 = _mm_set1_ps(w_ptr[1]);
            let w2 = _mm_set1_ps(w_ptr[2]);
            let w3 = _mm_set1_ps(w_ptr[3]);
            let bounds_start = bounds.start + jx;
            store = conv_horiz_rgba_4_u16::<FMA>(bounds_start, src, w0, w1, w2, w3, store);
            jx += 4;
        }

        while jx + 2 < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 2));
            let bounds_start = bounds.start + jx;
            let w0 = _mm_set1_ps(w_ptr[0]);
            let w1 = _mm_set1_ps(w_ptr[1]);
            store = conv_horiz_rgba_2_u16::<FMA>(bounds_start, src, w0, w1, store);
            jx += 2;
        }

        while jx < bounds_size {
            let w_ptr = weights.get_unchecked(jx..(jx + 1));
            let w0 = _mm_set1_ps(w_ptr[0]);
            let bounds_start = bounds.start + jx;
            store = conv_horiz_rgba_1_u16::<FMA>(bounds_start, src, w0, store);
            jx += 1;
        }

        let v_st = _mm_min_epi32(
            _mm_cvtps_epi32(_mm_max_ps(store, _mm_setzero_ps())),
            v_max_colors,
        );

        let store_16_0 = _mm_packus_epi32(v_st, v_st);
        _mm_storeu_si64(dst.as_mut_ptr() as *mut u8, store_16_0);
    }
}
