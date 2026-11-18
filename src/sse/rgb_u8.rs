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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::filter_weights::FilterWeights;
use crate::sse::{compress_i32, convolve_horizontal_parts_one_sse_rgb, shuffle};
use crate::support::ROUNDING_CONST;

pub(crate) fn convolve_horizontal_rgb_sse_rows_4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        if filter_weights.kernel_size >= 8 {
            convolve_horizontal_rgb_sse_rows_4_impl::<true>(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
            );
        } else {
            convolve_horizontal_rgb_sse_rows_4_impl::<false>(
                src,
                src_stride,
                dst,
                dst_stride,
                filter_weights,
            );
        }
    }
}

#[inline(always)]
fn load_rgb_x2(src: &[u8]) -> __m128i {
    unsafe {
        let mut rgb_pixel = _mm_setzero_si128();
        rgb_pixel = _mm_insert_epi32::<0>(rgb_pixel, (src.as_ptr() as *const i32).read_unaligned());
        rgb_pixel = _mm_insert_epi16::<2>(
            rgb_pixel,
            (src.get_unchecked(4..).as_ptr() as *const i16).read_unaligned() as i32,
        );
        rgb_pixel
    }
}

#[inline(always)]
fn load_rgb_x4(src: &[u8]) -> __m128i {
    unsafe {
        let mut rgb_pixel = _mm_loadu_si64(src.as_ptr());
        rgb_pixel = _mm_insert_epi32::<2>(
            rgb_pixel,
            (src.get_unchecked(8..).as_ptr() as *const i32).read_unaligned(),
        );
        rgb_pixel
    }
}

#[inline(always)]
fn convolve_rgb_x8(src: &[u8], mut sum: __m128i, weights: __m128i) -> __m128i {
    unsafe {
        // Their union is exactly 24 bytes, including the final packed RGB pixel.
        let first = _mm_loadu_si128(src.as_ptr().cast());
        let second = _mm_loadu_si128(src.as_ptr().add(8).cast());
        let sh01 = _mm_setr_epi8(0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1);
        let sh23 = _mm_setr_epi8(6, -1, 9, -1, 7, -1, 10, -1, 8, -1, 11, -1, -1, -1, -1, -1);
        let sh45 = _mm_setr_epi8(4, -1, 7, -1, 5, -1, 8, -1, 6, -1, 9, -1, -1, -1, -1, -1);
        let sh67 = _mm_setr_epi8(
            10, -1, 13, -1, 11, -1, 14, -1, 12, -1, 15, -1, -1, -1, -1, -1,
        );
        sum = _mm_add_epi32(
            sum,
            _mm_madd_epi16(
                _mm_shuffle_epi8(first, sh01),
                _mm_shuffle_epi32::<0x00>(weights),
            ),
        );
        sum = _mm_add_epi32(
            sum,
            _mm_madd_epi16(
                _mm_shuffle_epi8(first, sh23),
                _mm_shuffle_epi32::<0x55>(weights),
            ),
        );
        sum = _mm_add_epi32(
            sum,
            _mm_madd_epi16(
                _mm_shuffle_epi8(second, sh45),
                _mm_shuffle_epi32::<0xaa>(weights),
            ),
        );
        _mm_add_epi32(
            sum,
            _mm_madd_epi16(
                _mm_shuffle_epi8(second, sh67),
                _mm_shuffle_epi32::<0xff>(weights),
            ),
        )
    }
}

#[target_feature(enable = "sse4.1")]
fn convolve_horizontal_rgb_sse_rows_4_impl<const WIDE: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CN: usize = 3;

        #[rustfmt::skip]
        let shuffle_lo = _mm_setr_epi8(0, -1,
                                               3, -1,
                                               1, -1,
                                               4, -1,
                                               2, -1 ,
                                               5,-1,
                                               -1, -1,
                                               -1, -1);

        #[rustfmt::skip]
        let shuffle_hi = _mm_setr_epi8(6, -1,
                                               9, -1,
                                               7, -1,
                                               10, -1 ,
                                               8,-1,
                                               11, -1,
                                               -1, -1,
                                               -1, -1);

        let vld = _mm_set1_epi32(ROUNDING_CONST);

        let (row0_ref, rest) = dst.split_at_mut(dst_stride);
        let (row1_ref, rest) = rest.split_at_mut(dst_stride);
        let (row2_ref, row3_ref) = rest.split_at_mut(dst_stride);

        let iter_row0 = row0_ref.as_chunks_mut::<CN>().0.iter_mut();
        let iter_row1 = row1_ref.as_chunks_mut::<CN>().0.iter_mut();
        let iter_row2 = row2_ref.as_chunks_mut::<CN>().0.iter_mut();
        let iter_row3 = row3_ref.as_chunks_mut::<CN>().0.iter_mut();

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
            let mut store_0 = vld;
            let mut store_1 = vld;
            let mut store_2 = vld;
            let mut store_3 = vld;

            let src0 = src;
            let src1 = src0.get_unchecked(src_stride..);
            let src2 = src1.get_unchecked(src_stride..);
            let src3 = src2.get_unchecked(src_stride..);

            while WIDE && jx + 8 <= bounds.size {
                let weights8 = _mm_loadu_si128(weights.as_ptr().add(jx).cast());
                let start = (bounds.start + jx) * CN;
                store_0 = convolve_rgb_x8(src0.get_unchecked(start..), store_0, weights8);
                store_1 = convolve_rgb_x8(src1.get_unchecked(start..), store_1, weights8);
                store_2 = convolve_rgb_x8(src2.get_unchecked(start..), store_2, weights8);
                store_3 = convolve_rgb_x8(src3.get_unchecked(start..), store_3, weights8);
                jx += 8;
            }

            // Will make step in 4 items however since it is RGB it is necessary to make a safe offset
            while jx + 4 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = _mm_loadu_si64(w_ptr.as_ptr() as *const u8);
                const SHUFFLE_01: i32 = shuffle(0, 0, 0, 0);
                let weight01 = _mm_shuffle_epi32::<SHUFFLE_01>(weights);
                const SHUFFLE_23: i32 = shuffle(1, 1, 1, 1);
                let weight23 = _mm_shuffle_epi32::<SHUFFLE_23>(weights);
                let bounds_start = (bounds.start + jx) * CN;

                let rgb_pixel_0 = load_rgb_x4(src0.get_unchecked(bounds_start..));
                let rgb_pixel_1 = load_rgb_x4(src1.get_unchecked(bounds_start..));
                let rgb_pixel_2 = load_rgb_x4(src2.get_unchecked(bounds_start..));
                let rgb_pixel_4 = load_rgb_x4(src3.get_unchecked(bounds_start..));

                let hi_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_hi);
                let lo_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_lo);
                let hi_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_hi);
                let lo_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_lo);
                let hi_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_hi);
                let lo_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_lo);
                let hi_3 = _mm_shuffle_epi8(rgb_pixel_4, shuffle_hi);
                let lo_3 = _mm_shuffle_epi8(rgb_pixel_4, shuffle_lo);

                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo_0, weight01));
                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(hi_0, weight23));

                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo_1, weight01));
                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(hi_1, weight23));

                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo_2, weight01));
                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(hi_2, weight23));

                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo_3, weight01));
                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(hi_3, weight23));
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let bounds_start = (bounds.start + jx) * CN;
                let weight01 = _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());

                let rgb_pixel_0 = load_rgb_x2(src0.get_unchecked(bounds_start..));
                let rgb_pixel_1 = load_rgb_x2(src1.get_unchecked(bounds_start..));
                let rgb_pixel_2 = load_rgb_x2(src2.get_unchecked(bounds_start..));
                let rgb_pixel_4 = load_rgb_x2(src3.get_unchecked(bounds_start..));

                let lo_0 = _mm_shuffle_epi8(rgb_pixel_0, shuffle_lo);
                let lo_1 = _mm_shuffle_epi8(rgb_pixel_1, shuffle_lo);
                let lo_2 = _mm_shuffle_epi8(rgb_pixel_2, shuffle_lo);
                let lo_3 = _mm_shuffle_epi8(rgb_pixel_4, shuffle_lo);

                store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(lo_0, weight01));
                store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(lo_1, weight01));
                store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(lo_2, weight01));
                store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(lo_3, weight01));

                jx += 2;
            }

            while jx < bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let bounds_start = bounds.start + jx;

                let weight0 = _mm_set1_epi32(w_ptr.as_ptr().read_unaligned() as i32);

                store_0 =
                    convolve_horizontal_parts_one_sse_rgb(bounds_start, src0, weight0, store_0);
                store_1 =
                    convolve_horizontal_parts_one_sse_rgb(bounds_start, src1, weight0, store_1);
                store_2 =
                    convolve_horizontal_parts_one_sse_rgb(bounds_start, src2, weight0, store_2);
                store_3 =
                    convolve_horizontal_parts_one_sse_rgb(bounds_start, src3, weight0, store_3);
                jx += 1;
            }

            let shifted0 = _mm_srai_epi32::<{ crate::support::PRECISION }>(store_0);
            let shifted1 = _mm_srai_epi32::<{ crate::support::PRECISION }>(store_1);
            let shifted2 = _mm_srai_epi32::<{ crate::support::PRECISION }>(store_2);
            let shifted3 = _mm_srai_epi32::<{ crate::support::PRECISION }>(store_3);
            let packed01 = _mm_packus_epi32(shifted0, shifted1);
            let packed23 = _mm_packus_epi32(shifted2, shifted3);
            let packed = _mm_packus_epi16(packed01, packed23);
            let element_0 = _mm_extract_epi32::<0>(packed);
            let element_1 = _mm_extract_epi32::<1>(packed);
            let element_2 = _mm_extract_epi32::<2>(packed);
            let element_3 = _mm_extract_epi32::<3>(packed);

            let bytes = element_0.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (chunk0.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
            *chunk0.get_unchecked_mut(2) = bytes[2];

            let bytes = element_1.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (chunk1.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
            *chunk1.get_unchecked_mut(2) = bytes[2];

            let bytes = element_2.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (chunk2.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
            *chunk2.get_unchecked_mut(2) = bytes[2];

            let bytes = element_3.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (chunk3.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
            *chunk3.get_unchecked_mut(2) = bytes[2];
        }
    }
}

pub(crate) fn convolve_horizontal_rgb_sse_row_one(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
    _: u32,
) {
    unsafe {
        if filter_weights.kernel_size >= 8 {
            convolve_horizontal_rgb_sse_row_one_impl::<true>(src, dst, filter_weights);
        } else {
            convolve_horizontal_rgb_sse_row_one_impl::<false>(src, dst, filter_weights);
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn convolve_horizontal_rgb_sse_row_one_impl<const WIDE: bool>(
    src: &[u8],
    dst: &mut [u8],
    filter_weights: &FilterWeights<i16>,
) {
    unsafe {
        const CN: usize = 3;

        let shuffle_lo = _mm_setr_epi8(0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1);

        let shuffle_hi = _mm_setr_epi8(6, -1, 9, -1, 7, -1, 10, -1, 8, -1, 11, -1, -1, -1, -1, -1);

        for ((dst, bounds), weights) in dst
            .as_chunks_mut::<CN>()
            .0
            .iter_mut()
            .zip(filter_weights.bounds.iter())
            .zip(
                filter_weights
                    .weights
                    .chunks_exact(filter_weights.aligned_size),
            )
        {
            let bounds_size = bounds.size;
            let mut jx = 0usize;
            let mut store = _mm_set1_epi32(ROUNDING_CONST);

            while WIDE && jx + 8 <= bounds.size {
                let weights8 = _mm_loadu_si128(weights.as_ptr().add(jx).cast());
                let start = (bounds.start + jx) * CN;
                store = convolve_rgb_x8(src.get_unchecked(start..), store, weights8);
                jx += 8;
            }

            while jx + 4 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 4));
                let weights = _mm_loadu_si64(w_ptr.as_ptr() as *const u8);
                const SHUFFLE_01: i32 = shuffle(0, 0, 0, 0);
                let weight01 = _mm_shuffle_epi32::<SHUFFLE_01>(weights);
                const SHUFFLE_23: i32 = shuffle(1, 1, 1, 1);
                let weight23 = _mm_shuffle_epi32::<SHUFFLE_23>(weights);
                let bounds_start = bounds.start + jx;
                let src_ptr_0 = src.get_unchecked((bounds_start * CN)..);

                let rgb_pixel = load_rgb_x4(src_ptr_0);

                let hi = _mm_shuffle_epi8(rgb_pixel, shuffle_hi);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);

                store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight01));
                store = _mm_add_epi32(store, _mm_madd_epi16(hi, weight23));
                jx += 4;
            }

            while jx + 2 <= bounds.size {
                let w_ptr = weights.get_unchecked(jx..(jx + 2));
                let weight0 = _mm_set1_epi32((w_ptr.as_ptr() as *const i32).read_unaligned());
                let src_ptr = src.get_unchecked(((bounds.start + jx) * 3)..);
                let rgb_pixel = load_rgb_x2(src_ptr);
                let lo = _mm_shuffle_epi8(rgb_pixel, shuffle_lo);
                store = _mm_add_epi32(store, _mm_madd_epi16(lo, weight0));
                jx += 2;
            }

            while jx < bounds_size {
                let w_ptr = weights.get_unchecked(jx..(jx + 1));
                let weight0 = _mm_set1_epi32(w_ptr.as_ptr().read_unaligned() as i32);
                store =
                    convolve_horizontal_parts_one_sse_rgb(bounds.start + jx, src, weight0, store);
                jx += 1;
            }

            let store_16_8 = compress_i32(store);
            let store_16_8 = _mm_packus_epi16(store_16_8, store_16_8);

            let element = _mm_extract_epi32::<0>(store_16_8);
            let bytes = element.to_le_bytes();
            let first_byte = u16::from_le_bytes([bytes[0], bytes[1]]);
            (dst.as_mut_ptr() as *mut u16).write_unaligned(first_byte);
            *dst.get_unchecked_mut(2) = bytes[2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_weights::FilterBounds;

    fn check_rows(four_rows: bool) {
        if !std::arch::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for width in [1usize, 2, 3, 7, 19, 33] {
            for taps in [
                1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 31, 32,
                33, 47, 48, 49, 63, 64, 65, 127, 255,
            ] {
                let source_width = taps + width + 2;
                for padding in [0, 5] {
                    let source_stride = source_width * 3 + padding;
                    let mut source = vec![0x9b; source_stride * 3 + source_width * 3];
                    for row in 0..4 {
                        for x in 0..source_width {
                            for c in 0..3 {
                                source[row * source_stride + x * 3 + c] =
                                    (row * 67 + x * 29 + c * 43 + x * c * 11) as u8;
                            }
                        }
                    }
                    let aligned = taps.next_multiple_of(4);
                    for clipped in [false, true] {
                        let bounds: Vec<_> = (0..width)
                            .map(|x| {
                                let size = if clipped && x == 0 {
                                    taps.div_ceil(2)
                                } else if clipped {
                                    taps - (x % 4).min(taps - 1)
                                } else {
                                    taps
                                };
                                let start = if x == width - 1 {
                                    source_width - size
                                } else {
                                    x + 2
                                };
                                FilterBounds::new(start, size)
                            })
                            .collect();
                        for pattern in 0..4 {
                            // Unused coefficient slots are deliberately nonzero.
                            let mut weights = vec![i16::MIN; aligned * width];
                            for (x, bound) in bounds.iter().enumerate() {
                                let active = &mut weights[x * aligned..][..bound.size];
                                for (j, w) in active.iter_mut().enumerate() {
                                    *w = match pattern {
                                        0 => ((j * 997 + x * 53) % 4096) as i16 - 2048,
                                        1 => {
                                            if j % 2 == 0 {
                                                i16::MIN
                                            } else {
                                                i16::MAX
                                            }
                                        }
                                        2 => i16::MAX,
                                        _ => {
                                            if j < bound.size.div_ceil(2) {
                                                i16::MAX
                                            } else {
                                                i16::MIN
                                            }
                                        }
                                    };
                                }
                                if pattern == 0 {
                                    active[0] = 30000;
                                    let mut residual =
                                        32768 - active.iter().map(|&v| i32::from(v)).sum::<i32>();
                                    for w in active {
                                        let delta = residual.clamp(
                                            i32::from(i16::MIN) - i32::from(*w),
                                            i32::from(i16::MAX) - i32::from(*w),
                                        );
                                        *w = (i32::from(*w) + delta) as i16;
                                        residual -= delta;
                                    }
                                }
                            }
                            let filter = FilterWeights::new(
                                weights,
                                taps,
                                aligned,
                                width,
                                0,
                                bounds.clone(),
                            );
                            let target_stride = width * 3 + padding;
                            let len = target_stride * 3 + width * 3;
                            let mut expected = vec![0xa5; len + 2];
                            for row in 0..4 {
                                for (x, bound) in bounds.iter().enumerate() {
                                    for c in 0..3 {
                                        let mut sum = ROUNDING_CONST;
                                        for j in 0..bound.size {
                                            sum = sum.wrapping_add(
                                                i32::from(
                                                    source[row * source_stride
                                                        + (bound.start + j) * 3
                                                        + c],
                                                ) * i32::from(filter.weights[x * aligned + j]),
                                            );
                                        }
                                        // Preserve the existing unsigned-word then signed-word byte packing.
                                        let word = (sum >> 15).clamp(0, 65535) as u16;
                                        expected[1 + row * target_stride + x * 3 + c] =
                                            (word as i16).clamp(0, 255) as u8;
                                    }
                                }
                            }
                            let mut actual = vec![0xa5; len + 2];
                            if four_rows {
                                convolve_horizontal_rgb_sse_rows_4(
                                    &source,
                                    source_stride,
                                    &mut actual[1..len + 1],
                                    target_stride,
                                    &filter,
                                    8,
                                );
                            } else {
                                for row in 0..4 {
                                    convolve_horizontal_rgb_sse_row_one(
                                        &source[row * source_stride..][..source_width * 3],
                                        &mut actual[1 + row * target_stride..][..width * 3],
                                        &filter,
                                        8,
                                    );
                                }
                            }
                            assert_eq!(
                                actual, expected,
                                "four_rows={four_rows}, width={width}, taps={taps}, padding={padding}, clipped={clipped}, pattern={pattern}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sse_rgb_one_row_matches_scalar_q15_exact_ends() {
        check_rows(false);
    }

    #[test]
    fn sse_rgb_four_rows_match_scalar_q15_exact_ends() {
        check_rows(true);
    }
}
