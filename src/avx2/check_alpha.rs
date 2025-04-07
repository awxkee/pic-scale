/*
 * Copyright (c) Radzivon Bartoshyk 12/2024. All rights reserved.
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

/// Checks if image has constant alpha by xor rows
pub(crate) fn avx_has_non_constant_cap_alpha_rgba8(
    store: &[u8],
    width: usize,
    stride: usize,
) -> bool {
    unsafe { avx_has_non_constant_cap_alpha_rgba8_impl(store, width, stride) }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_has_non_constant_cap_alpha_rgba8_impl(
    store: &[u8],
    width: usize,
    stride: usize,
) -> bool {
    if store.is_empty() {
        return true;
    }

    let ash0 = _mm256_setr_epi8(
        3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1, 3, -1, -1, -1, 7, -1, -1, -1,
        11, -1, -1, -1, 15, -1, -1, -1,
    );

    let sh0 = _mm_setr_epi8(3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1);

    let first_alpha = store[3];
    let def_alpha = _mm256_set1_epi32(first_alpha as i32);

    for row in store.chunks_exact(stride) {
        let row = &row[0..width * 4];
        let mut sums = _mm256_set1_epi32(0);

        for chunk in row.chunks_exact(32 * 4) {
            let mut r0 = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let mut r1 = _mm256_loadu_si256(chunk.get_unchecked(32..).as_ptr() as *const __m256i);
            let mut r2 = _mm256_loadu_si256(chunk.get_unchecked(64..).as_ptr() as *const __m256i);
            let mut r3 = _mm256_loadu_si256(chunk.get_unchecked(96..).as_ptr() as *const __m256i);

            r0 = _mm256_xor_si256(_mm256_shuffle_epi8(r0, ash0), def_alpha);
            r1 = _mm256_xor_si256(_mm256_shuffle_epi8(r1, ash0), def_alpha);
            r2 = _mm256_xor_si256(_mm256_shuffle_epi8(r2, ash0), def_alpha);
            r3 = _mm256_xor_si256(_mm256_shuffle_epi8(r3, ash0), def_alpha);

            sums = _mm256_add_epi32(sums, r0);
            sums = _mm256_add_epi32(sums, r1);
            sums = _mm256_add_epi32(sums, r2);
            sums = _mm256_add_epi32(sums, r3);
        }

        let row = row.chunks_exact(32 * 4).remainder();

        for chunk in row.chunks_exact(32) {
            let mut r0 = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            r0 = _mm256_xor_si256(_mm256_shuffle_epi8(r0, ash0), def_alpha);

            sums = _mm256_add_epi32(sums, r0);
        }

        let row = row.chunks_exact(32).remainder();

        let mut sums = _mm_add_epi32(
            _mm256_castsi256_si128(sums),
            _mm256_extracti128_si256::<1>(sums),
        );
        let def_alpha = _mm_set1_epi32(first_alpha as i32);

        for chunk in row.chunks_exact(16) {
            let mut r0 = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);

            r0 = _mm_shuffle_epi8(r0, sh0);

            let alphas = _mm_xor_si128(r0, def_alpha);

            sums = _mm_add_epi32(sums, alphas);
        }

        let row = row.chunks_exact(16).remainder();

        use crate::avx2::routines::_mm_hsum_epi32;
        let mut h_sum = _mm_hsum_epi32(sums);

        for chunk in row.chunks_exact(4) {
            h_sum += chunk[3] as i32 ^ first_alpha as i32;
        }

        if h_sum != 0 {
            return true;
        }
    }
    false
}

/// Checks if image has constant alpha by xor rows for image 16bits
pub(crate) fn avx_has_non_constant_cap_alpha_rgba16(
    store: &[u16],
    width: usize,
    stride: usize,
) -> bool {
    unsafe { avx_has_non_constant_cap_alpha_rgba16_impl(store, width, stride) }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_has_non_constant_cap_alpha_rgba16_impl(
    store: &[u16],
    width: usize,
    stride: usize,
) -> bool {
    if store.is_empty() {
        return true;
    }

    let ash0 = _mm256_setr_epi8(
        6, 7, -1, -1, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, 7, -1, -1, 14, 15, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    );

    let first_alpha = store[3];
    let def_alpha = _mm256_set1_epi32(first_alpha as i32);

    for row in store.chunks_exact(stride) {
        let row = &row[0..width * 4];
        let mut sums = _mm256_set1_epi32(0);
        for chunk in row.chunks_exact(16 * 4) {
            let mut r0 = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let mut r1 = _mm256_loadu_si256(chunk.get_unchecked(16..).as_ptr() as *const __m256i);
            let mut r2 = _mm256_loadu_si256(chunk.get_unchecked(32..).as_ptr() as *const __m256i);
            let mut r3 = _mm256_loadu_si256(chunk.get_unchecked(48..).as_ptr() as *const __m256i);

            r0 = _mm256_shuffle_epi8(r0, ash0);
            r1 = _mm256_shuffle_epi8(r1, ash0);
            r2 = _mm256_shuffle_epi8(r2, ash0);
            r3 = _mm256_shuffle_epi8(r3, ash0);

            let r01 = _mm256_xor_si256(_mm256_unpacklo_epi32(r0, r1), def_alpha);
            let r23 = _mm256_xor_si256(_mm256_unpacklo_epi32(r2, r3), def_alpha);

            sums = _mm256_add_epi32(sums, r01);
            sums = _mm256_add_epi32(sums, r23);
        }

        let row = row.chunks_exact(16 * 4).remainder();

        for chunk in row.chunks_exact(16) {
            let mut r0 = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            r0 = _mm256_shuffle_epi8(r0, ash0);

            let alphas = _mm256_xor_si256(_mm256_unpacklo_epi32(r0, r0), def_alpha);

            sums = _mm256_add_epi32(sums, alphas);
        }

        let row = row.chunks_exact(16).remainder();

        use crate::avx2::routines::_mm_hsum_epi32;
        let mut h_sum = _mm_hsum_epi32(_mm_add_epi32(
            _mm256_castsi256_si128(sums),
            _mm256_extracti128_si256::<1>(sums),
        ));

        for chunk in row.chunks_exact(4) {
            h_sum += chunk[3] as i32 ^ first_alpha as i32;
        }

        if h_sum != 0 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_alpha_exists_rgba8() {
        let image_size = 256usize;
        let mut image = vec![0u8; image_size * image_size * 4];
        image[3 + 150 * 4] = 75;
        let has_alpha = avx_has_non_constant_cap_alpha_rgba8(&image, image_size, image_size * 4);
        assert_eq!(true, has_alpha);
    }

    #[test]
    fn check_alpha_not_exists_rgba8() {
        let image_size = 256usize;
        let image = vec![255u8; image_size * image_size * 4];
        let has_alpha = avx_has_non_constant_cap_alpha_rgba8(&image, image_size, image_size * 4);
        assert_eq!(false, has_alpha);
    }

    #[test]
    fn check_alpha_exists_rgba16() {
        let image_size = 256usize;
        let mut image = vec![0u16; image_size * image_size * 4];
        image[3] = 715;
        image[7] = 715;
        image[11] = 715;
        image[15] = 715;
        let has_alpha = avx_has_non_constant_cap_alpha_rgba16(&image, image_size, image_size * 4);
        assert_eq!(true, has_alpha);
    }

    #[test]
    fn check_alpha_not_exists_rgba16() {
        let image_size = 256usize;
        let image = vec![255u16; image_size * image_size * 4];
        let has_alpha = avx_has_non_constant_cap_alpha_rgba16(&image, image_size, image_size * 4);
        assert_eq!(false, has_alpha);
    }
}
