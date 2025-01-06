/*
 * Copyright (c) Radzivon Bartoshyk 01/2025. All rights reserved.
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
pub(crate) fn avx512_has_non_constant_cap_alpha_rgba8(
    store: &[u8],
    width: usize,
    stride: usize,
) -> bool {
    unsafe { avx512_has_non_constant_cap_alpha_rgba8_impl(store, width, stride) }
}

#[target_feature(enable = "avx512bw", enable = "avx512f")]
unsafe fn avx512_has_non_constant_cap_alpha_rgba8_impl(
    store: &[u8],
    width: usize,
    stride: usize,
) -> bool {
    if store.is_empty() {
        return true;
    }

    let sh0 = _mm512_set_epi8(
        -1, -1, -1, 63, -1, -1, -1, 59, -1, -1, -1, 55, -1, -1, -1, 51, -1, -1, -1, 47, -1, -1, -1,
        43, -1, -1, -1, 39, -1, -1, -1, 35, -1, -1, -1, 31, -1, -1, -1, 27, -1, -1, -1, 23, -1, -1,
        -1, 19, -1, -1, -1, 15, -1, -1, -1, 11, -1, -1, -1, 7, -1, -1, -1, 3,
    );

    let first_alpha = store[3];
    let base_mask = _mm512_set1_epi8(first_alpha as i8);
    let def_alpha = _mm512_set1_epi32(first_alpha as i32);

    for row in store.chunks_exact(stride) {
        let row = &row[0..width * 4];
        let mut sums = _mm512_set1_epi32(0);

        for chunk in row.chunks_exact(64) {
            let working_mask: __mmask64 = if chunk.len() == 64 {
                0xffff_ffff_ffff_ffff
            } else {
                0xffff_ffff_ffff_ffff >> (64 - chunk.len())
            };

            let mut r0 =
                _mm512_mask_loadu_epi8(base_mask, working_mask, chunk.as_ptr() as *const _);
            r0 = _mm512_shuffle_epi8(r0, sh0);
            r0 = _mm512_xor_si512(r0, def_alpha);

            sums = _mm512_add_epi32(sums, r0);
        }

        let h_sum = _mm512_reduce_add_epi32(sums);

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
        let has_alpha = avx512_has_non_constant_cap_alpha_rgba8(&image, image_size, image_size * 4);
        assert_eq!(true, has_alpha);
    }

    #[test]
    fn check_alpha_not_exists_rgba8() {
        let image_size = 256usize;
        let image = vec![255u8; image_size * image_size * 4];
        let has_alpha = avx512_has_non_constant_cap_alpha_rgba8(&image, image_size, image_size * 4);
        assert_eq!(false, has_alpha);
    }
}
