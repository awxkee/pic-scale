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
use std::arch::aarch64::*;

/// Checks if image has constant alpha by xor rows
pub(crate) fn neon_has_non_constant_cap_alpha_rgba8(
    store: &[u8],
    width: usize,
    stride: usize,
) -> bool {
    unsafe {
        if store.is_empty() {
            return true;
        }

        let first_alpha = store[3];

        let v_first_alpha = vdupq_n_u8(first_alpha);

        for row in store.chunks_exact(stride) {
            let row = &row[0..width * 4];
            let mut sums = vdupq_n_u32(0);
            for chunk in row.chunks_exact(16 * 4) {
                let loaded = vld4q_u8(chunk.as_ptr());
                let blend_result = veorq_u8(loaded.3, v_first_alpha);
                let blend32 = vpaddlq_u16(vpaddlq_u8(blend_result));
                sums = vaddq_u32(sums, blend32);
            }

            let row = row.chunks_exact(16 * 4).remainder();

            for chunk in row.chunks_exact(8 * 4) {
                let loaded = vld4_u8(chunk.as_ptr());
                let blend_result = veor_u8(loaded.3, vget_low_u8(v_first_alpha));
                let blend32 = vpaddl_u16(vpaddl_u8(blend_result));
                sums = vaddq_u32(sums, vcombine_u32(blend32, blend32));
            }

            let row = row.chunks_exact(8 * 4).remainder();

            let mut h_sum = vaddvq_u32(sums);

            for chunk in row.chunks_exact(4) {
                h_sum += chunk[3] as u32 ^ first_alpha as u32;
            }

            if h_sum != 0 {
                return true;
            }
        }

        false
    }
}

/// Checks if image has constant alpha by xor rows for image 16bits
pub(crate) fn neon_has_non_constant_cap_alpha_rgba16(
    store: &[u16],
    width: usize,
    stride: usize,
) -> bool {
    unsafe {
        if store.is_empty() {
            return true;
        }

        let first_alpha = store[3];
        let def_alpha = vdupq_n_u16(first_alpha);

        for row in store.chunks_exact(stride) {
            let row = &row[0..width * 4];
            let mut sums = vdupq_n_u32(0);
            for chunk in row.chunks_exact(8 * 4) {
                let r0 = vld4q_u16(chunk.as_ptr());

                let pxor = veorq_u16(r0.3, def_alpha);
                sums = vaddq_u32(sums, vpaddlq_u16(pxor));
            }

            let row = row.chunks_exact(8 * 4).remainder();

            for chunk in row.chunks_exact(4 * 4) {
                let r0 = vld4_u16(chunk.as_ptr());

                let pxor = veor_u16(r0.3, vget_low_u16(def_alpha));
                let pw = vpaddl_u16(pxor);
                sums = vaddq_u32(sums, vcombine_u32(pw, pw));
            }

            let row = row.chunks_exact(4 * 4).remainder();

            let mut h_sum = vaddvq_u32(sums);

            for chunk in row.chunks_exact(4) {
                h_sum += chunk[3] as u32 ^ first_alpha as u32;
            }

            if h_sum != 0 {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_alpha_exists_rgba8() {
        let image_size = 256usize;
        let mut image = vec![0u8; image_size * image_size * 4];
        image[3 + 150 * 4] = 75;
        let has_alpha = neon_has_non_constant_cap_alpha_rgba8(&image, image_size, image_size * 4);
        assert_eq!(true, has_alpha);
    }

    #[test]
    fn check_alpha_not_exists_rgba8() {
        let image_size = 256usize;
        let image = vec![255u8; image_size * image_size * 4];
        let has_alpha = neon_has_non_constant_cap_alpha_rgba8(&image, image_size, image_size * 4);
        assert_eq!(false, has_alpha);
    }

    #[test]
    fn check_alpha_not_exists_rgba16() {
        let image_size = 256usize;
        let image = vec![255u16; image_size * image_size * 4];
        let has_alpha = neon_has_non_constant_cap_alpha_rgba16(&image, image_size, image_size * 4);
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
        let has_alpha = neon_has_non_constant_cap_alpha_rgba16(&image, image_size, image_size * 4);
        assert_eq!(true, has_alpha);
    }
}
