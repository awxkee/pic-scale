/*
 * Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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

mod rgb_u8_dot;
mod vertical_u16_dot;
mod vertical_u8_dot;

pub(crate) use rgb_u8_dot::{
    sve_convolve_horizontal_rgb_neon_row_one_dot, sve_convolve_horizontal_rgb_neon_rows_4_dot,
};
pub(crate) use vertical_u8_dot::convolve_vertical_sve2_i8_dot;
pub(crate) use vertical_u16_dot::convolve_vertical_sve2_u16_dot;

#[cfg(test)]
fn test_vector_length(require_i8mm: bool) -> Option<usize> {
    let available = std::arch::is_aarch64_feature_detected!("sve2")
        && (!require_i8mm || std::arch::is_aarch64_feature_detected!("i8mm"));
    if !available {
        assert!(
            std::env::var_os("PIC_SCALE_REQUIRE_SVE2").is_none(),
            "SVE2 tests were required, but the necessary CPU features are unavailable",
        );
        eprintln!("Skipping SVE2 test: necessary CPU features are unavailable");
        return None;
    }
    let bytes = unsafe { std::arch::aarch64::svcntb() as usize };
    if let Ok(expected) = std::env::var("PIC_SCALE_SVE_VL_BITS") {
        assert_eq!(bytes * 8, expected.parse::<usize>().unwrap());
    }
    eprintln!("Testing SVE2 with {}-bit vectors", bytes * 8);
    Some(bytes)
}
