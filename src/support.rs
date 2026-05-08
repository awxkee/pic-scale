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
#![forbid(unsafe_code)]
pub(crate) const PRECISION: i32 = 15;
pub(crate) const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);

pub(crate) fn check_image_size_overflow(
    width: usize,
    height: usize,
    chan: usize,
    t_size: isize,
) -> bool {
    let Ok(w) = isize::try_from(width) else {
        return true;
    };
    let Ok(h) = isize::try_from(height) else {
        return true;
    };
    let Ok(n) = isize::try_from(chan) else {
        return true;
    };
    let Some(stride) = w.checked_mul(n) else {
        return true;
    };

    // stride * (height - 1) + width * N
    let Some(h_minus_1) = h.checked_sub(1) else {
        return true;
    };
    let Some(lhs) = stride.checked_mul(h_minus_1) else {
        return true;
    };
    lhs.checked_add(stride)
        .and_then(|x| x.checked_mul(t_size))
        .is_none()
}

pub(crate) fn check_image_size_overflow_with_stride(
    width: usize,
    height: usize,
    stride: usize,
    chan: usize,
    t_size: isize,
) -> bool {
    let Ok(w) = isize::try_from(width) else {
        return true;
    };
    let Ok(h) = isize::try_from(height) else {
        return true;
    };
    let Ok(n) = isize::try_from(chan) else {
        return true;
    };
    let Ok(stride) = isize::try_from(stride) else {
        return true;
    };

    // stride * (height - 1) + width * N
    let Some(h_minus_1) = h.checked_sub(1) else {
        return true;
    };
    let Some(lhs) = stride.checked_mul(h_minus_1) else {
        return true;
    };
    let Some(rhs) = w.checked_mul(n) else {
        return true;
    };
    lhs.checked_add(rhs)
        .and_then(|x| x.checked_mul(t_size))
        .is_none()
}
