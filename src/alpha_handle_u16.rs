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

#[macro_export]
macro_rules! unpremultiply_pixel_u16 {
    ($dst: expr, $src: expr, $pixel_offset: expr, $max_colors: expr) => {{
        let mut r = *unsafe { $src.get_unchecked($pixel_offset) } as i64;
        let mut g = *unsafe { $src.get_unchecked($pixel_offset + 1) } as i64;
        let mut b = *unsafe { $src.get_unchecked($pixel_offset + 2) } as i64;
        let a = *unsafe { $src.get_unchecked($pixel_offset + 3) } as i64;
        if a != 0 {
            r = ((r * $max_colors) / a).min($max_colors);
            g = ((g * $max_colors) / a).min($max_colors);
            b = ((b * $max_colors) / a).min($max_colors);
        } else {
            r = 0;
            g = 0;
            b = 0;
        }
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = r as u16;
            *$dst.get_unchecked_mut($pixel_offset + 1) = g as u16;
            *$dst.get_unchecked_mut($pixel_offset + 2) = b as u16;
            *$dst.get_unchecked_mut($pixel_offset + 3) = a as u16;
        }
    }};
}

#[macro_export]
macro_rules! premultiply_pixel_u16 {
    ($dst: expr, $src: expr, $pixel_offset: expr, $max_colors: expr) => {{
        let mut r = *unsafe { $src.get_unchecked($pixel_offset) } as i64;
        let mut g = *unsafe { $src.get_unchecked($pixel_offset + 1) } as i64;
        let mut b = *unsafe { $src.get_unchecked($pixel_offset + 2) } as i64;
        let a = *unsafe { $src.get_unchecked($pixel_offset + 3) } as i64;
        r *= a;
        g *= a;
        b *= a;
        r /= $max_colors;
        g /= $max_colors;
        b /= $max_colors;
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = r as u16;
            *$dst.get_unchecked_mut($pixel_offset + 1) = g as u16;
            *$dst.get_unchecked_mut($pixel_offset + 2) = b as u16;
            *$dst.get_unchecked_mut($pixel_offset + 3) = a as u16;
        }
    }};
}

fn premultiply_alpha_rgba_impl(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
) {
    let mut offset = 0usize;

    let max_colors = 2i64.pow(bit_depth as u32) - 1;

    for _ in 0..height {
        for x in 0..width {
            let px = x * 4;
            premultiply_pixel_u16!(dst, src, offset + px, max_colors);
        }

        offset += 4 * width;
    }
}

fn unpremultiply_alpha_rgba_impl(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
) {
    let mut offset = 0usize;

    let max_colors = 2i64.pow(bit_depth as u32) - 1;

    for _ in 0..height {
        for x in 0..width {
            let px = x * 4;
            let pixel_offset = offset + px;
            unpremultiply_pixel_u16!(dst, src, pixel_offset, max_colors);
        }

        offset += 4 * width;
    }
}

pub fn premultiply_alpha_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
) {
    let mut _dispatcher: fn(&mut [u16], &[u16], usize, usize, usize) = premultiply_alpha_rgba_impl;
    _dispatcher(dst, src, width, height, bit_depth);
}

pub fn unpremultiply_alpha_rgba_u16(
    dst: &mut [u16],
    src: &[u16],
    width: usize,
    height: usize,
    bit_depth: usize,
) {
    let mut _dispatcher: fn(&mut [u16], &[u16], usize, usize, usize) =
        unpremultiply_alpha_rgba_impl;
    _dispatcher(dst, src, width, height, bit_depth);
}
