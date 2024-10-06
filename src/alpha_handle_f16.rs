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
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::avx2::{avx_premultiply_alpha_rgba_f16, avx_unpremultiply_alpha_rgba_f16};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::cpu_features::is_aarch_f16_supported;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{neon_premultiply_alpha_rgba_f16, neon_unpremultiply_alpha_rgba_f16};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{neon_premultiply_alpha_rgba_f16_full, neon_unpremultiply_alpha_rgba_f16_full};
#[cfg(all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    feature = "riscv"
))]
use crate::risc::{
    risc_is_feature_supported, risc_premultiply_alpha_rgba_f16, risc_unpremultiply_alpha_rgba_f16,
};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{sse_premultiply_alpha_rgba_f16, sse_unpremultiply_alpha_rgba_f16};
use crate::ThreadingPolicy;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;

#[macro_export]
macro_rules! premultiply_pixel_f16 {
    ($dst: expr, $src: expr, $pixel_offset: expr) => {{
        let mut r = (*unsafe { $src.get_unchecked($pixel_offset) }).to_f32();
        let mut g = (*unsafe { $src.get_unchecked($pixel_offset + 1) }).to_f32();
        let mut b = (*unsafe { $src.get_unchecked($pixel_offset + 2) }).to_f32();
        let a = (*unsafe { $src.get_unchecked($pixel_offset + 3) }).to_f32();
        r *= a;
        g *= a;
        b *= a;
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = half::f16::from_f32(r);
            *$dst.get_unchecked_mut($pixel_offset + 1) = half::f16::from_f32(g);
            *$dst.get_unchecked_mut($pixel_offset + 2) = half::f16::from_f32(b);
            *$dst.get_unchecked_mut($pixel_offset + 3) = half::f16::from_f32(a);
        }
    }};
}

#[macro_export]
macro_rules! unpremultiply_pixel_f16 {
    ($dst: expr, $src: expr, $pixel_offset: expr) => {{
        let mut r = (*unsafe { $src.get_unchecked($pixel_offset) }).to_f32();
        let mut g = (*unsafe { $src.get_unchecked($pixel_offset + 1) }).to_f32();
        let mut b = (*unsafe { $src.get_unchecked($pixel_offset + 2) }).to_f32();
        let a = (*unsafe { $src.get_unchecked($pixel_offset + 3) }).to_f32();
        if a != 0. {
            r = r / a;
            g = g / a;
            b = b / a;
        } else {
            r = 0.;
            g = 0.;
            b = 0.;
        }
        unsafe {
            *$dst.get_unchecked_mut($pixel_offset) = half::f16::from_f32(r);
            *$dst.get_unchecked_mut($pixel_offset + 1) = half::f16::from_f32(g);
            *$dst.get_unchecked_mut($pixel_offset + 2) = half::f16::from_f32(b);
            *$dst.get_unchecked_mut($pixel_offset + 3) = half::f16::from_f32(a);
        }
    }};
}

fn premultiply_alpha_rgba_impl_f16(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
    threading_policy: ThreadingPolicy,
) {
    let allowed_threading = threading_policy.allowed_threading();

    if allowed_threading {
        src.par_chunks_exact(width * 4)
            .zip(dst.par_chunks_exact_mut(width * 4))
            .for_each(|(src, dst)| {
                let mut _cx = 0usize;

                for x in _cx..width {
                    let px = x * 4;
                    premultiply_pixel_f16!(dst, src, px);
                }
            });
    } else {
        let mut offset = 0usize;

        for _ in 0..height {
            let mut _cx = 0usize;

            for x in _cx..width {
                let px = x * 4;
                premultiply_pixel_f16!(dst, src, offset + px);
            }

            offset += 4 * width;
        }
    }
}

fn unpremultiply_alpha_rgba_impl_f16(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
    threading_policy: ThreadingPolicy,
) {
    let allowed_threading = threading_policy.allowed_threading();

    if allowed_threading {
        src.par_chunks_exact(width * 4)
            .zip(dst.par_chunks_exact_mut(width * 4))
            .for_each(|(src, dst)| {
                for x in 0..width {
                    let px = x * 4;
                    let pixel_offset = px;
                    unpremultiply_pixel_f16!(dst, src, pixel_offset);
                }
            });
    } else {
        let mut offset = 0usize;

        for _ in 0..height {
            let mut _cx = 0usize;

            for x in _cx..width {
                let px = x * 4;
                let pixel_offset = offset + px;
                unpremultiply_pixel_f16!(dst, src, pixel_offset);
            }

            offset += 4 * width;
        }
    }
}

pub fn premultiply_alpha_rgba_f16(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
    threading_policy: ThreadingPolicy,
) {
    let mut _dispatcher: fn(&mut [half::f16], &[half::f16], usize, usize, ThreadingPolicy) =
        premultiply_alpha_rgba_impl_f16;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_premultiply_alpha_rgba_f16;
        if is_aarch_f16_supported() {
            _dispatcher = neon_premultiply_alpha_rgba_f16_full;
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            _dispatcher = sse_premultiply_alpha_rgba_f16;
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            _dispatcher = avx_premultiply_alpha_rgba_f16;
        }
    }
    #[cfg(all(
        any(target_arch = "riscv64", target_arch = "riscv32"),
        feature = "riscv"
    ))]
    {
        if std::arch::is_riscv_feature_detected!("v") && risc_is_feature_supported("zvfh") {
            _dispatcher = risc_unpremultiply_alpha_rgba_f16;
        }
    }
    _dispatcher(dst, src, width, height, threading_policy);
}

pub fn unpremultiply_alpha_rgba_f16(
    dst: &mut [half::f16],
    src: &[half::f16],
    width: usize,
    height: usize,
    threading_policy: ThreadingPolicy,
) {
    let mut _dispatcher: fn(&mut [half::f16], &[half::f16], usize, usize, ThreadingPolicy) =
        unpremultiply_alpha_rgba_impl_f16;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        _dispatcher = neon_unpremultiply_alpha_rgba_f16;
        if is_aarch_f16_supported() {
            _dispatcher = neon_unpremultiply_alpha_rgba_f16_full;
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            _dispatcher = sse_unpremultiply_alpha_rgba_f16;
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            _dispatcher = avx_unpremultiply_alpha_rgba_f16;
        }
    }
    #[cfg(all(
        any(target_arch = "riscv64", target_arch = "riscv32"),
        feature = "riscv"
    ))]
    {
        if std::arch::is_riscv_feature_detected!("v") && risc_is_feature_supported("zvfh") {
            _dispatcher = risc_premultiply_alpha_rgba_f16;
        }
    }
    _dispatcher(dst, src, width, height, threading_policy);
}
