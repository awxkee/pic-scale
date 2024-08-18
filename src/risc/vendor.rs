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
use std::arch::asm;

#[inline]
#[target_feature(enable = "v")]
pub unsafe fn xvsetvlmax_e8m1() -> usize {
    let mut result: usize;
    asm!(
    "\
    li {t0}, -1
    vsetvli {0}, {t0}, e8, m1, ta, ma",
    out(reg) result,
    t0 = out(reg) _,
    options(pure, nomem, nostack));
    result
}

#[inline]
#[target_feature(enable = "v")]
pub unsafe fn xvsetvlmax_f32m1() -> usize {
    let mut result: usize;
    asm!(
    "\
    li {t0}, -1
    vsetvli {0}, {t0}, e32, m1, ta, ma\
    ",
    out(reg) result,
    t0 = out(reg) _,
    options(pure, nomem, nostack));
    result
}

#[inline]
#[allow(dead_code)]
#[target_feature(enable = "v,zfh")]
pub unsafe fn xvsetvlmax_f16m1() -> usize {
    let mut result: usize;
    asm!(
    "\
    li {t0}, -1
    vsetvli {0}, {t0}, e16, m1, ta, ma\
    ",
    out(reg) result,
    t0 = out(reg) _,
    options(pure, nomem, nostack));
    result
}


#[inline]
#[target_feature(enable = "v")]
#[allow(dead_code)]
pub unsafe fn xvsetvlmax_f32m1_k(k: usize) -> usize {
    let mut result: usize;
    asm!(
    "\
    vsetvli {0}, {1}, e32, m1, ta, ma\
    ",
    out(reg) result,
    in(reg) k,
    options(pure, nomem, nostack));
    result
}
