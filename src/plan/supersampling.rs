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
use crate::{ImageSize, ResamplingFunction};

/// Choose the cheapest pre-filter for the supersampling first pass.
///
/// The goal is to rapidly reduce the source to ~2× the target size so the
/// final quality filter has a manageable input. The pre-filter does not need
/// to be high quality — it just needs to be fast and not alias badly.
pub(crate) fn supersampling_prefilter(ratio_w: f64, ratio_h: f64) -> Option<ResamplingFunction> {
    let ratio = ratio_w.max(ratio_h);
    if ratio >= 4.0 {
        Some(ResamplingFunction::Nearest)
    } else if ratio >= 3.0 {
        Some(ResamplingFunction::Box)
    } else {
        None
    }
}

/// Compute the intermediate size for a supersampling pre-pass.
///
/// We target ~2× the destination in each axis, clamped to [dst, src].
/// This gives the quality filter a ~2× downscale to work with, which is
/// within every filter's optimal range.
pub(crate) fn supersampling_intermediate_size(src: ImageSize, dst: ImageSize) -> ImageSize {
    // 2× the destination, but never larger than source or smaller than dst.
    let w = (dst.width * 2).min(src.width).max(dst.width);
    let h = (dst.height * 2).min(src.height).max(dst.height);
    ImageSize::new(w, h)
}
