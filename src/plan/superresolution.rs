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

/// Compute the chain of intermediate sizes between `src` and `dst`.
/// Returns only the intermediate sizes — the final `dst` is not included
/// since the last plan targets it directly.
pub(crate) fn plan_intermediate_sizes(
    src: ImageSize,
    dst: ImageSize,
    function: ResamplingFunction,
) -> Vec<ImageSize> {
    let max_ratio = function
        .get_resampling_filter::<f32>()
        .min_kernel_size
        .max(1.5)
        .min(4.0) as f64;

    // For filters with no effective ratio limit just do a single step.
    if max_ratio == f64::MAX {
        return Vec::new();
    }

    // Number of steps needed per axis.
    let steps_w = if dst.width > src.width {
        let ratio = dst.width as f64 / src.width as f64;
        (ratio.log2() / max_ratio.log2()).ceil() as usize
    } else {
        0
    };
    let steps_h = if dst.height > src.height {
        let ratio = dst.height as f64 / src.height as f64;
        (ratio.log2() / max_ratio.log2()).ceil() as usize
    } else {
        0
    };
    let steps = steps_w.max(steps_h);

    if steps <= 1 {
        return Vec::new();
    }

    // Distribute steps evenly in log space.
    let mut sizes = Vec::with_capacity(steps - 1);
    for i in 1..steps {
        let t = i as f64 / steps as f64;
        let w = if dst.width > src.width {
            (src.width as f64 * (dst.width as f64 / src.width as f64).powf(t)).round() as usize
        } else {
            dst.width
        };
        let h = if dst.height > src.height {
            (src.height as f64 * (dst.height as f64 / src.height as f64).powf(t)).round() as usize
        } else {
            dst.height
        };
        let w = w.max(src.width).min(dst.width);
        let h = h.max(src.height).min(dst.height);
        sizes.push(ImageSize::new(w, h));
    }

    sizes.dedup_by(|a, b| a.width == b.width && a.height == b.height);

    sizes
}
