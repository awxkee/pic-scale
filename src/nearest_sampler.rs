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

pub fn resize_nearest<T: Copy, const CHANNELS: usize>(
    src: &[T],
    src_width: usize,
    src_height: usize,
    dst: &mut [T],
    dst_width: usize,
    dst_height: usize,
) where
    T: Copy,
{
    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    let clip_width = src_width as f32 - 1f32;
    let clip_height = src_height as f32 - 1f32;

    let dst_stride = dst_width * CHANNELS;
    let src_stride = src_width * CHANNELS;

    let mut dst_offset = 0usize;

    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = (x as f32 * x_scale + 0.5f32).min(clip_width).max(0f32) as usize;
            let src_y = (y as f32 * y_scale + 0.5f32).min(clip_height).max(0f32) as usize;
            let src_offset_y = src_y * src_stride;
            let src_px = src_x * CHANNELS;
            let dst_px = x * CHANNELS;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_offset_y + src_px),
                    dst.as_mut_ptr().add(dst_offset + dst_px),
                    CHANNELS,
                );
            }
        }

        dst_offset += dst_stride;
    }
}
