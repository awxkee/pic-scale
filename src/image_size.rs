/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[derive(Copy, Clone)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

impl ImageSize {
    pub fn new(width: usize, height: usize) -> ImageSize {
        ImageSize { width, height }
    }
}