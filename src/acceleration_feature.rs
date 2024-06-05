/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
#[allow(dead_code)]
pub(crate) enum AccelerationFeature {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[allow(dead_code)]
    Neon,
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[allow(dead_code)]
    Sse,
    #[allow(dead_code)]
    Native
}