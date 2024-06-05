/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

pub fn chunked<I>(
    a: impl IntoIterator<Item = I>,
    chunk_size: usize,
) -> impl Iterator<Item = Vec<I>> {
    let mut a = a.into_iter();
    std::iter::from_fn(move || {
        Some(a.by_ref().take(chunk_size).collect()).filter(|chunk: &Vec<_>| !chunk.is_empty())
    })
}
