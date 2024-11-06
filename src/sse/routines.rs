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

macro_rules! load_4_weights {
    ($src_ptr: expr) => {{
        let weight = _mm_loadu_ps($src_ptr);
        const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
        let weight0 = _mm_shuffle_ps::<SHUFFLE_0>(weight, weight);
        const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
        let weight1 = _mm_shuffle_ps::<SHUFFLE_1>(weight, weight);
        const SHUFFLE_2: i32 = shuffle(2, 2, 2, 2);
        let weight2 = _mm_shuffle_ps::<SHUFFLE_2>(weight, weight);
        const SHUFFLE_3: i32 = shuffle(3, 3, 3, 3);
        let weight3 = _mm_shuffle_ps::<SHUFFLE_3>(weight, weight);
        (weight0, weight1, weight2, weight3)
    }};
}

pub(crate) use load_4_weights;

macro_rules! load_4_weights_group_2_avx {
    ($src_ptr: expr) => {{
        let weight = _mm_loadu_ps($src_ptr);
        const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
        let weight0 = _mm_shuffle_ps::<SHUFFLE_0>(weight, weight);
        const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
        let weight1 = _mm_shuffle_ps::<SHUFFLE_1>(weight, weight);
        const SHUFFLE_2: i32 = shuffle(2, 2, 2, 2);
        let weight2 = _mm_shuffle_ps::<SHUFFLE_2>(weight, weight);
        const SHUFFLE_3: i32 = shuffle(3, 3, 3, 3);
        let weight3 = _mm_shuffle_ps::<SHUFFLE_3>(weight, weight);
        (
            avx_combine_ps(weight0, weight1),
            avx_combine_ps(weight2, weight3),
        )
    }};
}

pub(crate) use load_4_weights_group_2_avx;

macro_rules! load_8_weights_group_4_avx {
    ($src_ptr: expr) => {{
        let weight_row_0 = _mm_loadu_ps($src_ptr);
        const SHUFFLE_0: i32 = shuffle(0, 0, 0, 0);
        let weight0 = _mm_shuffle_ps::<SHUFFLE_0>(weight_row_0, weight_row_0);
        const SHUFFLE_1: i32 = shuffle(1, 1, 1, 1);
        let weight1 = _mm_shuffle_ps::<SHUFFLE_1>(weight_row_0, weight_row_0);
        const SHUFFLE_2: i32 = shuffle(2, 2, 2, 2);
        let weight2 = _mm_shuffle_ps::<SHUFFLE_2>(weight_row_0, weight_row_0);
        const SHUFFLE_3: i32 = shuffle(3, 3, 3, 3);
        let weight3 = _mm_shuffle_ps::<SHUFFLE_3>(weight_row_0, weight_row_0);

        let weight_row_1 = _mm_loadu_ps($src_ptr.add(4));
        let weight4 = _mm_shuffle_ps::<SHUFFLE_0>(weight_row_1, weight_row_1);
        let weight5 = _mm_shuffle_ps::<SHUFFLE_1>(weight_row_1, weight_row_1);
        let weight6 = _mm_shuffle_ps::<SHUFFLE_2>(weight_row_1, weight_row_1);
        let weight7 = _mm_shuffle_ps::<SHUFFLE_3>(weight_row_1, weight_row_1);
        (
            avx_combine_ps(weight0, weight1),
            avx_combine_ps(weight2, weight3),
            avx_combine_ps(weight4, weight5),
            avx_combine_ps(weight6, weight7),
        )
    }};
}

pub(crate) use load_8_weights_group_4_avx;
