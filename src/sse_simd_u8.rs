#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod sse_convolve_u8 {

    use crate::filter_weights::FilterBounds;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    use crate::support::ROUNDING_APPROX;

    #[inline(always)]
    pub fn compress_i32(x: __m128i) -> __m128i {
        let store_32 = unsafe { _mm_srai_epi32::<12>(_mm_max_epi32(x, _mm_setzero_si128())) };
        let store_16 = unsafe { _mm_packus_epi32(store_32, store_32) };
        let store_16_8 = unsafe { _mm_packus_epi16(store_16, store_16) };
        store_16_8
    }

    #[inline(always)]
    pub(crate) unsafe fn convolve_horizontal_parts_one_sse_rgb(
        start_x: usize,
        src: *const u8,
        weight0: __m128i,
        store_0: __m128i,
    ) -> __m128i {
        const COMPONENTS: usize = 3;
        let src_ptr = src.add(start_x * COMPONENTS);
        let vl = i32::from_le_bytes([*src_ptr, *src_ptr.add(1), *src_ptr.add(2), 0]);
        let m_vl = _mm_cvtsi32_si128(vl);
        let lo = _mm_cvtepu8_epi16(m_vl);
        let acc = _mm_add_epi32(store_0, _mm_mullo_epi32(_mm_cvtepi16_epi32(lo), weight0));
        acc
    }

    #[inline(always)]
    pub(crate) unsafe fn convolve_horizontal_parts_two_sse_rgb(
        start_x: usize,
        src: *const u8,
        weight01: __m128i,
        store_0: __m128i,
        shuffle: __m128i,
    ) -> __m128i {
        const COMPONENTS: usize = 3;
        let src_ptr = src.add(start_x * COMPONENTS);
        let rgb_pixel = _mm_loadu_si64(src_ptr);
        let lo = _mm_shuffle_epi8(rgb_pixel, shuffle);
        let acc = _mm_add_epi32(store_0, _mm_madd_epi16(lo, weight01));
        acc
    }

    #[inline(always)]
    pub(crate) unsafe fn convolve_vertical_part_sse_16(
        start_y: usize,
        start_x: usize,
        src: *const u8,
        src_stride: usize,
        dst: *mut u8,
        filter: *const i16,
        bounds: &FilterBounds,
    ) {
        let vld = _mm_set1_epi32(ROUNDING_APPROX);
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;

        let px = start_x;

        let zeros = _mm_setzero_si128();

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = *unsafe { filter.add(j) };
            let v_weight = _mm_set1_epi32(weight as i32);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row = _mm_loadu_si128(s_ptr as *const __m128i);

            let low = _mm_cvtepu8_epi16(item_row);
            let high = _mm_unpackhi_epi8(item_row, zeros);

            store_0 = _mm_add_epi32(store_0, _mm_mullo_epi32(_mm_cvtepi16_epi32(low), v_weight));
            store_1 = _mm_add_epi32(
                store_1,
                _mm_mullo_epi32(_mm_unpackhi_epi16(low, zeros), v_weight),
            );
            store_2 = _mm_add_epi32(store_2, _mm_mullo_epi32(_mm_cvtepi16_epi32(high), v_weight));
            store_3 = _mm_add_epi32(
                store_3,
                _mm_mullo_epi32(_mm_unpackhi_epi16(high, zeros), v_weight),
            );
        }

        store_0 = _mm_max_epi32(store_0, zeros);
        store_1 = _mm_max_epi32(store_1, zeros);
        store_2 = _mm_max_epi32(store_2, zeros);
        store_3 = _mm_max_epi32(store_3, zeros);

        let low_16 = _mm_packs_epi32(_mm_srai_epi32::<12>(store_0), _mm_srai_epi32::<12>(store_1));
        let high_16 = _mm_packs_epi32(_mm_srai_epi32::<12>(store_2), _mm_srai_epi32::<12>(store_3));

        let item = _mm_packus_epi16(low_16, high_16);

        let dst_ptr = dst.add(px);
        _mm_storeu_si128(dst_ptr as *mut __m128i, item);
    }

    #[inline]
    pub(crate) unsafe fn convolve_vertical_part_sse_32(
        start_y: usize,
        start_x: usize,
        src: *const u8,
        src_stride: usize,
        dst: *mut u8,
        filter: *const i16,
        bounds: &FilterBounds,
    ) {
        let zeros = _mm_setzero_si128();
        let vld = _mm_set1_epi32(ROUNDING_APPROX);
        let mut store_0 = vld;
        let mut store_1 = vld;
        let mut store_2 = vld;
        let mut store_3 = vld;
        let mut store_4 = vld;
        let mut store_5 = vld;
        let mut store_6 = vld;
        let mut store_7 = vld;

        let px = start_x;

        let mut jj = 0usize;

        while jj < bounds.size.saturating_sub(2) {
            let py = start_y + jj;
            let f_ptr = filter.add(jj) as *const i32;
            let v_weight_2 = _mm_set1_epi32(f_ptr.read_unaligned());
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let s_ptr_next = s_ptr.add(src_stride);

            let item_row_0 = _mm_loadu_si128(s_ptr as *const __m128i);
            let item_row_1 = _mm_loadu_si128(s_ptr_next as *const __m128i);

            let interleaved = _mm_unpacklo_epi8(item_row_0, item_row_1);
            let pix = _mm_unpacklo_epi8(interleaved, zeros);
            store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(pix, v_weight_2));
            let pix = _mm_unpackhi_epi8(interleaved, zeros);
            store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(pix, v_weight_2));

            let interleaved = _mm_unpackhi_epi8(item_row_0, item_row_1);
            let pix = _mm_unpacklo_epi8(interleaved, zeros);
            store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(pix, v_weight_2));
            let pix = _mm_unpackhi_epi8(interleaved, zeros);
            store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(pix, v_weight_2));

            let item_row_0 = _mm_loadu_si128(s_ptr.add(16) as *const __m128i);
            let item_row_1 = _mm_loadu_si128(s_ptr_next.add(16) as *const __m128i);

            let interleaved = _mm_unpacklo_epi8(item_row_0, item_row_1);
            let pix = _mm_unpacklo_epi8(interleaved, zeros);
            store_4 = _mm_add_epi32(store_4, _mm_madd_epi16(pix, v_weight_2));
            let pix = _mm_unpackhi_epi8(interleaved, zeros);
            store_5 = _mm_add_epi32(store_5, _mm_madd_epi16(pix, v_weight_2));

            let interleaved = _mm_unpackhi_epi8(item_row_0, item_row_1);
            let pix = _mm_unpacklo_epi8(interleaved, zeros);
            store_6 = _mm_add_epi32(store_6, _mm_madd_epi16(pix, v_weight_2));
            let pix = _mm_unpackhi_epi8(interleaved, zeros);
            store_7 = _mm_add_epi32(store_7, _mm_madd_epi16(pix, v_weight_2));

            jj += 2;
        }

        for j in jj..bounds.size {
            let py = start_y + j;
            let weight = *unsafe { filter.add(j) };
            let v_weight = _mm_set1_epi32(weight as i32);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row_0 = _mm_loadu_si128(s_ptr as *const __m128i);
            let item_row_1 = _mm_loadu_si128(s_ptr.add(16) as *const __m128i);

            let interleaved = _mm_unpacklo_epi8(item_row_0, zeros);
            let pix = _mm_unpacklo_epi8(interleaved, zeros);
            store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(pix, v_weight));
            let pix = _mm_unpackhi_epi8(interleaved, zeros);
            store_1 = _mm_add_epi32(store_1, _mm_madd_epi16(pix, v_weight));

            let interleaved = _mm_unpackhi_epi8(item_row_0, zeros);
            let pix = _mm_unpacklo_epi8(interleaved, zeros);
            store_2 = _mm_add_epi32(store_2, _mm_madd_epi16(pix, v_weight));
            let pix = _mm_unpackhi_epi8(interleaved, zeros);
            store_3 = _mm_add_epi32(store_3, _mm_madd_epi16(pix, v_weight));

            let interleaved = _mm_unpacklo_epi8(item_row_1, zeros);
            let pix = _mm_unpacklo_epi8(interleaved, zeros);
            store_4 = _mm_add_epi32(store_4, _mm_madd_epi16(pix, v_weight));
            let pix = _mm_unpackhi_epi8(interleaved, zeros);
            store_5 = _mm_add_epi32(store_5, _mm_madd_epi16(pix, v_weight));

            let interleaved = _mm_unpackhi_epi8(item_row_1, zeros);
            let pix = _mm_unpacklo_epi8(interleaved, zeros);
            store_6 = _mm_add_epi32(store_6, _mm_madd_epi16(pix, v_weight));
            let pix = _mm_unpackhi_epi8(interleaved, zeros);
            store_7 = _mm_add_epi32(store_7, _mm_madd_epi16(pix, v_weight));
        }

        store_0 = _mm_srai_epi32::<12>(store_0);
        store_1 = _mm_srai_epi32::<12>(store_1);
        store_2 = _mm_srai_epi32::<12>(store_2);
        store_3 = _mm_srai_epi32::<12>(store_3);
        store_4 = _mm_srai_epi32::<12>(store_4);
        store_5 = _mm_srai_epi32::<12>(store_5);
        store_6 = _mm_srai_epi32::<12>(store_6);
        store_7 = _mm_srai_epi32::<12>(store_7);

        let rgb0 = _mm_packs_epi32(store_0, store_1);
        let rgb2 = _mm_packs_epi32(store_2, store_3);
        let rgb = _mm_packus_epi16(rgb0, rgb2);

        let dst_ptr = dst.add(px);
        _mm_storeu_si128(dst_ptr as *mut __m128i, rgb);

        let rgb0 = _mm_packs_epi32(store_4, store_5);
        let rgb2 = _mm_packs_epi32(store_6, store_7);
        let rgb = _mm_packus_epi16(rgb0, rgb2);

        let dst_ptr = dst.add(px + 16);
        _mm_storeu_si128(dst_ptr as *mut __m128i, rgb);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[inline(always)]
    pub(crate) unsafe fn convolve_horizontal_parts_one_rgba_sse(
        start_x: usize,
        src: *const u8,
        weight0: __m128i,
        store_0: __m128i,
    ) -> __m128i {
        const COMPONENTS: usize = 4;
        let src_ptr = src.add(start_x * COMPONENTS);

        let src_ptr_32 = src_ptr as *const i32;
        let rgba_pixel = _mm_cvtsi32_si128(*src_ptr_32);
        let lo = _mm_cvtepu8_epi16(rgba_pixel);

        let acc = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(lo), weight0));
        acc
    }

    #[inline(always)]
    pub(crate) unsafe fn convolve_vertical_part_sse_8<const USE_BLENDING: bool>(
        start_y: usize,
        start_x: usize,
        src: *const u8,
        src_stride: usize,
        dst: *mut u8,
        filter: *const i16,
        bounds: &FilterBounds,
        blend_length: usize,
    ) {
        let vld = _mm_set1_epi32(ROUNDING_APPROX);
        let mut store_0 = vld;
        let mut store_1 = vld;

        let zeros = _mm_setzero_si128();

        let px = start_x;

        for j in 0..bounds.size {
            let py = start_y + j;
            let weight = *unsafe { filter.add(j) };
            let v_weight = _mm_set1_epi32(weight as i32);
            let src_ptr = src.add(src_stride * py);

            let s_ptr = src_ptr.add(px);
            let item_row = if USE_BLENDING {
                let mut transient: [u8; 8] = [0; 8];
                std::ptr::copy_nonoverlapping(s_ptr, transient.as_mut_ptr(), blend_length);
                _mm_loadu_si64(transient.as_ptr())
            } else {
                _mm_loadu_si64(s_ptr)
            };

            let low = _mm_cvtepu8_epi16(item_row);
            store_0 = _mm_add_epi32(store_0, _mm_madd_epi16(_mm_cvtepi16_epi32(low), v_weight));
            store_1 = _mm_add_epi32(
                store_1,
                _mm_madd_epi16(_mm_unpackhi_epi16(low, zeros), v_weight),
            );
        }

        store_0 = _mm_max_epi32(store_0, zeros);
        store_1 = _mm_max_epi32(store_1, zeros);

        let low_16 = _mm_packus_epi32(_mm_srai_epi32::<12>(store_0), _mm_srai_epi32::<12>(store_1));

        let item = _mm_packus_epi16(low_16, low_16);

        let dst_ptr = dst.add(px);
        if USE_BLENDING {
            let mut transient: [u8; 8] = [0; 8];
            std::ptr::copy_nonoverlapping(
                &item as *const _ as *const u8,
                transient.as_mut_ptr(),
                8,
            );
            std::ptr::copy_nonoverlapping(transient.as_ptr(), dst_ptr, blend_length);
        } else {
            std::ptr::copy_nonoverlapping(&item as *const _ as *const u8, dst_ptr, 8);
        }
    }
}
