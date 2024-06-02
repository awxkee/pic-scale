use crate::filter_weights::FilterBounds;

#[inline(always)]
#[allow(unused)]
pub(crate) unsafe fn convolve_vertical_part<const PART: usize, const CHANNELS: usize>(
    start_y: usize,
    start_x: usize,
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    filter: *const i16,
    bounds: &FilterBounds,
) {
    let mut store: [[i32; CHANNELS]; PART] = [[0; CHANNELS]; PART];

    for j in 0..bounds.size {
        let py = start_y + j;
        let weight = *unsafe { filter.add(j) } as i32;
        let src_ptr = src.add(src_stride * py);
        for x in 0..PART {
            let px = (start_x + x) * CHANNELS;
            let s_ptr = src_ptr.add(px);
            for c in 0..CHANNELS {
                let store_p = store.get_unchecked_mut(x);
                let store_v = store_p.get_unchecked_mut(c);
                *store_v += unsafe { *s_ptr.add(c) } as i32 * weight;
            }
        }
    }

    for x in 0..PART {
        let px = (start_x + x) * CHANNELS;
        let dst_ptr = dst.add(px);
        for c in 0..CHANNELS {
            let vl = *(*store.get_unchecked_mut(x)).get_unchecked_mut(c);
            let ck = vl >> 12;
            *dst_ptr.add(c) = ck.max(0).min(255) as u8;
        }
    }
}
