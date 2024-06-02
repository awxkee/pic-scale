pub fn resize_nearest<T, const CHANNELS: usize>(
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
