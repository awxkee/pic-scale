#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

constexpr static const uint32_t PIC_SCALE_PREMULTIPLY_ALPHA = 1;

constexpr static const uint32_t PIC_SCALE_USE_MULTITHREADING = 2;

enum class ScalingFilter {
  Nearest,
  Bilinear,
  Lanczos3,
  MitchellNetravalli,
  Bicubic,
  CatmullRom,
};

extern "C" {

/// Resizes an RGBA8 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
///
uintptr_t pic_scale_resize_rgba8(const uint8_t *src,
                                 uintptr_t src_stride,
                                 uint32_t width,
                                 uint32_t height,
                                 uint8_t *dst,
                                 uintptr_t dst_stride,
                                 uint32_t new_width,
                                 uint32_t new_height,
                                 ScalingFilter resizing_filter,
                                 uint32_t flags);

/// Resizes an RGB8 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
///
uintptr_t pic_scale_resize_rgb8(const uint8_t *src,
                                uintptr_t src_stride,
                                uint32_t width,
                                uint32_t height,
                                uint8_t *dst,
                                uintptr_t dst_stride,
                                uint32_t new_width,
                                uint32_t new_height,
                                ScalingFilter resizing_filter,
                                uint32_t flags);

/// Resizes an CbCr8 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
///
uintptr_t pic_scale_resize_cbcr8(const uint8_t *src,
                                 uintptr_t src_stride,
                                 uint32_t width,
                                 uint32_t height,
                                 uint8_t *dst,
                                 uintptr_t dst_stride,
                                 uint32_t new_width,
                                 uint32_t new_height,
                                 ScalingFilter resizing_filter,
                                 uint32_t flags);

/// Resizes an Planar8 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
///
uintptr_t pic_scale_resize_planar8(const uint8_t *src,
                                   uintptr_t src_stride,
                                   uint32_t width,
                                   uint32_t height,
                                   uint8_t *dst,
                                   uintptr_t dst_stride,
                                   uint32_t new_width,
                                   uint32_t new_height,
                                   ScalingFilter resizing_filter,
                                   uint32_t flags);

/// Resizes an RGBA16 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `bit_depth`: Image bit-depth
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
///
uintptr_t pic_scale_resize_rgba16(const uint16_t *src,
                                  uintptr_t src_stride,
                                  uint32_t width,
                                  uint32_t height,
                                  uint16_t *dst,
                                  uintptr_t dst_stride,
                                  uint32_t new_width,
                                  uint32_t new_height,
                                  uint32_t bit_depth,
                                  ScalingFilter resizing_filter,
                                  uint32_t flags);

/// Resizes an RGB16 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `bit_depth`: Image bit-depth
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
///
uintptr_t pic_scale_resize_rgb16(const uint16_t *src,
                                 uintptr_t src_stride,
                                 uint32_t width,
                                 uint32_t height,
                                 uint16_t *dst,
                                 uintptr_t dst_stride,
                                 uint32_t new_width,
                                 uint32_t new_height,
                                 uint32_t bit_depth,
                                 ScalingFilter resizing_filter,
                                 uint32_t flags);

/// Resizes an CbCr16 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `bit_depth`: Image bit-depth
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
///
uintptr_t pic_scale_resize_cbcr16(const uint16_t *src,
                                  uintptr_t src_stride,
                                  uint32_t width,
                                  uint32_t height,
                                  uint16_t *dst,
                                  uintptr_t dst_stride,
                                  uint32_t new_width,
                                  uint32_t new_height,
                                  uint32_t bit_depth,
                                  ScalingFilter resizing_filter,
                                  uint32_t flags);

/// Resizes an Planar16 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `bit_depth`: Image bit-depth
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
///
uintptr_t pic_scale_resize_planar16(const uint16_t *src,
                                    uintptr_t src_stride,
                                    uint32_t width,
                                    uint32_t height,
                                    uint16_t *dst,
                                    uintptr_t dst_stride,
                                    uint32_t new_width,
                                    uint32_t new_height,
                                    uint32_t bit_depth,
                                    ScalingFilter resizing_filter,
                                    uint32_t flags);

/// Resizes an RGBAF32 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
uintptr_t pic_scale_resize_rgba_f32(const float *src,
                                    uintptr_t src_stride,
                                    uint32_t width,
                                    uint32_t height,
                                    float *dst,
                                    uintptr_t dst_stride,
                                    uint32_t new_width,
                                    uint32_t new_height,
                                    ScalingFilter resizing_filter,
                                    uint32_t flags);

/// Resizes an RGBF32 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
uintptr_t pic_scale_resize_rgb_f32(const float *src,
                                   uintptr_t src_stride,
                                   uint32_t width,
                                   uint32_t height,
                                   float *dst,
                                   uintptr_t dst_stride,
                                   uint32_t new_width,
                                   uint32_t new_height,
                                   ScalingFilter resizing_filter,
                                   uint32_t flags);

/// Resizes an CbCrF32 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
uintptr_t pic_scale_resize_cbcr_f32(const float *src,
                                    uintptr_t src_stride,
                                    uint32_t width,
                                    uint32_t height,
                                    float *dst,
                                    uintptr_t dst_stride,
                                    uint32_t new_width,
                                    uint32_t new_height,
                                    ScalingFilter resizing_filter,
                                    uint32_t flags);

/// Resizes an PlanarF32 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
uintptr_t pic_scale_resize_planar_f32(const float *src,
                                      uintptr_t src_stride,
                                      uint32_t width,
                                      uint32_t height,
                                      float *dst,
                                      uintptr_t dst_stride,
                                      uint32_t new_width,
                                      uint32_t new_height,
                                      ScalingFilter resizing_filter,
                                      uint32_t flags);

/// Resizes an RGBAF16 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
uintptr_t pic_scale_resize_rgba_f16(const uint16_t *src,
                                    uintptr_t src_stride,
                                    uint32_t width,
                                    uint32_t height,
                                    uint16_t *dst,
                                    uintptr_t dst_stride,
                                    uint32_t new_width,
                                    uint32_t new_height,
                                    ScalingFilter resizing_filter,
                                    uint32_t flags);

/// Resizes an RGBAF16 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
uintptr_t pic_scale_resize_rgb_f16(const uint16_t *src,
                                   uintptr_t src_stride,
                                   uint32_t width,
                                   uint32_t height,
                                   uint16_t *dst,
                                   uintptr_t dst_stride,
                                   uint32_t new_width,
                                   uint32_t new_height,
                                   ScalingFilter resizing_filter,
                                   uint32_t flags);

/// Resizes an CbCrF16 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
uintptr_t pic_scale_resize_cbcr_f16(const uint16_t *src,
                                    uintptr_t src_stride,
                                    uint32_t width,
                                    uint32_t height,
                                    uint16_t *dst,
                                    uintptr_t dst_stride,
                                    uint32_t new_width,
                                    uint32_t new_height,
                                    ScalingFilter resizing_filter,
                                    uint32_t flags);

/// Resizes an PlanarF16 image
///
/// # Arguments
///
/// * `src`: Source image pointer
/// * `src_stride`: Source image stride
/// * `width`: Source image width
/// * `height`: Source image height
/// * `dst`: Destination pointer
/// * `dst_stride`: Destination stride
/// * `new_width`: New image width
/// * `new_height`: New image height
/// * `resizing_filter`: One of [ScalingFilter]
/// * `flags`: Flags of: [PIC_SCALE_PREMULTIPLY_ALPHA], [PIC_SCALE_USE_MULTITHREADING]
///
/// returns: 0 if success, for error codes refers to [PicScaleError::code]
uintptr_t pic_scale_resize_planar_f16(const uint16_t *src,
                                      uintptr_t src_stride,
                                      uint32_t width,
                                      uint32_t height,
                                      uint16_t *dst,
                                      uintptr_t dst_stride,
                                      uint32_t new_width,
                                      uint32_t new_height,
                                      ScalingFilter resizing_filter,
                                      uint32_t flags);

} // extern "C"
