#[cfg(all(feature = "half", target_feature = "f16c"))]
mod alpha_f16;
mod alpha_f32;
mod alpha_u8;
pub mod utils;
mod vertical_f16;
mod vertical_f32;
mod vertical_u8;

#[cfg(all(feature = "half", target_feature = "f16c"))]
pub use alpha_f16::{avx_premultiply_alpha_rgba_f16, avx_unpremultiply_alpha_rgba_f16};
pub use alpha_f32::avx_premultiply_alpha_rgba_f32;
pub use alpha_f32::avx_unpremultiply_alpha_rgba_f32;
pub use alpha_u8::avx_premultiply_alpha_rgba;
pub use alpha_u8::avx_unpremultiply_alpha_rgba;
#[cfg(all(feature = "half", target_feature = "f16c"))]
pub use vertical_f16::convolve_vertical_avx_row_f16;
pub use vertical_f32::convolve_vertical_avx_row_f32;
pub use vertical_u8::convolve_vertical_avx_row;
