#[cfg(target_feature = "f16c")]
mod alpha_f16;
mod alpha_f32;
mod alpha_u8;
pub mod utils;

#[cfg(target_feature = "f16c")]
pub use alpha_f16::{avx_premultiply_alpha_rgba_f16, avx_unpremultiply_alpha_rgba_f16};
pub use alpha_f32::avx_premultiply_alpha_rgba_f32;
pub use alpha_f32::avx_unpremultiply_alpha_rgba_f32;
pub use alpha_u8::avx_premultiply_alpha_rgba;
pub use alpha_u8::avx_unpremultiply_alpha_rgba;
