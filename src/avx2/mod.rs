mod alpha_f32;
mod alpha_u8;
pub mod utils;

pub use alpha_f32::avx_premultiply_alpha_rgba_f32;
pub use alpha_f32::avx_unpremultiply_alpha_rgba_f32;
pub use alpha_u8::avx_premultiply_alpha_rgba;
pub use alpha_u8::avx_unpremultiply_alpha_rgba;
