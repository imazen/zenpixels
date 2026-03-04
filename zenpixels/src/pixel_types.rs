//! Type aliases for gray+alpha pixel types from the `rgb` crate.
//!
//! The underlying `rgb::GrayA<T>` is `#[repr(C)]` with `Pod`/`Zeroable`
//! (via the `bytemuck` feature). Fields: `.v` (gray value), `.a` (alpha).

/// Grayscale + alpha, 8-bit per channel.
pub type GrayAlpha8 = rgb::GrayA<u8>;

/// Grayscale + alpha, 16-bit per channel.
pub type GrayAlpha16 = rgb::GrayA<u16>;

/// Grayscale + alpha, f32 per channel.
pub type GrayAlphaF32 = rgb::GrayA<f32>;
