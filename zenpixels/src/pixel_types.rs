//! Custom pixel types for gray+alpha formats.
//!
//! These types are `#[repr(C)]` and derive `Pod`/`Zeroable` for zero-cost
//! byte reinterpretation via `bytemuck`.

/// Grayscale + alpha, 8-bit.
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct GrayAlpha8 {
    /// Gray value.
    pub v: u8,
    /// Alpha value.
    pub a: u8,
}

/// Grayscale + alpha, 16-bit.
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct GrayAlpha16 {
    /// Gray value.
    pub v: u16,
    /// Alpha value.
    pub a: u16,
}

/// Grayscale + alpha, f32.
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct GrayAlphaF32 {
    /// Gray value.
    pub v: f32,
    /// Alpha value.
    pub a: f32,
}

impl GrayAlpha8 {
    /// Create a new gray+alpha pixel.
    pub const fn new(v: u8, a: u8) -> Self {
        Self { v, a }
    }
}

impl GrayAlpha16 {
    /// Create a new gray+alpha pixel.
    pub const fn new(v: u16, a: u16) -> Self {
        Self { v, a }
    }
}

impl GrayAlphaF32 {
    /// Create a new gray+alpha pixel.
    pub const fn new(v: f32, a: f32) -> Self {
        Self { v, a }
    }
}
