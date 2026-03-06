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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn gray_alpha8_new_and_fields() {
        let px = GrayAlpha8::new(128, 200);
        assert_eq!(px.v, 128);
        assert_eq!(px.a, 200);
    }

    #[test]
    fn gray_alpha16_new_and_fields() {
        let px = GrayAlpha16::new(1000, 65535);
        assert_eq!(px.v, 1000);
        assert_eq!(px.a, 65535);
    }

    #[test]
    fn gray_alpha_f32_new_and_fields() {
        let px = GrayAlphaF32::new(0.5, 1.0);
        assert_eq!(px.v, 0.5);
        assert_eq!(px.a, 1.0);
    }

    #[test]
    fn gray_alpha8_default() {
        let px = GrayAlpha8::default();
        assert_eq!(px.v, 0);
        assert_eq!(px.a, 0);
    }

    #[test]
    fn gray_alpha16_default() {
        let px = GrayAlpha16::default();
        assert_eq!(px.v, 0);
        assert_eq!(px.a, 0);
    }

    #[test]
    fn gray_alpha_f32_default() {
        let px = GrayAlphaF32::default();
        assert_eq!(px.v, 0.0);
        assert_eq!(px.a, 0.0);
    }

    #[test]
    fn gray_alpha8_eq() {
        let a = GrayAlpha8::new(100, 200);
        let b = GrayAlpha8::new(100, 200);
        let c = GrayAlpha8::new(100, 201);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    #[cfg(feature = "std")]
    fn gray_alpha8_hash() {
        use core::hash::{Hash, Hasher};
        let a = GrayAlpha8::new(100, 200);
        let b = GrayAlpha8::new(100, 200);
        let mut h1 = std::hash::DefaultHasher::new();
        a.hash(&mut h1);
        let mut h2 = std::hash::DefaultHasher::new();
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn gray_alpha_f32_partial_eq() {
        let a = GrayAlphaF32::new(0.5, 1.0);
        let b = GrayAlphaF32::new(0.5, 1.0);
        let c = GrayAlphaF32::new(0.5, 0.9);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn gray_alpha8_debug() {
        let s = format!("{:?}", GrayAlpha8::new(10, 20));
        assert!(s.contains("10"));
        assert!(s.contains("20"));
    }

    #[test]
    fn gray_alpha8_clone_copy() {
        let a = GrayAlpha8::new(50, 100);
        let b = a;
        #[allow(clippy::clone_on_copy)]
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn gray_alpha8_bytemuck_roundtrip() {
        let pixels = [GrayAlpha8::new(10, 20), GrayAlpha8::new(30, 40)];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes, &[10, 20, 30, 40]);
        let back: &[GrayAlpha8] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &pixels);
    }

    #[test]
    fn gray_alpha16_bytemuck_roundtrip() {
        let pixels = [GrayAlpha16::new(1000, 2000)];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes.len(), 4);
        let back: &[GrayAlpha16] = bytemuck::cast_slice(bytes);
        assert_eq!(back[0], pixels[0]);
    }

    #[test]
    fn gray_alpha_f32_bytemuck_roundtrip() {
        let pixels = [GrayAlphaF32::new(0.5, 1.0)];
        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes.len(), 8);
        let back: &[GrayAlphaF32] = bytemuck::cast_slice(bytes);
        assert_eq!(back[0], pixels[0]);
    }
}
