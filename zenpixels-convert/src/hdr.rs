//! HDR processing utilities.
//!
//! Provides HDR metadata types (content light level, mastering display),
//! and basic tone mapping helpers. The core PQ/HLG EOTF/OETF math is
//! always available through the main conversion pipeline in
//! [`ConvertPlan`](crate::ConvertPlan).

use crate::TransferFunction;

/// HDR content light level metadata (CEA-861.3 / CTA-861-H).
///
/// Describes the peak brightness characteristics of HDR content.
/// Used by AVIF, JXL, PNG (cLLi chunk), and video containers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ContentLightLevel {
    /// Maximum Content Light Level (MaxCLL) in cd/m² (nits).
    /// Peak luminance of any single pixel in the content.
    pub max_content_light_level: u16,
    /// Maximum Frame-Average Light Level (MaxFALL) in cd/m².
    /// Peak average luminance of any single frame.
    pub max_frame_average_light_level: u16,
}

impl ContentLightLevel {
    /// Create content light level metadata.
    pub const fn new(max_content_light_level: u16, max_frame_average_light_level: u16) -> Self {
        Self {
            max_content_light_level,
            max_frame_average_light_level,
        }
    }
}

/// Mastering display color volume metadata (SMPTE ST 2086).
///
/// Describes the display on which the content was mastered, enabling
/// downstream displays to reproduce the creator's intent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MasteringDisplay {
    /// RGB primaries of the mastering display in CIE 1931 xy coordinates.
    /// `[[rx, ry], [gx, gy], [bx, by]]`.
    pub primaries_xy: [[f32; 2]; 3],
    /// White point in CIE 1931 xy coordinates `[wx, wy]`.
    pub white_point_xy: [f32; 2],
    /// Maximum display luminance in cd/m².
    pub max_luminance: f32,
    /// Minimum display luminance in cd/m².
    pub min_luminance: f32,
}

impl MasteringDisplay {
    /// Create mastering display metadata from CIE 1931 xy coordinates and cd/m² luminances.
    pub const fn new(
        primaries_xy: [[f32; 2]; 3],
        white_point_xy: [f32; 2],
        max_luminance: f32,
        min_luminance: f32,
    ) -> Self {
        Self {
            primaries_xy,
            white_point_xy,
            max_luminance,
            min_luminance,
        }
    }

    /// BT.2020 primaries with D65 white point, 10000 nits peak (HDR10 reference).
    pub const HDR10_REFERENCE: Self = Self {
        primaries_xy: [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        white_point_xy: [0.3127, 0.3290],
        max_luminance: 10000.0,
        min_luminance: 0.0001,
    };

    /// Display P3 primaries with D65 white point, 1000 nits.
    pub const DISPLAY_P3_1000: Self = Self {
        primaries_xy: [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        white_point_xy: [0.3127, 0.3290],
        max_luminance: 1000.0,
        min_luminance: 0.0001,
    };
}

/// Describes the HDR characteristics of pixel data.
///
/// Bundles transfer function, content light level, and mastering display
/// metadata to provide everything needed for HDR processing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrMetadata {
    /// Transfer function (PQ, HLG, sRGB, Linear, etc.).
    pub transfer: TransferFunction,
    /// Content light level (MaxCLL/MaxFALL). Optional.
    pub content_light_level: Option<ContentLightLevel>,
    /// Mastering display color volume. Optional.
    pub mastering_display: Option<MasteringDisplay>,
}

impl HdrMetadata {
    /// True if this describes HDR content (PQ or HLG transfer function).
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer, TransferFunction::Pq | TransferFunction::Hlg)
    }

    /// True if this describes SDR content.
    #[must_use]
    pub fn is_sdr(&self) -> bool {
        !self.is_hdr()
    }

    /// Create HDR10 metadata with PQ transfer.
    pub fn hdr10(cll: ContentLightLevel) -> Self {
        Self {
            transfer: TransferFunction::Pq,
            content_light_level: Some(cll),
            mastering_display: Some(MasteringDisplay::HDR10_REFERENCE),
        }
    }

    /// Create HLG metadata.
    pub fn hlg() -> Self {
        Self {
            transfer: TransferFunction::Hlg,
            content_light_level: None,
            mastering_display: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Naive HDR ↔ SDR tone mapping (built-in, no deps)
// ---------------------------------------------------------------------------

/// Simple Reinhard-style tone mapping: HDR linear → SDR linear.
///
/// Maps linear light [0, ∞) → [0, 1) using `v / (1 + v)`.
/// Preserves relative brightness ordering. Does not use any display
/// metadata — for proper tone mapping, use a dedicated HDR tone mapping
/// library.
#[inline]
#[must_use]
pub fn reinhard_tonemap(v: f32) -> f32 {
    v / (1.0 + v)
}

/// Inverse Reinhard: SDR linear → HDR linear.
///
/// Maps [0, 1) → [0, ∞) using `v / (1 - v)`.
#[inline]
#[must_use]
pub fn reinhard_inverse(v: f32) -> f32 {
    if v >= 1.0 {
        return f32::MAX;
    }
    v / (1.0 - v)
}

/// Simple exposure-based tone mapping.
///
/// `exposure` is in stops relative to 1.0. Positive values brighten,
/// negative darken. The result is clamped to [0, 1].
///
/// Requires `std` because `f32::powf` is not available in `no_std`.
#[cfg(feature = "std")]
#[inline]
#[must_use]
pub fn exposure_tonemap(v: f32, exposure: f32) -> f32 {
    (v * 2.0f32.powf(exposure)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reinhard_boundaries() {
        assert_eq!(reinhard_tonemap(0.0), 0.0);
        assert!((reinhard_tonemap(1.0) - 0.5).abs() < 1e-6);
        assert!(reinhard_tonemap(1000.0) > 0.99);
        assert!(reinhard_tonemap(1000.0) < 1.0);
    }

    #[test]
    fn reinhard_roundtrip() {
        for &v in &[0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0] {
            let mapped = reinhard_tonemap(v);
            let unmapped = reinhard_inverse(mapped);
            assert!(
                (unmapped - v).abs() < 1e-4,
                "Reinhard roundtrip failed for {v}: got {unmapped}"
            );
        }
    }

    #[test]
    fn hdr_metadata_is_hdr() {
        assert!(HdrMetadata::hdr10(ContentLightLevel::default()).is_hdr());
        assert!(HdrMetadata::hlg().is_hdr());
        assert!(
            HdrMetadata {
                transfer: TransferFunction::Srgb,
                content_light_level: None,
                mastering_display: None,
            }
            .is_sdr()
        );
    }

    #[test]
    fn content_light_level_new() {
        let cll = ContentLightLevel::new(1000, 500);
        assert_eq!(cll.max_content_light_level, 1000);
        assert_eq!(cll.max_frame_average_light_level, 500);
    }

    #[test]
    fn content_light_level_default() {
        let cll = ContentLightLevel::default();
        assert_eq!(cll.max_content_light_level, 0);
        assert_eq!(cll.max_frame_average_light_level, 0);
    }

    #[test]
    fn mastering_display_new() {
        let md = MasteringDisplay::new(
            [[0.68, 0.32], [0.265, 0.69], [0.15, 0.06]],
            [0.3127, 0.329],
            1000.0,
            0.001,
        );
        assert_eq!(md.max_luminance, 1000.0);
        assert_eq!(md.min_luminance, 0.001);
    }

    #[test]
    fn mastering_display_constants() {
        assert_eq!(MasteringDisplay::HDR10_REFERENCE.max_luminance, 10000.0);
        assert_eq!(MasteringDisplay::DISPLAY_P3_1000.max_luminance, 1000.0);
    }

    #[test]
    fn hdr10_constructor() {
        let cll = ContentLightLevel::new(4000, 1000);
        let meta = HdrMetadata::hdr10(cll);
        assert!(meta.is_hdr());
        assert_eq!(meta.transfer, TransferFunction::Pq);
        assert_eq!(meta.content_light_level, Some(cll));
        assert!(meta.mastering_display.is_some());
    }

    #[test]
    fn hlg_constructor() {
        let meta = HdrMetadata::hlg();
        assert!(meta.is_hdr());
        assert_eq!(meta.transfer, TransferFunction::Hlg);
        assert!(meta.content_light_level.is_none());
        assert!(meta.mastering_display.is_none());
    }

    #[test]
    fn exposure_tonemap_values() {
        // 0 stops = unchanged (clamped to [0,1]).
        assert!((exposure_tonemap(0.5, 0.0) - 0.5).abs() < 1e-6);
        // +1 stop = doubled.
        assert!((exposure_tonemap(0.25, 1.0) - 0.5).abs() < 1e-5);
        // -1 stop = halved.
        assert!((exposure_tonemap(0.5, -1.0) - 0.25).abs() < 1e-5);
        // Clamped to [0,1].
        assert_eq!(exposure_tonemap(0.8, 1.0), 1.0);
        assert_eq!(exposure_tonemap(0.0, 5.0), 0.0);
    }

    #[test]
    fn reinhard_inverse_at_one() {
        assert_eq!(reinhard_inverse(1.0), f32::MAX);
    }

    #[test]
    fn content_light_level_clone_eq_hash() {
        use core::hash::{Hash, Hasher};
        let a = ContentLightLevel::new(100, 50);
        let b = a;
        assert_eq!(a, b);
        let mut h1 = std::hash::DefaultHasher::new();
        a.hash(&mut h1);
        let mut h2 = std::hash::DefaultHasher::new();
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn hdr_metadata_clone_partial_eq() {
        let a = HdrMetadata::hlg();
        let b = a;
        assert_eq!(a, b);
    }
}
