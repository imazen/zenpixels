//! HDR metadata types.
//!
//! Pure data types for HDR content description. These travel with pixel
//! data alongside [`Cicp`](crate::Cicp) and [`ColorContext`](crate::ColorContext).
//!
//! For tone mapping and HDR processing functions, see
//! [`zenpixels-convert::hdr`](https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/).
//!
//! For SOTA content-light-level measurement (MaxCLL / MaxFALL — histogram-
//! based, percentile-aware, SIMD-accelerated), see the `measure` module
//! and the `CllMeasure` extension trait in `zenpixels-convert`.

/// The absolute luminance, in cd/m² (nits), that a relative-linear sample
/// value of `1.0` represents — the "diffuse white" (a.k.a. nominal diffuse
/// white / SDR reference white) anchor that bridges relative-linear pixel
/// data to absolute display light.
///
/// This is the single scalar the rest of the industry uses for that bridge:
/// OpenEXR's `whiteLuminance` ("nits of RGB (1,1,1)"), JPEG XL's
/// `intensity_target`, libheif's `ndwt` (nominal diffuse white), and
/// libplacebo's SDR-white constant. The cross-vendor default is
/// [`BT2408`](Self::BT2408) = 203 cd/m².
///
/// It is a *typed* anchor on purpose: HDR code mixes nits, PQ-encoded `[0,1]`,
/// log2 gain, and headroom ratios — passing a bare `f32` invites unit
/// confusion. Use [`DiffuseWhite::new`] / [`DiffuseWhite::nits`].
#[derive(Clone, Copy, Debug)]
pub struct DiffuseWhite(f32);

// Bit-exact equality so `DiffuseWhite` — and therefore `ColorContext` — keeps
// `Eq` despite wrapping `f32`. A luminance anchor is always a sane, finite,
// positive cd/m² value (203, 100, 10000, …), so a bitwise compare is reflexive
// and consistent; the -0.0 / NaN cases a value compare would treat differently
// never occur for an anchor.
impl PartialEq for DiffuseWhite {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for DiffuseWhite {}
impl core::hash::Hash for DiffuseWhite {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl DiffuseWhite {
    /// ITU-R BT.2408 HDR reference white: **203 cd/m²**. The cross-industry
    /// default anchor for relative-linear HDR (matches Chrome `SDRWhiteLevel`,
    /// Skia skcms, CSS `rec2100-linear`, and libplacebo).
    pub const BT2408: Self = Self(203.0);

    /// [`new`](Self::new), or `None` unless `nits` is finite and strictly positive.
    #[must_use]
    pub const fn try_new(nits: f32) -> Option<Self> {
        if nits.is_finite() && nits > 0.0 {
            Some(Self(nits))
        } else {
            None
        }
    }

    /// An anchor of `nits` cd/m² (the luminance that relative-linear `1.0`
    /// represents).
    ///
    /// # Panics
    ///
    /// Panics if `nits` is zero, negative, NaN, or infinite.
    #[track_caller]
    #[must_use]
    pub const fn new(nits: f32) -> Self {
        assert!(
            nits.is_finite() && nits > 0.0,
            "diffuse white must be finite and strictly positive"
        );
        Self(nits)
    }

    /// The anchor in cd/m² (nits).
    #[must_use]
    pub const fn nits(self) -> f32 {
        self.0
    }
}

impl Default for DiffuseWhite {
    /// [`BT2408`](Self::BT2408) — 203 cd/m².
    fn default() -> Self {
        Self::BT2408
    }
}

/// A finite, strictly positive luminance in cd/m².
/// Used for source/display peaks independently of the relative-linear white anchor.
#[derive(Clone, Copy, Debug)]
pub struct Nits(f32);
impl Nits {
    /// Return a luminance when it is finite and strictly positive.
    pub const fn try_new(value: f32) -> Option<Self> {
        if value.is_finite() && value > 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }
    /// Construct a luminance.
    /// # Panics
    /// Panics for zero, negative, NaN or infinite values.
    #[track_caller]
    pub const fn new(value: f32) -> Self {
        assert!(
            value.is_finite() && value > 0.0,
            "luminance must be finite and strictly positive"
        );
        Self(value)
    }
    /// Luminance in cd/m².
    pub const fn nits(self) -> f32 {
        self.0
    }
}
impl PartialEq for Nits {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for Nits {}
impl core::hash::Hash for Nits {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// How to obtain the source peak for HDR tone mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HdrPeak {
    /// Measure the current buffer's peak. Row plans need an explicit measurement.
    #[default]
    Measured,
    /// Use the source peak declared in its color context; refuse if absent.
    Declared,
    /// Use this explicit peak instead of measuring or consulting metadata.
    Nits(Nits),
}

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
#[non_exhaustive]
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diffuse_white_defaults_to_bt2408() {
        assert_eq!(DiffuseWhite::default(), DiffuseWhite::BT2408);
        assert_eq!(DiffuseWhite::BT2408.nits(), 203.0);
        assert_eq!(DiffuseWhite::new(100.0).nits(), 100.0);
    }

    #[test]
    fn diffuse_white_custom_anchor_round_trips() {
        // Anchor metadata is byte-identical-preserved through the constructor:
        // a custom 100 cd/m² (HDR home-tier mastering) and 10 000 cd/m² (PQ
        // peak) both round-trip through `new` → `nits` losslessly.
        assert_eq!(DiffuseWhite::new(100.0).nits(), 100.0);
        assert_eq!(DiffuseWhite::new(10_000.0).nits(), 10_000.0);
        // PartialEq honours bit equality (see the impl above) so two
        // independently constructed anchors compare equal.
        assert_eq!(DiffuseWhite::new(203.0), DiffuseWhite::BT2408);
    }

    #[test]
    fn diffuse_white_accepts_positive_finite_bounds_and_const_construction() {
        const CUSTOM: DiffuseWhite = DiffuseWhite::new(100.0);
        assert_eq!(CUSTOM.nits(), 100.0);
        for nits in [f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX] {
            assert_eq!(DiffuseWhite::new(nits).nits().to_bits(), nits.to_bits());
        }
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_zero() {
        let _ = DiffuseWhite::new(0.0);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_negative_zero() {
        let _ = DiffuseWhite::new(-0.0);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_negative() {
        let _ = DiffuseWhite::new(-100.0);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_nan() {
        let _ = DiffuseWhite::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_infinity() {
        let _ = DiffuseWhite::new(f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_negative_infinity() {
        let _ = DiffuseWhite::new(f32::NEG_INFINITY);
    }

    #[test]
    fn content_light_level_clone_eq() {
        let a = ContentLightLevel::new(100, 50);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    #[cfg(feature = "std")]
    fn content_light_level_hash() {
        use core::hash::{Hash, Hasher};
        let a = ContentLightLevel::new(100, 50);
        let b = a;
        let mut h1 = std::hash::DefaultHasher::new();
        a.hash(&mut h1);
        let mut h2 = std::hash::DefaultHasher::new();
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn mastering_display_constants() {
        assert_eq!(MasteringDisplay::HDR10_REFERENCE.max_luminance, 10000.0);
        assert_eq!(MasteringDisplay::DISPLAY_P3_1000.max_luminance, 1000.0);
    }
}

/// Exact mastering-display codes, separate from the lossy floating-point view.
/// Chromaticities are in units of 0.00002; luminances in units of 0.0001 cd/m².
/// Byte serialization uses RGB primary order, then white, maximum and minimum
/// luminance, all big-endian. Adapters for formats with another primary order
/// must reorder explicitly. No validation discards unknown or out-of-range codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct St2086Codes([u8; 24]);
impl St2086Codes {
    /// Preserve all 24 bytes in the documented RGB-order representation.
    pub const fn from_be_bytes(bytes: [u8; 24]) -> Self {
        Self(bytes)
    }
    /// Return the exact original code bytes, without a float round trip.
    pub const fn to_be_bytes(self) -> [u8; 24] {
        self.0
    }
    /// Interpret codes for display decisions. This view can lose precision;
    /// retain this carrier when exact metadata must survive a transcode.
    pub fn semantic(self) -> MasteringDisplay {
        let xy = |i| f32::from(u16::from_be_bytes([self.0[i], self.0[i + 1]])) / 50_000.0;
        let lum = |i| u32::from_be_bytes(self.0[i..i + 4].try_into().unwrap()) as f32 / 10_000.0;
        MasteringDisplay::new(
            [[xy(0), xy(2)], [xy(4), xy(6)], [xy(8), xy(10)]],
            [xy(12), xy(14)],
            lum(16),
            lum(20),
        )
    }
}

/// Explicit HLG display rendering conditions. No implicit default exists.
///
/// `system_gamma` includes the caller's surround adjustment. The reference
/// preset assumes a 5 cd/m² surround and an ideal zero-black 1000-nit display.
/// CICP, mastering metadata and MaxCLL do not select these conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HlgRendering {
    peak: Nits,
    black_bits: u32,
    gamma_bits: u32,
}
impl HlgRendering {
    /// Explicit ideal-black reference rendering: 1000 nits, gamma 1.2,
    /// reference surround (5 cd/m²). Opt in; never inferred from an HLG tag.
    pub const REFERENCE_1000_NIT: Self = Self {
        peak: Nits::new(1000.0),
        black_bits: 0,
        gamma_bits: 1.2f32.to_bits(),
    };
    /// Supply resolved display peak, black luminance and surround-adjusted gamma.
    /// Supports finite gamma 0.5..=3 and black 0..=1% of peak. Other conditions
    /// refuse rather than applying an unqualified extrapolation.
    pub const fn new(peak: Nits, black_nits: f32, system_gamma: f32) -> Option<Self> {
        if !black_nits.is_finite()
            || black_nits < 0.0
            || black_nits > peak.nits() * 0.01
            || !system_gamma.is_finite()
            || system_gamma < 0.5
            || system_gamma > 3.0
        {
            return None;
        }
        Some(Self {
            peak,
            black_bits: black_nits.to_bits(),
            gamma_bits: system_gamma.to_bits(),
        })
    }
    /// Target/reference display peak, not a measurement of source content.
    pub const fn peak(self) -> Nits {
        self.peak
    }
    /// Display black luminance in cd/m².
    pub const fn black_nits(self) -> f32 {
        f32::from_bits(self.black_bits)
    }
    /// Effective gamma including the explicitly selected surround assumption.
    pub const fn system_gamma(self) -> f32 {
        f32::from_bits(self.gamma_bits)
    }
}
