//! HDR metadata and luminance anchor types.
//!
//! Pure data types for HDR content description, diffuse white anchors,
//! explicit HLG display rendering conditions, and peak luminance sources.
//!
//! For tone mapping, gamut compression, and content light-level measurement,
//! see `zenpixels_convert::tonemap` and `zenpixels_convert::measure`.

use core::fmt;
use core::num::NonZeroU32;

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
///
/// Internally stored as a [`NonZeroU32`] IEEE 754 bit pattern so
/// `Option<DiffuseWhite>` fits in 4 bytes via Rust niche optimization.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiffuseWhite(NonZeroU32);

impl fmt::Debug for DiffuseWhite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DiffuseWhite").field(&self.nits()).finish()
    }
}

impl DiffuseWhite {
    /// ITU-R BT.2408 HDR reference white: **203 cd/m²**. The cross-industry
    /// default anchor for relative-linear HDR (matches Chrome `SDRWhiteLevel`,
    /// Skia skcms, CSS `rec2100-linear`, and libplacebo).
    pub const BT2408: Self = Self::new_or_panic(203.0);

    /// Construct a diffuse white anchor of `nits` cd/m², returning `None` unless
    /// `nits` is finite and strictly positive.
    #[inline]
    #[must_use]
    pub const fn new(nits: f32) -> Option<Self> {
        if nits.is_finite() && nits > 0.0 {
            match NonZeroU32::new(nits.to_bits()) {
                Some(bits) => Some(Self(bits)),
                None => None,
            }
        } else {
            None
        }
    }

    /// Construct a diffuse white anchor of `nits` cd/m² (the luminance that
    /// relative-linear `1.0` represents), panicking on invalid input.
    ///
    /// Intended for `const` items and validated literals; prefer [`new`](Self::new)
    /// for runtime inputs.
    ///
    /// # Panics
    ///
    /// Panics if `nits` is zero, negative, NaN, or infinite.
    #[track_caller]
    #[inline]
    #[must_use]
    pub const fn new_or_panic(nits: f32) -> Self {
        assert!(
            nits.is_finite() && nits > 0.0,
            "diffuse white must be finite and strictly positive"
        );
        match NonZeroU32::new(nits.to_bits()) {
            Some(bits) => Self(bits),
            None => unreachable!(),
        }
    }

    /// The anchor in cd/m² (nits).
    #[inline]
    #[must_use]
    pub const fn nits(self) -> f32 {
        f32::from_bits(self.0.get())
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
///
/// Internally stored as a [`NonZeroU32`] IEEE 754 bit pattern so `Option<Nits>`
/// fits in 4 bytes via Rust niche optimization.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Nits(NonZeroU32);

impl fmt::Debug for Nits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Nits").field(&self.nits()).finish()
    }
}

impl Nits {
    /// Construct a luminance in cd/m², returning `None` unless `value` is finite and strictly positive.
    #[inline]
    #[must_use]
    pub const fn new(value: f32) -> Option<Self> {
        if value.is_finite() && value > 0.0 {
            match NonZeroU32::new(value.to_bits()) {
                Some(bits) => Some(Self(bits)),
                None => None,
            }
        } else {
            None
        }
    }
    /// Construct a luminance in cd/m², panicking on invalid input.
    ///
    /// Intended for `const` items and validated literals; prefer [`new`](Self::new)
    /// for runtime inputs.
    ///
    /// # Panics
    /// Panics for zero, negative, NaN or infinite values.
    #[track_caller]
    #[inline]
    #[must_use]
    pub const fn new_or_panic(value: f32) -> Self {
        assert!(
            value.is_finite() && value > 0.0,
            "luminance must be finite and strictly positive"
        );
        match NonZeroU32::new(value.to_bits()) {
            Some(bits) => Self(bits),
            None => unreachable!(),
        }
    }
    /// Luminance in cd/m².
    #[inline]
    #[must_use]
    pub const fn nits(self) -> f32 {
        f32::from_bits(self.0.get())
    }
}

/// How an image-level conversion obtains the source peak for HDR tone mapping.
///
/// Row-only plans accept an explicit `Option<Nits>` instead of `PeakSource`,
/// making it a compile-time error to request [`PeakSource::Measured`] on a
/// single-row converter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PeakSource {
    /// Streamingly measure the image's peak luminance in O(1) row memory.
    #[default]
    Measured,
    /// Use the source peak declared in [`HdrMetadata`](crate::HdrMetadata); refuse if absent.
    Declared,
    /// Use this explicit peak instead of measuring or consulting container metadata.
    Fixed(Nits),
}

/// HDR content light level metadata (CEA-861.3 / CTA-861-H / ITU-T H.274 §8.10).
///
/// Describes the peak light-level characteristics of HDR content.
/// Used by AVIF, JXL, PNG (`cLLI` chunk, scaled in `0.0001 cd/m²` units), and video containers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ContentLightLevel {
    /// Maximum Content Light Level (MaxCLL) in cd/m² (nits).
    /// Peak `max(R, G, B)` light level of any single pixel in the content (ITU-T H.274 §8.10.2).
    pub max_content_light_level: u16,
    /// Maximum Frame-Average Light Level (MaxFALL) in cd/m².
    /// Peak frame-average `max(R, G, B)` light level of any single frame.
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
        assert_eq!(DiffuseWhite::new(100.0).unwrap().nits(), 100.0);
    }

    #[test]
    fn diffuse_white_custom_anchor_round_trips() {
        // Anchor metadata is byte-identical-preserved through the constructor:
        // a custom 100 cd/m² (HDR home-tier mastering) and 10 000 cd/m² (PQ
        // peak) both round-trip through `new` → `nits` losslessly.
        assert_eq!(DiffuseWhite::new(100.0).unwrap().nits(), 100.0);
        assert_eq!(DiffuseWhite::new(10_000.0).unwrap().nits(), 10_000.0);
        // PartialEq honours bit equality (see the impl above) so two
        // independently constructed anchors compare equal.
        assert_eq!(DiffuseWhite::new(203.0), Some(DiffuseWhite::BT2408));
    }

    #[test]
    fn diffuse_white_accepts_positive_finite_bounds_and_const_construction() {
        const CUSTOM: DiffuseWhite = DiffuseWhite::new_or_panic(100.0);
        assert_eq!(CUSTOM.nits(), 100.0);
        for nits in [f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX] {
            assert_eq!(
                DiffuseWhite::new(nits).unwrap().nits().to_bits(),
                nits.to_bits()
            );
        }
    }

    #[test]
    fn diffuse_white_new_returns_none_for_invalid() {
        for invalid in [
            0.0,
            -0.0,
            -100.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert_eq!(DiffuseWhite::new(invalid), None);
        }
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_zero() {
        let _ = DiffuseWhite::new_or_panic(0.0);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_negative_zero() {
        let _ = DiffuseWhite::new_or_panic(-0.0);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_negative() {
        let _ = DiffuseWhite::new_or_panic(-100.0);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_nan() {
        let _ = DiffuseWhite::new_or_panic(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_infinity() {
        let _ = DiffuseWhite::new_or_panic(f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "diffuse white must be finite and strictly positive")]
    fn diffuse_white_rejects_negative_infinity() {
        let _ = DiffuseWhite::new_or_panic(f32::NEG_INFINITY);
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

/// Exact PNG 3 / ITU-T H.274 `cLLI` content light-level codes (`0.0001 cd/m²` precision).
///
/// Separate from the integer-nit [`ContentLightLevel`] view so sub-nit PNG `cLLI`
/// payloads (`max_cll`, `max_fall` as big-endian `u32` in units of `0.0001 cd/m²`)
/// survive lossless transcodes without quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClliCodes([u8; 8]);

impl ClliCodes {
    /// Preserve all 8 big-endian bytes (`[max_cll: u32 BE, max_fall: u32 BE]`).
    #[inline]
    #[must_use]
    pub const fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Construct from exact `0.0001 cd/m²` integer units (PNG `cLLI` wire scale).
    #[inline]
    #[must_use]
    pub const fn from_ten_thousandths(max_cll_10000: u32, max_fall_10000: u32) -> Self {
        let a = max_cll_10000.to_be_bytes();
        let b = max_fall_10000.to_be_bytes();
        Self([a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]])
    }

    /// Return the exact original 8 big-endian bytes without float or integer rounding.
    #[inline]
    #[must_use]
    pub const fn to_be_bytes(self) -> [u8; 8] {
        self.0
    }

    /// Exact MaxCLL in units of `0.0001 cd/m²`.
    #[inline]
    #[must_use]
    pub const fn max_cll_ten_thousandths(self) -> u32 {
        u32::from_be_bytes([self.0[0], self.0[1], self.0[2], self.0[3]])
    }

    /// Exact MaxFALL in units of `0.0001 cd/m²`.
    #[inline]
    #[must_use]
    pub const fn max_fall_ten_thousandths(self) -> u32 {
        u32::from_be_bytes([self.0[4], self.0[5], self.0[6], self.0[7]])
    }

    /// MaxCLL in cd/m² (`f64`).
    #[inline]
    #[must_use]
    pub const fn max_cll_nits(self) -> f64 {
        self.max_cll_ten_thousandths() as f64 / 10_000.0
    }

    /// MaxFALL in cd/m² (`f64`).
    #[inline]
    #[must_use]
    pub const fn max_fall_nits(self) -> f64 {
        self.max_fall_ten_thousandths() as f64 / 10_000.0
    }

    /// Project to integer-nit [`ContentLightLevel`] (rounded to nearest nit, clamped to `u16::MAX`).
    ///
    /// Retain `ClliCodes` when exact `0.0001 cd/m²` precision must survive a transcode.
    #[inline]
    #[must_use]
    pub const fn semantic(self) -> ContentLightLevel {
        let cll = (self.max_cll_ten_thousandths() as u64 + 5_000) / 10_000;
        let fall = (self.max_fall_ten_thousandths() as u64 + 5_000) / 10_000;
        ContentLightLevel::new(
            if cll > u16::MAX as u64 {
                u16::MAX
            } else {
                cll as u16
            },
            if fall > u16::MAX as u64 {
                u16::MAX
            } else {
                fall as u16
            },
        )
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
        peak: Nits::new_or_panic(1000.0),
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
        let black_nits = if black_nits == 0.0 { 0.0 } else { black_nits };
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
