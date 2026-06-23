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

// `PixelFormat` / `PixelSlice` / `TransferFunction` are intentionally NOT
// imported here. The 0.2.15 release left them in for the deprecated
// `ContentLightLevel::measure` inherent method; that method was removed in
// 0.2.16 (per CHANGELOG QUEUED BREAKING CHANGES) — the canonical replacement
// is `zenpixels_convert::hdr::measure::CllMeasure::measure_max` (and the
// other histogram-based readouts on the trait). This module is now pure
// metadata / anchor types: scalar `DiffuseWhite`, the `ContentLightLevel`
// struct, and `MasteringDisplay`, with no pixel-data reduction.

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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl DiffuseWhite {
    /// ITU-R BT.2408 HDR reference white: **203 cd/m²**. The cross-industry
    /// default anchor for relative-linear HDR (matches Chrome `SDRWhiteLevel`,
    /// Skia skcms, CSS `rec2100-linear`, and libplacebo).
    pub const BT2408: Self = Self(203.0);

    /// An anchor of `nits` cd/m² (the luminance that relative-linear `1.0`
    /// represents).
    #[must_use]
    pub const fn new(nits: f32) -> Self {
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

/// HDR content light level metadata (CEA-861.3 / CTA-861-H).
///
/// Describes the peak brightness characteristics of HDR content.
/// Used by AVIF, JXL, PNG (cLLi chunk), and video containers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Tail-tightest percentile for explicit percentile-based MaxCLL.
    ///
    /// `0.99999` — the 99.999th percentile, dropping the top 0.001 % of
    /// pixels. Empirically the tail-tightest tested value in the
    /// 2026-06-22 audited HDR→SDR shootout (76 imazen-26 samples × 20
    /// curves × 4 peak methods, scored on tail-aware metrics + OKLab
    /// Euclidean ΔE against the producer SDR base): won every per-image
    /// tail metric (`de2000_p95`, `de2000_p99`, `de_ok_p95`) by 1.4-1.8 %
    /// over the literal-max alternative.
    ///
    /// **The recommended production default is the literal max**
    /// (`CllMeasure::measure_max`), NOT this percentile — the same
    /// shootout showed `measure_max` wins on 3 of 6 metrics including
    /// the user-visible `pct_above_de5` (11 % fewer clearly-different
    /// pixels). This constant exists for callers who explicitly opt
    /// into percentile-based measurement via `measure_percentile`
    /// because their content policy needs the tail-tighter trade-off
    /// (defect-noisy capture path, single hot pixels would over-drive
    /// downstream tone-mapping). See
    /// `zen/zentone/benchmarks/shootout_2026-06-22_findings_v2.md`.
    ///
    /// **Sparse-bright cliff.** Content that occupies < 0.001 % of
    /// pixels is silently dropped at any image size. For 24 MP that's
    /// anything below ~240 pixels; for 1 MP, below ~10 pixels; for
    /// small images (< 100 000 pixels) the fraction rounds to "any
    /// single bright pixel". Astrophotography, fireworks, and
    /// candle-in-dark-room content where every bright pixel is
    /// legitimate should use the literal max reading instead.
    ///
    /// **Bin quantisation.** The percentile readout reports the
    /// lower edge of the log2 histogram bin that contains the
    /// percentile-threshold pixel — up to ~2 % (one bin = ~0.02 stops)
    /// below the literal max even when all content lives in one bin.
    /// Acceptable for HDR metadata at the u16-nits granularity CTA-861.3
    /// encodes, but documented so callers comparing against the literal
    /// max know to expect this.
    ///
    /// Used by `CllMeasure::measure_robust` in `zenpixels-convert`.
    /// Explicit callers who want a non-default percentile pass their own
    /// value to `CllMeasure::measure_percentile`.
    #[doc(hidden)]
    pub const DEFAULT_PERCENTILE: f32 = 0.99999;
}

/// Mastering display color volume metadata (SMPTE ST 2086).
///
/// Describes the display on which the content was mastered, enabling
/// downstream displays to reproduce the creator's intent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    fn default_percentile_constant_is_stable() {
        // Pin the constant — `zenpixels-convert::CllMeasure::measure_percentile`
        // reads this as its industry-tail default. Any change here breaks the
        // documented production tail metric.
        assert_eq!(ContentLightLevel::DEFAULT_PERCENTILE, 0.99999);
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
