//! HDR processing utilities.
//!
//! Re-exports [`ContentLightLevel`] and [`MasteringDisplay`] from the
//! `zenpixels` crate for convenience. Adds [`HdrMetadata`] (which bundles
//! transfer function with the metadata types) and tone mapping helpers.
//!
//! The core PQ/HLG EOTF/OETF math is always available through the main
//! conversion pipeline in [`ConvertPlan`](crate::ConvertPlan).

use crate::TransferFunction;

// Re-export metadata types from the core crate.
pub use zenpixels::hdr::{ContentLightLevel, MasteringDisplay};

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
    ///
    /// The mastering display is [`MasteringDisplay::HDR10_REFERENCE`] — the
    /// generic 1000-nit reference mastering volume, **not** measured
    /// metadata from any real mastering session. Replace it when the
    /// source carries an actual SMPTE ST 2086 record.
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
/// Maps linear light `[0, ∞]` → `[0, 1]` using `v / (1 + v)`.
///
/// Out-of-domain inputs are clamped rather than propagated: **negative
/// values and NaN map to 0.0** (linear HDR buffers can legitimately carry
/// small negatives from gamut-mapping ringing — pre-clamp, `-1.0` produced
/// `-inf` and `-2.0` produced `+2.0`), and **`+∞` maps to 1.0** (the
/// mathematical limit). The output never leaves `[0, 1]`; it reaches 1.0
/// only at the float saturation edge.
///
/// Preserves relative brightness ordering. Does not use any display
/// metadata — for proper tone mapping, use a dedicated HDR tone mapping
/// library.
#[inline]
#[must_use]
pub fn reinhard_tonemap(v: f32) -> f32 {
    // f32::max(NaN, 0.0) == 0.0, so one clamp handles negatives and NaN.
    let v = v.max(0.0);
    if v == f32::INFINITY {
        return 1.0;
    }
    v / (1.0 + v)
}

/// Inverse Reinhard: SDR linear → HDR linear.
///
/// Maps `[0, 1)` → `[0, ∞)` using `v / (1 - v)`. Inputs ≥ 1.0 saturate to
/// `f32::MAX` (1.0 has no finite preimage); **negative values and NaN map
/// to 0.0**, mirroring [`reinhard_tonemap`]'s domain clamp.
#[inline]
#[must_use]
pub fn reinhard_inverse(v: f32) -> f32 {
    let v = v.max(0.0);
    if v >= 1.0 {
        return f32::MAX;
    }
    v / (1.0 - v)
}

/// Simple exposure-based tone mapping.
///
/// `exposure` is in stops relative to 1.0. Positive values brighten,
/// negative darken. The result is clamped to [0, 1]; **NaN input maps to
/// 0.0** (consistent with [`reinhard_tonemap`]'s domain clamp).
///
/// Requires `std` because `f32::powf` is not available in `no_std`.
#[cfg(feature = "std")]
#[inline]
#[must_use]
pub fn exposure_tonemap(v: f32, exposure: f32) -> f32 {
    // .max then .min instead of .clamp: max(NaN, 0.0) == 0.0 makes the
    // NaN result deterministic, where clamp would propagate it.
    (v * 2.0f32.powf(exposure)).max(0.0).min(1.0)
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
    #[cfg(feature = "std")]
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
    fn hdr_metadata_clone_partial_eq() {
        let a = HdrMetadata::hlg();
        let b = a;
        assert_eq!(a, b);
    }

    // -- Rung 1 hardening (zenpixels#39): domain contracts + properties --

    /// Independent f64 oracle for the f32 implementation.
    fn reinhard_f64(v: f64) -> f64 {
        v / (1.0 + v)
    }

    #[test]
    fn reinhard_clamps_negatives_and_nan_to_zero() {
        // Pre-clamp hazards: -1.0 → -inf, -2.0 → +2.0 (outside [0,1]).
        assert_eq!(reinhard_tonemap(-0.25), 0.0);
        assert_eq!(reinhard_tonemap(-1.0), 0.0);
        assert_eq!(reinhard_tonemap(-2.0), 0.0);
        assert_eq!(reinhard_tonemap(f32::NEG_INFINITY), 0.0);
        assert_eq!(reinhard_tonemap(f32::NAN), 0.0);

        assert_eq!(reinhard_inverse(-0.25), 0.0);
        assert_eq!(reinhard_inverse(-1.0), 0.0);
        assert_eq!(reinhard_inverse(f32::NAN), 0.0);
    }

    #[test]
    fn reinhard_infinity_saturates_to_one() {
        // inf/(1+inf) would be NaN; the limit is 1.0.
        assert_eq!(reinhard_tonemap(f32::INFINITY), 1.0);
        // The float saturation edge also rounds to 1.0 (MAX + 1 == MAX).
        assert_eq!(reinhard_tonemap(f32::MAX), 1.0);
    }

    #[test]
    fn reinhard_output_range_and_monotonicity() {
        let grid: [f32; 13] = [
            0.0, 1e-6, 1e-3, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 1e3, 1e6, 1e9, f32::MAX,
        ];
        let mut prev = -1.0f32;
        for &v in &grid {
            let out = reinhard_tonemap(v);
            assert!(
                (0.0..=1.0).contains(&out) && out.is_finite(),
                "reinhard_tonemap({v}) = {out} escapes [0, 1]"
            );
            assert!(out >= prev, "not monotonic at {v}: {out} < {prev}");
            // Strictly increasing while far from the saturation edge.
            if v <= 1e6 && prev >= 0.0 {
                assert!(out > prev, "not strictly increasing at {v}");
            }
            prev = out;
        }
    }

    #[test]
    fn reinhard_matches_f64_oracle() {
        for &v in &[0.0f32, 1e-6, 1e-3, 0.1, 0.5, 1.0, 2.0, 10.0, 1e3, 1e5] {
            let got = reinhard_tonemap(v) as f64;
            let want = reinhard_f64(v as f64);
            assert!(
                (got - want).abs() < 1e-6,
                "f32 impl diverges from f64 oracle at {v}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn reinhard_roundtrip_relative_error_bound() {
        // inverse(tonemap(v)) ≈ v across eight decades. The inverse
        // amplifies the f32 quantization of t = v/(1+v) (whose spacing is
        // ~ε once t nears 1.0) by dv/dt = (1+v)², so the relative
        // round-trip error grows ~linearly in v; bound it at 4ε·(1+v).
        let mut v = 1e-4f32;
        while v <= 1e4 {
            let rt = reinhard_inverse(reinhard_tonemap(v));
            let rel = ((f64::from(rt) - f64::from(v)) / f64::from(v)).abs();
            let bound = 4.0 * f64::from(f32::EPSILON) * (1.0 + f64::from(v));
            assert!(
                rel < bound,
                "roundtrip rel err {rel} > bound {bound} at {v} (got {rt})"
            );
            v *= 3.7;
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn exposure_tonemap_nan_maps_to_zero() {
        assert_eq!(exposure_tonemap(f32::NAN, 0.0), 0.0);
        assert_eq!(exposure_tonemap(f32::NAN, 2.0), 0.0);
        // Negative input still clamps to 0 (unchanged behavior).
        assert_eq!(exposure_tonemap(-0.5, 0.0), 0.0);
    }
}
