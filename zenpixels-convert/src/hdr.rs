//! HDR processing utilities.
//!
//! Re-exports [`ContentLightLevel`] and [`MasteringDisplay`] from the
//! `zenpixels` crate for convenience. Adds [`HdrMetadata`] (which bundles
//! transfer function with the metadata types) and tone mapping helpers.
//!
//! The core PQ/HLG EOTF/OETF math is always available through the main
//! conversion pipeline in [`ConvertPlan`](crate::ConvertPlan).

use crate::adapt::convert_buffer;
use crate::error::ConvertError;
use crate::{PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice, TransferFunction};
use alloc::vec::Vec;
use whereat::At;

// Re-export metadata types from the core crate.
pub use zenpixels::hdr::{ContentLightLevel, MasteringDisplay};
// `quantize_to` reads the anchor from the source's `ColorContext`; the
// canonical public home for the type is `zenpixels::hdr::DiffuseWhite`
// (reachable through the core crate — not re-exported here).
use zenpixels::hdr::DiffuseWhite;

/// Describes the HDR characteristics of pixel data.
///
/// Bundles transfer function, content light level, and mastering display
/// metadata to provide everything needed for HDR processing.
///
/// # Deprecated
///
/// This bundle is a redundant, weaker duplicate of the codec-layer carrier
/// `zencodec::Metadata` (which the codecs actually populate, and which also
/// carries CICP, ICC, EXIF/XMP, and orientation). It bundles `transfer` with
/// CLL/mastering, which the prior art uniformly keeps separate (transfer
/// belongs on the [`PixelDescriptor`](crate::PixelDescriptor); CLL and
/// mastering are independent optional metadata). It has frozen public fields
/// (not `#[non_exhaustive]`), so the absolute-luminance anchor and gain-map
/// fields HDR needs cannot be added without a break. Scheduled for removal in
/// 0.3.0 — see `CHANGELOG.md` "QUEUED BREAKING CHANGES". Carry CLL and
/// mastering as the standalone [`ContentLightLevel`] / [`MasteringDisplay`]
/// types, or use `zencodec::Metadata`.
#[deprecated(
    since = "0.2.14",
    note = "redundant with zencodec::Metadata and frozen-shaped; carry ContentLightLevel / MasteringDisplay directly. Removal queued for 0.3.0."
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrMetadata {
    /// Transfer function (PQ, HLG, sRGB, Linear, etc.).
    pub transfer: TransferFunction,
    /// Content light level (MaxCLL/MaxFALL). Optional.
    pub content_light_level: Option<ContentLightLevel>,
    /// Mastering display color volume. Optional.
    pub mastering_display: Option<MasteringDisplay>,
}

#[allow(deprecated)]
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
// Not clamp(): max(NaN, 0.0) == 0.0 makes the NaN result deterministic
// (the documented contract above), where clamp would propagate NaN.
#[allow(clippy::manual_clamp)]
pub fn exposure_tonemap(v: f32, exposure: f32) -> f32 {
    (v * 2.0f32.powf(exposure)).max(0.0).min(1.0)
}

// ---------------------------------------------------------------------------
// HDR quantization (relative-linear f32 → a PQ HDR descriptor)
// ---------------------------------------------------------------------------

/// Read one native-endian f32 sample from a pixel's bytes.
#[inline]
fn sample_f32(bytes: &[u8], k: usize) -> f32 {
    f32::from_ne_bytes([
        bytes[4 * k],
        bytes[4 * k + 1],
        bytes[4 * k + 2],
        bytes[4 * k + 3],
    ])
}

/// Quantize relative-linear RGB(A) f32 pixels to a **PQ** HDR target
/// descriptor (e.g. [`PixelDescriptor::RGB16_BT2100_PQ`]).
///
/// The absolute-luminance anchor — the nits that sample `1.0` represents — is
/// read from the source `ColorContext`'s `diffuse_white`, defaulting to
/// [`DiffuseWhite::BT2408`] (203, the cross-vendor relative-linear convention)
/// when unsignaled. Attach a custom anchor with
/// `ColorContext::with_diffuse_white` (e.g. a buffer reconstructed at a
/// different reference white). The anchor is the one thing the pipeline's fixed
/// `1.0 = 10000 cd/m²` PQ domain can't yet express; the transfer (linear → PQ)
/// and depth (f32 → u16) quantization reuse
/// [`convert_buffer`](crate::adapt::convert_buffer). The anchor scales the
/// linear samples by `anchor / 10000`, clamps to the PQ peak, and drops any
/// alpha lane. PQ codes match the f64 ST 2084 oracle within ±1.
///
/// **Primaries are not converted** — the source gamut is signaled as the
/// target's (feed BT.2020-relative-linear for `RGB16_BT2100_PQ`). Measure CLL
/// separately with [`ContentLightLevel::measure`].
///
/// The successor to the withdrawn `encode_pq16` (rationale:
/// `docs/hdr-design-survey-2026-06-13.md`). It collapses fully into
/// `convert_buffer` once the anchor threads into the PQ/HLG `ConvertStep`s
/// themselves — tracked as a refinement in the M×N HDR epic (#45).
///
/// # Errors
///
/// - [`ConvertError::NoMatch`] if `px` is not `RgbF32`/`RgbaF32`;
///   [`ConvertError::UnsupportedTransfer`] if it is not `Linear`.
/// - [`ConvertError::NoPath`] if `target`'s transfer is not PQ (HLG's
///   scene-referred anchor differs and is not handled here).
/// - [`ConvertError::InvalidWidth`] for zero-area input, or any error the
///   inner [`convert_buffer`](crate::adapt::convert_buffer) raises.
pub fn quantize_to(
    px: PixelSlice<'_>,
    target: PixelDescriptor,
) -> Result<PixelBuffer, At<ConvertError>> {
    // The anchor travels with the pixels (S1a): read it from the source's
    // ColorContext, default to the BT.2408 relative-linear convention (203).
    let white = px
        .color_context()
        .and_then(|c| c.diffuse_white)
        .unwrap_or(DiffuseWhite::BT2408);
    let desc = px.descriptor();
    let channels = match desc.pixel_format() {
        PixelFormat::RgbF32 => 3,
        PixelFormat::RgbaF32 => 4,
        _ => return Err(whereat::at!(ConvertError::NoMatch { source: desc })),
    };
    if desc.transfer != TransferFunction::Linear {
        return Err(whereat::at!(ConvertError::UnsupportedTransfer {
            from: desc.transfer,
            to: TransferFunction::Linear,
        }));
    }
    // The pipeline anchors PQ at 1.0 = 10000 cd/m²; only PQ targets are handled.
    if target.transfer != TransferFunction::Pq {
        return Err(whereat::at!(ConvertError::NoPath {
            from: desc,
            to: target,
        }));
    }
    let w = px.width();
    let h = px.rows();
    if w == 0 || h == 0 {
        return Err(whereat::at!(ConvertError::InvalidWidth(w)));
    }

    // Pre-scale relative-linear (1.0 = white nits) into the pipeline's PQ
    // domain (1.0 = 10000 nits), clamp to the peak, drop alpha → tight RGB f32.
    let factor = (f64::from(white.nits()) / 10_000.0) as f32;
    let stride = px.stride();
    let bytes = px.as_strided_bytes();
    let row_len = w as usize * channels * 4;
    let mut scaled: Vec<u8> = Vec::with_capacity(w as usize * h as usize * 3 * 4);
    for row in 0..h as usize {
        let rb = &bytes[row * stride..row * stride + row_len];
        for pxl in rb.chunks_exact(channels * 4) {
            for k in 0..3 {
                let v = (sample_f32(pxl, k).max(0.0) * factor).min(1.0);
                scaled.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }

    // Reuse the pipeline for linear→PQ + f32→u16. Tag the scratch with the
    // target's primaries so no gamut step is inserted (value-only quantize).
    let src = PixelDescriptor::RGBF32_LINEAR.with_primaries(target.primaries);
    let out = convert_buffer(&scaled, w, h, src, target)?;
    PixelBuffer::from_vec(out, w, h, target)
        .map_err(|_| whereat::at!(ConvertError::AllocationFailed))
}

#[cfg(test)]
// These tests exercise the deprecated-but-still-present HdrMetadata API.
#[allow(deprecated)]
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
            0.0,
            1e-6,
            1e-3,
            0.05,
            0.1,
            0.5,
            1.0,
            2.0,
            10.0,
            1e3,
            1e6,
            1e9,
            f32::MAX,
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

    // -- quantize_to (PQ16) parity with the f64 ST 2084 oracle --

    use alloc::vec::Vec;

    /// f64 SMPTE ST 2084 inverse-EOTF oracle (exact constants).
    fn pq_oracle(x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let m1 = 2610.0 / 16384.0;
        let m2 = 2523.0 / 4096.0 * 128.0;
        let c1 = 3424.0 / 4096.0;
        let c2 = 2413.0 / 4096.0 * 32.0;
        let c3 = 2392.0 / 4096.0 * 32.0;
        let xp = x.powf(m1);
        ((c1 + c2 * xp) / (1.0 + c3 * xp)).powf(m2)
    }

    fn rgbf32(pixels: &[[f32; 3]], w: u32, h: u32) -> PixelBuffer {
        let mut data = Vec::with_capacity(pixels.len() * 12);
        for p in pixels {
            for c in p {
                data.extend_from_slice(&c.to_ne_bytes());
            }
        }
        PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBF32_LINEAR).unwrap()
    }

    #[test]
    fn quantize_to_pq16_white_and_peak() {
        // 1.0 @ 203 nits → PQ(203/10000); 10000/203 → PQ(1.0) = code 65535.
        let peak = 10_000.0 / 203.0;
        let buf = rgbf32(&[[1.0; 3], [peak; 3]], 2, 1);
        let out = quantize_to(buf.as_slice(), PixelDescriptor::RGB16_BT2100_PQ).unwrap();
        assert_eq!(out.descriptor(), PixelDescriptor::RGB16_BT2100_PQ);
        let bytes = out.as_slice().as_strided_bytes();
        let code = |i: usize| u16::from_ne_bytes([bytes[2 * i], bytes[2 * i + 1]]);

        let want_white = (pq_oracle(203.0 / 10_000.0) * 65535.0).round() as i64;
        assert!((i64::from(code(0)) - want_white).abs() <= 1);
        assert_eq!(code(3), 65535, "10000-nit peak clips to full code");
    }

    #[test]
    fn quantize_to_pq16_matches_oracle_across_decades() {
        let values = [0.001f32, 0.01, 0.1, 0.5, 1.0, 2.0, 8.0, 20.0, 49.0];
        let pixels: Vec<[f32; 3]> = values.iter().map(|&v| [v; 3]).collect();
        let buf = rgbf32(&pixels, values.len() as u32, 1);
        let out = quantize_to(buf.as_slice(), PixelDescriptor::RGB16_BT2100_PQ).unwrap();
        let bytes = out.as_slice().as_strided_bytes();
        for (i, &v) in values.iter().enumerate() {
            let got = i64::from(u16::from_ne_bytes([bytes[6 * i], bytes[6 * i + 1]]));
            let x = f64::from(v) * 203.0 / 10_000.0;
            let want = (pq_oracle(x) * 65535.0).round() as i64;
            assert!(
                (got - want).abs() <= 1,
                "PQ16 at {v}: got {got}, oracle {want}"
            );
        }
    }

    #[test]
    fn quantize_to_rejects_non_pq_target_and_non_linear_src() {
        let buf = rgbf32(&[[0.5; 3]], 1, 1);
        // HLG target → NoPath (anchor semantics differ).
        let err = quantize_to(buf.as_slice(), PixelDescriptor::RGB16_BT2100_HLG).unwrap_err();
        assert!(matches!(*err.error(), ConvertError::NoPath { .. }));
        // Non-linear source → UnsupportedTransfer.
        let srgb = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
        let mut d = Vec::new();
        for c in [0.5f32; 3] {
            d.extend_from_slice(&c.to_ne_bytes());
        }
        let nb = PixelBuffer::from_vec(d, 1, 1, srgb).unwrap();
        assert!(quantize_to(nb.as_slice(), PixelDescriptor::RGB16_BT2100_PQ).is_err());
    }

    #[test]
    fn quantize_to_reads_anchor_from_color_context() {
        use alloc::sync::Arc;
        use zenpixels::{Cicp, ColorContext};
        // A 100-nit anchor on the ColorContext (not the 203 default) must
        // change the PQ scale — proving the anchor travels with the pixels.
        let buf = rgbf32(&[[1.0; 3]], 1, 1).with_color_context(Arc::new(
            ColorContext::from_cicp(Cicp::BT2100_PQ).with_diffuse_white(DiffuseWhite::new(100.0)),
        ));
        let out = quantize_to(buf.as_slice(), PixelDescriptor::RGB16_BT2100_PQ).unwrap();
        let bytes = out.as_slice().as_strided_bytes();
        let got = i64::from(u16::from_ne_bytes([bytes[0], bytes[1]]));
        let want = (pq_oracle(100.0 / 10_000.0) * 65535.0).round() as i64;
        assert!(
            (got - want).abs() <= 1,
            "100-nit anchor: got {got}, want {want}"
        );
        // The 100-nit result differs from the 203-nit default for the same input.
        let want_203 = (pq_oracle(203.0 / 10_000.0) * 65535.0).round() as i64;
        assert_ne!(want, want_203);
    }
}
