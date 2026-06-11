//! HDR processing utilities.
//!
//! Re-exports [`ContentLightLevel`] and [`MasteringDisplay`] from the
//! `zenpixels` crate for convenience. Adds [`HdrMetadata`] (which bundles
//! transfer function with the metadata types) and tone mapping helpers.
//!
//! The core PQ/HLG EOTF/OETF math is always available through the main
//! conversion pipeline in [`ConvertPlan`](crate::ConvertPlan).

use crate::TransferFunction;
use crate::error::ConvertError;
use crate::{PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice};
use alloc::vec::Vec;
use whereat::At;

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

// ---------------------------------------------------------------------------
// Display-light measurement + PQ16 output encoding (zenpixels#39 Rung 2)
// ---------------------------------------------------------------------------

/// BT.2408 reference diffuse white in cd/m² — the conventional anchor for
/// "linear 1.0" in relative-linear HDR buffers, and the customary
/// `diffuse_white_nits` argument for [`compute_content_light_level`] and
/// [`encode_pq16`].
pub const REFERENCE_DIFFUSE_WHITE_NITS: f32 = 203.0;

/// Gate shared by the Rung-2 helpers: relative-linear RGB(A) f32 only.
#[inline]
fn require_linear_f32_rgb(desc: &PixelDescriptor) -> Result<usize, At<ConvertError>> {
    let channels = match desc.pixel_format() {
        PixelFormat::RgbF32 => 3,
        PixelFormat::RgbaF32 => 4,
        _ => return Err(whereat::at!(ConvertError::NoMatch { source: *desc })),
    };
    if desc.transfer != TransferFunction::Linear {
        return Err(whereat::at!(ConvertError::UnsupportedTransfer {
            from: desc.transfer,
            to: TransferFunction::Linear,
        }));
    }
    Ok(channels)
}

#[inline]
fn nits_to_u16(nits: f64) -> u16 {
    nits.round().clamp(0.0, 65535.0) as u16
}

/// Read one native-endian f32 sample (per the `PixelSlice` endianness
/// contract); no alignment assumption on the backing bytes.
#[inline]
fn sample_f32(bytes: &[u8], k: usize) -> f32 {
    f32::from_ne_bytes([bytes[4 * k], bytes[4 * k + 1], bytes[4 * k + 2], bytes[4 * k + 3]])
}

/// Measure MaxCLL / MaxFALL (CTA-861.3-A) from relative-linear RGB(A) f32
/// pixels.
///
/// `diffuse_white_nits` anchors the relative-linear scale: a sample value of
/// `1.0` is that many cd/m² ([`REFERENCE_DIFFUSE_WHITE_NITS`] — 203, per
/// BT.2408 — is the convention). Semantics per CTA-861.3-A as PNG 3rd ed
/// §11.3.2.8 imports it for stills ("each frame is analyzed"; one still is
/// one frame): **MaxCLL** is the brightest pixel's `max(R, G, B)` in cd/m²,
/// and **MaxFALL** is this image's average of per-pixel `max(R, G, B)`.
///
/// Negative and NaN samples clamp to 0 (the same domain convention as
/// [`reinhard_tonemap`]); an alpha lane is ignored. Strided rows are
/// handled. (Zero-area input cannot be constructed — buffer/slice
/// validation rejects it; the `(0, 0)` early return is defensive.)
///
/// # Errors
///
/// [`ConvertError::NoMatch`] unless the descriptor is `RgbF32`/`RgbaF32`,
/// [`ConvertError::UnsupportedTransfer`] unless its transfer is `Linear`.
pub fn compute_content_light_level(
    px: PixelSlice<'_>,
    diffuse_white_nits: f32,
) -> Result<ContentLightLevel, At<ConvertError>> {
    let desc = px.descriptor();
    let channels = require_linear_f32_rgb(&desc)?;
    let w = px.width() as usize;
    let h = px.rows() as usize;
    if w == 0 || h == 0 {
        return Ok(ContentLightLevel::new(0, 0));
    }

    let stride = px.stride();
    let bytes = px.as_strided_bytes();
    let row_len = w * channels * 4;

    let mut max_nits = 0.0f64;
    let mut sum_max_nits = 0.0f64;
    for row in 0..h {
        let row_bytes = &bytes[row * stride..row * stride + row_len];
        for pxl in row_bytes.chunks_exact(channels * 4) {
            // Fold from 0.0 so NaN and negative samples drop out.
            let m = 0.0f32
                .max(sample_f32(pxl, 0))
                .max(sample_f32(pxl, 1))
                .max(sample_f32(pxl, 2));
            let nits = f64::from(m) * f64::from(diffuse_white_nits);
            max_nits = max_nits.max(nits);
            sum_max_nits += nits;
        }
    }
    let fall = sum_max_nits / (w as f64 * h as f64);
    Ok(ContentLightLevel::new(nits_to_u16(max_nits), nits_to_u16(fall)))
}

/// Quantize relative-linear RGB(A) f32 pixels to PQ-encoded 16-bit RGB
/// (SMPTE ST 2084, full range), measuring [`ContentLightLevel`] in the same
/// pass.
///
/// `diffuse_white_nits` anchors the scale exactly as in
/// [`compute_content_light_level`]. Each channel maps
/// `pq(clamp(sample, 0, ∞) · diffuse_white_nits / 10000)` onto `0..=65535`;
/// luminance above the 10 000 cd/m² PQ peak clips to code 65535. An alpha
/// lane is dropped. Negative/NaN samples clamp to 0.
///
/// The output buffer's descriptor is [`PixelDescriptor::RGB16_BT2100_PQ`]
/// (BT.2020 primaries, PQ transfer, full range — CICP `(9, 16, 0, full)`).
/// **Signal it CICP-natively** where the container has a carrier (PNG
/// `cICP`, AVIF `nclx`, JXL enum color); `synthesize_icc_for_cicp` can
/// supply a compatibility ICC, but the PQ `curv`-LUT profile softens ≈8 %
/// at ~1 nit, so the ICC is a fallback — never the primary signal.
///
/// Note: this quantizes values only — it does **not** convert primaries.
/// Feed it BT.2020-relative linear data (or accept the source gamut being
/// signaled as-is by the caller, as corpus tooling does deliberately).
///
/// # Errors
///
/// As [`compute_content_light_level`], plus
/// [`ConvertError::AllocationFailed`] if the output buffer cannot be
/// built. (The zero-area guard is defensive — such slices cannot be
/// constructed.)
pub fn encode_pq16(
    px: PixelSlice<'_>,
    diffuse_white_nits: f32,
) -> Result<(PixelBuffer, ContentLightLevel), At<ConvertError>> {
    let desc = px.descriptor();
    let channels = require_linear_f32_rgb(&desc)?;
    let w = px.width() as usize;
    let h = px.rows() as usize;
    if w == 0 || h == 0 {
        return Err(whereat::at!(ConvertError::InvalidWidth(px.width())));
    }

    let stride = px.stride();
    let bytes = px.as_strided_bytes();
    let row_len = w * channels * 4;
    // sample → PQ-domain input: nits / 10000 = sample · (diffuse_white / 10000)
    let to_pq_domain = f64::from(diffuse_white_nits) / 10_000.0;

    let mut out: Vec<u8> = Vec::with_capacity(w * h * 3 * 2);
    let mut max_nits = 0.0f64;
    let mut sum_max_nits = 0.0f64;
    for row in 0..h {
        let row_bytes = &bytes[row * stride..row * stride + row_len];
        for pxl in row_bytes.chunks_exact(channels * 4) {
            let mut px_max = 0.0f32;
            for k in 0..3 {
                let c = sample_f32(pxl, k).max(0.0); // NaN/negative → 0
                px_max = px_max.max(c);
                let x = (f64::from(c) * to_pq_domain).min(1.0) as f32;
                let q = (linear_srgb::tf::linear_to_pq(x) * 65535.0).round() as u16;
                out.extend_from_slice(&q.to_ne_bytes());
            }
            let nits = f64::from(px_max) * f64::from(diffuse_white_nits);
            max_nits = max_nits.max(nits);
            sum_max_nits += nits;
        }
    }

    let buffer = PixelBuffer::from_vec(
        out,
        px.width(),
        px.rows(),
        PixelDescriptor::RGB16_BT2100_PQ,
    )
    .map_err(|_| whereat::at!(ConvertError::AllocationFailed))?;
    let fall = sum_max_nits / (w as f64 * h as f64);
    Ok((
        buffer,
        ContentLightLevel::new(nits_to_u16(max_nits), nits_to_u16(fall)),
    ))
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

    // -- Rung 2 (zenpixels#39): CLL measurement + PQ16 encoding --

    use alloc::vec;

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

    fn rgbf32_buffer(pixels: &[[f32; 3]], w: u32, h: u32) -> PixelBuffer {
        let mut data = Vec::with_capacity(pixels.len() * 12);
        for p in pixels {
            for c in p {
                data.extend_from_slice(&c.to_ne_bytes());
            }
        }
        PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBF32_LINEAR).unwrap()
    }

    #[test]
    fn cll_two_grays_matches_cta_stills_semantics() {
        // MaxCLL = brightest pixel's max(R,G,B); MaxFALL = image average of
        // per-pixel max — for [1.0, 2.0] @ 203 nits: (406, round(304.5)=305).
        let buf = rgbf32_buffer(&[[1.0; 3], [2.0; 3]], 2, 1);
        let cll =
            compute_content_light_level(buf.as_slice(), REFERENCE_DIFFUSE_WHITE_NITS).unwrap();
        assert_eq!(cll.max_content_light_level, 406);
        assert_eq!(cll.max_frame_average_light_level, 305);
    }

    #[test]
    fn cll_clamps_nan_and_negative_samples() {
        let buf = rgbf32_buffer(&[[-1.0, f32::NAN, 0.5]], 1, 1);
        let cll =
            compute_content_light_level(buf.as_slice(), REFERENCE_DIFFUSE_WHITE_NITS).unwrap();
        // max(R,G,B) folds from 0.0 → 0.5 · 203 = 101.5 → 102.
        assert_eq!(cll.max_content_light_level, 102);
        assert_eq!(cll.max_frame_average_light_level, 102);
    }

    #[test]
    fn cll_ignores_alpha_lane() {
        let mut data = Vec::new();
        for c in [0.5f32, 0.5, 0.5, 7.0] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, PixelDescriptor::RGBAF32_LINEAR).unwrap();
        let cll =
            compute_content_light_level(buf.as_slice(), REFERENCE_DIFFUSE_WHITE_NITS).unwrap();
        assert_eq!(cll.max_content_light_level, 102); // alpha 7.0 not a light level
    }

    #[test]
    fn cll_strided_matches_tight() {
        let pixels = [[0.25f32, 0.5, 1.0], [2.0, 0.1, 0.3], [0.0, 0.0, 4.0], [1.5; 3]];
        let tight = rgbf32_buffer(&pixels, 2, 2);
        let want =
            compute_content_light_level(tight.as_slice(), REFERENCE_DIFFUSE_WHITE_NITS).unwrap();

        // Same pixels with one whole padding pixel (12 bytes) per row —
        // stride must be pixel-aligned. Back the slice with a Vec<f32> so
        // the pointer satisfies the f32 alignment check in `PixelSlice::new`.
        let stride_f32 = 2 * 3 + 3;
        let mut padded = vec![0.0f32; stride_f32 * 2];
        for row in 0..2 {
            for col in 0..2 {
                for (k, &c) in pixels[row * 2 + col].iter().enumerate() {
                    padded[row * stride_f32 + col * 3 + k] = c;
                }
            }
        }
        let bytes: &[u8] = bytemuck::cast_slice(&padded);
        let slice = PixelSlice::new(bytes, 2, 2, stride_f32 * 4, PixelDescriptor::RGBF32_LINEAR)
            .unwrap();
        let got = compute_content_light_level(slice, REFERENCE_DIFFUSE_WHITE_NITS).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn cll_rejects_u8_and_nonlinear() {
        let u8buf = PixelBuffer::from_vec(vec![0u8; 3], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        assert!(compute_content_light_level(u8buf.as_slice(), 203.0).is_err());

        let nonlinear = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
        let mut data = Vec::new();
        for c in [0.5f32; 3] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, nonlinear).unwrap();
        assert!(compute_content_light_level(buf.as_slice(), 203.0).is_err());
    }

    #[test]
    fn pq16_golden_reference_white_and_peak() {
        // 1.0 @ 203 nits → PQ(0.0203); 10000/203 → PQ(1.0) = code 65535.
        let peak = 10_000.0 / REFERENCE_DIFFUSE_WHITE_NITS;
        let buf = rgbf32_buffer(&[[1.0; 3], [peak; 3]], 2, 1);
        let (out, cll) = encode_pq16(buf.as_slice(), REFERENCE_DIFFUSE_WHITE_NITS).unwrap();

        assert_eq!(out.descriptor(), PixelDescriptor::RGB16_BT2100_PQ);
        let bytes = out.as_slice().as_strided_bytes();
        let code = |i: usize| u16::from_ne_bytes([bytes[2 * i], bytes[2 * i + 1]]);

        let want_white = (pq_oracle(203.0 / 10_000.0) * 65535.0).round() as i64;
        let got_white = i64::from(code(0));
        assert!(
            (got_white - want_white).abs() <= 1,
            "PQ code for 203-nit white: got {got_white}, oracle {want_white}"
        );
        assert_eq!(code(3), 65535, "10000-nit peak must clip to full code");

        assert_eq!(cll.max_content_light_level, 10_000);
        // Average of (203, 10000) = 5101.5 → 5102.
        assert_eq!(cll.max_frame_average_light_level, 5102);
    }

    #[test]
    fn pq16_matches_oracle_across_decades() {
        let values = [0.001f32, 0.01, 0.1, 0.5, 1.0, 2.0, 8.0, 20.0, 49.0];
        let pixels: Vec<[f32; 3]> = values.iter().map(|&v| [v; 3]).collect();
        let buf = rgbf32_buffer(&pixels, values.len() as u32, 1);
        let (out, _) = encode_pq16(buf.as_slice(), REFERENCE_DIFFUSE_WHITE_NITS).unwrap();
        let bytes = out.as_slice().as_strided_bytes();
        for (i, &v) in values.iter().enumerate() {
            let got = i64::from(u16::from_ne_bytes([bytes[6 * i], bytes[6 * i + 1]]));
            let x = f64::from(v) * f64::from(REFERENCE_DIFFUSE_WHITE_NITS) / 10_000.0;
            let want = (pq_oracle(x) * 65535.0).round() as i64;
            assert!(
                (got - want).abs() <= 1,
                "PQ16 at {v}: got {got}, oracle {want}"
            );
        }
    }

    #[test]
    fn pq16_cll_agrees_with_compute() {
        let pixels = [[0.25f32, 0.5, 1.0], [2.0, 0.1, 0.3], [0.0; 3], [1.5; 3]];
        let buf = rgbf32_buffer(&pixels, 2, 2);
        let direct =
            compute_content_light_level(buf.as_slice(), REFERENCE_DIFFUSE_WHITE_NITS).unwrap();
        let (_, inline) = encode_pq16(buf.as_slice(), REFERENCE_DIFFUSE_WHITE_NITS).unwrap();
        assert_eq!(direct, inline);
    }

}
