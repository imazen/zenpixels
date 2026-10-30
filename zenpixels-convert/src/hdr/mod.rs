//! HDR processing utilities.
//!
//! Re-exports [`ContentLightLevel`] and [`MasteringDisplay`] from the
//! `zenpixels::hdr` module. Provides explicit measurement, tone mapping
//! and anchor-aware quantization.
//!
//! The core PQ/HLG EOTF/OETF math is always available through the main
//! conversion pipeline in [`ConvertPlan`](crate::ConvertPlan).
//!
//! # Experimental: `measure` — content-light-level measurement
//!
//! Behind the `hdr-experimental` Cargo feature, the `measure` submodule
//! exposes the `measure::CllMeasure` extension trait (with
//! `measure_max` / `measure_robust` / `measure_max_smoothed` /
//! `measure_percentile` / `measure_histogram` on
//! [`ContentLightLevel`]), the `measure::LightLevelHistogram`
//! primitive, and the `measure::LightLevelMethod` enum. (Plain code
//! spans, not intra-doc links — the targets only exist when the feature
//! is enabled, and this page renders either way.)
//! Measurement stays explicitly named and separate from the shared metadata
//! types. The extension trait is sealed; scans are opt-in via these methods.

/// BT.2020 NCL luma coefficients shared by the HDR submodules:
/// `Y = 0.2627·R + 0.6780·G + 0.0593·B`. Used by [`measure`]'s
/// `LuminanceBt2020` reduction and by the [`Bt2446A`] curve (both the
/// SIMD body and the scalar remainder tail of `bt2446a_tier`). Pinned at
/// the parent-module scope so the call sites stay in lock-step.
#[cfg(feature = "hdr-experimental")]
pub(super) const BT2020_LR: f32 = 0.2627;
#[cfg(feature = "hdr-experimental")]
pub(super) const BT2020_LG: f32 = 0.6780;
#[cfg(feature = "hdr-experimental")]
pub(super) const BT2020_LB: f32 = 0.0593;

/// Content-light-level measurement (experimental).
///
/// Gated behind the `hdr-experimental` Cargo feature. See the parent
/// module docs for stability notes.
#[cfg(feature = "hdr-experimental")]
pub mod measure;

/// BT.2446 Method A tone-mapper (experimental).
///
/// Linear-light HDR → linear-light SDR curve. Gated behind
/// `hdr-experimental` while the cross-crate API surface settles.
#[cfg(feature = "hdr-experimental")]
mod bt2446a;

/// Soft chroma compression in OKLch with a precomputed gamut boundary LUT.
/// Gated behind `hdr-experimental`.
#[cfg(feature = "hdr-experimental")]
mod gamut_compress;

/// Re-exports of the experimental [`measure`] surface at the [`hdr`](self)
/// boundary, so callers can write
/// `use zenpixels_convert::hdr::CllMeasure;` instead of the full
/// `zenpixels_convert::hdr::measure::CllMeasure` path. Same gating.
#[cfg(feature = "hdr-experimental")]
pub use measure::{CllMeasure, LightLevelHistogram, LightLevelMethod};

// HDR → SDR conversion now lives in the main `ConvertPlan` infrastructure:
// build via [`crate::ConvertPlan::new_with_hdr_peak`] /
// [`crate::ConvertPlan::new_with_hdr_config`] and run through the standard
// [`crate::RowConverter`] / [`crate::convert_buffer`] entry points. The
// underlying `Bt2446A` and `SoftCompress` primitives stay public for advanced
// callers that want to drive the math directly.
#[cfg(feature = "hdr-experimental")]
pub use bt2446a::Bt2446A;
#[cfg(feature = "hdr-experimental")]
pub use gamut_compress::{GamutBoundaryLut, SoftCompress};

use crate::adapt::{convert_buffer_with_anchor, convert_into_with_anchor};
use crate::error::ConvertError;
use crate::{PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice, TransferFunction};
use alloc::sync::Arc;
use whereat::At;
use zenpixels::{Cicp, ColorContext};

// Re-export metadata types from the core crate.
pub use zenpixels::hdr::{ContentLightLevel, MasteringDisplay};
// `quantize_to` reads the anchor from the source's `ColorContext`; the
// canonical public home for the type is `zenpixels::hdr::DiffuseWhite`
// (reachable through the core crate — not re-exported here).
use zenpixels::hdr::DiffuseWhite;

// HDR quantization (relative-linear f32 → a PQ HDR descriptor)
// ---------------------------------------------------------------------------

/// Shared `quantize_*` setup: read the anchor from the source `ColorContext`
/// (default [`DiffuseWhite::BT2408`] = 203), validate the source is linear
/// RGB(A) f32 and the target is PQ, and return the gamut-tagged source
/// descriptor + anchor + dimensions. The source descriptor carries the target's
/// primaries so no gamut step is planned (value-only quantize); the target's
/// channel count then drives whether alpha is dropped or preserved.
fn quantize_setup(
    px: &PixelSlice<'_>,
    target: PixelDescriptor,
) -> Result<(PixelDescriptor, DiffuseWhite, u32, u32), At<ConvertError>> {
    let diffuse_white = px
        .color_context()
        .and_then(|c| c.diffuse_white)
        .unwrap_or(DiffuseWhite::BT2408);
    let desc = px.descriptor();
    let src = match desc.pixel_format() {
        PixelFormat::RgbF32 => PixelDescriptor::RGBF32_LINEAR,
        PixelFormat::RgbaF32 => PixelDescriptor::RGBAF32_LINEAR,
        _ => return Err(whereat::at!(ConvertError::NoMatch { source: desc })),
    }
    .with_primaries(target.primaries);
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
    Ok((src, diffuse_white, w, h))
}

/// Quantize relative-linear RGB(A) f32 pixels to a **PQ** HDR target
/// descriptor (e.g. [`PixelDescriptor::RGB16_BT2100_PQ`]).
///
/// The absolute-luminance anchor — the nits that sample `1.0` represents — is
/// read from the source `ColorContext`'s `diffuse_white`, defaulting to
/// [`DiffuseWhite::BT2408`] (203, the cross-vendor relative-linear convention)
/// when unsignaled. Attach a custom anchor with
/// `ColorContext::with_diffuse_white` (e.g. a buffer reconstructed at a
/// different reference white). The anchor threads **into the PQ `ConvertStep`s
/// themselves**: the linear → PQ kernel scales the RGB lanes by `anchor / 10000`
/// across the relative-linear ↔ PQ-absolute boundary, so this is a thin wrapper
/// that hands the source — **strided and RGBA accepted as-is, no repack** —
/// straight to the pipeline. Negatives fold to 0 and the PQ peak clamps
/// in-kernel; codes match the f64 ST 2084 oracle within ±1.
///
/// **Alpha follows the target.** An RGB target (e.g.
/// [`RGB16_BT2100_PQ`](PixelDescriptor::RGB16_BT2100_PQ)) drops alpha; an RGBA
/// PQ target (`RGBA16.with_transfer(Pq).with_primaries(…)`) preserves it,
/// carried linearly and **never PQ-encoded or anchor-scaled**.
///
/// **Primaries are not converted** — the source gamut is signaled as the
/// target's (feed BT.2020-relative-linear for `RGB16_BT2100_PQ`). Measure CLL
/// separately with `hdr::measure::CllMeasure::measure_max`.
///
/// The successor to the withdrawn `encode_pq16` (rationale:
/// `docs/hdr-design-survey-2026-06-13.md`). With the anchor living on the plan
/// (#45 S2), the quantizer is now a straight pass to
/// `convert_buffer_with_anchor`; HLG is still excluded (its scene-referred
/// anchor differs).
///
/// # Errors
///
/// - [`ConvertError::NoMatch`] if `px` is not `RgbF32`/`RgbaF32`;
///   [`ConvertError::UnsupportedTransfer`] if it is not `Linear`.
/// - [`ConvertError::NoPath`] if `target`'s transfer is not PQ (HLG's
///   scene-referred anchor differs and is not handled here).
/// - [`ConvertError::InvalidWidth`] for zero-area input, or any error the
///   inner anchored conversion raises.
pub fn quantize_to(
    px: PixelSlice<'_>,
    target: PixelDescriptor,
) -> Result<PixelBuffer, At<ConvertError>> {
    // Hand the (possibly strided, possibly RGBA) source straight to the anchored
    // pipeline — no caller-side pre-scale or repack. The PQ kernel applies
    // `white / 10000` to the RGB lanes, folds negatives to 0, and the plan
    // drops or preserves alpha per `target`. The anchor travels with the pixels
    // (S1a): `quantize_setup` reads it from the source `ColorContext`.
    let (src, diffuse_white, w, h) = quantize_setup(&px, target)?;
    let out = convert_buffer_with_anchor(
        px.as_strided_bytes(),
        w,
        h,
        px.stride(),
        src,
        target,
        diffuse_white,
    )?;
    // Carry the envelope forward. `diffuse_white` is a *reference* — it survives
    // the encode (it's the SDR-white nits a downstream encoder signals as
    // `ndwt`), so the output self-describes it rather than silently dropping the
    // anchor we just applied. The target's CICP (transfer/primaries/range) rides
    // along so the buffer is fully described for re-encode.
    let context = match Cicp::from_descriptor(&target) {
        Some(cicp) => ColorContext::from_cicp(cicp),
        None => ColorContext::default(),
    }
    .with_diffuse_white(diffuse_white);
    Ok(out.with_color_context(Arc::new(context)))
}

/// [`quantize_to`] writing into a caller-provided `dst` — no output allocation.
///
/// The result is written at `dst_stride` bytes per row (pass
/// `width * target.bytes_per_pixel()` for packed, or a larger stride to write
/// into a sub-region of a bigger buffer); `dst` must hold
/// `(rows - 1) * dst_stride + width * target.bpp` bytes, else
/// [`ConvertError::BufferSize`]. Anchor sourcing, strided-**source** handling,
/// and target-driven alpha (drop for an RGB target, preserve for an RGBA one)
/// are identical to [`quantize_to`]; this only avoids allocating the output. Unlike
/// [`quantize_to`], it writes raw bytes with no `PixelBuffer` to tag, so the
/// caller owns the output's color envelope (e.g. re-attaching the
/// `diffuse_white` anchor for a downstream encode).
///
/// Kept `pub(crate)` for now: the no-allocation capability is built and tested,
/// but per the "no speculative `pub`" rule (and the §3.2 design doc, which routes
/// the public HDR-convert surface through a future `PixelBuffer`-level entry and
/// keeps the byte-level convert internal) it is not yet a public commitment.
/// Promote the instant a concrete external consumer or §3.2 lands.
///
/// # Errors
///
/// The same validation errors as [`quantize_to`], plus
/// [`ConvertError::BufferSize`] when `dst` is too small.
// Exercised by the `quantize_into_*` unit tests; no non-test in-crate caller yet
// (it is a staged, ready-to-promote public candidate — see above).
#[allow(dead_code)]
pub(crate) fn quantize_into(
    px: PixelSlice<'_>,
    target: PixelDescriptor,
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<(), At<ConvertError>> {
    let (src, diffuse_white, w, h) = quantize_setup(&px, target)?;
    convert_into_with_anchor(
        px.as_strided_bytes(),
        w,
        h,
        px.stride(),
        src,
        target,
        diffuse_white,
        dst,
        dst_stride,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- quantize_to (PQ16) parity with the f64 ST 2084 oracle --

    use alloc::vec;
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

    fn rgbaf32(pixels: &[[f32; 4]], w: u32, h: u32) -> PixelBuffer {
        let mut data = Vec::with_capacity(pixels.len() * 16);
        for p in pixels {
            for c in p {
                data.extend_from_slice(&c.to_ne_bytes());
            }
        }
        PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBAF32_LINEAR).unwrap()
    }

    /// RGBA16 PQ target (BT.2020), matching `RGB16_BT2100_PQ` plus an alpha lane.
    fn rgba16_pq() -> PixelDescriptor {
        PixelDescriptor::RGBA16
            .with_transfer(TransferFunction::Pq)
            .with_primaries(PixelDescriptor::RGB16_BT2100_PQ.primaries)
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

    #[test]
    fn quantize_to_preserves_alpha_for_rgba_target() {
        // RGBA f32 linear → RGBA16 PQ: RGB take the anchored PQ OETF; alpha rides
        // through linearly (never PQ-encoded). PQ-encoding 0.5 would give a code
        // far from the linear 32768, so the assertion is a real discriminator.
        let target = rgba16_pq();
        let buf = rgbaf32(&[[1.0, 1.0, 1.0, 0.5], [2.0, 2.0, 2.0, 0.25]], 2, 1);
        let out = quantize_to(buf.as_slice(), target).unwrap();
        assert_eq!(out.descriptor(), target);
        let bytes = out.as_slice().as_strided_bytes();
        let code = |i: usize| u16::from_ne_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        for (px, g, a) in [(0usize, 1.0f64, 0.5f64), (1, 2.0, 0.25)] {
            let r = i64::from(code(px * 4));
            let want_rgb = (pq_oracle(g * 203.0 / 10_000.0) * 65535.0).round() as i64;
            assert!(
                (r - want_rgb).abs() <= 1,
                "rgb @203: got {r} want {want_rgb}"
            );
            let alpha = code(px * 4 + 3);
            let want_a = (a * 65535.0).round() as u16;
            assert_eq!(
                alpha, want_a,
                "alpha linear passthrough: got {alpha} want {want_a}"
            );
        }
    }

    #[test]
    fn quantize_to_honors_strided_input() {
        // A padded source stride (one sentinel pixel per row) must quantize
        // identically to the equivalent packed buffer.
        let target = PixelDescriptor::RGB16_BT2100_PQ;
        let stride = 2 * 12 + 12; // two RGB f32 pixels + one padding pixel
        let mut data = vec![0u8; stride * 2];
        for y in 0..2usize {
            let mut off = y * stride;
            for c in [0.1f32, 0.1, 0.1, 1.0, 1.0, 1.0] {
                data[off..off + 4].copy_from_slice(&c.to_ne_bytes());
                off += 4;
            }
            data[off..off + 4].copy_from_slice(&999.0f32.to_ne_bytes()); // sentinel
        }
        let strided = PixelSlice::new(&data, 2, 2, stride, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let got = quantize_to(strided, target).unwrap();

        let packed = rgbf32(&[[0.1; 3], [1.0; 3], [0.1; 3], [1.0; 3]], 2, 2);
        let want = quantize_to(packed.as_slice(), target).unwrap();
        assert_eq!(
            got.as_slice().as_strided_bytes(),
            want.as_slice().as_strided_bytes(),
            "strided input must quantize identically to packed"
        );
    }

    #[test]
    fn quantize_into_matches_quantize_to() {
        let target = PixelDescriptor::RGB16_BT2100_PQ;
        let buf = rgbf32(&[[0.1; 3], [1.0; 3], [2.0; 3]], 3, 1);
        let want = quantize_to(buf.as_slice(), target).unwrap();
        let row = 3 * target.bytes_per_pixel();
        let mut dst = vec![0u8; row];
        quantize_into(buf.as_slice(), target, &mut dst, row).unwrap();
        assert_eq!(dst, want.as_slice().as_strided_bytes());
    }

    #[test]
    fn quantize_into_honors_dst_stride() {
        // Write two PQ16 rows into a padded destination; the padding bytes must
        // be untouched and the row content must match a packed quantize.
        let target = PixelDescriptor::RGB16_BT2100_PQ;
        let buf = rgbf32(&[[0.1; 3], [1.0; 3], [0.1; 3], [1.0; 3]], 2, 2);
        let want = quantize_to(buf.as_slice(), target).unwrap();
        let want_bytes = want.as_slice().as_strided_bytes();
        let row = 2 * target.bytes_per_pixel(); // packed row width
        let dst_stride = row + 8; // 8 bytes of padding per row
        let mut dst = vec![0xAAu8; dst_stride * 2];
        quantize_into(buf.as_slice(), target, &mut dst, dst_stride).unwrap();
        for y in 0..2 {
            assert_eq!(
                &dst[y * dst_stride..y * dst_stride + row],
                &want_bytes[y * row..(y + 1) * row]
            );
            assert!(
                dst[y * dst_stride + row..y * dst_stride + dst_stride]
                    .iter()
                    .all(|&b| b == 0xAA),
                "padding row {y} must be untouched"
            );
        }
    }

    #[test]
    fn quantize_into_rejects_undersized_dst() {
        let target = PixelDescriptor::RGB16_BT2100_PQ;
        let buf = rgbf32(&[[1.0; 3]], 1, 1);
        let mut dst = vec![0u8; 2]; // one RGB16 pixel needs 6 bytes
        let row = target.bytes_per_pixel();
        let err = quantize_into(buf.as_slice(), target, &mut dst, row).unwrap_err();
        assert!(matches!(*err.error(), ConvertError::BufferSize { .. }));
    }

    #[test]
    fn quantize_to_carries_diffuse_white_anchor_onto_output() {
        use alloc::sync::Arc;
        use zenpixels::{Cicp, ColorContext};
        let target = PixelDescriptor::RGB16_BT2100_PQ;

        // A signaled 100-nit anchor must ride out on the output's ColorContext —
        // the encode applied it, so the buffer self-describes it (the `ndwt`
        // signal a downstream encoder needs), rather than silently dropping it.
        let buf = rgbf32(&[[1.0; 3]], 1, 1).with_color_context(Arc::new(
            ColorContext::from_cicp(Cicp::BT2100_PQ).with_diffuse_white(DiffuseWhite::new(100.0)),
        ));
        let out = quantize_to(buf.as_slice(), target).unwrap();
        let ctx = out.color_context().expect("output carries a ColorContext");
        assert_eq!(ctx.diffuse_white, Some(DiffuseWhite::new(100.0)));
        assert!(ctx.cicp.is_some(), "target CICP rides along for re-encode");

        // An unsignaled source still yields a self-describing output at the 203 default.
        let plain = rgbf32(&[[1.0; 3]], 1, 1);
        let out = quantize_to(plain.as_slice(), target).unwrap();
        assert_eq!(
            out.color_context().unwrap().diffuse_white,
            Some(DiffuseWhite::BT2408)
        );
    }
}
