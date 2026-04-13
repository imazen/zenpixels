//! Lightweight CMS backend using fused SIMD gamut kernels.
//!
//! [`ZenCmsLite`] implements [`ColorManagement`] for named-profile conversions
//! (sRGB, Display P3, BT.2020, Adobe RGB, DCI-P3) without ICC profile parsing.
//! It delegates to [`fast_gamut`](crate::fast_gamut) kernels for the actual
//! conversion work — fused TRC + matrix + TRC in a single SIMD pass.
//!
//! # When to use
//!
//! Use `ZenCmsLite` when:
//! - All source/destination color spaces are known named profiles or CICP codes
//! - No custom ICC profiles are involved
//! - You want maximum conversion speed (1.5–2x faster than moxcms for f32)
//!
//! Use `MoxCms` (or another full CMS) when:
//! - Source or destination is a custom ICC profile
//! - Profile identification from ICC bytes is needed
//!
//! # Supported conversions
//!
//! Any combination of `ColorProfileSource::PrimariesTransferPair`,
//! `ColorProfileSource::Named`, or `ColorProfileSource::Cicp` where both
//! primaries sets have known chromaticities. ICC profile sources are not
//! supported — `build_transform` returns an error, `identify_profile`
//! returns `None`.
//!
//! # Example
//!
//! ```rust,ignore
//! use zenpixels_convert::cms_lite::ZenCmsLite;
//! use zenpixels_convert::output::{finalize_for_output, OutputProfile};
//! use zenpixels_convert::Cicp;
//!
//! let ready = finalize_for_output(
//!     &buffer, &origin,
//!     OutputProfile::Named(Cicp::SRGB),
//!     PixelFormat::Rgb8,
//!     &ZenCmsLite,
//! )?;
//! ```

use alloc::boxed::Box;
use alloc::format;

use crate::cms::{ColorManagement, RowTransform};
use crate::fast_gamut;
use crate::icc;
use crate::{ChannelType, Cicp, ColorPrimaries, PixelFormat, TransferFunction};

/// Lightweight CMS using fused SIMD gamut conversion kernels.
///
/// Handles conversions between any color spaces that can be described by
/// a (primaries, transfer) pair. This includes:
///
/// - **ICC profiles**: identified via 132-profile hash table (~100ns) and
///   CICP-in-ICC extraction. Covers sRGB, Display P3, BT.2020, Adobe RGB,
///   ProPhoto, and their variants across ICC v2–v5.
/// - **CICP codes**: mapped directly to primaries + transfer.
/// - **Named profiles**: decomposed to primaries + transfer.
/// - **Explicit primaries + transfer pairs**.
///
/// Custom/unknown ICC profiles that don't match any known hash return `None`,
/// signaling the caller to fall back to a full CMS (e.g., moxcms).
///
/// # Extended range
///
/// By default, f32 conversions clamp to \[0, 1\] and use fused SIMD kernels
/// (~4 GiB/s). Set `extended: true` to preserve out-of-gamut values
/// (negatives, >1.0) for HDR or cross-gamut workflows that need to defer
/// tone mapping or gamut mapping to a later stage. The extended path is
/// scalar (~200 MiB/s) because sign-preserving TRC requires per-channel
/// branching.
///
/// ```rust,ignore
/// // Fast (default): clamp to [0,1], fused SIMD
/// let cms = ZenCmsLite::default();
///
/// // Extended: preserve out-of-gamut, scalar powf
/// let cms = ZenCmsLite::extended();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ZenCmsLite {
    /// Preserve out-of-gamut values (negatives, >1.0) in f32 conversions.
    /// Default: `false` (clamp to \[0,1\], fused SIMD).
    pub extended: bool,
}

impl Default for ZenCmsLite {
    fn default() -> Self {
        Self { extended: false }
    }
}

impl ZenCmsLite {
    /// Create a `ZenCmsLite` with extended range enabled.
    ///
    /// Preserves out-of-gamut f32 values (negatives, >1.0) for HDR/cross-gamut
    /// workflows. Slower than the default clamped SIMD path.
    pub const fn extended() -> Self {
        Self { extended: true }
    }
}

/// Error from the lightweight CMS.
#[derive(Debug, Clone)]
pub struct ZenCmsLiteError(pub alloc::string::String);

impl core::fmt::Display for ZenCmsLiteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Extract (primaries, transfer) from a `ColorProfileSource`.
///
/// For ICC sources, tries hash-based identification (~100ns) and CICP-in-ICC
/// extraction. Returns `None` only for truly unknown custom profiles.
fn resolve_primaries_transfer(
    src: &crate::ColorProfileSource<'_>,
) -> Option<(ColorPrimaries, TransferFunction)> {
    match src {
        crate::ColorProfileSource::Icc(icc_bytes) => {
            // Try hash-based identification first (covers 132 known profiles).
            if let Some(id) = icc::identify_common(icc_bytes, icc::Tolerance::Intent) {
                return Some((id.primaries, id.transfer));
            }
            // Try CICP-in-ICC tag (ICC v4.4+).
            if let Some(cicp) = icc::extract_cicp(icc_bytes) {
                let p = ColorPrimaries::from_cicp(cicp.color_primaries)?;
                let t = TransferFunction::from_cicp(cicp.transfer_characteristics)?;
                return Some((p, t));
            }
            None
        }
        other => other.primaries_transfer(),
    }
}

impl ColorManagement for ZenCmsLite {
    type Error = ZenCmsLiteError;

    fn build_transform(
        &self,
        src_icc: &[u8],
        dst_icc: &[u8],
    ) -> Result<Box<dyn RowTransform>, Self::Error> {
        self.build_transform_for_format(src_icc, dst_icc, PixelFormat::Rgb8, PixelFormat::Rgb8)
    }

    fn build_transform_for_format(
        &self,
        src_icc: &[u8],
        dst_icc: &[u8],
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> Result<Box<dyn RowTransform>, Self::Error> {
        let src = crate::ColorProfileSource::Icc(src_icc);
        let dst = crate::ColorProfileSource::Icc(dst_icc);
        match self.build_source_transform(src, dst, src_format, dst_format) {
            Some(result) => result,
            None => Err(ZenCmsLiteError(
                "unrecognized ICC profile (not in known-profile table)".into(),
            )),
        }
    }

    fn identify_profile(&self, icc_bytes: &[u8]) -> Option<Cicp> {
        // Hash-based identification → CICP.
        if let Some(id) = icc::identify_common(icc_bytes, icc::Tolerance::Intent) {
            return id.to_cicp();
        }
        // CICP-in-ICC tag (ICC v4.4+).
        icc::extract_cicp(icc_bytes)
    }

    fn build_source_transform(
        &self,
        src: crate::ColorProfileSource<'_>,
        dst: crate::ColorProfileSource<'_>,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> Option<Result<Box<dyn RowTransform>, Self::Error>> {
        let (src_p, src_t) = resolve_primaries_transfer(&src)?;
        let (dst_p, dst_t) = resolve_primaries_transfer(&dst)?;

        // Same color space — no conversion needed.
        if src_p as u8 == dst_p as u8 && src_t as u8 == dst_t as u8 {
            return None;
        }

        // Compute the gamut matrix. If primaries are the same but TRC differs,
        // the matrix is identity — but we still need TRC conversion.
        let matrix = if src_p as u8 == dst_p as u8 {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        } else {
            match src_p.gamut_matrix_to(dst_p) {
                Some(m) => m,
                None => {
                    return Some(Err(ZenCmsLiteError(format!(
                        "no gamut matrix for {src_p:?} → {dst_p:?}"
                    ))));
                }
            }
        };

        // Verify TRC functions are supported.
        if fast_gamut::scalar_linearize(src_t).is_none() {
            return Some(Err(ZenCmsLiteError(format!(
                "unsupported source transfer function: {src_t:?}"
            ))));
        }
        if fast_gamut::scalar_encode(dst_t).is_none() {
            return Some(Err(ZenCmsLiteError(format!(
                "unsupported destination transfer function: {dst_t:?}"
            ))));
        }

        // Determine pixel layout from the format we'll actually process.
        // Use src_format to determine channel type and alpha.
        let channel_type = src_format.channel_type();
        let has_alpha = src_format.has_alpha_bytes();

        // Pre-build linearization LUT for u8 path (1KB, one-time cost).
        let linearize_fn = fast_gamut::scalar_linearize(src_t).unwrap();
        let linearize_lut = if matches!(channel_type, ChannelType::U8) {
            Some(fast_gamut::build_linearize_lut(linearize_fn))
        } else {
            None
        };

        // Extended range (sign-preserving, no clamping) is opt-in via
        // ZenCmsLite::extended(). Only applies to f32 — u8/u16 always clamp.
        let extended = self.extended && matches!(channel_type, ChannelType::F32);

        Some(Ok(Box::new(LiteTransform {
            matrix,
            src_trc: src_t,
            dst_trc: dst_t,
            linearize: linearize_fn,
            encode: fast_gamut::scalar_encode(dst_t).unwrap(),
            encode_u8: fast_gamut::scalar_encode_u8(dst_t).unwrap(),
            linearize_lut,
            has_alpha,
            channel_type,
            extended,
            _dst_format: dst_format,
        })))
    }
}

struct LiteTransform {
    matrix: [[f32; 3]; 3],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
    linearize: fn(f32) -> f32,
    encode: fn(f32) -> f32,
    /// f32→u8 encode. Uses LUT for sRGB, polynomial+quantize for others.
    encode_u8: fn(f32) -> u8,
    /// Pre-built u8→f32 linearization LUT. Only allocated for u8 formats.
    linearize_lut: Option<alloc::boxed::Box<[f32; 256]>>,
    has_alpha: bool,
    channel_type: ChannelType,
    /// Use extended range (sign-preserving, no clamping) for f32 path.
    /// Required for HDR → SDR gamut conversion where out-of-gamut values
    /// must survive until tone mapping.
    extended: bool,
    _dst_format: PixelFormat,
}

// LiteTransform contains only Copy types and fn pointers, which are all Send.
// Rust auto-derives Send for this, but static_assertions would catch regressions.

impl RowTransform for LiteTransform {
    fn transform_row(&self, src: &[u8], dst: &mut [u8], width: u32) {
        match self.channel_type {
            ChannelType::U8 => self.transform_u8(src, dst, width),
            ChannelType::U16 => self.transform_u16(src, dst, width),
            ChannelType::F32 | ChannelType::F16 | _ => self.transform_f32(src, dst),
        }
    }
}

impl LiteTransform {
    fn transform_u8(&self, src: &[u8], dst: &mut [u8], _width: u32) {
        if let Some(lut) = &self.linearize_lut {
            // Fast path: LUT linearize + LUT/fast encode (no polynomial on hot path)
            if self.has_alpha {
                fast_gamut::convert_u8_rgba_simd_lut(&self.matrix, src, dst, lut, self.encode_u8);
            } else {
                // RGB: SIMD-batched LUT→matrix→LUT (8 pixels at a time)
                fast_gamut::convert_u8_rgb_simd_lut(&self.matrix, src, dst, lut, self.encode_u8);
            }
        } else {
            // Fallback: per-channel function calls
            if self.has_alpha {
                fast_gamut::convert_u8_rgba(&self.matrix, src, dst, self.linearize, self.encode);
            } else {
                fast_gamut::convert_u8_rgb(&self.matrix, src, dst, self.linearize, self.encode);
            }
        }
    }

    fn transform_u16(&self, src: &[u8], dst: &mut [u8], _width: u32) {
        let src_u16: &[u16] = bytemuck::cast_slice(src);
        let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(dst);
        if self.has_alpha {
            convert_u16_rgba(&self.matrix, src_u16, dst_u16, self.linearize, self.encode);
        } else {
            fast_gamut::convert_u16_rgb(
                &self.matrix,
                src_u16,
                dst_u16,
                self.linearize,
                self.encode,
            );
        }
    }

    fn transform_f32(&self, src: &[u8], dst: &mut [u8]) {
        // Copy src → dst, then transform in place.
        dst.copy_from_slice(src);
        let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
        if self.extended {
            // Extended range: sign-preserving, no clamping (for HDR/out-of-gamut).
            if self.has_alpha {
                fast_gamut::convert_f32_rgba_extended(
                    &self.matrix,
                    dst_f32,
                    self.src_trc,
                    self.dst_trc,
                );
            } else {
                fast_gamut::convert_f32_rgb_extended(
                    &self.matrix,
                    dst_f32,
                    self.src_trc,
                    self.dst_trc,
                );
            }
        } else if self.has_alpha {
            fast_gamut::convert_f32_rgba_dispatch(
                &self.matrix,
                dst_f32,
                self.src_trc,
                self.dst_trc,
            );
        } else {
            fast_gamut::convert_f32_rgb_dispatch(&self.matrix, dst_f32, self.src_trc, self.dst_trc);
        }
    }
}

/// Convert u16 RGBA source to u16 RGBA dest via gamut conversion. Alpha copied.
fn convert_u16_rgba(
    m: &[[f32; 3]; 3],
    src: &[u16],
    dst: &mut [u16],
    linearize_fn: fn(f32) -> f32,
    encode_fn: fn(f32) -> f32,
) {
    debug_assert_eq!(src.len() % 4, 0);
    debug_assert_eq!(src.len(), dst.len());
    for (src_px, dst_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let r = linearize_fn(src_px[0] as f32 / 65535.0);
        let g = linearize_fn(src_px[1] as f32 / 65535.0);
        let b = linearize_fn(src_px[2] as f32 / 65535.0);
        let (nr, ng, nb) = fast_gamut::mat3x3_scalar(m, r, g, b);
        dst_px[0] = (encode_fn(nr) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        dst_px[1] = (encode_fn(ng) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        dst_px[2] = (encode_fn(nb) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        dst_px[3] = src_px[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cms::ColorManagement;
    use crate::{ColorProfileSource, NamedProfile, PixelFormat};

    #[test]
    fn build_source_transform_p3_to_srgb_f32() {
        let cms = ZenCmsLite::default();
        let src = ColorProfileSource::Named(NamedProfile::DisplayP3);
        let dst = ColorProfileSource::Named(NamedProfile::Srgb);
        let result = cms.build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32);
        assert!(result.is_some(), "should support P3→sRGB");
        let transform = result.unwrap().expect("should not error");

        // White should map to white
        let src_px: [f32; 3] = [1.0, 1.0, 1.0];
        let mut dst_px = [0.0f32; 3];
        transform.transform_row(
            bytemuck::cast_slice(&src_px),
            bytemuck::cast_slice_mut(&mut dst_px),
            1,
        );
        for ch in 0..3 {
            assert!(
                (dst_px[ch] - 1.0).abs() < 1e-4,
                "white ch{ch}: {}",
                dst_px[ch]
            );
        }
    }

    #[test]
    fn build_source_transform_p3_to_srgb_u8() {
        let cms = ZenCmsLite::default();
        let src = ColorProfileSource::Named(NamedProfile::DisplayP3);
        let dst = ColorProfileSource::Named(NamedProfile::Srgb);
        let result = cms.build_source_transform(src, dst, PixelFormat::Rgb8, PixelFormat::Rgb8);
        assert!(result.is_some());
        let transform = result.unwrap().expect("should not error");

        // White (255,255,255) should map to (255,255,255)
        let src_px = [255u8, 255, 255];
        let mut dst_px = [0u8; 3];
        transform.transform_row(&src_px, &mut dst_px, 1);
        assert_eq!(dst_px, [255, 255, 255]);

        // Black should map to black
        let src_px = [0u8, 0, 0];
        let mut dst_px = [0u8; 3];
        transform.transform_row(&src_px, &mut dst_px, 1);
        assert_eq!(dst_px, [0, 0, 0]);
    }

    #[test]
    fn same_color_space_returns_none() {
        let cms = ZenCmsLite::default();
        let src = ColorProfileSource::Named(NamedProfile::Srgb);
        let dst = ColorProfileSource::Named(NamedProfile::Srgb);
        assert!(
            cms.build_source_transform(src, dst, PixelFormat::Rgb8, PixelFormat::Rgb8)
                .is_none(),
            "same color space should return None (no transform needed)"
        );
    }

    #[test]
    fn icc_profile_not_supported() {
        let cms = ZenCmsLite::default();
        assert!(cms.build_transform(&[0; 100], &[0; 100]).is_err());
        assert!(cms.identify_profile(&[0; 100]).is_none());
    }

    #[test]
    fn cicp_source_supported() {
        let cms = ZenCmsLite::default();
        let src = ColorProfileSource::Cicp(Cicp::DISPLAY_P3);
        let dst = ColorProfileSource::Cicp(Cicp::SRGB);
        let result = cms.build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32);
        assert!(result.is_some(), "CICP source should be supported");
        assert!(result.unwrap().is_ok());
    }

    #[test]
    fn primaries_transfer_pair_supported() {
        let cms = ZenCmsLite::default();
        let src = ColorProfileSource::PrimariesTransferPair {
            primaries: ColorPrimaries::DisplayP3,
            transfer: TransferFunction::Srgb,
        };
        let dst = ColorProfileSource::PrimariesTransferPair {
            primaries: ColorPrimaries::Bt709,
            transfer: TransferFunction::Srgb,
        };
        let result = cms.build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32);
        assert!(result.is_some());
        assert!(result.unwrap().is_ok());
    }

    #[test]
    fn cross_trc_conversion() {
        let cms = ZenCmsLite::default();
        // BT.2020 PQ → sRGB: different primaries AND different TRC
        let src = ColorProfileSource::Named(NamedProfile::Bt2020Pq);
        let dst = ColorProfileSource::Named(NamedProfile::Srgb);
        let result = cms.build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32);
        assert!(result.is_some());
        let transform = result.unwrap().expect("should not error");

        // Black should still map to black
        let src_px = [0.0f32, 0.0, 0.0];
        let mut dst_px = [0.0f32; 3];
        transform.transform_row(
            bytemuck::cast_slice(&src_px),
            bytemuck::cast_slice_mut(&mut dst_px),
            1,
        );
        for ch in 0..3 {
            assert!(dst_px[ch].abs() < 1e-5, "black ch{ch}: {}", dst_px[ch]);
        }
    }

    // --- ICC profile identification ---

    #[test]
    fn identify_profile_p3_icc() {
        let cms = ZenCmsLite::default();
        let p3_icc = crate::icc_profiles::DISPLAY_P3_V4;
        let cicp = cms.identify_profile(p3_icc);
        assert!(cicp.is_some(), "should identify Display P3 ICC profile");
        let cicp = cicp.unwrap();
        assert_eq!(cicp.color_primaries, 12, "should be Display P3 (cp=12)");
        assert_eq!(
            cicp.transfer_characteristics, 13,
            "should be sRGB TRC (tc=13)"
        );
    }

    #[test]
    fn identify_profile_unknown_returns_none() {
        let cms = ZenCmsLite::default();
        assert!(cms.identify_profile(&[0; 100]).is_none());
        assert!(cms.identify_profile(&[]).is_none());
    }

    #[test]
    fn build_transform_for_format_icc_to_icc() {
        let cms = ZenCmsLite::default();
        let p3_icc = crate::icc_profiles::DISPLAY_P3_V4;
        let bt2020_icc = crate::icc_profiles::REC2020_V4;

        // P3 ICC → BT.2020 ICC should work (both are recognized profiles)
        let result = cms.build_transform_for_format(
            p3_icc,
            bt2020_icc,
            PixelFormat::RgbF32,
            PixelFormat::RgbF32,
        );
        assert!(result.is_ok(), "should handle recognized ICC→ICC");

        let transform = result.unwrap();
        // White should map to white
        let src_px: [f32; 3] = [1.0, 1.0, 1.0];
        let mut dst_px = [0.0f32; 3];
        transform.transform_row(
            bytemuck::cast_slice(&src_px),
            bytemuck::cast_slice_mut(&mut dst_px),
            1,
        );
        for ch in 0..3 {
            assert!(
                (dst_px[ch] - 1.0).abs() < 1e-4,
                "ICC→ICC white ch{ch}: {}",
                dst_px[ch]
            );
        }
    }

    #[test]
    fn build_transform_unrecognized_icc_fails() {
        let cms = ZenCmsLite::default();
        let garbage = [0u8; 200];
        assert!(cms.build_transform(&garbage, &garbage).is_err());
    }

    #[test]
    fn build_source_transform_icc_src() {
        let cms = ZenCmsLite::default();
        let p3_icc = crate::icc_profiles::DISPLAY_P3_V4;
        let src = ColorProfileSource::Icc(p3_icc);
        let dst = ColorProfileSource::Named(NamedProfile::Srgb);
        let result = cms.build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32);
        assert!(result.is_some(), "ICC P3 src should be recognized");
        assert!(result.unwrap().is_ok());
    }

    // --- Clamped vs extended ---

    #[test]
    fn default_clamps_out_of_gamut() {
        // Default (clamped): P3 green → sRGB should clamp negatives to 0.
        let cms = ZenCmsLite::default();
        let src = ColorProfileSource::Named(NamedProfile::DisplayP3);
        let dst = ColorProfileSource::Named(NamedProfile::Srgb);
        let transform = cms
            .build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32)
            .unwrap()
            .unwrap();

        let src_px: [f32; 3] = [0.0, 1.0, 0.0];
        let mut dst_px = [0.0f32; 3];
        transform.transform_row(
            bytemuck::cast_slice(&src_px),
            bytemuck::cast_slice_mut(&mut dst_px),
            1,
        );
        // Clamped path: red should be 0 (clamped from negative), green ≤ 1.0
        assert!(
            dst_px[0] >= 0.0,
            "clamped path should not produce negatives: {}",
            dst_px[0]
        );
        assert!(
            dst_px[1] <= 1.0 + 1e-5,
            "clamped path should not produce >1: {}",
            dst_px[1]
        );
    }

    #[test]
    fn extended_range_preserves_negative_values() {
        // P3 pure green → sRGB produces negative red (out of sRGB gamut).
        // Extended range must preserve these negatives, not clamp to 0.
        let cms = ZenCmsLite::extended();
        let src = ColorProfileSource::Named(NamedProfile::DisplayP3);
        let dst = ColorProfileSource::Named(NamedProfile::Srgb);
        let result = cms.build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32);
        let transform = result.unwrap().unwrap();

        let src_px: [f32; 3] = [0.0, 1.0, 0.0]; // P3 pure green
        let mut dst_px = [0.0f32; 3];
        transform.transform_row(
            bytemuck::cast_slice(&src_px),
            bytemuck::cast_slice_mut(&mut dst_px),
            1,
        );
        // P3 green maps outside sRGB: red should be negative, green > 1.0
        assert!(
            dst_px[0] < 0.0,
            "P3 green should have negative sRGB red: {}",
            dst_px[0]
        );
        assert!(
            dst_px[1] > 1.0,
            "P3 green should have >1.0 sRGB green: {}",
            dst_px[1]
        );
    }

    #[test]
    fn extended_range_hdr_preserves_supernormal() {
        // BT.2020 PQ → sRGB: PQ signal 1.0 = 10000 nits, far above SDR range.
        let cms = ZenCmsLite::extended();
        let src = ColorProfileSource::Named(NamedProfile::Bt2020Pq);
        let dst = ColorProfileSource::Named(NamedProfile::Srgb);
        let result = cms.build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32);
        let transform = result.unwrap().unwrap();

        let src_px: [f32; 3] = [0.5, 0.5, 0.5]; // ~100 nits in PQ
        let mut dst_px = [0.0f32; 3];
        transform.transform_row(
            bytemuck::cast_slice(&src_px),
            bytemuck::cast_slice_mut(&mut dst_px),
            1,
        );
        // Should produce values (possibly >1.0 or <0.0) — not clamped to [0,1]
        // The exact values depend on tone mapping, but they should be finite.
        for ch in 0..3 {
            assert!(
                dst_px[ch].is_finite(),
                "HDR→SDR should produce finite values: ch{ch}={}",
                dst_px[ch]
            );
        }
    }
}
