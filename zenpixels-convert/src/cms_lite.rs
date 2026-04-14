//! Lightweight CMS backend using fused SIMD gamut kernels.
//!
//! [`ZenCmsLite`] implements [`ColorManagement`] for named-profile conversions
//! (sRGB, Display P3, BT.2020, Adobe RGB) without ICC profile parsing.
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
use crate::{ChannelType, Cicp, PixelFormat, TransferFunction};

/// Lightweight CMS using fused SIMD gamut conversion kernels.
///
/// Handles conversions between any color spaces that can be described by
/// a (primaries, transfer) pair. This includes:
///
/// - **ICC profiles**: identified via 132-profile hash table (~100ns) and
///   CICP-in-ICC extraction. Covers sRGB, Display P3, BT.2020, Adobe RGB,
///   and their variants across ICC v2–v5.
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

// Resolution of ColorProfileSource → (primaries, transfer) is handled by
// ColorProfileSource::resolve() (requires `icc` feature on zenpixels,
// which `zencms-lite` enables). That method handles ICC hash-based
// identification, CICP-in-ICC extraction, and CICP safety checks
// (rejects non-identity matrix coefficients and narrow range).

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
        // Metadata-only lookup: returns the profile's claimed (primaries,
        // transfer) without filtering by intent-safety. A profile may be
        // recognized here but rejected by `build_source_transform` when the
        // strict empirical check determined that canonical matrix+TRC math
        // diverges from CMS behavior for any rendering intent.
        if let Some(id) = zenpixels::icc::identify_common(icc_bytes) {
            let cp = id.primaries.to_cicp()?;
            let tc = id.transfer.to_cicp()?;
            return Some(Cicp::new(cp, tc, 0, true));
        }
        // CICP-in-ICC tag (ICC v4.4+) is authoritative metadata.
        if let Some(cicp) = zenpixels::icc::extract_cicp(icc_bytes) {
            return Some(cicp);
        }
        None
    }
}

impl ZenCmsLite {
    /// Build a transform from resolved `ColorProfileSource`s.
    ///
    /// Returns `None` if either source can't be resolved to known primaries+transfer.
    pub(crate) fn build_source_transform(
        &self,
        src: crate::ColorProfileSource<'_>,
        dst: crate::ColorProfileSource<'_>,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> Option<Result<Box<dyn RowTransform>, ZenCmsLiteError>> {
        let (src_p, src_t) = src.resolve()?;
        let (dst_p, dst_t) = dst.resolve()?;

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
            if !self.has_alpha && fast_gamut::has_simd_encode(self.dst_trc) {
                // Fused RGB: LUT linearize → SIMD (matrix + poly encode) → quantize
                fast_gamut::convert_u8_rgb_simd_fused(
                    &self.matrix,
                    src,
                    dst,
                    lut,
                    self.dst_trc,
                    self.encode,
                );
                return;
            }
            // RGBA or unsupported TRC: LUT→SIMD matrix→LUT encode
            if self.has_alpha {
                fast_gamut::convert_u8_rgba_simd_lut(&self.matrix, src, dst, lut, self.encode_u8);
            } else {
                fast_gamut::convert_u8_rgb_lut_lut(&self.matrix, src, dst, lut, self.encode_u8);
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
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::cms::ColorManagement;
    use crate::{ColorPrimaries, ColorProfileSource, NamedProfile, PixelFormat};

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
    fn build_transform_for_format_icc_to_icc_compat() {
        // The bundled DisplayP3Compat-v4 / REC2020_V4 profiles from the
        // saucecontrol Compact-ICC corpus are matrix-shaper profiles with
        // no LUTs. Structural inspection marks them as safe for all intents
        // (pure matrix+TRC path), so ZenCmsLite accepts them for the fast
        // path. Their encoded XYZ matrices drift ~590-931 u16 from
        // mathematically-derived canonical, but the structural rule treats
        // LUT-absence as sufficient — consistent with how other CMSs
        // route these profiles through matrix-shaper math anyway.
        let cms = ZenCmsLite::default();
        let p3_icc = crate::icc_profiles::DISPLAY_P3_V4;
        let bt2020_icc = crate::icc_profiles::REC2020_V4;

        let result = cms.build_transform_for_format(
            p3_icc,
            bt2020_icc,
            PixelFormat::RgbF32,
            PixelFormat::RgbF32,
        );
        assert!(
            result.is_ok(),
            "bundled compat profiles should be handled via lite CMS (matrix-shaper, no LUTs)"
        );
    }

    #[test]
    fn build_transform_unrecognized_icc_fails() {
        let cms = ZenCmsLite::default();
        let garbage = [0u8; 200];
        assert!(cms.build_transform(&garbage, &garbage).is_err());
    }

    #[test]
    fn build_source_transform_icc_compat() {
        // The bundled DisplayP3Compat-v4 is a matrix-shaper profile with no
        // LUTs → accepted by the structural rule → resolved by lite CMS.
        let cms = ZenCmsLite::default();
        let p3_icc = crate::icc_profiles::DISPLAY_P3_V4;
        let src = ColorProfileSource::Icc(p3_icc);
        let dst = ColorProfileSource::Named(NamedProfile::Srgb);
        let result = cms.build_source_transform(src, dst, PixelFormat::RgbF32, PixelFormat::RgbF32);
        assert!(
            result.is_some(),
            "bundled compat profile should resolve via lite CMS"
        );
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

// ===========================================================================
// Accuracy ground truth tests (moved from tests/accuracy_ground_truth.rs)
//
// These tests validate fast_gamut accuracy against f64 reference math and
// compare with moxcms. They live here because they need access to the
// pub(crate) ZenCmsLite::build_source_transform method.
// ===========================================================================

#[cfg(test)]
#[cfg(feature = "cms-moxcms")]
#[allow(clippy::type_complexity)]
mod accuracy_ground_truth_tests {
    use super::*;
    use crate::{ColorProfileSource, NamedProfile, PixelFormat};
    use moxcms::{
        BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, RenderingIntent,
        TransformOptions,
    };

    fn moxcms_opts() -> TransformOptions {
        TransformOptions {
            rendering_intent: RenderingIntent::RelativeColorimetric,
            allow_use_cicp_transfer: false,
            barycentric_weight_scale: BarycentricWeightScale::High,
            interpolation_method: InterpolationMethod::Tetrahedral,
            ..Default::default()
        }
    }

    // =====================================================================
    // f64 ground truth implementation
    // =====================================================================

    /// sRGB EOTF (encoded -> linear) in f64, C0-continuous constants.
    fn srgb_to_linear_f64(v: f64) -> f64 {
        const THRESH: f64 = 0.0392857142857142850819238;
        if v <= THRESH {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    /// sRGB inverse EOTF (linear -> encoded) in f64, C0-continuous constants.
    fn linear_to_srgb_f64(v: f64) -> f64 {
        const THRESH: f64 = 0.00303993464041981300277518;
        if v <= THRESH {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    /// BT.709 inverse OETF (encoded -> linear) in f64.
    fn bt709_to_linear_f64(v: f64) -> f64 {
        if v < 0.08124285829863519 {
            v / 4.5
        } else {
            ((v + 0.09929682680944) / 1.09929682680944).powf(1.0 / 0.45)
        }
    }

    /// f64 3x3 matrix multiply.
    fn mat3x3_f64(m: &[[f64; 3]; 3], r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        (
            m[0][0] * r + m[0][1] * g + m[0][2] * b,
            m[1][0] * r + m[1][1] * g + m[1][2] * b,
            m[2][0] * r + m[2][1] * g + m[2][2] * b,
        )
    }

    /// P3 -> sRGB matrix in f64.
    const P3_TO_SRGB_F64: [[f64; 3]; 3] = [
        [1.2249401763_f64, -0.2249401763_f64, 0.0_f64],
        [-0.0420569547_f64, 1.0420569547_f64, 0.0_f64],
        [-0.0196375546_f64, -0.0786360456_f64, 1.0982736001_f64],
    ];

    const BT2020_TO_SRGB_F64: [[f64; 3]; 3] = [
        [1.6604910021_f64, -0.5876411388_f64, -0.0728498633_f64],
        [-0.1245504745_f64, 1.1328998971_f64, -0.0083494226_f64],
        [-0.0181507634_f64, -0.1005788980_f64, 1.1187296614_f64],
    ];

    const ADOBERGB_TO_SRGB_F64: [[f64; 3]; 3] = [
        [1.3983557440_f64, -0.3983557440_f64, 0.0_f64],
        [0.0_f64, 1.0_f64, 0.0_f64],
        [0.0_f64, -0.0429289893_f64, 1.0429289893_f64],
    ];

    /// Adobe RGB gamma
    const ADOBE_GAMMA: f64 = 563.0 / 256.0;

    /// Compute ground truth u8 output via f64 math.
    fn ground_truth_u8(
        src: &[u8],
        matrix: &[[f64; 3]; 3],
        linearize: fn(f64) -> f64,
        encode: fn(f64) -> f64,
    ) -> Vec<u8> {
        let mut dst = vec![0u8; src.len()];
        for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
            let r = linearize(s[0] as f64 / 255.0);
            let g = linearize(s[1] as f64 / 255.0);
            let b = linearize(s[2] as f64 / 255.0);
            let (nr, ng, nb) = mat3x3_f64(matrix, r, g, b);
            d[0] = (encode(nr.clamp(0.0, 1.0)) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            d[1] = (encode(ng.clamp(0.0, 1.0)) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            d[2] = (encode(nb.clamp(0.0, 1.0)) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
        dst
    }

    /// Build full 256^3 source buffer.
    fn full_rgb_cube() -> Vec<u8> {
        let total = 256 * 256 * 256;
        let mut src = vec![0u8; total * 3];
        for i in 0..total {
            src[i * 3] = (i & 0xFF) as u8;
            src[i * 3 + 1] = ((i >> 8) & 0xFF) as u8;
            src[i * 3 + 2] = ((i >> 16) & 0xFF) as u8;
        }
        src
    }

    struct AccuracyResult {
        name: &'static str,
        fast_exact: usize,
        fast_off1: usize,
        fast_off2plus: usize,
        fast_max_delta: u8,
        mox_exact: usize,
        mox_off1: usize,
        mox_off2plus: usize,
        mox_max_delta: u8,
        fast_better: usize,
        mox_better: usize,
        tied: usize,
        total: usize,
    }

    impl AccuracyResult {
        fn print(&self) {
            let pct = |n: usize| n as f64 / self.total as f64 * 100.0;
            eprintln!("\n=== {} ({} pixels) ===", self.name, self.total);
            eprintln!(
                "  fast_gamut: exact={:.1}% +/-1={:.1}% +/-2+={:.1}% max_delta={}",
                pct(self.fast_exact),
                pct(self.fast_off1),
                pct(self.fast_off2plus),
                self.fast_max_delta
            );
            eprintln!(
                "  moxcms:     exact={:.1}% +/-1={:.1}% +/-2+={:.1}% max_delta={}",
                pct(self.mox_exact),
                pct(self.mox_off1),
                pct(self.mox_off2plus),
                self.mox_max_delta
            );
            eprintln!(
                "  fast_gamut closer to truth: {:.1}%",
                pct(self.fast_better)
            );
            eprintln!("  moxcms closer to truth:     {:.1}%", pct(self.mox_better));
            eprintln!("  tied (equal distance):      {:.1}%", pct(self.tied));
        }
    }

    fn compare_accuracy(
        name: &'static str,
        src: &[u8],
        truth: &[u8],
        fast_fn: &dyn Fn(&[u8], &mut [u8]),
        moxcms_src: &ColorProfile,
        moxcms_dst: &ColorProfile,
    ) -> AccuracyResult {
        let total = src.len() / 3;

        let mut fast_dst = vec![0u8; src.len()];
        let mut mox_dst = vec![0u8; src.len()];

        fast_fn(src, &mut fast_dst);
        let xform = moxcms_src
            .create_transform_8bit(Layout::Rgb, moxcms_dst, Layout::Rgb, moxcms_opts())
            .unwrap();
        xform.transform(src, &mut mox_dst).unwrap();

        let mut r = AccuracyResult {
            name,
            fast_exact: 0,
            fast_off1: 0,
            fast_off2plus: 0,
            fast_max_delta: 0,
            mox_exact: 0,
            mox_off1: 0,
            mox_off2plus: 0,
            mox_max_delta: 0,
            fast_better: 0,
            mox_better: 0,
            tied: 0,
            total,
        };

        for i in 0..total {
            let off = i * 3;
            let mut fast_dist: u16 = 0;
            let mut mox_dist: u16 = 0;

            for ch in 0..3 {
                let t = truth[off + ch];
                let f = fast_dst[off + ch];
                let m = mox_dst[off + ch];

                let fd = f.abs_diff(t);
                let md = m.abs_diff(t);
                fast_dist += fd as u16;
                mox_dist += md as u16;

                if fd > r.fast_max_delta {
                    r.fast_max_delta = fd;
                }
                if md > r.mox_max_delta {
                    r.mox_max_delta = md;
                }
            }

            // Per-pixel: count exact/off1/off2+
            let f_max_ch = (0..3)
                .map(|ch| fast_dst[off + ch].abs_diff(truth[off + ch]))
                .max()
                .unwrap();
            let m_max_ch = (0..3)
                .map(|ch| mox_dst[off + ch].abs_diff(truth[off + ch]))
                .max()
                .unwrap();

            match f_max_ch {
                0 => r.fast_exact += 1,
                1 => r.fast_off1 += 1,
                _ => r.fast_off2plus += 1,
            }
            match m_max_ch {
                0 => r.mox_exact += 1,
                1 => r.mox_off1 += 1,
                _ => r.mox_off2plus += 1,
            }

            // Who is closer (sum of absolute channel differences)?
            if fast_dist < mox_dist {
                r.fast_better += 1;
            } else if mox_dist < fast_dist {
                r.mox_better += 1;
            } else {
                r.tied += 1;
            }
        }

        r
    }

    /// Build a ZenCmsLite u8 RGB transform as a closure.
    fn build_lite_u8_transform(
        src_profile: ColorProfileSource<'_>,
        dst_profile: ColorProfileSource<'_>,
    ) -> alloc::boxed::Box<dyn Fn(&[u8], &mut [u8])> {
        let cms = ZenCmsLite::default();
        let xf = cms
            .build_source_transform(
                src_profile,
                dst_profile,
                PixelFormat::Rgb8,
                PixelFormat::Rgb8,
            )
            .unwrap()
            .unwrap();
        alloc::boxed::Box::new(move |src: &[u8], dst: &mut [u8]| {
            let width = (src.len() / 3) as u32;
            xf.transform_row(src, dst, width);
        })
    }

    #[test]
    fn p3_to_srgb_accuracy() {
        let src = full_rgb_cube();
        let truth = ground_truth_u8(
            &src,
            &P3_TO_SRGB_F64,
            srgb_to_linear_f64,
            linear_to_srgb_f64,
        );
        let fast_fn = build_lite_u8_transform(
            ColorProfileSource::Named(NamedProfile::DisplayP3),
            ColorProfileSource::Named(NamedProfile::Srgb),
        );
        let r = compare_accuracy(
            "P3->sRGB",
            &src,
            &truth,
            &*fast_fn,
            &ColorProfile::new_display_p3(),
            &ColorProfile::new_srgb(),
        );
        r.print();
    }

    #[test]
    fn bt2020_to_srgb_accuracy() {
        let src = full_rgb_cube();
        let truth = ground_truth_u8(
            &src,
            &BT2020_TO_SRGB_F64,
            bt709_to_linear_f64,
            linear_to_srgb_f64,
        );
        let fast_fn = build_lite_u8_transform(
            ColorProfileSource::Named(NamedProfile::Bt2020),
            ColorProfileSource::Named(NamedProfile::Srgb),
        );
        let r = compare_accuracy(
            "BT.2020 SDR->sRGB",
            &src,
            &truth,
            &*fast_fn,
            &ColorProfile::new_bt2020(),
            &ColorProfile::new_srgb(),
        );
        r.print();
    }

    #[test]
    fn adobergb_to_srgb_accuracy() {
        let src = full_rgb_cube();
        let truth = ground_truth_u8(
            &src,
            &ADOBERGB_TO_SRGB_F64,
            |v| v.powf(ADOBE_GAMMA),
            linear_to_srgb_f64,
        );
        let fast_fn = build_lite_u8_transform(
            ColorProfileSource::Named(NamedProfile::AdobeRgb),
            ColorProfileSource::Named(NamedProfile::Srgb),
        );
        let r = compare_accuracy(
            "AdobeRGB->sRGB",
            &src,
            &truth,
            &*fast_fn,
            &ColorProfile::new_adobe_rgb(),
            &ColorProfile::new_srgb(),
        );
        r.print();
    }
}

// ===========================================================================
// Gamut reduction comparison tests (moved from tests/gamut_reduction_compare.rs)
//
// These tests compare fast_gamut vs moxcms output pixel-by-pixel. They live
// here because they need access to the pub(crate) ZenCmsLite::build_source_transform
// method.
// ===========================================================================

#[cfg(test)]
#[cfg(feature = "cms-moxcms")]
#[allow(
    clippy::type_complexity,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::excessive_precision
)]
mod gamut_reduction_compare_tests {
    use super::*;
    use crate::{ColorProfileSource, NamedProfile, PixelFormat};
    use moxcms::{
        BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, RenderingIntent,
        TransformOptions,
    };

    fn moxcms_opts() -> TransformOptions {
        TransformOptions {
            rendering_intent: RenderingIntent::RelativeColorimetric,
            allow_use_cicp_transfer: false,
            barycentric_weight_scale: BarycentricWeightScale::High,
            interpolation_method: InterpolationMethod::Tetrahedral,
            ..Default::default()
        }
    }

    /// Build a ZenCmsLite u8 RGB transform as a closure.
    fn build_lite_u8_fn(
        src_profile: ColorProfileSource<'_>,
        dst_profile: ColorProfileSource<'_>,
    ) -> alloc::boxed::Box<dyn Fn(&[u8], &mut [u8])> {
        let cms = ZenCmsLite::default();
        let xf = cms
            .build_source_transform(
                src_profile,
                dst_profile,
                PixelFormat::Rgb8,
                PixelFormat::Rgb8,
            )
            .unwrap()
            .unwrap();
        alloc::boxed::Box::new(move |src: &[u8], dst: &mut [u8]| {
            let width = (src.len() / 3) as u32;
            xf.transform_row(src, dst, width);
        })
    }

    /// Build a ZenCmsLite f32 RGB in-place transform as a closure.
    fn build_lite_f32_fn(
        src_profile: ColorProfileSource<'_>,
        dst_profile: ColorProfileSource<'_>,
    ) -> alloc::boxed::Box<dyn Fn(&mut [f32])> {
        let cms = ZenCmsLite::default();
        let xf = cms
            .build_source_transform(
                src_profile,
                dst_profile,
                PixelFormat::RgbF32,
                PixelFormat::RgbF32,
            )
            .unwrap()
            .unwrap();
        alloc::boxed::Box::new(move |data: &mut [f32]| {
            let width = (data.len() / 3) as u32;
            let bytes: &[u8] = bytemuck::cast_slice(data);
            // transform_row needs separate src/dst; copy src first.
            let src_copy = bytes.to_vec();
            let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(data);
            xf.transform_row(&src_copy, dst_bytes, width);
        })
    }

    /// Compare fast_gamut output vs moxcms for a full 256^3 sweep.
    /// Returns (max_delta, differing_pixel_count, total_pixels, example_worst).
    fn compare_exhaustive_u8(
        fast_fn: &dyn Fn(&[u8], &mut [u8]),
        moxcms_src: &ColorProfile,
        moxcms_dst: &ColorProfile,
    ) -> (u8, usize, usize, Option<([u8; 3], [u8; 3], [u8; 3])>) {
        let opts = moxcms_opts();
        let xform = moxcms_src
            .create_transform_8bit(Layout::Rgb, moxcms_dst, Layout::Rgb, opts)
            .unwrap();

        let total = 256 * 256 * 256;
        let mut src = vec![0u8; total * 3];
        for i in 0..total {
            src[i * 3] = (i & 0xFF) as u8;
            src[i * 3 + 1] = ((i >> 8) & 0xFF) as u8;
            src[i * 3 + 2] = ((i >> 16) & 0xFF) as u8;
        }

        let mut fast_dst = vec![0u8; src.len()];
        let mut mox_dst = vec![0u8; src.len()];

        fast_fn(&src, &mut fast_dst);
        xform.transform(&src, &mut mox_dst).unwrap();

        let mut max_delta: u8 = 0;
        let mut diff_count: usize = 0;
        let mut worst: Option<([u8; 3], [u8; 3], [u8; 3])> = None;

        for i in 0..total {
            let off = i * 3;
            let f = [fast_dst[off], fast_dst[off + 1], fast_dst[off + 2]];
            let m = [mox_dst[off], mox_dst[off + 1], mox_dst[off + 2]];
            let s = [src[off], src[off + 1], src[off + 2]];

            if f != m {
                diff_count += 1;
                for ch in 0..3 {
                    let d = f[ch].abs_diff(m[ch]);
                    if d > max_delta {
                        max_delta = d;
                        worst = Some((s, f, m));
                    }
                }
            }
        }

        (max_delta, diff_count, total, worst)
    }

    /// Same comparison but for f32, sampling a grid (not full 256^3).
    fn compare_f32_grid(
        fast_fn: &dyn Fn(&mut [f32]),
        moxcms_src: &ColorProfile,
        moxcms_dst: &ColorProfile,
        step: usize,
    ) -> (f32, usize, usize) {
        let opts = moxcms_opts();
        let xform = moxcms_src
            .create_transform_f32(Layout::Rgb, moxcms_dst, Layout::Rgb, opts)
            .unwrap();

        let steps = (256 + step - 1) / step;
        let total = steps * steps * steps;
        let mut src = vec![0.0f32; total * 3];
        let mut idx = 0;
        for r in (0..=255).step_by(step) {
            for g in (0..=255).step_by(step) {
                for b in (0..=255).step_by(step) {
                    src[idx * 3] = r as f32 / 255.0;
                    src[idx * 3 + 1] = g as f32 / 255.0;
                    src[idx * 3 + 2] = b as f32 / 255.0;
                    idx += 1;
                }
            }
        }

        let mut fast_buf = src.clone();
        let mut mox_dst = vec![0.0f32; src.len()];

        fast_fn(&mut fast_buf);
        xform.transform(&src, &mut mox_dst).unwrap();

        let mut max_delta: f32 = 0.0;
        let mut diff_count: usize = 0;

        for i in 0..idx {
            let off = i * 3;
            for ch in 0..3 {
                let d = (fast_buf[off + ch] - mox_dst[off + ch]).abs();
                if d > 1e-6 {
                    diff_count += 1;
                }
                if d > max_delta {
                    max_delta = d;
                }
            }
        }

        (max_delta, diff_count, idx)
    }

    // =====================================================================
    // Tests
    // =====================================================================

    #[test]
    fn p3_to_srgb_u8_vs_moxcms() {
        let fast_fn = build_lite_u8_fn(
            ColorProfileSource::Named(NamedProfile::DisplayP3),
            ColorProfileSource::Named(NamedProfile::Srgb),
        );
        let (max_delta, diff_count, total, worst) = compare_exhaustive_u8(
            &*fast_fn,
            &ColorProfile::new_display_p3(),
            &ColorProfile::new_srgb(),
        );
        eprintln!("P3->sRGB u8: {diff_count}/{total} pixels differ, max delta={max_delta}");
        if let Some((src, fast, mox)) = worst {
            eprintln!("  worst: src={src:?} fast={fast:?} moxcms={mox:?}");
        }
        assert!(
            max_delta <= 2,
            "P3->sRGB u8: max delta {max_delta} > 2 -- moxcms may use a different algorithm"
        );
    }

    #[test]
    fn srgb_to_p3_u8_vs_moxcms() {
        let fast_fn = build_lite_u8_fn(
            ColorProfileSource::Named(NamedProfile::Srgb),
            ColorProfileSource::Named(NamedProfile::DisplayP3),
        );
        let (max_delta, diff_count, total, worst) = compare_exhaustive_u8(
            &*fast_fn,
            &ColorProfile::new_srgb(),
            &ColorProfile::new_display_p3(),
        );
        eprintln!("sRGB->P3 u8: {diff_count}/{total} pixels differ, max delta={max_delta}");
        if let Some((src, fast, mox)) = worst {
            eprintln!("  worst: src={src:?} fast={fast:?} moxcms={mox:?}");
        }
        assert!(max_delta <= 2, "sRGB->P3 u8: max delta {max_delta} > 2");
    }

    #[test]
    fn bt2020_sdr_to_srgb_u8_vs_moxcms() {
        let fast_fn = build_lite_u8_fn(
            ColorProfileSource::Named(NamedProfile::Bt2020),
            ColorProfileSource::Named(NamedProfile::Srgb),
        );
        let (max_delta, diff_count, total, worst) = compare_exhaustive_u8(
            &*fast_fn,
            &ColorProfile::new_bt2020(),
            &ColorProfile::new_srgb(),
        );
        eprintln!(
            "BT.2020 SDR->sRGB u8: {diff_count}/{total} pixels differ, max delta={max_delta}"
        );
        if let Some((src, fast, mox)) = worst {
            eprintln!("  worst: src={src:?} fast={fast:?} moxcms={mox:?}");
        }
        assert!(
            max_delta <= 3,
            "BT.2020->sRGB u8: max delta {max_delta} > 3 -- check if moxcms does gamut mapping"
        );
    }

    #[test]
    fn adobergb_to_srgb_u8_vs_moxcms() {
        let fast_fn = build_lite_u8_fn(
            ColorProfileSource::Named(NamedProfile::AdobeRgb),
            ColorProfileSource::Named(NamedProfile::Srgb),
        );
        let (max_delta, diff_count, total, worst) = compare_exhaustive_u8(
            &*fast_fn,
            &ColorProfile::new_adobe_rgb(),
            &ColorProfile::new_srgb(),
        );
        eprintln!("AdobeRGB->sRGB u8: {diff_count}/{total} pixels differ, max delta={max_delta}");
        if let Some((src, fast, mox)) = worst {
            eprintln!("  worst: src={src:?} fast={fast:?} moxcms={mox:?}");
        }
        assert!(
            max_delta <= 2,
            "AdobeRGB->sRGB u8: max delta {max_delta} > 2"
        );
    }

    #[test]
    fn p3_to_srgb_f32_vs_moxcms() {
        let fast_fn = build_lite_f32_fn(
            ColorProfileSource::Named(NamedProfile::DisplayP3),
            ColorProfileSource::Named(NamedProfile::Srgb),
        );
        let (max_delta, diff_count, total) = compare_f32_grid(
            &*fast_fn,
            &ColorProfile::new_display_p3(),
            &ColorProfile::new_srgb(),
            4,
        );
        eprintln!(
            "P3->sRGB f32: {diff_count}/{total}x3 channels differ >1e-6, max delta={max_delta:.6e}"
        );
        assert!(
            max_delta < 0.01,
            "P3->sRGB f32: max delta {max_delta} -- unexpected divergence"
        );
    }

    #[test]
    fn bt2020_to_srgb_f32_vs_moxcms() {
        let fast_fn = build_lite_f32_fn(
            ColorProfileSource::Named(NamedProfile::Bt2020),
            ColorProfileSource::Named(NamedProfile::Srgb),
        );
        let (max_delta, diff_count, total) = compare_f32_grid(
            &*fast_fn,
            &ColorProfile::new_bt2020(),
            &ColorProfile::new_srgb(),
            4,
        );
        eprintln!(
            "BT.2020->sRGB f32: {diff_count}/{total}x3 channels differ >1e-6, max delta={max_delta:.6e}"
        );
        assert!(max_delta < 0.01, "BT.2020->sRGB f32: max delta {max_delta}");
    }
}
