//! Color profile types for CMS integration.
//!
//! Provides a unified way to reference the source color space of decoded
//! pixels, suitable for passing to a CMS backend (e.g., moxcms, lcms2).
//!
//! [`ColorContext`] bundles ICC and CICP metadata for cheap sharing
//! (via `Arc`) across pixel slices and pipeline stages.
//! Current color state (transfer, primaries, alpha) is tracked on the
//! pixel descriptor itself, not as a separate enum.

use crate::TransferFunction;
use crate::cicp::Cicp;
use alloc::sync::Arc;

/// A source color profile — either ICC bytes or CICP parameters.
///
/// This unified type lets consumers pass decoded image color info
/// directly to a CMS backend without caring whether the source had
/// an ICC profile, CICP codes, or a well-known named profile.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorProfileSource<'a> {
    /// Raw ICC profile data.
    Icc(&'a [u8]),
    /// CICP parameters (a CMS can synthesize an equivalent profile).
    Cicp(Cicp),
    /// Well-known named profile.
    Named(NamedProfile),
}

/// Well-known color profiles that any CMS should recognize.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum NamedProfile {
    /// sRGB (IEC 61966-2-1). The web and desktop default.
    #[default]
    Srgb,
    /// Display P3 with sRGB transfer curve.
    DisplayP3,
    /// BT.2020 with BT.709 transfer (SDR wide gamut).
    Bt2020,
    /// BT.2020 with PQ transfer (HDR10, SMPTE ST 2084).
    Bt2020Pq,
    /// BT.2020 with HLG transfer (ARIB STD-B67, HDR broadcast).
    Bt2020Hlg,
    /// Adobe RGB (1998). Used in print workflows.
    AdobeRgb,
    /// Linear sRGB (sRGB primaries, gamma 1.0).
    LinearSrgb,
}

impl NamedProfile {
    /// Map CICP parameters to a well-known named profile.
    ///
    /// Recognizes sRGB, Display P3, BT.2020 (SDR), BT.2100 PQ, BT.2100 HLG,
    /// and Linear sRGB. Returns `None` for unrecognized combinations.
    pub const fn from_cicp(cicp: Cicp) -> Option<Self> {
        // Match on (primaries, transfer, matrix, full_range).
        // We match matrix_coefficients == 0 (Identity/RGB) for all RGB profiles.
        match (
            cicp.color_primaries,
            cicp.transfer_characteristics,
            cicp.matrix_coefficients,
        ) {
            (1, 13, 0) => Some(Self::Srgb),
            (12, 13, 0) => Some(Self::DisplayP3),
            (9, 1, 0) => Some(Self::Bt2020),
            (9, 16, _) => Some(Self::Bt2020Pq), // BT.2100 PQ (any matrix)
            (9, 18, _) => Some(Self::Bt2020Hlg), // BT.2100 HLG (any matrix)
            (1, 8, 0) => Some(Self::LinearSrgb),
            _ => None,
        }
    }

    /// Convert to CICP parameters, if a standard mapping exists.
    pub const fn to_cicp(self) -> Option<Cicp> {
        match self {
            Self::Srgb => Some(Cicp::SRGB),
            Self::DisplayP3 => Some(Cicp::DISPLAY_P3),
            Self::Bt2020 => Some(Cicp {
                color_primaries: 9,
                transfer_characteristics: 1,
                matrix_coefficients: 0,
                full_range: true,
            }),
            Self::Bt2020Pq => Some(Cicp::BT2100_PQ),
            Self::Bt2020Hlg => Some(Cicp::BT2100_HLG),
            Self::LinearSrgb => Some(Cicp {
                color_primaries: 1,
                transfer_characteristics: 8,
                matrix_coefficients: 0,
                full_range: true,
            }),
            Self::AdobeRgb => None,
        }
    }
}

/// Color space metadata for pixel data.
///
/// Bundles ICC profile bytes and/or CICP parameters into a single
/// shareable context. Carried via `Arc` on pixel slices and pipeline
/// sources so color metadata travels with pixel data without per-strip
/// cloning overhead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorContext {
    /// Raw ICC profile bytes.
    pub icc: Option<Arc<[u8]>>,
    /// CICP parameters (ITU-T H.273).
    pub cicp: Option<Cicp>,
}

impl ColorContext {
    /// Create from an ICC profile.
    pub fn from_icc(icc: impl Into<Arc<[u8]>>) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: None,
        }
    }

    /// Create from CICP parameters.
    pub fn from_cicp(cicp: Cicp) -> Self {
        Self {
            icc: None,
            cicp: Some(cicp),
        }
    }

    /// Create from both ICC and CICP.
    pub fn from_icc_and_cicp(icc: impl Into<Arc<[u8]>>, cicp: Cicp) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: Some(cicp),
        }
    }

    /// Get a [`ColorProfileSource`] reference for CMS integration.
    ///
    /// Returns CICP if present (takes precedence per AVIF/HEIF specs),
    /// otherwise returns the ICC profile bytes.
    pub fn as_profile_source(&self) -> Option<ColorProfileSource<'_>> {
        if let Some(cicp) = self.cicp {
            Some(ColorProfileSource::Cicp(cicp))
        } else {
            self.icc.as_deref().map(ColorProfileSource::Icc)
        }
    }

    /// Derive transfer function from CICP (or `Unknown` if no CICP).
    pub fn transfer_function(&self) -> TransferFunction {
        self.cicp
            .and_then(|c| TransferFunction::from_cicp(c.transfer_characteristics))
            .unwrap_or(TransferFunction::Unknown)
    }

    /// True if this describes sRGB (either via CICP or trivially).
    pub fn is_srgb(&self) -> bool {
        self.cicp == Some(Cicp::SRGB)
    }
}

// ---------------------------------------------------------------------------
// Color provenance — how the source described its color
// ---------------------------------------------------------------------------

/// How the source file described its color information.
///
/// Preserved for re-encoding and round-trip conversion decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorProvenance {
    /// Color was described via an embedded ICC profile.
    Icc,
    /// Color was described via CICP code points.
    Cicp,
    /// Color was described via PNG gAMA + cHRM chunks or similar.
    GamaChrm,
    /// Color was assumed (no explicit metadata in source).
    Assumed,
}

/// Immutable record of how the source file described its color.
///
/// Tracks the original color description from the decoded file so
/// encoders can make provenance-aware decisions (e.g., re-embed the
/// original ICC profile, or prefer CICP when re-encoding to AVIF).
///
/// `ColorOrigin` is immutable once set. It records how color was
/// described, not what the pixels currently are. The encoder uses
/// [`PixelDescriptor`](crate::PixelDescriptor) for the current state
/// and can consult `ColorOrigin` for provenance decisions.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ColorOrigin {
    /// Raw ICC profile bytes from the source file, if any.
    pub icc: Option<Arc<[u8]>>,
    /// CICP parameters from the source file, if any.
    pub cicp: Option<Cicp>,
    /// How the color information was originally described.
    pub provenance: ColorProvenance,
}

impl ColorOrigin {
    /// Create from an ICC profile.
    pub fn from_icc(icc: impl Into<Arc<[u8]>>) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: None,
            provenance: ColorProvenance::Icc,
        }
    }

    /// Create from CICP parameters.
    pub fn from_cicp(cicp: Cicp) -> Self {
        Self {
            icc: None,
            cicp: Some(cicp),
            provenance: ColorProvenance::Cicp,
        }
    }

    /// Create from both ICC and CICP (e.g., AVIF with both).
    pub fn from_icc_and_cicp(icc: impl Into<Arc<[u8]>>, cicp: Cicp) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: Some(cicp),
            provenance: ColorProvenance::Cicp,
        }
    }

    /// Create from gAMA/cHRM chunks (no ICC or CICP).
    pub fn from_gama_chrm() -> Self {
        Self {
            icc: None,
            cicp: None,
            provenance: ColorProvenance::GamaChrm,
        }
    }

    /// Create for assumed/default color (no explicit metadata).
    pub fn assumed() -> Self {
        Self {
            icc: None,
            cicp: None,
            provenance: ColorProvenance::Assumed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn named_profile_default_is_srgb() {
        assert_eq!(NamedProfile::default(), NamedProfile::Srgb);
    }

    #[test]
    fn named_profile_to_cicp() {
        assert_eq!(NamedProfile::Srgb.to_cicp(), Some(Cicp::SRGB));
        assert_eq!(NamedProfile::Bt2020Pq.to_cicp(), Some(Cicp::BT2100_PQ));
        assert!(NamedProfile::AdobeRgb.to_cicp().is_none());
    }

    #[test]
    fn named_profile_from_cicp() {
        assert_eq!(
            NamedProfile::from_cicp(Cicp::SRGB),
            Some(NamedProfile::Srgb)
        );
        assert_eq!(
            NamedProfile::from_cicp(Cicp::DISPLAY_P3),
            Some(NamedProfile::DisplayP3)
        );
        assert_eq!(
            NamedProfile::from_cicp(Cicp::BT2100_PQ),
            Some(NamedProfile::Bt2020Pq)
        );
        assert_eq!(
            NamedProfile::from_cicp(Cicp::BT2100_HLG),
            Some(NamedProfile::Bt2020Hlg)
        );
        // Linear sRGB
        assert_eq!(
            NamedProfile::from_cicp(Cicp::new(1, 8, 0, true)),
            Some(NamedProfile::LinearSrgb)
        );
        // BT.2020 SDR
        assert_eq!(
            NamedProfile::from_cicp(Cicp::new(9, 1, 0, true)),
            Some(NamedProfile::Bt2020)
        );
        // Unknown combo
        assert_eq!(NamedProfile::from_cicp(Cicp::new(99, 99, 0, true)), None);
    }

    #[test]
    fn named_profile_cicp_roundtrip() {
        for profile in [
            NamedProfile::Srgb,
            NamedProfile::DisplayP3,
            NamedProfile::Bt2020,
            NamedProfile::Bt2020Pq,
            NamedProfile::Bt2020Hlg,
            NamedProfile::LinearSrgb,
        ] {
            let cicp = profile.to_cicp().unwrap();
            assert_eq!(
                NamedProfile::from_cicp(cicp),
                Some(profile),
                "roundtrip failed for {profile:?}"
            );
        }
    }

    #[test]
    fn color_context_from_cicp() {
        let ctx = ColorContext::from_cicp(Cicp::SRGB);
        assert!(ctx.icc.is_none());
        assert_eq!(ctx.cicp, Some(Cicp::SRGB));
    }

    #[test]
    fn color_context_profile_source_prefers_cicp() {
        let ctx = ColorContext::from_icc_and_cicp(vec![1, 2, 3], Cicp::SRGB);
        let src = ctx.as_profile_source().unwrap();
        assert_eq!(src, ColorProfileSource::Cicp(Cicp::SRGB));
    }

    #[test]
    fn color_context_is_srgb() {
        assert!(ColorContext::from_cicp(Cicp::SRGB).is_srgb());
        assert!(!ColorContext::from_cicp(Cicp::BT2100_PQ).is_srgb());
    }

    #[test]
    fn color_context_transfer_function() {
        assert_eq!(
            ColorContext::from_cicp(Cicp::SRGB).transfer_function(),
            TransferFunction::Srgb
        );
        assert_eq!(
            ColorContext::from_icc(vec![1]).transfer_function(),
            TransferFunction::Unknown
        );
    }

    // --- ColorContext additional coverage ---

    #[test]
    fn color_context_from_icc() {
        let ctx = ColorContext::from_icc(vec![10, 20, 30]);
        assert!(ctx.icc.is_some());
        assert_eq!(ctx.icc.as_deref(), Some(&[10u8, 20, 30][..]));
        assert!(ctx.cicp.is_none());
    }

    #[test]
    fn color_context_from_icc_and_cicp() {
        let ctx = ColorContext::from_icc_and_cicp(vec![1, 2], Cicp::BT2100_PQ);
        assert!(ctx.icc.is_some());
        assert_eq!(ctx.cicp, Some(Cicp::BT2100_PQ));
    }

    #[test]
    fn color_context_profile_source_icc_only() {
        let ctx = ColorContext::from_icc(vec![42]);
        let src = ctx.as_profile_source().unwrap();
        assert_eq!(src, ColorProfileSource::Icc(&[42]));
    }

    #[test]
    fn color_context_profile_source_none() {
        let ctx = ColorContext {
            icc: None,
            cicp: None,
        };
        assert!(ctx.as_profile_source().is_none());
    }

    #[test]
    fn color_context_pq_transfer() {
        assert_eq!(
            ColorContext::from_cicp(Cicp::BT2100_PQ).transfer_function(),
            TransferFunction::Pq
        );
    }

    #[test]
    fn color_context_hlg_transfer() {
        assert_eq!(
            ColorContext::from_cicp(Cicp::BT2100_HLG).transfer_function(),
            TransferFunction::Hlg
        );
    }

    #[test]
    fn color_context_eq_and_clone() {
        let a = ColorContext::from_cicp(Cicp::SRGB);
        let b = a.clone();
        assert_eq!(a, b);
        let c = ColorContext::from_icc(vec![1]);
        assert_ne!(a, c);
    }

    #[test]
    fn color_context_debug() {
        let ctx = ColorContext::from_cicp(Cicp::SRGB);
        let s = alloc::format!("{ctx:?}");
        assert!(s.contains("ColorContext"));
    }

    // --- ColorProfileSource coverage ---

    #[test]
    fn color_profile_source_named() {
        let src = ColorProfileSource::Named(NamedProfile::DisplayP3);
        assert_eq!(src, ColorProfileSource::Named(NamedProfile::DisplayP3));
        assert_ne!(src, ColorProfileSource::Named(NamedProfile::Srgb));
    }

    #[test]
    fn color_profile_source_cicp() {
        let src = ColorProfileSource::Cicp(Cicp::BT2100_PQ);
        assert_eq!(src, ColorProfileSource::Cicp(Cicp::BT2100_PQ));
    }

    #[test]
    fn color_profile_source_icc() {
        let data: &[u8] = &[1, 2, 3];
        let src = ColorProfileSource::Icc(data);
        assert_eq!(src, ColorProfileSource::Icc(&[1, 2, 3]));
    }

    #[test]
    fn color_profile_source_debug_clone() {
        let src = ColorProfileSource::Named(NamedProfile::Srgb);
        let s = alloc::format!("{src:?}");
        assert!(s.contains("Named"));
        let src2 = src.clone();
        assert_eq!(src, src2);
    }

    // --- NamedProfile coverage ---

    #[test]
    fn named_profile_all_variants_to_cicp() {
        assert!(NamedProfile::Srgb.to_cicp().is_some());
        assert!(NamedProfile::DisplayP3.to_cicp().is_some());
        assert!(NamedProfile::Bt2020.to_cicp().is_some());
        assert!(NamedProfile::Bt2020Pq.to_cicp().is_some());
        assert!(NamedProfile::Bt2020Hlg.to_cicp().is_some());
        assert!(NamedProfile::LinearSrgb.to_cicp().is_some());
        assert!(NamedProfile::AdobeRgb.to_cicp().is_none());
    }

    #[test]
    fn named_profile_debug_clone_eq() {
        let p = NamedProfile::DisplayP3;
        let _ = alloc::format!("{p:?}");
        let p2 = p;
        assert_eq!(p, p2);
    }

    #[test]
    #[cfg(feature = "std")]
    fn named_profile_hash() {
        use core::hash::{Hash, Hasher};
        let p = NamedProfile::DisplayP3;
        let mut h = std::hash::DefaultHasher::new();
        p.hash(&mut h);
        let _ = h.finish();
    }

    // --- ColorProvenance coverage ---

    #[test]
    fn color_provenance_variants() {
        assert_ne!(ColorProvenance::Icc, ColorProvenance::Cicp);
        assert_ne!(ColorProvenance::GamaChrm, ColorProvenance::Assumed);
        let a = ColorProvenance::Icc;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn color_provenance_debug() {
        let p = ColorProvenance::Cicp;
        let _ = alloc::format!("{p:?}");
    }

    #[test]
    #[cfg(feature = "std")]
    fn color_provenance_hash() {
        use core::hash::{Hash, Hasher};
        let p = ColorProvenance::Cicp;
        let mut h = std::hash::DefaultHasher::new();
        p.hash(&mut h);
        let _ = h.finish();
    }

    // --- ColorOrigin coverage ---

    #[test]
    fn color_origin_from_icc() {
        let o = ColorOrigin::from_icc(vec![1, 2, 3]);
        assert!(o.icc.is_some());
        assert!(o.cicp.is_none());
        assert_eq!(o.provenance, ColorProvenance::Icc);
    }

    #[test]
    fn color_origin_from_cicp() {
        let o = ColorOrigin::from_cicp(Cicp::SRGB);
        assert!(o.icc.is_none());
        assert_eq!(o.cicp, Some(Cicp::SRGB));
        assert_eq!(o.provenance, ColorProvenance::Cicp);
    }

    #[test]
    fn color_origin_from_icc_and_cicp() {
        let o = ColorOrigin::from_icc_and_cicp(vec![10], Cicp::BT2100_PQ);
        assert!(o.icc.is_some());
        assert_eq!(o.cicp, Some(Cicp::BT2100_PQ));
        assert_eq!(o.provenance, ColorProvenance::Cicp);
    }

    #[test]
    fn color_origin_from_gama_chrm() {
        let o = ColorOrigin::from_gama_chrm();
        assert!(o.icc.is_none());
        assert!(o.cicp.is_none());
        assert_eq!(o.provenance, ColorProvenance::GamaChrm);
    }

    #[test]
    fn color_origin_assumed() {
        let o = ColorOrigin::assumed();
        assert!(o.icc.is_none());
        assert!(o.cicp.is_none());
        assert_eq!(o.provenance, ColorProvenance::Assumed);
    }

    #[test]
    fn color_origin_eq_clone_debug() {
        let a = ColorOrigin::from_cicp(Cicp::SRGB);
        let b = a.clone();
        assert_eq!(a, b);
        let _ = alloc::format!("{a:?}");
        let c = ColorOrigin::assumed();
        assert_ne!(a, c);
    }
}
