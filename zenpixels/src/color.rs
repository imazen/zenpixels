//! Color profile types for CMS integration.
//!
//! Provides a unified way to reference the source color space of decoded
//! pixels, suitable for passing to a CMS backend (e.g., moxcms, lcms2).
//!
//! [`ColorContext`] bundles ICC and CICP metadata for cheap sharing
//! (via `Arc`) across pixel slices and pipeline stages.
//! Current color state (transfer, primaries, alpha) is tracked on the
//! pixel descriptor itself, not as a separate enum.

use crate::cicp::Cicp;
use crate::{ColorPrimaries, TransferFunction};
use alloc::sync::Arc;

/// A source color profile — either ICC bytes, CICP parameters, a named
/// profile, or a primaries + transfer function pair.
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
    /// Color primaries + transfer function pair.
    ///
    /// Covers the full `ColorPrimaries × TransferFunction` matrix,
    /// including combinations that don't have a [`NamedProfile`] variant
    /// or a CICP mapping (e.g., Adobe RGB).
    ///
    /// A CMS backend that handles this variant can avoid ICC profile
    /// parsing entirely for known primaries/transfer combinations.
    PrimariesTransferPair {
        /// Color primaries (gamut + white point).
        primaries: ColorPrimaries,
        /// Transfer function (EOTF encoding).
        transfer: TransferFunction,
    },
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
    /// BT.2020 with HLG transfer (BT.2100 HLG).
    Bt2020Hlg,
    /// Adobe RGB (1998). Used in print workflows.
    AdobeRgb,
    /// Linear sRGB (sRGB primaries, gamma 1.0).
    LinearSrgb,
}

impl NamedProfile {
    /// Map CICP parameters to a well-known named profile.
    ///
    /// Recognizes sRGB, Display P3, BT.2020 (SDR), BT.2100 PQ,
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

    /// Decompose into primaries + transfer function.
    pub const fn to_primaries_transfer(self) -> (ColorPrimaries, TransferFunction) {
        match self {
            Self::Srgb => (ColorPrimaries::Bt709, TransferFunction::Srgb),
            Self::DisplayP3 => (ColorPrimaries::DisplayP3, TransferFunction::Srgb),
            Self::Bt2020 => (ColorPrimaries::Bt2020, TransferFunction::Bt709),
            Self::Bt2020Pq => (ColorPrimaries::Bt2020, TransferFunction::Pq),
            Self::Bt2020Hlg => (ColorPrimaries::Bt2020, TransferFunction::Hlg),
            Self::AdobeRgb => (ColorPrimaries::AdobeRgb, TransferFunction::Gamma22),
            Self::LinearSrgb => (ColorPrimaries::Bt709, TransferFunction::Linear),
        }
    }

    /// Try to construct from a primaries + transfer pair.
    ///
    /// Returns `None` for combinations that don't have a named profile
    /// (e.g., Display P3 + PQ, or Adobe RGB + Linear).
    pub const fn from_primaries_transfer(
        primaries: ColorPrimaries,
        transfer: TransferFunction,
    ) -> Option<Self> {
        match (primaries, transfer) {
            (ColorPrimaries::Bt709, TransferFunction::Srgb) => Some(Self::Srgb),
            (ColorPrimaries::DisplayP3, TransferFunction::Srgb) => Some(Self::DisplayP3),
            (ColorPrimaries::Bt2020, TransferFunction::Bt709) => Some(Self::Bt2020),
            (ColorPrimaries::Bt2020, TransferFunction::Pq) => Some(Self::Bt2020Pq),
            (ColorPrimaries::Bt2020, TransferFunction::Hlg) => Some(Self::Bt2020Hlg),
            (ColorPrimaries::AdobeRgb, TransferFunction::Gamma22) => Some(Self::AdobeRgb),
            (ColorPrimaries::Bt709, TransferFunction::Linear) => Some(Self::LinearSrgb),
            _ => None,
        }
    }
}

impl<'a> ColorProfileSource<'a> {
    /// Create from primaries + transfer function.
    pub const fn from_primaries_transfer(
        primaries: ColorPrimaries,
        transfer: TransferFunction,
    ) -> Self {
        Self::PrimariesTransferPair {
            primaries,
            transfer,
        }
    }

    /// Try to extract primaries + transfer, regardless of variant.
    ///
    /// - `PrimariesTransferPair` — returns directly
    /// - `Named` — decomposes via [`NamedProfile::to_primaries_transfer`]
    /// - `Cicp` — maps via [`ColorPrimaries::from_cicp`] and [`TransferFunction::from_cicp`]
    /// - `Icc` — returns `None` (requires profile parsing)
    pub const fn primaries_transfer(&self) -> Option<(ColorPrimaries, TransferFunction)> {
        match self {
            Self::PrimariesTransferPair {
                primaries,
                transfer,
            } => Some((*primaries, *transfer)),
            Self::Named(named) => Some(named.to_primaries_transfer()),
            Self::Cicp(cicp) => {
                let p = ColorPrimaries::from_cicp(cicp.color_primaries);
                let t = TransferFunction::from_cicp(cicp.transfer_characteristics);
                match (p, t) {
                    (Some(p), Some(t)) => Some((p, t)),
                    _ => None,
                }
            }
            Self::Icc(_) => None,
        }
    }

    /// Resolve to a (primaries, transfer) pair using all available methods,
    /// including ICC profile identification when the `icc` feature is enabled.
    ///
    /// This is the most complete resolution path:
    /// - `PrimariesTransferPair` — returns directly
    /// - `Named` — decomposes via [`NamedProfile::to_primaries_transfer`]
    /// - `Cicp` — maps via `from_cicp`, but returns `None` if `matrix_coefficients`
    ///   is non-zero (YCbCr data requires matrix conversion first) or `full_range`
    ///   is false (narrow-range data needs range expansion first)
    /// - `Icc` — hash-based identification (~100ns, 135 known profiles) + CICP-in-ICC
    ///   extraction. Returns `None` for unknown custom profiles.
    ///
    /// Returns `None` when the profile is unknown or when reducing to
    /// (primaries, transfer) would discard significant information
    /// (YCbCr matrix coefficients, narrow signal range).
    #[cfg(feature = "icc")]
    #[allow(unreachable_patterns)]
    pub fn resolve(&self) -> Option<(ColorPrimaries, TransferFunction)> {
        match self {
            Self::PrimariesTransferPair {
                primaries,
                transfer,
            } => Some((*primaries, *transfer)),
            Self::Named(named) => Some(named.to_primaries_transfer()),
            Self::Cicp(cicp) => {
                // Non-identity matrix coefficients mean YCbCr data — can't reduce
                // to just primaries+transfer without a YCbCr→RGB matrix step.
                if cicp.matrix_coefficients != 0 {
                    return None;
                }
                // Narrow range needs expansion before primaries+transfer applies.
                if !cicp.full_range {
                    return None;
                }
                let p = ColorPrimaries::from_cicp(cicp.color_primaries)?;
                let t = TransferFunction::from_cicp(cicp.transfer_characteristics)?;
                Some((p, t))
            }
            Self::Icc(icc_bytes) => {
                // Use the intent-safe variant — rejects profiles whose CMS
                // output would differ from our matrix+TRC math (non-Bradford
                // chad, LUT tags, Lab PCS) for the colorimetric intent. For
                // approximation, use `crate::icc::identify_common` directly.
                if let Some(id) = crate::icc::identify_common_for(
                    icc_bytes,
                    crate::icc::Tolerance::Intent,
                    crate::icc::CoalesceForUse::RelativeColorimetric,
                ) {
                    return Some((id.primaries, id.transfer));
                }
                // CICP-in-ICC tag (ICC v4.4+) is authoritative — accept it.
                if let Some(cicp) = crate::icc::extract_cicp(icc_bytes) {
                    if cicp.matrix_coefficients != 0 || !cicp.full_range {
                        return None;
                    }
                    let p = ColorPrimaries::from_cicp(cicp.color_primaries)?;
                    let t = TransferFunction::from_cicp(cicp.transfer_characteristics)?;
                    return Some((p, t));
                }
                None
            }
            _ => None,
        }
    }
}

/// Which color metadata a CMS should prefer when building transforms.
///
/// When both ICC and CICP are present, this determines which one the CMS
/// uses. When only one is present, it is used regardless of this flag —
/// the authority controls precedence, not exclusivity.
///
/// **Codec contract:** codecs should set this to match the data they
/// actually provide. Setting `Icc` without populating `icc_profile` (or
/// `Cicp` without `cicp`) is a codec bug. Implementations *may* fall back
/// to the other field when the authoritative one is absent, but are not
/// required to — they may also treat the mismatch as an error or assume
/// sRGB.
///
/// [`ColorContext::as_profile_source`] implements a lenient resolution:
/// preferred field → other field → `None`. Stricter consumers can inspect
/// the authority and the fields directly.
///
/// The codec sets this during decode based on the format's specification:
///
/// - **JPEG, WebP, TIFF**: `Icc` (ICC is the only color signal)
/// - **PNG 3rd Ed**: `Cicp` when cICP chunk present, `Icc` when only iCCP
/// - **AVIF/MIAF**: `Icc` when ICC colr box present, `Cicp` otherwise
/// - **HEIC/HEIF**: `Cicp` when nclx colr box present, `Icc` otherwise
/// - **JPEG XL**: `Cicp` when enum encoding (`want_icc=false`), `Icc` when embedded ICC
///
/// Both `icc_profile` and `cicp` may be populated regardless of this flag —
/// the non-authoritative field is preserved for metadata roundtripping.
///
/// This is distinct from [`ColorProvenance`] which records *how the source
/// described* its color (for re-encoding decisions). `ColorAuthority` says
/// which field the CMS should *prefer* for building transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorAuthority {
    /// ICC profile bytes take precedence for CMS transforms.
    ///
    /// The CMS should parse `icc_profile` and use its TRC curves directly.
    /// Any CICP tag embedded inside the ICC should NOT override the TRC
    /// (i.e., `allow_use_cicp_transfer: false` in moxcms terms).
    ///
    /// Codecs should only set this when `icc_profile` is populated.
    /// Lenient consumers may fall back to `cicp` if ICC is absent.
    #[default]
    Icc,
    /// CICP codes take precedence for CMS transforms.
    ///
    /// The CMS should build a source profile from the `cicp` field
    /// (e.g., `ColorProfile::new_from_cicp()` in moxcms). The ICC profile,
    /// if present, is for backwards-compatible metadata roundtripping only.
    ///
    /// Codecs should only set this when `cicp` is populated.
    /// Lenient consumers may fall back to `icc_profile` if CICP is absent.
    Cicp,
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
    ///
    /// **Deprecated:** Codecs should populate only the authoritative field —
    /// use [`from_icc()`](Self::from_icc) or [`from_cicp()`](Self::from_cicp).
    /// Roundtrip metadata belongs on [`ColorOrigin`], not `ColorContext`.
    #[deprecated(
        since = "0.2.6",
        note = "use from_icc() or from_cicp(); roundtrip metadata belongs on ColorOrigin"
    )]
    pub fn from_icc_and_cicp(icc: impl Into<Arc<[u8]>>, cicp: Cicp) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: Some(cicp),
        }
    }

    /// Get a [`ColorProfileSource`] reference for CMS integration.
    ///
    /// Returns ICC if present, otherwise CICP. Returns `None` when neither
    /// is present.
    ///
    /// Codecs should populate only the authoritative field on `ColorContext`
    /// at decode time, using [`ColorAuthority`] on [`ColorOrigin`] to
    /// determine which. When only one field is set, this method returns it.
    pub fn as_profile_source(&self) -> Option<ColorProfileSource<'_>> {
        if let Some(icc) = self.icc.as_deref() {
            Some(ColorProfileSource::Icc(icc))
        } else {
            self.cicp.map(ColorProfileSource::Cicp)
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
    /// Which field a CMS should treat as authoritative for transforms.
    pub color_authority: ColorAuthority,
}

impl ColorOrigin {
    /// Create from an ICC profile.
    pub fn from_icc(icc: impl Into<Arc<[u8]>>) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: None,
            provenance: ColorProvenance::Icc,
            color_authority: ColorAuthority::Icc,
        }
    }

    /// Create from CICP parameters.
    pub fn from_cicp(cicp: Cicp) -> Self {
        Self {
            icc: None,
            cicp: Some(cicp),
            provenance: ColorProvenance::Cicp,
            color_authority: ColorAuthority::Cicp,
        }
    }

    /// Create from both ICC and CICP (e.g., AVIF with both).
    pub fn from_icc_and_cicp(icc: impl Into<Arc<[u8]>>, cicp: Cicp) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: Some(cicp),
            provenance: ColorProvenance::Cicp,
            color_authority: ColorAuthority::Cicp,
        }
    }

    /// Create from gAMA/cHRM chunks (no ICC or CICP).
    pub fn from_gama_chrm() -> Self {
        Self {
            icc: None,
            cicp: None,
            provenance: ColorProvenance::GamaChrm,
            color_authority: ColorAuthority::Icc,
        }
    }

    /// Create for assumed/default color (no explicit metadata).
    pub fn assumed() -> Self {
        Self {
            icc: None,
            cicp: None,
            provenance: ColorProvenance::Assumed,
            color_authority: ColorAuthority::Icc,
        }
    }

    /// Override the color authority.
    pub fn with_color_authority(mut self, authority: ColorAuthority) -> Self {
        self.color_authority = authority;
        self
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
    #[allow(deprecated)]
    fn color_context_profile_source_prefers_icc() {
        // When both are present, ICC wins (codecs should avoid this case)
        let ctx = ColorContext::from_icc_and_cicp(vec![1, 2, 3], Cicp::SRGB);
        let src = ctx.as_profile_source().unwrap();
        assert_eq!(src, ColorProfileSource::Icc(&[1, 2, 3]));
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
    #[allow(deprecated)]
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
