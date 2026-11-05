//! Color profile types for CMS integration.
//!
//! Provides a unified way to reference the source color space of decoded
//! pixels, suitable for passing to a CMS backend (e.g., moxcms, lcms2).
//!
//! [`ColorContext`] bundles ICC and CICP metadata for cheap sharing
//! (via `Arc`) across pixel slices and pipeline stages.
//! Current color state (transfer, primaries, alpha) is tracked on the
//! pixel descriptor itself, not as a separate enum.

use crate::PixelDescriptor;
use crate::cicp::Cicp;
use crate::hdr::DiffuseWhite;
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
            Self::Bt2020Pq => Some(Cicp::new(9, 16, 0, true)),
            Self::Bt2020Hlg => Some(Cicp::new(9, 18, 0, true)),
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
    /// - `Icc` — validated common-profile identification permitting matrix/TRC
    ///   substitution. Embedded CICP alone is insufficient; other ICCs need a CMS.
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
                // Only use the identification if matrix+TRC substitution is
                // safe — profiles with LUTs, non-Bradford chad, or Lab PCS
                // return MetadataOnly and need a full CMS.
                if let Some(id) = crate::icc::identify_common(icc_bytes) {
                    if id.valid_use == crate::icc::IdentificationUse::MatrixTrcSubstitution {
                        return Some((id.primaries, id.transfer));
                    }
                }
                // An embedded CICP tag does not prove the profile's TRCs,
                // LUTs or adaptation can be replaced. Preserve ICC authority;
                // callers choosing CICP explicitly pass ColorProfileSource::Cicp.
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
/// - **HEIC/HEIF**: `Icc` when ICC colr box present, `Cicp` otherwise
/// - **JPEG XL**: `Cicp` when enum encoding (`want_icc=false`), `Icc` when embedded ICC
///
/// Readers' measured behavior for conflicting signals is in
/// `docs/color-signal-precedence.md`.
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

impl ColorAuthority {
    /// Choose between the ICC and CICP form of a color source: the authoritative one
    /// when it is present, otherwise the other. Every reader of the authority should
    /// choose through this, so that they cannot disagree.
    #[inline]
    pub fn select<T>(self, icc: Option<T>, cicp: Option<T>) -> Option<T> {
        match self {
            Self::Icc => icc.or(cicp),
            Self::Cicp => cicp.or(icc),
        }
    }
}

/// Color space metadata for pixel data.
///
/// Bundles ICC profile bytes and/or CICP parameters into a single
/// shareable context. Carried via `Arc` on pixel slices and pipeline
/// sources so color metadata travels with pixel data without per-strip
/// cloning overhead.
// `Eq` retained: `DiffuseWhite` wraps `f32` but implements a bit-exact `Eq` (a
// luminance anchor is never NaN/-0.0), so adding the anchor field did not cost
// `ColorContext` its `Eq` — avoiding a removed-trait break vs 0.2.13.
// `#[non_exhaustive]` added so future HDR carrier fields (mastering display,
// content light level) extend this without another break. `Default` (all fields
// `None`) is the empty "no color signaling" context — the only way to construct
// one now that the struct literal is sealed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ColorContext {
    /// Raw ICC profile bytes.
    pub icc: Option<Arc<[u8]>>,
    /// CICP parameters (ITU-T H.273).
    pub cicp: Option<Cicp>,
    /// Absolute-luminance anchor: the cd/m² (nits) that a relative-linear
    /// sample value of `1.0` represents. `None` means unsignaled — consumers
    /// default to [`DiffuseWhite::BT2408`] (203 nits, the cross-vendor
    /// convention). Travels with the pixels so the HDR converter can map
    /// relative-linear light to absolute luminance for PQ/HLG encode and
    /// tone-mapping.
    pub diffuse_white: Option<DiffuseWhite>,
}

impl ColorContext {
    /// Create from an ICC profile.
    pub fn from_icc(icc: impl Into<Arc<[u8]>>) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: None,
            diffuse_white: None,
        }
    }

    /// Create from CICP parameters.
    pub fn from_cicp(cicp: Cicp) -> Self {
        Self {
            icc: None,
            cicp: Some(cicp),
            diffuse_white: None,
        }
    }

    /// Attach an absolute-luminance anchor — the nits that relative-linear
    /// `1.0` represents — returning the updated context.
    ///
    /// Set this on an HDR working buffer (e.g. after gain-map reconstruction,
    /// where the anchor is [`DiffuseWhite::BT2408`] = 203) so downstream
    /// conversion can map relative-linear light to absolute cd/m² for PQ/HLG
    /// encode and tone-mapping.
    #[must_use]
    pub fn with_diffuse_white(mut self, white: DiffuseWhite) -> Self {
        self.diffuse_white = Some(white);
        self
    }

    /// Set the [`Cicp`] colour signalling, returning the updated context.
    #[must_use]
    pub fn with_cicp(mut self, cicp: Cicp) -> Self {
        self.cicp = Some(cicp);
        self
    }

    /// Set the raw ICC profile, returning the updated context.
    #[must_use]
    pub fn with_icc(mut self, icc: impl Into<Arc<[u8]>>) -> Self {
        self.icc = Some(icc.into());
        self
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
/// [`PixelDescriptor`] for the current state
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
        assert_eq!(
            NamedProfile::Bt2020Pq.to_cicp(),
            Some(Cicp::new(9, 16, 0, true))
        );
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
        let ctx = ColorContext::from_icc(vec![1, 2, 3]).with_cicp(Cicp::SRGB);
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
        let ctx = ColorContext::from_icc(vec![1, 2]).with_cicp(Cicp::BT2100_PQ);
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
            diffuse_white: None,
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
    fn color_context_diffuse_white_defaults_none_and_round_trips() {
        // Unsignaled by default — consumers fall back to BT.2408 (203).
        assert_eq!(ColorContext::from_cicp(Cicp::BT2100_PQ).diffuse_white, None);
        assert_eq!(ColorContext::from_icc([0u8; 4]).diffuse_white, None);

        // The anchor round-trips through the builder, and is preserved by Clone.
        let ctx = ColorContext::from_cicp(Cicp::BT2100_PQ).with_diffuse_white(DiffuseWhite::BT2408);
        assert_eq!(ctx.diffuse_white, Some(DiffuseWhite::BT2408));
        assert_eq!(ctx.clone().diffuse_white, Some(DiffuseWhite::BT2408));

        // A custom anchor (e.g. a 100-nit SDR reference) carries through too.
        let custom =
            ColorContext::from_cicp(Cicp::BT2100_PQ).with_diffuse_white(DiffuseWhite::new(100.0));
        assert_eq!(custom.diffuse_white.map(DiffuseWhite::nits), Some(100.0));
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

/// Bytes aligned for every sample type a [`Color`] can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C, align(4))]
struct ColorSamples([u8; Color::MAX_BYTES]);

const _: () = assert!(core::mem::size_of::<Color>() == 24);

/// One color, and what its samples mean.
///
/// Holds one pixel of the descriptor's format, so `0.5` in linear Display P3 and `128` in sRGB
/// are different colors that both stay exact until converted. `zenpixels_convert::ColorConvertExt`
/// converts one to any other descriptor.
///
/// The bytes are aligned for every sample type, so casting [`as_bytes`](Self::as_bytes) to `f32`
/// or `u16` samples can't fail and doesn't copy. Comparison is bitwise: `0.0` and `-0.0` differ.
///
/// Colors with an ICC profile aren't representable, because a profile can't be `Copy`.
///
/// # Examples
///
/// ```
/// use zenpixels::{Color, ColorPrimaries, PixelDescriptor};
///
/// let red = Color::srgb8(255, 0, 0);
/// assert_eq!(red.as_bytes(), [255, 0, 0]);
///
/// // Display P3's red, in linear light.
/// let descriptor = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::DisplayP3);
/// let p3_red = Color::try_new(descriptor, bytemuck::bytes_of(&[1.0f32, 0.0, 0.0])).unwrap();
/// let samples: &[f32] = bytemuck::cast_slice(p3_red.as_bytes());
/// assert_eq!(samples, [1.0, 0.0, 0.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Color {
    samples: ColorSamples,
    descriptor: PixelDescriptor,
}

impl Color {
    /// The largest pixel a color holds: four `f32` samples.
    const MAX_BYTES: usize = 16;

    /// Whether one pixel fits, in bytes and in the storage's alignment.
    const fn fits(descriptor: PixelDescriptor) -> bool {
        descriptor.bytes_per_pixel() <= Self::MAX_BYTES && descriptor.min_alignment() <= 4
    }

    /// An opaque 8-bit sRGB color, like a CSS hex color.
    pub const fn srgb8(r: u8, g: u8, b: u8) -> Self {
        let mut samples = [0; Self::MAX_BYTES];
        samples[0] = r;
        samples[1] = g;
        samples[2] = b;
        Self {
            samples: ColorSamples(samples),
            descriptor: PixelDescriptor::RGB8_SRGB,
        }
    }

    /// Fully transparent 8-bit sRGB black, CSS's `transparent`.
    pub const TRANSPARENT: Self = Self::srgba8(0, 0, 0, 0);

    /// An 8-bit sRGB color with straight alpha, like a CSS `#rrggbbaa` color.
    pub const fn srgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        let mut samples = [0; Self::MAX_BYTES];
        samples[0] = r;
        samples[1] = g;
        samples[2] = b;
        samples[3] = a;
        Self {
            samples: ColorSamples(samples),
            descriptor: PixelDescriptor::RGBA8_SRGB,
        }
    }

    /// One pixel of `descriptor`'s format, as bytes in native sample order.
    ///
    /// # Errors
    ///
    /// [`BufferError::InvalidDimensions`](crate::buffer::BufferError::InvalidDimensions) when
    /// `pixel` isn't exactly one pixel, the format's pixels are wider than 16 bytes, or its samples
    /// need alignment above 4 bytes. [`AnyColor`] holds those.
    pub fn try_new(
        descriptor: PixelDescriptor,
        pixel: &[u8],
    ) -> Result<Self, crate::buffer::BufferError> {
        let bytes = descriptor.bytes_per_pixel();
        if pixel.len() != bytes || !Self::fits(descriptor) {
            return Err(crate::buffer::BufferError::InvalidDimensions);
        }
        let mut samples = [0; Self::MAX_BYTES];
        samples[..bytes].copy_from_slice(pixel);
        Ok(Self {
            samples: ColorSamples(samples),
            descriptor,
        })
    }

    /// What the samples mean.
    pub const fn descriptor(&self) -> PixelDescriptor {
        self.descriptor
    }

    /// The pixel, aligned to 4 bytes. Sample types wider than that need [`AnyColor`].
    pub fn as_bytes(&self) -> &[u8] {
        &self.samples.0[..self.descriptor.bytes_per_pixel()]
    }
}

/// A color in any format, including pixels wider than [`Color`] holds.
///
/// [`Color`] is `Copy` and holds up to 16 bytes, which covers every format at 8 and 16
/// bits and everything with four `f32` samples. It has to be `Copy` to live in
/// `ConvertOptions`. This type also holds wider pixels, such as RGBA in `f64`, on the
/// heap, and costs no more space. Comparison is bitwise.
///
/// ```
/// use zenpixels::{AnyColor, ChannelLayout, ChannelType, Color, PixelDescriptor, PixelFormat};
///
/// let small = AnyColor::from(Color::srgb8(255, 0, 0));
/// assert_eq!(small.as_bytes(), [255, 0, 0]);
/// assert!(small.as_color().is_some());
///
/// let format = PixelFormat::create_from(ChannelType::F64, ChannelLayout::Gray).unwrap();
/// let descriptor = PixelDescriptor::from_pixel_format(format);
/// let wide = AnyColor::try_new(descriptor, &0.5f64.to_ne_bytes()).unwrap();
/// let samples: &[f64] = bytemuck::cast_slice(wide.as_bytes());
/// assert_eq!(samples, [0.5]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AnyColor(AnyColorRepr);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum AnyColorRepr {
    Compact(Color),
    Wide(alloc::boxed::Box<WideColor>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WideColor {
    descriptor: PixelDescriptor,
    /// Eight-byte words, so the bytes are aligned for every sample type.
    words: alloc::boxed::Box<[u64]>,
}

const _: () = assert!(core::mem::size_of::<AnyColor>() == core::mem::size_of::<Color>());

impl AnyColor {
    /// One pixel of `descriptor`'s format, as bytes in native sample order.
    ///
    /// # Errors
    ///
    /// [`BufferError::InvalidDimensions`](crate::buffer::BufferError::InvalidDimensions) when
    /// `pixel` isn't exactly one pixel, and
    /// [`BufferError::AllocationFailed`](crate::buffer::BufferError::AllocationFailed).
    pub fn try_new(
        descriptor: PixelDescriptor,
        pixel: &[u8],
    ) -> Result<Self, crate::buffer::BufferError> {
        use crate::buffer::BufferError;
        let bytes = descriptor.bytes_per_pixel();
        if pixel.len() != bytes {
            return Err(BufferError::InvalidDimensions);
        }
        if Color::fits(descriptor) {
            return Color::try_new(descriptor, pixel).map(Self::from);
        }
        let mut words = alloc::vec::Vec::new();
        words
            .try_reserve_exact(bytes.div_ceil(8))
            .map_err(|_| BufferError::AllocationFailed)?;
        words.resize(bytes.div_ceil(8), 0u64);
        bytemuck::cast_slice_mut::<u64, u8>(&mut words)[..bytes].copy_from_slice(pixel);
        let wide = WideColor {
            descriptor,
            words: words.into_boxed_slice(),
        };
        Ok(Self(AnyColorRepr::Wide(alloc::boxed::Box::new(wide))))
    }

    /// What the samples mean.
    pub fn descriptor(&self) -> PixelDescriptor {
        match &self.0 {
            AnyColorRepr::Compact(color) => color.descriptor(),
            AnyColorRepr::Wide(wide) => wide.descriptor,
        }
    }

    /// The pixel, aligned for its sample type.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.0 {
            AnyColorRepr::Compact(color) => color.as_bytes(),
            AnyColorRepr::Wide(wide) => {
                &bytemuck::cast_slice(&wide.words)[..wide.descriptor.bytes_per_pixel()]
            }
        }
    }

    /// The compact form, or `None` when the pixel is wider than [`Color`] holds.
    pub fn as_color(&self) -> Option<Color> {
        match &self.0 {
            AnyColorRepr::Compact(color) => Some(*color),
            AnyColorRepr::Wide(_) => None,
        }
    }
}

impl From<Color> for AnyColor {
    fn from(color: Color) -> Self {
        Self(AnyColorRepr::Compact(color))
    }
}
