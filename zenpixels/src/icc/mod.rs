//! Lightweight ICC profile inspection and identification.
//!
//! Two capabilities, no dependencies beyond `core`:
//!
//! - **CICP extraction**: [`Cicp::from_icc`] and [`CicpTag::from_icc`] read the
//!   `cicp` tag from ICC v4.4+ profiles (~100ns, no allocation), including the
//!   44-byte custom-primaries `cicpType` extension (ICC amendment approved
//!   2026-10-19).
//!
//! - **Hash-based identification**: [`IccIdentification::from_icc`] recognizes
//!   retained profile fingerprints (sRGB, Display P3, BT.2020, Adobe RGB,
//!   grayscale) via paired normalized FNV-1a64/MurmurHash3_x86_32 lookup.
//!
//! The hash table covers profiles found in a web corpus of 55,539 spidered
//! images, plus Compact-ICC-Profiles, skcms, ICC.org, colord, Ghostscript,
//! moxcms, and zenpixels-convert bundled profiles. Metadata-only header
//! fields are zeroed before hashing to collapse functionally identical
//! profiles across ICC v2.0–v4.4 and v5/iccMAX. Stored curve-fit errors concern
//! red/gray TRCs; intent permissions carry separate historical qualification.
//! Neither establishes bit-exact full-transform equivalence. Exact original
//! coverage is documented in `docs/icc-retention-audit.md`; complete historical
//! tolerance requalification remains separate.
//!
//! # Example
//!
//! ```
//! use zenpixels::icc::{IccIdentification, IdentificationUse};
//!
//! # let icc_bytes: &[u8] = &[];
//! if let Some(id) = IccIdentification::from_icc(icc_bytes) {
//!     if id.valid_use == IdentificationUse::MatrixTrcSubstitution {
//!         // Eligible for matrix+TRC under the caller's approximation policy.
//!     }
//!     // id.primaries and id.transfer are always available for metadata
//! }
//! ```

pub use crate::cicp::CicpTag;
pub use crate::color::{IccValidationError, validate_icc_structure};
use crate::{Cicp, HlgMode, IccColorSpace, Primaries, Transfer};

impl Cicp {
    /// Read the `cicp` tag of an ICC profile (ICC v4.4+), or `None` when absent.
    #[must_use]
    pub fn from_icc(icc_bytes: &[u8]) -> Option<Self> {
        extract_cicp(icc_bytes)
    }
}

impl IccColorSpace {
    /// The data color space from an ICC profile's header (bytes 16 to 19), or
    /// `None` when the profile is too short or has an unrecognized signature.
    #[must_use]
    pub fn from_icc(icc_bytes: &[u8]) -> Option<Self> {
        profile_color_space(icc_bytes)
    }
}

// ── ICC header layout (ITU-T / ISO 15076-1) ──────────────────────────────

/// Minimum valid ICC profile size: 128-byte header + 4-byte tag count.
const ICC_MIN_SIZE: usize = 132;
/// Offset of the `'acsp'` magic signature in the ICC header.
const ICC_SIGNATURE_OFFSET: usize = 36;
/// Offset of the tag count (u32 BE) following the 128-byte header.
const ICC_TAG_COUNT_OFFSET: usize = 128;
/// Start of the tag table (immediately after the tag count).
const ICC_TAG_TABLE_OFFSET: usize = 132;
/// Size of each tag table entry: 4 (sig) + 4 (offset) + 4 (size).
const ICC_TAG_ENTRY_SIZE: usize = 12;
/// Maximum tag count we'll scan (prevents DoS from malformed profiles).
const ICC_MAX_TAG_COUNT: usize = 200;

/// ICC header byte ranges that are metadata-only (not color-critical).
/// Zeroed before hashing to collapse functionally identical profiles.
///
/// Per ICC spec v2.0–v5/iccMAX, these fields never affect colorimetry:
/// - `CMM_TYPE`:     preferred CMM (advisory hint)
/// - `DATE_TIME`:    profile creation date/time
/// - `PLATFORM`:     primary platform (APPL/MSFT/etc.)
/// - `DEVICE`:       device manufacturer + model
/// - `CREATOR_ID`:   profile creator + profile ID (reserved in v2, MD5 in v4)
const ICC_CMM_TYPE: core::ops::Range<usize> = 4..8;
const ICC_DATE_TIME: core::ops::Range<usize> = 24..36;
const ICC_PLATFORM: core::ops::Range<usize> = 40..44;
const ICC_DEVICE: core::ops::Range<usize> = 48..56;
const ICC_CREATOR_ID: core::ops::Range<usize> = 80..100;
/// Number of header bytes that need conditional zeroing.
const ICC_HEADER_NORMALIZE_END: usize = 100;

const fn is_metadata_field(i: usize) -> bool {
    (i >= ICC_CMM_TYPE.start && i < ICC_CMM_TYPE.end)
        || (i >= ICC_DATE_TIME.start && i < ICC_DATE_TIME.end)
        || (i >= ICC_PLATFORM.start && i < ICC_PLATFORM.end)
        || (i >= ICC_DEVICE.start && i < ICC_DEVICE.end)
        || (i >= ICC_CREATOR_ID.start && i < ICC_CREATOR_ID.end)
}

// ── CICP extraction ──────────────────────────────────────────────────────

/// Extract CICP (Coding-Independent Code Points) from an ICC profile's tag table.
///
/// Scans the ICC tag table for a `cicp` tag (ICC v4.4+, 12 bytes) and returns
/// a [`Cicp`] if found. Returns `None` for ICC v2 profiles (which never contain
/// cicp tags), profiles without a cicp tag, or malformed input.
///
/// This is a lightweight operation (~100ns) that reads only the 128-byte header
/// and tag table entries — no full profile parse required.
///
/// Read leniency: the tag is recovered regardless of the profile's data colour
/// space. ICC.1 restricts `cicpTag` to RGB/YCbCr/XYZ, so a GRAY/CMYK profile
/// carrying one is non-conformant — but a third-party encoder may still emit it,
/// and we honor it on input rather than drop recoverable colour signaling
/// (Postel's law). zenpixels itself never *emits* a cicp tag on a gray profile
/// (see the GRAY-class synthesis in `zenpixels-convert`); this leniency only
/// affects what we accept, never what we produce.
pub(crate) fn extract_cicp(data: &[u8]) -> Option<Cicp> {
    crate::cicp::extract_cicp_tag(data).map(CicpTag::cicp)
}

// ── Profile inspection ───────────────────────────────────────────────────

/// Profile features that can cause a CMS to produce output different from
/// what a pure matrix+TRC conversion would give.
///
/// A profile with any of these features should ideally be handled by a full
/// CMS (moxcms, lcms2) rather than identified-and-approximated via matrix math.
#[allow(dead_code)] // used in tests; will be used by zenpixels-convert CMS bypass
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(crate) struct ProfileFeatures {
    /// PCS is Lab instead of XYZ. Matrix math assumes XYZ.
    pub(crate) pcs_is_lab: bool,
    /// Has `chad` (chromatic adaptation) tag.
    pub(crate) has_chad: bool,
    /// `chad` tag is Bradford (within tolerance). Only meaningful if `has_chad`.
    pub(crate) chad_is_bradford: bool,
    /// Has `A2B0` LUT (default/relative-colorimetric device→PCS). A CMS
    /// typically prefers this over colorants when present.
    pub(crate) has_a2b0: bool,
    /// Has `A2B1` LUT (perceptual device→PCS, with gamut mapping).
    pub(crate) has_a2b1: bool,
    /// Has `A2B2` LUT (saturation device→PCS).
    pub(crate) has_a2b2: bool,
    /// Has `B2A0` LUT (default PCS→device).
    pub(crate) has_b2a0: bool,
    /// Has `B2A1` LUT (perceptual PCS→device).
    pub(crate) has_b2a1: bool,
    /// Has `B2A2` LUT (saturation PCS→device).
    pub(crate) has_b2a2: bool,
    /// Has `D2B0..3` or `B2D0..3` float multi-processing element tags (ICC.1:2022 §8.10.2).
    pub(crate) has_mpe_lut: bool,
    /// Has matrix-shaper tags (rXYZ + gXYZ + bXYZ + rTRC + gTRC + bTRC).
    pub(crate) has_matrix_shaper: bool,
}

impl ProfileFeatures {
    /// Whether a CMS would produce identical output to our matrix+TRC math.
    ///
    /// True when: PCS is XYZ, chad (if present) is Bradford, no LUT tags
    /// that a CMS would prefer. Matrix-shaper tags must be present.
    #[inline]
    #[allow(dead_code)] // will be used by zenpixels-convert CMS bypass
    pub(crate) fn is_safe_matrix_shaper(&self) -> bool {
        self.has_matrix_shaper
            && !self.pcs_is_lab
            && !self.has_a2b0
            && !self.has_a2b1
            && !self.has_a2b2
            && !self.has_b2a0
            && !self.has_b2a1
            && !self.has_b2a2
            && !self.has_mpe_lut
            && (!self.has_chad || self.chad_is_bradford)
    }
}

/// Bradford chromatic adaptation matrix (D65→D50 direction, ICC convention).
/// Used by ICC `chad` tag.
#[allow(dead_code)] // used by inspect_profile, kept for CMS bypass path
const BRADFORD_CHAD_D65_TO_D50: [f64; 9] = [
    1.0478, 0.0229, -0.0501, 0.0295, 0.9905, -0.0171, -0.0092, 0.0151, 0.7517,
];
/// Tolerance for `chad` matrix comparison (s15Fixed16 quantization).
#[allow(dead_code)] // used by inspect_profile, kept for CMS bypass path
const CHAD_TOL: f64 = 0.005;

/// Inspect an ICC profile and return which features it uses.
///
/// Use [`ProfileFeatures::is_safe_matrix_shaper`] to decide whether
/// matrix+TRC approximation via [`IccIdentification::from_icc`] is equivalent to a
/// full CMS's output.
///
/// Returns `None` if the bytes aren't a valid ICC profile.
#[allow(dead_code)] // used in tests; will be used by zenpixels-convert CMS bypass
pub(crate) fn inspect_profile(data: &[u8]) -> Option<ProfileFeatures> {
    if data.len() < ICC_MIN_SIZE {
        return None;
    }
    if data.get(ICC_SIGNATURE_OFFSET..ICC_SIGNATURE_OFFSET + 4)? != b"acsp" {
        return None;
    }

    // PCS: bytes 20..24 in header. "XYZ " vs "Lab "
    let mut feat = ProfileFeatures {
        pcs_is_lab: data.get(20..24)? == b"Lab ",
        ..ProfileFeatures::default()
    };

    let tag_count = u32::from_be_bytes(
        data[ICC_TAG_COUNT_OFFSET..ICC_TAG_COUNT_OFFSET + 4]
            .try_into()
            .ok()?,
    ) as usize;
    let tag_count = tag_count.min(ICC_MAX_TAG_COUNT);

    let mut has_rxyz = false;
    let mut has_gxyz = false;
    let mut has_bxyz = false;
    let mut has_rtrc = false;
    let mut has_gtrc = false;
    let mut has_btrc = false;
    let mut chad_span: Option<(usize, usize)> = None;

    for i in 0..tag_count {
        let entry_offset = ICC_TAG_TABLE_OFFSET + i * ICC_TAG_ENTRY_SIZE;
        let entry = data.get(entry_offset..entry_offset + ICC_TAG_ENTRY_SIZE)?;
        let sig = &entry[..4];
        let d_off = u32::from_be_bytes(entry[4..8].try_into().ok()?) as usize;
        let d_size = u32::from_be_bytes(entry[8..12].try_into().ok()?) as usize;
        match sig {
            b"rXYZ" => has_rxyz = true,
            b"gXYZ" => has_gxyz = true,
            b"bXYZ" => has_bxyz = true,
            b"rTRC" => has_rtrc = true,
            b"gTRC" => has_gtrc = true,
            b"bTRC" => has_btrc = true,
            b"A2B0" => feat.has_a2b0 = true,
            b"A2B1" => feat.has_a2b1 = true,
            b"A2B2" => feat.has_a2b2 = true,
            b"B2A0" => feat.has_b2a0 = true,
            b"B2A1" => feat.has_b2a1 = true,
            b"B2A2" => feat.has_b2a2 = true,
            b"D2B0" | b"D2B1" | b"D2B2" | b"D2B3" | b"B2D0" | b"B2D1" | b"B2D2" | b"B2D3" => {
                feat.has_mpe_lut = true;
            }
            b"chad" => chad_span = Some((d_off, d_size)),
            _ => {}
        }
    }

    feat.has_matrix_shaper = has_rxyz && has_gxyz && has_bxyz && has_rtrc && has_gtrc && has_btrc;

    // Parse chad matrix if present
    if let Some((off, d_size)) = chad_span {
        feat.has_chad = true;
        // chad tag: "sf32" (signature) + 4 reserved + 9 × s15Fixed16 (44 bytes minimum)
        if d_size >= 44
            && off.checked_add(d_size).is_some_and(|end| end <= data.len())
            && data.get(off..off + 8)? == b"sf32\0\0\0\0"
        {
            let mut m = [0.0f64; 9];
            let mut ok = true;
            for (i, slot) in m.iter_mut().enumerate() {
                let o = off + 8 + i * 4;
                if let Ok(bytes) = data[o..o + 4].try_into() {
                    *slot = i32::from_be_bytes(bytes) as f64 / 65536.0;
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                let mut max_diff = 0.0f64;
                for (mi, bi) in m.iter().zip(BRADFORD_CHAD_D65_TO_D50.iter()) {
                    let d = (mi - bi).abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                }
                feat.chad_is_bradford = max_diff < CHAD_TOL;
            }
        }
    }

    Some(feat)
}

// ── Return type ──────────────────────────────────────────────────────────

/// Result of identifying a well-known ICC profile.
///
/// Carries the recognized color primaries and transfer function.
/// These may or may not have CICP equivalents — [`AdobeRgb`](Primaries::AdobeRgb)
/// does not.
///
/// Use [`to_cicp`](Self::to_cicp) to convert to CICP codes when available.
/// Check [`valid_use`](Self::valid_use) before deciding whether to skip a CMS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct IccIdentification {
    /// Recognized color primaries.
    pub primaries: Primaries,
    /// Recognized transfer function.
    pub transfer: Transfer,
    /// What this identification can be used for.
    pub valid_use: IdentificationUse,
}

/// What operations an [`IccIdentification`] supports.
///
/// Always check this before deciding whether to skip a CMS — a profile
/// can be *recognized* (primaries/transfer known) without being *safe*
/// for fast-path substitution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IdentificationUse {
    /// Primaries and transfer are known, but the profile's structure may
    /// require a full CMS for accurate conversion (e.g., LUT-based
    /// profiles, non-Bradford chromatic adaptation, Lab PCS). Use for
    /// metadata, format negotiation, and display — not for pixel math.
    MetadataOnly,
    /// Matrix+TRC substitution produces output equivalent to a full CMS
    /// within the tolerance used for identification. Safe to use for
    /// pixel conversion without a CMS backend.
    MatrixTrcSubstitution,
}

impl IccIdentification {
    /// Recognize `icc_bytes` as a well-known profile from its normalized hash.
    ///
    /// `None` means not in the table, which says nothing about the profile. A
    /// match is an approximate color identification, as described on this type.
    /// For sRGB use [`is_common_srgb`](Self::is_common_srgb).
    #[must_use]
    pub fn from_icc(icc_bytes: &[u8]) -> Option<Self> {
        identify_common(icc_bytes)
    }

    /// Recognize `icc_bytes` only when it is safe for matrix+TRC substitution
    /// across all rendering intents (`IdentificationUse::MatrixTrcSubstitution`).
    ///
    /// Consults only the compact substitutable table so callers that never need
    /// `MetadataOnly` entries dead-strip that table.
    #[must_use]
    pub fn from_icc_substitutable(icc_bytes: &[u8]) -> Option<Self> {
        identify_substitutable(icc_bytes)
    }

    /// Recognize `icc_bytes` when matrix+TRC substitution is safe for the
    /// requested [`RenderingIntent`](crate::RenderingIntent).
    ///
    /// Always returns `None` for
    /// [`RenderingIntent::AbsoluteColorimetric`](crate::RenderingIntent::AbsoluteColorimetric)
    /// because D50-adapted relative matrix math cannot undo D65→D50 adaptation.
    /// When `exact_only` is `true`, only entries within ±1 u16 TRC tolerance
    /// (parametric v4 profiles) match.
    #[must_use]
    pub fn from_icc_for_intent(
        icc_bytes: &[u8],
        intent: crate::RenderingIntent,
        exact_only: bool,
    ) -> Option<Self> {
        let use_for = match intent {
            crate::RenderingIntent::Perceptual => CoalesceForUse::Perceptual,
            crate::RenderingIntent::RelativeColorimetric => CoalesceForUse::RelativeColorimetric,
            crate::RenderingIntent::Saturation => CoalesceForUse::Saturation,
            crate::RenderingIntent::AbsoluteColorimetric => CoalesceForUse::AbsoluteColorimetric,
        };
        let tol = if exact_only {
            Tolerance::Exact
        } else {
            Tolerance::Intent
        };
        identify_common_for(icc_bytes, tol, use_for)
    }

    /// Whether `icc_bytes` is a well-known sRGB profile, for skipping a CMS.
    ///
    /// `false` means not recognized, which doesn't mean it isn't sRGB.
    #[inline]
    #[must_use]
    pub fn is_common_srgb(icc_bytes: &[u8]) -> bool {
        identify_substitutable(icc_bytes).is_some_and(|id| id.is_srgb())
    }

    /// Create a new identification result.
    #[inline]
    pub(crate) fn new(
        primaries: Primaries,
        transfer: Transfer,
        valid_use: IdentificationUse,
    ) -> Self {
        Self {
            primaries,
            transfer,
            valid_use,
        }
    }

    /// Convert to CICP codes if both primaries and transfer have CICP equivalents.
    ///
    /// Returns `None` for non-CICP color spaces (Adobe RGB, etc.).
    #[inline]
    pub fn to_cicp(&self) -> Option<crate::Cicp> {
        let cp = self.primaries.to_cicp()?;
        let tc = self.transfer.to_cicp()?;
        Some(crate::Cicp::new(cp, tc, 0, true))
    }

    /// Whether this represents sRGB (BT.709 primaries + sRGB transfer).
    #[inline]
    pub fn is_srgb(&self) -> bool {
        matches!(self.primaries, Primaries::Bt709) && matches!(self.transfer, Transfer::Srgb)
    }
}

// ── Tolerance ────────────────────────────────────────────────────────────

/// TRC match tolerance for ICC profile identification.
///
/// Each entry stores a historical red/gray curve-fit error in u16 space
/// (0–65535). The generator evaluates that curve against reference EOTFs for
/// all 65536 input values; this is not a complete CMS transform measurement.
/// This tolerance controls which entries are accepted, without requalifying
/// the retained original at runtime.
///
/// A small curve error does not guarantee equal quantized output: values
/// near a rounding midpoint can differ. This is identification tolerance,
/// not proof of bit-exact CMS substitution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[must_use]
#[allow(dead_code)] // all levels kept for future fine-grained identification
pub(crate) enum Tolerance {
    /// ±1 in u16 — parametric v4 profiles only.
    Exact = 1,
    /// ±3 in u16 — includes v2-magic parametric approximations.
    Precise = 3,
    /// ±13 in u16 — includes LUT profiles and iPhone P3.
    Approximate = 13,
    /// ±56 in u16 — historical identification tolerance.
    Intent = 56,
}

// ── Intent-safety flags ──────────────────────────────────────────────────

/// Intent-safety flag: matrix+TRC math matches a CMS's output for the
/// relative/absolute colorimetric intent.
///
/// Set when PCS is XYZ, `chad` (if present) is Bradford, there is no `A2B0`
/// or `B2A0` LUT, and the matrix-shaper tags are complete.
pub(crate) const INTENT_COLORIMETRIC_SAFE: u8 = 1 << 0;
/// Intent-safety flag: matrix+TRC math matches a CMS's output for the
/// perceptual intent (in addition to [`INTENT_COLORIMETRIC_SAFE`]).
///
/// Additionally requires no `A2B1` or `B2A1` LUT.
pub(crate) const INTENT_PERCEPTUAL_SAFE: u8 = 1 << 1;
/// Intent-safety flag: matrix+TRC math matches a CMS's output for the
/// saturation intent (in addition to [`INTENT_COLORIMETRIC_SAFE`]).
///
/// Additionally requires no `A2B2` or `B2A2` LUT.
pub(crate) const INTENT_SATURATION_SAFE: u8 = 1 << 2;

/// ICC rendering intent — controls which intent-safety flags must be set
/// on a table entry for [`identify_common_for`] to return a match.
///
/// Names follow the ICC v4.4 specification (section 7.2.15) and CICP/moxcms
/// conventions. See also [`Tolerance`] for TRC error tolerance control.
#[allow(dead_code)] // intent dispatch kept for zenpixels-convert CMS bypass
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(crate) enum CoalesceForUse {
    /// All four ICC rendering intents must be safe.
    AnyIntent,
    /// Relative colorimetric — CMS uses colorants directly with white-point
    /// adaptation. The most common default for display-to-display workflows.
    /// Equivalent to absolute colorimetric for matrix-shaper math (we share
    /// one safety bit for both).
    RelativeColorimetric,
    /// Absolute colorimetric — preserves source white point literally.
    /// Same intent-safety requirements as relative colorimetric for
    /// matrix-shaper profiles.
    AbsoluteColorimetric,
    /// Perceptual rendering — CMS uses A2B1/B2A1 LUT for gamut mapping
    /// when present. Matrix math is only equivalent when no perceptual
    /// LUT exists in the profile.
    Perceptual,
    /// Saturation rendering — vivid business graphics. Matrix math is
    /// only equivalent when no A2B2/B2A2 LUT exists.
    Saturation,
}

impl CoalesceForUse {
    /// The intent-safety mask required for a profile to match this use.
    #[inline]
    const fn required_mask(self) -> u8 {
        match self {
            Self::AnyIntent => {
                INTENT_COLORIMETRIC_SAFE | INTENT_PERCEPTUAL_SAFE | INTENT_SATURATION_SAFE
            }
            Self::RelativeColorimetric | Self::AbsoluteColorimetric => INTENT_COLORIMETRIC_SAFE,
            Self::Perceptual => INTENT_PERCEPTUAL_SAFE,
            Self::Saturation => INTENT_SATURATION_SAFE,
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────

/// Identify a well-known ICC profile by normalized hash lookup.
///
/// Computes both normalized fingerprints (metadata fields zeroed) and
/// checks against tables of known RGB and grayscale ICC profiles.
///
/// Returns `Some(IccIdentification)` for recognized profiles, `None` for
/// unknown ones. Check [`IccIdentification::valid_use`] before deciding
/// whether to skip a CMS — a recognized profile may still require one.
///
/// Grayscale profiles return [`Bt709`](Primaries::Bt709) primaries
/// (grayscale has no gamut, but D65 white point is assumed).
///
/// ~100ns. For the long tail of vendor/monitor profiles, use structural
/// analysis via a CMS backend.
pub(crate) fn identify_common(icc_bytes: &[u8]) -> Option<IccIdentification> {
    identify_common_at(icc_bytes, Tolerance::Intent)
}

/// Identify a well-known ICC profile that is safe for unconditional
/// matrix+TRC substitution (`IdentificationUse::MatrixTrcSubstitution`).
///
/// Only consults the compact substitutable profile table (`SUBSTITUTABLE_*`),
/// allowing callers that never use `MetadataOnly` entries (such as
/// [`IccIdentification::from_icc_substitutable`] and
/// [`IccIdentification::is_common_srgb`]) to dead-strip the metadata-only table.
#[inline(never)]
pub(crate) fn identify_substitutable(icc_bytes: &[u8]) -> Option<IccIdentification> {
    let hash = normalized_hash(icc_bytes);
    let meta = lookup_substitutable_hash(hash)?;
    if unpack_err(meta) <= Tolerance::Intent as u8 && unpack_mask(meta) == Safe::AnyIntent {
        Some(IccIdentification::new(
            unpack_cp(meta),
            unpack_tf(meta),
            IdentificationUse::MatrixTrcSubstitution,
        ))
    } else {
        None
    }
}

/// Internal identification with configurable tolerance.
fn identify_common_at(icc_bytes: &[u8], tolerance: Tolerance) -> Option<IccIdentification> {
    let hash = normalized_hash(icc_bytes);
    let meta = lookup_substitutable_hash(hash).or_else(|| lookup_metadata_only_hash(hash))?;
    if unpack_err(meta) <= tolerance as u8 {
        Some(IccIdentification::new(
            unpack_cp(meta),
            unpack_tf(meta),
            use_from_mask(unpack_mask(meta)),
        ))
    } else {
        None
    }
}

/// Map an intent-safety mask to an [`IdentificationUse`].
#[inline]
fn use_from_mask(mask: u8) -> IdentificationUse {
    const ALL: u8 = INTENT_COLORIMETRIC_SAFE | INTENT_PERCEPTUAL_SAFE | INTENT_SATURATION_SAFE;
    if mask == ALL {
        IdentificationUse::MatrixTrcSubstitution
    } else {
        IdentificationUse::MetadataOnly
    }
}

/// Identify a profile only when matrix+TRC math is equivalent to a CMS's
/// output for the specified intent.
///
/// Each table entry carries a precomputed intent-safety mask derived at
/// table-generation time from the ICC profile's structural features
/// (PCS type, `chad` matrix, A2B/B2A LUT presence, matrix-shaper
/// completeness). Entries whose mask is missing any bit required by
/// `use_for` are rejected.
///
/// Use this when you need byte-exact equivalence with a full CMS for a
/// specific rendering intent. Use [`IccIdentification::from_icc`] for the common
/// colorimetric-intent case.
///
/// Returns `None` if the profile isn't in the table, its measured u16 error
/// exceeds `tolerance`, or it isn't safe for the requested intent.
#[allow(dead_code)] // will be used by zenpixels-convert intent-aware CMS bypass
pub(crate) fn identify_common_for(
    icc_bytes: &[u8],
    tolerance: Tolerance,
    use_for: CoalesceForUse,
) -> Option<IccIdentification> {
    // All table entries are D65 profiles; ICC-Absolute colorimetric (ICC.1 §6.3.2.3)
    // undoes D65->D50 adaptation in PCS, so D50-adapted relative matrix math is never equivalent.
    if matches!(use_for, CoalesceForUse::AbsoluteColorimetric) {
        return None;
    }
    let hash = normalized_hash(icc_bytes);
    let required = use_for.required_mask();

    // Metadata-only entries have intent_mask == 0 and can never satisfy a
    // non-zero `required` mask, so only the substitutable table is searched.
    let meta = lookup_substitutable_hash(hash)?;
    if unpack_err(meta) <= tolerance as u8 && (unpack_mask(meta) & required) == required {
        return Some(IccIdentification::new(
            unpack_cp(meta),
            unpack_tf(meta),
            IdentificationUse::MatrixTrcSubstitution,
        ));
    }

    None
}

/// Read the ICC profile's data color space from the header (bytes 16–19).
///
/// Returns the [`IccColorSpace`] corresponding to the four-byte signature
/// at offset 16 in the ICC header, or `None` if the profile is too short
/// or uses an unrecognized color space.
#[must_use]
pub fn profile_color_space(icc_bytes: &[u8]) -> Option<IccColorSpace> {
    if icc_bytes.len() < 20 {
        return None;
    }
    match &icc_bytes[16..20] {
        b"RGB " => Some(IccColorSpace::Rgb),
        b"GRAY" => Some(IccColorSpace::Gray),
        b"CMYK" => Some(IccColorSpace::Cmyk),
        b"Lab " => Some(IccColorSpace::Lab),
        b"XYZ " => Some(IccColorSpace::Xyz),
        b"YCbr" => Some(IccColorSpace::Ycbcr),
        _ => None,
    }
}

mod murmur3;
use self::murmur3 as murmur;
#[cfg(test)]
mod murmur3_tests;

// ── Hash function ────────────────────────────────────────────────────────

/// Pair of normalized ICC fingerprints: FNV-1a64 and MurmurHash3_x86_32.
///
/// Both components must match for identification or known-profile normalization.
/// Murmur3 uses seed `3_242_157_231` and explicit little-endian words on every
/// target. The return type changed from `u64` in the authorized 0.3.1 break.
///
/// Zeroes metadata-only header fields before hashing so that functionally
/// identical profiles (same colorants + TRC) produce the same hash even if
/// they differ in creation date, CMM, platform, creator, or profile ID.
///
/// Zeroed ranges (all non-colorimetric per ICC spec v2.0–v5/iccMAX):
/// - bytes  4– 7: preferred CMM type (advisory hint)
/// - bytes 24–35: creation date/time (metadata)
/// - bytes 40–43: primary platform (advisory hint)
/// - bytes 48–55: device manufacturer + device model (identification)
/// - bytes 80–99: profile creator + profile ID (reserved in v2, MD5 in v4)
///
/// This is the fingerprint used by the committed identification tables and by
/// `zenpixels-convert`'s explicit known-profile normalization. It is not a
/// cryptographic digest, validation, or a proof of colorimetric equivalence.
/// Bytes outside the listed header fields (including every tag) are hashed.
/// Matching a table's approximate color identification does not make profiles
/// interchangeable without a pixel conversion.
pub const fn normalized_hash(data: &[u8]) -> (u64, u32) {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    let mut header = [0u8; ICC_HEADER_NORMALIZE_END];

    // Phase 1: header bytes — zero metadata fields.
    let header_len = if data.len() < ICC_HEADER_NORMALIZE_END {
        data.len()
    } else {
        ICC_HEADER_NORMALIZE_END
    };
    let mut i = 0;
    while i < header_len {
        let b = if is_metadata_field(i) { 0u8 } else { data[i] };
        header[i] = b;
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }

    // Phase 2: remaining bytes — straight hash, no conditionals.
    while i < data.len() {
        hash ^= data[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    // Fixed Quickwit seed and x86_32 variant, including on 64-bit targets.
    const SEED: u32 = 3_242_157_231;
    let normalized_header = header.split_at(header_len).0;
    let murmur = if data.len() == header_len {
        murmur::murmur3_32(normalized_header, SEED)
    } else {
        let mut state = murmur::Murmur3_32::with_seed(SEED);
        state.update(normalized_header);
        state.update(data.split_at(header_len).1);
        state.digest()
    };
    (hash, murmur)
}

/// Compare every byte except the metadata-only header fields ignored by
/// [`normalized_hash`]. No allocation, profile parsing or hashing is performed.
///
/// This establishes equality under that exact normalization, including every
/// tag and rendering-intent byte. It does not establish that either input is a
/// valid ICC profile or that different transforms are colorimetrically close.
/// Matching fingerprints alone cannot establish equality; callers replacing a
/// profile can use them as a filter and then compare the retained bytes here.
/// Inputs of different lengths never compare equal; identical empty inputs do.
///
/// Used by `zenpixels-convert::OutputSignaling::normalize_known_icc` before
/// replacing a profile with its bundled canonical bytes.
#[must_use]
pub const fn normalized_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut i = 0;
    while i < left.len() {
        if !is_metadata_field(i) && left[i] != right[i] {
            return false;
        }
        i += 1;
    }
    true
}

// ── Hash tables ──────────────────────────────────────────────────────────
//
// Tables store enum variants directly — no code-to-enum mapping needed.
// Shorthand aliases keep the .inc files readable.

use Primaries as CP;
use Transfer as TF;

type ProfileHash = (u64, u32);

/// Intent-safety bitfield aliases used in the generated `.inc` tables.
///
/// `AnyIntent` (all three bits set) is the common case: matrix+TRC math
/// matches a CMS for every rendering intent, so the entry is safe to
/// substitute unconditionally. `IdOnly` means the profile is recognized
/// but substitution isn't safe for any intent — use it for identification,
/// not for math. Partial combinations name each possible subset; they
/// occur when a profile's structural features make one or more intents
/// non-equivalent to matrix+TRC math.
#[allow(non_upper_case_globals)]
struct Safe;
#[allow(non_upper_case_globals, dead_code)]
impl Safe {
    /// Matrix+TRC output matches a CMS for every rendering intent.
    const AnyIntent: u8 =
        INTENT_COLORIMETRIC_SAFE | INTENT_PERCEPTUAL_SAFE | INTENT_SATURATION_SAFE;
    /// Profile is recognized, but matrix+TRC substitution is not safe for
    /// any intent — identification only.
    const IdOnly: u8 = 0;
    /// Colorimetric intents only (relative + absolute share one bit).
    const Colorimetric: u8 = INTENT_COLORIMETRIC_SAFE;
    /// Perceptual intent only.
    const Perceptual: u8 = INTENT_PERCEPTUAL_SAFE;
    /// Saturation intent only.
    const Saturation: u8 = INTENT_SATURATION_SAFE;
    /// Colorimetric + perceptual.
    const ColorimetricPerceptual: u8 = INTENT_COLORIMETRIC_SAFE | INTENT_PERCEPTUAL_SAFE;
    /// Colorimetric + saturation.
    const ColorimetricSaturation: u8 = INTENT_COLORIMETRIC_SAFE | INTENT_SATURATION_SAFE;
    /// Perceptual + saturation (no colorimetric equivalence).
    const PerceptualSaturation: u8 = INTENT_PERCEPTUAL_SAFE | INTENT_SATURATION_SAFE;
}

/// Well-known RGB ICC profiles: `(normalized_hash, primaries, transfer, max_u16_err, intent_mask)`.
///
/// `max_u16_err` is historical red/gray curve-fit error in u16 units,
/// compared against [`Tolerance`] at query time. It is not a complete
/// transform error bound; new generator runs check all RGB curves. `intent_mask` is a [`Safe`] alias
/// (e.g. `Safe::AnyIntent`, `Safe::IdOnly`) precomputed from the profile's
/// structural features (PCS type, `chad` matrix, A2B/B2A LUT presence,
/// matrix-shaper completeness).
///
/// Sorted by normalized hash for binary search. Generated by
/// `scripts/icc-gen` from the ICC profile corpus.
#[rustfmt::skip]
const KNOWN_RGB_PROFILES: &[(ProfileHash, CP, TF, u8, u8)] =
    include!("icc_table_rgb.inc");

/// Well-known grayscale ICC profiles: `(normalized_hash, transfer, max_u16_err, intent_mask)`.
///
/// Columns match [`KNOWN_RGB_PROFILES`] minus the primaries column
/// (grayscale has no gamut; D65 white point is assumed).
///
/// Sorted by normalized hash for binary search.
#[rustfmt::skip]
const KNOWN_GRAY_PROFILES: &[(ProfileHash, TF, u8, u8)] =
    include!("icc_table_gray.inc");

// ── Compact split SoA runtime tables ─────────────────────────────────────
//
// `KNOWN_RGB_PROFILES` and `KNOWN_GRAY_PROFILES` are compile-time inputs only
// (never referenced at runtime, so they emit 0 bytes into `.rodata`).
//
// Storing `((u64, u32), CP, TF, u8, u8)` directly wastes 8–9 bytes of struct
// alignment padding per entry (24 B/entry × 441 = 10,584 B) and forces callers
// that only need substitutable profiles (`ColorProfileSource::resolve`,
// `is_common_srgb`) to carry all 236 `Safe::IdOnly` metadata-only entries.
//
// Packing into parallel arrays `(fnv64: [u64; N], murmur32: [u32; N], meta: [u16; N])`
// takes 14 B/entry losslessly (full 96-bit fingerprint + exact primaries,
// transfer, 3-bit intent mask, and 6-bit max_u16_err), splits substitutable
// (~2.9 KB) from metadata-only (~3.3 KB) so the latter dead-strips when unused,
// and collapses RGB + gray into a single cache-friendly `u64` binary search.

const fn pack_cp(cp: CP) -> u16 {
    match cp {
        CP::Bt709 => 0,
        CP::DisplayP3 => 1,
        CP::Bt2020 => 2,
        CP::AdobeRgb => 3,
        _ => panic!("unsupported ColorPrimaries in ICC hash table"),
    }
}

const fn unpack_cp(meta: u16) -> CP {
    match meta & 0x7 {
        0 => CP::Bt709,
        1 => CP::DisplayP3,
        2 => CP::Bt2020,
        3 => CP::AdobeRgb,
        _ => CP::Unknown,
    }
}

const fn pack_tf(tf: TF) -> u16 {
    match tf {
        TF::Linear => 0,
        TF::Srgb => 1,
        TF::Bt709 => 2,
        TF::Gamma22 => 3,
        TF::Pq => 4,
        TF::Hlg(_) => 5,
        TF::AdobeRgb => 6,
        _ => panic!("unsupported Transfer in ICC hash table"),
    }
}

const fn unpack_tf(meta: u16) -> TF {
    match (meta >> 3) & 0x7 {
        0 => TF::Linear,
        1 => TF::Srgb,
        2 => TF::Bt709,
        3 => TF::Gamma22,
        4 => TF::Pq,
        5 => TF::Hlg(HlgMode::Scene),
        6 => TF::AdobeRgb,
        _ => TF::Unknown,
    }
}

const fn pack_meta(cp: CP, tf: TF, err: u8, mask: u8) -> u16 {
    assert!(
        err <= 63,
        "ICC table max_u16_err exceeds 6-bit packed limit"
    );
    assert!(
        mask <= 7,
        "ICC table intent_mask exceeds 3-bit packed limit"
    );
    pack_cp(cp) | (pack_tf(tf) << 3) | ((mask as u16) << 6) | ((err as u16) << 9)
}

#[inline]
const fn unpack_mask(meta: u16) -> u8 {
    ((meta >> 6) & 0x7) as u8
}

#[inline]
const fn unpack_err(meta: u16) -> u8 {
    ((meta >> 9) & 0x3f) as u8
}

const fn count_compact(substitutable: bool) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < KNOWN_RGB_PROFILES.len() {
        if (KNOWN_RGB_PROFILES[i].4 != Safe::IdOnly) == substitutable {
            n += 1;
        }
        i += 1;
    }
    let mut j = 0;
    while j < KNOWN_GRAY_PROFILES.len() {
        if (KNOWN_GRAY_PROFILES[j].3 != Safe::IdOnly) == substitutable {
            n += 1;
        }
        j += 1;
    }
    n
}

struct CompactProfileTable<const N: usize> {
    fnv64: [u64; N],
    murmur32: [u32; N],
    meta: [u16; N],
}

const fn build_compact_table<const N: usize>(substitutable: bool) -> CompactProfileTable<N> {
    let mut fnv64 = [0u64; N];
    let mut murmur32 = [0u32; N];
    let mut meta = [0u16; N];

    let mut ri = 0;
    let mut gi = 0;
    let mut out = 0;

    while out < N {
        while ri < KNOWN_RGB_PROFILES.len()
            && (KNOWN_RGB_PROFILES[ri].4 != Safe::IdOnly) != substitutable
        {
            ri += 1;
        }
        while gi < KNOWN_GRAY_PROFILES.len()
            && (KNOWN_GRAY_PROFILES[gi].3 != Safe::IdOnly) != substitutable
        {
            gi += 1;
        }

        let take_rgb = if ri < KNOWN_RGB_PROFILES.len() && gi < KNOWN_GRAY_PROFILES.len() {
            let rh = KNOWN_RGB_PROFILES[ri].0;
            let gh = KNOWN_GRAY_PROFILES[gi].0;
            rh.0 < gh.0 || (rh.0 == gh.0 && rh.1 < gh.1)
        } else {
            ri < KNOWN_RGB_PROFILES.len()
        };

        let (h, m) = if take_rgb {
            let e = KNOWN_RGB_PROFILES[ri];
            ri += 1;
            (e.0, pack_meta(e.1, e.2, e.3, e.4))
        } else {
            let e = KNOWN_GRAY_PROFILES[gi];
            gi += 1;
            (e.0, pack_meta(CP::Bt709, e.1, e.2, e.3))
        };

        if out > 0 {
            // Binary search indexes by `fnv64` directly; assert strict uniqueness.
            assert!(
                fnv64[out - 1] < h.0,
                "duplicate or unsorted fnv64 hash in compact ICC table"
            );
        }
        fnv64[out] = h.0;
        murmur32[out] = h.1;
        meta[out] = m;
        out += 1;
    }

    CompactProfileTable {
        fnv64,
        murmur32,
        meta,
    }
}

const SUBSTITUTABLE_LEN: usize = count_compact(true);
const METADATA_ONLY_LEN: usize = count_compact(false);

const SUBSTITUTABLE_TABLE: CompactProfileTable<SUBSTITUTABLE_LEN> = build_compact_table(true);
const METADATA_ONLY_TABLE: CompactProfileTable<METADATA_ONLY_LEN> = build_compact_table(false);

static SUBSTITUTABLE_FNV64: [u64; SUBSTITUTABLE_LEN] = SUBSTITUTABLE_TABLE.fnv64;
static SUBSTITUTABLE_MURMUR32: [u32; SUBSTITUTABLE_LEN] = SUBSTITUTABLE_TABLE.murmur32;
static SUBSTITUTABLE_META: [u16; SUBSTITUTABLE_LEN] = SUBSTITUTABLE_TABLE.meta;

static METADATA_ONLY_FNV64: [u64; METADATA_ONLY_LEN] = METADATA_ONLY_TABLE.fnv64;
static METADATA_ONLY_MURMUR32: [u32; METADATA_ONLY_LEN] = METADATA_ONLY_TABLE.murmur32;
static METADATA_ONLY_META: [u16; METADATA_ONLY_LEN] = METADATA_ONLY_TABLE.meta;

fn lookup_compact(fnv64: &[u64], murmur32: &[u32], meta: &[u16], hash: ProfileHash) -> Option<u16> {
    let idx = fnv64.binary_search(&hash.0).ok()?;
    if murmur32[idx] == hash.1 {
        Some(meta[idx])
    } else {
        None
    }
}

#[inline]
fn lookup_substitutable_hash(hash: ProfileHash) -> Option<u16> {
    lookup_compact(
        &SUBSTITUTABLE_FNV64,
        &SUBSTITUTABLE_MURMUR32,
        &SUBSTITUTABLE_META,
        hash,
    )
}

#[inline]
fn lookup_metadata_only_hash(hash: ProfileHash) -> Option<u16> {
    lookup_compact(
        &METADATA_ONLY_FNV64,
        &METADATA_ONLY_MURMUR32,
        &METADATA_ONLY_META,
        hash,
    )
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_table_sorted() {
        for i in 1..KNOWN_RGB_PROFILES.len() {
            assert!(
                KNOWN_RGB_PROFILES[i - 1].0 < KNOWN_RGB_PROFILES[i].0,
                "KNOWN_RGB_PROFILES not sorted at index {i}: {:?} >= {:?}",
                KNOWN_RGB_PROFILES[i - 1].0,
                KNOWN_RGB_PROFILES[i].0,
            );
        }
    }

    #[test]
    fn gray_table_sorted() {
        for i in 1..KNOWN_GRAY_PROFILES.len() {
            assert!(
                KNOWN_GRAY_PROFILES[i - 1].0 < KNOWN_GRAY_PROFILES[i].0,
                "KNOWN_GRAY_PROFILES not sorted at index {i}: {:?} >= {:?}",
                KNOWN_GRAY_PROFILES[i - 1].0,
                KNOWN_GRAY_PROFILES[i].0,
            );
        }
    }

    #[test]
    fn no_duplicate_hashes() {
        let mut seen = alloc::collections::BTreeSet::new();
        for entry in KNOWN_RGB_PROFILES {
            assert!(seen.insert(entry.0), "duplicate RGB hash {:?}", entry.0);
        }
        for entry in KNOWN_GRAY_PROFILES {
            assert!(seen.insert(entry.0), "duplicate gray hash {:?}", entry.0);
        }
    }

    #[test]
    fn all_errors_within_intent() {
        for &(h, _, _, err, _) in KNOWN_RGB_PROFILES {
            assert!(err <= 56, "RGB {h:?} err={err} > 56");
        }
        for &(h, _, err, _) in KNOWN_GRAY_PROFILES {
            assert!(err <= 56, "gray {h:?} err={err} > 56");
        }
    }

    #[test]
    fn no_unknown_variants_in_tables() {
        for &(h, cp, tc, _, _) in KNOWN_RGB_PROFILES {
            assert_ne!(cp, Primaries::Unknown, "RGB {h:?}");
            assert_ne!(tc, Transfer::Unknown, "RGB {h:?}");
        }
        for &(h, tc, _, _) in KNOWN_GRAY_PROFILES {
            assert_ne!(tc, Transfer::Unknown, "gray {h:?}");
        }
    }

    #[test]
    fn intent_mask_reserved_bits_zero() {
        // Only bits 0–2 are defined; bits 3–7 must be zero.
        const ALL_DEFINED: u8 =
            INTENT_COLORIMETRIC_SAFE | INTENT_PERCEPTUAL_SAFE | INTENT_SATURATION_SAFE;
        for &(h, _, _, _, mask) in KNOWN_RGB_PROFILES {
            assert_eq!(
                mask & !ALL_DEFINED,
                0,
                "RGB {h:?} has reserved bits set: 0x{mask:02x}"
            );
        }
        for &(h, _, _, mask) in KNOWN_GRAY_PROFILES {
            assert_eq!(
                mask & !ALL_DEFINED,
                0,
                "gray {h:?} has reserved bits set: 0x{mask:02x}"
            );
        }
    }

    #[test]
    fn compact_tables_match_raw_tables_bit_exact() {
        let tolerances = [
            Tolerance::Exact,
            Tolerance::Precise,
            Tolerance::Approximate,
            Tolerance::Intent,
        ];
        let uses = [
            CoalesceForUse::AnyIntent,
            CoalesceForUse::RelativeColorimetric,
            CoalesceForUse::AbsoluteColorimetric,
            CoalesceForUse::Perceptual,
            CoalesceForUse::Saturation,
        ];

        assert_eq!(
            SUBSTITUTABLE_LEN + METADATA_ONLY_LEN,
            KNOWN_RGB_PROFILES.len() + KNOWN_GRAY_PROFILES.len(),
            "compact split tables must cover every raw entry"
        );
        // Verify compact table size is strictly 14 B/entry (u64 + u32 + u16).
        assert_eq!(
            core::mem::size_of_val(&SUBSTITUTABLE_FNV64)
                + core::mem::size_of_val(&SUBSTITUTABLE_MURMUR32)
                + core::mem::size_of_val(&SUBSTITUTABLE_META),
            SUBSTITUTABLE_LEN * 14
        );

        let verify_entry = |hash: ProfileHash, cp: CP, tf: TF, err: u8, mask: u8| {
            let meta = lookup_substitutable_hash(hash)
                .or_else(|| lookup_metadata_only_hash(hash))
                .expect("every raw entry must exist in compact tables");
            assert_eq!(unpack_cp(meta), cp);
            assert_eq!(unpack_tf(meta), tf);
            assert_eq!(unpack_err(meta), err);
            assert_eq!(unpack_mask(meta), mask);

            // Partition invariant: substitutable iff mask != 0.
            assert_eq!(
                lookup_substitutable_hash(hash).is_some(),
                mask != Safe::IdOnly
            );
            assert_eq!(
                lookup_metadata_only_hash(hash).is_some(),
                mask == Safe::IdOnly
            );

            // Check every tolerance and intent predicate.
            for tol in tolerances {
                let expected =
                    (err <= tol as u8).then(|| IccIdentification::new(cp, tf, use_from_mask(mask)));
                let actual = (unpack_err(meta) <= tol as u8).then(|| {
                    IccIdentification::new(
                        unpack_cp(meta),
                        unpack_tf(meta),
                        use_from_mask(unpack_mask(meta)),
                    )
                });
                assert_eq!(actual, expected);

                for use_for in uses {
                    let req = use_for.required_mask();
                    let expected_for = (!matches!(use_for, CoalesceForUse::AbsoluteColorimetric)
                        && err <= tol as u8
                        && (mask & req) == req)
                        .then(|| {
                            IccIdentification::new(cp, tf, IdentificationUse::MatrixTrcSubstitution)
                        });
                    let actual_for = if matches!(use_for, CoalesceForUse::AbsoluteColorimetric) {
                        None
                    } else {
                        lookup_substitutable_hash(hash).and_then(|m| {
                            (unpack_err(m) <= tol as u8 && (unpack_mask(m) & req) == req).then(
                                || {
                                    IccIdentification::new(
                                        unpack_cp(m),
                                        unpack_tf(m),
                                        IdentificationUse::MatrixTrcSubstitution,
                                    )
                                },
                            )
                        })
                    };
                    assert_eq!(actual_for, expected_for);
                }
            }
        };

        for &(h, cp, tf, err, mask) in KNOWN_RGB_PROFILES {
            verify_entry(h, cp, tf, err, mask);
        }
        for &(h, tf, err, mask) in KNOWN_GRAY_PROFILES {
            verify_entry(h, CP::Bt709, tf, err, mask);
        }

        // Wrong murmur32 with valid fnv64 must never match.
        let (fnv, murmur) = KNOWN_RGB_PROFILES[0].0;
        assert!(lookup_substitutable_hash((fnv, murmur ^ 1)).is_none());
        assert!(lookup_metadata_only_hash((fnv, murmur ^ 1)).is_none());
    }

    #[test]
    fn table_coverage() {
        let rgb_count = |cp: Primaries, tc: Transfer| {
            KNOWN_RGB_PROFILES
                .iter()
                .filter(|e| e.1 == cp && e.2 == tc)
                .count()
        };
        assert!(
            rgb_count(CP::Bt709, TF::Srgb) >= 25,
            "sRGB: {}",
            rgb_count(CP::Bt709, TF::Srgb)
        );
        assert!(
            rgb_count(CP::DisplayP3, TF::Srgb) >= 25,
            "Display P3: {}",
            rgb_count(CP::DisplayP3, TF::Srgb)
        );
        assert!(
            rgb_count(CP::AdobeRgb, TF::AdobeRgb) >= 15,
            "Adobe RGB: {}",
            rgb_count(CP::AdobeRgb, TF::AdobeRgb)
        );
        assert!(
            KNOWN_RGB_PROFILES.len() >= 90,
            "RGB total: {}",
            KNOWN_RGB_PROFILES.len()
        );
        assert!(
            KNOWN_GRAY_PROFILES.len() >= 10,
            "Gray total: {}",
            KNOWN_GRAY_PROFILES.len()
        );
    }

    #[test]
    fn zero_filled_no_false_positive() {
        for len in [410, 456, 480, 524, 548, 656, 736, 3024, 3144] {
            let zeros = alloc::vec![0u8; len];
            assert!(
                identify_common(&zeros).is_none(),
                "zeros({len}) falsely matched"
            );
        }
    }

    #[test]
    fn hash_deterministic() {
        // Independently computed FNV-1a64 + MurmurHash3_x86_32 (seed 3_242_157_231)
        // over a 160-byte fixture after zeroing ICC metadata-only header ranges.
        let mut fixture = [0u8; 160];
        for (i, slot) in fixture.iter_mut().enumerate() {
            *slot = (i.wrapping_mul(37).wrapping_add(11)) as u8;
        }
        const EXPECTED: (u64, u32) = (0x63a0_c773_7e54_3275, 0x103a_ebf3);
        assert_eq!(normalized_hash(&fixture), EXPECTED);

        // Editing an allowed metadata header byte (creation date byte 30) keeps the pair identical.
        let mut meta_edited = fixture;
        meta_edited[30] ^= 0xFF;
        assert_eq!(normalized_hash(&meta_edited), EXPECTED);
        assert!(normalized_eq(&fixture, &meta_edited));

        // Editing a tag payload byte (byte 140) changes the normalized fingerprint and equality.
        let mut tag_edited = fixture;
        tag_edited[140] ^= 0x01;
        assert_ne!(normalized_hash(&tag_edited), EXPECTED);
        assert!(!normalized_eq(&fixture, &tag_edited));
    }

    #[test]
    fn hash_distinct() {
        assert_ne!(normalized_hash(b"abc"), normalized_hash(b"abd"));
    }

    #[test]
    fn normalization_zeroes_metadata() {
        // Two profiles identical except for creation date (bytes 24-35)
        let mut a = alloc::vec![0u8; 200];
        let mut b = a.clone();
        a[30] = 0xFF; // different date byte
        b[30] = 0x01;
        assert_eq!(normalized_hash(&a), normalized_hash(&b));

        // But differ in color-critical field (byte 20 = PCS)
        let mut c = a.clone();
        c[20] = 0xFF;
        assert_ne!(normalized_hash(&a), normalized_hash(&c));
    }

    #[test]
    fn normalized_equality_checks_every_non_metadata_byte() {
        const {
            assert!(normalized_eq(b"same", b"same"));
            assert!(!normalized_eq(b"a", b"b"));
        }
        assert!(normalized_eq(&[], &[]));
        let a = [0x5au8; 300];
        assert!(!normalized_eq(&a[..299], &a));
        assert!(!normalized_eq(&a, &a[..299]));
        for index in 0..a.len() {
            let mut b = a;
            b[index] ^= 0xff;
            let metadata = [4..8, 24..36, 40..44, 48..56, 80..100]
                .iter()
                .any(|range| range.contains(&index));
            assert_eq!(normalized_eq(&a, &b), metadata, "byte {index}");
            assert_eq!(normalized_eq(&b, &a), metadata, "byte {index}");
            // The same predicate must hold when an input ends inside a field.
            assert_eq!(
                normalized_eq(&a[..=index], &b[..=index]),
                metadata,
                "truncated byte {index}"
            );
        }
    }

    #[test]
    fn is_common_srgb_rejects_empty() {
        assert!(!identify_common(&[]).is_some_and(|id| id.is_srgb()));
        assert!(!identify_common(&[0; 100]).is_some_and(|id| id.is_srgb()));
    }

    #[test]
    fn historical_gray_white_is_recognized_without_srgb_cms_bypass() {
        // Previously bundled C-white GRAY/sRGB profile. Its unchanged bytes
        // identify the approximate curve; they cannot stand in for D65 sRGB.
        let hex = [
            "000001a400000000044000006d6e74724752415958595a200000000000000000",
            "0000000061637370000000000000000000000000000000000000000000000000",
            "000000000000f6d6000100000000d32d00000000000000000000000000000000",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "0000000563686164000000c00000002c6b545243000000ec0000002077747074",
            "0000010c0000001464657363000001200000004a637072740000016c00000036",
            "736633320000000000010999000003e7fffff10c0000044e00010183fffffb27",
            "fffffce9000005420000b083706172610000000000030000000266660000f2a7",
            "00000d59000013d000000a5b58595a20000000000000fb240001000000012efd",
            "6d6c756300000000000000010000000c656e55530000002e0000001c00470072",
            "0061007900200048002e00320037003300200054004300310033002000430020",
            "0077006800690074006500006d6c756300000000000000010000000c656e5553",
            "0000001a0000001c005000750062006c0069006300200044006f006d00610069",
            "006e0000",
        ]
        .concat();
        let bytes: alloc::vec::Vec<u8> = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        let id = IccIdentification::from_icc(&bytes).expect("historical hash retained");
        assert!(id.is_srgb());
        assert_eq!(id.valid_use, IdentificationUse::MetadataOnly);
        assert!(!IccIdentification::is_common_srgb(&bytes));
        assert!(IccIdentification::from_icc_substitutable(&bytes).is_none());
    }

    // ── extract_cicp tests ───────────────────────────────────────────

    /// Helper: build a minimal ICC v4.4 Display profile with a cicp tag.
    fn build_icc_with_cicp(cp: u8, tc: u8, mc: u8, fr: bool) -> alloc::vec::Vec<u8> {
        let mut data = alloc::vec![0u8; 256];
        data[0..4].copy_from_slice(&256u32.to_be_bytes());
        data[8..12].copy_from_slice(&[4, 0x40, 0, 0]);
        data[12..16].copy_from_slice(b"mntr");
        data[16..20].copy_from_slice(b"RGB ");
        data[20..24].copy_from_slice(b"XYZ ");
        data[36..40].copy_from_slice(b"acsp");
        data[128..132].copy_from_slice(&1u32.to_be_bytes());
        data[132..136].copy_from_slice(b"cicp");
        data[136..140].copy_from_slice(&144u32.to_be_bytes());
        data[140..144].copy_from_slice(&12u32.to_be_bytes());
        data[144..148].copy_from_slice(b"cicp");
        data[152] = cp;
        data[153] = tc;
        data[154] = mc;
        data[155] = u8::from(fr);
        data
    }

    #[test]
    fn embedded_cicp_does_not_prove_icc_matrix_trc_substitution() {
        let icc = build_icc_with_cicp(1, 13, 0, true);
        assert!(extract_cicp(&icc).is_some());
        assert!(IccIdentification::from_icc_substitutable(&icc).is_none());
    }

    #[test]
    fn extract_cicp_srgb() {
        let icc = build_icc_with_cicp(1, 13, 0, true);
        let cicp = extract_cicp(&icc).unwrap();
        assert_eq!(cicp.color_primaries, 1);
        assert_eq!(cicp.transfer_characteristics, 13);
        assert_eq!(cicp.matrix_coefficients, 0);
        assert!(cicp.full_range);
    }

    #[test]
    fn extract_cicp_pq() {
        let icc = build_icc_with_cicp(9, 16, 0, true);
        let cicp = extract_cicp(&icc).unwrap();
        assert_eq!(cicp.color_primaries, 9);
        assert_eq!(cicp.transfer_characteristics, 16);
    }

    #[test]
    fn extract_cicp_empty() {
        assert!(extract_cicp(&[]).is_none());
        assert!(extract_cicp(&[0; 100]).is_none());
    }

    #[test]
    fn extract_cicp_no_acsp() {
        let mut icc = build_icc_with_cicp(1, 13, 0, true);
        icc[36..40].copy_from_slice(b"xxxx");
        assert!(extract_cicp(&icc).is_none());
    }

    #[test]
    fn extract_cicp_no_tag() {
        let mut data = build_icc_with_cicp(1, 13, 0, true);
        data[132..136].copy_from_slice(b"desc"); // not cicp
        assert!(extract_cicp(&data).is_none());
    }

    #[test]
    fn extract_cicp_tag_too_small() {
        let mut icc = build_icc_with_cicp(1, 13, 0, true);
        icc[140..144].copy_from_slice(&8u32.to_be_bytes()); // size < 12
        assert!(extract_cicp(&icc).is_none());
    }

    #[test]
    fn extract_cicp_type_mismatch() {
        let mut icc = build_icc_with_cicp(1, 13, 0, true);
        icc[144..148].copy_from_slice(b"xxxx"); // wrong type sig
        assert!(extract_cicp(&icc).is_none());
    }

    #[test]
    fn extract_cicp_malicious_tag_count() {
        let mut data = build_icc_with_cicp(1, 13, 0, true);
        data[132..136].copy_from_slice(b"desc");
        // Absurd tag count — capped to 200
        data[128..132].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(extract_cicp(&data).is_none());
    }

    #[test]
    fn extract_cicp_rejects_v2_link_reserved_truncation_and_bad_range() {
        // ICC v2.4 header -> rejected per ICC.1 §9.2.17
        let mut v2 = build_icc_with_cicp(1, 13, 0, true);
        v2[8..12].copy_from_slice(&[2, 0x40, 0, 0]);
        assert!(extract_cicp(&v2).is_none());

        // DeviceLink ('link') class -> rejected per ICC.1 §9.2.17
        let mut link = build_icc_with_cicp(1, 13, 0, true);
        link[12..16].copy_from_slice(b"link");
        assert!(extract_cicp(&link).is_none());

        // Non-zero reserved bytes 4..8 -> rejected per ICC.1 §10.4
        let mut reserved = build_icc_with_cicp(1, 13, 0, true);
        reserved[148] = 1;
        assert!(extract_cicp(&reserved).is_none());

        // Declared data_size extends past end of buffer -> rejected
        let mut truncated = build_icc_with_cicp(1, 13, 0, true);
        truncated[140..144].copy_from_slice(&500u32.to_be_bytes());
        assert!(extract_cicp(&truncated).is_none());

        // Invalid full_range byte (> 1) -> rejected per ICC.1 §10.4
        let mut bad_fr = build_icc_with_cicp(1, 13, 0, true);
        bad_fr[155] = 2;
        assert!(extract_cicp(&bad_fr).is_none());
    }

    /// Survey profile features across the local ICC cache.
    /// Run with: cargo test -p zenpixels survey_corpus_features -- --ignored --nocapture
    #[cfg(feature = "std")]
    #[test]
    #[ignore] // only run when explicitly requested
    fn survey_corpus_features() {
        let cache = std::env::var("ICC_CACHE").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.cache/zenpixels-icc")
        });
        let mut safe = 0;
        let mut unsafe_lab = 0;
        let mut unsafe_lut = 0;
        let mut unsafe_chad = 0;
        let mut no_matrix = 0;
        let mut total = 0;
        let entries = match std::fs::read_dir(&cache) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if !matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("icc" | "icm")
            ) {
                continue;
            }
            let data = match std::fs::read(&path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            total += 1;
            if let Some(feat) = inspect_profile(&data) {
                if feat.is_safe_matrix_shaper() {
                    safe += 1;
                } else if !feat.has_matrix_shaper {
                    no_matrix += 1;
                } else if feat.pcs_is_lab {
                    unsafe_lab += 1;
                } else if feat.has_a2b0
                    || feat.has_a2b1
                    || feat.has_a2b2
                    || feat.has_b2a0
                    || feat.has_b2a1
                    || feat.has_b2a2
                {
                    unsafe_lut += 1;
                } else if feat.has_chad && !feat.chad_is_bradford {
                    unsafe_chad += 1;
                }
            }
        }
        eprintln!("\n=== ICC Profile Features Survey (cache: {cache}) ===");
        eprintln!("Total profiles:                  {total}");
        eprintln!("Safe matrix-shaper:              {safe}");
        eprintln!("Has matrix tags + LUTs:          {unsafe_lut}");
        eprintln!("Lab PCS:                         {unsafe_lab}");
        eprintln!("Non-Bradford chad:               {unsafe_chad}");
        eprintln!("No matrix-shaper tags:           {no_matrix}");
    }

    // --- profile_color_space tests ---

    #[test]
    fn profile_color_space_detection() {
        // Minimal valid ICC header with CMYK color space at bytes 16-19
        let mut header = vec![0u8; 128];
        header[16..20].copy_from_slice(b"CMYK");
        assert_eq!(profile_color_space(&header), Some(IccColorSpace::Cmyk));

        header[16..20].copy_from_slice(b"RGB ");
        assert_eq!(profile_color_space(&header), Some(IccColorSpace::Rgb));

        header[16..20].copy_from_slice(b"GRAY");
        assert_eq!(profile_color_space(&header), Some(IccColorSpace::Gray));

        header[16..20].copy_from_slice(b"Lab ");
        assert_eq!(profile_color_space(&header), Some(IccColorSpace::Lab));

        header[16..20].copy_from_slice(b"XYZ ");
        assert_eq!(profile_color_space(&header), Some(IccColorSpace::Xyz));

        header[16..20].copy_from_slice(b"YCbr");
        assert_eq!(profile_color_space(&header), Some(IccColorSpace::Ycbcr));
    }

    #[test]
    fn profile_color_space_too_short() {
        assert_eq!(profile_color_space(&[0u8; 19]), None);
        assert_eq!(profile_color_space(&[]), None);
    }

    #[test]
    fn profile_color_space_unknown() {
        let mut header = vec![0u8; 128];
        header[16..20].copy_from_slice(b"5CLR");
        assert_eq!(profile_color_space(&header), None);
    }
}
