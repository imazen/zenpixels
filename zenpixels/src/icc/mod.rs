//! Lightweight ICC profile inspection and identification.
//!
//! Two capabilities, no dependencies beyond `core`:
//!
//! - **CICP extraction**: [`extract_cicp`] reads the `cicp` tag from ICC v4.4+
//!   profiles (~100ns, no allocation).
//!
//! - **Hash-based identification**: [`identify_common`] recognizes 181
//!   well-known profiles (sRGB, Display P3, BT.2020, Adobe RGB,
//!   grayscale) via normalized FNV-1a hash lookup (~100ns).
//!
//! The hash table covers profiles found in a web corpus of 55,539 spidered
//! images, plus Compact-ICC-Profiles, skcms, ICC.org, colord, Ghostscript,
//! moxcms, and zenpixels-convert bundled profiles. Metadata-only header
//! fields are zeroed before hashing to collapse functionally identical
//! profiles — safe across ICC v2.0–v4.4 and v5/iccMAX. Each entry is
//! verified against its reference EOTF for all 65536 u16 values.
//!
//! # Example
//!
//! ```
//! use zenpixels::icc::{identify_common, IdentificationUse};
//!
//! # let icc_bytes: &[u8] = &[];
//! if let Some(id) = identify_common(icc_bytes) {
//!     if id.valid_use == IdentificationUse::MatrixTrcSubstitution {
//!         // Safe to skip CMS — use matrix+TRC math directly
//!     }
//!     // id.primaries and id.transfer are always available for metadata
//! }
//! ```

use crate::{Cicp, ColorPrimaries, TransferFunction};

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

// ── CICP extraction ──────────────────────────────────────────────────────

/// Extract CICP (Coding-Independent Code Points) from an ICC profile's tag table.
///
/// Scans the ICC tag table for a `cicp` tag (ICC v4.4+, 12 bytes) and returns
/// a [`Cicp`] if found. Returns `None` for ICC v2 profiles (which never contain
/// cicp tags), profiles without a cicp tag, or malformed input.
///
/// This is a lightweight operation (~100ns) that reads only the 128-byte header
/// and tag table entries — no full profile parse required.
pub fn extract_cicp(data: &[u8]) -> Option<Cicp> {
    if data.len() < ICC_MIN_SIZE {
        return None;
    }
    if data.get(ICC_SIGNATURE_OFFSET..ICC_SIGNATURE_OFFSET + 4)? != b"acsp" {
        return None;
    }

    let tag_count = u32::from_be_bytes(
        data[ICC_TAG_COUNT_OFFSET..ICC_TAG_COUNT_OFFSET + 4]
            .try_into()
            .ok()?,
    ) as usize;
    let tag_count = tag_count.min(ICC_MAX_TAG_COUNT);

    for i in 0..tag_count {
        let entry_offset = ICC_TAG_TABLE_OFFSET + i * ICC_TAG_ENTRY_SIZE;
        let entry = data.get(entry_offset..entry_offset + ICC_TAG_ENTRY_SIZE)?;

        if entry[..4] != *b"cicp" {
            continue;
        }

        let data_offset = u32::from_be_bytes(entry[4..8].try_into().ok()?) as usize;
        let data_size = u32::from_be_bytes(entry[8..12].try_into().ok()?) as usize;

        if data_size < 12 {
            return None;
        }

        let tag_data = data.get(data_offset..data_offset + 12)?;

        // Tag data starts with type signature (should also be "cicp").
        if tag_data[..4] != *b"cicp" {
            return None;
        }
        // Bytes 4..8 are reserved (should be zero).
        // Bytes 8..12 are the four CICP fields.
        return Some(Cicp::new(
            tag_data[8],
            tag_data[9],
            tag_data[10],
            tag_data[11] != 0,
        ));
    }

    None
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
/// matrix+TRC approximation via [`identify_common`] is equivalent to a
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
    let mut chad_off = None;

    for i in 0..tag_count {
        let entry_offset = ICC_TAG_TABLE_OFFSET + i * ICC_TAG_ENTRY_SIZE;
        let entry = data.get(entry_offset..entry_offset + ICC_TAG_ENTRY_SIZE)?;
        let sig = &entry[..4];
        let d_off = u32::from_be_bytes(entry[4..8].try_into().ok()?) as usize;
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
            b"chad" => chad_off = Some(d_off),
            _ => {}
        }
    }

    feat.has_matrix_shaper = has_rxyz && has_gxyz && has_bxyz && has_rtrc && has_gtrc && has_btrc;

    // Parse chad matrix if present
    if let Some(off) = chad_off {
        feat.has_chad = true;
        // chad tag: "sf32" (signature) + 4 reserved + 9 × s15Fixed16 (f64 values)
        if data.get(off..off + 8)? == b"sf32\0\0\0\0" && data.len() >= off + 8 + 36 {
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
/// These may or may not have CICP equivalents — [`AdobeRgb`](ColorPrimaries::AdobeRgb)
/// does not.
///
/// Use [`to_cicp`](Self::to_cicp) to convert to CICP codes when available.
/// Check [`valid_use`](Self::valid_use) before deciding whether to skip a CMS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct IccIdentification {
    /// Recognized color primaries.
    pub primaries: ColorPrimaries,
    /// Recognized transfer function.
    pub transfer: TransferFunction,
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
    /// Create a new identification result.
    #[inline]
    pub(crate) fn new(
        primaries: ColorPrimaries,
        transfer: TransferFunction,
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
        matches!(self.primaries, ColorPrimaries::Bt709)
            && matches!(self.transfer, TransferFunction::Srgb)
    }
}

// ── Tolerance ────────────────────────────────────────────────────────────

/// TRC match tolerance for ICC profile identification.
///
/// Each entry in the hash table stores its measured maximum error in u16
/// space (0–65535), verified against the reference EOTF for all 65536
/// input values. This tolerance controls which entries are accepted.
///
/// **All levels produce identical results at 8-bit depth.** The worst case
/// (Intent, ±56/65535) shifts a u8 value by at most 0.22 of a step —
/// always rounds to the same u8. Even at 10-bit (0–1023), Intent is
/// ±0.87 of a code value. Visible differences require 14-bit or higher.
///
/// Use [`Intent`](Self::Intent) unless you are computing perceptual metrics
/// at 16-bit precision.
///
/// | Level | Max u16 error | 8-bit impact | What it covers |
/// |-------|--------------|--------------|----------------|
/// | `Exact` | ±1 | none | Parametric v4 profiles |
/// | `Precise` | ±3 | none | + v2-magic approximations |
/// | `Approximate` | ±13 | none | + LUT profiles, iPhone P3 |
/// | `Intent` | ±56 | none | + Facebook sRGB, Chrome nano |
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
    /// ±56 in u16 — honors encoder intent. Identical output at 8-bit and 10-bit.
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
/// Computes a normalized FNV-1a 64-bit hash (metadata fields zeroed) and
/// checks against tables of known RGB and grayscale ICC profiles.
///
/// Returns `Some(IccIdentification)` for recognized profiles, `None` for
/// unknown ones. Check [`IccIdentification::valid_use`] before deciding
/// whether to skip a CMS — a recognized profile may still require one.
///
/// Grayscale profiles return [`Bt709`](ColorPrimaries::Bt709) primaries
/// (grayscale has no gamut, but D65 white point is assumed).
///
/// ~100ns. For the long tail of vendor/monitor profiles, use structural
/// analysis via a CMS backend.
pub fn identify_common(icc_bytes: &[u8]) -> Option<IccIdentification> {
    identify_common_at(icc_bytes, Tolerance::Intent)
}

/// Internal identification with configurable tolerance.
fn identify_common_at(icc_bytes: &[u8], tolerance: Tolerance) -> Option<IccIdentification> {
    let hash = fnv1a_64_normalized(icc_bytes);
    if let Ok(idx) = KNOWN_RGB_PROFILES.binary_search_by_key(&hash, |e| e.0) {
        let entry = &KNOWN_RGB_PROFILES[idx];
        if entry.3 <= tolerance as u8 {
            return Some(IccIdentification::new(
                entry.1,
                entry.2,
                use_from_mask(entry.4),
            ));
        }
    }
    if let Ok(idx) = KNOWN_GRAY_PROFILES.binary_search_by_key(&hash, |e| e.0) {
        let entry = &KNOWN_GRAY_PROFILES[idx];
        if entry.2 <= tolerance as u8 {
            return Some(IccIdentification::new(
                ColorPrimaries::Bt709,
                entry.1,
                use_from_mask(entry.3),
            ));
        }
    }
    None
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
/// specific rendering intent. Use [`identify_common`] for the common
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
    let hash = fnv1a_64_normalized(icc_bytes);
    let required = use_for.required_mask();

    // Try RGB table first.
    if let Ok(idx) = KNOWN_RGB_PROFILES.binary_search_by_key(&hash, |e| e.0) {
        let entry = &KNOWN_RGB_PROFILES[idx];
        if entry.3 <= tolerance as u8 && (entry.4 & required) == required {
            return Some(IccIdentification::new(
                entry.1,
                entry.2,
                IdentificationUse::MatrixTrcSubstitution,
            ));
        }
    }

    // Try grayscale table.
    if let Ok(idx) = KNOWN_GRAY_PROFILES.binary_search_by_key(&hash, |e| e.0) {
        let entry = &KNOWN_GRAY_PROFILES[idx];
        if entry.2 <= tolerance as u8 && (entry.3 & required) == required {
            return Some(IccIdentification::new(
                ColorPrimaries::Bt709,
                entry.1,
                IdentificationUse::MatrixTrcSubstitution,
            ));
        }
    }

    None
}

/// Check if an ICC profile is a known sRGB profile.
///
/// Convenience wrapper — returns `true` if the profile is recognized as sRGB.
#[inline]
pub fn is_common_srgb(icc_bytes: &[u8]) -> bool {
    identify_common(icc_bytes).is_some_and(|id| id.is_srgb())
}

/// Read the ICC profile's data color space from the header (bytes 16–19).
///
/// Returns the [`ColorModel`](crate::ColorModel) corresponding to the four-byte signature
/// at offset 16 in the ICC header, or `None` if the profile is too short
/// or uses an unrecognized color space.
pub fn profile_color_space(icc_bytes: &[u8]) -> Option<crate::ColorModel> {
    if icc_bytes.len() < 20 {
        return None;
    }
    match &icc_bytes[16..20] {
        b"RGB " => Some(crate::ColorModel::Rgb),
        b"GRAY" => Some(crate::ColorModel::Gray),
        b"CMYK" => Some(crate::ColorModel::Cmyk),
        _ => None,
    }
}

// ── Hash function ────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of ICC profile bytes with metadata normalization.
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
fn fnv1a_64_normalized(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;

    fn is_metadata_field(i: usize) -> bool {
        ICC_CMM_TYPE.contains(&i)
            || ICC_DATE_TIME.contains(&i)
            || ICC_PLATFORM.contains(&i)
            || ICC_DEVICE.contains(&i)
            || ICC_CREATOR_ID.contains(&i)
    }

    // Phase 1: header bytes — zero metadata fields.
    let header_len = data.len().min(ICC_HEADER_NORMALIZE_END);
    let mut i = 0;
    while i < header_len {
        let b = if is_metadata_field(i) { 0u8 } else { data[i] };
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
    hash
}

// ── Hash tables ──────────────────────────────────────────────────────────
//
// Tables store enum variants directly — no code-to-enum mapping needed.
// Shorthand aliases keep the .inc files readable.

use ColorPrimaries as CP;
use TransferFunction as TF;

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
/// `max_u16_err` is the empirically measured maximum channel error (in u16
/// units) between matrix+TRC math and a reference CMS; compared directly
/// against [`Tolerance`] at query time. `intent_mask` is a [`Safe`] alias
/// (e.g. `Safe::AnyIntent`, `Safe::IdOnly`) precomputed from the profile's
/// structural features (PCS type, `chad` matrix, A2B/B2A LUT presence,
/// matrix-shaper completeness).
///
/// Sorted by normalized hash for binary search. Generated by
/// `scripts/icc-gen` from the ICC profile corpus.
#[rustfmt::skip]
const KNOWN_RGB_PROFILES: &[(u64, CP, TF, u8, u8)] =
    include!("icc_table_rgb.inc");

/// Well-known grayscale ICC profiles: `(normalized_hash, transfer, max_u16_err, intent_mask)`.
///
/// Columns match [`KNOWN_RGB_PROFILES`] minus the primaries column
/// (grayscale has no gamut; D65 white point is assumed).
///
/// Sorted by normalized hash for binary search.
#[rustfmt::skip]
const KNOWN_GRAY_PROFILES: &[(u64, TF, u8, u8)] =
    include!("icc_table_gray.inc");

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_table_sorted() {
        for i in 1..KNOWN_RGB_PROFILES.len() {
            assert!(
                KNOWN_RGB_PROFILES[i - 1].0 < KNOWN_RGB_PROFILES[i].0,
                "KNOWN_RGB_PROFILES not sorted at index {i}: 0x{:016x} >= 0x{:016x}",
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
                "KNOWN_GRAY_PROFILES not sorted at index {i}: 0x{:016x} >= 0x{:016x}",
                KNOWN_GRAY_PROFILES[i - 1].0,
                KNOWN_GRAY_PROFILES[i].0,
            );
        }
    }

    #[test]
    fn no_duplicate_hashes() {
        let mut seen = alloc::collections::BTreeSet::new();
        for entry in KNOWN_RGB_PROFILES {
            assert!(
                seen.insert(entry.0),
                "duplicate RGB hash 0x{:016x}",
                entry.0
            );
        }
        for entry in KNOWN_GRAY_PROFILES {
            assert!(
                seen.insert(entry.0),
                "duplicate gray hash 0x{:016x}",
                entry.0
            );
        }
    }

    #[test]
    fn all_errors_within_intent() {
        for &(h, _, _, err, _) in KNOWN_RGB_PROFILES {
            assert!(err <= 56, "RGB 0x{h:016x} err={err} > 56");
        }
        for &(h, _, err, _) in KNOWN_GRAY_PROFILES {
            assert!(err <= 56, "gray 0x{h:016x} err={err} > 56");
        }
    }

    #[test]
    fn no_unknown_variants_in_tables() {
        for &(h, cp, tc, _, _) in KNOWN_RGB_PROFILES {
            assert_ne!(cp, ColorPrimaries::Unknown, "RGB 0x{h:016x}");
            assert_ne!(tc, TransferFunction::Unknown, "RGB 0x{h:016x}");
        }
        for &(h, tc, _, _) in KNOWN_GRAY_PROFILES {
            assert_ne!(tc, TransferFunction::Unknown, "gray 0x{h:016x}");
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
                "RGB 0x{h:016x} has reserved bits set: 0x{mask:02x}"
            );
        }
        for &(h, _, _, mask) in KNOWN_GRAY_PROFILES {
            assert_eq!(
                mask & !ALL_DEFINED,
                0,
                "gray 0x{h:016x} has reserved bits set: 0x{mask:02x}"
            );
        }
    }

    #[test]
    fn table_coverage() {
        let rgb_count = |cp: ColorPrimaries, tc: TransferFunction| {
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
            rgb_count(CP::AdobeRgb, TF::Gamma22) >= 15,
            "Adobe RGB: {}",
            rgb_count(CP::AdobeRgb, TF::Gamma22)
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
        let data = b"test data for hashing";
        assert_eq!(fnv1a_64_normalized(data), fnv1a_64_normalized(data));
    }

    #[test]
    fn hash_distinct() {
        assert_ne!(fnv1a_64_normalized(b"abc"), fnv1a_64_normalized(b"abd"));
    }

    #[test]
    fn normalization_zeroes_metadata() {
        // Two profiles identical except for creation date (bytes 24-35)
        let mut a = alloc::vec![0u8; 200];
        let mut b = a.clone();
        a[30] = 0xFF; // different date byte
        b[30] = 0x01;
        assert_eq!(fnv1a_64_normalized(&a), fnv1a_64_normalized(&b));

        // But differ in color-critical field (byte 20 = PCS)
        let mut c = a.clone();
        c[20] = 0xFF;
        assert_ne!(fnv1a_64_normalized(&a), fnv1a_64_normalized(&c));
    }

    #[test]
    fn is_common_srgb_rejects_empty() {
        assert!(!is_common_srgb(&[]));
        assert!(!is_common_srgb(&[0; 100]));
    }

    // ── extract_cicp tests ───────────────────────────────────────────

    /// Helper: build a minimal ICC profile with a cicp tag.
    fn build_icc_with_cicp(cp: u8, tc: u8, mc: u8, fr: bool) -> alloc::vec::Vec<u8> {
        let mut data = alloc::vec![0u8; 256];
        data[0..4].copy_from_slice(&256u32.to_be_bytes());
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
        let mut data = alloc::vec![0u8; 256];
        data[0..4].copy_from_slice(&256u32.to_be_bytes());
        data[36..40].copy_from_slice(b"acsp");
        data[128..132].copy_from_slice(&1u32.to_be_bytes());
        data[132..136].copy_from_slice(b"desc"); // not cicp
        data[136..140].copy_from_slice(&144u32.to_be_bytes());
        data[140..144].copy_from_slice(&12u32.to_be_bytes());
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
        let mut data = alloc::vec![0u8; 256];
        data[0..4].copy_from_slice(&256u32.to_be_bytes());
        data[36..40].copy_from_slice(b"acsp");
        // Absurd tag count — capped to 200
        data[128..132].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(extract_cicp(&data).is_none());
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
        assert_eq!(profile_color_space(&header), Some(crate::ColorModel::Cmyk));

        header[16..20].copy_from_slice(b"RGB ");
        assert_eq!(profile_color_space(&header), Some(crate::ColorModel::Rgb));

        header[16..20].copy_from_slice(b"GRAY");
        assert_eq!(profile_color_space(&header), Some(crate::ColorModel::Gray));
    }

    #[test]
    fn profile_color_space_too_short() {
        assert_eq!(profile_color_space(&[0u8; 19]), None);
        assert_eq!(profile_color_space(&[]), None);
    }

    #[test]
    fn profile_color_space_unknown() {
        let mut header = vec![0u8; 128];
        header[16..20].copy_from_slice(b"Lab ");
        assert_eq!(profile_color_space(&header), None);
    }
}
