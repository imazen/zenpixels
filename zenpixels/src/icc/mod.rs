//! Lightweight ICC profile inspection and identification.
//!
//! Two capabilities, no dependencies beyond `core`:
//!
//! - **CICP extraction**: [`extract_cicp`] reads the `cicp` tag from ICC v4.4+
//!   profiles (~100ns, no allocation).
//!
//! - **Hash-based identification**: [`identify`] recognizes 132
//!   well-known profiles (sRGB, Display P3, BT.2020, Adobe RGB, ProPhoto,
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
//! use zenpixels::icc::{identify_common, Tolerance};
//!
//! # let icc_bytes: &[u8] = &[];
//! if let Some(id) = identify_common(icc_bytes, Tolerance::Intent) {
//!     // id.primaries: ColorPrimaries, id.transfer: TransferFunction
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

// ── Return type ──────────────────────────────────────────────────────────

/// Result of identifying a well-known ICC profile.
///
/// Carries the recognized color primaries and transfer function.
/// These may or may not have CICP equivalents — [`AdobeRgb`](ColorPrimaries::AdobeRgb)
/// and [`ProPhoto`](ColorPrimaries::ProPhoto) do not.
///
/// Use [`to_cicp`](Self::to_cicp) to convert to CICP codes when available.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct IccIdentification {
    /// Recognized color primaries.
    pub primaries: ColorPrimaries,
    /// Recognized transfer function.
    pub transfer: TransferFunction,
}

impl IccIdentification {
    /// Create a new identification result.
    #[inline]
    pub fn new(primaries: ColorPrimaries, transfer: TransferFunction) -> Self {
        Self {
            primaries,
            transfer,
        }
    }

    /// Convert to CICP codes if both primaries and transfer have CICP equivalents.
    ///
    /// Returns `None` for non-CICP color spaces (Adobe RGB, ProPhoto, etc.).
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
pub enum Tolerance {
    /// ±1 in u16 — parametric v4 profiles only.
    Exact = 1,
    /// ±3 in u16 — includes v2-magic parametric approximations.
    Precise = 3,
    /// ±13 in u16 — includes LUT profiles and iPhone P3.
    Approximate = 13,
    /// ±56 in u16 — honors encoder intent. Identical output at 8-bit and 10-bit.
    Intent = 56,
}

// ── Public API ───────────────────────────────────────────────────────────

/// Identify a well-known ICC profile by normalized hash lookup.
///
/// Computes a normalized FNV-1a 64-bit hash (metadata fields zeroed) and
/// checks against tables of known RGB and grayscale ICC profiles.
///
/// Returns `Some(IccIdentification)` for recognized profiles, `None` for
/// unknown ones. Grayscale profiles return [`Bt709`](ColorPrimaries::Bt709)
/// primaries (grayscale has no gamut, but D65 white point is assumed).
///
/// ~100ns. For the long tail of vendor/monitor profiles, use structural
/// analysis via a CMS backend.
pub fn identify_common(icc_bytes: &[u8], tolerance: Tolerance) -> Option<IccIdentification> {
    let hash = fnv1a_64_normalized(icc_bytes);

    // Try RGB table first.
    if let Ok(idx) = KNOWN_RGB_PROFILES.binary_search_by_key(&hash, |e| e.0) {
        let entry = &KNOWN_RGB_PROFILES[idx];
        if entry.3 <= tolerance as u8 {
            return Some(IccIdentification::new(entry.1, entry.2));
        }
    }

    // Try grayscale table.
    if let Ok(idx) = KNOWN_GRAY_PROFILES.binary_search_by_key(&hash, |e| e.0) {
        let entry = &KNOWN_GRAY_PROFILES[idx];
        if entry.2 <= tolerance as u8 {
            return Some(IccIdentification::new(ColorPrimaries::Bt709, entry.1));
        }
    }

    None
}

/// Check if an ICC profile is a known sRGB profile.
///
/// Convenience wrapper — returns `true` if the profile matches sRGB within
/// [`Intent`](Tolerance::Intent) tolerance.
#[inline]
pub fn is_common_srgb(icc_bytes: &[u8]) -> bool {
    identify_common(icc_bytes, Tolerance::Intent).is_some_and(|id| id.is_srgb())
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

/// Well-known RGB ICC profiles: `(normalized_hash, primaries, transfer, max_u16_err)`.
///
/// Sorted by normalized hash for binary search. Generated by
/// `scripts/gen_icc_tables.py` from the ICC profile corpus.
#[rustfmt::skip]
const KNOWN_RGB_PROFILES: &[(u64, CP, TF, u8)] =
    include!("icc_table_rgb.inc");

/// Well-known grayscale ICC profiles: `(normalized_hash, transfer, max_u16_err)`.
///
/// Sorted by normalized hash for binary search.
#[rustfmt::skip]
const KNOWN_GRAY_PROFILES: &[(u64, TF, u8)] =
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
        for &(h, _, _, err) in KNOWN_RGB_PROFILES {
            assert!(err <= 56, "RGB 0x{h:016x} err={err} > 56");
        }
        for &(h, _, err) in KNOWN_GRAY_PROFILES {
            assert!(err <= 56, "gray 0x{h:016x} err={err} > 56");
        }
    }

    #[test]
    fn no_unknown_variants_in_tables() {
        for &(h, cp, tc, _) in KNOWN_RGB_PROFILES {
            assert_ne!(cp, ColorPrimaries::Unknown, "RGB 0x{h:016x}");
            assert_ne!(tc, TransferFunction::Unknown, "RGB 0x{h:016x}");
        }
        for &(h, tc, _) in KNOWN_GRAY_PROFILES {
            assert_ne!(tc, TransferFunction::Unknown, "gray 0x{h:016x}");
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
                identify_common(&zeros, Tolerance::Intent).is_none(),
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
}
