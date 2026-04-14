//! Embedded CC0-licensed ICC profiles for common wide-gamut color spaces.
//!
//! All profiles are from [Compact-ICC-Profiles](https://github.com/saucecontrol/Compact-ICC-Profiles)
//! by Clinton Ingram, released under the [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
//! public domain dedication. They are embedded at compile time via `include_bytes!()`.
//!
//! # Available profiles
//!
//! | Constant | Color space | Format | Size | TRC form |
//! |----------|-------------|--------|------|----------|
//! | [`DISPLAY_P3_V4`] | Display P3 | ICC v4 | 480 bytes | paraType-3 sRGB |
//! | [`DISPLAY_P3_V2`] | Display P3 | ICC v2 | 736 bytes | paraType-3 sRGB |
//! | [`ADOBE_RGB`] | Adobe RGB (1998) | ICC v2 | 374 bytes | **curv count=1** (pure gamma) |
//! | [`REC2020_V4`] | Rec. 2020 | ICC v4 | 480 bytes | paraType-3 BT.709 |
//!
//! # Notes on choices
//!
//! ## Adobe RGB — pure gamma, not paraType-3
//!
//! We deliberately bundle the saucecontrol **v2** variant (`curv count=1`,
//! gamma 2.19921875, no linear toe) rather than their **v4** variant
//! (`paraType funcType=3` with linear toe). Rationale: surveying the in-the-wild
//! corpus, ~85% of Adobe RGB ICC profiles encode the TRC as pure gamma
//! (Adobe CS4 distribution, Windows `ClayRGB1998` / `AdobeRGB1998`, macOS
//! `AdobeRGB1998`, Linux `AdobeRGB1998`/`compatibleWithAdobeRGB1998`, Nikon,
//! and per-camera profiles). The Adobe RGB 1998 encoding spec (§4.3.4.2)
//! itself defines pure gamma with no toe. Bundling the pure-gamma form
//! matches both the spec and the majority of the ecosystem, and means the
//! embedded profile round-trips byte-exact against moxcms's `new_adobe_rgb()`
//! canonical reference.
//!
//! ## ProPhoto — not bundled
//!
//! Unlike Adobe RGB, real-world ProPhoto / ROMM ICC profiles are fragmented:
//! ~50% pure gamma 1.8, ~30% `paraType funcType=3` with the ISO 22028-2 toe
//! (`c=1/16, d=1/32`), some with a non-standard Apple `d=1/512`, one
//! `ProPhotoLin.icm` with a linear TRC despite the name, and two ISO 22028-2
//! v4 profiles that are mAB/mBA LUTs (no rTRC at all). Picking any single
//! "canonical" ProPhoto profile to embed would misrepresent the other
//! variants. Instead we leave ProPhoto un-accelerated: callers handing us a
//! ProPhoto ICC profile fall through to full CMS so the exact encoded curve
//! is honored.
//!
//! ## Display P3 / Rec. 2020 — kept as-is for embedding output
//!
//! The DisplayP3Compat and Rec2020Compat profiles use saucecontrol's
//! D50-sum-exact matrix rebalancing (truncating negative `rXYZ.Z` / `bXYZ.Z`
//! to clean s15.16 values, compensating via `chad` row 3). They diverge
//! ~500-900 u16 from canonical matrix math for the same reason — but they're
//! intended as **compact encoder-friendly output profiles**, not as
//! fast-path identification targets. They're bundled here for embedding in
//! encoded JPEG/PNG/WebP/etc. The ICC identification table in
//! `zenpixels/src/icc` deliberately excludes them from the safe-for-fast-path
//! set (see `scripts/icc-gen/src/main.rs`).
//!
//! # Lookup by primaries
//!
//! Use [`icc_profile_for_primaries`] to get the recommended ICC profile bytes
//! for a [`ColorPrimaries`] value. Returns `None` for `Bt709` (sRGB is assumed
//! by default and rarely needs an explicit ICC profile), `ProPhoto` (not
//! bundled; see above), and `Unknown`.
//!
//! # Precision warnings
//!
//! Rec. 2020 has a very wide gamut. Using 8-bit precision with Rec. 2020
//! will cause visible banding in gradients. Use 16-bit or f32 precision.

use crate::ColorPrimaries;

// ---------------------------------------------------------------------------
// Embedded ICC profiles (CC0 license from Compact-ICC-Profiles)
// https://github.com/saucecontrol/Compact-ICC-Profiles
// ---------------------------------------------------------------------------

/// Display P3 Compatible ICC profile, v4 format (480 bytes).
///
/// Recommended for modern software. ICC v4 profiles are more compact
/// and have better-defined semantics than v2.
///
/// Source: <https://github.com/saucecontrol/Compact-ICC-Profiles> (CC0)
pub const DISPLAY_P3_V4: &[u8] = include_bytes!("profiles/DisplayP3Compat-v4.icc");

/// Display P3 Compatible ICC profile, v2 format (736 bytes).
///
/// Use this for compatibility with older software that doesn't support ICC v4.
/// The "magic" variant includes workarounds for buggy v2 parsers.
///
/// Source: <https://github.com/saucecontrol/Compact-ICC-Profiles> (CC0)
pub const DISPLAY_P3_V2: &[u8] = include_bytes!("profiles/DisplayP3Compat-v2-magic.icc");

/// Adobe RGB (1998) Compatible ICC profile, v2 format (374 bytes, **pure gamma**).
///
/// Encodes the rTRC/gTRC/bTRC as `curv count=1` with gamma 2.19921875 (= 563/256),
/// matching the Adobe RGB 1998 spec and ~85% of real-world Adobe RGB ICC
/// profiles. See the module-level docs for the rationale on picking pure gamma
/// over the paraType-3 toe form.
///
/// Source: <https://github.com/saucecontrol/Compact-ICC-Profiles> (CC0)
pub const ADOBE_RGB: &[u8] = include_bytes!("profiles/AdobeCompat-v2.icc");

/// Deprecated alias for [`ADOBE_RGB`]. The v4 paraType-3 variant was replaced
/// with the v2 pure-gamma variant to match the spec and ~85% of the ecosystem.
#[deprecated(
    since = "0.2.4",
    note = "renamed to ADOBE_RGB (now v2 pure-gamma form)"
)]
pub const ADOBE_RGB_V4: &[u8] = ADOBE_RGB;

/// Deprecated: ProPhoto is not bundled due to TRC fragmentation.
/// See module-level docs for details.
#[deprecated(
    since = "0.2.4",
    note = "ProPhoto removed — TRC too fragmented to pick a canonical form"
)]
pub const PROPHOTO_V4: &[u8] = &[];

/// Rec. 2020 Compatible ICC profile, v4 format (480 bytes).
///
/// Rec. 2020 has a very wide gamut (~75% of visible colors).
/// **Use 16-bit or f32 precision** to avoid banding.
///
/// Source: <https://github.com/saucecontrol/Compact-ICC-Profiles> (CC0)
pub const REC2020_V4: &[u8] = include_bytes!("profiles/Rec2020Compat-v4.icc");

// ProPhoto / ROMM RGB is intentionally not bundled — see the module-level
// "ProPhoto — not bundled" note for the fragmentation analysis.

/// Get the recommended ICC profile for a set of color primaries.
///
/// Returns the recommended ICC profile bytes for the given primaries, or
/// `None` when no embedded profile is available. Returns `None` for:
/// - [`ColorPrimaries::Bt709`] — sRGB is the assumed default and rarely
///   needs an explicit ICC profile
/// - [`ColorPrimaries::Unknown`]
///
/// # Examples
///
/// ```
/// use zenpixels_convert::icc_profiles::icc_profile_for_primaries;
/// use zenpixels_convert::ColorPrimaries;
///
/// let p3_icc = icc_profile_for_primaries(ColorPrimaries::DisplayP3);
/// assert!(p3_icc.is_some());
/// assert_eq!(p3_icc.unwrap().len(), 480);
///
/// // sRGB returns None (assumed default)
/// assert!(icc_profile_for_primaries(ColorPrimaries::Bt709).is_none());
/// ```
#[inline]
pub const fn icc_profile_for_primaries(primaries: ColorPrimaries) -> Option<&'static [u8]> {
    match primaries {
        ColorPrimaries::DisplayP3 => Some(DISPLAY_P3_V4),
        ColorPrimaries::Bt2020 => Some(REC2020_V4),
        ColorPrimaries::AdobeRgb => Some(ADOBE_RGB),
        // BT.709/sRGB is the assumed default; no explicit ICC profile needed.
        // ProPhoto is deliberately not bundled — see module-level notes.
        ColorPrimaries::Bt709 | ColorPrimaries::Unknown | _ => None,
    }
}

/// Get the Display P3 ICC profile, choosing v4 or v2 format.
///
/// Returns v4 by default, or v2 if `prefer_v2` is true (for compatibility
/// with older software).
#[inline]
pub const fn display_p3_icc(prefer_v2: bool) -> &'static [u8] {
    if prefer_v2 {
        DISPLAY_P3_V2
    } else {
        DISPLAY_P3_V4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_profiles_valid_structure() {
        let profiles: &[(&[u8], &str)] = &[
            (DISPLAY_P3_V4, "Display P3 v4"),
            (REC2020_V4, "Rec. 2020 v4"),
        ];

        for (profile, name) in profiles {
            assert_eq!(profile.len(), 480, "{name}: expected 480 bytes");
            assert_eq!(
                &profile[36..40],
                b"acsp",
                "{name}: missing ICC 'acsp' signature at offset 36"
            );
            assert_eq!(
                &profile[12..16],
                b"mntr",
                "{name}: expected 'mntr' (monitor) profile class at offset 12"
            );
        }
    }

    #[test]
    fn adobe_rgb_profile_valid_structure() {
        assert_eq!(
            ADOBE_RGB.len(),
            374,
            "Adobe RGB: expected 374 bytes (ICC v2)"
        );
        assert_eq!(
            &ADOBE_RGB[36..40],
            b"acsp",
            "Adobe RGB: missing ICC 'acsp' signature at offset 36"
        );
        assert_eq!(
            &ADOBE_RGB[12..16],
            b"mntr",
            "Adobe RGB: expected 'mntr' (monitor) profile class at offset 12"
        );
        // TRC is `curv count=1` (pure gamma) — offset at rTRC tag, 14 bytes.
        // `curv` signature (4) + reserved (4) + count=1 (4) + u16 gamma (2) = 14.
        // This is the marker distinguishing the v2 pure-gamma variant from the
        // v4 paraType-3 variant (32+ bytes).
        let tag_count = u32::from_be_bytes([
            ADOBE_RGB[128],
            ADOBE_RGB[129],
            ADOBE_RGB[130],
            ADOBE_RGB[131],
        ]) as usize;
        let mut found_pure_gamma_trc = false;
        for i in 0..tag_count {
            let b = 132 + i * 12;
            if &ADOBE_RGB[b..b + 4] == b"rTRC" {
                let off = u32::from_be_bytes([
                    ADOBE_RGB[b + 4],
                    ADOBE_RGB[b + 5],
                    ADOBE_RGB[b + 6],
                    ADOBE_RGB[b + 7],
                ]) as usize;
                assert_eq!(
                    &ADOBE_RGB[off..off + 4],
                    b"curv",
                    "Adobe RGB: rTRC must be curveType (pure gamma)"
                );
                let count = u32::from_be_bytes([
                    ADOBE_RGB[off + 8],
                    ADOBE_RGB[off + 9],
                    ADOBE_RGB[off + 10],
                    ADOBE_RGB[off + 11],
                ]);
                assert_eq!(
                    count, 1,
                    "Adobe RGB: curveType count must be 1 (pure gamma, no toe)"
                );
                found_pure_gamma_trc = true;
                break;
            }
        }
        assert!(found_pure_gamma_trc, "Adobe RGB: rTRC tag not found");
    }

    #[test]
    fn v2_profile_valid_structure() {
        assert_eq!(
            DISPLAY_P3_V2.len(),
            736,
            "Display P3 v2: expected 736 bytes"
        );
        assert_eq!(
            &DISPLAY_P3_V2[36..40],
            b"acsp",
            "Display P3 v2: missing ICC 'acsp' signature at offset 36"
        );
        assert_eq!(
            &DISPLAY_P3_V2[12..16],
            b"mntr",
            "Display P3 v2: expected 'mntr' (monitor) profile class at offset 12"
        );
    }

    #[test]
    fn display_p3_icc_selector() {
        assert_eq!(display_p3_icc(false).len(), 480); // v4
        assert_eq!(display_p3_icc(true).len(), 736); // v2
    }

    #[test]
    fn icc_profile_for_primaries_mapping() {
        assert_eq!(
            icc_profile_for_primaries(ColorPrimaries::DisplayP3),
            Some(DISPLAY_P3_V4)
        );
        assert_eq!(
            icc_profile_for_primaries(ColorPrimaries::Bt2020),
            Some(REC2020_V4)
        );
        assert_eq!(
            icc_profile_for_primaries(ColorPrimaries::AdobeRgb),
            Some(ADOBE_RGB)
        );
        assert!(icc_profile_for_primaries(ColorPrimaries::Bt709).is_none());
        assert!(icc_profile_for_primaries(ColorPrimaries::Unknown).is_none());
    }
}
