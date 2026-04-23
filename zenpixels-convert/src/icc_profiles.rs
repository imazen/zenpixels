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

use crate::{ColorPrimaries, TransferFunction};

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

/// Get a bundled ICC profile matching both a primaries set and a transfer
/// function.
///
/// This is a finer-grained accessor than [`icc_profile_for_primaries`]: it
/// matches against the TRC encoded in each bundled profile, so a caller
/// that asks for `(Bt2020, Bt709)` gets the same Rec. 2020 profile, but a
/// caller asking for `(Bt2020, Pq)` gets `None` (no PQ profile bundled).
///
/// # Currently bundled combinations
///
/// | Primaries | Transfer | Returned profile |
/// |-----------|----------|------------------|
/// | [`Bt709`](ColorPrimaries::Bt709) | [`Srgb`](TransferFunction::Srgb) | `None` — sRGB is the assumed default |
/// | [`DisplayP3`](ColorPrimaries::DisplayP3) | [`Srgb`](TransferFunction::Srgb) | [`DISPLAY_P3_V4`] |
/// | [`Bt2020`](ColorPrimaries::Bt2020) | [`Bt709`](TransferFunction::Bt709) | [`REC2020_V4`] |
/// | [`AdobeRgb`](ColorPrimaries::AdobeRgb) | [`Gamma22`](TransferFunction::Gamma22) | [`ADOBE_RGB`] |
///
/// # Not bundled (returns `None`)
///
/// - HDR transfers ([`Pq`](TransferFunction::Pq), [`Hlg`](TransferFunction::Hlg))
///   on any primaries. Ultra HDR / HDR10 / HLG broadcast workflows that need
///   a PQ- or HLG-tagged profile should either generate one via a CMS crate
///   (e.g., `moxcms::ColorProfile::new_bt2020_pq().encode()`) or signal color
///   via CICP instead of ICC.
/// - [`Linear`](TransferFunction::Linear) on any primaries. Linear-light
///   working spaces are typically expressed with CICP transfer code 8
///   rather than an ICC profile.
/// - Adobe RGB with any transfer other than `Gamma22`.
/// - BT.2020 primaries with sRGB or BT.709 `Gamma22` / `Linear` transfers
///   other than the single bundled BT.709 paraType-3 form.
///
/// When this function returns `None`, call [`icc_profile_for_primaries`] as
/// a fallback if you can tolerate the profile's encoded TRC differing from
/// your requested transfer (e.g., accept the bundled BT.709 TRC for an
/// SDR BT.2020 export regardless of whether the caller asked for `Srgb` or
/// `Bt709`).
#[inline]
pub const fn icc_profile_for(
    primaries: ColorPrimaries,
    transfer: TransferFunction,
) -> Option<&'static [u8]> {
    match (primaries, transfer) {
        // Display P3 + sRGB: saucecontrol's DisplayP3Compat uses paraType-3
        // sRGB TRC, so both the sRGB and BT.709 transfer callers (which differ
        // only in the near-black linear segment) get the same profile.
        (ColorPrimaries::DisplayP3, TransferFunction::Srgb)
        | (ColorPrimaries::DisplayP3, TransferFunction::Bt709) => Some(DISPLAY_P3_V4),
        // BT.2020 + BT.709 TRC: saucecontrol's Rec2020Compat uses paraType-3
        // BT.709 TRC. Accept the sRGB request alias (same curve shape
        // outside the near-black toe), as this matches what ultrahdr-style
        // SDR BT.2020 base images need.
        (ColorPrimaries::Bt2020, TransferFunction::Bt709)
        | (ColorPrimaries::Bt2020, TransferFunction::Srgb) => Some(REC2020_V4),
        // Adobe RGB + Gamma22: bundled v2 pure-gamma variant.
        (ColorPrimaries::AdobeRgb, TransferFunction::Gamma22) => Some(ADOBE_RGB),
        // Everything else: no bundled profile with that exact TRC.
        _ => None,
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

    #[test]
    fn icc_profile_for_hits_bundled_combinations() {
        // Display P3 + sRGB / BT.709 paraType-3 curves both map to the bundled
        // DisplayP3Compat-v4 profile.
        assert_eq!(
            icc_profile_for(ColorPrimaries::DisplayP3, TransferFunction::Srgb),
            Some(DISPLAY_P3_V4)
        );
        assert_eq!(
            icc_profile_for(ColorPrimaries::DisplayP3, TransferFunction::Bt709),
            Some(DISPLAY_P3_V4)
        );
        // Rec 2020 SDR: BT.709 TRC profile is the canonical export for SDR
        // BT.2020 base images (matches the ultrahdr 8-bit base JPEG case).
        assert_eq!(
            icc_profile_for(ColorPrimaries::Bt2020, TransferFunction::Bt709),
            Some(REC2020_V4)
        );
        assert_eq!(
            icc_profile_for(ColorPrimaries::Bt2020, TransferFunction::Srgb),
            Some(REC2020_V4)
        );
        // Adobe RGB: pure-gamma-2.2 TRC.
        assert_eq!(
            icc_profile_for(ColorPrimaries::AdobeRgb, TransferFunction::Gamma22),
            Some(ADOBE_RGB)
        );
    }

    #[test]
    fn icc_profile_for_rejects_hdr_transfers() {
        // HDR PQ / HLG profiles aren't bundled; callers should use CICP or
        // a CMS-side generator.
        assert!(icc_profile_for(ColorPrimaries::Bt2020, TransferFunction::Pq).is_none());
        assert!(icc_profile_for(ColorPrimaries::Bt2020, TransferFunction::Hlg).is_none());
        assert!(icc_profile_for(ColorPrimaries::DisplayP3, TransferFunction::Pq).is_none());
        assert!(icc_profile_for(ColorPrimaries::DisplayP3, TransferFunction::Hlg).is_none());
        // Linear likewise isn't bundled — CICP 8 is the canonical signal.
        assert!(icc_profile_for(ColorPrimaries::Bt2020, TransferFunction::Linear).is_none());
        assert!(icc_profile_for(ColorPrimaries::DisplayP3, TransferFunction::Linear).is_none());
    }

    #[test]
    fn icc_profile_for_rejects_mismatched_trc_on_bundled_primaries() {
        // We only bundle Adobe RGB with gamma 2.2 — asking for sRGB TRC on
        // Adobe RGB primaries returns None rather than lying with a mismatched
        // curve. Callers who want a fallback use icc_profile_for_primaries.
        assert!(icc_profile_for(ColorPrimaries::AdobeRgb, TransferFunction::Srgb).is_none());
        assert!(icc_profile_for(ColorPrimaries::AdobeRgb, TransferFunction::Bt709).is_none());
        // Gamma 2.2 on DisplayP3 / Bt2020 isn't bundled either.
        assert!(icc_profile_for(ColorPrimaries::DisplayP3, TransferFunction::Gamma22).is_none());
        assert!(icc_profile_for(ColorPrimaries::Bt2020, TransferFunction::Gamma22).is_none());
    }

    #[test]
    fn icc_profile_for_bt709_returns_none() {
        // Same as icc_profile_for_primaries: BT.709 / sRGB is the assumed
        // default and isn't bundled.
        assert!(icc_profile_for(ColorPrimaries::Bt709, TransferFunction::Srgb).is_none());
        assert!(icc_profile_for(ColorPrimaries::Bt709, TransferFunction::Bt709).is_none());
        assert!(icc_profile_for(ColorPrimaries::Unknown, TransferFunction::Unknown).is_none());
    }
}
