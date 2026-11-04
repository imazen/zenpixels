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
//! ## Display P3 / Rec. 2020 — compact constants, not what synthesis emits
//!
//! The DisplayP3Compat and Rec2020Compat profiles use saucecontrol's
//! D50-sum-exact matrix rebalancing (truncating negative `rXYZ.Z` / `bXYZ.Z`
//! to clean s15.16 values, compensating via `chad` row 3). They diverge
//! ~500-900 u16 from canonical matrix math, and they carry no `cicp` tag.
//! Some readers identify a gamut by colorant tags within a tolerance:
//! libultrahdr (`IccHelper::readIccColorGamut`, 0.001) does not recognise
//! `DISPLAY_P3_V4` as Display P3 and treats the image as BT.709. With the
//! `icc-db` feature, [`synthesize_icc_for_cicp`] therefore serves the
//! generated, `cicp`-tagged profile for every pair and uses these constants
//! only as the fallback of a build without `icc-db`. They remain exported for
//! callers that embed them directly.
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

use alloc::borrow::Cow;

#[cfg(any(not(feature = "icc-db"), test))]
use crate::TransferFunction;
use crate::{Cicp, ColorPrimaries};

// Bundled, transfer-grouped CICP→ICC blob: a build-time-generated index
// (`cicp_bundle_index`) over an LZ4-compressed asset, decoded lazily per
// transfer group at runtime (`cicp_bundle`). This gives `synthesize_icc_for_cicp`
// full H.273-grid coverage with no CMS dependency — moxcms becomes a build-time
// generator only. Both submodules are internal implementation detail.
#[cfg(feature = "icc-db")]
mod cicp_bundle;
#[cfg(feature = "icc-db")]
#[rustfmt::skip]
mod cicp_bundle_index;

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
/// # Transfer-blind — prefer [`synthesize_icc_for_cicp`] when lowering a CICP
///
/// This function looks at primaries **only**, so the profile it returns always
/// carries the bundled SDR TRC for that gamut. Handing it a gamut whose source
/// is HDR — e.g. BT.2020 pixels that are actually **PQ** or **HLG** — gets back
/// the SDR-TRC Rec.2020 profile, which *mis-tags* the transfer. When you have a
/// full [`Cicp`] (primaries **and** transfer), call [`synthesize_icc_for_cicp`]
/// instead: it matches the TRC, returns a bundled profile for any assigned CICP
/// (HDR PQ/HLG included) with no CMS, and never substitutes a contradicting curve.
///
/// # Examples
///
/// ```
/// # #![allow(deprecated)] // demonstrating the (deprecated) primaries-only lookup
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
#[deprecated(
    since = "0.2.12",
    note = "transfer-blind (can mis-tag an HDR gamut with an SDR TRC); use \
            synthesize_icc_for_cicp for transfer-aware synthesis, or embed a \
            bundled const (DISPLAY_P3_V4 / REC2020_V4 / ADOBE_RGB) directly"
)]
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

/// Get a bundled `&'static` ICC profile whose tone curve is exactly the
/// requested transfer function on exactly the requested primaries.
///
/// This is a finer-grained accessor than [`icc_profile_for_primaries`]: it
/// matches against the TRC encoded in each bundled profile, so `(Bt2020,
/// Bt709)` gets the Rec. 2020 profile while `(Bt2020, Srgb)` and `(Bt2020,
/// Pq)` get `None`. sRGB and BT.709 are different curves (at linear 0.18 they
/// encode 0.461 and 0.409, 13 code values apart at 8 bits), so neither stands
/// in for the other.
///
/// # Currently bundled combinations
///
/// | Primaries | Transfer | Returned profile |
/// |-----------|----------|------------------|
/// | [`Bt709`](ColorPrimaries::Bt709) | [`Srgb`](crate::TransferFunction::Srgb) | `None` — sRGB is the assumed default |
/// | [`DisplayP3`](ColorPrimaries::DisplayP3) | [`Srgb`](crate::TransferFunction::Srgb) | [`DISPLAY_P3_V4`] |
/// | [`Bt2020`](ColorPrimaries::Bt2020) | [`Bt709`](crate::TransferFunction::Bt709) | [`REC2020_V4`] |
/// | [`AdobeRgb`](ColorPrimaries::AdobeRgb) | [`Gamma22`](crate::TransferFunction::Gamma22) | [`ADOBE_RGB`] |
///
/// # Not bundled (returns `None`)
///
/// - HDR transfers ([`Pq`](crate::TransferFunction::Pq), [`Hlg`](crate::TransferFunction::Hlg))
///   on any primaries. Ultra HDR / HDR10 / HLG broadcast workflows that need
///   a PQ- or HLG-tagged profile should either generate one via a CMS crate
///   (e.g., `moxcms::ColorProfile::new_bt2020_pq().encode()`) or signal color
///   via CICP instead of ICC.
/// - [`Linear`](crate::TransferFunction::Linear) on any primaries. Linear-light
///   working spaces are typically expressed with CICP transfer code 8
///   rather than an ICC profile.
/// - Adobe RGB with any transfer other than `Gamma22`.
/// - Display P3 with any transfer other than `Srgb`, and BT.2020 with any
///   transfer other than `Bt709`.
///
/// [`synthesize_icc_for_cicp`] serves every assigned pair (these included)
/// from the `icc-db` blob; this table is its fallback without that feature.
// Used by `synthesize_icc_for_cicp` only without `icc-db`.
#[cfg(any(not(feature = "icc-db"), test))]
#[inline]
pub(crate) const fn bundled_icc_profile(
    primaries: ColorPrimaries,
    transfer: TransferFunction,
) -> Option<&'static [u8]> {
    match (primaries, transfer) {
        // Display P3 + sRGB: saucecontrol's DisplayP3Compat carries the
        // paraType-3 sRGB TRC.
        (ColorPrimaries::DisplayP3, TransferFunction::Srgb) => Some(DISPLAY_P3_V4),
        // BT.2020 + BT.709: saucecontrol's Rec2020Compat carries the
        // paraType-3 BT.709 TRC.
        (ColorPrimaries::Bt2020, TransferFunction::Bt709) => Some(REC2020_V4),
        // Adobe RGB + Gamma22: bundled v2 pure-gamma variant.
        (ColorPrimaries::AdobeRgb, TransferFunction::Gamma22) => Some(ADOBE_RGB),
        // Everything else: no bundled profile with that exact TRC.
        _ => None,
    }
}

/// Outcome of [`synthesize_icc_for_cicp`] — what ICC profile (if any) to embed for a
/// CICP color description, and *why* there are no bytes when there aren't.
///
/// The point is to make lowering an `IccDisposition::SynthesizeFrom` explicit:
/// an implementer can tell apart "got bytes", "no profile needed", "we'd need a
/// CMS for this one", and "even the CMS couldn't" — instead of an ambiguous
/// `Option` that conflates them.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SynthesizedIcc {
    /// A profile is available — embed these bytes. From
    /// [`synthesize_icc_for_cicp`] this is always `Cow::Borrowed`: either a
    /// bundled `&'static` const, or a zero-copy slice of a decoded bundle group
    /// held in a `'static` cache (no allocation per call). `Cow::Owned` remains
    /// part of the type for callers that construct their own outcomes.
    Profile(Cow<'static, [u8]>),
    /// No ICC is needed: the CICP is the universally-assumed sRGB / BT.709 default,
    /// so omitting a profile loses nothing.
    ///
    /// In the encode pipeline this *shouldn't* occur — `resolve_color_emit` never
    /// asks to synthesize for sRGB (its `synth_worthwhile` test excludes it) — but
    /// it's surfaced explicitly rather than masquerading as a missing profile.
    NotNeeded,
    /// A CMS would be required to produce a profile for this CICP.
    ///
    /// Retained for API stability: [`synthesize_icc_for_cicp`] no longer returns
    /// this — the bundled blob serves supported assigned H.273 pairs with no
    /// CMS, so a no-CMS build reports [`CmsUnsupported`](Self::CmsUnsupported) for an
    /// unassigned / reserved code rather than `NeedsCms`.
    NeedsCms,
    /// The `cms-moxcms` CMS is enabled but could not generate a profile for this
    /// CICP — e.g. unrecognized primaries/transfer code points. Genuinely
    /// unavailable; carry the color via the CICP carrier instead.
    CmsUnsupported,
}

/// Resolve an ICC profile for a full [`Cicp`] color description (primaries **and**
/// transfer), returning a [`SynthesizedIcc`] that says exactly what happened.
///
/// This is the transfer-aware lowering for `IccDisposition::SynthesizeFrom`. Unlike
/// [`icc_profile_for_primaries`] (primaries-only — which would hand a BT.2020-**PQ**
/// source the SDR-TRC Rec.2020 profile), this never mis-tags: a CICP outside the
/// assigned H.273 grid (e.g. a reserved code point) returns
/// [`CmsUnsupported`](SynthesizedIcc::CmsUnsupported), so you carry the colour via the
/// CICP carrier or embed nothing — never a profile whose TRC contradicts the pixels.
///
/// Coverage, with **no CMS required** (a default `no_std` build serves all of it):
/// - **The supported assigned H.273 grid** (159 RGB primaries×transfer combos, **including HDR
///   PQ and HLG**), from a bundled LZ4-compressed blob (`icc-db`, on by default)
///   decoded once per transfer group on first use. These bytes are byte-for-byte what
///   the `cms-moxcms` CMS generates; every profile carries a `cicp` tag naming its
///   pair and the exact tone curve of its transfer.
/// - **Without `icc-db`**: the curated consts for the pairs whose curve they carry
///   exactly ([`DISPLAY_P3_V4`] for P3 + sRGB, [`REC2020_V4`] for BT.2020 + BT.709,
///   [`ADOBE_RGB`] for Adobe RGB + gamma 2.2); [`NeedsCms`](SynthesizedIcc::NeedsCms)
///   for the rest. These carry no `cicp` tag and are not recognised by colorant-matching
///   readers such as libultrahdr.
/// - **sRGB / unspecified defaults** → [`NotNeeded`](SynthesizedIcc::NotNeeded).
///
/// Primaries 10 (CIE XYZ) is refused as RGB in every build: its generated
/// RGB colorants are singular.
///
/// No CMS is involved — a default `no_std` build serves the full set above. (The
/// optional `cms-moxcms` feature is the separate `MoxCms` *transform* engine for
/// applying profiles to pixels — a different concern from synthesis.)
///
/// # HDR caveat (PQ)
/// HLG round-trips cleanly through a CMS. The **PQ** / P3-PQ profiles are faithful
/// above ~10 nits but, as ICC `curv`-LUT encodings, soften in the deep toe (≈8 %
/// relative at ~1 nit) — inherent to representing PQ's range as a finite ICC tone
/// curve, not specific to these bytes. Where the container signals PQ/HLG natively
/// (AVIF, JXL, HEIC, PNG `cICP`), prefer CICP-native signalling and treat an embedded
/// PQ ICC as a fallback for ICC-only containers (JPEG, WebP, PNG without `cICP`).
///
/// # Examples
/// ```
/// use zenpixels_convert::icc_profiles::{synthesize_icc_for_cicp, SynthesizedIcc};
/// use zenpixels_convert::Cicp;
///
/// // Display-P3 → a bundled profile.
/// assert!(matches!(synthesize_icc_for_cicp(Cicp::DISPLAY_P3), SynthesizedIcc::Profile(_)));
/// // sRGB → no profile needed.
/// assert_eq!(synthesize_icc_for_cicp(Cicp::SRGB), SynthesizedIcc::NotNeeded);
/// ```
pub fn synthesize_icc_for_cicp(cicp: Cicp) -> SynthesizedIcc {
    // sRGB / unspecified defaults need no embedded ICC. BT.709 transfer
    // with specified BT.709 primaries (1,1) is a different curve from sRGB.
    //
    // This is gated on the **raw** H.273 code points, NOT the enum mapping:
    // `color_primaries_enum` folds every gamut zenpixels doesn't model (BT.601 = 6,
    // SMPTE240 = 7, BT.470M/BG = 4/5, EBU3213 = 22, …) into `Unknown`, so an
    // enum-based check would treat those real, non-sRGB gamuts as "sRGB default"
    // and drop their colour — and worse, skip the blob lookup below that actually
    // carries them. Only colour-primaries 1 (BT.709) / 2 (unspecified) with
    // transfer 2 (unspecified) / 13 (sRGB) is the assumed default;
    // (2,1) has no specified gamut to describe.
    if (matches!(cicp.color_primaries, 1 | 2) && matches!(cicp.transfer_characteristics, 2 | 13))
        || (cicp.color_primaries == 2 && cicp.transfer_characteristics == 1)
    {
        return SynthesizedIcc::NotNeeded;
    }

    // H.273 primaries 10 denotes CIE XYZ. The generator's RGB model
    // collapses its colorants to rank 1; never serve those bytes, including
    // in builds without the database.
    if cicp.color_primaries == 10 {
        return SynthesizedIcc::CmsUnsupported;
    }

    // 1. The bundled, LZ4-compressed blob covering the supported H.273 grid
    //    (159 RGB combos) — `icc-db` feature. Keyed on the raw code points —
    //    moxcms-equivalent bytes, decoded once per transfer group and cached.
    //    Each profile carries the pair's exact tone curve and a `cicp` tag,
    //    so readers that identify a gamut by `cicp` tag or by colorants
    //    (libultrahdr) recognise it. It is consulted before the curated
    //    consts, which carry no `cicp` tag and truncated colorants.
    #[cfg(feature = "icc-db")]
    {
        if let Some(bytes) = cicp_bundle::bundled_profile_for_cicp(
            cicp.color_primaries,
            cicp.transfer_characteristics,
        ) {
            return SynthesizedIcc::Profile(bytes);
        }

        // 2. Not in the blob. The blob is generated from moxcms over the full
        //    assigned H.273 grid, so it already holds every profile moxcms
        //    could synthesize — reaching here means the CICP is outside that
        //    grid (e.g. a reserved / unspecified code point), genuinely
        //    unrepresentable as an ICC. A CMS cannot add what the blob lacks,
        //    so there is no moxcms fallback (the `cms-moxcms` feature drives
        //    the `MoxCms` transform engine, not synthesis).
        SynthesizedIcc::CmsUnsupported
    }
    // Without the database: a curated const whose curve is exactly the
    // requested transfer, else NeedsCms ("enable `icc-db`"), the variant's
    // documented build-dependent meaning.
    #[cfg(not(feature = "icc-db"))]
    {
        match bundled_icc_profile(cicp.color_primaries_enum(), cicp.transfer_function_enum()) {
            Some(bytes) => SynthesizedIcc::Profile(Cow::Borrowed(bytes)),
            None => SynthesizedIcc::NeedsCms,
        }
    }
}

/// Resolve a **GRAY-class** ICC profile for a [`Cicp`] color description —
/// the profile to embed alongside single-channel (Gray / GrayAlpha) output
/// whose color is described by CICP code points.
///
/// An ICC profile's header declares a device color space class (`'RGB '`,
/// `'GRAY'`, …), and consumers enforce it: libpng rejects a gray PNG paired
/// with an RGB-class profile. So a gray image can't reuse
/// [`synthesize_icc_for_cicp`]'s output — it needs this single-channel
/// variant: `kTRC` = the CICP transfer's tone curve (byte-identical to the
/// curve the RGB synthesis uses for the same transfer, including the PQ/HLG
/// `curv`-LUT encodings), media white point = the primaries' H.273 white
/// (D65 / C / E / DCI — the only thing primaries contribute to single-channel
/// data), and a per-white Bradford white→D50 `chad`. CICP matrix
/// coefficients are ignored (no chroma to de-correlate).
///
/// Gray keeps all 11 assigned primaries, including primaries 10: only its
/// equal-energy white and transfer are needed, not an RGB gamut matrix.
/// The 175 non-default pairs are served
/// from the same bundled, LZ4-compressed asset as the RGB profiles (64
/// unique gray profiles after white-point dedup, packed gray-first into the
/// shared per-transfer groups so the duplicated TRC payload compresses away
/// — the gray side adds ~7.6 KB to the committed blob; no CMS required,
/// `no_std`-safe); the sRGB / unspecified defaults return
/// [`NotNeeded`](SynthesizedIcc::NotNeeded) (viewers assume sRGB-gamma gray —
/// PNG can carry `gAMA`/`sRGB` instead); off-grid code points return
/// [`CmsUnsupported`](SynthesizedIcc::CmsUnsupported) in every build.
///
/// The RGB variant's HDR caveat applies equally here: the PQ / P3-PQ `curv`
/// LUT encodings are faithful above ~10 nits but soften in the deep toe —
/// prefer CICP-native PQ/HLG signalling where the container carries it.
///
/// # Examples
/// ```
/// use zenpixels_convert::icc_profiles::{synthesize_gray_icc_for_cicp, SynthesizedIcc};
/// use zenpixels_convert::Cicp;
///
/// // Display-P3-described gray content → a D65-white sRGB-TRC gray profile.
/// assert!(matches!(
///     synthesize_gray_icc_for_cicp(Cicp::DISPLAY_P3),
///     SynthesizedIcc::Profile(_)
/// ));
/// // sRGB gray → no profile needed.
/// assert_eq!(
///     synthesize_gray_icc_for_cicp(Cicp::SRGB),
///     SynthesizedIcc::NotNeeded
/// );
/// ```
pub fn synthesize_gray_icc_for_cicp(cicp: Cicp) -> SynthesizedIcc {
    // Same raw-code-point default gate as the RGB variant (see its comment
    // for why the enum mapping must not be used here).
    if (matches!(cicp.color_primaries, 1 | 2) && matches!(cicp.transfer_characteristics, 2 | 13))
        || (cicp.color_primaries == 2 && cicp.transfer_characteristics == 1)
    {
        return SynthesizedIcc::NotNeeded;
    }

    // The bundle's gray table covers the full assigned grid (`icc-db`);
    // there are no curated `&'static` gray consts (nothing in the wild to
    // byte-match).
    #[cfg(feature = "icc-db")]
    {
        if let Some(bytes) = cicp_bundle::bundled_gray_profile_for_cicp(
            cicp.color_primaries,
            cicp.transfer_characteristics,
        ) {
            return SynthesizedIcc::Profile(bytes);
        }

        // Outside the assigned grid — same terminal state as the RGB
        // variant.
        SynthesizedIcc::CmsUnsupported
    }
    #[cfg(not(feature = "icc-db"))]
    {
        SynthesizedIcc::NeedsCms
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
    #[allow(deprecated)] // exercises the deprecated primaries-only lookup
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
    fn bundled_icc_profile_hits_combinations() {
        // Each const serves only the pair whose curve it carries.
        assert_eq!(
            bundled_icc_profile(ColorPrimaries::DisplayP3, TransferFunction::Srgb),
            Some(DISPLAY_P3_V4)
        );
        assert_eq!(
            bundled_icc_profile(ColorPrimaries::Bt2020, TransferFunction::Bt709),
            Some(REC2020_V4)
        );
        // Adobe RGB: pure-gamma-2.2 TRC.
        assert_eq!(
            bundled_icc_profile(ColorPrimaries::AdobeRgb, TransferFunction::Gamma22),
            Some(ADOBE_RGB)
        );
    }

    #[test]
    fn bundled_icc_profile_rejects_hdr_transfers() {
        // HDR PQ / HLG profiles aren't bundled; callers should use CICP or
        // a CMS-side generator.
        assert!(bundled_icc_profile(ColorPrimaries::Bt2020, TransferFunction::Pq).is_none());
        assert!(bundled_icc_profile(ColorPrimaries::Bt2020, TransferFunction::Hlg).is_none());
        assert!(bundled_icc_profile(ColorPrimaries::DisplayP3, TransferFunction::Pq).is_none());
        assert!(bundled_icc_profile(ColorPrimaries::DisplayP3, TransferFunction::Hlg).is_none());
        // Linear likewise isn't bundled — CICP 8 is the canonical signal.
        assert!(bundled_icc_profile(ColorPrimaries::Bt2020, TransferFunction::Linear).is_none());
        assert!(bundled_icc_profile(ColorPrimaries::DisplayP3, TransferFunction::Linear).is_none());
    }

    #[test]
    fn bundled_icc_profile_rejects_mismatched_trc() {
        // We only bundle Adobe RGB with gamma 2.2 — asking for sRGB TRC on
        // Adobe RGB primaries returns None rather than lying with a mismatched
        // curve. Callers who want a fallback use icc_profile_for_primaries.
        assert!(bundled_icc_profile(ColorPrimaries::AdobeRgb, TransferFunction::Srgb).is_none());
        assert!(bundled_icc_profile(ColorPrimaries::AdobeRgb, TransferFunction::Bt709).is_none());
        // sRGB and BT.709 are different curves: neither stands in for the
        // other (zenpixels#86: BT.2020 + sRGB got the BT.709-curve profile).
        assert!(bundled_icc_profile(ColorPrimaries::Bt2020, TransferFunction::Srgb).is_none());
        assert!(bundled_icc_profile(ColorPrimaries::DisplayP3, TransferFunction::Bt709).is_none());
        // Gamma 2.2 on DisplayP3 / Bt2020 isn't bundled either.
        assert!(
            bundled_icc_profile(ColorPrimaries::DisplayP3, TransferFunction::Gamma22).is_none()
        );
        assert!(bundled_icc_profile(ColorPrimaries::Bt2020, TransferFunction::Gamma22).is_none());
    }

    #[test]
    fn bundled_icc_profile_bt709_returns_none() {
        // Same as icc_profile_for_primaries: BT.709 / sRGB is the assumed
        // default and isn't bundled.
        assert!(bundled_icc_profile(ColorPrimaries::Bt709, TransferFunction::Srgb).is_none());
        assert!(bundled_icc_profile(ColorPrimaries::Bt709, TransferFunction::Bt709).is_none());
        assert!(bundled_icc_profile(ColorPrimaries::Unknown, TransferFunction::Unknown).is_none());
    }

    #[test]
    fn synthesize_icc_for_cicp_srgb_is_not_needed() {
        // sRGB / BT.709 is the universally-assumed default — no profile to embed.
        assert_eq!(
            synthesize_icc_for_cicp(Cicp::SRGB),
            SynthesizedIcc::NotNeeded
        );
    }

    #[test]
    fn synthesize_icc_for_cicp_display_p3_is_bundled_borrowed() {
        // Display P3 (sRGB TRC) resolves with no CMS and no allocation: a
        // borrowed profile. With `icc-db` it is the blob's cicp-tagged
        // profile, not the curated const (zenpixels#87).
        match synthesize_icc_for_cicp(Cicp::DISPLAY_P3) {
            SynthesizedIcc::Profile(Cow::Borrowed(bytes)) => {
                #[cfg(feature = "icc-db")]
                assert_ne!(bytes, DISPLAY_P3_V4);
                #[cfg(not(feature = "icc-db"))]
                assert_eq!(bytes, DISPLAY_P3_V4);
            }
            other => panic!("expected bundled borrowed Display P3 profile, got {other:?}"),
        }
    }

    /// An s15Fixed16 XYZ triple from tag `sig` of an ICC profile.
    #[cfg(feature = "icc-db")]
    fn xyz_tag(icc: &[u8], sig: &[u8; 4]) -> Option<[f64; 3]> {
        let count = u32::from_be_bytes(icc[128..132].try_into().ok()?) as usize;
        (0..count).find_map(|i| {
            let e = 132 + 12 * i;
            let entry = icc.get(e..e + 12)?;
            if &entry[..4] != sig {
                return None;
            }
            let off = u32::from_be_bytes(entry[4..8].try_into().ok()?) as usize;
            let body = icc.get(off..off + 20)?;
            let s15 = |b: &[u8]| f64::from(i32::from_be_bytes(b.try_into().unwrap())) / 65536.0;
            Some([s15(&body[8..12]), s15(&body[12..16]), s15(&body[16..20])])
        })
    }

    #[test]
    #[cfg(feature = "icc-db")]
    fn synthesized_profiles_carry_their_own_cicp_over_the_full_grid() {
        // Every synthesized profile names its own pair in a `cicp` tag
        // (matrix 0, full range), so a reader resolving the tag gets the
        // signal the caller asked for. And identification, where it
        // recognises the bytes, agrees with the pair: the curated aliases
        // (zenpixels#86) would fail here.
        const ASSIGNED_PRIMARIES: &[u8] = &[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22];
        const ASSIGNED_TRANSFERS: &[u8] =
            &[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
        let mut checked = 0;
        for &p in ASSIGNED_PRIMARIES {
            for &t in ASSIGNED_TRANSFERS {
                let asked = Cicp::new(p, t, 0, true);
                let SynthesizedIcc::Profile(bytes) = synthesize_icc_for_cicp(asked) else {
                    continue;
                };
                let tag = zenpixels::icc::extract_cicp(&bytes)
                    .unwrap_or_else(|| panic!("({p}, {t}): no cicp tag"));
                assert_eq!(tag, asked, "({p}, {t}): cicp tag names another signal");
                // Identification, where it recognises the bytes, names the
                // same primaries and transfer wherever the CICP code maps to
                // a known enum value. (Code 4, BT.470M "gamma 2.2", maps to
                // Unknown while the profile identifies as Gamma22: a mapping
                // gap, not a profile error, so Unknown fields are skipped.)
                if let Some(id) = zenpixels::icc::identify_common(&bytes) {
                    let (want_p, want_t) =
                        (asked.color_primaries_enum(), asked.transfer_function_enum());
                    if want_p != ColorPrimaries::Unknown {
                        assert_eq!(id.primaries, want_p, "({p}, {t}): identified primaries");
                    }
                    if want_t != TransferFunction::Unknown {
                        assert_eq!(id.transfer, want_t, "({p}, {t}): identified transfer");
                    }
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 159);
    }

    #[test]
    #[cfg(feature = "icc-db")]
    fn display_p3_colorants_match_libultrahdr_within_its_tolerance() {
        // libultrahdr reads a `cicp` tag first, then matches colorant tags
        // against its own table within 0.001 (`lib/src/icc.cpp`,
        // `IccHelper::tagsEqualToMatrix`; table `kDisplayP3` in
        // `lib/include/ultrahdr/icc.h`). The curated DISPLAY_P3_V4 fails both
        // (no tag; red Z 0 and blue Z 0.783081), so libultrahdr decoded
        // Display P3 Ultra HDR bases as BT.709 (zenpixels#87).
        const KDISPLAYP3: [[f64; 3]; 3] = [
            [0.515102, 0.241182, -0.00104941],
            [0.291965, 0.692236, 0.0418818],
            [0.157153, 0.0665819, 0.784378],
        ];
        for transfer in [13u8, 1] {
            let SynthesizedIcc::Profile(bytes) =
                synthesize_icc_for_cicp(Cicp::new(12, transfer, 0, true))
            else {
                panic!("Display P3 (transfer {transfer}) not synthesized");
            };
            for (sig, want) in [b"rXYZ", b"gXYZ", b"bXYZ"].iter().zip(KDISPLAYP3) {
                let got = xyz_tag(&bytes, sig).expect("colorant tag");
                for (g, w) in got.iter().zip(want) {
                    assert!(
                        (g - w).abs() <= 0.001,
                        "transfer {transfer}: {} {got:?} vs libultrahdr {want:?}",
                        core::str::from_utf8(*sig).unwrap()
                    );
                }
            }
        }
        // The curated const misses, which is why synthesis must not serve it.
        let curated = xyz_tag(DISPLAY_P3_V4, b"rXYZ").unwrap();
        assert!((curated[2] - KDISPLAYP3[0][2]).abs() > 0.001);
    }

    #[test]
    #[cfg(feature = "icc-db")]
    fn sdr_bt2020_with_srgb_transfer_gets_the_srgb_curve() {
        // zenpixels#86: (BT.2020, sRGB) got REC2020_V4, whose curve is
        // BT.709's. The synthesized profile names (9, 13) and is not the
        // BT.709-curve const.
        let SynthesizedIcc::Profile(bytes) = synthesize_icc_for_cicp(Cicp::new(9, 13, 0, true))
        else {
            panic!("BT.2020 + sRGB not synthesized");
        };
        assert_ne!(&*bytes, REC2020_V4);
        assert_eq!(
            zenpixels::icc::extract_cicp(&bytes),
            Some(Cicp::new(9, 13, 0, true))
        );
    }

    #[test]
    fn synthesize_icc_for_cicp_hdr_pq_is_bundled_in_every_build() {
        // BT.2020 PQ: an HDR transfer with no `&'static` const, but it IS in the
        // full-grid blob now — so a profile comes back in BOTH builds, never a
        // mis-tagged SDR profile and never `NeedsCms`. (The bytes are the
        // blob's; byte-equality with moxcms is verified by the roundtrip test.)
        let bt2020_pq = Cicp::new(9, 16, 9, false); // primaries 9, transfer 16 (PQ)
        match synthesize_icc_for_cicp(bt2020_pq) {
            SynthesizedIcc::Profile(bytes) => {
                assert!(!bytes.is_empty());
                assert_eq!(&bytes[36..40], b"acsp", "must be a valid ICC profile");
            }
            other => panic!("expected a bundled PQ profile, got {other:?}"),
        }
    }

    #[test]
    fn synthesize_icc_for_cicp_reserved_transfer_is_unsupported_not_mistagged() {
        // Recognized primaries (BT.2020) but a *reserved* transfer code (0). It's
        // outside the assigned H.273 grid, so it isn't in the blob — synthesis
        // must refuse rather than mis-tag. The blob is the coverage source now,
        // so a no-CMS build reports `CmsUnsupported` (not `NeedsCms`): there is
        // no CMS that could add what the blob lacks for an unassigned code.
        let bt2020_reserved_trc = Cicp::new(9, 0, 9, false);
        assert_eq!(
            synthesize_icc_for_cicp(bt2020_reserved_trc),
            SynthesizedIcc::CmsUnsupported
        );
    }

    #[test]
    fn synthesize_icc_for_cicp_real_gamut_not_assumed_srgb() {
        // BT.601 primaries (code 6) + gamma-2.2 transfer (code 4): a real, non-sRGB
        // colour description that zenpixels' enum folds into `Unknown`/`Unknown`.
        // It must NOT be treated as the sRGB default — that would silently drop the
        // gamut. The `NotNeeded` gate keys on raw code points, and it's an assigned
        // grid point, so it resolves from the blob in EVERY build.
        let bt601_g22 = Cicp::new(6, 4, 0, false);
        let got = synthesize_icc_for_cicp(bt601_g22);
        assert_ne!(
            got,
            SynthesizedIcc::NotNeeded,
            "a real non-sRGB gamut must never be assumed to be the sRGB default"
        );
        assert!(
            matches!(got, SynthesizedIcc::Profile(_)),
            "an assigned H.273 grid point must resolve from the blob in every build, got {got:?}"
        );
    }

    #[test]
    fn synthesize_icc_for_cicp_full_grid_coverage_without_cms() {
        // The headline property: the blob gives a profile for every assigned-grid
        // combo except primaries 10 and the sRGB-default `NotNeeded` pair — with no dependence
        // on a CMS. The blob lookup sits *before* the moxcms fallback, so every
        // covered combo comes back as a `Cow::Borrowed` (the blob's zero-copy
        // slice, or one of the 4 `&'static` consts) — never a `Cow::Owned`, which
        // is what moxcms would produce. Asserting `Borrowed` proves the no-CMS
        // path serves these regardless of whether `cms-moxcms` is compiled in
        // (it's forced on for this crate's tests by a dev-dependency).
        const ASSIGNED_PRIMARIES: &[u8] = &[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22];
        const ASSIGNED_TRANSFERS: &[u8] =
            &[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
        let mut profiles = 0usize;
        let mut not_needed = 0usize;
        let mut unsupported = 0usize;
        for &p in ASSIGNED_PRIMARIES {
            for &t in ASSIGNED_TRANSFERS {
                let got = synthesize_icc_for_cicp(Cicp::new(p, t, 0, true));
                match got {
                    SynthesizedIcc::Profile(bytes) => {
                        assert!(
                            matches!(bytes, Cow::Borrowed(_)),
                            "({p}, {t}) was served by the CMS (Cow::Owned) — must come \
                             from the blob/const (Cow::Borrowed) so coverage holds without a CMS"
                        );
                        assert_eq!(&bytes[36..40], b"acsp", "({p}, {t}) produced non-ICC bytes");
                        profiles += 1;
                    }
                    SynthesizedIcc::NotNeeded => {
                        // Only the BT.709 sRGB-default pairs.
                        assert!(p == 1 && t == 13, "({p}, {t}) unexpectedly NotNeeded");
                        not_needed += 1;
                    }
                    SynthesizedIcc::CmsUnsupported if p == 10 => unsupported += 1,
                    other => panic!("({p}, {t}) gave {other:?} — expected full blob coverage"),
                }
            }
        }
        assert_eq!(profiles, 159, "expected 159 profile-yielding combos");
        assert_eq!(unsupported, 16, "all primaries-10 RGB pairs refused");
        assert_eq!(
            not_needed, 1,
            "expected exactly 1 sRGB-default NotNeeded combo"
        );
    }
}
