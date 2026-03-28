//! Embedded CC0-licensed ICC profiles for common wide-gamut color spaces.
//!
//! All profiles are from [Compact-ICC-Profiles](https://github.com/saucecontrol/Compact-ICC-Profiles)
//! by Clinton Ingram, released under the [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
//! public domain dedication. They are embedded at compile time via `include_bytes!()`.
//!
//! # Available profiles
//!
//! | Constant | Color space | Format | Size |
//! |----------|-------------|--------|------|
//! | [`DISPLAY_P3_V4`] | Display P3 | ICC v4 | 480 bytes |
//! | [`DISPLAY_P3_V2`] | Display P3 | ICC v2 | 736 bytes |
//! | [`ADOBE_RGB_V4`] | Adobe RGB (1998) | ICC v4 | 480 bytes |
//! | [`REC2020_V4`] | Rec. 2020 | ICC v4 | 480 bytes |
//! | [`PROPHOTO_V4`] | ProPhoto RGB | ICC v4 | 480 bytes |
//!
//! # Lookup by primaries
//!
//! Use [`icc_profile_for_primaries`] to get the recommended ICC profile bytes
//! for a [`ColorPrimaries`] value. Returns `None` for `Bt709` (sRGB is assumed
//! by default and rarely needs an explicit ICC profile) and `Unknown`.
//!
//! # Precision warnings
//!
//! Rec. 2020 and ProPhoto RGB have very wide gamuts. Using 8-bit precision
//! with these spaces will cause visible banding in gradients. Use 16-bit
//! or f32 precision when working with these profiles.

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

/// Adobe RGB (1998) Compatible ICC profile, v4 format (480 bytes).
///
/// Adobe RGB has a wider gamut than sRGB, particularly in cyan-green.
/// Common in photography workflows and print production.
///
/// Source: <https://github.com/saucecontrol/Compact-ICC-Profiles> (CC0)
pub const ADOBE_RGB_V4: &[u8] = include_bytes!("profiles/AdobeCompat-v4.icc");

/// Rec. 2020 Compatible ICC profile, v4 format (480 bytes).
///
/// Rec. 2020 has a very wide gamut (~75% of visible colors).
/// **Use 16-bit or f32 precision** to avoid banding.
///
/// Source: <https://github.com/saucecontrol/Compact-ICC-Profiles> (CC0)
pub const REC2020_V4: &[u8] = include_bytes!("profiles/Rec2020Compat-v4.icc");

/// ProPhoto RGB ICC profile, v4 format (480 bytes).
///
/// ProPhoto RGB has an extremely wide gamut (~90% of visible colors).
/// Some ProPhoto colors are outside human vision (imaginary colors).
/// **Use 16-bit or f32 precision** to avoid severe banding.
///
/// ProPhoto is used as an editing space in Lightroom and other RAW processors.
/// Convert to a smaller gamut (P3, sRGB) before final output.
///
/// Source: <https://github.com/saucecontrol/Compact-ICC-Profiles> (CC0)
pub const PROPHOTO_V4: &[u8] = include_bytes!("profiles/ProPhoto-v4.icc");

/// Get the recommended ICC profile for a set of color primaries.
///
/// Returns the v4 ICC profile bytes for the given primaries, or `None` if
/// no embedded profile is available. Returns `None` for [`ColorPrimaries::Bt709`]
/// because sRGB is the assumed default and rarely needs an explicit ICC profile.
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
        // BT.709/sRGB is the assumed default; no explicit ICC profile needed.
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
            (ADOBE_RGB_V4, "Adobe RGB v4"),
            (REC2020_V4, "Rec. 2020 v4"),
            (PROPHOTO_V4, "ProPhoto v4"),
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
        assert!(icc_profile_for_primaries(ColorPrimaries::Bt709).is_none());
        assert!(icc_profile_for_primaries(ColorPrimaries::Unknown).is_none());
    }
}
