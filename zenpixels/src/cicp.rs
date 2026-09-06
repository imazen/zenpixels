//! CICP (Coding-Independent Code Points) color description.
//!
//! ITU-T H.273 / ISO 23091-2 defines code points for color primaries,
//! transfer characteristics, and matrix coefficients. This struct
//! carries the four fields needed by [`ColorContext`](crate::color::ColorContext).

use core::fmt;

use crate::{ColorPrimaries, TransferFunction};

/// CICP color description (ITU-T H.273).
///
/// Coding-Independent Code Points describe the color space of an image
/// without requiring an ICC profile. Used by AVIF, HEIF, JPEG XL, and
/// video codecs (H.264, H.265, AV1).
///
/// Common combinations for RGB content (matrix_coefficients = 0 = Identity):
/// - sRGB: `(1, 13, 0, true)` — BT.709 primaries, sRGB transfer
/// - Display P3: `(12, 13, 0, true)` — P3 primaries, sRGB transfer
/// - BT.2100 PQ (HDR): `(9, 16, 0, true)` — BT.2020 primaries, PQ transfer
/// - BT.2100 HLG (HDR): `(9, 18, 0, true)` — BT.2020 primaries, HLG transfer
///
/// Video/YCbCr content uses non-zero matrix_coefficients (e.g., 6=BT.601, 9=BT.2020).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Cicp {
    /// Color primaries (ColourPrimaries). Common values:
    /// 1 = BT.709/sRGB, 9 = BT.2020, 12 = Display P3.
    pub color_primaries: u8,
    /// Transfer characteristics (TransferCharacteristics). Common values:
    /// 1 = BT.709, 13 = sRGB, 16 = PQ (HDR), 18 = HLG (HDR).
    pub transfer_characteristics: u8,
    /// Matrix coefficients (MatrixCoefficients). Common values:
    /// 0 = Identity/RGB, 1 = BT.709, 6 = BT.601, 9 = BT.2020.
    pub matrix_coefficients: u8,
    /// Whether pixel values use the full range (0-255 for 8-bit)
    /// or video/limited range (16-235 for 8-bit luma).
    pub full_range: bool,
}

impl Cicp {
    /// Create a CICP color description from raw code points.
    pub const fn new(
        color_primaries: u8,
        transfer_characteristics: u8,
        matrix_coefficients: u8,
        full_range: bool,
    ) -> Self {
        Self {
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            full_range,
        }
    }

    /// sRGB color space: BT.709 primaries, sRGB transfer, Identity (RGB) matrix, full range.
    pub const SRGB: Self = Self {
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
        full_range: true,
    };

    /// BT.2100 PQ (HDR10): BT.2020 primaries, PQ transfer, BT.2020 matrix, full range.
    pub const BT2100_PQ: Self = Self {
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
        full_range: true,
    };

    /// BT.2100 HLG (HDR): BT.2020 primaries, HLG transfer, BT.2020 matrix, full range.
    pub const BT2100_HLG: Self = Self {
        color_primaries: 9,
        transfer_characteristics: 18,
        matrix_coefficients: 9,
        full_range: true,
    };

    /// Display P3 with sRGB transfer: P3 primaries, sRGB transfer, Identity matrix, full range.
    pub const DISPLAY_P3: Self = Self {
        color_primaries: 12,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
        full_range: true,
    };

    /// Map the CICP `color_primaries` code to a [`ColorPrimaries`] enum.
    ///
    /// Returns [`Unknown`](ColorPrimaries::Unknown) for unrecognized codes.
    /// This is a convenience wrapper around [`ColorPrimaries::from_cicp`].
    pub fn color_primaries_enum(&self) -> ColorPrimaries {
        ColorPrimaries::from_cicp(self.color_primaries).unwrap_or(ColorPrimaries::Unknown)
    }

    /// Map the CICP `transfer_characteristics` code to a [`TransferFunction`] enum.
    ///
    /// Returns [`Unknown`](TransferFunction::Unknown) for unrecognized codes.
    /// This is a convenience wrapper around [`TransferFunction::from_cicp`].
    pub fn transfer_function_enum(&self) -> TransferFunction {
        TransferFunction::from_cicp(self.transfer_characteristics)
            .unwrap_or(TransferFunction::Unknown)
    }

    /// Create a CICP from a [`PixelDescriptor`](crate::PixelDescriptor).
    ///
    /// Returns `None` if the descriptor's transfer function or color primaries
    /// cannot be mapped to CICP code points (e.g., `Unknown` variants).
    pub fn from_descriptor(desc: &crate::PixelDescriptor) -> Option<Self> {
        let tc = desc.transfer.to_cicp()?;
        let cp = desc.primaries.to_cicp()?;
        let full_range = matches!(desc.signal_range, crate::SignalRange::Full);
        Some(Self {
            color_primaries: cp,
            transfer_characteristics: tc,
            matrix_coefficients: 0, // RGB content uses Identity
            full_range,
        })
    }

    /// Resolve the `matrix_coefficients` field to a concrete, decoder-usable value.
    ///
    /// Rec. ITU-T H.273 (V4) (07/2024), Table 4, defines three categories of
    /// `matrix_coefficients` (MC):
    ///
    /// | Category | MC values | Action |
    /// |---|---|---|
    /// | Self-contained | 0, 1, 4–11, 14–17 | `Ok(self)` — recipe is fixed |
    /// | Chromaticity-derived | 12, 13 | `Ok(self)` if `color_primaries` has known chromaticity; `Err` otherwise |
    /// | Unspecified / Reserved | 2, 3, 18–255 | `Ok` with `hint_mc` substituted in; `Err` if no valid hint |
    ///
    /// # MC=0 (Identity)
    ///
    /// MC=0 means the planes are already in RGB order — no matrix is applied.
    /// `Ok(self)` is returned without consuming the hint.
    ///
    /// # MC=12/13 (Chroma NCL / Chroma CL)
    ///
    /// These matrices are derived from `color_primaries` via KR/KB computation
    /// (H.273 §8.3). They are only resolvable when `color_primaries` carries
    /// defined chromaticity coordinates (H.273 Table 2: codes 1, 4–12, 22).
    /// If `color_primaries` is 0 (Reserved) or 2 (Unspecified) or an
    /// otherwise undefined code, `Err` is returned — the hint is **not**
    /// consumed for MC=12/13 (the matrix is fully specified by CP; a hint
    /// cannot override that).
    ///
    /// # MC=2 / Reserved (2, 3, 18–255)
    ///
    /// MC=2 (Unspecified) and reserved values require external disambiguation.
    /// If `hint_mc` is `Some(h)` and `h` is itself self-contained (category 1
    /// above) or chromaticity-derived with a valid CP, the returned `Cicp` has
    /// `matrix_coefficients` replaced with `h`. Otherwise `Err` is returned.
    ///
    /// # MC=15/16/17 (IPT-C2 / YCgCo-Re / YCgCo-Ro)
    ///
    /// These code points were reserved in editions up to H.273 (07/2021) but
    /// are **defined, self-contained recipes** since H.273 (09/2023): 15 is
    /// SMPTE IPT-PQ-C2 (equations 85–87), 16/17 are the even/odd-bit-shift
    /// reversible YCgCo variants (equations 58–65). They resolve to
    /// `Ok(self)` and are never hint-substituted. References written against
    /// pre-2023 editions list them as reserved — the pinned edition here is
    /// V4 (07/2024), where only 3 and 18–255 remain reserved.
    ///
    /// # Resolved ≠ Supported
    ///
    /// `resolve_matrix` normalizes **signaling**. A successful return means the
    /// matrix is unambiguous, not that the caller implements the corresponding
    /// decode math. Receiving `Ok` with MC=13 (Chroma CL), MC=14 (ICtCp), or
    /// MC=15 (IPT-C2) does not imply those transforms exist in the decoder —
    /// the caller must still reject or handle unsupported matrices explicitly.
    ///
    /// # Example
    ///
    /// ```
    /// use zenpixels::Cicp;
    ///
    /// // MC=2 (Unspecified) + hint → resolved
    /// let unspecified = Cicp::new(9, 16, 2, true);
    /// let resolved = unspecified.resolve_matrix(Some(9)).unwrap();
    /// assert_eq!(resolved.matrix_coefficients, 9);
    ///
    /// // MC=2 without hint → error
    /// assert!(unspecified.resolve_matrix(None).is_err());
    ///
    /// // MC=0 (Identity) → self-contained, no hint needed
    /// assert!(Cicp::SRGB.resolve_matrix(None).is_ok());
    /// ```
    pub const fn resolve_matrix(self, hint_mc: Option<u8>) -> Result<Self, UnspecifiedMatrixError> {
        let mc = self.matrix_coefficients;

        if mc_is_self_contained(mc) {
            return Ok(self);
        }

        if mc_is_chromaticity_derived(mc) {
            // MC=12/13: valid only when color_primaries has known chromaticity.
            // The hint is not used here — a hint cannot override a specified
            // chromaticity-derived matrix.
            if cp_has_known_chromaticity(self.color_primaries) {
                return Ok(self);
            }
            return Err(UnspecifiedMatrixError {
                color_primaries: self.color_primaries,
                transfer_characteristics: self.transfer_characteristics,
                matrix_coefficients: mc,
            });
        }

        // MC=2 (Unspecified) or reserved (3, 18–255).
        // Try to apply the hint.
        match hint_mc {
            Some(h) if mc_is_self_contained(h) => Ok(Self {
                matrix_coefficients: h,
                ..self
            }),
            Some(h) if mc_is_chromaticity_derived(h) => {
                if cp_has_known_chromaticity(self.color_primaries) {
                    Ok(Self {
                        matrix_coefficients: h,
                        ..self
                    })
                } else {
                    Err(UnspecifiedMatrixError {
                        color_primaries: self.color_primaries,
                        transfer_characteristics: self.transfer_characteristics,
                        matrix_coefficients: mc,
                    })
                }
            }
            _ => Err(UnspecifiedMatrixError {
                color_primaries: self.color_primaries,
                transfer_characteristics: self.transfer_characteristics,
                matrix_coefficients: mc,
            }),
        }
    }

    /// Convert to a [`PixelDescriptor`](crate::PixelDescriptor) with the given
    /// [`PixelFormat`](crate::PixelFormat).
    ///
    /// Maps the CICP code points to the corresponding enum variants.
    /// Unmapped codes become `Unknown`.
    pub fn to_descriptor(&self, format: crate::PixelFormat) -> crate::PixelDescriptor {
        let transfer = self.transfer_function_enum();
        let primaries = self.color_primaries_enum();
        let signal_range = if self.full_range {
            crate::SignalRange::Full
        } else {
            crate::SignalRange::Narrow
        };
        // Derive alpha from the pixel format's channel layout.
        let alpha = if format.layout().has_alpha() {
            Some(crate::AlphaMode::Straight)
        } else {
            None
        };
        crate::PixelDescriptor {
            format,
            transfer,
            alpha,
            primaries,
            signal_range,
        }
    }

    /// Human-readable name for the color primaries code (ITU-T H.273 Table 2).
    pub fn color_primaries_name(code: u8) -> &'static str {
        match code {
            0 => "Reserved",
            1 => "BT.709/sRGB",
            2 => "Unspecified",
            4 => "BT.470M",
            5 => "BT.601 (625)",
            6 => "BT.601 (525)",
            7 => "SMPTE 240M",
            8 => "Generic Film",
            9 => "BT.2020",
            10 => "XYZ",
            11 => "SMPTE 431 (DCI-P3)",
            12 => "Display P3",
            22 => "EBU Tech 3213",
            _ => "Unknown",
        }
    }

    /// Human-readable name for the transfer characteristics code (ITU-T H.273 Table 3).
    pub fn transfer_characteristics_name(code: u8) -> &'static str {
        match code {
            0 => "Reserved",
            1 => "BT.709",
            2 => "Unspecified",
            4 => "BT.470M (Gamma 2.2)",
            5 => "BT.470BG (Gamma 2.8)",
            6 => "BT.601",
            7 => "SMPTE 240M",
            8 => "Linear",
            9 => "Log 100:1",
            10 => "Log 316:1",
            11 => "IEC 61966-2-4",
            12 => "BT.1361",
            13 => "sRGB",
            14 => "BT.2020 (10-bit)",
            15 => "BT.2020 (12-bit)",
            16 => "PQ (HDR)",
            17 => "SMPTE 428",
            18 => "HLG (HDR)",
            _ => "Unknown",
        }
    }

    /// Human-readable name for the matrix coefficients code (ITU-T H.273 Table 4).
    pub fn matrix_coefficients_name(code: u8) -> &'static str {
        match code {
            0 => "Identity/RGB",
            1 => "BT.709",
            2 => "Unspecified",
            4 => "FCC",
            5 => "BT.470BG",
            6 => "BT.601",
            7 => "SMPTE 240M",
            8 => "YCgCo",
            9 => "BT.2020 NCL",
            10 => "BT.2020 CL",
            11 => "SMPTE 2085",
            12 => "Chroma NCL",
            13 => "Chroma CL",
            14 => "ICtCp",
            15 => "IPT-C2",
            16 => "YCgCo-Re",
            17 => "YCgCo-Ro",
            _ => "Unknown",
        }
    }
}

// ── resolve_matrix helpers ────────────────────────────────────────────────

/// Returns `true` for MC values that are fully self-contained
/// (Rec. ITU-T H.273 (V4) (07/2024), Table 4).
///
/// These are `Ok(self)` in [`Cicp::resolve_matrix`]: the matrix recipe is
/// fixed by the spec and does not depend on `color_primaries`.
///
/// 15 (IPT-C2), 16 (YCgCo-Re), and 17 (YCgCo-Ro) are defined since the
/// 09/2023 edition; only 3 and 18–255 remain reserved in V4.
const fn mc_is_self_contained(mc: u8) -> bool {
    matches!(
        mc,
        0 | 1 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 14 | 15 | 16 | 17
    )
}

/// Returns `true` for MC=12 (Chroma NCL) and MC=13 (Chroma CL).
///
/// These are chromaticity-derived: KR/KB are computed from `color_primaries`
/// (H.273 §8.3). They cannot be resolved without a known CP.
const fn mc_is_chromaticity_derived(mc: u8) -> bool {
    matches!(mc, 12 | 13)
}

/// Returns `true` when the CP code carries known chromaticity coordinates
/// (H.273 Table 2), making MC=12/13 derivation valid.
///
/// CP=0 (Reserved) and CP=2 (Unspecified) have no coordinates.
/// CP=3 and CP=13–21 and CP=23–255 are reserved/undefined.
const fn cp_has_known_chromaticity(cp: u8) -> bool {
    matches!(cp, 1 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 22)
}

// ── UnspecifiedMatrixError ────────────────────────────────────────────────

/// Error returned by [`Cicp::resolve_matrix`] when the matrix coefficients
/// cannot be determined without additional information.
///
/// Occurs when:
/// - `matrix_coefficients` is 2 (Unspecified per H.273) and no valid hint was supplied.
/// - `matrix_coefficients` is reserved (3 or 18–255 per H.273 V4 (07/2024)) and no valid hint was supplied.
/// - `matrix_coefficients` is 12 or 13 (chromaticity-derived; H.273 §8.3) and
///   `color_primaries` is 0 (Reserved), 2 (Unspecified), or otherwise has no
///   defined chromaticity coordinates in H.273 Table 2.
#[derive(Debug)]
#[non_exhaustive]
pub struct UnspecifiedMatrixError {
    color_primaries: u8,
    transfer_characteristics: u8,
    matrix_coefficients: u8,
}

impl UnspecifiedMatrixError {
    /// The CICP `color_primaries` of the input that failed to resolve.
    pub fn color_primaries(&self) -> u8 {
        self.color_primaries
    }

    /// The CICP `transfer_characteristics` of the input that failed to resolve.
    pub fn transfer_characteristics(&self) -> u8 {
        self.transfer_characteristics
    }

    /// The CICP `matrix_coefficients` of the input that failed to resolve —
    /// the unresolvable signaled value itself (2/reserved with no usable
    /// hint, or 12/13 with an underivable `color_primaries`), never the
    /// rejected hint.
    pub fn matrix_coefficients(&self) -> u8 {
        self.matrix_coefficients
    }
}

impl fmt::Display for UnspecifiedMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "matrix_coefficients={} cannot be resolved \
             (color_primaries={}, transfer_characteristics={})",
            self.matrix_coefficients, self.color_primaries, self.transfer_characteristics,
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UnspecifiedMatrixError {}

// ── Cicp impl ─────────────────────────────────────────────────────────────

impl fmt::Display for Cicp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} / {} / {} ({})",
            Self::color_primaries_name(self.color_primaries),
            Self::transfer_characteristics_name(self.transfer_characteristics),
            Self::matrix_coefficients_name(self.matrix_coefficients),
            if self.full_range {
                "full range"
            } else {
                "limited range"
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn cicp_new() {
        let c = Cicp::new(1, 13, 0, true);
        assert_eq!(c, Cicp::SRGB);
    }

    #[test]
    fn cicp_constants() {
        assert_eq!(Cicp::SRGB.color_primaries, 1);
        assert_eq!(Cicp::SRGB.transfer_characteristics, 13);
        assert_eq!(Cicp::BT2100_PQ.transfer_characteristics, 16);
        assert_eq!(Cicp::BT2100_HLG.transfer_characteristics, 18);
        assert_eq!(Cicp::DISPLAY_P3.color_primaries, 12);
    }

    #[test]
    fn color_primaries_enum() {
        assert_eq!(Cicp::SRGB.color_primaries_enum(), ColorPrimaries::Bt709);
        assert_eq!(
            Cicp::BT2100_PQ.color_primaries_enum(),
            ColorPrimaries::Bt2020
        );
        assert_eq!(
            Cicp::DISPLAY_P3.color_primaries_enum(),
            ColorPrimaries::DisplayP3
        );
        assert_eq!(
            Cicp::new(255, 0, 0, true).color_primaries_enum(),
            ColorPrimaries::Unknown
        );
    }

    #[test]
    fn transfer_function_enum() {
        assert_eq!(Cicp::SRGB.transfer_function_enum(), TransferFunction::Srgb);
        assert_eq!(
            Cicp::BT2100_PQ.transfer_function_enum(),
            TransferFunction::Pq
        );
        assert_eq!(
            Cicp::BT2100_HLG.transfer_function_enum(),
            TransferFunction::Hlg
        );
        assert_eq!(
            Cicp::new(1, 255, 0, true).transfer_function_enum(),
            TransferFunction::Unknown
        );
    }

    #[test]
    fn color_primaries_name_known() {
        assert_eq!(Cicp::color_primaries_name(0), "Reserved");
        assert_eq!(Cicp::color_primaries_name(1), "BT.709/sRGB");
        assert_eq!(Cicp::color_primaries_name(9), "BT.2020");
        assert_eq!(Cicp::color_primaries_name(12), "Display P3");
        assert_eq!(Cicp::color_primaries_name(200), "Unknown");
    }

    #[test]
    fn transfer_characteristics_name_known() {
        assert_eq!(Cicp::transfer_characteristics_name(8), "Linear");
        assert_eq!(Cicp::transfer_characteristics_name(13), "sRGB");
        assert_eq!(Cicp::transfer_characteristics_name(16), "PQ (HDR)");
        assert_eq!(Cicp::transfer_characteristics_name(18), "HLG (HDR)");
        assert_eq!(Cicp::transfer_characteristics_name(200), "Unknown");
    }

    #[test]
    fn matrix_coefficients_name_known() {
        assert_eq!(Cicp::matrix_coefficients_name(0), "Identity/RGB");
        assert_eq!(Cicp::matrix_coefficients_name(1), "BT.709");
        assert_eq!(Cicp::matrix_coefficients_name(9), "BT.2020 NCL");
        assert_eq!(Cicp::matrix_coefficients_name(200), "Unknown");
    }

    #[test]
    fn display_srgb() {
        let s = format!("{}", Cicp::SRGB);
        assert!(s.contains("BT.709/sRGB"));
        assert!(s.contains("sRGB"));
        assert!(s.contains("full range"));
    }

    #[test]
    fn display_limited_range() {
        let c = Cicp::new(1, 1, 1, false);
        let s = format!("{c}");
        assert!(s.contains("limited range"));
    }

    #[test]
    fn debug_and_clone() {
        let c = Cicp::SRGB;
        let _ = format!("{c:?}");
        let c2 = c;
        assert_eq!(c, c2);
    }

    #[test]
    #[cfg(feature = "std")]
    fn hash() {
        use core::hash::{Hash, Hasher};
        let mut h1 = std::hash::DefaultHasher::new();
        Cicp::SRGB.hash(&mut h1);
        let mut h2 = std::hash::DefaultHasher::new();
        Cicp::SRGB.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn from_descriptor_srgb() {
        use crate::{AlphaMode, PixelDescriptor, PixelFormat, SignalRange};
        let desc = PixelDescriptor {
            format: PixelFormat::Rgba8,
            transfer: TransferFunction::Srgb,
            alpha: Some(AlphaMode::Straight),
            primaries: ColorPrimaries::Bt709,
            signal_range: SignalRange::Full,
        };
        let cicp = Cicp::from_descriptor(&desc).unwrap();
        assert_eq!(cicp, Cicp::SRGB);
    }

    #[test]
    fn from_descriptor_unknown_returns_none() {
        use crate::{PixelDescriptor, PixelFormat, SignalRange};
        let desc = PixelDescriptor {
            format: PixelFormat::Rgb8,
            transfer: TransferFunction::Unknown,
            alpha: None,
            primaries: ColorPrimaries::Bt709,
            signal_range: SignalRange::Full,
        };
        assert!(Cicp::from_descriptor(&desc).is_none());
    }

    #[test]
    fn to_descriptor_srgb_rgba() {
        use crate::{AlphaMode, PixelFormat, SignalRange};
        let desc = Cicp::SRGB.to_descriptor(PixelFormat::Rgba8);
        assert_eq!(desc.format, PixelFormat::Rgba8);
        assert_eq!(desc.transfer, TransferFunction::Srgb);
        assert_eq!(desc.primaries, ColorPrimaries::Bt709);
        assert_eq!(desc.alpha, Some(AlphaMode::Straight));
        assert_eq!(desc.signal_range, SignalRange::Full);
    }

    #[test]
    fn to_descriptor_narrow_range() {
        use crate::{PixelFormat, SignalRange};
        let cicp = Cicp::new(1, 13, 0, false);
        let desc = cicp.to_descriptor(PixelFormat::Rgb8);
        assert_eq!(desc.signal_range, SignalRange::Narrow);
        assert!(desc.alpha.is_none());
    }

    #[test]
    fn descriptor_roundtrip() {
        use crate::PixelFormat;
        for cicp in [
            Cicp::SRGB,
            Cicp::BT2100_PQ,
            Cicp::BT2100_HLG,
            Cicp::DISPLAY_P3,
        ] {
            let desc = cicp.to_descriptor(PixelFormat::Rgb8);
            let back = Cicp::from_descriptor(&desc).unwrap();
            assert_eq!(back.color_primaries, cicp.color_primaries);
            assert_eq!(back.transfer_characteristics, cicp.transfer_characteristics);
            assert_eq!(back.full_range, cicp.full_range);
        }
    }

    // ── resolve_matrix tests ──────────────────────────────────────────────

    /// MC=0 (Identity) is self-contained — no hint needed, returns self.
    #[test]
    fn resolve_matrix_identity() {
        let c = Cicp::SRGB; // MC=0
        assert_eq!(c.matrix_coefficients, 0);
        let r = c.resolve_matrix(None).unwrap();
        assert_eq!(r, c);
    }

    /// All self-contained MC codes return Ok(self) with no hint.
    /// 15 (IPT-C2), 16 (YCgCo-Re), 17 (YCgCo-Ro) are defined recipes since
    /// H.273 (09/2023) — self-contained, never hint-substituted.
    #[test]
    fn resolve_matrix_self_contained() {
        for mc in [0u8, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17] {
            let c = Cicp::new(1, 13, mc, true);
            let r = c.resolve_matrix(None);
            assert!(r.is_ok(), "MC={mc} should be self-contained");
            assert_eq!(r.unwrap(), c, "MC={mc} resolve_matrix should return self");
        }
    }

    /// MC=14 (ICtCp) is self-contained (BT.2100 fixed recipe, not chromaticity-derived).
    #[test]
    fn resolve_matrix_ictcp_is_self_contained() {
        let c = Cicp::new(9, 16, 14, true); // BT.2020 PQ, ICtCp matrix
        let r = c.resolve_matrix(None).unwrap();
        assert_eq!(r.matrix_coefficients, 14);
    }

    /// MC=9 (BT.2020 NCL) with CP=9 is the canonical HDR pair — self-contained.
    #[test]
    fn resolve_matrix_bt2020_ncl_canonical() {
        let c = Cicp::BT2100_PQ; // CP=9, TC=16, MC=9
        let r = c.resolve_matrix(None).unwrap();
        assert_eq!(r, c);
    }

    /// MC=12 (Chroma NCL, chromaticity-derived) with a known CP → Ok(self).
    #[test]
    fn resolve_matrix_chroma_ncl_known_cp() {
        for cp in [1u8, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22] {
            let c = Cicp::new(cp, 1, 12, false);
            let r = c.resolve_matrix(None);
            assert!(r.is_ok(), "CP={cp}, MC=12 should resolve Ok");
            assert_eq!(r.unwrap(), c);
        }
    }

    /// MC=13 (Chroma CL, chromaticity-derived) with a known CP → Ok(self).
    #[test]
    fn resolve_matrix_chroma_cl_known_cp() {
        let c = Cicp::new(9, 16, 13, false);
        let r = c.resolve_matrix(None).unwrap();
        assert_eq!(r.matrix_coefficients, 13);
    }

    /// MC=12 with CP=2 (Unspecified) → Err; hint is not consumed.
    #[test]
    fn resolve_matrix_chroma_ncl_unspecified_cp_is_err() {
        let c = Cicp::new(2, 1, 12, false);
        let err = c.resolve_matrix(Some(6)).unwrap_err();
        // hint is NOT applied for MC=12/13 — CP is what's unresolvable
        assert_eq!(err.color_primaries(), 2);
        assert_eq!(err.transfer_characteristics(), 1);
    }

    /// MC=12 with CP=0 (Reserved) → Err.
    #[test]
    fn resolve_matrix_chroma_ncl_reserved_cp_is_err() {
        let c = Cicp::new(0, 1, 12, false);
        assert!(c.resolve_matrix(None).is_err());
    }

    /// MC=12 with an unassigned/reserved CP code → Err.
    #[test]
    fn resolve_matrix_chroma_ncl_unknown_cp_is_err() {
        for cp in [3u8, 13, 20, 100, 255] {
            let c = Cicp::new(cp, 1, 12, false);
            assert!(
                c.resolve_matrix(None).is_err(),
                "CP={cp} has no known chromaticity; MC=12 should be Err"
            );
        }
    }

    /// MC=2 (Unspecified) with a valid self-contained hint → resolved.
    #[test]
    fn resolve_matrix_unspecified_with_hint() {
        let c = Cicp::new(9, 16, 2, true);
        let r = c.resolve_matrix(Some(9)).unwrap();
        assert_eq!(r.matrix_coefficients, 9);
        assert_eq!(r.color_primaries, 9);
        assert_eq!(r.transfer_characteristics, 16);
        assert!(r.full_range);
    }

    /// MC=2 with no hint → Err, with correct CP/TC in the error.
    #[test]
    fn resolve_matrix_unspecified_no_hint_is_err() {
        let c = Cicp::new(1, 13, 2, true);
        let err = c.resolve_matrix(None).unwrap_err();
        assert_eq!(err.color_primaries(), 1);
        assert_eq!(err.transfer_characteristics(), 13);
    }

    /// MC=2 with hint=2 (itself Unspecified) → Err.
    #[test]
    fn resolve_matrix_unspecified_hint_also_unspecified_is_err() {
        let c = Cicp::new(1, 13, 2, true);
        assert!(c.resolve_matrix(Some(2)).is_err());
    }

    /// MC=3 (Reserved) without hint → Err.
    #[test]
    fn resolve_matrix_reserved_mc3_no_hint() {
        let c = Cicp::new(1, 13, 3, true);
        assert!(c.resolve_matrix(None).is_err());
    }

    /// MC=3 (Reserved) with a valid hint → resolved.
    #[test]
    fn resolve_matrix_reserved_mc3_with_hint() {
        let c = Cicp::new(1, 13, 3, true);
        let r = c.resolve_matrix(Some(1)).unwrap();
        assert_eq!(r.matrix_coefficients, 1);
    }

    /// MC=18 (first reserved value in H.273 V4) through MC=255 require a
    /// hint. (An earlier revision of this test used 15 as the first reserved
    /// value — wrong against H.273 ≥ 09/2023, where 15/16/17 are defined;
    /// corrected together with the classification fix.)
    #[test]
    fn resolve_matrix_large_reserved_mc() {
        for mc in [18u8, 100, 255] {
            let c = Cicp::new(1, 13, mc, true);
            assert!(
                c.resolve_matrix(None).is_err(),
                "MC={mc} (reserved) without hint should be Err"
            );
            let r = c.resolve_matrix(Some(6)).unwrap();
            assert_eq!(r.matrix_coefficients, 6, "MC={mc} hint=6 should resolve");
        }
    }

    /// MC=2 with hint=12 (Chroma NCL) and known CP → resolved.
    #[test]
    fn resolve_matrix_unspecified_hint_chroma_ncl_known_cp() {
        let c = Cicp::new(9, 16, 2, true);
        let r = c.resolve_matrix(Some(12)).unwrap();
        assert_eq!(r.matrix_coefficients, 12);
    }

    /// MC=2 with hint=12 (Chroma NCL) but CP=2 (Unspecified) → Err.
    #[test]
    fn resolve_matrix_unspecified_hint_chroma_ncl_unknown_cp_is_err() {
        let c = Cicp::new(2, 16, 2, true);
        assert!(c.resolve_matrix(Some(12)).is_err());
    }

    /// Hint is ignored when MC is already self-contained.
    #[test]
    fn resolve_matrix_hint_ignored_for_self_contained() {
        let c = Cicp::new(1, 13, 6, true); // MC=6, self-contained
        let r = c.resolve_matrix(Some(1)).unwrap(); // hint should be ignored
        assert_eq!(r.matrix_coefficients, 6); // unchanged
    }

    /// Error Display contains the matrix_coefficients, color_primaries, and
    /// transfer_characteristics.
    #[test]
    fn unspecified_matrix_error_display() {
        let c = Cicp::new(1, 13, 2, true);
        let err = c.resolve_matrix(None).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("color_primaries=1"), "got: {s}");
        assert!(s.contains("transfer_characteristics=13"), "got: {s}");
        assert!(s.contains("matrix_coefficients=2"), "got: {s}");
    }

    /// Error Debug is available (derived).
    #[test]
    fn unspecified_matrix_error_debug() {
        let c = Cicp::new(9, 16, 2, true);
        let err = c.resolve_matrix(None).unwrap_err();
        let _ = format!("{err:?}");
    }

    /// `resolve_matrix` is callable in a const context.
    #[test]
    fn resolve_matrix_is_const_callable() {
        const _R: Result<Cicp, UnspecifiedMatrixError> = Cicp::SRGB.resolve_matrix(None);
        assert!(_R.is_ok());
    }

    /// The error exposes the unresolvable signaled MC — the original value,
    /// never a rejected hint.
    #[test]
    fn unspecified_matrix_error_exposes_mc() {
        // MC=2, invalid hint: error carries 2, not the hint.
        let err = Cicp::new(1, 13, 2, true)
            .resolve_matrix(Some(3))
            .unwrap_err();
        assert_eq!(err.matrix_coefficients(), 2);
        // MC=12 with underivable CP: error carries 12.
        let err = Cicp::new(2, 1, 12, false)
            .resolve_matrix(Some(6))
            .unwrap_err();
        assert_eq!(err.matrix_coefficients(), 12);
        // Reserved MC=200, no hint: error carries 200.
        let err = Cicp::new(1, 13, 200, true)
            .resolve_matrix(None)
            .unwrap_err();
        assert_eq!(err.matrix_coefficients(), 200);
    }

    /// Exhaustive classification over the full MC byte range, pinned to
    /// Rec. ITU-T H.273 (V4) (07/2024), Table 4:
    /// - self-contained (0, 1, 4–11, 14–17): Ok(self) with or without hint;
    /// - chromaticity-derived (12, 13): Ok(self) iff CP has Table 2
    ///   chromaticity, hint never consulted;
    /// - unspecified/reserved (2, 3, 18–255): Err without hint, resolved to
    ///   the hint with a valid hint.
    #[test]
    fn resolve_matrix_exhaustive_full_byte_range() {
        const SELF_CONTAINED: [u8; 14] = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17];
        const CHROMA_DERIVED: [u8; 2] = [12, 13];
        for mc in 0..=255u8 {
            let known_cp = Cicp::new(1, 13, mc, true); // CP=1: has chromaticity
            let unknown_cp = Cicp::new(2, 13, mc, true); // CP=2: Unspecified
            if SELF_CONTAINED.contains(&mc) {
                // Ok(self) regardless of hint or CP.
                assert_eq!(known_cp.resolve_matrix(None).unwrap(), known_cp);
                assert_eq!(known_cp.resolve_matrix(Some(1)).unwrap(), known_cp);
                assert_eq!(unknown_cp.resolve_matrix(None).unwrap(), unknown_cp);
            } else if CHROMA_DERIVED.contains(&mc) {
                // CP-gated; the hint is never consulted.
                assert_eq!(known_cp.resolve_matrix(None).unwrap(), known_cp);
                assert!(unknown_cp.resolve_matrix(None).is_err(), "MC={mc} CP=2");
                assert!(
                    unknown_cp.resolve_matrix(Some(1)).is_err(),
                    "MC={mc} CP=2: a hint must not override a CP-derived matrix"
                );
            } else {
                // 2, 3, 18–255: hint-substituted or Err.
                assert!(known_cp.resolve_matrix(None).is_err(), "MC={mc} no hint");
                let r = known_cp.resolve_matrix(Some(6)).unwrap();
                assert_eq!(r.matrix_coefficients, 6, "MC={mc} hint=6");
            }
        }
    }

    /// Hint-validity matrix: for an unspecified MC, sweep every possible
    /// hint byte. A hint is consumed iff it is itself self-contained, or
    /// chromaticity-derived with a CP that has Table 2 chromaticity.
    #[test]
    fn resolve_matrix_hint_validity_matrix() {
        const SELF_CONTAINED: [u8; 14] = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17];
        const CHROMA_DERIVED: [u8; 2] = [12, 13];
        for hint in 0..=255u8 {
            let known_cp = Cicp::new(1, 13, 2, true);
            let unknown_cp = Cicp::new(2, 13, 2, true);
            if SELF_CONTAINED.contains(&hint) {
                // Valid regardless of CP.
                assert_eq!(
                    known_cp
                        .resolve_matrix(Some(hint))
                        .unwrap()
                        .matrix_coefficients,
                    hint,
                    "hint={hint} should substitute"
                );
                assert_eq!(
                    unknown_cp
                        .resolve_matrix(Some(hint))
                        .unwrap()
                        .matrix_coefficients,
                    hint,
                    "hint={hint} should substitute even with CP=2"
                );
            } else if CHROMA_DERIVED.contains(&hint) {
                // Valid only when CP can derive the matrix.
                assert_eq!(
                    known_cp
                        .resolve_matrix(Some(hint))
                        .unwrap()
                        .matrix_coefficients,
                    hint,
                    "hint={hint} with derivable CP should substitute"
                );
                assert!(
                    unknown_cp.resolve_matrix(Some(hint)).is_err(),
                    "hint={hint} with CP=2 cannot derive"
                );
            } else {
                // 2, 3, 18–255: never a valid hint.
                assert!(
                    known_cp.resolve_matrix(Some(hint)).is_err(),
                    "hint={hint} is unspecified/reserved and must be rejected"
                );
            }
        }
    }

    /// The CP gate for MC=12/13 accepts exactly the Table 2
    /// chromaticity-bearing codes: 1, 4–12, 22.
    #[test]
    fn resolve_matrix_cp_gate_exhaustive() {
        const CHROMA_CP: [u8; 11] = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22];
        for cp in 0..=255u8 {
            let c = Cicp::new(cp, 1, 12, false);
            if CHROMA_CP.contains(&cp) {
                assert!(c.resolve_matrix(None).is_ok(), "CP={cp} should derive");
            } else {
                assert!(c.resolve_matrix(None).is_err(), "CP={cp} cannot derive");
            }
        }
    }
}
