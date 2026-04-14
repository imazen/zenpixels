//! CICP (Coding-Independent Code Points) color description.
//!
//! ITU-T H.273 / ISO 23091-2 defines code points for color primaries,
//! transfer characteristics, and matrix coefficients. This struct
//! carries the four fields needed by [`ColorContext`](crate::color::ColorContext).

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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
            _ => "Unknown",
        }
    }
}

impl core::fmt::Display for Cicp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
            TransferFunction::Unknown
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
        for cicp in [Cicp::SRGB, Cicp::BT2100_PQ, Cicp::DISPLAY_P3] {
            let desc = cicp.to_descriptor(PixelFormat::Rgb8);
            let back = Cicp::from_descriptor(&desc).unwrap();
            assert_eq!(back.color_primaries, cicp.color_primaries);
            assert_eq!(back.transfer_characteristics, cicp.transfer_characteristics);
            assert_eq!(back.full_range, cicp.full_range);
        }
    }
}
