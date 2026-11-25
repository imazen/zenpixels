//! CICP (Coding-Independent Code Points, ITU-T H.273 / ISO 23091-2).

use core::fmt;
use whereat::At;

use crate::color::{AlphaMode, ColorSpace, Primaries, SignalRange, Transfer};

/// Failure parsing a raw wire CICP payload or projecting a [`Cicp`] tuple into an RGB [`ColorSpace`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CicpError {
    /// Non-zero matrix coefficients (YCbCr, ICtCp, etc.) require component reconstruction before RGB projection.
    NonIdentityMatrix(u8),
    /// `color_primaries` code has no built-in [`Primaries`] mapping.
    UnmappedPrimaries(u8),
    /// `transfer_characteristics` code has no built-in [`Transfer`] mapping.
    UnmappedTransfer(u8),
    /// Wire `video_full_range_flag` byte was neither `0` nor `1` (ITU-T H.273 §8.3 / PNG 3 §11.3.2.1).
    InvalidFullRangeFlag(u8),
}

impl fmt::Display for CicpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonIdentityMatrix(mc) => {
                write!(
                    f,
                    "CICP matrix {mc} requires YCbCr-to-RGB reconstruction first"
                )
            }
            Self::UnmappedPrimaries(cp) => write!(f, "unmapped CICP primaries code {cp}"),
            Self::UnmappedTransfer(tc) => write!(f, "unmapped CICP transfer code {tc}"),
            Self::InvalidFullRangeFlag(flag) => {
                write!(
                    f,
                    "invalid CICP video_full_range_flag byte {flag} (must be 0 or 1)"
                )
            }
        }
    }
}

impl core::error::Error for CicpError {}

/// Failure projecting a [`ColorSpace`] losslessly to an ITU-T H.273 [`Cicp`] tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CicpProjectionError {
    /// Color space is not tristimulus RGB (e.g. Gray, XYZ, Oklab, CMYK, ICC, Untagged).
    NotRgb,
    /// Primaries have no standardized H.273 `ColourPrimaries` code (e.g. Adobe RGB, Custom, Unknown).
    UnmappedPrimaries,
    /// Transfer curve has no standardized H.273 `TransferCharacteristics` code (e.g. Adobe RGB, custom gamma).
    UnmappedTransfer,
    /// HLG display-rendering parameters (`peak`, `black_nits`, `system_gamma`) cannot be represented
    /// in CICP TC=18, which carries scene-referred syntax only.
    HlgDisplayRenderingLost,
    /// Diffuse white luminance anchor (`DiffuseWhite`) cannot be represented in CICP.
    DiffuseWhiteLost,
}

impl fmt::Display for CicpProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotRgb => "only RGB color spaces project directly to CICP",
            Self::UnmappedPrimaries => "primaries have no standardized H.273 CICP code",
            Self::UnmappedTransfer => "transfer curve has no standardized H.273 CICP code",
            Self::HlgDisplayRenderingLost => {
                "CICP TC=18 cannot represent HLG display rendering parameters"
            }
            Self::DiffuseWhiteLost => "CICP cannot represent an explicit DiffuseWhite anchor",
        })
    }
}

impl core::error::Error for CicpProjectionError {}

/// 4-byte ITU-T H.273 color description (`CP`, `TC`, `MC`, `full_range`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Cicp {
    /// `ColourPrimaries` (1 = BT.709/sRGB, 4 = BT.470M, 5 = BT.470BG, 6 = BT.601,
    /// 7 = SMPTE 240M, 8 = Generic Film, 9 = BT.2020, 10 = XYZ, 11 = DCI-P3,
    /// 12 = Display P3, 22 = EBU Tech 3213).
    pub color_primaries: u8,
    /// `TransferCharacteristics` (1 = BT.709, 4 = Gamma 2.2, 5 = Gamma 2.8, 6 = BT.601,
    /// 7 = SMPTE 240M, 8 = Linear, 9 = Log100, 10 = Log100Sqrt10, 11 = IEC 61966-2-4,
    /// 12 = BT.1361, 13 = sRGB, 14 = BT.2020 10-bit, 15 = BT.2020 12-bit, 16 = PQ,
    /// 17 = SMPTE ST 428-1, 18 = HLG).
    pub transfer_characteristics: u8,
    /// `MatrixCoefficients` (0 = Identity/RGB, 1 = BT.709, 4 = FCC, 5 = BT.470BG,
    /// 6 = BT.601, 7 = SMPTE 240M, 8 = YCgCo, 9 = BT.2020 NCL, 12 = Chroma-derived NCL).
    pub matrix_coefficients: u8,
    /// `true` = full range, `false` = narrow/studio swing.
    pub full_range: bool,
}

impl Cicp {
    /// sRGB RGB `(1, 13, 0, true)`.
    pub const SRGB: Self = Self::new(1, 13, 0, true);
    /// DCI-P3 (SMPTE RP 431-2 theater white) RGB `(11, 13, 0, true)`.
    pub const DCI_P3: Self = Self::new(11, 13, 0, true);
    /// Display P3 (D65 white) RGB `(12, 13, 0, true)`.
    pub const DISPLAY_P3: Self = Self::new(12, 13, 0, true);
    /// Linear sRGB RGB `(1, 8, 0, true)`.
    pub const LINEAR_SRGB: Self = Self::new(1, 8, 0, true);
    /// BT.709 RGB `(1, 1, 0, true)`.
    pub const BT709_RGB: Self = Self::new(1, 1, 0, true);
    /// BT.2100 PQ RGB `(9, 16, 0, true)`.
    pub const BT2100_PQ_RGB: Self = Self::new(9, 16, 0, true);
    /// BT.2100 PQ YCbCr `(9, 16, 9, true)`.
    pub const BT2100_PQ: Self = Self::new(9, 16, 9, true);
    /// BT.2100 HLG RGB `(9, 18, 0, true)`.
    pub const BT2100_HLG_RGB: Self = Self::new(9, 18, 0, true);
    /// BT.2100 HLG YCbCr `(9, 18, 9, true)`.
    pub const BT2100_HLG: Self = Self::new(9, 18, 9, true);

    /// Construct a CICP tuple from raw H.273 codes.
    #[inline]
    #[must_use]
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

    /// Strictly decode a 4-byte wire payload `[CP, TC, MC, full_range_u8]`, rejecting
    /// invalid `video_full_range_flag` values (`> 1`) per ITU-T H.273 §8.3 / PNG 3.
    #[inline]
    pub fn try_from_bytes(bytes: [u8; 4]) -> Result<Self, At<CicpError>> {
        let full_range = match bytes[3] {
            0 => false,
            1 => true,
            other => return Err(whereat::at!(CicpError::InvalidFullRangeFlag(other))),
        };
        Ok(Self::new(bytes[0], bytes[1], bytes[2], full_range))
    }

    /// Permissively decode from a 4-byte wire payload `[CP, TC, MC, full_range_u8]`
    /// (treating any non-zero 4th byte as full range). For strict wire validation,
    /// use [`try_from_bytes`](Self::try_from_bytes).
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Self::new(bytes[0], bytes[1], bytes[2], bytes[3] != 0)
    }

    /// Encode to PNG `cICP` 4-byte wire payload.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 4] {
        [
            self.color_primaries,
            self.transfer_characteristics,
            self.matrix_coefficients,
            self.full_range as u8,
        ]
    }

    /// Human-readable ITU-T H.273 name for `self.color_primaries`, if known.
    #[inline]
    #[must_use]
    pub const fn primaries_name(self) -> Option<&'static str> {
        color_primaries_name(self.color_primaries)
    }

    /// Human-readable ITU-T H.273 name for `self.transfer_characteristics`, if known.
    #[inline]
    #[must_use]
    pub const fn transfer_name(self) -> Option<&'static str> {
        transfer_characteristics_name(self.transfer_characteristics)
    }

    /// Human-readable ITU-T H.273 name for `self.matrix_coefficients`, if known.
    #[inline]
    #[must_use]
    pub const fn matrix_name(self) -> Option<&'static str> {
        matrix_coefficients_name(self.matrix_coefficients)
    }

    /// Project an Identity-matrix (`MC == 0`) CICP tuple into an algebraic [`ColorSpace::Rgb`].
    pub fn to_color_space(self, alpha: AlphaMode) -> Result<ColorSpace, At<CicpError>> {
        if self.matrix_coefficients != 0 {
            return Err(whereat::at!(CicpError::NonIdentityMatrix(
                self.matrix_coefficients
            )));
        }
        let primaries = Primaries::from_cicp(self.color_primaries)
            .ok_or_else(|| whereat::at!(CicpError::UnmappedPrimaries(self.color_primaries)))?;
        let transfer = Transfer::from_cicp(self.transfer_characteristics).ok_or_else(|| {
            whereat::at!(CicpError::UnmappedTransfer(self.transfer_characteristics))
        })?;
        Ok(ColorSpace::Rgb {
            primaries,
            transfer,
            range: if self.full_range {
                SignalRange::Full
            } else {
                SignalRange::Narrow
            },
            alpha,
            white: None,
        })
    }
}

/// Human-readable ITU-T H.273 Table 2 `ColourPrimaries` label.
#[inline]
#[must_use]
pub const fn color_primaries_name(code: u8) -> Option<&'static str> {
    match code {
        1 => Some("BT.709"),
        2 => Some("Unspecified"),
        4 => Some("BT.470M"),
        5 => Some("BT.470BG"),
        6 => Some("BT.601"),
        7 => Some("SMPTE 240M"),
        8 => Some("Generic film"),
        9 => Some("BT.2020"),
        10 => Some("XYZ"),
        11 => Some("SMPTE 431-2 (DCI-P3)"),
        12 => Some("SMPTE 432-1 (Display P3)"),
        22 => Some("EBU Tech 3213"),
        _ => None,
    }
}

/// Human-readable ITU-T H.273 Table 3 `TransferCharacteristics` label.
#[inline]
#[must_use]
pub const fn transfer_characteristics_name(code: u8) -> Option<&'static str> {
    match code {
        1 => Some("BT.709"),
        2 => Some("Unspecified"),
        4 => Some("BT.470M (Gamma 2.2)"),
        5 => Some("BT.470BG (Gamma 2.8)"),
        6 => Some("BT.601"),
        7 => Some("SMPTE 240M"),
        8 => Some("Linear"),
        9 => Some("Log 100"),
        10 => Some("Log 100 sqrt10"),
        11 => Some("IEC 61966-2-4"),
        12 => Some("BT.1361"),
        13 => Some("sRGB"),
        14 => Some("BT.2020 10-bit"),
        15 => Some("BT.2020 12-bit"),
        16 => Some("PQ (SMPTE ST 2084)"),
        17 => Some("SMPTE ST 428-1"),
        18 => Some("HLG (ARIB STD-B67)"),
        _ => None,
    }
}

/// Human-readable ITU-T H.273 Table 4 `MatrixCoefficients` label.
#[inline]
#[must_use]
pub const fn matrix_coefficients_name(code: u8) -> Option<&'static str> {
    match code {
        0 => Some("Identity (RGB)"),
        1 => Some("BT.709"),
        2 => Some("Unspecified"),
        4 => Some("FCC"),
        5 => Some("BT.470BG"),
        6 => Some("BT.601"),
        7 => Some("SMPTE 240M"),
        8 => Some("YCgCo"),
        9 => Some("BT.2020 NCL"),
        10 => Some("BT.2020 CL"),
        11 => Some("SMPTE 2085"),
        12 => Some("Chroma-derived NCL"),
        13 => Some("Chroma-derived CL"),
        14 => Some("ICtCp"),
        _ => None,
    }
}

impl fmt::Display for Cicp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let range = if self.full_range { "full" } else { "narrow" };
        match (
            self.primaries_name(),
            self.transfer_name(),
            self.matrix_name(),
        ) {
            (Some(p), Some(t), Some(m)) => write!(
                f,
                "CICP({}/{}/{}, {}) [{} / {} / {}]",
                self.color_primaries,
                self.transfer_characteristics,
                self.matrix_coefficients,
                range,
                p,
                t,
                m
            ),
            _ => write!(
                f,
                "CICP({}/{}/{}, {})",
                self.color_primaries,
                self.transfer_characteristics,
                self.matrix_coefficients,
                range
            ),
        }
    }
}
