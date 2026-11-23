//! CICP (Coding-Independent Code Points, ITU-T H.273 / ISO 23091-2).

use core::fmt;
use whereat::At;

use crate::color::{AlphaMode, ColorSpace, Primaries, SignalRange, Transfer};

/// Failure projecting a raw [`Cicp`] tuple into an RGB [`ColorSpace`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CicpError {
    /// Non-zero matrix coefficients (YCbCr, ICtCp, etc.) require component reconstruction before RGB projection.
    NonIdentityMatrix(u8),
    /// `color_primaries` code has no built-in [`Primaries`] mapping.
    UnmappedPrimaries(u8),
    /// `transfer_characteristics` code has no built-in [`Transfer`] mapping.
    UnmappedTransfer(u8),
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
        }
    }
}

impl core::error::Error for CicpError {}

/// 4-byte ITU-T H.273 color description (`CP`, `TC`, `MC`, `full_range`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Cicp {
    /// `ColourPrimaries` (1 = BT.709/sRGB, 9 = BT.2020, 12 = Display P3).
    pub color_primaries: u8,
    /// `TransferCharacteristics` (1 = BT.709, 8 = Linear, 13 = sRGB, 16 = PQ, 18 = HLG).
    pub transfer_characteristics: u8,
    /// `MatrixCoefficients` (0 = Identity/RGB, 1 = BT.709, 6 = BT.601, 9 = BT.2020).
    pub matrix_coefficients: u8,
    /// `true` = full range, `false` = narrow/studio swing.
    pub full_range: bool,
}

impl Cicp {
    /// sRGB RGB `(1, 13, 0, true)`.
    pub const SRGB: Self = Self::new(1, 13, 0, true);
    /// Display P3 RGB `(12, 13, 0, true)`.
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

    /// Decode from PNG `cICP` 4-byte wire payload `[CP, TC, MC, full_range_u8]`.
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
