//! CICP (Coding-Independent Code Points, ITU-T H.273 / ISO 23091-2) and
//! ICC.1 `cicpType` tag codec including the 2026-10-19 custom-primaries extension.

use alloc::vec::Vec;
use core::fmt;
use whereat::At;

use crate::color::{
    AlphaMode, ColorSpace, CustomPrimaries, HlgMode, Primaries, SignalRange, Transfer,
};

/// Failure parsing a raw wire CICP payload or projecting a [`Cicp`] / [`CicpTag`] into an RGB [`ColorSpace`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CicpError {
    /// Non-zero matrix coefficients (YCbCr, ICtCp, etc.) require component reconstruction before RGB projection.
    NonIdentityMatrix(u8),
    /// `color_primaries` code has no built-in [`Primaries`] mapping (and no custom-primaries extension was supplied).
    UnmappedPrimaries(u8),
    /// `transfer_characteristics` code has no built-in [`Transfer`] mapping.
    UnmappedTransfer(u8),
    /// Wire `video_full_range_flag` byte was neither `0` nor `1` (ITU-T H.273 §8.3 / PNG 3 §11.3.2.1).
    InvalidFullRangeFlag(u8),
    /// Raw ICC `cicpType` tag bytes had an invalid length, type signature, or non-zero reserved field.
    MalformedTag,
    /// ICC `cicpType` 32-byte custom-primaries extension contained invalid, non-physical, or degenerate chromaticities.
    InvalidCustomPrimaries,
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
            Self::MalformedTag => {
                write!(
                    f,
                    "malformed ICC cicpType tag (invalid signature, reserved bytes, or declared length)"
                )
            }
            Self::InvalidCustomPrimaries => {
                write!(
                    f,
                    "ICC cicpType custom-primaries extension has invalid or degenerate chromaticities"
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

/// 1-byte lossless ITU-T H.273 Table 4 `MatrixCoefficients` code.
///
/// Preserves arbitrary `u8` wire codes losslessly (including unknown, reserved, or future
/// codes such as `10` = BT.2020 CL or `14` = ICtCp), while exposing named constants for
/// every standardized ITU-T H.273 Table 4 entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MatrixCoefficients(u8);

impl MatrixCoefficients {
    /// Identity / RGB / GBR (`MC = 0`).
    pub const IDENTITY: Self = Self(0);
    /// ITU-R BT.709 (`MC = 1`).
    pub const BT709: Self = Self(1);
    /// Unspecified (`MC = 2`).
    pub const UNSPECIFIED: Self = Self(2);
    /// FCC Title 47 CFR 73.682 (`MC = 4`).
    pub const FCC: Self = Self(4);
    /// ITU-R BT.470-6 System B, G (`MC = 5`).
    pub const BT470BG: Self = Self(5);
    /// SMPTE 170M / ITU-R BT.601 (`MC = 6`).
    pub const BT601: Self = Self(6);
    /// SMPTE 240M (`MC = 7`).
    pub const SMPTE240M: Self = Self(7);
    /// YCgCo (`MC = 8`).
    pub const YCGCO: Self = Self(8);
    /// ITU-R BT.2020 non-constant luminance (`MC = 9`).
    pub const BT2020_NCL: Self = Self(9);
    /// ITU-R BT.2020 constant luminance (`MC = 10`).
    pub const BT2020_CL: Self = Self(10);
    /// SMPTE ST 2085 (`MC = 11`).
    pub const SMPTE2085: Self = Self(11);
    /// Chromaticity-derived non-constant luminance (`MC = 12`).
    pub const CHROMA_DERIVED_NCL: Self = Self(12);
    /// Chromaticity-derived constant luminance (`MC = 13`).
    pub const CHROMA_DERIVED_CL: Self = Self(13);
    /// ITU-R BT.2100 ICtCp (`MC = 14`).
    pub const ICTCP: Self = Self(14);

    /// Wrap any raw ITU-T H.273 Table 4 `matrix_coefficients` byte losslessly.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        Self(code)
    }

    /// Raw ITU-T H.273 Table 4 `matrix_coefficients` byte.
    #[inline]
    #[must_use]
    pub const fn code(self) -> u8 {
        self.0
    }

    /// Human-readable ITU-T H.273 Table 4 name for this code, if standardized.
    #[inline]
    #[must_use]
    pub const fn name(self) -> Option<&'static str> {
        matrix_coefficients_name(self.0)
    }
}

impl fmt::Display for MatrixCoefficients {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(f, "{name} (MC={})", self.0),
            None => write!(f, "MatrixCoefficients({})", self.0),
        }
    }
}

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

    /// Typed [`MatrixCoefficients`] wrapper for `self.matrix_coefficients`.
    #[inline]
    #[must_use]
    pub const fn matrix(self) -> MatrixCoefficients {
        MatrixCoefficients::from_code(self.matrix_coefficients)
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

/// ICC.1:2022 §10.3 `cicpType` tag payload, including the ICC custom-primaries amendment
/// (approved 2026-10-19).
///
/// - **Base 12-byte form:** `'cicp'` + 4 zero reserved bytes + 4-byte [`Cicp`] tuple.
/// - **Extended 44-byte form:** valid only when `cicp.color_primaries == 2` (Unspecified)
///   in an ICC v4+ Input (`scnr`), Display (`mntr`), or ColorSpace (`spac`) profile whose
///   header data colour space is `RGB ` or `YCbr` (ICC `cicpType` Amendment §4.1 & §4.3
///   Table 32). Bytes `12..44` encode eight signed `s15Fixed16Number` CIE 1931 `(x, y)`
///   coordinates `[xR, yR, xG, yG, xB, yB, xw, yw]`, supporting negative chromaticities
///   such as ACES AP0 (`y_B = -0.0770`).
///
/// The invariant `custom_primaries.is_some() => cicp.color_primaries == 2` is enforced
/// by construction.
///
/// # `s15Fixed16Number` wire quantization
///
/// Encoding the 44-byte extended form quantizes each CIE 1931 `(x, y)` coordinate to
/// signed `s15Fixed16Number` (`1 / 65536 ≈ 1.53e-5`), so a decimal `f32` preset such as
/// `0.7347` decodes back as `48149 / 65536 ≈ 0.73469543`. Wire bytes and
/// [`CustomPrimaries::to_s15fixed16`] round-trip bit-exact; compare those (or a
/// `1 / 65536` tolerance) rather than raw decimal `f32` presets after serialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CicpTag {
    cicp: Cicp,
    custom_primaries: Option<CustomPrimaries>,
}

impl CicpTag {
    /// Byte length of the base ICC.1:2022 §10.3 `cicpType` tag (`12` bytes).
    pub const BASE_TAG_LEN: usize = 12;

    /// Byte length of the extended `cicpType` tag carrying eight `s15Fixed16Number`
    /// custom chromaticities (`12 + 32 = 44` bytes).
    pub const EXTENDED_TAG_LEN: usize = 44;

    /// Construct a 12-byte base `cicpType` tag with no custom-primaries extension.
    #[inline]
    #[must_use]
    pub const fn from_cicp(cicp: Cicp) -> Self {
        Self {
            cicp,
            custom_primaries: None,
        }
    }

    /// Construct a 44-byte extended `cicpType` tag (`ColourPrimaries = 2` + custom
    /// CIE 1931 primaries and white point).
    #[inline]
    #[must_use]
    pub const fn with_custom_primaries(
        primaries: CustomPrimaries,
        transfer_characteristics: u8,
        matrix_coefficients: u8,
        full_range: bool,
    ) -> Self {
        Self {
            cicp: Cicp::new(2, transfer_characteristics, matrix_coefficients, full_range),
            custom_primaries: Some(primaries),
        }
    }

    /// Construct a `cicpType` tag from any known [`Primaries`] (emitting the 12-byte form
    /// when standardized in H.273 Table 2, or the 44-byte custom-primaries form for
    /// [`Primaries::AdobeRgb`] and [`Primaries::Custom`]).
    ///
    /// Returns `None` for [`Primaries::Unknown`].
    #[inline]
    #[must_use]
    pub const fn from_primaries_and_cicp(
        primaries: Primaries,
        transfer_characteristics: u8,
        matrix_coefficients: u8,
        full_range: bool,
    ) -> Option<Self> {
        if let Some(cp_code) = primaries.to_cicp() {
            return Some(Self::from_cicp(Cicp::new(
                cp_code,
                transfer_characteristics,
                matrix_coefficients,
                full_range,
            )));
        }
        let Some(custom) = primaries.to_custom() else {
            return None;
        };
        Some(Self::with_custom_primaries(
            custom,
            transfer_characteristics,
            matrix_coefficients,
            full_range,
        ))
    }

    /// Losslessly project an RGB [`ColorSpace`] to an ICC `cicpType` tag, using the
    /// 12-byte enumerated form when standardized or the 44-byte custom-primaries form
    /// for [`Primaries::AdobeRgb`] and [`Primaries::Custom`].
    ///
    /// Refuses [`HlgMode::Display`], explicit [`DiffuseWhite`](crate::DiffuseWhite),
    /// non-RGB color spaces, [`Primaries::Unknown`], and non-H.273 transfers.
    pub fn from_color_space(color: &ColorSpace) -> Result<Self, CicpProjectionError> {
        match *color {
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                white,
                ..
            } => {
                if matches!(transfer, Transfer::Hlg(HlgMode::Display(_))) {
                    return Err(CicpProjectionError::HlgDisplayRenderingLost);
                }
                if white.is_some() {
                    return Err(CicpProjectionError::DiffuseWhiteLost);
                }
                let tc = transfer
                    .to_cicp()
                    .ok_or(CicpProjectionError::UnmappedTransfer)?;
                Self::from_primaries_and_cicp(primaries, tc, 0, range == SignalRange::Full)
                    .ok_or(CicpProjectionError::UnmappedPrimaries)
            }
            _ => Err(CicpProjectionError::NotRgb),
        }
    }

    /// Underlying 4-byte H.273 [`Cicp`] tuple (`color_primaries == 2` when extended).
    #[inline]
    #[must_use]
    pub const fn cicp(self) -> Cicp {
        self.cicp
    }

    /// Embedded custom primaries from bytes `12..44`, if present.
    #[inline]
    #[must_use]
    pub const fn custom_primaries(self) -> Option<CustomPrimaries> {
        self.custom_primaries
    }

    /// Resolved [`Primaries`]: [`Primaries::Custom`] when the 32-byte extension is present,
    /// otherwise [`Primaries::from_cicp`] on the H.273 `ColourPrimaries` code.
    #[inline]
    #[must_use]
    pub const fn primaries(self) -> Option<Primaries> {
        match self.custom_primaries {
            Some(cp) => Some(Primaries::Custom(cp)),
            None => Primaries::from_cicp(self.cicp.color_primaries),
        }
    }

    /// Encoded ICC `cicpType` tag length in bytes: `12` (base) or `44` (extended).
    #[inline]
    #[must_use]
    pub const fn encoded_len(self) -> usize {
        if self.custom_primaries.is_some() {
            Self::EXTENDED_TAG_LEN
        } else {
            Self::BASE_TAG_LEN
        }
    }

    /// Project an Identity-matrix (`MC == 0`) `cicpType` tag into [`ColorSpace::Rgb`],
    /// resolving the 32-byte custom-primaries extension into [`Primaries::Custom`] when present.
    pub fn to_color_space(self, alpha: AlphaMode) -> Result<ColorSpace, At<CicpError>> {
        if self.cicp.matrix_coefficients != 0 {
            return Err(whereat::at!(CicpError::NonIdentityMatrix(
                self.cicp.matrix_coefficients
            )));
        }
        let primaries = self
            .primaries()
            .ok_or_else(|| whereat::at!(CicpError::UnmappedPrimaries(self.cicp.color_primaries)))?;
        let transfer =
            Transfer::from_cicp(self.cicp.transfer_characteristics).ok_or_else(|| {
                whereat::at!(CicpError::UnmappedTransfer(
                    self.cicp.transfer_characteristics
                ))
            })?;
        Ok(ColorSpace::Rgb {
            primaries,
            transfer,
            range: if self.cicp.full_range {
                SignalRange::Full
            } else {
                SignalRange::Narrow
            },
            alpha,
            white: None,
        })
    }

    /// Strictly decode raw ICC `cicpType` tag bytes (12-byte base or 44-byte extended)
    /// per ICC.1:2022 §10.3 and the 2026-10-19 custom-primaries amendment.
    ///
    /// - Rejects lengths `< 12`, wrong `'cicp'` signature, non-zero reserved bytes `4..8`,
    ///   or `VideoFullRangeFlag > 1`.
    /// - When `ColourPrimaries != 2`, rejects declared lengths `!= 12` (§4.3 NOTE 1).
    /// - When `ColourPrimaries == 2` and `tag_bytes.len() == 12`, returns `custom_primaries = None`
    ///   (§4.3 NOTE 2).
    /// - When `ColourPrimaries == 2` and `tag_bytes.len() > 12`, requires at least 44 bytes and
    ///   validates bytes `12..44` as eight `s15Fixed16Number` chromaticities.
    pub fn from_tag_bytes(tag_bytes: &[u8]) -> Result<Self, At<CicpError>> {
        if tag_bytes.len() < Self::BASE_TAG_LEN
            || &tag_bytes[0..4] != b"cicp"
            || tag_bytes[4..8] != [0; 4]
        {
            return Err(whereat::at!(CicpError::MalformedTag));
        }
        let full_range = match tag_bytes[11] {
            0 => false,
            1 => true,
            other => return Err(whereat::at!(CicpError::InvalidFullRangeFlag(other))),
        };
        let cicp = Cicp::new(tag_bytes[8], tag_bytes[9], tag_bytes[10], full_range);
        if cicp.color_primaries == 2 {
            if tag_bytes.len() == Self::BASE_TAG_LEN {
                return Ok(Self {
                    cicp,
                    custom_primaries: None,
                });
            }
            if tag_bytes.len() < Self::EXTENDED_TAG_LEN {
                return Err(whereat::at!(CicpError::MalformedTag));
            }
            let ext: [u8; 32] = tag_bytes[12..44]
                .try_into()
                .map_err(|_| whereat::at!(CicpError::MalformedTag))?;
            let custom = CustomPrimaries::from_cicp_extension_bytes(ext)
                .ok_or_else(|| whereat::at!(CicpError::InvalidCustomPrimaries))?;
            return Ok(Self {
                cicp,
                custom_primaries: Some(custom),
            });
        }
        if tag_bytes.len() != Self::BASE_TAG_LEN {
            return Err(whereat::at!(CicpError::MalformedTag));
        }
        Ok(Self {
            cicp,
            custom_primaries: None,
        })
    }

    /// Encode the 12-byte base `cicpType` header + 4-byte CICP payload.
    #[inline]
    #[must_use]
    pub const fn to_base_tag_bytes(self) -> [u8; 12] {
        [
            b'c',
            b'i',
            b'c',
            b'p',
            0,
            0,
            0,
            0,
            self.cicp.color_primaries,
            self.cicp.transfer_characteristics,
            self.cicp.matrix_coefficients,
            self.cicp.full_range as u8,
        ]
    }

    /// Encode the full 44-byte extended `cicpType` payload when `custom_primaries` is present,
    /// or `None` for 12-byte base tags.
    #[must_use]
    pub const fn to_extended_tag_bytes(self) -> Option<[u8; 44]> {
        let Some(cp) = self.custom_primaries else {
            return None;
        };
        let base = self.to_base_tag_bytes();
        let ext = cp.to_cicp_extension_bytes();
        let mut out = [0u8; 44];
        let mut i = 0;
        while i < 12 {
            out[i] = base[i];
            i += 1;
        }
        let mut j = 0;
        while j < 32 {
            out[12 + j] = ext[j];
            j += 1;
        }
        Some(out)
    }

    /// Write the encoded `cicpType` tag (`12` or `44` bytes) into `dst` without allocating,
    /// returning the number of bytes written (`None` if `dst.len() < self.encoded_len()`).
    #[must_use]
    pub fn write_tag_bytes(self, dst: &mut [u8]) -> Option<usize> {
        let len = self.encoded_len();
        if dst.len() < len {
            return None;
        }
        match self.to_extended_tag_bytes() {
            Some(ext) => dst[..44].copy_from_slice(&ext),
            None => dst[..12].copy_from_slice(&self.to_base_tag_bytes()),
        }
        Some(len)
    }

    /// Encode the `cicpType` tag into an owned `Vec<u8>` (`12` or `44` bytes).
    #[must_use]
    pub fn to_tag_bytes_vec(self) -> Vec<u8> {
        match self.to_extended_tag_bytes() {
            Some(ext) => ext.to_vec(),
            None => self.to_base_tag_bytes().to_vec(),
        }
    }

    /// Extract the `cicp` tag from an ICC v4+ profile (Input `scnr`, Display `mntr`, or
    /// ColorSpace `spac`), parsing the optional 32-byte custom-primaries extension when
    /// `ColourPrimaries == 2`, declared tag length is `>= 44`, and the profile's header
    /// data colour space is `RGB ` or `YCbr` (ICC `cicpType` Amendment §4.1 & §4.3).
    ///
    /// When the profile header's data colour space is `XYZ ` (or non-RGB/YCbCr), the
    /// custom-primaries extension does not apply (§4.1 NOTE) and `custom_primaries()` is `None`.
    #[must_use]
    pub fn from_icc(icc_bytes: &[u8]) -> Option<Self> {
        extract_cicp_tag(icc_bytes)
    }
}

impl From<Cicp> for CicpTag {
    #[inline]
    fn from(cicp: Cicp) -> Self {
        Self::from_cicp(cicp)
    }
}

impl fmt::Display for CicpTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.custom_primaries {
            Some(cp) => write!(
                f,
                "{} + custom(R=({:.4},{:.4}), G=({:.4},{:.4}), B=({:.4},{:.4}), W=({:.4},{:.4}))",
                self.cicp,
                cp.red().x(),
                cp.red().y(),
                cp.green().x(),
                cp.green().y(),
                cp.blue().x(),
                cp.blue().y(),
                cp.white_point().x(),
                cp.white_point().y(),
            ),
            None => fmt::Display::fmt(&self.cicp, f),
        }
    }
}

/// Internal zero-allocation ICC `cicp` tag extractor shared by `CicpTag::from_icc` and `Cicp::from_icc`.
pub(crate) fn extract_cicp_tag(data: &[u8]) -> Option<CicpTag> {
    const ICC_MIN_SIZE: usize = 132;
    const ICC_MAX_TAG_COUNT: usize = 200;

    if data.len() < ICC_MIN_SIZE {
        return None;
    }
    if data.get(36..40)? != b"acsp" {
        return None;
    }
    // ICC.1 §9.2.17 (and HDR Profiles Amendment §4.4): `cicpTag` exists in v4+ Input, Display, or ColorSpace profiles.
    let version = u32::from_be_bytes(data.get(8..12)?.try_into().ok()?);
    if version < 0x0400_0000 {
        return None;
    }
    if !matches!(data.get(12..16)?, b"scnr" | b"mntr" | b"spac") {
        return None;
    }
    let data_color_space = data.get(16..20)?;
    let custom_primaries_applicable = matches!(data_color_space, b"RGB " | b"YCbr");

    let tag_count = u32::from_be_bytes(data[128..132].try_into().ok()?) as usize;
    let tag_count = tag_count.min(ICC_MAX_TAG_COUNT);

    for i in 0..tag_count {
        let entry_offset = 132 + i * 12;
        let entry = data.get(entry_offset..entry_offset + 12)?;
        if entry[..4] != *b"cicp" {
            continue;
        }
        let data_offset = u32::from_be_bytes(entry[4..8].try_into().ok()?) as usize;
        let data_size = u32::from_be_bytes(entry[8..12].try_into().ok()?) as usize;
        if data_size < CicpTag::BASE_TAG_LEN {
            return None;
        }
        let tag_span = data.get(data_offset..data_offset.checked_add(data_size)?)?;
        if tag_span[..4] != *b"cicp" || tag_span[4..8] != [0; 4] || tag_span[11] > 1 {
            return None;
        }
        let cicp = Cicp::new(tag_span[8], tag_span[9], tag_span[10], tag_span[11] == 1);
        if cicp.color_primaries == 2 && data_size > CicpTag::BASE_TAG_LEN {
            if data_size < CicpTag::EXTENDED_TAG_LEN {
                return None;
            }
            if !custom_primaries_applicable {
                // §4.1 NOTE: amendment does not apply when header data colour space is XYZ (or non-RGB/YCbCr).
                return Some(CicpTag::from_cicp(cicp));
            }
            let ext: [u8; 32] = tag_span[12..44].try_into().ok()?;
            let custom = CustomPrimaries::from_cicp_extension_bytes(ext)?;
            return Some(CicpTag {
                cicp,
                custom_primaries: Some(custom),
            });
        }
        if cicp.color_primaries != 2 && data_size != CicpTag::BASE_TAG_LEN {
            // §4.3 NOTE 1: when ColourPrimaries != 2, declared length is exactly 12 bytes.
            return None;
        }
        return Some(CicpTag::from_cicp(cicp));
    }

    None
}
