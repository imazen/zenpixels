//! Algebraic color-space model, image specification, provenance, and gain-map metadata.
//!
//! [`ColorSpace`] is an algebraic enum so mutually exclusive states — such as
//! Oklab carrying BT.709 primaries, CMYK carrying a PQ transfer curve, or an
//! authoritative ICC profile paired with conflicting primaries — are
//! **unrepresentable by construction**.

use alloc::sync::Arc;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::num::NonZeroU32;
use whereat::At;

use crate::cicp::{Cicp, CicpProjectionError, CicpTag};
use crate::hdr::{ContentLightLevel, DiffuseWhite, HlgRendering, MasteringDisplay};
use crate::layout::{ChannelOrder, InteropTarget, LayoutError, SampleLayout, SampleType};

/// Validated CIE 1931 `(x, y)` chromaticity coordinate with bit-exact equality & hashing.
#[derive(Clone, Copy, Debug)]
pub struct Chromaticity {
    x: f32,
    y: f32,
}

impl Chromaticity {
    /// CIE 1931 D65 standard illuminant `(0.3127, 0.3290)`.
    pub const D65: Self = Self {
        x: 0.3127,
        y: 0.3290,
    };
    /// CIE 1931 D60 ACES illuminant `(0.32168, 0.33767)`.
    pub const D60: Self = Self {
        x: 0.32168,
        y: 0.33767,
    };
    /// CIE 1931 D50 ICC PCS illuminant `(0.3457, 0.3585)`.
    pub const D50: Self = Self {
        x: 0.3457,
        y: 0.3585,
    };
    /// SMPTE RP 431-2 DCI theater white point `(0.314, 0.351)`.
    pub const DCI: Self = Self { x: 0.314, y: 0.351 };
    /// CIE Standard Illuminant C `(0.310, 0.316)` — NTSC 1953 (BT.470M) & Generic Film.
    pub const C: Self = Self { x: 0.310, y: 0.316 };
    /// Equal-energy illuminant E `(1/3, 1/3)`.
    pub const E: Self = Self {
        x: 1.0 / 3.0,
        y: 1.0 / 3.0,
    };

    /// Validate and construct a physical CIE 1931 `(x, y)` chromaticity coordinate
    /// (`x > 0`, `y > 0`, `x + y < 1`, finite — guarantees strictly positive `X, Y, Z`).
    ///
    /// Use [`new_primary`](Self::new_primary) for boundary or virtual/imaginary RGB
    /// primaries such as CIE XYZ (`R = (1, 0), B = (0, 0)`) and ACES2065-1 AP0
    /// (`G = (0, 1)`, `B = (0.0001, -0.0770)`).
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Option<Self> {
        if !x.is_finite() || !y.is_finite() || x <= 0.0 || y <= 0.0 || (x + y) >= 1.0 {
            return None;
        }
        Some(Self { x, y })
    }

    /// Validate and construct an RGB primary coordinate, permitting boundary (`y == 0` such as
    /// CIE XYZ `R = (1, 0), B = (0, 0)`) and imaginary primaries such as ACES2065-1 AP0
    /// (`G = (0.0, 1.0)`, `B = (0.0001, -0.0770)`) and ProPhoto RGB (`x + y = 1.0`).
    ///
    /// Requires finite coordinates with `|x|, |y| <= 4.0`. Non-singularity of the resulting
    /// projective RGB → XYZ matrix is verified by [`CustomPrimaries::new`].
    #[inline]
    #[must_use]
    pub const fn new_primary(x: f32, y: f32) -> Option<Self> {
        if !x.is_finite() || !y.is_finite() || x < -4.0 || x > 4.0 || y < -4.0 || y > 4.0 {
            return None;
        }
        // Normalize `-0.0` to `+0.0` so bitwise Eq/Hash stay canonical.
        let x = if x == 0.0 { 0.0 } else { x };
        let y = if y == 0.0 { 0.0 } else { y };
        Some(Self { x, y })
    }

    /// Validate and construct a physical CIE 1931 `(x, y)` chromaticity coordinate from
    /// ICC `s15Fixed16Number` fixed-point values (`value = raw / 65536.0`, ICC.1:2022 §4.6).
    ///
    /// Every representable coordinate in `[-4.0, 4.0]` converts to `f32` without rounding
    /// error (`|raw| <= 262_144 < 2^24`), so valid `s15Fixed16Number` values round-trip
    /// bit-exact through [`to_s15fixed16`](Self::to_s15fixed16).
    #[inline]
    #[must_use]
    pub const fn from_s15fixed16(x: i32, y: i32) -> Option<Self> {
        Self::new((x as f64 / 65536.0) as f32, (y as f64 / 65536.0) as f32)
    }

    /// Validate and construct an RGB primary coordinate from ICC `s15Fixed16Number` values,
    /// permitting negative or boundary chromaticities such as ACES AP0 `y_B = -0.0770`.
    #[inline]
    #[must_use]
    pub const fn from_s15fixed16_primary(x: i32, y: i32) -> Option<Self> {
        Self::new_primary((x as f64 / 65536.0) as f32, (y as f64 / 65536.0) as f32)
    }

    /// Encode this chromaticity as a pair of ICC `s15Fixed16Number` values (`round(v * 65536)`).
    #[inline]
    #[must_use]
    pub const fn to_s15fixed16(self) -> (i32, i32) {
        (round_s15fixed16(self.x), round_s15fixed16(self.y))
    }

    /// Chromaticity `x` coordinate.
    #[inline]
    #[must_use]
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Chromaticity `y` coordinate.
    #[inline]
    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }

    /// Tuple `(x, y)`.
    #[inline]
    #[must_use]
    pub const fn xy(self) -> (f32, f32) {
        (self.x, self.y)
    }
}

#[inline]
const fn round_s15fixed16(v: f32) -> i32 {
    let scaled = (v as f64) * 65536.0;
    if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        (scaled - 0.5) as i32
    }
}

impl PartialEq for Chromaticity {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x.to_bits() == other.x.to_bits() && self.y.to_bits() == other.y.to_bits()
    }
}
impl Eq for Chromaticity {}
impl Hash for Chromaticity {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

/// Reference white point for XYZ, Gray, or custom-primary RGB spaces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WhitePoint {
    /// CIE D65 `(0.3127, 0.3290)` — sRGB, Display P3, BT.709, BT.2020, Adobe RGB.
    #[default]
    D65,
    /// CIE D50 `(0.3457, 0.3585)` — ICC PCS / ProPhoto RGB.
    D50,
    /// SMPTE RP 431-2 DCI theater white `(0.314, 0.351)`.
    Dci,
    /// CIE Standard Illuminant C `(0.310, 0.316)` — NTSC 1953 (BT.470M) & Generic Film.
    C,
    /// CIE Illuminant E `(1/3, 1/3)` — JPEG XL / XYZ equal-energy.
    E,
    /// Explicit validated CIE 1931 `(x, y)` white point.
    Custom(Chromaticity),
}

impl WhitePoint {
    /// CIE 1931 `(x, y)` coordinates of this white point.
    #[inline]
    #[must_use]
    pub const fn xy(self) -> (f32, f32) {
        match self {
            Self::D65 => Chromaticity::D65.xy(),
            Self::D50 => Chromaticity::D50.xy(),
            Self::Dci => Chromaticity::DCI.xy(),
            Self::C => Chromaticity::C.xy(),
            Self::E => Chromaticity::E.xy(),
            Self::Custom(c) => c.xy(),
        }
    }
}

/// Validated custom tristimulus RGB primaries + white point (ICC `cicpType` amendment,
/// JXL, OpenEXR, PNG `cHRM`, HAGC Mode 3 / SMPTE ST 2094-50).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CustomPrimaries {
    r: Chromaticity,
    g: Chromaticity,
    b: Chromaticity,
    white: Chromaticity,
}

impl CustomPrimaries {
    /// ACES2065-1 (SMPTE ST 2065-1) AP0 primaries with D60 white point (negative `y_B = -0.0770`).
    pub const ACES_AP0: Self = match Self::new(
        (0.7347, 0.2653),
        (0.0, 1.0),
        (0.0001, -0.0770),
        (0.32168, 0.33767),
    ) {
        Some(cp) => cp,
        None => unreachable!(),
    };

    /// ACEScg (Academy S-2014-004) AP1 primaries with D60 white point.
    pub const ACES_CG: Self = match Self::new(
        (0.713, 0.293),
        (0.165, 0.830),
        (0.128, 0.044),
        (0.32168, 0.33767),
    ) {
        Some(cp) => cp,
        None => unreachable!(),
    };

    /// ProPhoto RGB / ROMM RGB (ISO 22028-2) primaries with D50 white point.
    pub const PROPHOTO_RGB: Self = match Self::new(
        (0.7347, 0.2653),
        (0.1596, 0.8404),
        (0.0366, 0.0001),
        (0.3457, 0.3585),
    ) {
        Some(cp) => cp,
        None => unreachable!(),
    };

    /// Construct validated custom primaries, verifying a physical white point
    /// (`x > 0, y > 0, x + y < 1`), valid primary coordinates (including imaginary
    /// primaries such as ACES2065-1 AP0), and a finite, non-singular RGB → XYZ matrix.
    #[must_use]
    pub const fn new(
        r: (f32, f32),
        g: (f32, f32),
        b: (f32, f32),
        white: (f32, f32),
    ) -> Option<Self> {
        let (Some(r), Some(g), Some(b), Some(white)) = (
            Chromaticity::new_primary(r.0, r.1),
            Chromaticity::new_primary(g.0, g.1),
            Chromaticity::new_primary(b.0, b.1),
            Chromaticity::new(white.0, white.1),
        ) else {
            return None;
        };
        let cp = Self { r, g, b, white };
        let Some(m) = rgb_to_xyz_from_xy(cp.r.xy(), cp.g.xy(), cp.b.xy(), cp.white.xy()) else {
            return None;
        };
        if inv3_f64(m).is_none() {
            return None;
        }
        Some(cp)
    }

    /// Construct validated custom primaries from eight ICC `s15Fixed16Number` values
    /// in the canonical field order `[xR, yR, xG, yG, xB, yB, xw, yw]` (ICC.1:2022 §10.3
    /// `cicpType` amendment Table 32 / HAGC Mode 3 / SMPTE ST 2094-50).
    #[must_use]
    pub const fn from_s15fixed16(coords: [i32; 8]) -> Option<Self> {
        let (Some(r), Some(g), Some(b), Some(white)) = (
            Chromaticity::from_s15fixed16_primary(coords[0], coords[1]),
            Chromaticity::from_s15fixed16_primary(coords[2], coords[3]),
            Chromaticity::from_s15fixed16_primary(coords[4], coords[5]),
            Chromaticity::from_s15fixed16(coords[6], coords[7]),
        ) else {
            return None;
        };
        let cp = Self { r, g, b, white };
        let Some(m) = rgb_to_xyz_from_xy(cp.r.xy(), cp.g.xy(), cp.b.xy(), cp.white.xy()) else {
            return None;
        };
        if inv3_f64(m).is_none() {
            return None;
        }
        Some(cp)
    }

    /// Encode as eight ICC `s15Fixed16Number` values `[xR, yR, xG, yG, xB, yB, xw, yw]`.
    #[inline]
    #[must_use]
    pub const fn to_s15fixed16(self) -> [i32; 8] {
        let (xr, yr) = self.r.to_s15fixed16();
        let (xg, yg) = self.g.to_s15fixed16();
        let (xb, yb) = self.b.to_s15fixed16();
        let (xw, yw) = self.white.to_s15fixed16();
        [xr, yr, xg, yg, xb, yb, xw, yw]
    }

    /// Decode the 32-byte big-endian `cicpType` custom-primaries extension (ICC.1 §10.3
    /// bytes `12..44`, eight `s15Fixed16Number` values `[xR, yR, xG, yG, xB, yB, xw, yw]`).
    #[must_use]
    pub const fn from_cicp_extension_bytes(bytes: [u8; 32]) -> Option<Self> {
        Self::from_s15fixed16([
            i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            i32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            i32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            i32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            i32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            i32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
            i32::from_be_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            i32::from_be_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
        ])
    }

    /// Encode the 32-byte big-endian `cicpType` custom-primaries extension (ICC.1 §10.3
    /// bytes `12..44`).
    #[must_use]
    pub const fn to_cicp_extension_bytes(self) -> [u8; 32] {
        let v = self.to_s15fixed16();
        let mut out = [0u8; 32];
        let mut i = 0;
        while i < 8 {
            let b = v[i].to_be_bytes();
            out[i * 4] = b[0];
            out[i * 4 + 1] = b[1];
            out[i * 4 + 2] = b[2];
            out[i * 4 + 3] = b[3];
            i += 1;
        }
        out
    }

    /// Derive the ICC PCS D50-adapted 3×3 primary matrix (`M_D50 = M_adapt · M_native`) per
    /// ICC.1 `cicpType` Amendment Annex 1 (A1.1–A1.4) using Bradford chromatic adaptation to
    /// the ICC PCS adopted white `(0.9642, 1.0000, 0.8249)` (ICC.1:2022 §7.2.16).
    ///
    /// Columns `0`, `1`, `2` (`[m[0][c], m[1][c], m[2][c]]`) are the conforming
    /// `redMatrixColumnTag`, `greenMatrixColumnTag`, and `blueMatrixColumnTag` XYZ triples.
    /// Supports negative chromaticities such as ACES AP0 (`y_B = -0.0770`) without special cases.
    #[must_use]
    pub const fn to_icc_d50_matrix_f64(self) -> Option<[[f64; 3]; 3]> {
        let Some(m_native) =
            rgb_to_xyz_from_xy(self.r.xy(), self.g.xy(), self.b.xy(), self.white.xy())
        else {
            return None;
        };
        let Some(m_adapt) = bradford_adapt_to_icc_d50_f64(self.white.xy()) else {
            return None;
        };
        Some(mul3_mat_f64(m_adapt, m_native))
    }

    /// Red primary `(x, y)`.
    #[inline]
    #[must_use]
    pub const fn r(self) -> Chromaticity {
        self.r
    }

    /// Green primary `(x, y)`.
    #[inline]
    #[must_use]
    pub const fn g(self) -> Chromaticity {
        self.g
    }

    /// Blue primary `(x, y)`.
    #[inline]
    #[must_use]
    pub const fn b(self) -> Chromaticity {
        self.b
    }

    /// White point `(x, y)`.
    #[inline]
    #[must_use]
    pub const fn white(self) -> Chromaticity {
        self.white
    }
}

/// Chromaticity primaries (and reference white point) for tristimulus RGB color spaces.
///
/// Covers all 11 standardized ITU-T H.273 Table 2 `ColourPrimaries` codes losslessly
/// (`Primaries::from_cicp(cp)?.to_cicp() == Some(cp)`), plus Adobe RGB (1998) and
/// validated arbitrary [`CustomPrimaries`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Primaries {
    /// ITU-R BT.709 / sRGB primaries, D65 white (CICP 1).
    #[default]
    Bt709,
    /// ITU-R BT.470-6 System M (NTSC 1953) primaries, Illuminant C white `(0.310, 0.316)` (CICP 4).
    Bt470M,
    /// ITU-R BT.470-6 System B, G (PAL/SECAM) primaries, D65 white (CICP 5).
    Bt470Bg,
    /// ITU-R BT.601-7 525 / SMPTE 170M primaries, D65 white (CICP 6).
    Bt601,
    /// SMPTE 240M primaries, D65 white (CICP 7; identical chromaticities to [`Self::Bt601`], distinct code).
    Smpte240M,
    /// Generic film (colour filters using Illuminant C), Illuminant C white `(0.310, 0.316)` (CICP 8).
    GenericFilm,
    /// ITU-R BT.2020 / BT.2100 wide-gamut primaries, D65 white (CICP 9).
    Bt2020,
    /// SMPTE ST 428-1 CIE 1931 XYZ primaries (`R=(1,0), G=(0,1), B=(0,0)`), Equal-Energy E white (CICP 10).
    Xyz,
    /// SMPTE RP 431-2 DCI-P3 theater primaries, DCI white `(0.314, 0.351)` (CICP 11).
    DciP3,
    /// Apple Display P3 / SMPTE EG 432-1 primaries, D65 white (CICP 12).
    DisplayP3,
    /// EBU Tech. 3213-E / JEDEC P22 phosphors, D65 white (CICP 22).
    Ebu3213,
    /// Adobe RGB (1998) primaries, D65 white (no standard H.273 code).
    AdobeRgb,
    /// Validated custom CIE 1931 primaries + white point (JXL, OpenEXR, PNG `cHRM`).
    Custom(CustomPrimaries),
    /// Unspecified / unknown primaries.
    Unknown,
}

impl Primaries {
    /// CIE 1931 D65 white point `(x, y)`.
    pub const WHITE_D65: (f32, f32) = (0.3127, 0.3290);
    /// SMPTE RP 431-2 DCI theater white point `(x, y)`.
    pub const WHITE_DCI: (f32, f32) = (0.314, 0.351);
    /// CIE Standard Illuminant C white point `(x, y)` (BT.470M / Generic Film).
    pub const WHITE_C: (f32, f32) = (0.310, 0.316);
    /// CIE Equal-Energy Illuminant E white point `(x, y)` (SMPTE ST 428-1 XYZ).
    pub const WHITE_E: (f32, f32) = (1.0 / 3.0, 1.0 / 3.0);

    /// CIE 1931 `(x, y)` coordinates of `(R, G, B)` primaries, or `None` for [`Self::Unknown`].
    #[allow(clippy::type_complexity)]
    #[inline]
    #[must_use]
    pub const fn chromaticity(self) -> Option<((f32, f32), (f32, f32), (f32, f32))> {
        match self {
            Self::Bt709 => Some(((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))),
            Self::Bt470M => Some(((0.67, 0.33), (0.21, 0.71), (0.14, 0.08))),
            Self::Bt470Bg => Some(((0.64, 0.33), (0.29, 0.60), (0.15, 0.06))),
            Self::Bt601 | Self::Smpte240M => Some(((0.630, 0.340), (0.310, 0.595), (0.155, 0.070))),
            Self::GenericFilm => Some(((0.681, 0.319), (0.243, 0.692), (0.145, 0.049))),
            Self::Bt2020 => Some(((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))),
            Self::Xyz => Some(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
            Self::DciP3 | Self::DisplayP3 => Some(((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))),
            Self::Ebu3213 => Some(((0.630, 0.340), (0.295, 0.605), (0.155, 0.077))),
            Self::AdobeRgb => Some(((0.64, 0.33), (0.21, 0.71), (0.15, 0.06))),
            Self::Custom(c) => Some((c.r.xy(), c.g.xy(), c.b.xy())),
            Self::Unknown => None,
        }
    }

    /// Reference white point `(x, y)` of these primaries, or `None` for [`Self::Unknown`].
    #[inline]
    #[must_use]
    pub const fn white_point(self) -> Option<(f32, f32)> {
        match self {
            Self::Bt709
            | Self::Bt470Bg
            | Self::Bt601
            | Self::Smpte240M
            | Self::Bt2020
            | Self::DisplayP3
            | Self::Ebu3213
            | Self::AdobeRgb => Some(Self::WHITE_D65),
            Self::Bt470M | Self::GenericFilm => Some(Self::WHITE_C),
            Self::Xyz => Some(Self::WHITE_E),
            Self::DciP3 => Some(Self::WHITE_DCI),
            Self::Custom(c) => Some(c.white.xy()),
            Self::Unknown => None,
        }
    }

    /// Convert any known [`Primaries`] variant (including named presets such as
    /// [`Self::AdobeRgb`] and [`Self::Bt709`]) into explicit [`CustomPrimaries`],
    /// or `None` for [`Self::Unknown`].
    #[inline]
    #[must_use]
    pub const fn to_custom(self) -> Option<CustomPrimaries> {
        match self {
            Self::Custom(cp) => Some(cp),
            Self::Unknown => None,
            other => {
                let (Some((r, g, b)), Some(w)) = (other.chromaticity(), other.white_point()) else {
                    return None;
                };
                CustomPrimaries::new(r, g, b, w)
            }
        }
    }

    /// Derive the ICC PCS D50-adapted 3×3 primary matrix (`M_D50 = M_adapt · M_native`) per
    /// ICC.1 `cicpType` Amendment Annex 1 using Bradford adaptation to PCS D50
    /// `(0.9642, 1.0000, 0.8249)` (ICC.1:2022 §7.2.16).
    ///
    /// Columns `0`, `1`, `2` are the `rXYZ`, `gXYZ`, `bXYZ` matrix column tags.
    #[must_use]
    pub const fn to_icc_d50_matrix_f64(self) -> Option<[[f64; 3]; 3]> {
        let (Some(m_native), Some(w)) = (self.rgb_to_xyz_f64(), self.white_point()) else {
            return None;
        };
        let Some(m_adapt) = bradford_adapt_to_icc_d50_f64(w) else {
            return None;
        };
        Some(mul3_mat_f64(m_adapt, m_native))
    }

    /// Whether `self` and `other` have identical CIE 1931 primaries and white point
    /// (for example [`Self::Bt601`] and [`Self::Smpte240M`], or a [`Self::Custom`] matching
    /// a named variant). Returns `false` if either side is [`Self::Unknown`] unless both are `Unknown`.
    #[inline]
    #[must_use]
    pub const fn same_chromaticity(self, other: Self) -> bool {
        same_primaries_const(self, other)
    }

    /// Map an ITU-T H.273 Table 2 `ColourPrimaries` code to [`Primaries`].
    ///
    /// Lossless across all 11 standardized codes (`1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22`).
    #[inline]
    #[must_use]
    pub const fn from_cicp(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Bt709),
            4 => Some(Self::Bt470M),
            5 => Some(Self::Bt470Bg),
            6 => Some(Self::Bt601),
            7 => Some(Self::Smpte240M),
            8 => Some(Self::GenericFilm),
            9 => Some(Self::Bt2020),
            10 => Some(Self::Xyz),
            11 => Some(Self::DciP3),
            12 => Some(Self::DisplayP3),
            22 => Some(Self::Ebu3213),
            _ => None,
        }
    }

    /// Map [`Primaries`] to an ITU-T H.273 Table 2 `ColourPrimaries` code, when standardized.
    #[inline]
    #[must_use]
    pub const fn to_cicp(self) -> Option<u8> {
        match self {
            Self::Bt709 => Some(1),
            Self::Bt470M => Some(4),
            Self::Bt470Bg => Some(5),
            Self::Bt601 => Some(6),
            Self::Smpte240M => Some(7),
            Self::GenericFilm => Some(8),
            Self::Bt2020 => Some(9),
            Self::Xyz => Some(10),
            Self::DciP3 => Some(11),
            Self::DisplayP3 => Some(12),
            Self::Ebu3213 => Some(22),
            Self::AdobeRgb | Self::Custom(_) | Self::Unknown => None,
        }
    }

    /// Linear-light luminance weights `[Yr, Yg, Yb]` derived from the primaries' RGB→XYZ `Y` row,
    /// or `None` for [`Self::Unknown`] (use [`Self::luma_weights`] only when a BT.709 fallback is intended).
    #[inline]
    #[must_use]
    pub const fn try_luma_weights(self) -> Option<[f32; 3]> {
        match self {
            Self::Bt709 => Some([0.2126, 0.7152, 0.0722]),
            Self::DisplayP3 => Some([0.228_974_56, 0.691_738_5, 0.079_286_91]),
            Self::Bt2020 => Some([0.2627, 0.6780, 0.0593]),
            Self::AdobeRgb => Some([0.297_344_98, 0.627_363_56, 0.075_291_46]),
            Self::Xyz => Some([0.0, 1.0, 0.0]),
            Self::Bt470M
            | Self::Bt470Bg
            | Self::Bt601
            | Self::Smpte240M
            | Self::GenericFilm
            | Self::DciP3
            | Self::Ebu3213
            | Self::Custom(_) => match self.rgb_to_xyz_f64() {
                Some(m) => Some([m[1][0] as f32, m[1][1] as f32, m[1][2] as f32]),
                None => None,
            },
            Self::Unknown => None,
        }
    }

    /// Linear-light luminance weights `[Yr, Yg, Yb]` derived from the primaries' RGB→XYZ `Y` row
    /// (falls back to BT.709 for [`Self::Unknown`]; prefer [`try_luma_weights`](Self::try_luma_weights)
    /// when unknown primaries must be refused).
    #[inline]
    #[must_use]
    pub const fn luma_weights(self) -> [f32; 3] {
        match self.try_luma_weights() {
            Some(w) => w,
            None => [0.2126, 0.7152, 0.0722],
        }
    }

    /// Linear RGB → CIE XYZ (`Y_white = 1.0`, relative to `self.white_point()`) 3×3 matrix in `f64`.
    #[must_use]
    pub const fn rgb_to_xyz_f64(self) -> Option<[[f64; 3]; 3]> {
        if matches!(self, Self::Xyz) {
            return Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        }
        let (Some((r, g, b)), Some(w)) = (self.chromaticity(), self.white_point()) else {
            return None;
        };
        rgb_to_xyz_from_xy(r, g, b, w)
    }

    /// Exact 3×3 linear-light gamut conversion matrix from `self` to `dst`, applying
    /// Bradford chromatic adaptation automatically when `self.white_point() != dst.white_point()`.
    ///
    /// Returns the identity matrix when `self == dst` (for known primaries),
    /// and `None` if either side is [`Self::Unknown`].
    #[must_use]
    pub const fn gamut_matrix_to(self, dst: Self) -> Option<[[f32; 3]; 3]> {
        let (Some(src_xyz), Some(dst_xyz), Some(sw), Some(dw)) = (
            self.rgb_to_xyz_f64(),
            dst.rgb_to_xyz_f64(),
            self.white_point(),
            dst.white_point(),
        ) else {
            return None;
        };
        if same_primaries_const(self, dst) {
            return Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        }
        let Some(dst_inv) = inv3_f64(dst_xyz) else {
            return None;
        };
        let adapted_src = if sw.0.to_bits() == dw.0.to_bits() && sw.1.to_bits() == dw.1.to_bits() {
            src_xyz
        } else {
            let Some(bradford) = bradford_adapt_f64(sw, dw) else {
                return None;
            };
            mul3_mat_f64(bradford, src_xyz)
        };
        let m = mul3_mat_f64(dst_inv, adapted_src);
        Some([
            [m[0][0] as f32, m[0][1] as f32, m[0][2] as f32],
            [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32],
            [m[2][0] as f32, m[2][1] as f32, m[2][2] as f32],
        ])
    }

    /// Bradford-adapted 3×3 matrix converting CIE XYZ (at `src_white`) → linear RGB in `self`.
    #[must_use]
    pub const fn xyz_to_rgb_f32(self, src_white: WhitePoint) -> Option<[[f32; 3]; 3]> {
        let (Some(dst_xyz), Some(dw)) = (self.rgb_to_xyz_f64(), self.white_point()) else {
            return None;
        };
        let Some(dst_inv) = inv3_f64(dst_xyz) else {
            return None;
        };
        let sw = src_white.xy();
        let m = if sw.0.to_bits() == dw.0.to_bits() && sw.1.to_bits() == dw.1.to_bits() {
            dst_inv
        } else {
            let Some(adapt) = bradford_adapt_f64(sw, dw) else {
                return None;
            };
            mul3_mat_f64(dst_inv, adapt)
        };
        Some([
            [m[0][0] as f32, m[0][1] as f32, m[0][2] as f32],
            [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32],
            [m[2][0] as f32, m[2][1] as f32, m[2][2] as f32],
        ])
    }

    /// Bradford-adapted 3×3 matrix converting linear RGB in `self` → CIE XYZ (at `dst_white`).
    #[must_use]
    pub const fn rgb_to_xyz_f32(self, dst_white: WhitePoint) -> Option<[[f32; 3]; 3]> {
        let (Some(src_xyz), Some(sw)) = (self.rgb_to_xyz_f64(), self.white_point()) else {
            return None;
        };
        let dw = dst_white.xy();
        let m = if sw.0.to_bits() == dw.0.to_bits() && sw.1.to_bits() == dw.1.to_bits() {
            src_xyz
        } else {
            let Some(adapt) = bradford_adapt_f64(sw, dw) else {
                return None;
            };
            mul3_mat_f64(adapt, src_xyz)
        };
        Some([
            [m[0][0] as f32, m[0][1] as f32, m[0][2] as f32],
            [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32],
            [m[2][0] as f32, m[2][1] as f32, m[2][2] as f32],
        ])
    }
}

const fn same_primaries_const(a: Primaries, b: Primaries) -> bool {
    match (
        a.chromaticity(),
        b.chromaticity(),
        a.white_point(),
        b.white_point(),
    ) {
        (Some((ar, ag, ab)), Some((br, bg, bb)), Some(aw), Some(bw)) => {
            ar.0.to_bits() == br.0.to_bits()
                && ar.1.to_bits() == br.1.to_bits()
                && ag.0.to_bits() == bg.0.to_bits()
                && ag.1.to_bits() == bg.1.to_bits()
                && ab.0.to_bits() == bb.0.to_bits()
                && ab.1.to_bits() == bb.1.to_bits()
                && aw.0.to_bits() == bw.0.to_bits()
                && aw.1.to_bits() == bw.1.to_bits()
        }
        (None, None, None, None) => true,
        _ => false,
    }
}

/// Projective RGB → CIE XYZ matrix solver (`Q * t = W`, `M_ij = Q_ij * t_j`).
///
/// Uses `[x, y, 1 - x - y]` column vectors directly without dividing by primary `y`,
/// so boundary primaries with `y == 0` (such as CIE XYZ `R=(1,0), B=(0,0)`) solve cleanly
/// and yield exact finite matrices.
const fn rgb_to_xyz_from_xy(
    r: (f32, f32),
    g: (f32, f32),
    b: (f32, f32),
    white: (f32, f32),
) -> Option<[[f64; 3]; 3]> {
    let (rx, ry) = (r.0 as f64, r.1 as f64);
    let (gx, gy) = (g.0 as f64, g.1 as f64);
    let (bx, by) = (b.0 as f64, b.1 as f64);
    let (wx, wy) = (white.0 as f64, white.1 as f64);
    if wy <= 0.0 || wx <= 0.0 || (wx + wy) >= 1.0 {
        return None;
    }
    let q = [
        [rx, gx, bx],
        [ry, gy, by],
        [1.0 - rx - ry, 1.0 - gx - gy, 1.0 - bx - by],
    ];
    let Some(q_inv) = inv3_f64(q) else {
        return None;
    };
    let w = [wx / wy, 1.0, (1.0 - wx - wy) / wy];
    let t = mul3_vec_f64(q_inv, w);
    if !t[0].is_finite()
        || !t[1].is_finite()
        || !t[2].is_finite()
        || t[0].abs() < 1e-9
        || t[1].abs() < 1e-9
        || t[2].abs() < 1e-9
    {
        return None;
    }
    let m = [
        [q[0][0] * t[0], q[0][1] * t[1], q[0][2] * t[2]],
        [q[1][0] * t[0], q[1][1] * t[1], q[1][2] * t[2]],
        [q[2][0] * t[0], q[2][1] * t[1], q[2][2] * t[2]],
    ];
    let mut row = 0;
    while row < 3 {
        let mut col = 0;
        while col < 3 {
            if !m[row][col].is_finite() {
                return None;
            }
            col += 1;
        }
        row += 1;
    }
    Some(m)
}

/// Bradford chromatic adaptation matrix from source white `(swx, swy)` to destination white `(dwx, dwy)`.
pub const fn bradford_adapt_f64(sw: (f32, f32), dw: (f32, f32)) -> Option<[[f64; 3]; 3]> {
    const M_BFD: [[f64; 3]; 3] = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    let (swx, swy) = (sw.0 as f64, sw.1 as f64);
    let (dwx, dwy) = (dw.0 as f64, dw.1 as f64);
    if swy <= 0.0 || dwy <= 0.0 {
        return None;
    }
    let ws = [swx / swy, 1.0, (1.0 - swx - swy) / swy];
    let wd = [dwx / dwy, 1.0, (1.0 - dwx - dwy) / dwy];
    let cs = mul3_vec_f64(M_BFD, ws);
    let cd = mul3_vec_f64(M_BFD, wd);
    if cs[0] == 0.0 || cs[1] == 0.0 || cs[2] == 0.0 {
        return None;
    }
    let Some(m_inv) = inv3_f64(M_BFD) else {
        return None;
    };
    let scaled = [
        [
            M_BFD[0][0] * (cd[0] / cs[0]),
            M_BFD[0][1] * (cd[0] / cs[0]),
            M_BFD[0][2] * (cd[0] / cs[0]),
        ],
        [
            M_BFD[1][0] * (cd[1] / cs[1]),
            M_BFD[1][1] * (cd[1] / cs[1]),
            M_BFD[1][2] * (cd[1] / cs[1]),
        ],
        [
            M_BFD[2][0] * (cd[2] / cs[2]),
            M_BFD[2][1] * (cd[2] / cs[2]),
            M_BFD[2][2] * (cd[2] / cs[2]),
        ],
    ];
    Some(mul3_mat_f64(m_inv, scaled))
}

/// Bradford chromatic adaptation matrix from source white `(swx, swy)` to the ICC.1:2022 §7.2.16
/// PCS adopted white D50 `nCIEXYZ = (0.9642, 1.0000, 0.8249)` (ICC `cicpType` Amendment Annex 1 A1.3).
const fn bradford_adapt_to_icc_d50_f64(sw: (f32, f32)) -> Option<[[f64; 3]; 3]> {
    const M_BFD: [[f64; 3]; 3] = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    const ICC_PCS_D50_XYZ: [f64; 3] = [0.9642, 1.0, 0.8249];
    let (swx, swy) = (sw.0 as f64, sw.1 as f64);
    if swy <= 0.0 || swx <= 0.0 || (swx + swy) >= 1.0 {
        return None;
    }
    let ws = [swx / swy, 1.0, (1.0 - swx - swy) / swy];
    let cs = mul3_vec_f64(M_BFD, ws);
    let cd = mul3_vec_f64(M_BFD, ICC_PCS_D50_XYZ);
    if cs[0].abs() < 1e-12 || cs[1].abs() < 1e-12 || cs[2].abs() < 1e-12 {
        return None;
    }
    let Some(m_inv) = inv3_f64(M_BFD) else {
        return None;
    };
    let scaled = [
        [
            M_BFD[0][0] * (cd[0] / cs[0]),
            M_BFD[0][1] * (cd[0] / cs[0]),
            M_BFD[0][2] * (cd[0] / cs[0]),
        ],
        [
            M_BFD[1][0] * (cd[1] / cs[1]),
            M_BFD[1][1] * (cd[1] / cs[1]),
            M_BFD[1][2] * (cd[1] / cs[1]),
        ],
        [
            M_BFD[2][0] * (cd[2] / cs[2]),
            M_BFD[2][1] * (cd[2] / cs[2]),
            M_BFD[2][2] * (cd[2] / cs[2]),
        ],
    ];
    Some(mul3_mat_f64(m_inv, scaled))
}

const fn inv3_f64(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let a = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let b = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * a + m[0][1] * b + m[0][2] * c;
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            a * inv,
            -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            b * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv,
        ],
        [
            c * inv,
            -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

const fn mul3_vec_f64(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

const fn mul3_mat_f64(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    let mut r = 0;
    while r < 3 {
        let mut c = 0;
        while c < 3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
            c += 1;
        }
        r += 1;
    }
    out
}

/// Interpretation of an HLG signal: bare scene-referred vs explicit BT.2100-3 display rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HlgMode {
    /// Raw scene-referred HLG signal (ARIB STD-B67 / BT.2100 OETF without display OOTF).
    #[default]
    Scene,
    /// Display-referred HLG rendered with explicit BT.2100-3 Table 5 display conditions.
    Display(HlgRendering),
}

/// Validated pure power decoding gamma (`L = v^gamma`, OETF `v^(1/gamma)`).
///
/// Unambiguously stores the **decoding** (EOTF) exponent (e.g. `2.2` for standard
/// gamma 2.2, `2.6` for DCI-P3 theater, `1.8` for legacy ProPhoto/Apple).
///
/// Stored internally as [`NonZeroU32`] IEEE 754 bit pattern so `Option<Gamma>` fits in 4 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Gamma(NonZeroU32);

impl fmt::Debug for Gamma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Gamma").field(&self.exponent()).finish()
    }
}

impl Gamma {
    /// Pure power decoding gamma `2.6` (distinct from H.273 TC=17 [`Transfer::DciCinema`],
    /// which scales encoded cinema white by `48 / 52.37`).
    pub const DCI_2_6: Self = match Self::new(2.6) {
        Some(g) => g,
        None => unreachable!(),
    };
    /// Standard pure decoding gamma `2.2`.
    pub const GAMMA_2_2: Self = match Self::new(2.2) {
        Some(g) => g,
        None => unreachable!(),
    };
    /// ITU-R BT.470-6 System B, G pure decoding gamma `2.8`.
    pub const GAMMA_2_8: Self = match Self::new(2.8) {
        Some(g) => g,
        None => unreachable!(),
    };
    /// Adobe RGB (1998) pure decoding gamma `563 / 256` (`2.19921875`).
    pub const ADOBE_RGB: Self = match Self::new(563.0 / 256.0) {
        Some(g) => g,
        None => unreachable!(),
    };
    /// Legacy decoding gamma `1.8`.
    pub const GAMMA_1_8: Self = match Self::new(1.8) {
        Some(g) => g,
        None => unreachable!(),
    };

    /// Construct a validated decoding gamma (`0.0 < exponent <= 100.0`, finite).
    #[inline]
    #[must_use]
    pub const fn new(exponent: f32) -> Option<Self> {
        if !exponent.is_finite() || exponent <= 0.0 || exponent > 100.0 {
            return None;
        }
        match NonZeroU32::new(exponent.to_bits()) {
            Some(bits) => Some(Self(bits)),
            None => None,
        }
    }

    /// Construct from an encoding gamma (e.g. JPEG XL `gamma` / PNG `gAMA`, `v = L^enc_gamma`),
    /// storing the reciprocal decoding exponent `1.0 / enc_gamma`.
    #[inline]
    #[must_use]
    pub const fn from_encoding_gamma(encoding_gamma: f32) -> Option<Self> {
        if !encoding_gamma.is_finite() || encoding_gamma <= 0.0 {
            return None;
        }
        Self::new(1.0 / encoding_gamma)
    }

    /// Decoding (EOTF) exponent `gamma` where `L = v^gamma`.
    #[inline]
    #[must_use]
    pub const fn exponent(self) -> f32 {
        f32::from_bits(self.0.get())
    }
}

/// Electro-optical transfer function (non-linear curve between stored values and light).
///
/// Covers all 16 standardized ITU-T H.273 Table 3 `TransferCharacteristics` codes losslessly
/// (`Transfer::from_cicp(tc)?.to_cicp() == Some(tc)`), plus Adobe RGB (`563/256`) and
/// arbitrary validated pure-power [`Gamma`].
///
/// Distinct variants that share an identical mathematical curve (e.g. [`Self::Bt709`],
/// [`Self::Bt601`], [`Self::Iec61966_2_4`], [`Self::Bt2020_10bit`], [`Self::Bt2020_12bit`])
/// preserve their exact H.273 code on round-trip while reporting `true` from
/// [`Transfer::same_curve`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Transfer {
    /// Linear light (`L = v`, CICP 8).
    Linear,
    /// IEC 61966-2-1 sRGB curve (CICP 13).
    #[default]
    Srgb,
    /// ITU-R BT.709-6 camera curve (CICP 1).
    Bt709,
    /// Pure power curve `v^2.2` (ITU-R BT.470-6 System M, CICP 4).
    ///
    /// Distinct from [`Self::AdobeRgb`] (`v^(563/256)` = `2.19921875`).
    Gamma22,
    /// Pure power curve `v^2.8` (ITU-R BT.470-6 System B, G, CICP 5).
    Gamma28,
    /// ITU-R BT.601-7 / SMPTE 170M curve (CICP 6; identical curve to [`Self::Bt709`], distinct code).
    Bt601,
    /// SMPTE 240M curve (`α = 1.1115`, slope `4.0`, CICP 7).
    Smpte240M,
    /// Logarithmic transfer (`100:1` dynamic range, CICP 9).
    Log100,
    /// Logarithmic transfer (`100 * sqrt(10):1` dynamic range, CICP 10).
    Log100Sqrt10,
    /// IEC 61966-2-4 extended xvYCC curve (CICP 11; odd-symmetric BT.709).
    Iec61966_2_4,
    /// ITU-R BT.1361-0 extended colour gamut system (CICP 12).
    Bt1361,
    /// ITU-R BT.2020-2 10-bit curve (CICP 14; identical curve to [`Self::Bt709`], distinct code).
    Bt2020_10bit,
    /// ITU-R BT.2020-2 12-bit curve (CICP 15; identical curve to [`Self::Bt709`], distinct code).
    Bt2020_12bit,
    /// Adobe RGB (1998) pure power curve `v^(563/256)` (`2.19921875`, no exact H.273 code).
    AdobeRgb,
    /// SMPTE ST 428-1 digital cinema transfer (CICP 17: `L = (52.37 / 48.0) * v^2.6`,
    /// where `L = 1.0` = 48 cd/m² theater white encodes to `(48 / 52.37)^(1/2.6) ≈ 0.967043`).
    ///
    /// Distinct from pure power [`Self::Gamma(Gamma::DCI_2_6)`](Gamma::DCI_2_6) (`L = v^2.6`).
    DciCinema,
    /// Explicit validated pure decoding gamma `L = v^gamma` (JXL, pure 2.6, PNG `gAMA`).
    Gamma(Gamma),
    /// SMPTE ST 2084 Perceptual Quantizer (absolute `0..=10,000 cd/m²`, CICP 16).
    Pq,
    /// ARIB STD-B67 / ITU-R BT.2100 Hybrid Log-Gamma (CICP 18).
    Hlg(HlgMode),
    /// Unspecified / unknown transfer curve.
    Unknown,
}

impl Transfer {
    /// Whether this transfer encodes linear light directly.
    #[inline]
    #[must_use]
    pub const fn is_linear(self) -> bool {
        matches!(self, Self::Linear)
    }

    /// Whether this transfer is an HDR transfer (`Pq` or `Hlg`).
    #[inline]
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg(_))
    }

    /// Whether this transfer uses the exact BT.709 / BT.601 / BT.2020 / xvYCC transfer equation.
    #[inline]
    #[must_use]
    pub const fn is_bt709_curve(self) -> bool {
        matches!(
            self,
            Self::Bt709
                | Self::Bt601
                | Self::Iec61966_2_4
                | Self::Bt2020_10bit
                | Self::Bt2020_12bit
        )
    }

    /// Whether `self` and `other` evaluate the exact same mathematical transfer curve
    /// (for instance [`Self::Bt709`], [`Self::Bt601`], [`Self::Iec61966_2_4`],
    /// [`Self::Bt2020_10bit`], and [`Self::Bt2020_12bit`], or [`Self::Gamma`] matching a named
    /// pure-power curve bit-for-bit).
    #[inline]
    #[must_use]
    pub const fn same_curve(self, other: Self) -> bool {
        if self.is_bt709_curve() && other.is_bt709_curve() {
            return true;
        }
        match (self, other) {
            (Self::Linear, Self::Linear)
            | (Self::Srgb, Self::Srgb)
            | (Self::Gamma22, Self::Gamma22)
            | (Self::Gamma28, Self::Gamma28)
            | (Self::Smpte240M, Self::Smpte240M)
            | (Self::Log100, Self::Log100)
            | (Self::Log100Sqrt10, Self::Log100Sqrt10)
            | (Self::Bt1361, Self::Bt1361)
            | (Self::AdobeRgb, Self::AdobeRgb)
            | (Self::DciCinema, Self::DciCinema)
            | (Self::Pq, Self::Pq)
            | (Self::Unknown, Self::Unknown) => true,
            (Self::Gamma(a), Self::Gamma(b)) => a.0.get() == b.0.get(),
            (Self::Gamma22, Self::Gamma(g)) | (Self::Gamma(g), Self::Gamma22) => {
                g.0.get() == Gamma::GAMMA_2_2.0.get()
            }
            (Self::Gamma28, Self::Gamma(g)) | (Self::Gamma(g), Self::Gamma28) => {
                g.0.get() == Gamma::GAMMA_2_8.0.get()
            }
            (Self::AdobeRgb, Self::Gamma(g)) | (Self::Gamma(g), Self::AdobeRgb) => {
                g.0.get() == Gamma::ADOBE_RGB.0.get()
            }
            (Self::Linear, Self::Gamma(g)) | (Self::Gamma(g), Self::Linear) => {
                g.0.get() == 1.0f32.to_bits()
            }
            (Self::Hlg(HlgMode::Scene), Self::Hlg(HlgMode::Scene)) => true,
            (Self::Hlg(HlgMode::Display(a)), Self::Hlg(HlgMode::Display(b))) => {
                a.peak().nits().to_bits() == b.peak().nits().to_bits()
                    && a.black_nits().to_bits() == b.black_nits().to_bits()
                    && a.system_gamma().to_bits() == b.system_gamma().to_bits()
            }
            _ => false,
        }
    }

    /// Map an ITU-T H.273 Table 3 `TransferCharacteristics` code to [`Transfer`].
    ///
    /// Lossless bijection across all 16 standardized codes
    /// (`1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18`).
    #[inline]
    #[must_use]
    pub const fn from_cicp(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Bt709),
            4 => Some(Self::Gamma22),
            5 => Some(Self::Gamma28),
            6 => Some(Self::Bt601),
            7 => Some(Self::Smpte240M),
            8 => Some(Self::Linear),
            9 => Some(Self::Log100),
            10 => Some(Self::Log100Sqrt10),
            11 => Some(Self::Iec61966_2_4),
            12 => Some(Self::Bt1361),
            13 => Some(Self::Srgb),
            14 => Some(Self::Bt2020_10bit),
            15 => Some(Self::Bt2020_12bit),
            16 => Some(Self::Pq),
            17 => Some(Self::DciCinema),
            18 => Some(Self::Hlg(HlgMode::Scene)),
            _ => None,
        }
    }

    /// Map [`Transfer`] to an ITU-T H.273 Table 3 `TransferCharacteristics` code, when standardized.
    ///
    /// Note: [`Self::AdobeRgb`] (`563/256`) and pure [`Self::Gamma`] return `None` so they
    /// are never silently relabeled as H.273 TC=4 (`2.2`) or TC=17 (`SMPTE ST 428-1`).
    #[inline]
    #[must_use]
    pub const fn to_cicp(self) -> Option<u8> {
        match self {
            Self::Bt709 => Some(1),
            Self::Gamma22 => Some(4),
            Self::Gamma28 => Some(5),
            Self::Bt601 => Some(6),
            Self::Smpte240M => Some(7),
            Self::Linear => Some(8),
            Self::Log100 => Some(9),
            Self::Log100Sqrt10 => Some(10),
            Self::Iec61966_2_4 => Some(11),
            Self::Bt1361 => Some(12),
            Self::Srgb => Some(13),
            Self::Bt2020_10bit => Some(14),
            Self::Bt2020_12bit => Some(15),
            Self::Pq => Some(16),
            Self::DciCinema => Some(17),
            Self::Hlg(_) => Some(18),
            Self::AdobeRgb | Self::Gamma(_) | Self::Unknown => None,
        }
    }
}

/// Signal quantization range for integer samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SignalRange {
    /// Full range (`0..=255` in `U8`, `0..=65535` in `U16`).
    #[default]
    Full,
    /// Broadcast studio / limited swing (`16..=235` in `U8`, `4096..=60160` in `U16`).
    Narrow,
}

/// Interpretation of an image's alpha (or padding) channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlphaMode {
    /// No alpha channel is present in the layout.
    #[default]
    None,
    /// Alpha channel is present and logically fully opaque (`1.0` / `255` / `65535`).
    ///
    /// Freshly allocated buffers ([`PixelBuf::new`](crate::PixelBuf::new)) initialize stored alpha
    /// lanes to channel maximum and set `SampleFacts::all_opaque = true`. Zero-copy constructors
    /// ([`PixelBuf::from_typed`](crate::PixelBuf::from_typed), [`PixelBuf::from_bytes`](crate::PixelBuf::from_bytes),
    /// [`PixelView::new`](crate::PixelView::new)) do not mutate caller-supplied storage; converters
    /// and exporters canonicalize unverified `Opaque` alpha lanes to channel maximum unless
    /// `SampleFacts::all_opaque` is proven.
    Opaque,
    /// Unassociated (straight) alpha: RGB is independent of A.
    Straight,
    /// Associated (premultiplied) alpha: RGB is already multiplied by A.
    Premultiplied,
    /// Padding channel (`RGBx` / `BGRx`): bytes carry unspecified values and are ignored.
    Undefined,
}

/// Perceptual / cylindrical color spaces beyond Oklab.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PerceptualSpace {
    /// Polar Oklab (`[L, C, h_deg]`).
    Oklch,
    /// CIE 1976 `L*a*b*` (D65).
    CieLab,
    /// Polar CIE `L*C*h°` (D65).
    CieLch,
    /// Perceptual HDR `Jzazbz` (Safdar et al., PQ-based).
    Jzazbz,
    /// Polar `Jzczhz`.
    Jzczhz,
    /// Cylindrical HSL (`[H_deg, S, L]` over sRGB).
    Hsl,
    /// Cylindrical HSV (`[H_deg, S, V]` over sRGB).
    Hsv,
}

/// ICC rendering intent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RenderingIntent {
    /// Perceptual intent (gamut compression preserving visual relationships).
    #[default]
    Perceptual,
    /// Media-relative colorimetric intent.
    RelativeColorimetric,
    /// Saturation intent.
    Saturation,
    /// ICC-absolute colorimetric intent.
    AbsoluteColorimetric,
}

/// High-level data color space recorded in an ICC profile header (bytes 16..20).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IccColorSpace {
    /// `'RGB '`
    Rgb,
    /// `'GRAY'`
    Gray,
    /// `'CMYK'`
    Cmyk,
    /// `'Lab '`
    Lab,
    /// `'XYZ '`
    Xyz,
    /// `'YCbr'` (YCbCr, ICC.1:2022 Table 19)
    Ycbcr,
}

/// Authoritative color meaning of pixel samples — an algebraic enum with zero
/// contradictory or impossible combinations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorSpace {
    /// Standard tristimulus RGB color space.
    Rgb {
        /// Chromaticity primaries (D65).
        primaries: Primaries,
        /// Non-linear or linear transfer function.
        transfer: Transfer,
        /// Full or narrow studio signal range.
        range: SignalRange,
        /// Alpha or padding interpretation.
        alpha: AlphaMode,
        /// Luminance of relative-linear `1.0` in cd/m² (required when crossing PQ ↔ relative light).
        white: Option<DiffuseWhite>,
    },
    /// Achromatic single-channel luminance / luma.
    Gray {
        /// Transfer function.
        transfer: Transfer,
        /// Full or narrow range.
        range: SignalRange,
        /// Alpha interpretation (`None`, `Opaque`, `Straight`, `Premultiplied`).
        alpha: AlphaMode,
        /// Luminance of relative-linear `1.0` in cd/m².
        white: Option<DiffuseWhite>,
    },
    /// CIE 1931 linear XYZ tristimulus space (e.g. TIFF SGILog / LogLuv).
    Xyz {
        /// Reference white point (e.g. `WhitePoint::D65`, `WhitePoint::E`, `WhitePoint::D50`).
        white_point: WhitePoint,
        /// Alpha interpretation (`None`, `Opaque`, `Straight`, `Premultiplied`).
        alpha: AlphaMode,
        /// Luminance of `Y = 1.0` in cd/m², if known.
        white: Option<DiffuseWhite>,
    },
    /// Björn Ottosson's Oklab — CIE XYZ D65-anchored (no primaries or transfer field!).
    Oklab {
        /// Alpha interpretation (`None`, `Opaque`, `Straight`, `Premultiplied`).
        alpha: AlphaMode,
    },
    /// Additional perceptual or cylindrical color space (`Oklch`, `CieLab`, `Jzazbz`, `Hsl`, `Hsv`, …).
    Perceptual {
        /// Which perceptual/cylindrical model the channels represent.
        space: PerceptualSpace,
        /// Alpha interpretation.
        alpha: AlphaMode,
    },
    /// Subtractive printer ink (`CMYK`) — requires an explicit CMS backend and ICC profile to convert to display color.
    Cmyk {
        /// Whether ink polarity is inverted (`max = no ink`, Adobe JPEG/JXL convention).
        inverted: bool,
        /// Alpha interpretation (`None` for 4-channel CMYK).
        alpha: AlphaMode,
        /// Authoritative CMYK ICC profile bytes, if attached.
        icc: Option<Arc<[u8]>>,
    },
    /// Arbitrary authoritative ICC profile — no conflicting CICP primaries/transfer alongside it.
    Icc {
        /// Raw ICC profile bytes.
        profile: Arc<[u8]>,
        /// Number of color channels (excluding alpha, e.g. 3 for RGB, 1 for Gray, 4 for CMYK).
        channels: u8,
        /// Alpha interpretation.
        alpha: AlphaMode,
        /// Preferred rendering intent.
        intent: RenderingIntent,
    },
    /// Non-color numeric data (depth maps, normals, masks, generic signed/wide payloads).
    Untagged {
        /// Total channel count.
        channels: u8,
        /// Alpha interpretation (`None` for pure numeric data).
        alpha: AlphaMode,
    },
}

impl ColorSpace {
    /// Standard full-range sRGB (opaque / no alpha).
    pub const SRGB: Self = Self::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Srgb,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };

    /// Standard full-range sRGB with straight alpha.
    pub const SRGB_ALPHA: Self = Self::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Srgb,
        range: SignalRange::Full,
        alpha: AlphaMode::Straight,
        white: None,
    };

    /// Linear-light BT.709 / sRGB (no alpha).
    pub const LINEAR_SRGB: Self = Self::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };

    /// Linear-light BT.709 / sRGB with straight alpha.
    pub const LINEAR_SRGB_ALPHA: Self = Self::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::Straight,
        white: None,
    };

    /// Display P3 with sRGB transfer (no alpha).
    pub const DISPLAY_P3: Self = Self::Rgb {
        primaries: Primaries::DisplayP3,
        transfer: Transfer::Srgb,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };

    /// Display P3 with sRGB transfer and straight alpha.
    pub const DISPLAY_P3_ALPHA: Self = Self::Rgb {
        primaries: Primaries::DisplayP3,
        transfer: Transfer::Srgb,
        range: SignalRange::Full,
        alpha: AlphaMode::Straight,
        white: None,
    };

    /// BT.2100 PQ HDR (BT.2020 primaries, ST 2084 PQ, full range, no alpha).
    pub const BT2100_PQ: Self = Self::Rgb {
        primaries: Primaries::Bt2020,
        transfer: Transfer::Pq,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };

    /// BT.2100 HLG HDR scene-referred (no display OOTF selected yet).
    pub const BT2100_HLG_SCENE: Self = Self::Rgb {
        primaries: Primaries::Bt2020,
        transfer: Transfer::Hlg(HlgMode::Scene),
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };

    /// BT.2100 HLG HDR with explicit 1000-nit reference display rendering.
    pub const BT2100_HLG_1000: Self = Self::Rgb {
        primaries: Primaries::Bt2020,
        transfer: Transfer::Hlg(HlgMode::Display(HlgRendering::REFERENCE_1000_NIT)),
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };

    /// Grayscale with sRGB transfer (no alpha).
    pub const GRAY_SRGB: Self = Self::Gray {
        transfer: Transfer::Srgb,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };

    /// Linear grayscale (no alpha).
    pub const GRAY_LINEAR: Self = Self::Gray {
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };

    /// Oklab perceptual space (CIE XYZ D65-anchored, no alpha).
    pub const OKLAB: Self = Self::Oklab {
        alpha: AlphaMode::None,
    };

    /// Oklab + straight alpha.
    pub const OKLABA: Self = Self::Oklab {
        alpha: AlphaMode::Straight,
    };

    /// Return the alpha mode of this color space.
    #[inline]
    #[must_use]
    pub const fn alpha(&self) -> AlphaMode {
        match *self {
            Self::Rgb { alpha, .. }
            | Self::Gray { alpha, .. }
            | Self::Xyz { alpha, .. }
            | Self::Oklab { alpha }
            | Self::Perceptual { alpha, .. }
            | Self::Cmyk { alpha, .. }
            | Self::Icc { alpha, .. }
            | Self::Untagged { alpha, .. } => alpha,
        }
    }

    /// Return a copy with `alpha` replaced.
    #[inline]
    #[must_use]
    pub fn with_alpha(mut self, alpha: AlphaMode) -> Self {
        match &mut self {
            Self::Rgb { alpha: a, .. }
            | Self::Gray { alpha: a, .. }
            | Self::Xyz { alpha: a, .. }
            | Self::Oklab { alpha: a }
            | Self::Perceptual { alpha: a, .. }
            | Self::Cmyk { alpha: a, .. }
            | Self::Icc { alpha: a, .. }
            | Self::Untagged { alpha: a, .. } => *a = alpha,
        }
        self
    }

    /// Set primaries on an [`Self::Rgb`] color space (no-op on non-RGB variants).
    ///
    /// Prefer [`try_with_primaries`](Self::try_with_primaries) when applying to an unknown variant
    /// should be detected rather than silently ignored.
    #[inline]
    #[must_use]
    pub fn with_primaries(mut self, primaries: Primaries) -> Self {
        if let Self::Rgb { primaries: p, .. } = &mut self {
            *p = primaries;
        }
        self
    }

    /// Set primaries on an [`Self::Rgb`] color space, returning `None` on non-RGB variants.
    #[inline]
    #[must_use]
    pub fn try_with_primaries(mut self, primaries: Primaries) -> Option<Self> {
        match &mut self {
            Self::Rgb { primaries: p, .. } => {
                *p = primaries;
                Some(self)
            }
            _ => None,
        }
    }

    /// Set transfer on an [`Self::Rgb`] or [`Self::Gray`] color space (no-op on other variants).
    ///
    /// Prefer [`try_with_transfer`](Self::try_with_transfer) when applying to an incompatible
    /// variant should be detected rather than silently ignored.
    #[inline]
    #[must_use]
    pub fn with_transfer(mut self, transfer: Transfer) -> Self {
        match &mut self {
            Self::Rgb { transfer: t, .. } | Self::Gray { transfer: t, .. } => *t = transfer,
            _ => {}
        }
        self
    }

    /// Set transfer on an [`Self::Rgb`] or [`Self::Gray`] color space, returning `None` on other variants.
    #[inline]
    #[must_use]
    pub fn try_with_transfer(mut self, transfer: Transfer) -> Option<Self> {
        match &mut self {
            Self::Rgb { transfer: t, .. } | Self::Gray { transfer: t, .. } => {
                *t = transfer;
                Some(self)
            }
            _ => None,
        }
    }

    /// Return the signal range of an RGB or Gray color space (`None` for non-range-bearing models).
    #[inline]
    #[must_use]
    pub const fn range(&self) -> Option<SignalRange> {
        match *self {
            Self::Rgb { range, .. } | Self::Gray { range, .. } => Some(range),
            _ => None,
        }
    }

    /// Set signal range on an [`Self::Rgb`] or [`Self::Gray`] color space (no-op on other variants).
    ///
    /// Prefer [`try_with_range`](Self::try_with_range) when applying to an incompatible variant
    /// should be detected rather than silently ignored.
    #[inline]
    #[must_use]
    pub fn with_range(mut self, range: SignalRange) -> Self {
        match &mut self {
            Self::Rgb { range: r, .. } | Self::Gray { range: r, .. } => *r = range,
            _ => {}
        }
        self
    }

    /// Set signal range on an [`Self::Rgb`] or [`Self::Gray`] color space, returning `None` on other variants.
    #[inline]
    #[must_use]
    pub fn try_with_range(mut self, range: SignalRange) -> Option<Self> {
        match &mut self {
            Self::Rgb { range: r, .. } | Self::Gray { range: r, .. } => {
                *r = range;
                Some(self)
            }
            _ => None,
        }
    }

    /// Attach a [`DiffuseWhite`] luminance anchor to an [`Self::Rgb`], [`Self::Gray`], or [`Self::Xyz`] space
    /// (no-op on other variants).
    ///
    /// Prefer [`try_with_white`](Self::try_with_white) when applying to an incompatible variant
    /// should be detected rather than silently ignored.
    #[inline]
    #[must_use]
    pub fn with_white(mut self, white: DiffuseWhite) -> Self {
        match &mut self {
            Self::Rgb { white: w, .. }
            | Self::Gray { white: w, .. }
            | Self::Xyz { white: w, .. } => *w = Some(white),
            _ => {}
        }
        self
    }

    /// Attach a [`DiffuseWhite`] luminance anchor to an [`Self::Rgb`], [`Self::Gray`], or [`Self::Xyz`]
    /// color space, returning `None` on other variants.
    #[inline]
    #[must_use]
    pub fn try_with_white(mut self, white: DiffuseWhite) -> Option<Self> {
        match &mut self {
            Self::Rgb { white: w, .. }
            | Self::Gray { white: w, .. }
            | Self::Xyz { white: w, .. } => {
                *w = Some(white);
                Some(self)
            }
            _ => None,
        }
    }

    /// Return the attached [`DiffuseWhite`] anchor, if any.
    #[inline]
    #[must_use]
    pub const fn white(&self) -> Option<DiffuseWhite> {
        match *self {
            Self::Rgb { white, .. } | Self::Gray { white, .. } | Self::Xyz { white, .. } => white,
            _ => None,
        }
    }

    /// Project this color space to a raw ITU-T H.273 RGB [`Cicp`] syntax tuple when possible.
    ///
    /// Note: this returns `(9, 18, 0, true)` for both [`HlgMode::Scene`] and [`HlgMode::Display`].
    /// When lossless round-trip fidelity of display rendering parameters and diffuse white is
    /// required, use [`project_cicp`](Self::project_cicp).
    #[must_use]
    pub fn to_cicp(&self) -> Option<Cicp> {
        match *self {
            Self::Rgb {
                primaries,
                transfer,
                range,
                ..
            } => Some(Cicp::new(
                primaries.to_cicp()?,
                transfer.to_cicp()?,
                0,
                range == SignalRange::Full,
            )),
            _ => None,
        }
    }

    /// Losslessly project this color space to an ITU-T H.273 [`Cicp`] tuple, refusing when
    /// CICP cannot represent semantics carried on `self` (such as [`HlgMode::Display`] rendering
    /// parameters or an explicit [`DiffuseWhite`] anchor).
    pub fn project_cicp(&self) -> Result<Cicp, CicpProjectionError> {
        match *self {
            Self::Rgb {
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
                let cp = primaries
                    .to_cicp()
                    .ok_or(CicpProjectionError::UnmappedPrimaries)?;
                let tc = transfer
                    .to_cicp()
                    .ok_or(CicpProjectionError::UnmappedTransfer)?;
                Ok(Cicp::new(cp, tc, 0, range == SignalRange::Full))
            }
            _ => Err(CicpProjectionError::NotRgb),
        }
    }

    /// Losslessly project this color space to an ICC.1 `cicpType` tag ([`CicpTag`]),
    /// emitting the 12-byte base form for standardized H.273 primaries or the 44-byte
    /// custom-primaries extension (ICC `cicpType` Amendment, 2026-10-19) for
    /// [`Primaries::AdobeRgb`] and [`Primaries::Custom`].
    pub fn project_cicp_tag(&self) -> Result<CicpTag, CicpProjectionError> {
        CicpTag::from_color_space(self)
    }

    /// Whether this color space is HDR (`Pq` or `Hlg`).
    #[inline]
    #[must_use]
    pub const fn is_hdr(&self) -> bool {
        match *self {
            Self::Rgb { transfer, .. } | Self::Gray { transfer, .. } => transfer.is_hdr(),
            _ => false,
        }
    }
}

/// Structural validation error for raw ICC profile bytes (ISO 15076-1 / ICC.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IccValidationError {
    /// Profile is shorter than the 132-byte minimum (128-byte header + 4-byte tag count).
    TooShort,
    /// Header bytes `36..40` do not contain the `'acsp'` magic signature.
    MissingSignature,
    /// Header bytes `0..4` declared size is `< 132` or exceeds the slice length.
    InvalidDeclaredSize,
    /// Header bytes `12..16` profile class is not an image color-space class (`mntr`, `scnr`, `prtr`, `spac`).
    UnsupportedProfileClass,
    /// Tag table extends past the declared profile size or overflows.
    TagTableOutOfBounds,
    /// A tag entry's `(offset, size)` overlaps the header/tag-table or extends past the declared profile size.
    TagSpanOutOfBounds,
}

impl fmt::Display for IccValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TooShort => "ICC profile is shorter than 132-byte header + tag count",
            Self::MissingSignature => "ICC profile header is missing 'acsp' signature",
            Self::InvalidDeclaredSize => {
                "ICC profile declared size is smaller than 132 bytes or exceeds buffer length"
            }
            Self::UnsupportedProfileClass => {
                "ICC profile class must be Display ('mntr'), Input ('scnr'), Output ('prtr'), or ColorSpace ('spac')"
            }
            Self::TagTableOutOfBounds => "ICC tag table exceeds declared profile size",
            Self::TagSpanOutOfBounds => "ICC tag data span lies outside declared profile bounds",
        })
    }
}

impl core::error::Error for IccValidationError {}

/// Validate an ICC profile's structural framing per ISO 15076-1 / ICC.1:
///
/// - Minimum length of 132 bytes (128-byte header + 4-byte tag count)
/// - `'acsp'` magic signature at bytes `36..40`
/// - Big-endian declared profile size in bytes `0..4` (`132 <= declared_size <= bytes.len()`)
/// - Image color profile class in bytes `12..16` (`mntr`, `scnr`, `prtr`, or `spac`; rejects
///   DeviceLink `link`, Abstract `abst`, NamedColor `nmcl`, and zero/unknown classes)
/// - Tag table length and every tag entry's `(offset, size)` span strictly within
///   `[table_end, declared_size]`
///
/// Does not allocate and requires no `icc` feature tables.
pub fn validate_icc_structure(bytes: &[u8]) -> Result<(), IccValidationError> {
    if bytes.len() < 132 {
        return Err(IccValidationError::TooShort);
    }
    if &bytes[36..40] != b"acsp" {
        return Err(IccValidationError::MissingSignature);
    }
    let declared_size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if declared_size < 132 || declared_size > bytes.len() {
        return Err(IccValidationError::InvalidDeclaredSize);
    }
    if !matches!(&bytes[12..16], b"mntr" | b"scnr" | b"prtr" | b"spac") {
        return Err(IccValidationError::UnsupportedProfileClass);
    }
    let tag_count = u32::from_be_bytes([bytes[128], bytes[129], bytes[130], bytes[131]]) as usize;
    let table_end = tag_count
        .checked_mul(12)
        .and_then(|n| n.checked_add(132))
        .ok_or(IccValidationError::TagTableOutOfBounds)?;
    if table_end > declared_size {
        return Err(IccValidationError::TagTableOutOfBounds);
    }
    for i in 0..tag_count {
        let base = 132 + i * 12;
        let off = u32::from_be_bytes([
            bytes[base + 4],
            bytes[base + 5],
            bytes[base + 6],
            bytes[base + 7],
        ]) as usize;
        let len = u32::from_be_bytes([
            bytes[base + 8],
            bytes[base + 9],
            bytes[base + 10],
            bytes[base + 11],
        ]) as usize;
        if off < table_end || off.checked_add(len).is_none_or(|end| end > declared_size) {
            return Err(IccValidationError::TagSpanOutOfBounds);
        }
    }
    Ok(())
}

/// `Copy`, `const`, drop-free color model for every non-ICC [`ColorSpace`] variant.
///
/// Paired with [`SampleLayout`] inside [`PixelFormat`] so codec capability tables,
/// format presets, and config structs stay `Copy`, `!Drop`, and statically promotable
/// (`&[PixelFormat::RGB8_SRGB, PixelFormat::RGBA8_SRGB]`) while converting losslessly
/// into [`ColorSpace`] / [`ImageSpec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParametricColor {
    /// Standard tristimulus RGB color space.
    Rgb {
        /// Chromaticity primaries.
        primaries: Primaries,
        /// Non-linear or linear transfer function.
        transfer: Transfer,
        /// Full or narrow studio signal range.
        range: SignalRange,
        /// Alpha or padding interpretation.
        alpha: AlphaMode,
        /// Luminance of relative-linear `1.0` in cd/m².
        white: Option<DiffuseWhite>,
    },
    /// Achromatic single-channel luminance / luma.
    Gray {
        /// Transfer function.
        transfer: Transfer,
        /// Full or narrow range.
        range: SignalRange,
        /// Alpha interpretation.
        alpha: AlphaMode,
        /// Luminance of relative-linear `1.0` in cd/m².
        white: Option<DiffuseWhite>,
    },
    /// CIE 1931 linear XYZ tristimulus space.
    Xyz {
        /// Reference white point.
        white_point: WhitePoint,
        /// Alpha interpretation.
        alpha: AlphaMode,
        /// Luminance of `Y = 1.0` in cd/m², if known.
        white: Option<DiffuseWhite>,
    },
    /// Björn Ottosson's Oklab (CIE XYZ D65-anchored).
    Oklab {
        /// Alpha interpretation.
        alpha: AlphaMode,
    },
    /// Additional perceptual or cylindrical color space.
    Perceptual {
        /// Perceptual/cylindrical model.
        space: PerceptualSpace,
        /// Alpha interpretation.
        alpha: AlphaMode,
    },
    /// Unprofiled subtractive printer ink (`CMYK` without attached ICC bytes).
    Cmyk {
        /// Whether ink polarity is inverted (`max = no ink`, Adobe JPEG/JXL convention).
        inverted: bool,
        /// Alpha interpretation.
        alpha: AlphaMode,
    },
    /// Non-color numeric data (depth maps, normals, gain maps, masks).
    Untagged {
        /// Total channel count.
        channels: u8,
        /// Alpha interpretation.
        alpha: AlphaMode,
    },
}

impl ParametricColor {
    /// Standard full-range sRGB (no alpha).
    pub const SRGB: Self = Self::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Srgb,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    /// Standard full-range sRGB with straight alpha.
    pub const SRGB_ALPHA: Self = Self::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Srgb,
        range: SignalRange::Full,
        alpha: AlphaMode::Straight,
        white: None,
    };
    /// Linear-light BT.709 / sRGB (no alpha).
    pub const LINEAR_SRGB: Self = Self::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    /// Linear-light BT.709 / sRGB with straight alpha.
    pub const LINEAR_SRGB_ALPHA: Self = Self::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::Straight,
        white: None,
    };
    /// Grayscale with sRGB transfer (no alpha).
    pub const GRAY_SRGB: Self = Self::Gray {
        transfer: Transfer::Srgb,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    /// Linear grayscale (no alpha).
    pub const GRAY_LINEAR: Self = Self::Gray {
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    /// BT.2100 PQ HDR (BT.2020 primaries, ST 2084 PQ, full range, no alpha).
    pub const BT2100_PQ: Self = Self::Rgb {
        primaries: Primaries::Bt2020,
        transfer: Transfer::Pq,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    /// BT.2100 HLG HDR scene-referred.
    pub const BT2100_HLG_SCENE: Self = Self::Rgb {
        primaries: Primaries::Bt2020,
        transfer: Transfer::Hlg(HlgMode::Scene),
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    /// BT.2100 HLG HDR with explicit 1000-nit reference display rendering.
    pub const BT2100_HLG_1000: Self = Self::Rgb {
        primaries: Primaries::Bt2020,
        transfer: Transfer::Hlg(HlgMode::Display(HlgRendering::REFERENCE_1000_NIT)),
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    /// Oklab (no alpha).
    pub const OKLAB: Self = Self::Oklab {
        alpha: AlphaMode::None,
    };
    /// Oklab + straight alpha.
    pub const OKLABA: Self = Self::Oklab {
        alpha: AlphaMode::Straight,
    };

    /// Convert losslessly into a full [`ColorSpace`] in `const` context.
    #[inline]
    #[must_use]
    pub const fn to_color_space(self) -> ColorSpace {
        match self {
            Self::Rgb {
                primaries,
                transfer,
                range,
                alpha,
                white,
            } => ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                alpha,
                white,
            },
            Self::Gray {
                transfer,
                range,
                alpha,
                white,
            } => ColorSpace::Gray {
                transfer,
                range,
                alpha,
                white,
            },
            Self::Xyz {
                white_point,
                alpha,
                white,
            } => ColorSpace::Xyz {
                white_point,
                alpha,
                white,
            },
            Self::Oklab { alpha } => ColorSpace::Oklab { alpha },
            Self::Perceptual { space, alpha } => ColorSpace::Perceptual { space, alpha },
            Self::Cmyk { inverted, alpha } => ColorSpace::Cmyk {
                inverted,
                alpha,
                icc: None,
            },
            Self::Untagged { channels, alpha } => ColorSpace::Untagged { channels, alpha },
        }
    }

    /// Borrow-free projection from a [`ColorSpace`]; returns `None` if `color` owns an ICC profile.
    #[inline]
    #[must_use]
    pub const fn from_color_space(color: &ColorSpace) -> Option<Self> {
        match *color {
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                alpha,
                white,
            } => Some(Self::Rgb {
                primaries,
                transfer,
                range,
                alpha,
                white,
            }),
            ColorSpace::Gray {
                transfer,
                range,
                alpha,
                white,
            } => Some(Self::Gray {
                transfer,
                range,
                alpha,
                white,
            }),
            ColorSpace::Xyz {
                white_point,
                alpha,
                white,
            } => Some(Self::Xyz {
                white_point,
                alpha,
                white,
            }),
            ColorSpace::Oklab { alpha } => Some(Self::Oklab { alpha }),
            ColorSpace::Perceptual { space, alpha } => Some(Self::Perceptual { space, alpha }),
            ColorSpace::Cmyk {
                inverted,
                alpha,
                icc: None,
            } => Some(Self::Cmyk { inverted, alpha }),
            ColorSpace::Cmyk { icc: Some(_), .. } | ColorSpace::Icc { .. } => None,
            ColorSpace::Untagged { channels, alpha } => Some(Self::Untagged { channels, alpha }),
        }
    }

    /// Return the alpha mode.
    #[inline]
    #[must_use]
    pub const fn alpha(self) -> AlphaMode {
        match self {
            Self::Rgb { alpha, .. }
            | Self::Gray { alpha, .. }
            | Self::Xyz { alpha, .. }
            | Self::Oklab { alpha }
            | Self::Perceptual { alpha, .. }
            | Self::Cmyk { alpha, .. }
            | Self::Untagged { alpha, .. } => alpha,
        }
    }

    /// Replace the alpha mode.
    #[inline]
    #[must_use]
    pub const fn with_alpha(mut self, alpha: AlphaMode) -> Self {
        match &mut self {
            Self::Rgb { alpha: a, .. }
            | Self::Gray { alpha: a, .. }
            | Self::Xyz { alpha: a, .. }
            | Self::Oklab { alpha: a }
            | Self::Perceptual { alpha: a, .. }
            | Self::Cmyk { alpha: a, .. }
            | Self::Untagged { alpha: a, .. } => *a = alpha,
        }
        self
    }

    /// Replace primaries on an RGB color (returns `None` on non-RGB variants).
    #[inline]
    #[must_use]
    pub const fn with_primaries(mut self, primaries: Primaries) -> Option<Self> {
        match &mut self {
            Self::Rgb { primaries: p, .. } => {
                *p = primaries;
                Some(self)
            }
            _ => None,
        }
    }

    /// Replace transfer on an RGB or Gray color (returns `None` on other variants).
    #[inline]
    #[must_use]
    pub const fn with_transfer(mut self, transfer: Transfer) -> Option<Self> {
        match &mut self {
            Self::Rgb { transfer: t, .. } | Self::Gray { transfer: t, .. } => {
                *t = transfer;
                Some(self)
            }
            _ => None,
        }
    }

    /// Replace signal range on an RGB or Gray color (returns `None` on other variants).
    #[inline]
    #[must_use]
    pub const fn with_range(mut self, range: SignalRange) -> Option<Self> {
        match &mut self {
            Self::Rgb { range: r, .. } | Self::Gray { range: r, .. } => {
                *r = range;
                Some(self)
            }
            _ => None,
        }
    }

    /// Attach a [`DiffuseWhite`] anchor on RGB, Gray, or XYZ (returns `None` on other variants).
    #[inline]
    #[must_use]
    pub const fn with_white(mut self, white: DiffuseWhite) -> Option<Self> {
        match &mut self {
            Self::Rgb { white: w, .. }
            | Self::Gray { white: w, .. }
            | Self::Xyz { white: w, .. } => {
                *w = Some(white);
                Some(self)
            }
            _ => None,
        }
    }
}

impl From<ParametricColor> for ColorSpace {
    #[inline]
    fn from(c: ParametricColor) -> Self {
        c.to_color_space()
    }
}

/// `Copy`, `const`, drop-free companion to [`ImageSpec`] covering every non-ICC pixel format.
///
/// Ideal for static codec capability tables (`&[PixelFormat::RGB8_SRGB, PixelFormat::RGBA8_SRGB]`),
/// `#[derive(Copy)]` decoder/encoder configuration structs, and const-context format helpers.
/// Converts losslessly into [`ImageSpec`] via [`to_spec`](Self::to_spec) or `From<PixelFormat>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PixelFormat {
    /// Physical byte/bit layout.
    pub layout: SampleLayout,
    /// Non-ICC algebraic color meaning.
    pub color: ParametricColor,
}

impl PixelFormat {
    /// 8-bit sRGB RGB (`RGB8`).
    pub const RGB8_SRGB: Self = Self::new(SampleLayout::RGB8, ParametricColor::SRGB);
    /// 8-bit sRGB RGBA straight alpha (`RGBA8`).
    pub const RGBA8_SRGB: Self = Self::new(SampleLayout::RGBA8, ParametricColor::SRGB_ALPHA);
    /// 8-bit sRGB BGR (`BGR8`).
    pub const BGR8_SRGB: Self = Self::new(SampleLayout::BGR8, ParametricColor::SRGB);
    /// 8-bit sRGB BGRA straight alpha (`BGRA8`).
    pub const BGRA8_SRGB: Self = Self::new(SampleLayout::BGRA8, ParametricColor::SRGB_ALPHA);
    /// 8-bit sRGB RGBx (4th byte undefined padding).
    pub const RGBX8_SRGB: Self = Self::new(
        SampleLayout::RGBX8,
        ParametricColor::SRGB.with_alpha(AlphaMode::Undefined),
    );
    /// 8-bit sRGB BGRx (4th byte undefined padding).
    pub const BGRX8_SRGB: Self = Self::new(
        SampleLayout::BGRX8,
        ParametricColor::SRGB.with_alpha(AlphaMode::Undefined),
    );
    /// 16-bit sRGB RGB (`RGB16`).
    pub const RGB16_SRGB: Self = Self::new(SampleLayout::RGB16, ParametricColor::SRGB);
    /// 16-bit sRGB RGBA straight alpha (`RGBA16`).
    pub const RGBA16_SRGB: Self = Self::new(SampleLayout::RGBA16, ParametricColor::SRGB_ALPHA);
    /// 32-bit float linear RGB (`RGBF32`).
    pub const RGBF32_LINEAR: Self = Self::new(SampleLayout::RGBF32, ParametricColor::LINEAR_SRGB);
    /// 32-bit float linear RGBA straight alpha (`RGBAF32`).
    pub const RGBAF32_LINEAR: Self =
        Self::new(SampleLayout::RGBAF32, ParametricColor::LINEAR_SRGB_ALPHA);
    /// 16-bit half-float linear RGB (`RGBF16`).
    pub const RGBF16_LINEAR: Self = Self::new(SampleLayout::RGBF16, ParametricColor::LINEAR_SRGB);
    /// 16-bit half-float linear RGBA straight alpha (`RGBAF16`).
    pub const RGBAF16_LINEAR: Self =
        Self::new(SampleLayout::RGBAF16, ParametricColor::LINEAR_SRGB_ALPHA);
    /// 8-bit sRGB grayscale (`GRAY8`).
    pub const GRAY8_SRGB: Self = Self::new(SampleLayout::GRAY8, ParametricColor::GRAY_SRGB);
    /// 8-bit sRGB grayscale + straight alpha (`GRAYA8`).
    pub const GRAYA8_SRGB: Self = Self::new(
        SampleLayout::GRAYA8,
        ParametricColor::GRAY_SRGB.with_alpha(AlphaMode::Straight),
    );
    /// 16-bit sRGB grayscale (`GRAY16`).
    pub const GRAY16_SRGB: Self = Self::new(SampleLayout::GRAY16, ParametricColor::GRAY_SRGB);
    /// 32-bit float linear grayscale (`GRAYF32`).
    pub const GRAYF32_LINEAR: Self = Self::new(SampleLayout::GRAYF32, ParametricColor::GRAY_LINEAR);
    /// 16-bit BT.2100 PQ HDR RGB (`RGB16`, BT.2020, ST 2084).
    pub const RGB16_BT2100_PQ: Self = Self::new(SampleLayout::RGB16, ParametricColor::BT2100_PQ);
    /// 16-bit BT.2100 HLG HDR RGB, scene-referred.
    pub const RGB16_BT2100_HLG_SCENE: Self =
        Self::new(SampleLayout::RGB16, ParametricColor::BT2100_HLG_SCENE);
    /// 16-bit BT.2100 HLG HDR RGB with explicit 1000-nit reference rendering.
    pub const RGB16_BT2100_HLG_1000: Self =
        Self::new(SampleLayout::RGB16, ParametricColor::BT2100_HLG_1000);
    /// 32-bit float Oklab (no alpha).
    pub const OKLAB_F32: Self = Self::new(SampleLayout::OKLABF32, ParametricColor::OKLAB);
    /// 32-bit float Oklab + straight alpha.
    pub const OKLABA_F32: Self = Self::new(SampleLayout::OKLABAF32, ParametricColor::OKLABA);
    /// 8-bit unprofiled CMYK (`0` = no ink).
    pub const CMYK8: Self = Self::new(
        SampleLayout::CMYK8,
        ParametricColor::Cmyk {
            inverted: false,
            alpha: AlphaMode::None,
        },
    );

    /// Construct a [`PixelFormat`] from a [`SampleLayout`] and [`ParametricColor`].
    #[inline]
    #[must_use]
    pub const fn new(layout: SampleLayout, color: ParametricColor) -> Self {
        Self { layout, color }
    }

    /// Convert losslessly into an [`ImageSpec`] in `const` context.
    #[inline]
    #[must_use]
    pub const fn to_spec(self) -> ImageSpec {
        ImageSpec::new(self.layout, self.color.to_color_space())
    }

    /// Validate that `layout` and `color` agree on channels, alpha/padding, range, and sample types.
    #[inline]
    pub fn validate(self) -> Result<(), At<ColorError>> {
        self.to_spec().validate()
    }

    /// Replace primaries on an RGB format (no-op on non-RGB; use
    /// [`try_with_primaries`](Self::try_with_primaries) to detect unsupported variants).
    #[inline]
    #[must_use]
    pub const fn with_primaries(mut self, primaries: Primaries) -> Self {
        if let Some(c) = self.color.with_primaries(primaries) {
            self.color = c;
        }
        self
    }

    /// Replace primaries on an RGB format, returning `None` on non-RGB.
    #[inline]
    #[must_use]
    pub const fn try_with_primaries(mut self, primaries: Primaries) -> Option<Self> {
        match self.color.with_primaries(primaries) {
            Some(c) => {
                self.color = c;
                Some(self)
            }
            None => None,
        }
    }

    /// Replace transfer on an RGB or Gray format (no-op on other variants; use
    /// [`try_with_transfer`](Self::try_with_transfer) to detect unsupported variants).
    #[inline]
    #[must_use]
    pub const fn with_transfer(mut self, transfer: Transfer) -> Self {
        if let Some(c) = self.color.with_transfer(transfer) {
            self.color = c;
        }
        self
    }

    /// Replace transfer on an RGB or Gray format, returning `None` on other variants.
    #[inline]
    #[must_use]
    pub const fn try_with_transfer(mut self, transfer: Transfer) -> Option<Self> {
        match self.color.with_transfer(transfer) {
            Some(c) => {
                self.color = c;
                Some(self)
            }
            None => None,
        }
    }

    /// Replace signal range on an RGB or Gray format (no-op on other variants; use
    /// [`try_with_range`](Self::try_with_range) to detect unsupported variants).
    #[inline]
    #[must_use]
    pub const fn with_range(mut self, range: SignalRange) -> Self {
        if let Some(c) = self.color.with_range(range) {
            self.color = c;
        }
        self
    }

    /// Replace signal range on an RGB or Gray format, returning `None` on other variants.
    #[inline]
    #[must_use]
    pub const fn try_with_range(mut self, range: SignalRange) -> Option<Self> {
        match self.color.with_range(range) {
            Some(c) => {
                self.color = c;
                Some(self)
            }
            None => None,
        }
    }

    /// Replace alpha mode on this format.
    #[inline]
    #[must_use]
    pub const fn with_alpha(mut self, alpha: AlphaMode) -> Self {
        self.color = self.color.with_alpha(alpha);
        self
    }

    /// Attach a [`DiffuseWhite`] anchor (no-op on non-RGB/Gray/XYZ variants; use
    /// [`try_with_white`](Self::try_with_white) to detect unsupported variants).
    #[inline]
    #[must_use]
    pub const fn with_white(mut self, white: DiffuseWhite) -> Self {
        if let Some(c) = self.color.with_white(white) {
            self.color = c;
        }
        self
    }

    /// Attach a [`DiffuseWhite`] anchor, returning `None` on non-RGB/Gray/XYZ variants.
    #[inline]
    #[must_use]
    pub const fn try_with_white(mut self, white: DiffuseWhite) -> Option<Self> {
        match self.color.with_white(white) {
            Some(c) => {
                self.color = c;
                Some(self)
            }
            None => None,
        }
    }
}

impl From<PixelFormat> for ImageSpec {
    #[inline]
    fn from(fmt: PixelFormat) -> Self {
        fmt.to_spec()
    }
}

/// Borrowed-or-owned [`ImageSpec`] carried inside [`PixelView<'a>`](crate::PixelView) and
/// [`PixelViewMut<'a>`](crate::PixelViewMut).
///
/// Eliminates `Arc` refcount bumps on `PixelBuf::as_view()`, `as_view_mut()`, `crop()`,
/// and `sub_rows()`, while storing owned specs behind an [`Arc`] so [`PixelView`](crate::PixelView)
/// and [`PixelViewMut`](crate::PixelViewMut) stay compact (`size_of::<SpecCow<'_>>() == 16`).
///
/// Ordinary view constructors ([`PixelView::new`](crate::PixelView::new),
/// [`PixelViewMut::new`](crate::PixelViewMut::new)) accept `&'a ImageSpec` directly (guaranteed
/// zero heap allocation — pass [`ImageSpec::rgb8_srgb_ref()`] or `&local_spec`). When a view
/// genuinely needs to own a dynamically constructed [`ImageSpec`], pass an explicit [`SpecCow`]
/// via [`PixelView::new_with_spec_cow`](crate::PixelView::new_with_spec_cow) using
/// [`SpecCow::from_owned`], [`SpecCow::from_format`], or [`SpecCow::from_arc`].
#[derive(Clone, Debug)]
pub enum SpecCow<'a> {
    /// Zero-clone reference to an [`ImageSpec`] owned by a buffer, caller, or static preset.
    Borrowed(&'a ImageSpec),
    /// Heap-shared owned [`ImageSpec`] (e.g. a dynamically constructed custom or ICC spec).
    Owned(Arc<ImageSpec>),
}

impl<'a> SpecCow<'a> {
    /// Wrap an owned [`ImageSpec`], borrowing a zero-allocation `&'static ImageSpec` when `spec`
    /// matches a built-in preset, or allocating an [`Arc<ImageSpec>`] otherwise.
    #[inline]
    #[must_use]
    pub fn from_owned(spec: ImageSpec) -> Self {
        if let Some(static_ref) = spec.as_static_ref() {
            Self::Borrowed(static_ref)
        } else {
            Self::Owned(Arc::new(spec))
        }
    }

    /// Wrap a `Copy` [`PixelFormat`], borrowing a zero-allocation `&'static ImageSpec` for all
    /// built-in presets, or allocating an [`Arc<ImageSpec>`] for custom parametric formats.
    #[inline]
    #[must_use]
    pub fn from_format(format: PixelFormat) -> Self {
        Self::from_owned(format.to_spec())
    }

    /// Wrap an existing [`Arc<ImageSpec>`] with zero additional allocation.
    #[inline]
    #[must_use]
    pub const fn from_arc(spec: Arc<ImageSpec>) -> Self {
        Self::Owned(spec)
    }

    /// Borrow the underlying [`ImageSpec`].
    #[inline]
    #[must_use]
    pub fn as_spec(&self) -> &ImageSpec {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(s) => s,
        }
    }

    /// Re-borrow as a zero-clone [`SpecCow::Borrowed`] tied to `&self`.
    #[inline]
    #[must_use]
    pub fn to_borrowed(&self) -> SpecCow<'_> {
        SpecCow::Borrowed(self.as_spec())
    }

    /// Convert into an owned [`ImageSpec`] (cloning only if borrowed or shared).
    #[inline]
    #[must_use]
    pub fn into_owned(self) -> ImageSpec {
        match self {
            Self::Borrowed(s) => s.clone(),
            Self::Owned(s) => Arc::unwrap_or_clone(s),
        }
    }
}

impl core::ops::Deref for SpecCow<'_> {
    type Target = ImageSpec;
    #[inline]
    fn deref(&self) -> &ImageSpec {
        self.as_spec()
    }
}

impl AsRef<ImageSpec> for SpecCow<'_> {
    #[inline]
    fn as_ref(&self) -> &ImageSpec {
        self.as_spec()
    }
}

impl PartialEq for SpecCow<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_spec() == other.as_spec()
    }
}

impl Eq for SpecCow<'_> {}

impl Hash for SpecCow<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_spec().hash(state);
    }
}

impl<'a> From<&'a ImageSpec> for SpecCow<'a> {
    #[inline]
    fn from(spec: &'a ImageSpec) -> Self {
        Self::Borrowed(spec)
    }
}

impl From<Arc<ImageSpec>> for SpecCow<'_> {
    #[inline]
    fn from(spec: Arc<ImageSpec>) -> Self {
        Self::Owned(spec)
    }
}

/// Complete, validated description of an interleaved image: physical [`SampleLayout`] + algebraic [`ColorSpace`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ImageSpec {
    /// Physical byte/bit layout.
    pub layout: SampleLayout,
    /// Authoritative color meaning.
    pub color: ColorSpace,
}

impl AsRef<ImageSpec> for ImageSpec {
    #[inline]
    fn as_ref(&self) -> &ImageSpec {
        self
    }
}

static STATIC_RGB8_SRGB: ImageSpec = ImageSpec::RGB8_SRGB;
static STATIC_RGBA8_SRGB: ImageSpec = ImageSpec::RGBA8_SRGB;
static STATIC_BGR8_SRGB: ImageSpec = ImageSpec::BGR8_SRGB;
static STATIC_BGRA8_SRGB: ImageSpec = ImageSpec::BGRA8_SRGB;
static STATIC_RGBX8_SRGB: ImageSpec = ImageSpec::RGBX8_SRGB;
static STATIC_BGRX8_SRGB: ImageSpec = ImageSpec::BGRX8_SRGB;
static STATIC_RGB16_SRGB: ImageSpec = ImageSpec::RGB16_SRGB;
static STATIC_RGBA16_SRGB: ImageSpec = ImageSpec::RGBA16_SRGB;
static STATIC_RGBF32_LINEAR: ImageSpec = ImageSpec::RGBF32_LINEAR;
static STATIC_RGBAF32_LINEAR: ImageSpec = ImageSpec::RGBAF32_LINEAR;
static STATIC_RGBF16_LINEAR: ImageSpec = ImageSpec::RGBF16_LINEAR;
static STATIC_RGBAF16_LINEAR: ImageSpec = ImageSpec::RGBAF16_LINEAR;
static STATIC_GRAY8_SRGB: ImageSpec = ImageSpec::GRAY8_SRGB;
static STATIC_GRAYA8_SRGB: ImageSpec = ImageSpec::GRAYA8_SRGB;
static STATIC_GRAY16_SRGB: ImageSpec = ImageSpec::GRAY16_SRGB;
static STATIC_GRAYA16_SRGB: ImageSpec = ImageSpec::new(
    SampleLayout::GRAYA16,
    ColorSpace::Gray {
        transfer: Transfer::Srgb,
        range: SignalRange::Full,
        alpha: AlphaMode::Straight,
        white: None,
    },
);
static STATIC_GRAYF32_LINEAR: ImageSpec = ImageSpec::GRAYF32_LINEAR;
static STATIC_RGBF32_SRGB: ImageSpec = ImageSpec::new(SampleLayout::RGBF32, ColorSpace::SRGB);
static STATIC_RGBAF32_SRGB: ImageSpec =
    ImageSpec::new(SampleLayout::RGBAF32, ColorSpace::SRGB_ALPHA);
static STATIC_RGB16_BT2100_PQ: ImageSpec = ImageSpec::RGB16_BT2100_PQ;
static STATIC_RGB16_BT2100_HLG_SCENE: ImageSpec = ImageSpec::RGB16_BT2100_HLG_SCENE;
static STATIC_RGB16_BT2100_HLG_1000: ImageSpec = ImageSpec::RGB16_BT2100_HLG_1000;
static STATIC_OKLAB_F32: ImageSpec = ImageSpec::OKLAB_F32;
static STATIC_OKLABA_F32: ImageSpec = ImageSpec::OKLABA_F32;

impl ImageSpec {
    /// 8-bit sRGB RGB (`RGB8`).
    ///
    /// Note: because [`ImageSpec`] contains `ColorSpace::Icc { profile: Arc<[u8]> }` (non-trivial
    /// `Drop`), Rust cannot static-promote `&ImageSpec::RGB8_SRGB`. Use [`Self::rgb8_srgb_ref`]
    /// (or a local `let` / `static` binding) when borrowing `&ImageSpec` for [`PixelView::new`](crate::PixelView::new).
    pub const RGB8_SRGB: Self = PixelFormat::RGB8_SRGB.to_spec();
    /// 8-bit sRGB RGBA straight alpha (`RGBA8`). See [`Self::rgba8_srgb_ref`] for `&'static ImageSpec`.
    pub const RGBA8_SRGB: Self = PixelFormat::RGBA8_SRGB.to_spec();
    /// 8-bit sRGB BGR (`BGR8`). See [`Self::bgr8_srgb_ref`] for `&'static ImageSpec`.
    pub const BGR8_SRGB: Self = PixelFormat::BGR8_SRGB.to_spec();
    /// 8-bit sRGB BGRA straight alpha (`BGRA8`). See [`Self::bgra8_srgb_ref`] for `&'static ImageSpec`.
    pub const BGRA8_SRGB: Self = PixelFormat::BGRA8_SRGB.to_spec();
    /// 8-bit sRGB RGBx (4th byte undefined padding). See [`Self::rgbx8_srgb_ref`] for `&'static ImageSpec`.
    pub const RGBX8_SRGB: Self = PixelFormat::RGBX8_SRGB.to_spec();
    /// 8-bit sRGB BGRx (4th byte undefined padding). See [`Self::bgrx8_srgb_ref`] for `&'static ImageSpec`.
    pub const BGRX8_SRGB: Self = PixelFormat::BGRX8_SRGB.to_spec();
    /// 16-bit sRGB RGB (`RGB16`). See [`Self::rgb16_srgb_ref`] for `&'static ImageSpec`.
    pub const RGB16_SRGB: Self = PixelFormat::RGB16_SRGB.to_spec();
    /// 16-bit sRGB RGBA straight alpha (`RGBA16`). See [`Self::rgba16_srgb_ref`] for `&'static ImageSpec`.
    pub const RGBA16_SRGB: Self = PixelFormat::RGBA16_SRGB.to_spec();
    /// 32-bit float linear RGB (`RGBF32`). See [`Self::rgbf32_linear_ref`] for `&'static ImageSpec`.
    pub const RGBF32_LINEAR: Self = PixelFormat::RGBF32_LINEAR.to_spec();
    /// 32-bit float linear RGBA straight alpha (`RGBAF32`). See [`Self::rgbaf32_linear_ref`] for `&'static ImageSpec`.
    pub const RGBAF32_LINEAR: Self = PixelFormat::RGBAF32_LINEAR.to_spec();
    /// 16-bit half-float linear RGB (`RGBF16`). See [`Self::rgbf16_linear_ref`] for `&'static ImageSpec`.
    pub const RGBF16_LINEAR: Self = PixelFormat::RGBF16_LINEAR.to_spec();
    /// 16-bit half-float linear RGBA straight alpha (`RGBAF16`). See [`Self::rgbaf16_linear_ref`] for `&'static ImageSpec`.
    pub const RGBAF16_LINEAR: Self = PixelFormat::RGBAF16_LINEAR.to_spec();
    /// 8-bit sRGB grayscale (`GRAY8`). See [`Self::gray8_srgb_ref`] for `&'static ImageSpec`.
    pub const GRAY8_SRGB: Self = PixelFormat::GRAY8_SRGB.to_spec();
    /// 8-bit sRGB grayscale + straight alpha (`GRAYA8`). See [`Self::graya8_srgb_ref`] for `&'static ImageSpec`.
    pub const GRAYA8_SRGB: Self = PixelFormat::GRAYA8_SRGB.to_spec();
    /// 16-bit sRGB grayscale (`GRAY16`). See [`Self::gray16_srgb_ref`] for `&'static ImageSpec`.
    pub const GRAY16_SRGB: Self = PixelFormat::GRAY16_SRGB.to_spec();
    /// 32-bit float linear grayscale (`GRAYF32`). See [`Self::grayf32_linear_ref`] for `&'static ImageSpec`.
    pub const GRAYF32_LINEAR: Self = PixelFormat::GRAYF32_LINEAR.to_spec();
    /// 16-bit BT.2100 PQ HDR RGB (`RGB16`, BT.2020, ST 2084). See [`Self::rgb16_bt2100_pq_ref`] for `&'static ImageSpec`.
    pub const RGB16_BT2100_PQ: Self = PixelFormat::RGB16_BT2100_PQ.to_spec();
    /// 16-bit BT.2100 HLG HDR RGB, scene-referred. See [`Self::rgb16_bt2100_hlg_scene_ref`] for `&'static ImageSpec`.
    pub const RGB16_BT2100_HLG_SCENE: Self = PixelFormat::RGB16_BT2100_HLG_SCENE.to_spec();
    /// 16-bit BT.2100 HLG HDR RGB with explicit 1000-nit reference rendering. See [`Self::rgb16_bt2100_hlg_1000_ref`] for `&'static ImageSpec`.
    pub const RGB16_BT2100_HLG_1000: Self = PixelFormat::RGB16_BT2100_HLG_1000.to_spec();
    /// 32-bit float Oklab (no alpha). See [`Self::oklab_f32_ref`] for `&'static ImageSpec`.
    pub const OKLAB_F32: Self = PixelFormat::OKLAB_F32.to_spec();
    /// 32-bit float Oklab + straight alpha. See [`Self::oklaba_f32_ref`] for `&'static ImageSpec`.
    pub const OKLABA_F32: Self = PixelFormat::OKLABA_F32.to_spec();

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGB8_SRGB`].
    #[inline]
    #[must_use]
    pub fn rgb8_srgb_ref() -> &'static Self {
        &STATIC_RGB8_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGBA8_SRGB`].
    #[inline]
    #[must_use]
    pub fn rgba8_srgb_ref() -> &'static Self {
        &STATIC_RGBA8_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::BGR8_SRGB`].
    #[inline]
    #[must_use]
    pub fn bgr8_srgb_ref() -> &'static Self {
        &STATIC_BGR8_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::BGRA8_SRGB`].
    #[inline]
    #[must_use]
    pub fn bgra8_srgb_ref() -> &'static Self {
        &STATIC_BGRA8_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGBX8_SRGB`].
    #[inline]
    #[must_use]
    pub fn rgbx8_srgb_ref() -> &'static Self {
        &STATIC_RGBX8_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::BGRX8_SRGB`].
    #[inline]
    #[must_use]
    pub fn bgrx8_srgb_ref() -> &'static Self {
        &STATIC_BGRX8_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGB16_SRGB`].
    #[inline]
    #[must_use]
    pub fn rgb16_srgb_ref() -> &'static Self {
        &STATIC_RGB16_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGBA16_SRGB`].
    #[inline]
    #[must_use]
    pub fn rgba16_srgb_ref() -> &'static Self {
        &STATIC_RGBA16_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGBF32_LINEAR`].
    #[inline]
    #[must_use]
    pub fn rgbf32_linear_ref() -> &'static Self {
        &STATIC_RGBF32_LINEAR
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGBAF32_LINEAR`].
    #[inline]
    #[must_use]
    pub fn rgbaf32_linear_ref() -> &'static Self {
        &STATIC_RGBAF32_LINEAR
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGBF16_LINEAR`].
    #[inline]
    #[must_use]
    pub fn rgbf16_linear_ref() -> &'static Self {
        &STATIC_RGBF16_LINEAR
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGBAF16_LINEAR`].
    #[inline]
    #[must_use]
    pub fn rgbaf16_linear_ref() -> &'static Self {
        &STATIC_RGBAF16_LINEAR
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::GRAY8_SRGB`].
    #[inline]
    #[must_use]
    pub fn gray8_srgb_ref() -> &'static Self {
        &STATIC_GRAY8_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::GRAYA8_SRGB`].
    #[inline]
    #[must_use]
    pub fn graya8_srgb_ref() -> &'static Self {
        &STATIC_GRAYA8_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::GRAY16_SRGB`].
    #[inline]
    #[must_use]
    pub fn gray16_srgb_ref() -> &'static Self {
        &STATIC_GRAY16_SRGB
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::GRAYF32_LINEAR`].
    #[inline]
    #[must_use]
    pub fn grayf32_linear_ref() -> &'static Self {
        &STATIC_GRAYF32_LINEAR
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGB16_BT2100_PQ`].
    #[inline]
    #[must_use]
    pub fn rgb16_bt2100_pq_ref() -> &'static Self {
        &STATIC_RGB16_BT2100_PQ
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGB16_BT2100_HLG_SCENE`].
    #[inline]
    #[must_use]
    pub fn rgb16_bt2100_hlg_scene_ref() -> &'static Self {
        &STATIC_RGB16_BT2100_HLG_SCENE
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::RGB16_BT2100_HLG_1000`].
    #[inline]
    #[must_use]
    pub fn rgb16_bt2100_hlg_1000_ref() -> &'static Self {
        &STATIC_RGB16_BT2100_HLG_1000
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::OKLAB_F32`].
    #[inline]
    #[must_use]
    pub fn oklab_f32_ref() -> &'static Self {
        &STATIC_OKLAB_F32
    }

    /// Zero-allocation `&'static ImageSpec` for [`Self::OKLABA_F32`].
    #[inline]
    #[must_use]
    pub fn oklaba_f32_ref() -> &'static Self {
        &STATIC_OKLABA_F32
    }

    /// Return a `'static` preset reference if `self` equals one of the built-in common specs.
    ///
    /// Used by [`SpecCow::from_owned`] and interop bridges (`from_pixels`, `from_imgref`,
    /// `from_image_buffer`) to avoid heap allocation for standard sRGB / linear / HDR / Oklab specs.
    #[inline]
    #[must_use]
    pub fn as_static_ref(&self) -> Option<&'static Self> {
        let candidate: &'static Self = match self.layout {
            SampleLayout::RGB8 => &STATIC_RGB8_SRGB,
            SampleLayout::RGBA8 => &STATIC_RGBA8_SRGB,
            SampleLayout::BGR8 => &STATIC_BGR8_SRGB,
            SampleLayout::BGRA8 => &STATIC_BGRA8_SRGB,
            SampleLayout::RGBX8 => &STATIC_RGBX8_SRGB,
            SampleLayout::BGRX8 => &STATIC_BGRX8_SRGB,
            SampleLayout::RGB16 => {
                if self == &STATIC_RGB16_SRGB {
                    return Some(&STATIC_RGB16_SRGB);
                } else if self == &STATIC_RGB16_BT2100_PQ {
                    return Some(&STATIC_RGB16_BT2100_PQ);
                } else if self == &STATIC_RGB16_BT2100_HLG_SCENE {
                    return Some(&STATIC_RGB16_BT2100_HLG_SCENE);
                } else if self == &STATIC_RGB16_BT2100_HLG_1000 {
                    return Some(&STATIC_RGB16_BT2100_HLG_1000);
                }
                return None;
            }
            SampleLayout::RGBA16 => &STATIC_RGBA16_SRGB,
            SampleLayout::RGBF32 => {
                if self == &STATIC_RGBF32_LINEAR {
                    return Some(&STATIC_RGBF32_LINEAR);
                } else if self == &STATIC_RGBF32_SRGB {
                    return Some(&STATIC_RGBF32_SRGB);
                }
                return None;
            }
            SampleLayout::RGBAF32 => {
                if self == &STATIC_RGBAF32_LINEAR {
                    return Some(&STATIC_RGBAF32_LINEAR);
                } else if self == &STATIC_RGBAF32_SRGB {
                    return Some(&STATIC_RGBAF32_SRGB);
                }
                return None;
            }
            SampleLayout::RGBF16 => &STATIC_RGBF16_LINEAR,
            SampleLayout::RGBAF16 => &STATIC_RGBAF16_LINEAR,
            SampleLayout::GRAY8 => &STATIC_GRAY8_SRGB,
            SampleLayout::GRAYA8 => &STATIC_GRAYA8_SRGB,
            SampleLayout::GRAY16 => &STATIC_GRAY16_SRGB,
            SampleLayout::GRAYA16 => &STATIC_GRAYA16_SRGB,
            SampleLayout::GRAYF32 => &STATIC_GRAYF32_LINEAR,
            SampleLayout::OKLABF32 => &STATIC_OKLAB_F32,
            SampleLayout::OKLABAF32 => &STATIC_OKLABA_F32,
            _ => return None,
        };
        if self == candidate {
            Some(candidate)
        } else {
            None
        }
    }

    /// Construct an [`ImageSpec`] from a layout and color space.
    #[inline]
    #[must_use]
    pub const fn new(layout: SampleLayout, color: ColorSpace) -> Self {
        Self { layout, color }
    }

    /// Construct an [`ImageSpec`] from a drop-free `Copy` [`PixelFormat`].
    #[inline]
    #[must_use]
    pub const fn from_format(format: PixelFormat) -> Self {
        format.to_spec()
    }

    /// Project this spec into a `Copy` [`PixelFormat`], returning `None` if it owns an ICC profile.
    #[inline]
    #[must_use]
    pub const fn as_format(&self) -> Option<PixelFormat> {
        match ParametricColor::from_color_space(&self.color) {
            Some(color) => Some(PixelFormat::new(self.layout, color)),
            None => None,
        }
    }

    /// 8-bit sRGB RGB (`RGB8`).
    #[inline]
    #[must_use]
    pub const fn rgb8_srgb() -> Self {
        Self::new(SampleLayout::RGB8, ColorSpace::SRGB)
    }

    /// 8-bit sRGB RGBA straight alpha (`RGBA8`).
    #[inline]
    #[must_use]
    pub const fn rgba8_srgb() -> Self {
        Self::new(SampleLayout::RGBA8, ColorSpace::SRGB_ALPHA)
    }

    /// 8-bit sRGB BGR (`BGR8`).
    #[inline]
    #[must_use]
    pub const fn bgr8_srgb() -> Self {
        Self::new(SampleLayout::BGR8, ColorSpace::SRGB)
    }

    /// 8-bit sRGB BGRA straight alpha (`BGRA8`).
    #[inline]
    #[must_use]
    pub const fn bgra8_srgb() -> Self {
        Self::new(SampleLayout::BGRA8, ColorSpace::SRGB_ALPHA)
    }

    /// 8-bit sRGB RGBx (4th byte undefined padding).
    #[inline]
    #[must_use]
    pub const fn rgbx8_srgb() -> Self {
        Self::new(
            SampleLayout::RGBX8,
            ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: AlphaMode::Undefined,
                white: None,
            },
        )
    }

    /// 8-bit sRGB BGRx (4th byte undefined padding).
    #[inline]
    #[must_use]
    pub const fn bgrx8_srgb() -> Self {
        Self::new(
            SampleLayout::BGRX8,
            ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: AlphaMode::Undefined,
                white: None,
            },
        )
    }

    /// 16-bit sRGB RGB (`RGB16`).
    #[inline]
    #[must_use]
    pub const fn rgb16_srgb() -> Self {
        Self::new(SampleLayout::RGB16, ColorSpace::SRGB)
    }

    /// 16-bit sRGB RGBA straight alpha (`RGBA16`).
    #[inline]
    #[must_use]
    pub const fn rgba16_srgb() -> Self {
        Self::new(SampleLayout::RGBA16, ColorSpace::SRGB_ALPHA)
    }

    /// 32-bit float linear RGB (`RGBF32`).
    #[inline]
    #[must_use]
    pub const fn rgbf32_linear() -> Self {
        Self::new(SampleLayout::RGBF32, ColorSpace::LINEAR_SRGB)
    }

    /// 32-bit float linear RGBA straight alpha (`RGBAF32`).
    #[inline]
    #[must_use]
    pub const fn rgbaf32_linear() -> Self {
        Self::new(SampleLayout::RGBAF32, ColorSpace::LINEAR_SRGB_ALPHA)
    }

    /// 16-bit half-float linear RGB (`RGBF16`).
    #[inline]
    #[must_use]
    pub const fn rgbf16_linear() -> Self {
        Self::new(SampleLayout::RGBF16, ColorSpace::LINEAR_SRGB)
    }

    /// 16-bit half-float linear RGBA straight alpha (`RGBAF16`).
    #[inline]
    #[must_use]
    pub const fn rgbaf16_linear() -> Self {
        Self::new(SampleLayout::RGBAF16, ColorSpace::LINEAR_SRGB_ALPHA)
    }

    /// 8-bit sRGB grayscale (`GRAY8`).
    #[inline]
    #[must_use]
    pub const fn gray8_srgb() -> Self {
        Self::new(SampleLayout::GRAY8, ColorSpace::GRAY_SRGB)
    }

    /// 8-bit sRGB grayscale + straight alpha (`GRAYA8`).
    #[inline]
    #[must_use]
    pub const fn graya8_srgb() -> Self {
        Self::new(
            SampleLayout::GRAYA8,
            ColorSpace::Gray {
                transfer: Transfer::Srgb,
                range: SignalRange::Full,
                alpha: AlphaMode::Straight,
                white: None,
            },
        )
    }

    /// 16-bit sRGB grayscale (`GRAY16`).
    #[inline]
    #[must_use]
    pub const fn gray16_srgb() -> Self {
        Self::new(SampleLayout::GRAY16, ColorSpace::GRAY_SRGB)
    }

    /// 32-bit float linear grayscale (`GRAYF32`).
    #[inline]
    #[must_use]
    pub const fn grayf32_linear() -> Self {
        Self::new(SampleLayout::GRAYF32, ColorSpace::GRAY_LINEAR)
    }

    /// 16-bit BT.2100 PQ HDR RGB (`RGB16`, BT.2020, ST 2084).
    #[inline]
    #[must_use]
    pub const fn rgb16_bt2100_pq() -> Self {
        Self::new(SampleLayout::RGB16, ColorSpace::BT2100_PQ)
    }

    /// 16-bit BT.2100 HLG HDR RGB, scene-referred.
    #[inline]
    #[must_use]
    pub const fn rgb16_bt2100_hlg_scene() -> Self {
        Self::new(SampleLayout::RGB16, ColorSpace::BT2100_HLG_SCENE)
    }

    /// 16-bit BT.2100 HLG HDR RGB with explicit 1000-nit reference rendering.
    #[inline]
    #[must_use]
    pub const fn rgb16_bt2100_hlg_1000() -> Self {
        Self::new(SampleLayout::RGB16, ColorSpace::BT2100_HLG_1000)
    }

    /// 32-bit float Oklab (CIE XYZ D65-anchored, no alpha).
    #[inline]
    #[must_use]
    pub const fn oklab_f32() -> Self {
        Self::new(SampleLayout::OKLABF32, ColorSpace::OKLAB)
    }

    /// 32-bit float Oklab + straight alpha.
    #[inline]
    #[must_use]
    pub const fn oklaba_f32() -> Self {
        Self::new(SampleLayout::OKLABAF32, ColorSpace::OKLABA)
    }

    /// 8-bit CMYK (`0` = no ink, optional ICC profile).
    #[inline]
    #[must_use]
    pub const fn cmyk8(icc: Option<Arc<[u8]>>) -> Self {
        Self::new(
            SampleLayout::CMYK8,
            ColorSpace::Cmyk {
                inverted: false,
                alpha: AlphaMode::None,
                icc,
            },
        )
    }

    /// Replace primaries on an RGB spec (no-op on non-RGB).
    #[inline]
    #[must_use]
    pub fn with_primaries(mut self, primaries: Primaries) -> Self {
        self.color = self.color.with_primaries(primaries);
        self
    }

    /// Replace primaries on an RGB spec, returning `None` on non-RGB variants.
    #[inline]
    #[must_use]
    pub fn try_with_primaries(mut self, primaries: Primaries) -> Option<Self> {
        self.color = self.color.try_with_primaries(primaries)?;
        Some(self)
    }

    /// Replace transfer on an RGB or Gray spec (no-op on other variants).
    #[inline]
    #[must_use]
    pub fn with_transfer(mut self, transfer: Transfer) -> Self {
        self.color = self.color.with_transfer(transfer);
        self
    }

    /// Replace transfer on an RGB or Gray spec, returning `None` on other variants.
    #[inline]
    #[must_use]
    pub fn try_with_transfer(mut self, transfer: Transfer) -> Option<Self> {
        self.color = self.color.try_with_transfer(transfer)?;
        Some(self)
    }

    /// Replace signal range on an RGB or Gray spec (no-op on other variants).
    #[inline]
    #[must_use]
    pub fn with_range(mut self, range: SignalRange) -> Self {
        self.color = self.color.with_range(range);
        self
    }

    /// Replace signal range on an RGB or Gray spec, returning `None` on other variants.
    #[inline]
    #[must_use]
    pub fn try_with_range(mut self, range: SignalRange) -> Option<Self> {
        self.color = self.color.try_with_range(range)?;
        Some(self)
    }

    /// Replace alpha mode on this spec.
    #[inline]
    #[must_use]
    pub fn with_alpha(mut self, alpha: AlphaMode) -> Self {
        self.color = self.color.with_alpha(alpha);
        self
    }

    /// Attach a [`DiffuseWhite`] anchor (no-op on non-RGB/Gray/XYZ variants).
    #[inline]
    #[must_use]
    pub fn with_white(mut self, white: DiffuseWhite) -> Self {
        self.color = self.color.with_white(white);
        self
    }

    /// Attach a [`DiffuseWhite`] anchor, returning `None` on non-RGB/Gray/XYZ variants.
    #[inline]
    #[must_use]
    pub fn try_with_white(mut self, white: DiffuseWhite) -> Option<Self> {
        self.color = self.color.try_with_white(white)?;
        Some(self)
    }

    /// Whether this spec can represent HDR light levels above SDR diffuse white (`> 1.0`).
    ///
    /// Returns `true` for PQ and HLG transfers (any sample type), linear floating-point
    /// RGB/Gray (`F16`/`F32`/`F64`), floating-point CIE XYZ, and ICC profiles backed by
    /// floating-point storage or carrying an HDR (`PQ`/`HLG`) `cicp` tag.
    #[inline]
    #[must_use]
    pub fn can_represent_hdr(&self) -> bool {
        match &self.color {
            ColorSpace::Rgb { transfer, .. } | ColorSpace::Gray { transfer, .. } => {
                matches!(transfer, Transfer::Pq | Transfer::Hlg(_))
                    || (self.layout.sample.is_float() && transfer.is_linear())
            }
            ColorSpace::Xyz { .. } => self.layout.sample.is_float(),
            ColorSpace::Icc { profile, .. } => {
                if self.layout.sample.is_float() {
                    return true;
                }
                #[cfg(feature = "icc")]
                {
                    if let Some(cicp) = crate::Cicp::from_icc(profile) {
                        return matches!(cicp.transfer_characteristics, 16 | 18);
                    }
                }
                let _ = profile;
                false
            }
            _ => false,
        }
    }

    /// Construct a validated [`ImageSpec`] by assuming a conventional default color space for `layout`.
    ///
    /// **Assumption, not signal detection:** chooses full-range sRGB for integer RGB/Gray layouts,
    /// linear sRGB/Gray for floating-point layouts, D65 for XYZ, straight/undefined alpha matching
    /// the layout's alpha/padding slot, and untagged otherwise. Prefer explicit color spaces
    /// whenever container metadata is available.
    pub fn with_default_color_for_layout(layout: SampleLayout) -> Result<Self, At<ColorError>> {
        layout
            .validate()
            .map_err(|e| whereat::at!(ColorError::Layout(e)))?;
        let order = layout.order;
        let alpha = if order.has_padding_slot() {
            AlphaMode::Undefined
        } else if order.has_alpha_slot() {
            AlphaMode::Straight
        } else {
            AlphaMode::None
        };
        let tf = if layout.sample.is_float() {
            Transfer::Linear
        } else {
            Transfer::Srgb
        };
        let color = if order.is_rgb_family() {
            ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: tf,
                range: SignalRange::Full,
                alpha,
                white: None,
            }
        } else if order.is_gray_family() {
            ColorSpace::Gray {
                transfer: tf,
                range: SignalRange::Full,
                alpha,
                white: None,
            }
        } else if order.is_xyz_family() {
            ColorSpace::Xyz {
                white_point: WhitePoint::D65,
                alpha,
                white: None,
            }
        } else if order.is_oklab_family() {
            ColorSpace::Oklab { alpha }
        } else if order.is_cmyk_family() {
            ColorSpace::Cmyk {
                inverted: matches!(
                    order,
                    ChannelOrder::CmykInverted | ChannelOrder::CmykaInverted
                ),
                alpha,
                icc: None,
            }
        } else if matches!(order, ChannelOrder::Perceptual3 | ChannelOrder::Perceptual4) {
            ColorSpace::Perceptual {
                space: PerceptualSpace::Oklch,
                alpha,
            }
        } else {
            ColorSpace::Untagged {
                channels: order.channels() as u8,
                alpha,
            }
        };
        let spec = Self::new(layout, color);
        spec.validate()?;
        Ok(spec)
    }

    /// Validate that `layout` and `color` agree on channels, alpha/padding, range, and sample types.
    pub fn validate(&self) -> Result<(), At<ColorError>> {
        self.layout
            .validate()
            .map_err(|e| whereat::at!(ColorError::Layout(e)))?;
        let order = self.layout.order;
        let sample = self.layout.sample;
        let alpha = self.color.alpha();

        // Alpha / padding consistency with layout slots.
        if order.has_padding_slot() {
            if alpha != AlphaMode::Undefined {
                return Err(whereat::at!(ColorError::AlphaLayoutMismatch));
            }
        } else if order.has_alpha_slot() {
            if matches!(alpha, AlphaMode::None | AlphaMode::Undefined) {
                return Err(whereat::at!(ColorError::AlphaLayoutMismatch));
            }
        } else if alpha != AlphaMode::None {
            return Err(whereat::at!(ColorError::AlphaLayoutMismatch));
        }

        match &self.color {
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                ..
            } => {
                if !order.is_rgb_family() {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
                if *range == SignalRange::Narrow
                    && !matches!(sample, SampleType::U8 | SampleType::U16)
                {
                    return Err(whereat::at!(ColorError::NarrowRangeRequiresInteger));
                }
                if matches!(transfer, Transfer::Hlg(HlgMode::Display(_)))
                    && (*primaries != Primaries::Bt2020 || *range != SignalRange::Full)
                {
                    return Err(whereat::at!(ColorError::HlgDisplayRequiresBt2020Full));
                }
            }
            ColorSpace::Gray {
                transfer, range, ..
            } => {
                if !order.is_gray_family() {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
                if *range == SignalRange::Narrow
                    && !matches!(sample, SampleType::U8 | SampleType::U16)
                {
                    return Err(whereat::at!(ColorError::NarrowRangeRequiresInteger));
                }
                if matches!(transfer, Transfer::Hlg(HlgMode::Display(_))) {
                    return Err(whereat::at!(ColorError::HlgDisplayRequiresBt2020Full));
                }
            }
            ColorSpace::Xyz { .. } => {
                if !order.is_xyz_family() || !matches!(sample, SampleType::F16 | SampleType::F32) {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
            }
            ColorSpace::Oklab { .. } => {
                if !order.is_oklab_family() || sample != SampleType::F32 {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
            }
            ColorSpace::Perceptual { .. } => {
                if !matches!(order, ChannelOrder::Perceptual3 | ChannelOrder::Perceptual4)
                    || sample != SampleType::F32
                {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
            }
            ColorSpace::Cmyk {
                inverted,
                alpha,
                icc,
            } => {
                let want = match (*inverted, *alpha) {
                    (false, AlphaMode::None) => ChannelOrder::Cmyk,
                    (true, AlphaMode::None) => ChannelOrder::CmykInverted,
                    (false, _) => ChannelOrder::Cmyka,
                    (true, _) => ChannelOrder::CmykaInverted,
                };
                if order != want {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
                if let Some(profile) = icc {
                    if validate_icc_structure(profile).is_err() {
                        return Err(whereat::at!(ColorError::InvalidIccProfile));
                    }
                    if &profile[16..20] != b"CMYK" {
                        return Err(whereat::at!(ColorError::InvalidIccProfile));
                    }
                }
            }
            ColorSpace::Icc {
                profile, channels, ..
            } => {
                if validate_icc_structure(profile).is_err() {
                    return Err(whereat::at!(ColorError::InvalidIccProfile));
                }
                let expected = (*channels as usize)
                    + usize::from(order.has_alpha_slot() || order.has_padding_slot());
                if order.channels() != expected {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
                match &profile[16..20] {
                    b"GRAY" => {
                        if *channels != 1
                            || !(order.is_gray_family()
                                || matches!(order, ChannelOrder::Generic(1 | 2)))
                        {
                            return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                        }
                    }
                    b"RGB " => {
                        if *channels != 3
                            || !(order.is_rgb_family()
                                || matches!(order, ChannelOrder::Generic(3 | 4)))
                        {
                            return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                        }
                    }
                    b"CMYK" => {
                        if *channels != 4
                            || !(order.is_cmyk_family()
                                || matches!(order, ChannelOrder::Generic(4 | 5)))
                        {
                            return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                        }
                    }
                    b"Lab " => {
                        if *channels != 3
                            || !matches!(
                                order,
                                ChannelOrder::Perceptual3
                                    | ChannelOrder::Perceptual4
                                    | ChannelOrder::Generic(3 | 4)
                            )
                        {
                            return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                        }
                    }
                    b"XYZ "
                        if *channels != 3
                            || !(order.is_xyz_family()
                                || matches!(order, ChannelOrder::Generic(3 | 4))) =>
                    {
                        return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                    }
                    _ => {}
                }
            }
            ColorSpace::Untagged { channels, .. } => {
                if order.channels() != *channels as usize {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
            }
        }
        Ok(())
    }

    /// Check whether this spec satisfies an [`InteropTarget`]'s zero-copy contract.
    #[must_use]
    pub fn supports_interop(&self, target: InteropTarget) -> bool {
        if self.validate().is_err() || !self.layout.packing.is_full() {
            return false;
        }
        let s = self.layout.sample;
        let o = self.layout.order;
        match target {
            InteropTarget::Tight => true,
            InteropTarget::Imgref => {
                s.is_standard_color()
                    && matches!(
                        o,
                        ChannelOrder::Gray
                            | ChannelOrder::GrayAlpha
                            | ChannelOrder::Rgb
                            | ChannelOrder::Rgba
                            | ChannelOrder::Bgr
                            | ChannelOrder::Bgra
                            | ChannelOrder::Rgbx
                            | ChannelOrder::Bgrx
                    )
                    && (!(o.has_padding_slot()) || s == SampleType::U8)
            }
            InteropTarget::FlatSamples => {
                s.is_standard_color()
                    && matches!(
                        o,
                        ChannelOrder::Gray
                            | ChannelOrder::GrayAlpha
                            | ChannelOrder::Rgb
                            | ChannelOrder::Rgba
                    )
            }
            InteropTarget::ImageBuffer => {
                let sample_ok = matches!(s, SampleType::U8 | SampleType::U16 | SampleType::F32);
                let order_ok = matches!(
                    o,
                    ChannelOrder::Gray
                        | ChannelOrder::GrayAlpha
                        | ChannelOrder::Rgb
                        | ChannelOrder::Rgba
                );
                let alpha_ok = matches!(
                    self.color.alpha(),
                    AlphaMode::None | AlphaMode::Straight | AlphaMode::Opaque
                );
                let color_ok = match &self.color {
                    ColorSpace::Rgb {
                        primaries,
                        transfer,
                        range,
                        white,
                        ..
                    } => {
                        *range == SignalRange::Full
                            && white.is_none()
                            && matches!(
                                primaries,
                                Primaries::Bt709 | Primaries::DisplayP3 | Primaries::Bt2020
                            )
                            && matches!(
                                transfer,
                                Transfer::Srgb
                                    | Transfer::Linear
                                    | Transfer::Bt709
                                    | Transfer::Pq
                                    | Transfer::Hlg(HlgMode::Scene)
                            )
                    }
                    ColorSpace::Gray {
                        transfer,
                        range,
                        white,
                        ..
                    } => {
                        *range == SignalRange::Full
                            && white.is_none()
                            && matches!(
                                transfer,
                                Transfer::Srgb
                                    | Transfer::Linear
                                    | Transfer::Bt709
                                    | Transfer::Pq
                                    | Transfer::Hlg(HlgMode::Scene)
                            )
                    }
                    _ => false,
                };
                sample_ok && order_ok && alpha_ok && color_ok
            }
            InteropTarget::DynamicImage => {
                self.supports_interop(InteropTarget::ImageBuffer)
                    && !(s == SampleType::F32
                        && matches!(o, ChannelOrder::Gray | ChannelOrder::GrayAlpha))
            }
        }
    }
}

/// Descriptive HDR container metadata (CLL / ST 2086 mastering display).
///
/// Purely informational — never silently selects a display or tone-mapping peak.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HdrMetadata {
    /// Content light level (`MaxCLL` / `MaxFALL`).
    pub cll: Option<ContentLightLevel>,
    /// Mastering display color volume (SMPTE ST 2086 / `mDCV`).
    pub mastering: Option<MasteringDisplay>,
}

impl HdrMetadata {
    /// Whether neither content light level nor mastering display metadata is present.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.cll.is_none() && self.mastering.is_none()
    }
}

/// How the original file signaled its color space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OriginKind {
    /// No explicit color signaling was present in the file.
    #[default]
    Assumed,
    /// Only a CICP / `nclx` / `cICP` chunk was present.
    Cicp,
    /// Only an ICC profile was present.
    Icc,
    /// Both an ICC profile and a CICP chunk were present in the container.
    IccAndCicp,
    /// Legacy PNG `gAMA` + `cHRM` chunks.
    GamaChrm,
}

/// Immutable provenance of what the original file claimed on disk.
///
/// Kept strictly separate from active [`ColorSpace`] so container provenance
/// can guide encoder re-embedding (`SameAsOrigin`) without ever overriding
/// converted pixel values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ColorOrigin {
    /// Kind of original file signaling.
    pub kind: OriginKind,
    /// Original raw ICC profile bytes, if present.
    pub icc: Option<Arc<[u8]>>,
    /// Original raw CICP tuple, if present.
    pub cicp: Option<Cicp>,
}

impl ColorOrigin {
    /// No explicit color metadata in the source file.
    #[inline]
    #[must_use]
    pub const fn assumed() -> Self {
        Self {
            kind: OriginKind::Assumed,
            icc: None,
            cicp: None,
        }
    }

    /// Source carried only CICP.
    #[inline]
    #[must_use]
    pub const fn from_cicp(cicp: Cicp) -> Self {
        Self {
            kind: OriginKind::Cicp,
            icc: None,
            cicp: Some(cicp),
        }
    }

    /// Source carried only ICC.
    #[inline]
    #[must_use]
    pub fn from_icc(icc: Arc<[u8]>) -> Self {
        Self {
            kind: OriginKind::Icc,
            icc: Some(icc),
            cicp: None,
        }
    }

    /// Source carried both ICC and CICP chunks.
    #[inline]
    #[must_use]
    pub fn from_icc_and_cicp(icc: Arc<[u8]>, cicp: Cicp) -> Self {
        Self {
            kind: OriginKind::IccAndCicp,
            icc: Some(icc),
            cicp: Some(cicp),
        }
    }

    #[inline]
    pub(crate) const fn is_assumed(&self) -> bool {
        matches!(self.kind, OriginKind::Assumed) && self.icc.is_none() && self.cicp.is_none()
    }
}

/// Exact unreduced signed rational (`i32 / NonZeroU32`) per ISO 21496-1 wire format.
///
/// Preserves exact numerator and denominator pairs across lossless transcodes
/// (e.g. `1/64` vs `2/128`) while supporting exact cross-product comparison and
/// `f32`/`f64` projection for pixel math.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SignedFraction {
    /// Signed numerator.
    pub num: i32,
    /// Strictly positive denominator.
    pub den: NonZeroU32,
}

impl SignedFraction {
    /// Exact `0 / 1`.
    pub const ZERO: Self = Self {
        num: 0,
        den: NonZeroU32::MIN,
    };

    /// Standard ISO 21496-1 / UltraHDR `1 / 64` offset.
    pub const ONE_64TH: Self = Self {
        num: 1,
        den: match NonZeroU32::new(64) {
            Some(d) => d,
            None => unreachable!(),
        },
    };

    /// Construct from `(num, den: NonZeroU32)`.
    #[inline]
    #[must_use]
    pub const fn new(num: i32, den: NonZeroU32) -> Self {
        Self { num, den }
    }

    /// Construct from `(num, den: u32)`, returning `None` if `den == 0`.
    #[inline]
    #[must_use]
    pub const fn try_new(num: i32, den: u32) -> Option<Self> {
        match NonZeroU32::new(den) {
            Some(d) => Some(Self { num, den: d }),
            None => None,
        }
    }

    /// Exact integer `n / 1`.
    #[inline]
    #[must_use]
    pub const fn from_int(num: i32) -> Self {
        Self {
            num,
            den: NonZeroU32::MIN,
        }
    }

    /// Approximate a finite `f32` as a rational (using denominator `1_000_000` or exact `64` for `1/64` multiples).
    #[must_use]
    pub fn from_f32_approx(v: f32) -> Option<Self> {
        if !v.is_finite() {
            return None;
        }
        let s64 = (v as f64) * 64.0;
        let r64 = if s64 >= 0.0 {
            (s64 + 0.5) as i64
        } else {
            (s64 - 0.5) as i64
        };
        if (s64 - (r64 as f64)).abs() < 1e-7 && r64 >= i32::MIN as i64 && r64 <= i32::MAX as i64 {
            if r64 % 64 == 0 {
                return Some(Self::from_int((r64 / 64) as i32));
            }
            return Self::try_new(r64 as i32, 64);
        }
        const SCALE: f64 = 1_000_000.0;
        let scaled = (v as f64) * SCALE;
        if scaled < i32::MIN as f64 || scaled > i32::MAX as f64 {
            return None;
        }
        let n = if scaled >= 0.0 {
            (scaled + 0.5) as i32
        } else {
            (scaled - 0.5) as i32
        };
        Self::try_new(n, 1_000_000)
    }

    /// Exact rational value as `f64`.
    #[inline]
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        (self.num as f64) / (self.den.get() as f64)
    }

    /// Rational value as `f32`.
    #[inline]
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Exact rational comparison (`a.num * b.den` vs `b.num * a.den` in `i128`).
    #[inline]
    #[must_use]
    pub const fn cmp_rational(self, other: Self) -> Ordering {
        let lhs = (self.num as i128) * (other.den.get() as i128);
        let rhs = (other.num as i128) * (self.den.get() as i128);
        if lhs < rhs {
            Ordering::Less
        } else if lhs > rhs {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

/// Exact unreduced unsigned rational (`u32 / NonZeroU32`) per ISO 21496-1 wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnsignedFraction {
    /// Unsigned numerator.
    pub num: u32,
    /// Strictly positive denominator.
    pub den: NonZeroU32,
}

impl UnsignedFraction {
    /// Exact `0 / 1`.
    pub const ZERO: Self = Self {
        num: 0,
        den: NonZeroU32::MIN,
    };

    /// Exact `1 / 1`.
    pub const ONE: Self = Self {
        num: 1,
        den: NonZeroU32::MIN,
    };

    /// Construct from `(num, den: NonZeroU32)`.
    #[inline]
    #[must_use]
    pub const fn new(num: u32, den: NonZeroU32) -> Self {
        Self { num, den }
    }

    /// Construct from `(num, den: u32)`, returning `None` if `den == 0`.
    #[inline]
    #[must_use]
    pub const fn try_new(num: u32, den: u32) -> Option<Self> {
        match NonZeroU32::new(den) {
            Some(d) => Some(Self { num, den: d }),
            None => None,
        }
    }

    /// Exact unsigned integer `n / 1`.
    #[inline]
    #[must_use]
    pub const fn from_uint(num: u32) -> Self {
        Self {
            num,
            den: NonZeroU32::MIN,
        }
    }

    /// Approximate a finite non-negative `f32` as an unsigned rational.
    #[must_use]
    pub fn from_f32_approx(v: f32) -> Option<Self> {
        if !v.is_finite() || v < 0.0 {
            return None;
        }
        let s64 = (v as f64) * 64.0;
        let r64 = (s64 + 0.5) as u64;
        if (s64 - (r64 as f64)).abs() < 1e-7 && r64 <= u32::MAX as u64 {
            if r64 % 64 == 0 {
                return Some(Self::from_uint((r64 / 64) as u32));
            }
            return Self::try_new(r64 as u32, 64);
        }
        const SCALE: f64 = 1_000_000.0;
        let scaled = (v as f64) * SCALE;
        if scaled > u32::MAX as f64 {
            return None;
        }
        Self::try_new((scaled + 0.5) as u32, 1_000_000)
    }

    /// Exact rational value as `f64`.
    #[inline]
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        (self.num as f64) / (self.den.get() as f64)
    }

    /// Rational value as `f32`.
    #[inline]
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Exact rational comparison (`a.num * b.den` vs `b.num * a.den` in `u64`).
    #[inline]
    #[must_use]
    pub const fn cmp_rational(self, other: Self) -> Ordering {
        let lhs = (self.num as u64) * (other.den.get() as u64);
        let rhs = (other.num as u64) * (self.den.get() as u64);
        if lhs < rhs {
            Ordering::Less
        } else if lhs > rhs {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

/// Per-channel ISO 21496-1 gain-map parameters in exact rational form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GainMapComponent {
    /// Minimum log₂ gain (often `0/1` or negative).
    pub min_gain_stops: SignedFraction,
    /// Maximum log₂ gain (e.g. `2.3` stops for ~5× / 1000-nit headroom).
    pub max_gain_stops: SignedFraction,
    /// Gain-map encoding gamma (strictly positive; default `1/1`).
    pub gamma: UnsignedFraction,
    /// Linear offset added to base rendition before applying gain (may be negative per ISO 21496-1; typical `1/64`).
    pub base_offset: SignedFraction,
    /// Linear offset subtracted from alternate rendition (may be negative per ISO 21496-1; typical `1/64`).
    pub alternate_offset: SignedFraction,
}

impl GainMapComponent {
    /// Validate `gamma > 0` and `min_gain_stops <= max_gain_stops` using exact rational math.
    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.gamma.num > 0
            && !matches!(
                self.min_gain_stops.cmp_rational(self.max_gain_stops),
                Ordering::Greater
            )
    }
}

/// Metadata record arity per ISO 21496-1 §5.2.5.1 — strictly independent of gain-map image channels (1 or 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GainMapChannels {
    /// Single shared parameter record (`is_multichannel = false`).
    Shared(GainMapComponent),
    /// Three explicit per-channel `[R, G, B]` parameter records (`is_multichannel = true`).
    PerChannel([GainMapComponent; 3]),
}

impl GainMapChannels {
    /// Access parameter record for channel `c` (`0..3`).
    #[inline]
    #[must_use]
    pub const fn channel(self, c: usize) -> GainMapComponent {
        match self {
            Self::Shared(comp) => comp,
            Self::PerChannel(arr) => arr[c],
        }
    }

    /// Whether three explicit per-channel records were authored (`is_multichannel`).
    #[inline]
    #[must_use]
    pub const fn is_multichannel(self) -> bool {
        matches!(self, Self::PerChannel(_))
    }

    /// Validate all component records.
    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        match self {
            Self::Shared(c) => c.is_valid(),
            Self::PerChannel([r, g, b]) => r.is_valid() && g.is_valid() && b.is_valid(),
        }
    }
}

/// Which rendition's linear RGB primaries host channelwise gain application (ISO 21496-1 `use_base_colour_space`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GainMapColorSpace {
    /// Apply channelwise gain in the base rendition's linear RGB primaries (`use_base_colour_space = true`).
    #[default]
    Base,
    /// Convert base to the alternate rendition's linear RGB primaries before applying gain (`use_base_colour_space = false`).
    Alternate,
}

/// Rendition headroom ordering derived from exact rational comparison of `base_hdr_headroom` vs `alternate_hdr_headroom`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GainMapDirection {
    /// SDR (or lower-headroom) base → HDR (higher-headroom) alternate (`H_alt > H_base`, positive weight sign).
    Forward,
    /// HDR (higher-headroom) base → SDR (lower-headroom) alternate (`H_base > H_alt`, negative weight sign).
    Reverse,
}

/// ISO 21496-1 / UltraHDR gain-map reconstruction parameters in exact rational form.
///
/// Preserves exact unreduced rational pairs for lossless metadata transcoding across
/// JPEG APP2, AVIF `tmap`, and JXL `jhgm` while projecting cleanly to float for rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct GainMapMetadata {
    /// Shared (1-record) or per-channel (3-record) gain curve parameters.
    pub channels: GainMapChannels,
    /// Minimum display HDR headroom (log₂ stops) where the base rendition is shown unchanged.
    pub base_hdr_headroom: UnsignedFraction,
    /// Display HDR headroom (log₂ stops) at which full alternate rendition is reached.
    pub alternate_hdr_headroom: UnsignedFraction,
    /// Whether channelwise gain runs in base or alternate linear RGB primaries.
    pub application_space: GainMapColorSpace,
    /// Optional explicit ISO 21496-1 `backward_direction` flag override (`None` derives
    /// direction from comparing `base_hdr_headroom` vs `alternate_hdr_headroom`).
    pub backward_direction: Option<bool>,
}

impl GainMapMetadata {
    /// Construct and validate an exact ISO 21496-1 parameter set.
    ///
    /// Rejects non-positive `gamma`, `min > max`, or equal `base_hdr_headroom == alternate_hdr_headroom`
    /// (ISO 21496-1 §5.2.7 requires unequal headrooms).
    #[must_use]
    pub const fn new(
        channels: GainMapChannels,
        base_hdr_headroom: UnsignedFraction,
        alternate_hdr_headroom: UnsignedFraction,
        application_space: GainMapColorSpace,
    ) -> Option<Self> {
        let m = Self {
            channels,
            base_hdr_headroom,
            alternate_hdr_headroom,
            application_space,
            backward_direction: None,
        };
        if m.is_valid() { Some(m) } else { None }
    }

    /// Set an explicit ISO 21496-1 `backward_direction` flag (`true` = reverse, `false` = forward).
    #[inline]
    #[must_use]
    pub const fn with_backward_direction(mut self, backward: bool) -> Self {
        self.backward_direction = Some(backward);
        self
    }

    /// Fallibly construct a symmetric forward SDR-base → HDR-alternate parameter set with
    /// standard `1/64` offsets.
    ///
    /// Returns `None` if `max_boost_stops` is non-finite, `<= 0.0`, or rounds to zero at
    /// `1/1_000_000` precision.
    #[must_use]
    pub fn try_symmetric(max_boost_stops: f32) -> Option<Self> {
        if !max_boost_stops.is_finite() || max_boost_stops <= 0.0 {
            return None;
        }
        let max_f = SignedFraction::from_f32_approx(max_boost_stops)?;
        let alt_h = UnsignedFraction::from_f32_approx(max_boost_stops)?;
        if max_f.num <= 0 || alt_h.num == 0 {
            return None;
        }
        Self::new(
            GainMapChannels::Shared(GainMapComponent {
                min_gain_stops: SignedFraction::ZERO,
                max_gain_stops: max_f,
                gamma: UnsignedFraction::ONE,
                base_offset: SignedFraction::ONE_64TH,
                alternate_offset: SignedFraction::ONE_64TH,
            }),
            UnsignedFraction::ZERO,
            alt_h,
            GainMapColorSpace::Base,
        )
    }

    /// Construct a symmetric forward SDR-base → HDR-alternate parameter set with standard `1/64` offsets.
    ///
    /// # Panics
    ///
    /// Panics if `max_boost_stops` is not finite and strictly positive (use [`try_symmetric`](Self::try_symmetric)
    /// for untrusted inputs).
    #[must_use]
    pub fn symmetric(max_boost_stops: f32) -> Self {
        Self::try_symmetric(max_boost_stops)
            .expect("GainMapMetadata::symmetric requires finite max_boost_stops > 0.0")
    }

    /// Validate all exact rational invariants (`gamma > 0`, `min <= max`, `H_base != H_alt`,
    /// and any explicit [`backward_direction`](Self::backward_direction) flag agrees with the
    /// headroom ordering).
    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        let ordering_ok = !matches!(
            (
                self.base_hdr_headroom
                    .cmp_rational(self.alternate_hdr_headroom),
                self.backward_direction,
            ),
            (Ordering::Equal, _) | (Ordering::Less, Some(true)) | (Ordering::Greater, Some(false))
        );
        self.channels.is_valid() && ordering_ok
    }

    /// Exact gain-map direction (`Forward` if `H_alt > H_base`, `Reverse` if `H_base > H_alt`),
    /// returning `None` if headrooms are equal or an explicit [`backward_direction`](Self::backward_direction)
    /// flag contradicts the headroom ordering.
    #[inline]
    #[must_use]
    pub const fn direction(self) -> Option<GainMapDirection> {
        match (
            self.base_hdr_headroom
                .cmp_rational(self.alternate_hdr_headroom),
            self.backward_direction,
        ) {
            (Ordering::Less, None | Some(false)) => Some(GainMapDirection::Forward),
            (Ordering::Greater, None | Some(true)) => Some(GainMapDirection::Reverse),
            _ => None,
        }
    }

    /// Access parameter record for channel `c` (`0..3`).
    #[inline]
    #[must_use]
    pub const fn channel(self, c: usize) -> GainMapComponent {
        self.channels.channel(c)
    }
}

/// Single-pixel color value used for explicit alpha matting and fills.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Linear-light RGB in `primaries` (`1.0` = SDR diffuse white).
    pub rgb_linear: [f32; 3],
    /// Primaries of `rgb_linear`.
    pub primaries: Primaries,
}

impl Color {
    /// Opaque sRGB black `[0, 0, 0]`.
    pub const BLACK: Self = Self {
        rgb_linear: [0.0, 0.0, 0.0],
        primaries: Primaries::Bt709,
    };

    /// Opaque sRGB diffuse white `[1, 1, 1]`.
    pub const WHITE: Self = Self {
        rgb_linear: [1.0, 1.0, 1.0],
        primaries: Primaries::Bt709,
    };

    /// Construct from linear RGB values in `primaries`.
    #[inline]
    #[must_use]
    pub const fn linear(rgb: [f32; 3], primaries: Primaries) -> Self {
        Self {
            rgb_linear: rgb,
            primaries,
        }
    }

    /// Convert this color's linear RGB into `dst` primaries.
    #[inline]
    #[must_use]
    pub fn to_primaries(self, dst: Primaries) -> Option<[f32; 3]> {
        let m = self.primaries.gamut_matrix_to(dst)?;
        let [r, g, b] = self.rgb_linear;
        Some([
            m[0][0] * r + m[0][1] * g + m[0][2] * b,
            m[1][0] * r + m[1][1] * g + m[1][2] * b,
            m[2][0] * r + m[2][1] * g + m[2][2] * b,
        ])
    }
}

/// Validation errors for [`ImageSpec`] and [`ColorSpace`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorError {
    /// Underlying sample layout error.
    Layout(LayoutError),
    /// Alpha mode disagrees with whether the layout has an alpha or padding channel.
    AlphaLayoutMismatch,
    /// Color space variant disagrees with the layout's channel family or sample type.
    ColorLayoutMismatch,
    /// Narrow studio range is only valid on `U8` and `U16` integer samples.
    NarrowRangeRequiresInteger,
    /// Display-rendered HLG requires full-range BT.2020 RGB/RGBA.
    HlgDisplayRequiresBt2020Full,
    /// Attached ICC profile is empty, has a disallowed profile class (DeviceLink/Abstract/NamedColor),
    /// or its header data color space disagrees with the declared color model.
    InvalidIccProfile,
}

impl fmt::Display for ColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(e) => e.fmt(f),
            Self::AlphaLayoutMismatch => {
                f.write_str("alpha mode does not match layout's alpha/padding slot")
            }
            Self::ColorLayoutMismatch => {
                f.write_str("color space is incompatible with sample layout")
            }
            Self::NarrowRangeRequiresInteger => {
                f.write_str("narrow signal range requires U8 or U16 integer samples")
            }
            Self::HlgDisplayRequiresBt2020Full => {
                f.write_str("HLG display rendering requires full-range BT.2020 RGB/RGBA")
            }
            Self::InvalidIccProfile => {
                f.write_str("ICC profile is empty, unsupported class, or mismatches color space")
            }
        }
    }
}

impl core::error::Error for ColorError {}
