//! Color profile types for CMS integration.
//!
//! Provides a unified way to reference the source color space of decoded
//! pixels, suitable for passing to a CMS backend (e.g., moxcms, lcms2).
//!
//! [`ColorContext`] bundles ICC and CICP metadata for cheap sharing
//! (via `Arc`) across pixel slices and pipeline stages.
//! Current color state (transfer, primaries, alpha) is tracked on the
//! pixel descriptor itself, not as a separate enum.

use crate::PixelDescriptor;
use crate::cicp::Cicp;
use crate::hdr::DiffuseWhite;
use crate::{ColorPrimaries, TransferFunction};
use alloc::sync::Arc;

/// CIE 1931 `(x, y)` chromaticity coordinates with finite `f32` components.
///
/// Negative coordinates are permitted because wide/virtual gamuts such as ACES
/// AP0 (`blue.y = -0.077`) and JPEG XL custom chromaticities (`[-2, 2]`) use
/// imaginary primaries outside the spectral locus. NaN and infinities are
/// rejected at construction, and `-0.0` is canonicalized to `+0.0` so bitwise
/// [`Eq`] and [`Hash`](core::hash::Hash) are reflexive and consistent.
#[derive(Clone, Copy, Debug)]
pub struct CieXy {
    x: f32,
    y: f32,
}

impl PartialEq for CieXy {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x.to_bits() == other.x.to_bits() && self.y.to_bits() == other.y.to_bits()
    }
}

impl Eq for CieXy {}

impl core::hash::Hash for CieXy {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

impl CieXy {
    /// CIE Standard Illuminant D65 (`(0.3127, 0.3290)`).
    pub const D65: Self = Self::new(0.3127, 0.3290);
    /// CIE Standard Illuminant D50 (`(0.34567, 0.35850)`), ICC PCS white.
    pub const D50: Self = Self::new(0.34567, 0.35850);
    /// SMPTE ST 2065-1 ACES reference white D60 (`(0.32168, 0.33767)`).
    pub const D60: Self = Self::new(0.32168, 0.33767);
    /// CIE Standard Illuminant C (`(0.310, 0.316)`), ITU-T H.273 Table 2 (CICP 4 / 8).
    pub const C: Self = Self::new(0.310, 0.316);
    /// Equal-energy illuminant E (`(1/3, 1/3)`).
    pub const E: Self = Self::new(1.0 / 3.0, 1.0 / 3.0);
    /// SMPTE RP 431-2 theatrical DCI white (`(0.314, 0.351)`).
    pub const DCI: Self = Self::new(0.314, 0.351);

    /// Construct `(x, y)` chromaticity coordinates, returning `None` if either
    /// coordinate is NaN or infinite.
    #[inline]
    #[must_use]
    pub const fn try_new(x: f32, y: f32) -> Option<Self> {
        if x.is_finite() && y.is_finite() {
            let x = if x == 0.0 { 0.0 } else { x };
            let y = if y == 0.0 { 0.0 } else { y };
            Some(Self { x, y })
        } else {
            None
        }
    }

    /// Construct `(x, y)` chromaticity coordinates.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is NaN or infinite.
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        match Self::try_new(x, y) {
            Some(xy) => xy,
            None => panic!("CIE xy chromaticity coordinates must be finite"),
        }
    }

    /// CIE 1931 `x` chromaticity coordinate.
    #[inline]
    #[must_use]
    pub const fn x(self) -> f32 {
        self.x
    }

    /// CIE 1931 `y` chromaticity coordinate.
    #[inline]
    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }

    /// `(x, y)` chromaticity pair.
    #[inline]
    #[must_use]
    pub const fn to_tuple(self) -> (f32, f32) {
        (self.x, self.y)
    }
}

/// Reference white point for [`ParametricColor`].
///
/// Named variants preserve compact header enum tags (such as JPEG XL's
/// `WhitePoint::D65`, `E`, and `Dci`) instead of flattening them into
/// approximate floats, while [`Self::Custom`] carries arbitrary CIE 1931 `(x, y)`
/// coordinates (PNG `cHRM`, JPEG XL `WhitePoint::Custom`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WhitePoint {
    /// CIE Standard Illuminant D65 (`(0.3127, 0.3290)`).
    #[default]
    D65,
    /// CIE Standard Illuminant D50 (`(0.34567, 0.35850)`), ICC PCS white.
    D50,
    /// SMPTE ST 2065-1 ACES reference white D60 (`(0.32168, 0.33767)`).
    D60,
    /// CIE Standard Illuminant C (`(0.310, 0.316)`).
    C,
    /// Equal-energy illuminant E (`(1/3, 1/3)`).
    E,
    /// SMPTE RP 431-2 theatrical DCI white (`(0.314, 0.351)`).
    Dci,
    /// Arbitrary CIE 1931 `(x, y)` white point.
    Custom(CieXy),
}

impl WhitePoint {
    /// CIE 1931 `(x, y)` coordinates of this white point.
    #[inline]
    #[must_use]
    pub const fn cie_xy(self) -> CieXy {
        match self {
            Self::D65 => CieXy::D65,
            Self::D50 => CieXy::D50,
            Self::D60 => CieXy::D60,
            Self::C => CieXy::C,
            Self::E => CieXy::E,
            Self::Dci => CieXy::DCI,
            Self::Custom(xy) => xy,
        }
    }
}

/// RGB primary chromaticities for [`ParametricColor`], independent of white point.
///
/// Unlike [`ColorPrimaries`] (which bundles RGB primaries with D65 on
/// [`PixelDescriptor`]), `ParametricPrimaries` separates the RGB triangle from
/// [`WhitePoint`] so combinations like `P3` + [`WhitePoint::Dci`] (theatrical
/// DCI-P3) and `P3` + [`WhitePoint::D65`] (Display P3), as well as custom `(x, y)`
/// primaries from JPEG XL headers or PNG `cHRM`, round-trip losslessly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParametricPrimaries {
    /// ITU-R BT.709 / sRGB RGB primaries.
    Bt709,
    /// ITU-R BT.2020 / BT.2100 RGB primaries.
    Bt2020,
    /// SMPTE RP 431-2 / EG 432-1 P3 RGB primaries.
    P3,
    /// Adobe RGB (1998) RGB primaries.
    AdobeRgb,
    /// ISO 22028-2 ROMM RGB / ProPhoto RGB primaries (nominally paired with [`WhitePoint::D50`]).
    ProPhotoRgb,
    /// ACEScg (AP1) working-space primaries (nominally paired with [`WhitePoint::D60`]).
    AcesCg,
    /// SMPTE ST 2065-1 ACES archival (AP0) primaries (nominally paired with [`WhitePoint::D60`]).
    Aces2065_1,
    /// ITU-R BT.601-7 525-line / SMPTE 170M NTSC primaries (CICP 6, paired with [`WhitePoint::D65`]).
    Bt601_525,
    /// ITU-R BT.601-7 625-line / BT.470 System B, G PAL/SECAM primaries (CICP 5, paired with [`WhitePoint::D65`]).
    Bt601_625,
    /// Arbitrary CIE 1931 `(x, y)` red, green, and blue primaries.
    Custom {
        /// Red primary chromaticity.
        red: CieXy,
        /// Green primary chromaticity.
        green: CieXy,
        /// Blue primary chromaticity.
        blue: CieXy,
    },
}

impl ParametricPrimaries {
    /// `(red, green, blue)` CIE 1931 chromaticity coordinates.
    #[inline]
    #[must_use]
    pub const fn cie_xy(self) -> (CieXy, CieXy, CieXy) {
        match self {
            Self::Bt709 => (
                CieXy::new(0.64, 0.33),
                CieXy::new(0.30, 0.60),
                CieXy::new(0.15, 0.06),
            ),
            Self::Bt2020 => (
                CieXy::new(0.708, 0.292),
                CieXy::new(0.170, 0.797),
                CieXy::new(0.131, 0.046),
            ),
            Self::P3 => (
                CieXy::new(0.680, 0.320),
                CieXy::new(0.265, 0.690),
                CieXy::new(0.150, 0.060),
            ),
            Self::AdobeRgb => (
                CieXy::new(0.64, 0.33),
                CieXy::new(0.21, 0.71),
                CieXy::new(0.15, 0.06),
            ),
            Self::ProPhotoRgb => (
                CieXy::new(0.734699, 0.265301),
                CieXy::new(0.159597, 0.840403),
                CieXy::new(0.036598, 0.000105),
            ),
            Self::AcesCg => (
                CieXy::new(0.713, 0.293),
                CieXy::new(0.165, 0.830),
                CieXy::new(0.128, 0.044),
            ),
            Self::Aces2065_1 => (
                CieXy::new(0.7347, 0.2653),
                CieXy::new(0.0000, 1.0000),
                CieXy::new(0.0001, -0.0770),
            ),
            Self::Bt601_525 => (
                CieXy::new(0.630, 0.340),
                CieXy::new(0.310, 0.595),
                CieXy::new(0.155, 0.070),
            ),
            Self::Bt601_625 => (
                CieXy::new(0.64, 0.33),
                CieXy::new(0.29, 0.60),
                CieXy::new(0.15, 0.06),
            ),
            Self::Custom { red, green, blue } => (red, green, blue),
        }
    }
}

/// Pure power-law transfer exponent with explicit OETF vs EOTF direction.
///
/// Image formats disagree on whether "gamma" names the encoding exponent `g`
/// (`encoded = linear^g`, ~`0.45455`, used by JPEG XL `TransferFunction::Gamma`
/// and PNG `gAMA`) or the decoding/display EOTF exponent `γ`
/// (`linear = encoded^γ`, ~`2.2`, used by ICC `curv` tags and CSS/display specs).
/// `GammaCurve` has no bare-`f32` constructor so callers must state the
/// convention explicitly at the call site.
#[derive(Clone, Copy, Debug)]
pub struct GammaCurve(f32);

impl PartialEq for GammaCurve {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for GammaCurve {}

impl core::hash::Hash for GammaCurve {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl GammaCurve {
    /// Adobe RGB (1998) power law (`encoded = linear^(256/563)`, `linear = encoded^(563/256)`).
    pub const ADOBE_RGB: Self = Self::from_encoding_exponent(256.0 / 563.0);
    /// ITU-R BT.470M 2.2 display power law (`encoded = linear^(1/2.2)`, `linear = encoded^2.2`).
    pub const GAMMA_22: Self = Self::from_eotf_exponent(2.2);
    /// SMPTE ST 428-1 2.6 theatrical power law (`encoded = linear^(1/2.6)`, `linear = encoded^2.6`).
    pub const DCI: Self = Self::from_eotf_exponent(2.6);

    /// Construct from the encoding (OETF) exponent `g` where `encoded = linear^g`
    /// (JPEG XL `gamma` and PNG `gAMA / 100_000`), returning `None` unless `g` is
    /// finite and strictly positive.
    #[inline]
    #[must_use]
    pub const fn try_from_encoding_exponent(g: f32) -> Option<Self> {
        if g.is_finite() && g > 0.0 {
            Some(Self(g))
        } else {
            None
        }
    }

    /// Construct from the encoding (OETF) exponent `g` where `encoded = linear^g`.
    ///
    /// # Panics
    ///
    /// Panics if `g` is zero, negative, NaN, or infinite.
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn from_encoding_exponent(g: f32) -> Self {
        match Self::try_from_encoding_exponent(g) {
            Some(curve) => curve,
            None => panic!("gamma encoding exponent must be finite and strictly positive"),
        }
    }

    /// Construct from the decoding (EOTF) exponent `gamma` where `linear = encoded^gamma`
    /// (e.g. `2.2`), returning `None` unless both `gamma` and `1.0 / gamma` are
    /// finite and strictly positive.
    #[inline]
    #[must_use]
    pub const fn try_from_eotf_exponent(gamma: f32) -> Option<Self> {
        if gamma.is_finite() && gamma > 0.0 {
            Self::try_from_encoding_exponent(1.0 / gamma)
        } else {
            None
        }
    }

    /// Construct from the decoding (EOTF) exponent `gamma` where `linear = encoded^gamma`.
    ///
    /// # Panics
    ///
    /// Panics if `gamma` or `1.0 / gamma` is zero, negative, NaN, or infinite.
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn from_eotf_exponent(gamma: f32) -> Self {
        match Self::try_from_eotf_exponent(gamma) {
            Some(curve) => curve,
            None => panic!("gamma EOTF exponent must be finite and strictly positive"),
        }
    }

    /// Encoding (OETF) exponent `g` in `encoded = linear^g` (for example ~`0.45455`
    /// for 2.2 display gamma). Matches JPEG XL `TransferFunction::Gamma` and PNG `gAMA`.
    #[inline]
    #[must_use]
    pub const fn encoding_exponent(self) -> f32 {
        self.0
    }

    /// Decoding (EOTF) exponent `1 / g` in `linear = encoded^(1/g)` (for example
    /// ~`2.2` for standard display gamma).
    #[inline]
    #[must_use]
    pub const fn eotf_exponent(self) -> f32 {
        1.0 / self.0
    }
}

/// ICC.1 §10.18 `parametricCurveType` (`para`) EOTF transfer curve (`encoded -> linear`).
///
/// Represents the five standard ICC parametric curve forms (function types `0..=4`):
/// - **Type 0** (`[g]`): `Y = X^g` (`X >= 0`, else `0`)
/// - **Type 1** (`[g, a, b]`, CIE 122-1966): `Y = (aX + b)^g` (`X >= -b/a`, else `0`)
/// - **Type 2** (`[g, a, b, c]`, IEC 61966-3): `Y = (aX + b)^g + c` (`X >= -b/a`, else `c`)
/// - **Type 3** (`[g, a, b, c, d]`, IEC 61966-2-1): `Y = (aX + b)^g` (`X >= d`), `Y = cX` (`X < d`)
/// - **Type 4** (`[g, a, b, c, d, e, f]`): `Y = (aX + b)^g + e` (`X >= d`), `Y = cX + f` (`X < d`)
///
/// All active parameters are validated to be finite with `g > 0.0` (and `a != 0.0`
/// for types `1` and `2`, which divide by `a`). `-0.0` is canonicalized to `+0.0`
/// and unused parameter slots are zeroed so bitwise [`Eq`] and [`Hash`](core::hash::Hash)
/// are reflexive and canonical.
#[derive(Clone, Copy, Debug)]
pub struct IccParametricCurve {
    func_type: u8,
    params: [f32; 7],
}

impl PartialEq for IccParametricCurve {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.func_type == other.func_type
            && self.params[0].to_bits() == other.params[0].to_bits()
            && self.params[1].to_bits() == other.params[1].to_bits()
            && self.params[2].to_bits() == other.params[2].to_bits()
            && self.params[3].to_bits() == other.params[3].to_bits()
            && self.params[4].to_bits() == other.params[4].to_bits()
            && self.params[5].to_bits() == other.params[5].to_bits()
            && self.params[6].to_bits() == other.params[6].to_bits()
    }
}

impl Eq for IccParametricCurve {}

impl core::hash::Hash for IccParametricCurve {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.func_type.hash(state);
        for p in self.params {
            p.to_bits().hash(state);
        }
    }
}

impl IccParametricCurve {
    /// IEC 61966-2-1 sRGB curve (`funcType = 3`: `g = 2.4, a = 1/1.055, b = 0.055/1.055, c = 1/12.92, d = 0.04045`).
    pub const SRGB: Self =
        Self::type3_iec61966_2_1(2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045);
    /// ITU-R BT.709 OETF inverse (`funcType = 3`: `g = 1/0.45, a = 1/1.099, b = 0.099/1.099, c = 1/4.5, d = 0.081`).
    pub const BT709: Self =
        Self::type3_iec61966_2_1(1.0 / 0.45, 1.0 / 1.099, 0.099 / 1.099, 1.0 / 4.5, 0.081);
    /// ISO 22028-2:2013 §6.3 ROMM RGB / ProPhoto RGB curve
    /// (`funcType = 3`: `g = 1.8, a = 1.0, b = 0.0, c = 1/16, d = 1/32`).
    pub const PROPHOTO_RGB: Self = Self::type3_iec61966_2_1(1.8, 1.0, 0.0, 1.0 / 16.0, 1.0 / 32.0);

    #[inline]
    const fn canon(x: f32) -> f32 {
        if x == 0.0 { 0.0 } else { x }
    }

    /// Expected ICC parameter count for `func_type` (`0..=4`), or `None` for reserved types.
    #[inline]
    #[must_use]
    pub const fn param_count(func_type: u16) -> Option<usize> {
        match func_type {
            0 => Some(1),
            1 => Some(3),
            2 => Some(4),
            3 => Some(5),
            4 => Some(7),
            _ => None,
        }
    }

    /// Construct from an ICC `parametricCurveType` function code (`0..=4`) and parameter slice.
    ///
    /// Returns `None` if `func_type > 4`, `params.len()` does not match [`Self::param_count`],
    /// any parameter is NaN/infinite, `g <= 0.0`, or `a == 0.0` for function types `1` or `2`.
    #[must_use]
    pub const fn try_from_icc_para(func_type: u16, params: &[f32]) -> Option<Self> {
        match (func_type, params) {
            (0, &[g]) => Self::try_type0_gamma(g),
            (1, &[g, a, b]) => Self::try_type1_cie122(g, a, b),
            (2, &[g, a, b, c]) => Self::try_type2_iec61966_3(g, a, b, c),
            (3, &[g, a, b, c, d]) => Self::try_type3_iec61966_2_1(g, a, b, c, d),
            (4, &[g, a, b, c, d, e, f]) => Self::try_type4(g, a, b, c, d, e, f),
            _ => None,
        }
    }

    /// ICC function type `0`: `Y = X^g` (`g > 0`).
    #[inline]
    #[must_use]
    pub const fn try_type0_gamma(g: f32) -> Option<Self> {
        if g.is_finite() && g > 0.0 {
            Some(Self {
                func_type: 0,
                params: [Self::canon(g), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            })
        } else {
            None
        }
    }

    /// ICC function type `0`: `Y = X^g`, panicking if `g` is invalid.
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn type0_gamma(g: f32) -> Self {
        match Self::try_type0_gamma(g) {
            Some(c) => c,
            None => panic!("ICC parametric type 0 requires finite g > 0"),
        }
    }

    /// ICC function type `1` (CIE 122-1966): `Y = (aX + b)^g` for `X >= -b/a`, else `0`.
    #[inline]
    #[must_use]
    pub const fn try_type1_cie122(g: f32, a: f32, b: f32) -> Option<Self> {
        if g.is_finite() && g > 0.0 && a.is_finite() && a != 0.0 && b.is_finite() {
            Some(Self {
                func_type: 1,
                params: [
                    Self::canon(g),
                    Self::canon(a),
                    Self::canon(b),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            })
        } else {
            None
        }
    }

    /// ICC function type `1` (CIE 122-1966), panicking if parameters are invalid.
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn type1_cie122(g: f32, a: f32, b: f32) -> Self {
        match Self::try_type1_cie122(g, a, b) {
            Some(c) => c,
            None => {
                panic!("ICC parametric type 1 requires finite parameters with g > 0 and a != 0")
            }
        }
    }

    /// ICC function type `2` (IEC 61966-3): `Y = (aX + b)^g + c` for `X >= -b/a`, else `c`.
    #[inline]
    #[must_use]
    pub const fn try_type2_iec61966_3(g: f32, a: f32, b: f32, c: f32) -> Option<Self> {
        if g.is_finite() && g > 0.0 && a.is_finite() && a != 0.0 && b.is_finite() && c.is_finite() {
            Some(Self {
                func_type: 2,
                params: [
                    Self::canon(g),
                    Self::canon(a),
                    Self::canon(b),
                    Self::canon(c),
                    0.0,
                    0.0,
                    0.0,
                ],
            })
        } else {
            None
        }
    }

    /// ICC function type `2` (IEC 61966-3), panicking if parameters are invalid.
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn type2_iec61966_3(g: f32, a: f32, b: f32, c: f32) -> Self {
        match Self::try_type2_iec61966_3(g, a, b, c) {
            Some(curve) => curve,
            None => {
                panic!("ICC parametric type 2 requires finite parameters with g > 0 and a != 0")
            }
        }
    }

    /// ICC function type `3` (IEC 61966-2-1): `Y = (aX + b)^g` for `X >= d`, `Y = cX` for `X < d`.
    #[inline]
    #[must_use]
    pub const fn try_type3_iec61966_2_1(g: f32, a: f32, b: f32, c: f32, d: f32) -> Option<Self> {
        if g.is_finite()
            && g > 0.0
            && a.is_finite()
            && b.is_finite()
            && c.is_finite()
            && d.is_finite()
        {
            Some(Self {
                func_type: 3,
                params: [
                    Self::canon(g),
                    Self::canon(a),
                    Self::canon(b),
                    Self::canon(c),
                    Self::canon(d),
                    0.0,
                    0.0,
                ],
            })
        } else {
            None
        }
    }

    /// ICC function type `3` (IEC 61966-2-1), panicking if parameters are invalid.
    #[inline]
    #[track_caller]
    #[must_use]
    pub const fn type3_iec61966_2_1(g: f32, a: f32, b: f32, c: f32, d: f32) -> Self {
        match Self::try_type3_iec61966_2_1(g, a, b, c, d) {
            Some(curve) => curve,
            None => panic!("ICC parametric type 3 requires finite parameters with g > 0"),
        }
    }

    /// ICC function type `4`: `Y = (aX + b)^g + e` for `X >= d`, `Y = cX + f` for `X < d`.
    #[inline]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn try_type4(g: f32, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Option<Self> {
        if g.is_finite()
            && g > 0.0
            && a.is_finite()
            && b.is_finite()
            && c.is_finite()
            && d.is_finite()
            && e.is_finite()
            && f.is_finite()
        {
            Some(Self {
                func_type: 4,
                params: [
                    Self::canon(g),
                    Self::canon(a),
                    Self::canon(b),
                    Self::canon(c),
                    Self::canon(d),
                    Self::canon(e),
                    Self::canon(f),
                ],
            })
        } else {
            None
        }
    }

    /// ICC function type `4`, panicking if parameters are invalid.
    #[inline]
    #[track_caller]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn type4(g: f32, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Self {
        match Self::try_type4(g, a, b, c, d, e, f) {
            Some(curve) => curve,
            None => panic!("ICC parametric type 4 requires finite parameters with g > 0"),
        }
    }

    /// ICC `parametricCurveType` function type (`0..=4`).
    #[inline]
    #[must_use]
    pub const fn function_type(self) -> u8 {
        self.func_type
    }

    /// Active ICC parameters (`1`, `3`, `4`, `5`, or `7` elements according to [`Self::function_type`]).
    #[inline]
    #[must_use]
    pub fn icc_params(&self) -> &[f32] {
        let len = match self.func_type {
            0 => 1,
            1 => 3,
            2 => 4,
            3 => 5,
            _ => 7,
        };
        &self.params[..len]
    }

    /// Full 7-slot parameter array `[g, a, b, c, d, e, f]` (with unused trailing slots zeroed).
    #[inline]
    #[must_use]
    pub const fn all_params(self) -> [f32; 7] {
        self.params
    }
}

/// Transfer curve for [`ParametricColor`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParametricTransfer {
    /// Linear light (`gamma = 1.0`, CICP 8).
    Linear,
    /// IEC 61966-2-1 sRGB piecewise transfer (CICP 13).
    Srgb,
    /// ITU-R BT.709 / BT.601 / BT.2020 SDR transfer (CICP 1 / 6 / 14 / 15).
    Bt709,
    /// SMPTE ST 2084 Perceptual Quantizer (CICP 16).
    Pq,
    /// ITU-R BT.2100 Hybrid Log-Gamma (CICP 18).
    Hlg,
    /// SMPTE ST 428-1 theatrical 2.6 power law (CICP 17, JPEG XL `TransferFunction::Dci`).
    Dci,
    /// Pure power-law transfer curve (`encoded = linear^g`).
    Gamma(GammaCurve),
    /// ICC.1 §10.18 `parametricCurveType` EOTF curve (`funcType` `0..=4`).
    IccParametric(IccParametricCurve),
    /// Unspecified or unknown transfer curve.
    #[default]
    Unknown,
}

impl ParametricTransfer {
    /// Convert from a compact [`TransferFunction`].
    #[inline]
    #[must_use]
    pub const fn from_transfer_function(tf: TransferFunction) -> Self {
        match tf {
            TransferFunction::Linear => Self::Linear,
            TransferFunction::Srgb => Self::Srgb,
            TransferFunction::Bt709 => Self::Bt709,
            TransferFunction::Pq => Self::Pq,
            TransferFunction::Hlg => Self::Hlg,
            TransferFunction::Gamma22 => Self::Gamma(GammaCurve::ADOBE_RGB),
            TransferFunction::Unknown => Self::Unknown,
        }
    }

    /// Map to a compact [`TransferFunction`] when an exact built-in curve exists.
    ///
    /// Returns `None` for [`Self::Dci`], [`Self::IccParametric`], [`Self::Unknown`],
    /// and [`Self::Gamma`] exponents other than [`GammaCurve::ADOBE_RGB`] (`256/563`).
    #[inline]
    #[must_use]
    pub const fn to_transfer_function(self) -> Option<TransferFunction> {
        match self {
            Self::Linear => Some(TransferFunction::Linear),
            Self::Srgb => Some(TransferFunction::Srgb),
            Self::Bt709 => Some(TransferFunction::Bt709),
            Self::Pq => Some(TransferFunction::Pq),
            Self::Hlg => Some(TransferFunction::Hlg),
            Self::Gamma(g)
                if g.encoding_exponent().to_bits()
                    == GammaCurve::ADOBE_RGB.encoding_exponent().to_bits() =>
            {
                Some(TransferFunction::Gamma22)
            }
            Self::Dci | Self::Gamma(_) | Self::IccParametric(_) | Self::Unknown => None,
        }
    }
}

/// ICC / PNG `sRGB` / JPEG XL rendering intent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RenderingIntent {
    /// Perceptual gamut compression (ICC / PNG / JPEG XL code `0`).
    Perceptual,
    /// Media-relative colorimetric, preserving in-gamut colors relative to white (code `1`).
    #[default]
    RelativeColorimetric,
    /// Saturation-preserving business graphics intent (code `2`).
    Saturation,
    /// ICC-absolute colorimetric, preserving white-point chromaticity (code `3`).
    AbsoluteColorimetric,
}

impl RenderingIntent {
    /// Decode from the 0..=3 integer code shared by ICC headers, PNG `sRGB` chunks,
    /// and JPEG XL `ColorEncoding`.
    #[inline]
    #[must_use]
    pub const fn from_icc_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Perceptual),
            1 => Some(Self::RelativeColorimetric),
            2 => Some(Self::Saturation),
            3 => Some(Self::AbsoluteColorimetric),
            _ => None,
        }
    }

    /// Encode to the 0..=3 integer code shared by ICC headers, PNG `sRGB` chunks,
    /// and JPEG XL `ColorEncoding`.
    #[inline]
    #[must_use]
    pub const fn to_icc_code(self) -> u8 {
        match self {
            Self::Perceptual => 0,
            Self::RelativeColorimetric => 1,
            Self::Saturation => 2,
            Self::AbsoluteColorimetric => 3,
        }
    }
}

/// Non-ICC parametric color description for boxed [`ColorContext`] and [`ColorOrigin`].
///
/// Captures compact parametric color headers — JPEG XL's non-ICC `ColorEncoding`
/// (`primaries`, `white_point`, `transfer_function`, `rendering_intent`, including
/// custom CIE `xy` chromaticities and explicit power-law gamma) as well as PNG
/// `gAMA` + `cHRM` (+ `sRGB` rendering intent) — without inflating [`PixelDescriptor`]
/// on inner row/window loops or synthesizing fake ICC bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParametricColor {
    /// RGB primaries, or `None` for grayscale color encodings.
    pub primaries: Option<ParametricPrimaries>,
    /// Reference white point.
    pub white_point: WhitePoint,
    /// Electro-optical / opto-electronic transfer curve.
    pub transfer: ParametricTransfer,
    /// Rendering intent signaled alongside the parametric header.
    pub rendering_intent: RenderingIntent,
}

impl ParametricColor {
    /// Standard sRGB (`Bt709` + `D65` + `Srgb` + `RelativeColorimetric`).
    pub const SRGB: Self = Self::rgb(
        ParametricPrimaries::Bt709,
        WhitePoint::D65,
        ParametricTransfer::Srgb,
    );
    /// Apple Display P3 (`P3` + `D65` + `Srgb` + `RelativeColorimetric`).
    pub const DISPLAY_P3: Self = Self::rgb(
        ParametricPrimaries::P3,
        WhitePoint::D65,
        ParametricTransfer::Srgb,
    );
    /// Theatrical SMPTE DCI-P3 (`P3` + `Dci` + `Dci` + `RelativeColorimetric`).
    pub const DCI_P3: Self = Self::rgb(
        ParametricPrimaries::P3,
        WhitePoint::Dci,
        ParametricTransfer::Dci,
    );
    /// Adobe RGB (1998) (`AdobeRgb` + `D65` + `Gamma(256/563)` + `RelativeColorimetric`).
    pub const ADOBE_RGB: Self = Self::rgb(
        ParametricPrimaries::AdobeRgb,
        WhitePoint::D65,
        ParametricTransfer::Gamma(GammaCurve::ADOBE_RGB),
    );
    /// ITU-R BT.2100 PQ (`Bt2020` + `D65` + `Pq` + `RelativeColorimetric`).
    pub const BT2100_PQ: Self = Self::rgb(
        ParametricPrimaries::Bt2020,
        WhitePoint::D65,
        ParametricTransfer::Pq,
    );
    /// ITU-R BT.2100 HLG (`Bt2020` + `D65` + `Hlg` + `RelativeColorimetric`).
    pub const BT2100_HLG: Self = Self::rgb(
        ParametricPrimaries::Bt2020,
        WhitePoint::D65,
        ParametricTransfer::Hlg,
    );
    /// Linear-light sRGB / BT.709 (`Bt709` + `D65` + `Linear` + `RelativeColorimetric`).
    pub const LINEAR_SRGB: Self = Self::rgb(
        ParametricPrimaries::Bt709,
        WhitePoint::D65,
        ParametricTransfer::Linear,
    );
    /// ISO 22028-2 ROMM RGB / ProPhoto RGB (`ProPhotoRgb` + `D50` + `IccParametric::PROPHOTO_RGB` + `RelativeColorimetric`).
    pub const PROPHOTO_RGB: Self = Self::rgb(
        ParametricPrimaries::ProPhotoRgb,
        WhitePoint::D50,
        ParametricTransfer::IccParametric(IccParametricCurve::PROPHOTO_RGB),
    );
    /// ACEScg (AP1) linear working space (`AcesCg` + `D60` + `Linear` + `RelativeColorimetric`).
    pub const ACES_CG: Self = Self::rgb(
        ParametricPrimaries::AcesCg,
        WhitePoint::D60,
        ParametricTransfer::Linear,
    );
    /// SMPTE ST 2065-1 ACES archival (AP0) linear space (`Aces2065_1` + `D60` + `Linear` + `RelativeColorimetric`).
    pub const ACES_2065_1: Self = Self::rgb(
        ParametricPrimaries::Aces2065_1,
        WhitePoint::D60,
        ParametricTransfer::Linear,
    );

    /// Construct an RGB parametric color description with default
    /// [`RenderingIntent::RelativeColorimetric`].
    #[inline]
    #[must_use]
    pub const fn rgb(
        primaries: ParametricPrimaries,
        white_point: WhitePoint,
        transfer: ParametricTransfer,
    ) -> Self {
        Self {
            primaries: Some(primaries),
            white_point,
            transfer,
            rendering_intent: RenderingIntent::RelativeColorimetric,
        }
    }

    /// Construct a grayscale parametric color description (`primaries = None`)
    /// with default [`RenderingIntent::RelativeColorimetric`].
    #[inline]
    #[must_use]
    pub const fn gray(white_point: WhitePoint, transfer: ParametricTransfer) -> Self {
        Self {
            primaries: None,
            white_point,
            transfer,
            rendering_intent: RenderingIntent::RelativeColorimetric,
        }
    }

    /// Override the rendering intent.
    #[inline]
    #[must_use]
    pub const fn with_rendering_intent(mut self, intent: RenderingIntent) -> Self {
        self.rendering_intent = intent;
        self
    }

    /// Decompose to a compact `(ColorPrimaries, TransferFunction)` pair when this
    /// description uses exact [`WhitePoint::D65`], named RGB primaries, and a
    /// built-in [`TransferFunction`].
    ///
    /// Returns `None` for grayscale (`primaries: None`), non-D65 white points
    /// (such as theatrical `P3` + [`WhitePoint::Dci`]), custom CIE `xy` primaries,
    /// [`ParametricTransfer::Dci`], or arbitrary [`GammaCurve`] exponents.
    #[inline]
    #[must_use]
    pub const fn to_primaries_transfer(&self) -> Option<(ColorPrimaries, TransferFunction)> {
        if !matches!(self.white_point, WhitePoint::D65) {
            return None;
        }
        let primaries = match self.primaries {
            Some(ParametricPrimaries::Bt709) => ColorPrimaries::Bt709,
            Some(ParametricPrimaries::Bt2020) => ColorPrimaries::Bt2020,
            Some(ParametricPrimaries::P3) => ColorPrimaries::DisplayP3,
            Some(ParametricPrimaries::AdobeRgb) => ColorPrimaries::AdobeRgb,
            Some(
                ParametricPrimaries::ProPhotoRgb
                | ParametricPrimaries::AcesCg
                | ParametricPrimaries::Aces2065_1
                | ParametricPrimaries::Bt601_525
                | ParametricPrimaries::Bt601_625
                | ParametricPrimaries::Custom { .. },
            )
            | None => return None,
        };
        match self.transfer.to_transfer_function() {
            Some(transfer) => Some((primaries, transfer)),
            None => None,
        }
    }

    /// Construct from full-range RGB ITU-T H.273 [`Cicp`] (`matrix_coefficients == 0`
    /// and `full_range == true`) when `color_primaries` and `transfer_characteristics`
    /// map to known parametric presets.
    #[inline]
    #[must_use]
    pub const fn from_cicp(cicp: Cicp) -> Option<Self> {
        if cicp.matrix_coefficients != 0 || !cicp.full_range {
            return None;
        }
        let (primaries, white_point) = match cicp.color_primaries {
            1 => (ParametricPrimaries::Bt709, WhitePoint::D65),
            5 => (ParametricPrimaries::Bt601_625, WhitePoint::D65),
            6 | 7 => (ParametricPrimaries::Bt601_525, WhitePoint::D65),
            9 => (ParametricPrimaries::Bt2020, WhitePoint::D65),
            11 => (ParametricPrimaries::P3, WhitePoint::Dci),
            12 => (ParametricPrimaries::P3, WhitePoint::D65),
            _ => return None,
        };
        let transfer = match cicp.transfer_characteristics {
            1 | 6 | 14 | 15 => ParametricTransfer::Bt709,
            4 => ParametricTransfer::Gamma(GammaCurve::GAMMA_22),
            8 => ParametricTransfer::Linear,
            13 => ParametricTransfer::Srgb,
            16 => ParametricTransfer::Pq,
            17 => ParametricTransfer::Dci,
            18 => ParametricTransfer::Hlg,
            _ => return None,
        };
        Some(Self::rgb(primaries, white_point, transfer))
    }

    /// Convert to full-range RGB ITU-T H.273 [`Cicp`] (`matrix_coefficients = 0`,
    /// `full_range = true`) when an exact CICP code point exists for
    /// `(primaries, white_point, transfer)`.
    ///
    /// Note that CICP does not carry [`RenderingIntent`]; inspect
    /// `self.rendering_intent` separately when rendering intent is load-bearing.
    #[inline]
    #[must_use]
    pub const fn to_cicp(&self) -> Option<Cicp> {
        let cp = match (self.primaries, self.white_point) {
            (Some(ParametricPrimaries::Bt709), WhitePoint::D65) => 1,
            (Some(ParametricPrimaries::Bt601_625), WhitePoint::D65) => 5,
            (Some(ParametricPrimaries::Bt601_525), WhitePoint::D65) => 6,
            (Some(ParametricPrimaries::Bt2020), WhitePoint::D65) => 9,
            (Some(ParametricPrimaries::P3), WhitePoint::Dci) => 11,
            (Some(ParametricPrimaries::P3), WhitePoint::D65) => 12,
            _ => return None,
        };
        let tc = match self.transfer {
            ParametricTransfer::Bt709 => 1,
            ParametricTransfer::Gamma(g)
                if g.encoding_exponent().to_bits()
                    == GammaCurve::GAMMA_22.encoding_exponent().to_bits() =>
            {
                4
            }
            ParametricTransfer::Linear => 8,
            ParametricTransfer::Srgb => 13,
            ParametricTransfer::Pq => 16,
            ParametricTransfer::Dci => 17,
            ParametricTransfer::Hlg => 18,
            _ => return None,
        };
        Some(Cicp::new(cp, tc, 0, true))
    }
}

/// A source color profile — ICC bytes, CICP parameters, parametric header
/// signaling, a named profile, or a primaries + transfer function pair.
///
/// This unified type lets consumers pass decoded image color info
/// directly to a CMS backend without caring whether the source had
/// an ICC profile, CICP codes, parametric chromaticities/gamma, or a
/// well-known named profile.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorProfileSource<'a> {
    /// Raw ICC profile data.
    Icc(&'a [u8]),
    /// CICP parameters (a CMS can synthesize an equivalent profile).
    Cicp(Cicp),
    /// Non-ICC parametric color header (JPEG XL `ColorEncoding`, PNG `gAMA`+`cHRM`).
    Parametric(ParametricColor),
    /// Well-known named profile.
    Named(NamedProfile),
    /// Color primaries + transfer function pair.
    ///
    /// Covers the full `ColorPrimaries × TransferFunction` matrix,
    /// including combinations that don't have a [`NamedProfile`] variant
    /// or a CICP mapping (e.g., Adobe RGB).
    ///
    /// A CMS backend that handles this variant can avoid ICC profile
    /// parsing entirely for known primaries/transfer combinations.
    PrimariesTransferPair {
        /// Color primaries (gamut + white point).
        primaries: ColorPrimaries,
        /// Transfer function (EOTF encoding).
        transfer: TransferFunction,
    },
}

/// Well-known color profiles that any CMS should recognize.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum NamedProfile {
    /// sRGB (IEC 61966-2-1). The web and desktop default.
    #[default]
    Srgb,
    /// Display P3 with sRGB transfer curve.
    DisplayP3,
    /// BT.2020 with BT.709 transfer (SDR wide gamut).
    Bt2020,
    /// BT.2020 with PQ transfer (HDR10, SMPTE ST 2084).
    Bt2020Pq,
    /// BT.2020 with HLG transfer (BT.2100 HLG).
    Bt2020Hlg,
    /// Adobe RGB (1998). Used in print workflows.
    AdobeRgb,
    /// Linear sRGB (sRGB primaries, gamma 1.0).
    LinearSrgb,
}

impl NamedProfile {
    /// Map CICP parameters to a well-known named profile.
    ///
    /// Recognizes sRGB, Display P3, BT.2020 (SDR), BT.2100 PQ,
    /// and Linear sRGB. Returns `None` for unrecognized combinations.
    pub const fn from_cicp(cicp: Cicp) -> Option<Self> {
        // Match on (primaries, transfer, matrix, full_range).
        // We match matrix_coefficients == 0 (Identity/RGB) for all RGB profiles.
        match (
            cicp.color_primaries,
            cicp.transfer_characteristics,
            cicp.matrix_coefficients,
        ) {
            (1, 13, 0) => Some(Self::Srgb),
            (12, 13, 0) => Some(Self::DisplayP3),
            (9, 1, 0) => Some(Self::Bt2020),
            (9, 16, _) => Some(Self::Bt2020Pq), // BT.2100 PQ (any matrix)
            (9, 18, _) => Some(Self::Bt2020Hlg), // BT.2100 HLG (any matrix)
            (1, 8, 0) => Some(Self::LinearSrgb),
            _ => None,
        }
    }

    /// Convert to CICP parameters, if a standard mapping exists.
    pub const fn to_cicp(self) -> Option<Cicp> {
        match self {
            Self::Srgb => Some(Cicp::SRGB),
            Self::DisplayP3 => Some(Cicp::DISPLAY_P3),
            Self::Bt2020 => Some(Cicp {
                color_primaries: 9,
                transfer_characteristics: 1,
                matrix_coefficients: 0,
                full_range: true,
            }),
            Self::Bt2020Pq => Some(Cicp::new(9, 16, 0, true)),
            Self::Bt2020Hlg => Some(Cicp::new(9, 18, 0, true)),
            Self::LinearSrgb => Some(Cicp {
                color_primaries: 1,
                transfer_characteristics: 8,
                matrix_coefficients: 0,
                full_range: true,
            }),
            Self::AdobeRgb => None,
        }
    }

    /// Decompose into primaries + transfer function.
    pub const fn to_primaries_transfer(self) -> (ColorPrimaries, TransferFunction) {
        match self {
            Self::Srgb => (ColorPrimaries::Bt709, TransferFunction::Srgb),
            Self::DisplayP3 => (ColorPrimaries::DisplayP3, TransferFunction::Srgb),
            Self::Bt2020 => (ColorPrimaries::Bt2020, TransferFunction::Bt709),
            Self::Bt2020Pq => (ColorPrimaries::Bt2020, TransferFunction::Pq),
            Self::Bt2020Hlg => (ColorPrimaries::Bt2020, TransferFunction::Hlg),
            Self::AdobeRgb => (ColorPrimaries::AdobeRgb, TransferFunction::Gamma22),
            Self::LinearSrgb => (ColorPrimaries::Bt709, TransferFunction::Linear),
        }
    }

    /// Try to construct from a primaries + transfer pair.
    ///
    /// Returns `None` for combinations that don't have a named profile
    /// (e.g., Display P3 + PQ, or Adobe RGB + Linear).
    pub const fn from_primaries_transfer(
        primaries: ColorPrimaries,
        transfer: TransferFunction,
    ) -> Option<Self> {
        match (primaries, transfer) {
            (ColorPrimaries::Bt709, TransferFunction::Srgb) => Some(Self::Srgb),
            (ColorPrimaries::DisplayP3, TransferFunction::Srgb) => Some(Self::DisplayP3),
            (ColorPrimaries::Bt2020, TransferFunction::Bt709) => Some(Self::Bt2020),
            (ColorPrimaries::Bt2020, TransferFunction::Pq) => Some(Self::Bt2020Pq),
            (ColorPrimaries::Bt2020, TransferFunction::Hlg) => Some(Self::Bt2020Hlg),
            (ColorPrimaries::AdobeRgb, TransferFunction::Gamma22) => Some(Self::AdobeRgb),
            (ColorPrimaries::Bt709, TransferFunction::Linear) => Some(Self::LinearSrgb),
            _ => None,
        }
    }
}

impl<'a> ColorProfileSource<'a> {
    /// Create from primaries + transfer function.
    pub const fn from_primaries_transfer(
        primaries: ColorPrimaries,
        transfer: TransferFunction,
    ) -> Self {
        Self::PrimariesTransferPair {
            primaries,
            transfer,
        }
    }

    /// Try to extract primaries + transfer, regardless of variant.
    ///
    /// - `PrimariesTransferPair` — returns directly
    /// - `Named` — decomposes via [`NamedProfile::to_primaries_transfer`]
    /// - `Cicp` — maps via [`ColorPrimaries::from_cicp`] and [`TransferFunction::from_cicp`]
    /// - `Parametric` — decomposes via [`ParametricColor::to_primaries_transfer`]
    /// - `Icc` — returns `None` (requires profile parsing)
    pub const fn primaries_transfer(&self) -> Option<(ColorPrimaries, TransferFunction)> {
        match self {
            Self::PrimariesTransferPair {
                primaries,
                transfer,
            } => Some((*primaries, *transfer)),
            Self::Named(named) => Some(named.to_primaries_transfer()),
            Self::Cicp(cicp) => {
                let p = ColorPrimaries::from_cicp(cicp.color_primaries);
                let t = TransferFunction::from_cicp(cicp.transfer_characteristics);
                match (p, t) {
                    (Some(p), Some(t)) => Some((p, t)),
                    _ => None,
                }
            }
            Self::Parametric(p) => p.to_primaries_transfer(),
            Self::Icc(_) => None,
        }
    }

    /// Resolve to a (primaries, transfer) pair using all available methods,
    /// including ICC profile identification when the `icc` feature is enabled.
    ///
    /// This is the most complete resolution path:
    /// - `PrimariesTransferPair` — returns directly
    /// - `Named` — decomposes via [`NamedProfile::to_primaries_transfer`]
    /// - `Cicp` — maps via `from_cicp`, but returns `None` if `matrix_coefficients`
    ///   is non-zero (YCbCr data requires matrix conversion first) or `full_range`
    ///   is false (narrow-range data needs range expansion first)
    /// - `Parametric` — decomposes via [`ParametricColor::to_primaries_transfer`]
    /// - `Icc` — validated common-profile identification permitting matrix/TRC
    ///   substitution. Embedded CICP alone is insufficient; other ICCs need a CMS.
    ///
    /// Returns `None` when the profile is unknown or when reducing to
    /// (primaries, transfer) would discard significant information
    /// (YCbCr matrix coefficients, narrow signal range, custom chromaticities,
    /// non-D65 white point, or arbitrary gamma).
    #[cfg(feature = "icc")]
    #[allow(unreachable_patterns)]
    pub fn resolve(&self) -> Option<(ColorPrimaries, TransferFunction)> {
        match self {
            Self::PrimariesTransferPair {
                primaries,
                transfer,
            } => Some((*primaries, *transfer)),
            Self::Named(named) => Some(named.to_primaries_transfer()),
            Self::Cicp(cicp) => {
                // Non-identity matrix coefficients mean YCbCr data — can't reduce
                // to just primaries+transfer without a YCbCr→RGB matrix step.
                if cicp.matrix_coefficients != 0 {
                    return None;
                }
                // Narrow range needs expansion before primaries+transfer applies.
                if !cicp.full_range {
                    return None;
                }
                let p = ColorPrimaries::from_cicp(cicp.color_primaries)?;
                let t = TransferFunction::from_cicp(cicp.transfer_characteristics)?;
                Some((p, t))
            }
            Self::Parametric(p) => p.to_primaries_transfer(),
            Self::Icc(icc_bytes) => {
                // Only use the identification if matrix+TRC substitution is
                // safe — profiles with LUTs, non-Bradford chad, or Lab PCS
                // return MetadataOnly and need a full CMS.
                if let Some(id) = crate::icc::identify_common(icc_bytes) {
                    if id.valid_use == crate::icc::IdentificationUse::MatrixTrcSubstitution {
                        return Some((id.primaries, id.transfer));
                    }
                }
                // An embedded CICP tag does not prove the profile's TRCs,
                // LUTs or adaptation can be replaced. Preserve ICC authority;
                // callers choosing CICP explicitly pass ColorProfileSource::Cicp.
                None
            }
            _ => None,
        }
    }
}

/// Which color metadata a CMS should prefer when building transforms.
///
/// When both ICC and CICP are present, this determines which one the CMS
/// uses. When only one is present, it is used regardless of this flag —
/// the authority controls precedence, not exclusivity.
///
/// **Codec contract:** codecs should set this to match the data they
/// actually provide. Setting `Icc` without populating `icc_profile` (or
/// `Cicp` without `cicp`) is a codec bug. Implementations *may* fall back
/// to the other field when the authoritative one is absent, but are not
/// required to — they may also treat the mismatch as an error or assume
/// sRGB.
///
/// [`ColorContext::as_profile_source`] implements a lenient resolution:
/// preferred field → other field → `None`. Stricter consumers can inspect
/// the authority and the fields directly.
///
/// The codec sets this during decode based on the format's specification:
///
/// - **JPEG, WebP, TIFF**: `Icc` (ICC is the only color signal)
/// - **PNG 3rd Ed**: `Cicp` when cICP chunk present, `Icc` when only iCCP
/// - **AVIF/MIAF**: `Icc` when ICC colr box present, `Cicp` otherwise
/// - **HEIC/HEIF**: `Icc` when ICC colr box present, `Cicp` otherwise
/// - **JPEG XL**: `Cicp` when enum encoding (`want_icc=false`), `Icc` when embedded ICC
///
/// Readers' measured behavior for conflicting signals is in
/// `docs/color-signal-precedence.md`.
///
/// Both `icc_profile` and `cicp` may be populated regardless of this flag —
/// the non-authoritative field is preserved for metadata roundtripping.
///
/// This is distinct from [`ColorProvenance`] which records *how the source
/// described* its color (for re-encoding decisions). `ColorAuthority` says
/// which field the CMS should *prefer* for building transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ColorAuthority {
    /// ICC profile bytes take precedence for CMS transforms.
    ///
    /// The CMS should parse `icc_profile` and use its TRC curves directly.
    /// Any CICP tag embedded inside the ICC should NOT override the TRC
    /// (i.e., `allow_use_cicp_transfer: false` in moxcms terms).
    ///
    /// Codecs should only set this when `icc_profile` is populated.
    /// Lenient consumers may fall back to `cicp` if ICC is absent.
    #[default]
    Icc,
    /// CICP codes take precedence for CMS transforms.
    ///
    /// The CMS should build a source profile from the `cicp` field
    /// (e.g., `ColorProfile::new_from_cicp()` in moxcms). The ICC profile,
    /// if present, is for backwards-compatible metadata roundtripping only.
    ///
    /// Codecs should only set this when `cicp` is populated.
    /// Lenient consumers may fall back to `icc_profile` if CICP is absent.
    Cicp,
}

impl ColorAuthority {
    /// Choose between the ICC and CICP form of a color source: the authoritative one
    /// when it is present, otherwise the other. Every reader of the authority should
    /// choose through this, so that they cannot disagree.
    #[inline]
    pub fn select<T>(self, icc: Option<T>, cicp: Option<T>) -> Option<T> {
        match self {
            Self::Icc => icc.or(cicp),
            Self::Cicp => cicp.or(icc),
        }
    }
}

/// Authoritative extended color profile carried by [`ColorContext`].
///
/// A current pixel buffer has at most one active extended profile authority:
/// ITU-T H.273 [`Cicp`] parameters, raw ICC profile bytes, or a non-ICC
/// [`ParametricColor`] header description — never more than one simultaneously.
/// File-level multi-carrier signaling (for example PNG `cICP` + `iCCP` or AVIF
/// `colr` boxes) belongs on [`ColorOrigin`] alongside [`ColorAuthority`], which
/// resolves to a single authority at decode time.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ColorProfile {
    /// ITU-T H.273 CICP parameters.
    Cicp(Cicp),
    /// Raw ICC profile bytes.
    Icc(Arc<[u8]>),
    /// Non-ICC parametric color description (JPEG XL `ColorEncoding`, PNG `gAMA`+`cHRM`).
    Parametric(ParametricColor),
}

impl PartialEq for ColorProfile {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cicp(a), Self::Cicp(b)) => a == b,
            (Self::Icc(a), Self::Icc(b)) => Arc::ptr_eq(a, b) || a == b,
            (Self::Parametric(a), Self::Parametric(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for ColorProfile {}

#[cfg(feature = "hash")]
impl core::hash::Hash for ColorProfile {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Cicp(cicp) => cicp.hash(state),
            Self::Icc(icc) => icc[..].hash(state),
            Self::Parametric(p) => p.hash(state),
        }
    }
}

impl From<Cicp> for ColorProfile {
    fn from(cicp: Cicp) -> Self {
        Self::Cicp(cicp)
    }
}

impl From<Arc<[u8]>> for ColorProfile {
    fn from(icc: Arc<[u8]>) -> Self {
        Self::Icc(icc)
    }
}

impl From<ParametricColor> for ColorProfile {
    fn from(parametric: ParametricColor) -> Self {
        Self::Parametric(parametric)
    }
}

impl ColorProfile {
    /// Borrow as a [`ColorProfileSource`] for CMS integration.
    #[inline]
    pub fn as_profile_source(&self) -> ColorProfileSource<'_> {
        match self {
            Self::Cicp(cicp) => ColorProfileSource::Cicp(*cicp),
            Self::Icc(icc) => ColorProfileSource::Icc(icc),
            Self::Parametric(p) => ColorProfileSource::Parametric(*p),
        }
    }

    /// The [`Cicp`] parameters when this profile is `Cicp`.
    #[inline]
    pub const fn cicp(&self) -> Option<Cicp> {
        match self {
            Self::Cicp(cicp) => Some(*cicp),
            _ => None,
        }
    }

    /// Borrowed ICC profile bytes when this profile is `Icc`.
    #[inline]
    pub fn icc(&self) -> Option<&[u8]> {
        match self {
            Self::Icc(icc) => Some(icc),
            _ => None,
        }
    }

    /// Shared ICC profile allocation when this profile is `Icc`.
    #[inline]
    pub const fn icc_arc(&self) -> Option<&Arc<[u8]>> {
        match self {
            Self::Icc(icc) => Some(icc),
            _ => None,
        }
    }

    /// The [`ParametricColor`] parameters when this profile is `Parametric`.
    #[inline]
    pub const fn parametric(&self) -> Option<ParametricColor> {
        match self {
            Self::Parametric(p) => Some(*p),
            _ => None,
        }
    }
}

/// Extended color space and luminance metadata for current pixel data.
///
/// Bundles at most one authoritative [`ColorProfile`] (ICC bytes, CICP
/// parameters, or [`ParametricColor`], never more than one) plus optional
/// rendering intent and scene/display luminance anchors into a single shareable
/// context. Carried boxed out-of-line via `Option<Arc<ColorContext>>` on pixel
/// slices and buffers so inner window/row loops stay compact in registers while
/// color metadata travels without per-strip byte cloning.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ColorContext {
    profile: Option<ColorProfile>,
    // Always `None` when `profile` is `Some(ColorProfile::Parametric(_))` so
    // `ParametricColor.rendering_intent` is the single canonical slot.
    rendering_intent: Option<RenderingIntent>,
    diffuse_white: Option<DiffuseWhite>,
    hdr_peak: Option<crate::hdr::Nits>,
}

#[cfg(feature = "hash")]
impl core::hash::Hash for ColorContext {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.icc().hash(state);
        self.cicp().hash(state);
        self.diffuse_white.hash(state);
        self.hdr_peak.hash(state);
        if self.parametric().is_some() || self.rendering_intent.is_some() {
            state.write(b"zenpixels/color/ext1\0");
            self.parametric().hash(state);
            self.rendering_intent.hash(state);
        }
    }
}

impl ColorContext {
    /// Create an empty context with no profile, rendering intent, or luminance anchors.
    #[inline]
    pub const fn new() -> Self {
        Self {
            profile: None,
            rendering_intent: None,
            diffuse_white: None,
            hdr_peak: None,
        }
    }

    /// Create from an authoritative [`ColorProfile`].
    #[inline]
    pub const fn from_profile(profile: ColorProfile) -> Self {
        Self {
            profile: Some(profile),
            rendering_intent: None,
            diffuse_white: None,
            hdr_peak: None,
        }
    }

    /// Create from an ICC profile.
    pub fn from_icc(icc: impl Into<Arc<[u8]>>) -> Self {
        Self::from_profile(ColorProfile::Icc(icc.into()))
    }

    /// Create from CICP parameters.
    #[inline]
    pub const fn from_cicp(cicp: Cicp) -> Self {
        Self::from_profile(ColorProfile::Cicp(cicp))
    }

    /// Create from a non-ICC [`ParametricColor`] header description.
    #[inline]
    pub const fn from_parametric(parametric: ParametricColor) -> Self {
        Self::from_profile(ColorProfile::Parametric(parametric))
    }

    /// The active extended [`ColorProfile`], if any.
    #[inline]
    pub const fn profile(&self) -> Option<&ColorProfile> {
        self.profile.as_ref()
    }

    /// Borrowed ICC profile bytes when the active profile is [`ColorProfile::Icc`].
    #[inline]
    pub fn icc(&self) -> Option<&[u8]> {
        match &self.profile {
            Some(ColorProfile::Icc(icc)) => Some(icc),
            _ => None,
        }
    }

    /// Shared ICC profile allocation when the active profile is [`ColorProfile::Icc`].
    #[inline]
    pub const fn icc_arc(&self) -> Option<&Arc<[u8]>> {
        match &self.profile {
            Some(ColorProfile::Icc(icc)) => Some(icc),
            _ => None,
        }
    }

    /// CICP parameters when the active profile is [`ColorProfile::Cicp`].
    #[inline]
    pub const fn cicp(&self) -> Option<Cicp> {
        match &self.profile {
            Some(ColorProfile::Cicp(cicp)) => Some(*cicp),
            _ => None,
        }
    }

    /// Parametric color parameters when the active profile is [`ColorProfile::Parametric`].
    #[inline]
    pub const fn parametric(&self) -> Option<ParametricColor> {
        match &self.profile {
            Some(ColorProfile::Parametric(p)) => Some(*p),
            _ => None,
        }
    }

    /// Explicit rendering intent when signaled (either standalone on the context
    /// or inside an active [`ColorProfile::Parametric`]).
    #[inline]
    pub const fn rendering_intent(&self) -> Option<RenderingIntent> {
        match &self.profile {
            Some(ColorProfile::Parametric(p)) => Some(p.rendering_intent),
            _ => self.rendering_intent,
        }
    }

    /// Absolute-luminance anchor: the cd/m² (nits) that a relative-linear
    /// sample value of `1.0` represents, or `None` when unsignaled.
    #[inline]
    pub const fn diffuse_white(&self) -> Option<DiffuseWhite> {
        self.diffuse_white
    }

    /// Declared source peak luminance, used only when explicitly requested.
    #[inline]
    pub const fn hdr_peak(&self) -> Option<crate::hdr::Nits> {
        self.hdr_peak
    }

    /// True when no profile, rendering intent, or luminance anchor is set.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.profile.is_none()
            && self.rendering_intent.is_none()
            && self.diffuse_white.is_none()
            && self.hdr_peak.is_none()
    }

    /// Set or clear the active [`ColorProfile`], preserving luminance anchors.
    #[must_use]
    pub fn with_profile(mut self, mut profile: Option<ColorProfile>) -> Self {
        if let Some(ColorProfile::Parametric(p)) = &mut profile {
            if let Some(intent) = self.rendering_intent.take() {
                p.rendering_intent = intent;
            }
        }
        self.profile = profile;
        self
    }

    /// Set explicit [`RenderingIntent`] on this context (updating
    /// `ParametricColor::rendering_intent` in place when the active profile is
    /// [`ColorProfile::Parametric`]).
    #[must_use]
    pub const fn with_rendering_intent(mut self, intent: RenderingIntent) -> Self {
        if let Some(ColorProfile::Parametric(p)) = &mut self.profile {
            p.rendering_intent = intent;
            self.rendering_intent = None;
        } else {
            self.rendering_intent = Some(intent);
        }
        self
    }

    /// Attach an absolute-luminance anchor — the nits that relative-linear
    /// `1.0` represents — returning the updated context.
    ///
    /// Set this on an HDR working buffer (e.g. after gain-map reconstruction,
    /// where the anchor is [`DiffuseWhite::BT2408`] = 203) so downstream
    /// conversion can map relative-linear light to absolute cd/m² for PQ/HLG
    /// encode and tone-mapping.
    #[must_use]
    pub const fn with_diffuse_white(mut self, white: DiffuseWhite) -> Self {
        self.diffuse_white = Some(white);
        self
    }

    /// Set declared source peak luminance for `HdrPeak::Declared` conversion.
    #[must_use]
    pub const fn with_hdr_peak(mut self, peak: crate::hdr::Nits) -> Self {
        self.hdr_peak = Some(peak);
        self
    }

    /// Set [`Cicp`] as the active profile, replacing any prior profile.
    #[must_use]
    pub fn with_cicp(mut self, cicp: Cicp) -> Self {
        self.profile = Some(ColorProfile::Cicp(cicp));
        self
    }

    /// Set raw ICC bytes as the active profile, replacing any prior profile.
    #[must_use]
    pub fn with_icc(mut self, icc: impl Into<Arc<[u8]>>) -> Self {
        self.profile = Some(ColorProfile::Icc(icc.into()));
        self
    }

    /// Set [`ParametricColor`] as the active profile, replacing any prior profile.
    #[must_use]
    pub fn with_parametric(mut self, parametric: ParametricColor) -> Self {
        self.profile = Some(ColorProfile::Parametric(parametric));
        self.rendering_intent = None;
        self
    }

    /// Get a [`ColorProfileSource`] reference for CMS integration.
    ///
    /// Returns [`ColorProfileSource::Icc`], [`ColorProfileSource::Cicp`], or
    /// [`ColorProfileSource::Parametric`] according to the single active
    /// [`ColorProfile`], or `None` when no extended profile is set.
    pub fn as_profile_source(&self) -> Option<ColorProfileSource<'_>> {
        self.profile.as_ref().map(ColorProfile::as_profile_source)
    }

    /// Derive the selected current profile's transfer function.
    ///
    /// CICP or [`ParametricColor`] is used when it is the active profile. ICC
    /// remains opaque here, so an ICC-backed context returns `Unknown` without
    /// identifying or approximating ICC profiles.
    pub fn transfer_function(&self) -> TransferFunction {
        match self.as_profile_source() {
            Some(ColorProfileSource::Cicp(c)) => {
                TransferFunction::from_cicp(c.transfer_characteristics)
                    .unwrap_or(TransferFunction::Unknown)
            }
            Some(ColorProfileSource::Parametric(p)) => p
                .transfer
                .to_transfer_function()
                .unwrap_or(TransferFunction::Unknown),
            _ => TransferFunction::Unknown,
        }
    }

    /// True when the active profile is explicitly sRGB via CICP or [`ParametricColor`].
    pub fn is_srgb(&self) -> bool {
        self.cicp() == Some(Cicp::SRGB)
            || self
                .parametric()
                .is_some_and(|p| p.to_cicp() == Some(Cicp::SRGB))
    }
}

/// Complete pixel interpretation for an interleaved view or buffer: its compact
/// [`PixelDescriptor`] paired with an optional borrowed [`ColorContext`].
///
/// `PixelEncoding` is `16` bytes on 64-bit targets (`12` bytes on 32-bit
/// targets), `Copy`, and incurs zero `Arc` reference-count traffic. Conversion
/// and output planners accept `PixelEncoding<'_>` so callers pass
/// `slice.encoding()`, `buffer.encoding()`, or `descriptor.encoding()` as a
/// single coherent value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "hash", derive(Hash))]
pub struct PixelEncoding<'a> {
    descriptor: PixelDescriptor,
    color: Option<&'a ColorContext>,
}

const _: () = assert!(core::mem::size_of::<PixelEncoding<'static>>() <= 16);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<PixelEncoding<'static>>() == 16);

impl From<PixelDescriptor> for PixelEncoding<'static> {
    #[inline]
    fn from(descriptor: PixelDescriptor) -> Self {
        Self::from_descriptor(descriptor)
    }
}

impl<'a> PixelEncoding<'a> {
    /// Combine a [`PixelDescriptor`] with an optional borrowed [`ColorContext`].
    #[inline]
    pub const fn new(descriptor: PixelDescriptor, color: Option<&'a ColorContext>) -> Self {
        Self { descriptor, color }
    }

    /// Describe pixels that have only compact [`PixelDescriptor`] signaling.
    #[inline]
    pub const fn from_descriptor(descriptor: PixelDescriptor) -> Self {
        Self {
            descriptor,
            color: None,
        }
    }

    /// The compact format and color descriptor.
    #[inline]
    pub const fn descriptor(self) -> PixelDescriptor {
        self.descriptor
    }

    /// The borrowed extended [`ColorContext`], if attached.
    #[inline]
    pub const fn color_context(self) -> Option<&'a ColorContext> {
        self.color
    }

    /// Borrowed ICC profile bytes when the attached context holds [`ColorProfile::Icc`].
    #[inline]
    pub fn icc(self) -> Option<&'a [u8]> {
        self.color.and_then(ColorContext::icc)
    }

    /// CICP parameters when the attached context holds [`ColorProfile::Cicp`].
    #[inline]
    pub const fn cicp(self) -> Option<Cicp> {
        match self.color {
            Some(ctx) => ctx.cicp(),
            None => None,
        }
    }

    /// Parametric color parameters when the attached context holds [`ColorProfile::Parametric`].
    #[inline]
    pub const fn parametric(self) -> Option<ParametricColor> {
        match self.color {
            Some(ctx) => ctx.parametric(),
            None => None,
        }
    }

    /// Signaled rendering intent on the attached context, if any.
    #[inline]
    pub const fn rendering_intent(self) -> Option<RenderingIntent> {
        match self.color {
            Some(ctx) => ctx.rendering_intent(),
            None => None,
        }
    }

    /// Signaled diffuse-white anchor on the attached context, if any.
    #[inline]
    pub const fn diffuse_white(self) -> Option<DiffuseWhite> {
        match self.color {
            Some(ctx) => ctx.diffuse_white(),
            None => None,
        }
    }

    /// Declared source HDR peak on the attached context, if any.
    #[inline]
    pub const fn hdr_peak(self) -> Option<crate::hdr::Nits> {
        match self.color {
            Some(ctx) => ctx.hdr_peak(),
            None => None,
        }
    }

    /// Resolve a [`ColorProfileSource`] from the extended context if it holds a
    /// profile, falling back to the compact descriptor's primaries and transfer.
    #[inline]
    pub fn color_profile_source(self) -> ColorProfileSource<'a> {
        self.color
            .and_then(ColorContext::as_profile_source)
            .unwrap_or_else(|| self.descriptor.color_profile_source())
    }
}

// ---------------------------------------------------------------------------
// Color provenance — how the source described its color
// ---------------------------------------------------------------------------

/// How the source file described its color information.
///
/// Preserved for re-encoding and round-trip conversion decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorProvenance {
    /// Color was described via an embedded ICC profile.
    Icc,
    /// Color was described via CICP code points.
    Cicp,
    /// Color was described via PNG gAMA + cHRM chunks or similar.
    GamaChrm,
    /// Color was assume-defaulted (no explicit metadata in source).
    Assumed,
    /// Color was described via a non-ICC parametric header (e.g. JPEG XL `ColorEncoding`).
    Parametric,
}

/// Immutable record of how the source file described its color.
///
/// Tracks the original color description from the decoded file so
/// encoders can make provenance-aware decisions (e.g., re-embed the
/// original ICC profile, re-emit a compact JPEG XL `ColorEncoding`,
/// or prefer CICP when re-encoding to AVIF).
///
/// `ColorOrigin` is immutable once set. It records how color was
/// described, not what the pixels currently are. The encoder uses
/// [`PixelEncoding`] for the current state
/// and can consult `ColorOrigin` for provenance decisions.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "hash", derive(Hash))]
#[non_exhaustive]
pub struct ColorOrigin {
    /// Raw ICC profile bytes from the source file, if any.
    pub icc: Option<Arc<[u8]>>,
    /// CICP parameters from the source file, if any.
    pub cicp: Option<Cicp>,
    /// Parametric header color description (JPEG XL `ColorEncoding` or PNG `gAMA`+`cHRM`), if any.
    pub parametric: Option<ParametricColor>,
    /// How the color information was originally described.
    pub provenance: ColorProvenance,
    /// Which field a CMS should treat as authoritative for transforms.
    pub color_authority: ColorAuthority,
}

impl ColorOrigin {
    /// Create from an ICC profile.
    pub fn from_icc(icc: impl Into<Arc<[u8]>>) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: None,
            parametric: None,
            provenance: ColorProvenance::Icc,
            color_authority: ColorAuthority::Icc,
        }
    }

    /// Create from CICP parameters.
    pub fn from_cicp(cicp: Cicp) -> Self {
        Self {
            icc: None,
            cicp: Some(cicp),
            parametric: None,
            provenance: ColorProvenance::Cicp,
            color_authority: ColorAuthority::Cicp,
        }
    }

    /// Create from both ICC and CICP, with CICP authoritative. Formats where ICC wins, such as AVIF with
    /// an ICC `colr` box, set that with [`with_color_authority`](Self::with_color_authority).
    pub fn from_icc_and_cicp(icc: impl Into<Arc<[u8]>>, cicp: Cicp) -> Self {
        Self {
            icc: Some(icc.into()),
            cicp: Some(cicp),
            parametric: None,
            provenance: ColorProvenance::Cicp,
            color_authority: ColorAuthority::Cicp,
        }
    }

    /// Create from a non-ICC [`ParametricColor`] header (e.g. JPEG XL `ColorEncoding`).
    pub const fn from_parametric(parametric: ParametricColor) -> Self {
        Self {
            icc: None,
            cicp: parametric.to_cicp(),
            parametric: Some(parametric),
            provenance: ColorProvenance::Parametric,
            color_authority: ColorAuthority::Cicp,
        }
    }

    /// Create from PNG `gAMA` / `cHRM` chunks with preserved [`ParametricColor`] values.
    pub const fn from_gama_chrm_parametric(parametric: ParametricColor) -> Self {
        Self {
            icc: None,
            cicp: None,
            parametric: Some(parametric),
            provenance: ColorProvenance::GamaChrm,
            color_authority: ColorAuthority::Cicp,
        }
    }

    /// Create from gAMA/cHRM chunks (no ICC or CICP).
    pub fn from_gama_chrm() -> Self {
        Self {
            icc: None,
            cicp: None,
            parametric: None,
            provenance: ColorProvenance::GamaChrm,
            color_authority: ColorAuthority::Icc,
        }
    }

    /// Create for assumed/default color (no explicit metadata).
    pub fn assumed() -> Self {
        Self {
            icc: None,
            cicp: None,
            parametric: None,
            provenance: ColorProvenance::Assumed,
            color_authority: ColorAuthority::Icc,
        }
    }

    /// Override the color authority.
    pub fn with_color_authority(mut self, authority: ColorAuthority) -> Self {
        self.color_authority = authority;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn named_profile_default_is_srgb() {
        assert_eq!(NamedProfile::default(), NamedProfile::Srgb);
    }

    #[test]
    fn named_profile_to_cicp() {
        assert_eq!(NamedProfile::Srgb.to_cicp(), Some(Cicp::SRGB));
        assert_eq!(
            NamedProfile::Bt2020Pq.to_cicp(),
            Some(Cicp::new(9, 16, 0, true))
        );
        assert!(NamedProfile::AdobeRgb.to_cicp().is_none());
    }

    #[test]
    fn named_profile_from_cicp() {
        assert_eq!(
            NamedProfile::from_cicp(Cicp::SRGB),
            Some(NamedProfile::Srgb)
        );
        assert_eq!(
            NamedProfile::from_cicp(Cicp::DISPLAY_P3),
            Some(NamedProfile::DisplayP3)
        );
        assert_eq!(
            NamedProfile::from_cicp(Cicp::BT2100_PQ),
            Some(NamedProfile::Bt2020Pq)
        );
        // Linear sRGB
        assert_eq!(
            NamedProfile::from_cicp(Cicp::new(1, 8, 0, true)),
            Some(NamedProfile::LinearSrgb)
        );
        // BT.2020 SDR
        assert_eq!(
            NamedProfile::from_cicp(Cicp::new(9, 1, 0, true)),
            Some(NamedProfile::Bt2020)
        );
        // Unknown combo
        assert_eq!(NamedProfile::from_cicp(Cicp::new(99, 99, 0, true)), None);
    }

    #[test]
    fn named_profile_cicp_roundtrip() {
        for profile in [
            NamedProfile::Srgb,
            NamedProfile::DisplayP3,
            NamedProfile::Bt2020,
            NamedProfile::Bt2020Pq,
            NamedProfile::LinearSrgb,
        ] {
            let cicp = profile.to_cicp().unwrap();
            assert_eq!(
                NamedProfile::from_cicp(cicp),
                Some(profile),
                "roundtrip failed for {profile:?}"
            );
        }
    }

    #[test]
    fn color_context_from_cicp() {
        let ctx = ColorContext::from_cicp(Cicp::SRGB);
        assert!(ctx.icc().is_none());
        assert_eq!(ctx.cicp(), Some(Cicp::SRGB));
        assert_eq!(ctx.profile(), Some(&ColorProfile::Cicp(Cicp::SRGB)));
        assert!(!ctx.is_empty());
    }

    #[test]
    fn color_context_profile_variants_are_mutually_exclusive() {
        let ctx = ColorContext::from_icc(vec![1, 2, 3])
            .with_diffuse_white(DiffuseWhite::BT2408)
            .with_cicp(Cicp::SRGB);
        assert!(ctx.icc().is_none());
        assert_eq!(ctx.cicp(), Some(Cicp::SRGB));
        assert_eq!(ctx.diffuse_white(), Some(DiffuseWhite::BT2408));
        assert_eq!(
            ctx.as_profile_source(),
            Some(ColorProfileSource::Cicp(Cicp::SRGB))
        );

        let replaced = ctx.with_icc(vec![9, 8, 7]);
        assert_eq!(replaced.icc(), Some(&[9u8, 8, 7][..]));
        assert!(replaced.cicp().is_none());
        assert_eq!(replaced.diffuse_white(), Some(DiffuseWhite::BT2408));
        assert_eq!(
            replaced.as_profile_source(),
            Some(ColorProfileSource::Icc(&[9, 8, 7]))
        );
    }

    #[test]
    fn current_color_predicates_follow_the_selected_authority() {
        for icc in [&[][..], &[1, 2, 3][..]] {
            let context = ColorContext::from_icc(icc);
            assert_eq!(
                context.as_profile_source(),
                Some(ColorProfileSource::Icc(icc))
            );
            assert_eq!(context.transfer_function(), TransferFunction::Unknown);
            assert!(!context.is_srgb());
            assert!(context.cicp().is_none());
            assert_eq!(context.icc(), Some(icc));
        }
        let unknown = ColorContext::default();
        assert!(unknown.is_empty());
        assert_eq!(unknown.transfer_function(), TransferFunction::Unknown);
        assert!(!unknown.is_srgb());
    }

    #[test]
    fn color_context_is_srgb() {
        assert!(ColorContext::from_cicp(Cicp::SRGB).is_srgb());
        assert!(!ColorContext::from_cicp(Cicp::BT2100_PQ).is_srgb());
    }

    #[test]
    fn color_context_transfer_function() {
        assert_eq!(
            ColorContext::from_cicp(Cicp::SRGB).transfer_function(),
            TransferFunction::Srgb
        );
        assert_eq!(
            ColorContext::from_icc(vec![1]).transfer_function(),
            TransferFunction::Unknown
        );
    }

    // --- ColorContext additional coverage ---

    #[test]
    fn color_context_from_icc() {
        let ctx = ColorContext::from_icc(vec![10, 20, 30]);
        assert!(ctx.icc().is_some());
        assert_eq!(ctx.icc(), Some(&[10u8, 20, 30][..]));
        assert!(ctx.cicp().is_none());
    }

    #[test]
    fn color_context_with_profile_clears_profile_and_retains_anchors() {
        let ctx = ColorContext::from_icc(vec![1, 2])
            .with_diffuse_white(DiffuseWhite::BT2408)
            .with_profile(None);
        assert!(ctx.profile().is_none());
        assert!(ctx.icc().is_none());
        assert!(ctx.cicp().is_none());
        assert_eq!(ctx.diffuse_white(), Some(DiffuseWhite::BT2408));
        assert!(!ctx.is_empty());
    }

    #[test]
    fn color_context_profile_source_icc_only() {
        let ctx = ColorContext::from_icc(vec![42]);
        let src = ctx.as_profile_source().unwrap();
        assert_eq!(src, ColorProfileSource::Icc(&[42]));
    }

    #[test]
    fn color_context_profile_source_none() {
        let ctx = ColorContext::new();
        assert!(ctx.is_empty());
        assert!(ctx.as_profile_source().is_none());
    }

    #[test]
    fn pixel_encoding_combines_descriptor_and_borrowed_context() {
        let bare = PixelDescriptor::RGB8_SRGB.encoding();
        assert_eq!(bare, PixelEncoding::from(PixelDescriptor::RGB8_SRGB));
        assert_eq!(bare.descriptor(), PixelDescriptor::RGB8_SRGB);
        assert!(bare.color_context().is_none());
        assert!(bare.icc().is_none());
        assert!(bare.cicp().is_none());
        assert!(bare.diffuse_white().is_none());

        let ctx = ColorContext::from_cicp(Cicp::BT2100_PQ).with_diffuse_white(DiffuseWhite::BT2408);
        let enc = PixelEncoding::new(PixelDescriptor::RGB16_BT2100_PQ, Some(&ctx));
        assert_eq!(enc.cicp(), Some(Cicp::BT2100_PQ));
        assert_eq!(enc.diffuse_white(), Some(DiffuseWhite::BT2408));
        assert_eq!(
            enc.color_profile_source(),
            ColorProfileSource::Cicp(Cicp::BT2100_PQ)
        );
    }

    #[test]
    fn color_context_pq_transfer() {
        assert_eq!(
            ColorContext::from_cicp(Cicp::BT2100_PQ).transfer_function(),
            TransferFunction::Pq
        );
    }

    #[test]
    fn color_context_diffuse_white_defaults_none_and_round_trips() {
        // Unsignaled by default — consumers fall back to BT.2408 (203).
        assert_eq!(
            ColorContext::from_cicp(Cicp::BT2100_PQ).diffuse_white(),
            None
        );
        assert_eq!(ColorContext::from_icc([0u8; 4]).diffuse_white(), None);

        // The anchor round-trips through the builder, and is preserved by Clone.
        let ctx = ColorContext::from_cicp(Cicp::BT2100_PQ).with_diffuse_white(DiffuseWhite::BT2408);
        assert_eq!(ctx.diffuse_white(), Some(DiffuseWhite::BT2408));
        assert_eq!(ctx.clone().diffuse_white(), Some(DiffuseWhite::BT2408));

        // A custom anchor (e.g. a 100-nit SDR reference) carries through too.
        let custom =
            ColorContext::from_cicp(Cicp::BT2100_PQ).with_diffuse_white(DiffuseWhite::new(100.0));
        assert_eq!(custom.diffuse_white().map(DiffuseWhite::nits), Some(100.0));
    }

    #[test]
    fn color_context_eq_and_clone() {
        let a = ColorContext::from_cicp(Cicp::SRGB);
        let b = a.clone();
        assert_eq!(a, b);
        let c = ColorContext::from_icc(vec![1]);
        assert_ne!(a, c);
    }

    #[test]
    fn color_context_debug() {
        let ctx = ColorContext::from_cicp(Cicp::SRGB);
        let s = alloc::format!("{ctx:?}");
        assert!(s.contains("ColorContext"));
    }

    // --- ColorProfileSource coverage ---

    #[test]
    fn color_profile_source_named() {
        let src = ColorProfileSource::Named(NamedProfile::DisplayP3);
        assert_eq!(src, ColorProfileSource::Named(NamedProfile::DisplayP3));
        assert_ne!(src, ColorProfileSource::Named(NamedProfile::Srgb));
    }

    #[test]
    fn color_profile_source_cicp() {
        let src = ColorProfileSource::Cicp(Cicp::BT2100_PQ);
        assert_eq!(src, ColorProfileSource::Cicp(Cicp::BT2100_PQ));
    }

    #[test]
    fn color_profile_source_icc() {
        let data: &[u8] = &[1, 2, 3];
        let src = ColorProfileSource::Icc(data);
        assert_eq!(src, ColorProfileSource::Icc(&[1, 2, 3]));
    }

    #[test]
    fn color_profile_source_debug_clone() {
        let src = ColorProfileSource::Named(NamedProfile::Srgb);
        let s = alloc::format!("{src:?}");
        assert!(s.contains("Named"));
        let src2 = src.clone();
        assert_eq!(src, src2);
    }

    // --- NamedProfile coverage ---

    #[test]
    fn named_profile_all_variants_to_cicp() {
        assert!(NamedProfile::Srgb.to_cicp().is_some());
        assert!(NamedProfile::DisplayP3.to_cicp().is_some());
        assert!(NamedProfile::Bt2020.to_cicp().is_some());
        assert!(NamedProfile::Bt2020Pq.to_cicp().is_some());
        assert!(NamedProfile::LinearSrgb.to_cicp().is_some());
        assert!(NamedProfile::AdobeRgb.to_cicp().is_none());
    }

    #[test]
    fn named_profile_debug_clone_eq() {
        let p = NamedProfile::DisplayP3;
        let _ = alloc::format!("{p:?}");
        let p2 = p;
        assert_eq!(p, p2);
    }

    #[test]
    #[cfg(feature = "std")]
    fn named_profile_hash() {
        use core::hash::{Hash, Hasher};
        let p = NamedProfile::DisplayP3;
        let mut h = std::hash::DefaultHasher::new();
        p.hash(&mut h);
        let _ = h.finish();
    }

    // --- ColorProvenance coverage ---

    #[test]
    fn color_provenance_variants() {
        assert_ne!(ColorProvenance::Icc, ColorProvenance::Cicp);
        assert_ne!(ColorProvenance::GamaChrm, ColorProvenance::Assumed);
        let a = ColorProvenance::Icc;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn color_provenance_debug() {
        let p = ColorProvenance::Cicp;
        let _ = alloc::format!("{p:?}");
    }

    #[test]
    #[cfg(feature = "std")]
    fn color_provenance_hash() {
        use core::hash::{Hash, Hasher};
        let p = ColorProvenance::Cicp;
        let mut h = std::hash::DefaultHasher::new();
        p.hash(&mut h);
        let _ = h.finish();
    }

    // --- ColorOrigin coverage ---

    #[test]
    fn color_origin_from_icc() {
        let o = ColorOrigin::from_icc(vec![1, 2, 3]);
        assert!(o.icc.is_some());
        assert!(o.cicp.is_none());
        assert_eq!(o.provenance, ColorProvenance::Icc);
    }

    #[test]
    fn color_origin_from_cicp() {
        let o = ColorOrigin::from_cicp(Cicp::SRGB);
        assert!(o.icc.is_none());
        assert_eq!(o.cicp, Some(Cicp::SRGB));
        assert_eq!(o.provenance, ColorProvenance::Cicp);
    }

    #[test]
    fn color_origin_from_icc_and_cicp() {
        let o = ColorOrigin::from_icc_and_cicp(vec![10], Cicp::BT2100_PQ);
        assert!(o.icc.is_some());
        assert_eq!(o.cicp, Some(Cicp::BT2100_PQ));
        assert_eq!(o.provenance, ColorProvenance::Cicp);
    }

    #[test]
    fn color_origin_from_gama_chrm() {
        let o = ColorOrigin::from_gama_chrm();
        assert!(o.icc.is_none());
        assert!(o.cicp.is_none());
        assert_eq!(o.provenance, ColorProvenance::GamaChrm);
    }

    #[test]
    fn color_origin_assumed() {
        let o = ColorOrigin::assumed();
        assert!(o.icc.is_none());
        assert!(o.cicp.is_none());
        assert_eq!(o.provenance, ColorProvenance::Assumed);
    }

    #[test]
    fn color_origin_eq_clone_debug() {
        let a = ColorOrigin::from_cicp(Cicp::SRGB);
        let b = a.clone();
        assert_eq!(a, b);
        let _ = alloc::format!("{a:?}");
        let c = ColorOrigin::assumed();
        assert_ne!(a, c);
    }

    #[test]
    fn cie_xy_validates_finite_and_normalizes_negative_zero() {
        assert_eq!(CieXy::try_new(f32::NAN, 0.3), None);
        assert_eq!(CieXy::try_new(0.3, f32::INFINITY), None);
        // Imaginary primaries such as ACES AP0 blue (y = -0.077) are valid.
        let ap0_blue = CieXy::new(0.18, -0.077);
        assert_eq!(ap0_blue.to_tuple(), (0.18, -0.077));
        // -0.0 canonicalizes to +0.0 so bitwise Eq/Hash agree with numeric zero.
        let neg = CieXy::new(-0.0, -0.0);
        let pos = CieXy::new(0.0, 0.0);
        assert_eq!(neg, pos);
        assert_eq!(neg.x().to_bits(), 0);
        assert_eq!(neg.y().to_bits(), 0);
    }

    #[test]
    fn gamma_curve_distinguishes_oetf_from_eotf_exponent() {
        assert_eq!(GammaCurve::try_from_encoding_exponent(0.0), None);
        assert_eq!(GammaCurve::try_from_encoding_exponent(-0.5), None);
        assert_eq!(GammaCurve::try_from_eotf_exponent(f32::NAN), None);

        let jxl_22 = GammaCurve::from_eotf_exponent(2.2);
        assert!((jxl_22.encoding_exponent() - (1.0 / 2.2)).abs() < 1e-7);
        assert!((jxl_22.eotf_exponent() - 2.2).abs() < 1e-6);

        let png_gama = GammaCurve::from_encoding_exponent(45455.0 / 100_000.0);
        assert_eq!(png_gama.encoding_exponent(), 0.45455);
        assert!((png_gama.eotf_exponent() - 2.2).abs() < 1e-4);
    }

    #[test]
    fn parametric_color_cicp_and_primaries_transfer_resolution() {
        assert_eq!(ParametricColor::SRGB.to_cicp(), Some(Cicp::SRGB));
        assert_eq!(
            ParametricColor::SRGB.to_primaries_transfer(),
            Some((ColorPrimaries::Bt709, TransferFunction::Srgb))
        );
        assert_eq!(
            ParametricColor::DISPLAY_P3.to_cicp(),
            Some(Cicp::DISPLAY_P3)
        );
        assert_eq!(
            ParametricColor::DISPLAY_P3.to_primaries_transfer(),
            Some((ColorPrimaries::DisplayP3, TransferFunction::Srgb))
        );
        // Theatrical DCI-P3 has CICP (11, 17, 0, true), NOT DisplayP3 (D65).
        assert_eq!(
            ParametricColor::DCI_P3.to_cicp(),
            Some(Cicp::new(11, 17, 0, true))
        );
        assert_eq!(ParametricColor::DCI_P3.to_primaries_transfer(), None);

        // Adobe RGB maps to compact (AdobeRgb, Gamma22) but has no CICP code.
        assert_eq!(ParametricColor::ADOBE_RGB.to_cicp(), None);
        assert_eq!(
            ParametricColor::ADOBE_RGB.to_primaries_transfer(),
            Some((ColorPrimaries::AdobeRgb, TransferFunction::Gamma22))
        );

        // Custom primaries / non-D65 white point / arbitrary gamma do not collapse.
        let custom = ParametricColor::rgb(
            ParametricPrimaries::Custom {
                red: CieXy::new(0.7347, 0.2653),
                green: CieXy::new(0.0, 1.0),
                blue: CieXy::new(0.0001, -0.077),
            },
            WhitePoint::Custom(CieXy::new(0.32168, 0.33767)),
            ParametricTransfer::Gamma(GammaCurve::from_encoding_exponent(0.5)),
        )
        .with_rendering_intent(RenderingIntent::Perceptual);
        assert_eq!(custom.to_cicp(), None);
        assert_eq!(custom.to_primaries_transfer(), None);
    }

    #[test]
    fn color_context_parametric_and_canonical_rendering_intent() {
        let a = ColorContext::from_parametric(
            ParametricColor::DCI_P3.with_rendering_intent(RenderingIntent::Perceptual),
        );
        let b = ColorContext::from_parametric(ParametricColor::DCI_P3)
            .with_rendering_intent(RenderingIntent::Perceptual);
        assert_eq!(a, b);
        assert_eq!(a.rendering_intent(), Some(RenderingIntent::Perceptual));
        assert_eq!(
            a.parametric().map(|p| p.rendering_intent),
            Some(RenderingIntent::Perceptual)
        );

        let cicp_perceptual =
            ColorContext::from_cicp(Cicp::SRGB).with_rendering_intent(RenderingIntent::Perceptual);
        assert_eq!(cicp_perceptual.cicp(), Some(Cicp::SRGB));
        assert_eq!(
            cicp_perceptual.rendering_intent(),
            Some(RenderingIntent::Perceptual)
        );
        assert_ne!(cicp_perceptual, ColorContext::from_cicp(Cicp::SRGB));

        let origin = ColorOrigin::from_parametric(ParametricColor::DCI_P3);
        assert_eq!(origin.provenance, ColorProvenance::Parametric);
        assert_eq!(origin.parametric, Some(ParametricColor::DCI_P3));
        assert_eq!(origin.cicp, Some(Cicp::new(11, 17, 0, true)));
    }

    #[test]
    fn icc_parametric_curve_and_working_space_presets() {
        assert_eq!(IccParametricCurve::PROPHOTO_RGB.function_type(), 3);
        assert_eq!(
            IccParametricCurve::PROPHOTO_RGB.icc_params(),
            &[1.8, 1.0, 0.0, 1.0 / 16.0, 1.0 / 32.0]
        );
        // -0.0 canonicalizes to +0.0 for Eq/Hash.
        let neg_zero =
            IccParametricCurve::try_from_icc_para(3, &[1.8, 1.0, -0.0, 1.0 / 16.0, 1.0 / 32.0])
                .unwrap();
        assert_eq!(neg_zero, IccParametricCurve::PROPHOTO_RGB);
        // Invalid parameter counts, NaN, g <= 0, and a == 0 for types 1/2 reject cleanly.
        assert_eq!(IccParametricCurve::try_from_icc_para(0, &[1.8, 1.0]), None);
        assert_eq!(IccParametricCurve::try_from_icc_para(5, &[1.0]), None);
        assert_eq!(IccParametricCurve::try_type0_gamma(0.0), None);
        assert_eq!(IccParametricCurve::try_type0_gamma(f32::NAN), None);
        assert_eq!(IccParametricCurve::try_type1_cie122(2.2, 0.0, 0.0), None);
        assert_eq!(
            IccParametricCurve::try_type2_iec61966_3(2.2, 0.0, 0.0, 0.0),
            None
        );
        assert!(IccParametricCurve::try_type4(2.4, 0.9, 0.1, 0.08, 0.04, 0.01, -0.01).is_some());

        // Working-space presets and CICP round-trip for BT.601 525/625.
        assert_eq!(
            ParametricColor::LINEAR_SRGB.to_cicp(),
            Some(Cicp::new(1, 8, 0, true))
        );
        assert_eq!(
            ParametricColor::LINEAR_SRGB.to_primaries_transfer(),
            Some((ColorPrimaries::Bt709, TransferFunction::Linear))
        );
        assert_eq!(ParametricColor::PROPHOTO_RGB.white_point, WhitePoint::D50);
        assert_eq!(ParametricColor::ACES_CG.white_point, WhitePoint::D60);
        assert_eq!(ParametricColor::ACES_2065_1.white_point, WhitePoint::D60);
        assert_eq!(WhitePoint::C.cie_xy(), CieXy::C);
        assert_eq!(ParametricColor::PROPHOTO_RGB.to_cicp(), None);
        assert_eq!(ParametricColor::ACES_CG.to_cicp(), None);

        for code in [1u8, 5, 6, 9, 11, 12] {
            let cicp = Cicp::new(code, 1, 0, true);
            let p = ParametricColor::from_cicp(cicp).unwrap();
            assert_eq!(p.to_cicp(), Some(cicp));
        }
    }
}

/// Bytes aligned for every sample type a [`Color`] can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C, align(4))]
struct ColorSamples([u8; Color::MAX_BYTES]);

const _: () = assert!(core::mem::size_of::<Color>() == 24);

/// One color, and what its samples mean.
///
/// Holds one pixel of the descriptor's format, so `0.5` in linear Display P3 and `128` in sRGB
/// are different colors that both stay exact until converted. `zenpixels_convert::ColorConvertExt`
/// converts one to any other descriptor.
///
/// The bytes are aligned for every sample type, so casting [`as_bytes`](Self::as_bytes) to `f32`
/// or `u16` samples can't fail and doesn't copy. Comparison is bitwise: `0.0` and `-0.0` differ.
///
/// Colors with an ICC profile aren't representable, because a profile can't be `Copy`.
///
/// # Examples
///
/// ```
/// use zenpixels::{Color, ColorPrimaries, PixelDescriptor};
///
/// let red = Color::srgb8(255, 0, 0);
/// assert_eq!(red.as_bytes(), [255, 0, 0]);
///
/// // Display P3's red, in linear light.
/// let descriptor = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::DisplayP3);
/// let p3_red = Color::try_new(descriptor, bytemuck::bytes_of(&[1.0f32, 0.0, 0.0])).unwrap();
/// let samples: &[f32] = bytemuck::cast_slice(p3_red.as_bytes());
/// assert_eq!(samples, [1.0, 0.0, 0.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Color {
    samples: ColorSamples,
    descriptor: PixelDescriptor,
}

impl Color {
    /// The largest pixel a color holds: four `f32` samples.
    const MAX_BYTES: usize = 16;

    /// Whether one pixel fits, in bytes and in the storage's alignment.
    const fn fits(descriptor: PixelDescriptor) -> bool {
        descriptor.bytes_per_pixel() <= Self::MAX_BYTES && descriptor.min_alignment() <= 4
    }

    /// An opaque 8-bit sRGB color, like a CSS hex color.
    pub const fn srgb8(r: u8, g: u8, b: u8) -> Self {
        let mut samples = [0; Self::MAX_BYTES];
        samples[0] = r;
        samples[1] = g;
        samples[2] = b;
        Self {
            samples: ColorSamples(samples),
            descriptor: PixelDescriptor::RGB8_SRGB,
        }
    }

    /// Fully transparent 8-bit sRGB black, CSS's `transparent`.
    pub const TRANSPARENT: Self = Self::srgba8(0, 0, 0, 0);

    /// An 8-bit sRGB color with straight alpha, like a CSS `#rrggbbaa` color.
    pub const fn srgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        let mut samples = [0; Self::MAX_BYTES];
        samples[0] = r;
        samples[1] = g;
        samples[2] = b;
        samples[3] = a;
        Self {
            samples: ColorSamples(samples),
            descriptor: PixelDescriptor::RGBA8_SRGB,
        }
    }

    /// One pixel of `descriptor`'s format, as bytes in native sample order.
    ///
    /// # Errors
    ///
    /// [`BufferError::InvalidDimensions`](crate::buffer::BufferError::InvalidDimensions) when
    /// `pixel` isn't exactly one pixel, the format's pixels are wider than 16 bytes, or its samples
    /// need alignment above 4 bytes. [`AnyColor`] holds those.
    pub fn try_new(
        descriptor: PixelDescriptor,
        pixel: &[u8],
    ) -> Result<Self, crate::buffer::BufferError> {
        let bytes = descriptor.bytes_per_pixel();
        if pixel.len() != bytes || !Self::fits(descriptor) {
            return Err(crate::buffer::BufferError::InvalidDimensions);
        }
        let mut samples = [0; Self::MAX_BYTES];
        samples[..bytes].copy_from_slice(pixel);
        Ok(Self {
            samples: ColorSamples(samples),
            descriptor,
        })
    }

    /// What the samples mean.
    pub const fn descriptor(&self) -> PixelDescriptor {
        self.descriptor
    }

    /// The pixel, aligned to 4 bytes. Sample types wider than that need [`AnyColor`].
    pub fn as_bytes(&self) -> &[u8] {
        &self.samples.0[..self.descriptor.bytes_per_pixel()]
    }
}

/// A color in any format, including pixels wider than [`Color`] holds.
///
/// [`Color`] is `Copy` and holds up to 16 bytes, which covers every format at 8 and 16
/// bits and everything with four `f32` samples. It has to be `Copy` to live in
/// `ConvertOptions`. This type also holds wider pixels, such as RGBA in `f64`, on the
/// heap, and costs no more space. Comparison is bitwise.
///
/// ```
/// use zenpixels::{AnyColor, ChannelLayout, ChannelType, Color, PixelDescriptor, PixelFormat};
///
/// let small = AnyColor::from(Color::srgb8(255, 0, 0));
/// assert_eq!(small.as_bytes(), [255, 0, 0]);
/// assert!(small.as_color().is_some());
///
/// let format = PixelFormat::from_parts(ChannelType::F64, ChannelLayout::Gray, None).unwrap();
/// let descriptor = PixelDescriptor::from_pixel_format(format);
/// let wide = AnyColor::try_new(descriptor, &0.5f64.to_ne_bytes()).unwrap();
/// let samples: &[f64] = bytemuck::cast_slice(wide.as_bytes());
/// assert_eq!(samples, [0.5]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AnyColor(AnyColorRepr);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum AnyColorRepr {
    Compact(Color),
    Wide(alloc::boxed::Box<WideColor>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WideColor {
    descriptor: PixelDescriptor,
    /// Eight-byte words, so the bytes are aligned for every sample type.
    words: alloc::boxed::Box<[u64]>,
}

const _: () = assert!(core::mem::size_of::<AnyColor>() == core::mem::size_of::<Color>());

impl AnyColor {
    /// One pixel of `descriptor`'s format, as bytes in native sample order.
    ///
    /// # Errors
    ///
    /// [`BufferError::InvalidDimensions`](crate::buffer::BufferError::InvalidDimensions) when
    /// `pixel` isn't exactly one pixel, and
    /// [`BufferError::AllocationFailed`](crate::buffer::BufferError::AllocationFailed).
    pub fn try_new(
        descriptor: PixelDescriptor,
        pixel: &[u8],
    ) -> Result<Self, crate::buffer::BufferError> {
        use crate::buffer::BufferError;
        let bytes = descriptor.bytes_per_pixel();
        if pixel.len() != bytes {
            return Err(BufferError::InvalidDimensions);
        }
        if Color::fits(descriptor) {
            return Color::try_new(descriptor, pixel).map(Self::from);
        }
        let mut words = bytemuck::allocation::try_zeroed_vec::<u64>(bytes.div_ceil(8))
            .map_err(|()| BufferError::AllocationFailed)?;
        bytemuck::cast_slice_mut::<u64, u8>(&mut words)[..bytes].copy_from_slice(pixel);
        // `Box::new(wide)` stays infallible: a single fixed ~32 B control
        // allocation — not worth a fallible-`Box` workaround's extra copy.
        let wide = WideColor {
            descriptor,
            words: words.into_boxed_slice(),
        };
        Ok(Self(AnyColorRepr::Wide(alloc::boxed::Box::new(wide))))
    }

    /// What the samples mean.
    pub fn descriptor(&self) -> PixelDescriptor {
        match &self.0 {
            AnyColorRepr::Compact(color) => color.descriptor(),
            AnyColorRepr::Wide(wide) => wide.descriptor,
        }
    }

    /// The pixel, aligned for its sample type.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.0 {
            AnyColorRepr::Compact(color) => color.as_bytes(),
            AnyColorRepr::Wide(wide) => {
                &bytemuck::cast_slice(&wide.words)[..wide.descriptor.bytes_per_pixel()]
            }
        }
    }

    /// The compact form, or `None` when the pixel is wider than [`Color`] holds.
    pub fn as_color(&self) -> Option<Color> {
        match &self.0 {
            AnyColorRepr::Compact(color) => Some(*color),
            AnyColorRepr::Wide(_) => None,
        }
    }
}

impl From<Color> for AnyColor {
    fn from(color: Color) -> Self {
        Self(AnyColorRepr::Compact(color))
    }
}
