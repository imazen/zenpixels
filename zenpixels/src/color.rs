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

use crate::cicp::Cicp;
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
    /// CIE 1931 D50 ICC PCS illuminant `(0.3457, 0.3585)`.
    pub const D50: Self = Self {
        x: 0.3457,
        y: 0.3585,
    };
    /// SMPTE RP 431-2 DCI theater white point `(0.314, 0.351)`.
    pub const DCI: Self = Self { x: 0.314, y: 0.351 };
    /// Equal-energy illuminant E `(1/3, 1/3)`.
    pub const E: Self = Self {
        x: 1.0 / 3.0,
        y: 1.0 / 3.0,
    };

    /// Validate and construct a CIE 1931 `(x, y)` chromaticity coordinate.
    #[inline]
    #[must_use]
    pub const fn try_new(x: f32, y: f32) -> Option<Self> {
        if !x.is_finite() || !y.is_finite() || x <= 0.0 || y <= 0.0 || x >= 1.0 || y >= 1.0 {
            return None;
        }
        Some(Self { x, y })
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
            Self::E => Chromaticity::E.xy(),
            Self::Custom(c) => c.xy(),
        }
    }
}

/// Validated custom tristimulus RGB primaries + white point (JXL, OpenEXR, PNG `cHRM`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CustomPrimaries {
    r: Chromaticity,
    g: Chromaticity,
    b: Chromaticity,
    white: Chromaticity,
}

impl CustomPrimaries {
    /// Construct validated custom primaries, verifying finite positive chromaticities
    /// and a non-singular RGB → XYZ matrix.
    #[must_use]
    pub const fn try_new(
        r: (f32, f32),
        g: (f32, f32),
        b: (f32, f32),
        white: (f32, f32),
    ) -> Option<Self> {
        let (Some(r), Some(g), Some(b), Some(white)) = (
            Chromaticity::try_new(r.0, r.1),
            Chromaticity::try_new(g.0, g.1),
            Chromaticity::try_new(b.0, b.1),
            Chromaticity::try_new(white.0, white.1),
        ) else {
            return None;
        };
        let cp = Self { r, g, b, white };
        if rgb_to_xyz_from_xy(cp.r.xy(), cp.g.xy(), cp.b.xy(), cp.white.xy()).is_none() {
            return None;
        }
        Some(cp)
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Primaries {
    /// ITU-R BT.709 / sRGB primaries, D65 white (CICP 1).
    #[default]
    Bt709,
    /// SMPTE RP 431-2 DCI-P3 theater primaries, DCI white `(0.314, 0.351)` (CICP 11).
    DciP3,
    /// Apple Display P3 / DCI-P3 primaries, D65 white (CICP 12).
    DisplayP3,
    /// ITU-R BT.2020 / BT.2100 wide-gamut primaries, D65 white (CICP 9).
    Bt2020,
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

    /// CIE 1931 `(x, y)` coordinates of `(R, G, B)` primaries, or `None` for [`Self::Unknown`].
    #[allow(clippy::type_complexity)]
    #[inline]
    #[must_use]
    pub const fn chromaticity(self) -> Option<((f32, f32), (f32, f32), (f32, f32))> {
        match self {
            Self::Bt709 => Some(((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))),
            Self::DciP3 | Self::DisplayP3 => Some(((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))),
            Self::Bt2020 => Some(((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))),
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
            Self::Bt709 | Self::DisplayP3 | Self::Bt2020 | Self::AdobeRgb => Some(Self::WHITE_D65),
            Self::DciP3 => Some(Self::WHITE_DCI),
            Self::Custom(c) => Some(c.white.xy()),
            Self::Unknown => None,
        }
    }

    /// Map an ITU-T H.273 `ColourPrimaries` code to [`Primaries`].
    #[inline]
    #[must_use]
    pub const fn from_cicp(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Bt709),
            9 => Some(Self::Bt2020),
            11 => Some(Self::DciP3),
            12 => Some(Self::DisplayP3),
            _ => None,
        }
    }

    /// Map [`Primaries`] to an ITU-T H.273 `ColourPrimaries` code, when standardized.
    #[inline]
    #[must_use]
    pub const fn to_cicp(self) -> Option<u8> {
        match self {
            Self::Bt709 => Some(1),
            Self::Bt2020 => Some(9),
            Self::DciP3 => Some(11),
            Self::DisplayP3 => Some(12),
            Self::AdobeRgb | Self::Custom(_) | Self::Unknown => None,
        }
    }

    /// Linear-light luminance weights `[Yr, Yg, Yb]` derived from the primaries' RGB→XYZ `Y` row.
    #[inline]
    #[must_use]
    pub const fn luma_weights(self) -> [f32; 3] {
        match self {
            Self::Bt709 | Self::Unknown => [0.2126, 0.7152, 0.0722],
            Self::DisplayP3 => [0.228_974_56, 0.691_738_5, 0.079_286_91],
            Self::Bt2020 => [0.2627, 0.6780, 0.0593],
            Self::AdobeRgb => [0.297_344_98, 0.627_363_56, 0.075_291_46],
            Self::DciP3 | Self::Custom(_) => match self.rgb_to_xyz_f64() {
                Some(m) => [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32],
                None => [0.2126, 0.7152, 0.0722],
            },
        }
    }

    /// Linear RGB → CIE XYZ (`Y_white = 1.0`, relative to `self.white_point()`) 3×3 matrix in `f64`.
    #[must_use]
    pub const fn rgb_to_xyz_f64(self) -> Option<[[f64; 3]; 3]> {
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
    match (a, b) {
        (Primaries::Bt709, Primaries::Bt709)
        | (Primaries::DciP3, Primaries::DciP3)
        | (Primaries::DisplayP3, Primaries::DisplayP3)
        | (Primaries::Bt2020, Primaries::Bt2020)
        | (Primaries::AdobeRgb, Primaries::AdobeRgb) => true,
        (Primaries::Custom(x), Primaries::Custom(y)) => {
            x.r.x.to_bits() == y.r.x.to_bits()
                && x.r.y.to_bits() == y.r.y.to_bits()
                && x.g.x.to_bits() == y.g.x.to_bits()
                && x.g.y.to_bits() == y.g.y.to_bits()
                && x.b.x.to_bits() == y.b.x.to_bits()
                && x.b.y.to_bits() == y.b.y.to_bits()
                && x.white.x.to_bits() == y.white.x.to_bits()
                && x.white.y.to_bits() == y.white.y.to_bits()
        }
        _ => false,
    }
}

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
    if ry <= 0.0 || gy <= 0.0 || by <= 0.0 || wy <= 0.0 {
        return None;
    }
    let p = [
        [rx / ry, gx / gy, bx / by],
        [1.0, 1.0, 1.0],
        [
            (1.0 - rx - ry) / ry,
            (1.0 - gx - gy) / gy,
            (1.0 - bx - by) / by,
        ],
    ];
    let Some(p_inv) = inv3_f64(p) else {
        return None;
    };
    let w = [wx / wy, 1.0, (1.0 - wx - wy) / wy];
    let s = mul3_vec_f64(p_inv, w);
    Some([
        [p[0][0] * s[0], p[0][1] * s[1], p[0][2] * s[2]],
        [p[1][0] * s[0], p[1][1] * s[1], p[1][2] * s[2]],
        [p[2][0] * s[0], p[2][1] * s[1], p[2][2] * s[2]],
    ])
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

const fn inv3_f64(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let a = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let b = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * a + m[0][1] * b + m[0][2] * c;
    if det == 0.0 {
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
#[derive(Clone, Copy, Debug)]
pub struct Gamma(f32);

impl Gamma {
    /// DCI-P3 theater decoding gamma `2.6` (SMPTE RP 431-2).
    pub const DCI_2_6: Self = Self(2.6);
    /// Standard pure decoding gamma `2.2`.
    pub const GAMMA_2_2: Self = Self(2.2);
    /// Legacy decoding gamma `1.8`.
    pub const GAMMA_1_8: Self = Self(1.8);

    /// Construct a validated decoding gamma (`0.0 < exponent <= 16.0`, finite).
    #[inline]
    #[must_use]
    pub const fn try_new(exponent: f32) -> Option<Self> {
        if !exponent.is_finite() || exponent <= 0.0 || exponent > 16.0 {
            return None;
        }
        Some(Self(exponent))
    }

    /// Construct from an encoding gamma (e.g. JPEG XL `gamma` / PNG `gAMA`, `v = L^enc_gamma`),
    /// storing the reciprocal decoding exponent `1.0 / enc_gamma`.
    #[inline]
    #[must_use]
    pub const fn from_encoding_gamma(encoding_gamma: f32) -> Option<Self> {
        if !encoding_gamma.is_finite() || encoding_gamma <= 0.0 {
            return None;
        }
        Self::try_new(1.0 / encoding_gamma)
    }

    /// Decoding (EOTF) exponent `gamma` where `L = v^gamma`.
    #[inline]
    #[must_use]
    pub const fn exponent(self) -> f32 {
        self.0
    }
}

impl PartialEq for Gamma {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for Gamma {}
impl Hash for Gamma {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Electro-optical transfer function (non-linear curve between stored values and light).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Transfer {
    /// Linear light (`L = v`, CICP 8).
    Linear,
    /// IEC 61966-2-1 sRGB curve (CICP 13).
    #[default]
    Srgb,
    /// ITU-R BT.709 / BT.601 / BT.2020 camera curve (CICP 1).
    Bt709,
    /// Pure power curve `v^(563/256)` used by Adobe RGB (1998).
    Gamma22,
    /// Explicit validated pure decoding gamma `L = v^gamma` (JXL, DCI 2.6, PNG `gAMA`).
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

    /// Map an ITU-T H.273 `TransferCharacteristics` code to [`Transfer`].
    #[inline]
    #[must_use]
    pub const fn from_cicp(code: u8) -> Option<Self> {
        match code {
            1 | 6 | 14 | 15 => Some(Self::Bt709),
            4 => Some(Self::Gamma22),
            8 => Some(Self::Linear),
            13 => Some(Self::Srgb),
            16 => Some(Self::Pq),
            17 => Some(Self::Gamma(Gamma::DCI_2_6)),
            18 => Some(Self::Hlg(HlgMode::Scene)),
            _ => None,
        }
    }

    /// Map [`Transfer`] to an ITU-T H.273 `TransferCharacteristics` code, when standardized.
    #[inline]
    #[must_use]
    pub const fn to_cicp(self) -> Option<u8> {
        match self {
            Self::Bt709 => Some(1),
            Self::Gamma22 => Some(4),
            Self::Linear => Some(8),
            Self::Srgb => Some(13),
            Self::Pq => Some(16),
            Self::Gamma(g) if g.exponent().to_bits() == Gamma::DCI_2_6.exponent().to_bits() => {
                Some(17)
            }
            Self::Gamma(_) => None,
            Self::Hlg(_) => Some(18),
            Self::Unknown => None,
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
    /// Alpha channel is present and guaranteed fully opaque (`1.0` / `255` / `65535`).
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
    #[inline]
    #[must_use]
    pub fn with_primaries(mut self, primaries: Primaries) -> Self {
        if let Self::Rgb { primaries: p, .. } = &mut self {
            *p = primaries;
        }
        self
    }

    /// Set transfer on an [`Self::Rgb`] or [`Self::Gray`] color space.
    #[inline]
    #[must_use]
    pub fn with_transfer(mut self, transfer: Transfer) -> Self {
        match &mut self {
            Self::Rgb { transfer: t, .. } | Self::Gray { transfer: t, .. } => *t = transfer,
            _ => {}
        }
        self
    }

    /// Set signal range on an [`Self::Rgb`] or [`Self::Gray`] color space.
    #[inline]
    #[must_use]
    pub fn with_range(mut self, range: SignalRange) -> Self {
        match &mut self {
            Self::Rgb { range: r, .. } | Self::Gray { range: r, .. } => *r = range,
            _ => {}
        }
        self
    }

    /// Attach a [`DiffuseWhite`] luminance anchor to an [`Self::Rgb`], [`Self::Gray`], or [`Self::Xyz`] space.
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

    /// Return the attached [`DiffuseWhite`] anchor, if any.
    #[inline]
    #[must_use]
    pub const fn white(&self) -> Option<DiffuseWhite> {
        match *self {
            Self::Rgb { white, .. } | Self::Gray { white, .. } | Self::Xyz { white, .. } => white,
            _ => None,
        }
    }

    /// Project this color space to a standardized RGB [`Cicp`] tuple when possible.
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

/// Complete, validated description of an interleaved image: physical [`SampleLayout`] + algebraic [`ColorSpace`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ImageSpec {
    /// Physical byte/bit layout.
    pub layout: SampleLayout,
    /// Authoritative color meaning.
    pub color: ColorSpace,
}

impl ImageSpec {
    /// Construct an [`ImageSpec`] from a layout and color space.
    #[inline]
    #[must_use]
    pub const fn new(layout: SampleLayout, color: ColorSpace) -> Self {
        Self { layout, color }
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

    /// Replace primaries on an RGB spec.
    #[inline]
    #[must_use]
    pub fn with_primaries(mut self, primaries: Primaries) -> Self {
        self.color = self.color.with_primaries(primaries);
        self
    }

    /// Replace transfer on an RGB or Gray spec.
    #[inline]
    #[must_use]
    pub fn with_transfer(mut self, transfer: Transfer) -> Self {
        self.color = self.color.with_transfer(transfer);
        self
    }

    /// Replace signal range on an RGB or Gray spec.
    #[inline]
    #[must_use]
    pub fn with_range(mut self, range: SignalRange) -> Self {
        self.color = self.color.with_range(range);
        self
    }

    /// Replace alpha mode on this spec.
    #[inline]
    #[must_use]
    pub fn with_alpha(mut self, alpha: AlphaMode) -> Self {
        self.color = self.color.with_alpha(alpha);
        self
    }

    /// Attach a [`DiffuseWhite`] anchor.
    #[inline]
    #[must_use]
    pub fn with_white(mut self, white: DiffuseWhite) -> Self {
        self.color = self.color.with_white(white);
        self
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
                inverted, alpha, ..
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
            }
            ColorSpace::Icc {
                profile, channels, ..
            } => {
                let expected = (*channels as usize)
                    + usize::from(order.has_alpha_slot() || order.has_padding_slot());
                if order.channels() != expected {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
                #[cfg(feature = "icc")]
                if profile.len() >= 20 {
                    if let Some(sig) = crate::icc::profile_color_space(profile) {
                        let sig_ok = match sig {
                            IccColorSpace::Gray => *channels == 1,
                            IccColorSpace::Rgb | IccColorSpace::Lab | IccColorSpace::Xyz => {
                                *channels == 3
                            }
                            IccColorSpace::Cmyk => *channels == 4,
                        };
                        if !sig_ok {
                            return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                        }
                    }
                }
                #[cfg(not(feature = "icc"))]
                let _ = profile;
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
                        ..
                    } => {
                        *range == SignalRange::Full
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
                                    | Transfer::Hlg(_)
                            )
                    }
                    ColorSpace::Gray {
                        transfer, range, ..
                    } => {
                        *range == SignalRange::Full
                            && matches!(
                                transfer,
                                Transfer::Srgb
                                    | Transfer::Linear
                                    | Transfer::Bt709
                                    | Transfer::Pq
                                    | Transfer::Hlg(_)
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
}

impl GainMapMetadata {
    /// Construct and validate an exact ISO 21496-1 parameter set.
    ///
    /// Rejects non-positive `gamma`, `min > max`, or equal `base_hdr_headroom == alternate_hdr_headroom`
    /// (ISO 21496-1 §5.2.7 requires unequal headrooms).
    #[must_use]
    pub const fn try_new(
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
        };
        if m.is_valid() { Some(m) } else { None }
    }

    /// Construct a symmetric forward SDR-base → HDR-alternate parameter set with standard `1/64` offsets.
    #[must_use]
    pub fn symmetric(max_boost_stops: f32) -> Self {
        let max_f =
            SignedFraction::from_f32_approx(max_boost_stops).unwrap_or(SignedFraction::from_int(2));
        let alt_h = UnsignedFraction::from_f32_approx(max_boost_stops.max(1.0 / 64.0))
            .unwrap_or(UnsignedFraction::from_uint(2));
        Self {
            channels: GainMapChannels::Shared(GainMapComponent {
                min_gain_stops: SignedFraction::ZERO,
                max_gain_stops: max_f,
                gamma: UnsignedFraction::ONE,
                base_offset: SignedFraction::ONE_64TH,
                alternate_offset: SignedFraction::ONE_64TH,
            }),
            base_hdr_headroom: UnsignedFraction::ZERO,
            alternate_hdr_headroom: alt_h,
            application_space: GainMapColorSpace::Base,
        }
    }

    /// Validate all exact rational invariants (`gamma > 0`, `min <= max`, `H_base != H_alt`).
    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.channels.is_valid()
            && !matches!(
                self.base_hdr_headroom
                    .cmp_rational(self.alternate_hdr_headroom),
                Ordering::Equal
            )
    }

    /// Exact gain-map direction (`Forward` if `H_alt > H_base`, `Reverse` if `H_base > H_alt`), or `None` if equal.
    #[inline]
    #[must_use]
    pub const fn direction(self) -> Option<GainMapDirection> {
        match self
            .base_hdr_headroom
            .cmp_rational(self.alternate_hdr_headroom)
        {
            Ordering::Less => Some(GainMapDirection::Forward),
            Ordering::Greater => Some(GainMapDirection::Reverse),
            Ordering::Equal => None,
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
        }
    }
}

impl core::error::Error for ColorError {}
