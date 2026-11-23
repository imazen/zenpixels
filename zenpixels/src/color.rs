//! Algebraic color-space model, image specification, provenance, and gain-map metadata.
//!
//! [`ColorSpace`] is an algebraic enum so mutually exclusive states — such as
//! Oklab carrying BT.709 primaries, CMYK carrying a PQ transfer curve, or an
//! authoritative ICC profile paired with conflicting primaries — are
//! **unrepresentable by construction**.

use alloc::sync::Arc;
use core::fmt;
use whereat::At;

use crate::cicp::Cicp;
use crate::hdr::{ContentLightLevel, DiffuseWhite, HlgRendering, MasteringDisplay};
use crate::layout::{ChannelOrder, InteropTarget, LayoutError, SampleLayout, SampleType};

/// Chromaticity primaries for tristimulus RGB color spaces (all D65 white point).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Primaries {
    /// ITU-R BT.709 / sRGB primaries (CICP 1).
    #[default]
    Bt709,
    /// Apple Display P3 / DCI-P3 D65 primaries (CICP 12).
    DisplayP3,
    /// ITU-R BT.2020 / BT.2100 wide-gamut primaries (CICP 9).
    Bt2020,
    /// Adobe RGB (1998) primaries (no standard H.273 code).
    AdobeRgb,
    /// Unspecified / unknown primaries.
    Unknown,
}

impl Primaries {
    /// CIE 1931 D65 white point `(x, y)`.
    pub const WHITE_D65: (f32, f32) = (0.3127, 0.3290);

    /// CIE 1931 `(x, y)` coordinates of `(R, G, B)` primaries, or `None` for [`Self::Unknown`].
    #[allow(clippy::type_complexity)]
    #[inline]
    #[must_use]
    pub const fn chromaticity(self) -> Option<((f32, f32), (f32, f32), (f32, f32))> {
        match self {
            Self::Bt709 => Some(((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))),
            Self::DisplayP3 => Some(((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))),
            Self::Bt2020 => Some(((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))),
            Self::AdobeRgb => Some(((0.64, 0.33), (0.21, 0.71), (0.15, 0.06))),
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
            Self::DisplayP3 => Some(12),
            Self::AdobeRgb | Self::Unknown => None,
        }
    }

    /// Linear-light luminance weights `[Yr, Yg, Yb]` derived from the primaries' D65 RGB→XYZ `Y` row.
    #[inline]
    #[must_use]
    pub const fn luma_weights(self) -> [f32; 3] {
        match self {
            Self::Bt709 | Self::Unknown => [0.2126, 0.7152, 0.0722],
            Self::DisplayP3 => [0.228_974_56, 0.691_738_5, 0.079_286_91],
            Self::Bt2020 => [0.2627, 0.6780, 0.0593],
            Self::AdobeRgb => [0.297_344_98, 0.627_363_56, 0.075_291_46],
        }
    }

    /// Linear RGB → CIE XYZ (D65, `Y_white = 1.0`) 3×3 matrix in `f64`, computed from exact chromaticities.
    #[must_use]
    pub const fn rgb_to_xyz_f64(self) -> Option<[[f64; 3]; 3]> {
        let Some(((rx, ry), (gx, gy), (bx, by))) = self.chromaticity() else {
            return None;
        };
        let (rx, ry) = (rx as f64, ry as f64);
        let (gx, gy) = (gx as f64, gy as f64);
        let (bx, by) = (bx as f64, by as f64);
        let (wx, wy) = (0.3127_f64, 0.3290_f64);

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

    /// Exact 3×3 linear-light gamut conversion matrix from `self` to `dst`.
    ///
    /// Returns the identity matrix when `self == dst` (even for known primaries),
    /// and `None` if either side is [`Self::Unknown`].
    #[must_use]
    pub const fn gamut_matrix_to(self, dst: Self) -> Option<[[f32; 3]; 3]> {
        let (Some(src_xyz), Some(dst_xyz)) = (self.rgb_to_xyz_f64(), dst.rgb_to_xyz_f64()) else {
            return None;
        };
        if matches!(
            (self, dst),
            (Self::Bt709, Self::Bt709)
                | (Self::DisplayP3, Self::DisplayP3)
                | (Self::Bt2020, Self::Bt2020)
                | (Self::AdobeRgb, Self::AdobeRgb)
        ) {
            return Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        }
        let Some(dst_inv) = inv3_f64(dst_xyz) else {
            return None;
        };
        let m = mul3_mat_f64(dst_inv, src_xyz);
        Some([
            [m[0][0] as f32, m[0][1] as f32, m[0][2] as f32],
            [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32],
            [m[2][0] as f32, m[2][1] as f32, m[2][2] as f32],
        ])
    }
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
            Self::Hlg(_) => Some(18),
            Self::Unknown => None,
        }
    }
}

/// Signal quantization range for integer samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SignalRange {
    /// Full range (`0..=255` in `U8`, `0..=65535` in `U16`).
    #[default]
    Full,
    /// Broadcast studio / limited swing (`16..=235` in `U8`, `4096..=60160` in `U16`).
    Narrow,
}

/// Interpretation of an image's alpha (or padding) channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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

    /// Attach a [`DiffuseWhite`] luminance anchor to an [`Self::Rgb`] or [`Self::Gray`] space.
    #[inline]
    #[must_use]
    pub fn with_white(mut self, white: DiffuseWhite) -> Self {
        match &mut self {
            Self::Rgb { white: w, .. } | Self::Gray { white: w, .. } => *w = Some(white),
            _ => {}
        }
        self
    }

    /// Return the attached [`DiffuseWhite`] anchor, if any.
    #[inline]
    #[must_use]
    pub const fn white(&self) -> Option<DiffuseWhite> {
        match *self {
            Self::Rgb { white, .. } | Self::Gray { white, .. } => white,
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
            ColorSpace::Gray { range, .. } => {
                if !order.is_gray_family() {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
                if *range == SignalRange::Narrow
                    && !matches!(sample, SampleType::U8 | SampleType::U16)
                {
                    return Err(whereat::at!(ColorError::NarrowRangeRequiresInteger));
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
            ColorSpace::Cmyk { inverted, .. } => {
                let want = if *inverted {
                    ChannelOrder::CmykInverted
                } else {
                    ChannelOrder::Cmyk
                };
                if order != want {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
                }
            }
            ColorSpace::Icc { channels, .. } => {
                let expected = (*channels as usize)
                    + usize::from(order.has_alpha_slot() || order.has_padding_slot());
                if order.channels() != expected {
                    return Err(whereat::at!(ColorError::ColorLayoutMismatch));
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

/// ISO 21496-1 / UltraHDR gain-map reconstruction parameters.
///
/// All boost and headroom values are expressed in log2 stops relative to SDR white.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GainMapMetadata {
    /// Minimum log2 gain per channel `[R, G, B]` (often `0.0` or negative).
    pub min_boost_stops: [f32; 3],
    /// Maximum log2 gain per channel `[R, G, B]` (e.g. `[2.3, 2.3, 2.3]` for ~5× / 1000-nit headroom).
    pub max_boost_stops: [f32; 3],
    /// Gain-map encoding gamma per channel `[R, G, B]` (default `[1.0; 3]`).
    pub gamma: [f32; 3],
    /// Linear offset added to base image before applying gain (`[1/64; 3]` typical).
    pub base_offset: [f32; 3],
    /// Linear offset subtracted from alternate (HDR) image (`[1/64; 3]` typical).
    pub alternate_offset: [f32; 3],
    /// Minimum display HDR headroom (log2 stops) where base rendition is shown unchanged.
    pub base_hdr_headroom: f32,
    /// Display HDR headroom (log2 stops) at which full gain is applied.
    pub alternate_hdr_headroom: f32,
    /// Whether gain application runs in the base image's primaries (true) or alternate primaries (false).
    pub use_base_color_space: bool,
}

impl GainMapMetadata {
    /// Construct a symmetric scalar gain-map parameter set with standard `1/64` offsets.
    #[must_use]
    pub const fn symmetric(max_boost_stops: f32) -> Self {
        Self {
            min_boost_stops: [0.0; 3],
            max_boost_stops: [max_boost_stops; 3],
            gamma: [1.0; 3],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: max_boost_stops,
            use_base_color_space: true,
        }
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
