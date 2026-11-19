#![allow(clippy::excessive_precision)]
//! Closed-form perceptual and polar/cylindrical color conversions:
//! - Polar <-> Cartesian (`Oklch <-> Oklab`, `Lch <-> Lab`, `Jzczhz <-> Jzazbz`)
//! - Safdar et al. (2017) HDR perceptual `Jzazbz` <-> linear / ST 2084 PQ RGB
//! - CIE 1976 D50 `L*a*b*` / `L*C*h°` <-> linear RGB (Bradford D65 <-> D50)
//! - Cylindrical `HSL` / `HSV` <-> RGB (`F32` degrees `[0, 360)`, integer `[0, 1)` turns)

use crate::gamut::GamutMatrix;
use crate::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};

#[inline]
fn primaries_to_xyz(primaries: ColorPrimaries) -> Option<&'static GamutMatrix> {
    GamutMatrix::to_xyz(primaries)
}

#[inline]
fn primaries_from_xyz(primaries: ColorPrimaries) -> Option<&'static GamutMatrix> {
    GamutMatrix::from_xyz(primaries)
}

/// Elementary step inside `ConvertStep::Perceptual`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PerceptualStep {
    /// `(L, a, b[, A]) -> (L, C, h°[, A])` on `F32` (`Oklab -> Oklch`, `Lab -> Lch`, `Jzazbz -> Jzczhz`).
    CartesianToPolar { target_layout: ChannelLayout },
    /// `(L, C, h°[, A]) -> (L, a, b[, A])` on `F32` (`Oklch -> Oklab`, `Lch -> Lab`, `Jzczhz -> Jzazbz`).
    PolarToCartesian { target_layout: ChannelLayout },
    /// Linear `F32` RGB(A) -> `Jzazbz`(A) `F32` (anchored by `pq_anchor_scale` when `anchored`, else `203` cd/m²).
    LinearRgbToJzazbz { has_alpha: bool, anchored: bool },
    /// `Jzazbz`(A) `F32` -> linear `F32` RGB(A).
    JzazbzToLinearRgb {
        has_alpha: bool,
        anchored: bool,
        primaries: ColorPrimaries,
    },
    /// ST 2084 PQ `F32` RGB(A) -> `Jzazbz`(A) `F32` (`0..10,000` cd/m² direct HDR <-> HDR).
    PqRgbToJzazbz { has_alpha: bool },
    /// `Jzazbz`(A) `F32` -> ST 2084 PQ `F32` RGB(A).
    JzazbzToPqRgb {
        has_alpha: bool,
        primaries: ColorPrimaries,
    },
    /// Linear `F32` RGB(A) (D65 primaries) -> CIE 1976 D50 `L*C*h°`(A) `F32`.
    LinearRgbToLch { has_alpha: bool },
    /// CIE 1976 D50 `L*C*h°`(A) `F32` -> linear `F32` RGB(A) (D65 primaries).
    LchToLinearRgb {
        has_alpha: bool,
        primaries: ColorPrimaries,
    },
    /// Encoded `F32` RGB(A) -> cylindrical `HSL`(A) `F32` (`H ∈ [0, 360), S, L ∈ [0, 1]`).
    RgbToHsl { has_alpha: bool },
    /// Cylindrical `HSL`(A) `F32` -> encoded `F32` RGB(A).
    HslToRgb { has_alpha: bool },
    /// Encoded `F32` RGB(A) -> cylindrical `HSV`(A) `F32` (`H ∈ [0, 360), S, V ∈ [0, 1]`).
    RgbToHsv { has_alpha: bool },
    /// Cylindrical `HSV`(A) `F32` -> encoded `F32` RGB(A).
    HsvToRgb { has_alpha: bool },
    /// Normalized `U8`/`U16` perceptual/polar samples -> canonical `F32` domain.
    DecodeIntToF32 {
        from_type: ChannelType,
        layout: ChannelLayout,
    },
    /// Canonical `F32` perceptual/polar domain -> normalized `U8`/`U16` samples.
    EncodeF32ToInt {
        to_type: ChannelType,
        layout: ChannelLayout,
    },
}

impl PerceptualStep {
    #[inline]
    pub(crate) const fn variant_name(&self) -> &'static str {
        match self {
            Self::CartesianToPolar { .. } => "CartesianToPolar",
            Self::PolarToCartesian { .. } => "PolarToCartesian",
            Self::LinearRgbToJzazbz { .. } => "LinearRgbToJzazbz",
            Self::JzazbzToLinearRgb { .. } => "JzazbzToLinearRgb",
            Self::PqRgbToJzazbz { .. } => "PqRgbToJzazbz",
            Self::JzazbzToPqRgb { .. } => "JzazbzToPqRgb",
            Self::LinearRgbToLch { .. } => "LinearRgbToLch",
            Self::LchToLinearRgb { .. } => "LchToLinearRgb",
            Self::RgbToHsl { .. } => "RgbToHsl",
            Self::HslToRgb { .. } => "HslToRgb",
            Self::RgbToHsv { .. } => "RgbToHsv",
            Self::HsvToRgb { .. } => "HsvToRgb",
            Self::DecodeIntToF32 { .. } => "DecodeIntToF32",
            Self::EncodeF32ToInt { .. } => "EncodeF32ToInt",
        }
    }

    #[inline]
    pub(crate) fn is_inverse_of(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::CartesianToPolar { .. }, Self::PolarToCartesian { .. })
            | (Self::PolarToCartesian { .. }, Self::CartesianToPolar { .. })
            | (Self::PqRgbToJzazbz { .. }, Self::JzazbzToPqRgb { .. })
            | (Self::JzazbzToPqRgb { .. }, Self::PqRgbToJzazbz { .. })
            | (Self::LinearRgbToLch { .. }, Self::LchToLinearRgb { .. })
            | (Self::LchToLinearRgb { .. }, Self::LinearRgbToLch { .. })
            | (Self::RgbToHsl { .. }, Self::HslToRgb { .. })
            | (Self::HslToRgb { .. }, Self::RgbToHsl { .. }) => true,
            (
                Self::LinearRgbToJzazbz { anchored: a1, .. },
                Self::JzazbzToLinearRgb { anchored: a2, .. },
            )
            | (
                Self::JzazbzToLinearRgb { anchored: a1, .. },
                Self::LinearRgbToJzazbz { anchored: a2, .. },
            ) => a1 == a2,
            _ => false,
        }
    }

    pub(crate) fn output_descriptor(&self, current: PixelDescriptor) -> PixelDescriptor {
        let mut out_primaries = current.primaries;
        let out = match *self {
            Self::CartesianToPolar { target_layout } | Self::PolarToCartesian { target_layout } => {
                PixelDescriptor::new(
                    ChannelType::F32,
                    target_layout,
                    current.alpha(),
                    current.transfer(),
                )
            }
            Self::LinearRgbToJzazbz { has_alpha, .. } | Self::PqRgbToJzazbz { has_alpha } => {
                out_primaries = ColorPrimaries::Bt709;
                let layout = if has_alpha {
                    ChannelLayout::JzazbzAlpha
                } else {
                    ChannelLayout::Jzazbz
                };
                PixelDescriptor::new(
                    ChannelType::F32,
                    layout,
                    current.alpha(),
                    TransferFunction::Unknown,
                )
            }
            Self::JzazbzToLinearRgb {
                has_alpha,
                primaries,
                ..
            }
            | Self::LchToLinearRgb {
                has_alpha,
                primaries,
            } => {
                out_primaries = primaries;
                let layout = if has_alpha {
                    ChannelLayout::Rgba
                } else {
                    ChannelLayout::Rgb
                };
                PixelDescriptor::new(
                    ChannelType::F32,
                    layout,
                    current.alpha(),
                    TransferFunction::Linear,
                )
            }
            Self::JzazbzToPqRgb {
                has_alpha,
                primaries,
            } => {
                out_primaries = primaries;
                let layout = if has_alpha {
                    ChannelLayout::Rgba
                } else {
                    ChannelLayout::Rgb
                };
                PixelDescriptor::new(
                    ChannelType::F32,
                    layout,
                    current.alpha(),
                    TransferFunction::Pq,
                )
            }
            Self::LinearRgbToLch { has_alpha } => {
                out_primaries = ColorPrimaries::Bt709;
                let layout = if has_alpha {
                    ChannelLayout::LchAlpha
                } else {
                    ChannelLayout::Lch
                };
                PixelDescriptor::new(
                    ChannelType::F32,
                    layout,
                    current.alpha(),
                    TransferFunction::Unknown,
                )
            }
            Self::RgbToHsl { has_alpha } => {
                let layout = if has_alpha {
                    ChannelLayout::HslAlpha
                } else {
                    ChannelLayout::Hsl
                };
                PixelDescriptor::new(
                    ChannelType::F32,
                    layout,
                    current.alpha(),
                    current.transfer(),
                )
            }
            Self::RgbToHsv { has_alpha } => {
                let layout = if has_alpha {
                    ChannelLayout::HsvAlpha
                } else {
                    ChannelLayout::Hsv
                };
                PixelDescriptor::new(
                    ChannelType::F32,
                    layout,
                    current.alpha(),
                    current.transfer(),
                )
            }
            Self::HslToRgb { has_alpha } | Self::HsvToRgb { has_alpha } => {
                let layout = if has_alpha {
                    ChannelLayout::Rgba
                } else {
                    ChannelLayout::Rgb
                };
                PixelDescriptor::new(
                    ChannelType::F32,
                    layout,
                    current.alpha(),
                    current.transfer(),
                )
            }
            Self::DecodeIntToF32 { layout, .. } => PixelDescriptor::new(
                ChannelType::F32,
                layout,
                current.alpha(),
                current.transfer(),
            ),
            Self::EncodeF32ToInt { to_type, layout } => {
                PixelDescriptor::new(to_type, layout, current.alpha(), current.transfer())
            }
        };
        out.with_primaries(out_primaries)
            .with_signal_range(current.signal_range)
    }
}

/// True when `layout` is one of the generic perceptual / polar layouts handled here.
#[inline]
pub(crate) const fn is_perceptual_or_polar_layout(layout: ChannelLayout) -> bool {
    matches!(
        layout,
        ChannelLayout::Oklch
            | ChannelLayout::OklchAlpha
            | ChannelLayout::Lch
            | ChannelLayout::LchAlpha
            | ChannelLayout::Jzazbz
            | ChannelLayout::JzazbzAlpha
            | ChannelLayout::Jzczhz
            | ChannelLayout::JzczhzAlpha
            | ChannelLayout::Hsl
            | ChannelLayout::HslAlpha
            | ChannelLayout::Hsv
            | ChannelLayout::HsvAlpha
    )
}

/// True when `(from, to)` is a direct `Lab <-> Lch` polar/Cartesian pair (which needs no CMS).
#[inline]
pub(crate) fn is_direct_lab_lch_pair(from: &PixelDescriptor, to: &PixelDescriptor) -> bool {
    matches!(
        (from.layout(), to.layout()),
        (ChannelLayout::Lab, ChannelLayout::Lch)
            | (ChannelLayout::Lch, ChannelLayout::Lab)
            | (ChannelLayout::LabAlpha, ChannelLayout::LchAlpha)
            | (ChannelLayout::LchAlpha, ChannelLayout::LabAlpha)
    ) && from.alpha() == to.alpha()
        && from.signal_range == to.signal_range
}

/// Compute an intermediate pivot descriptor for `plan_native` when `from` or `to`
/// uses `Oklch`, `Lch`, `Jzazbz`, `Jzczhz`, `Hsl`, or `Hsv`, or `None` when the
/// pair is already adjacent (`direct_step` handles it directly).
pub(crate) fn pivot(from: PixelDescriptor, to: PixelDescriptor) -> Option<PixelDescriptor> {
    if from == to {
        return None;
    }

    // 1. Widen/narrow non-F32 perceptual/polar endpoints to F32 first so all
    //    nonlinear/trigonometric math executes in F32.
    if is_perceptual_or_polar_layout(from.layout()) && from.channel_type() != ChannelType::F32 {
        let f32_from = PixelDescriptor::new(
            ChannelType::F32,
            from.layout(),
            from.alpha(),
            from.transfer(),
        )
        .with_primaries(from.primaries)
        .with_signal_range(from.signal_range);
        if f32_from != to {
            return Some(f32_from);
        }
        return None;
    }
    if is_perceptual_or_polar_layout(to.layout()) && to.channel_type() != ChannelType::F32 {
        let f32_to = PixelDescriptor::new(ChannelType::F32, to.layout(), to.alpha(), to.transfer())
            .with_primaries(to.primaries)
            .with_signal_range(to.signal_range);
        if f32_to != from {
            return Some(f32_to);
        }
        return None;
    }

    // Also widen F16 Lab to F32 when converting directly against Lch.
    if matches!(from.layout(), ChannelLayout::Lab | ChannelLayout::LabAlpha)
        && matches!(to.layout(), ChannelLayout::Lch | ChannelLayout::LchAlpha)
        && from.channel_type() == ChannelType::F16
        && from.alpha() == to.alpha()
    {
        return Some(
            PixelDescriptor::new(
                ChannelType::F32,
                from.layout(),
                from.alpha(),
                from.transfer(),
            )
            .with_primaries(from.primaries)
            .with_signal_range(from.signal_range),
        );
    }
    if matches!(to.layout(), ChannelLayout::Lab | ChannelLayout::LabAlpha)
        && matches!(from.layout(), ChannelLayout::Lch | ChannelLayout::LchAlpha)
        && to.channel_type() == ChannelType::F16
        && from.alpha() == to.alpha()
    {
        return Some(
            PixelDescriptor::new(ChannelType::F32, to.layout(), to.alpha(), to.transfer())
                .with_primaries(to.primaries)
                .with_signal_range(to.signal_range),
        );
    }

    // Direct single-step pairs need no pivot.
    if direct_step(from, to).is_some() {
        return None;
    }

    let valid_pivot = |p: PixelDescriptor| Some(p).filter(|d| *d != from && *d != to);

    // 2. Polar Oklch <-> Cartesian Oklab (preserving alpha presence).
    if matches!(
        from.layout(),
        ChannelLayout::Oklch | ChannelLayout::OklchAlpha
    ) && from.transfer() == TransferFunction::Unknown
    {
        let cart = if from.layout().has_alpha() {
            ChannelLayout::OklabA
        } else {
            ChannelLayout::Oklab
        };
        let p = PixelDescriptor::new(
            ChannelType::F32,
            cart,
            from.alpha(),
            TransferFunction::Unknown,
        )
        .with_primaries(from.primaries)
        .with_signal_range(from.signal_range);
        return valid_pivot(p);
    }
    if matches!(
        to.layout(),
        ChannelLayout::Oklch | ChannelLayout::OklchAlpha
    ) && to.transfer() == TransferFunction::Unknown
    {
        let cart = if to.layout().has_alpha() {
            ChannelLayout::OklabA
        } else {
            ChannelLayout::Oklab
        };
        let p = PixelDescriptor::new(
            ChannelType::F32,
            cart,
            to.alpha(),
            TransferFunction::Unknown,
        )
        .with_primaries(to.primaries)
        .with_signal_range(to.signal_range);
        return valid_pivot(p);
    }

    // 3. Polar Jzczhz <-> Cartesian Jzazbz.
    if matches!(
        from.layout(),
        ChannelLayout::Jzczhz | ChannelLayout::JzczhzAlpha
    ) && from.transfer() == TransferFunction::Unknown
    {
        let cart = if from.layout().has_alpha() {
            ChannelLayout::JzazbzAlpha
        } else {
            ChannelLayout::Jzazbz
        };
        let p = PixelDescriptor::new(
            ChannelType::F32,
            cart,
            from.alpha(),
            TransferFunction::Unknown,
        )
        .with_primaries(from.primaries)
        .with_signal_range(from.signal_range);
        return valid_pivot(p);
    }
    if matches!(
        to.layout(),
        ChannelLayout::Jzczhz | ChannelLayout::JzczhzAlpha
    ) && to.transfer() == TransferFunction::Unknown
    {
        let cart = if to.layout().has_alpha() {
            ChannelLayout::JzazbzAlpha
        } else {
            ChannelLayout::Jzazbz
        };
        let p = PixelDescriptor::new(
            ChannelType::F32,
            cart,
            to.alpha(),
            TransferFunction::Unknown,
        )
        .with_primaries(to.primaries)
        .with_signal_range(to.signal_range);
        return valid_pivot(p);
    }

    // 4. HSL / HSV <-> encoded F32 RGB(A) at the HSL/HSV descriptor's (primaries, transfer).
    let is_hsl_hsv = |l: ChannelLayout| {
        matches!(
            l,
            ChannelLayout::Hsl
                | ChannelLayout::HslAlpha
                | ChannelLayout::Hsv
                | ChannelLayout::HsvAlpha
        )
    };
    if is_hsl_hsv(from.layout()) {
        let rgb_layout = if from.layout().has_alpha() {
            ChannelLayout::Rgba
        } else {
            ChannelLayout::Rgb
        };
        let p = PixelDescriptor::new(ChannelType::F32, rgb_layout, from.alpha(), from.transfer())
            .with_primaries(from.primaries)
            .with_signal_range(from.signal_range);
        return valid_pivot(p);
    }
    if is_hsl_hsv(to.layout()) {
        let rgb_layout = if to.layout().has_alpha() {
            ChannelLayout::Rgba
        } else {
            ChannelLayout::Rgb
        };
        let p = PixelDescriptor::new(ChannelType::F32, rgb_layout, to.alpha(), to.transfer())
            .with_primaries(to.primaries)
            .with_signal_range(to.signal_range);
        return valid_pivot(p);
    }

    // 5. Jzazbz <-> PQ F32 RGB(A) (when the other side uses PQ) or Linear F32 RGB(A).
    //    Use the RGB side's primaries so wide-gamut Rec.2020/P3/AdobeRGB converts
    //    directly to/from XYZ D65 without routing through BT.709.
    if matches!(
        from.layout(),
        ChannelLayout::Jzazbz | ChannelLayout::JzazbzAlpha
    ) && from.transfer() == TransferFunction::Unknown
    {
        let rgb_layout = if from.layout().has_alpha() {
            ChannelLayout::Rgba
        } else {
            ChannelLayout::Rgb
        };
        let tf = if to.transfer() == TransferFunction::Pq {
            TransferFunction::Pq
        } else {
            TransferFunction::Linear
        };
        let rgb_primaries = if matches!(
            to.color_model(),
            crate::ColorModel::Rgb | crate::ColorModel::Gray
        ) {
            to.primaries
        } else {
            from.primaries
        };
        let p = PixelDescriptor::new(ChannelType::F32, rgb_layout, from.alpha(), tf)
            .with_primaries(rgb_primaries)
            .with_signal_range(from.signal_range);
        return valid_pivot(p);
    }
    if matches!(
        to.layout(),
        ChannelLayout::Jzazbz | ChannelLayout::JzazbzAlpha
    ) && to.transfer() == TransferFunction::Unknown
    {
        let rgb_layout = if to.layout().has_alpha() {
            ChannelLayout::Rgba
        } else {
            ChannelLayout::Rgb
        };
        let tf = if from.transfer() == TransferFunction::Pq {
            TransferFunction::Pq
        } else {
            TransferFunction::Linear
        };
        let rgb_primaries = if matches!(
            from.color_model(),
            crate::ColorModel::Rgb | crate::ColorModel::Gray
        ) {
            from.primaries
        } else {
            to.primaries
        };
        let p = PixelDescriptor::new(ChannelType::F32, rgb_layout, to.alpha(), tf)
            .with_primaries(rgb_primaries)
            .with_signal_range(to.signal_range);
        return valid_pivot(p);
    }

    // 6. CIE L*C*h° <-> Linear F32 RGB(A).
    if matches!(from.layout(), ChannelLayout::Lch | ChannelLayout::LchAlpha)
        && from.transfer() == TransferFunction::Unknown
    {
        let rgb_layout = if from.layout().has_alpha() {
            ChannelLayout::Rgba
        } else {
            ChannelLayout::Rgb
        };
        let rgb_primaries = if matches!(
            to.color_model(),
            crate::ColorModel::Rgb | crate::ColorModel::Gray
        ) {
            to.primaries
        } else {
            from.primaries
        };
        let p = PixelDescriptor::new(
            ChannelType::F32,
            rgb_layout,
            from.alpha(),
            TransferFunction::Linear,
        )
        .with_primaries(rgb_primaries)
        .with_signal_range(from.signal_range);
        return valid_pivot(p);
    }
    if matches!(to.layout(), ChannelLayout::Lch | ChannelLayout::LchAlpha)
        && to.transfer() == TransferFunction::Unknown
    {
        let rgb_layout = if to.layout().has_alpha() {
            ChannelLayout::Rgba
        } else {
            ChannelLayout::Rgb
        };
        let rgb_primaries = if matches!(
            from.color_model(),
            crate::ColorModel::Rgb | crate::ColorModel::Gray
        ) {
            from.primaries
        } else {
            to.primaries
        };
        let p = PixelDescriptor::new(
            ChannelType::F32,
            rgb_layout,
            to.alpha(),
            TransferFunction::Linear,
        )
        .with_primaries(rgb_primaries)
        .with_signal_range(to.signal_range);
        return valid_pivot(p);
    }

    None
}

/// Direct single-step perceptual/polar operation when `from` and `to` are adjacent.
/// Returns `Ok(None)` when neither side is an adjacent perceptual step, or `Err(())`
/// when they are adjacent perceptual endpoints whose primaries/alpha disallow conversion
/// (e.g. `ColorPrimaries::Unknown` on `Jzazbz` or `Lch`).
pub(crate) fn try_direct_step(
    from: PixelDescriptor,
    to: PixelDescriptor,
) -> Result<Option<super::ConvertStep>, ()> {
    if from.signal_range != to.signal_range {
        return Ok(None);
    }

    // Same-layout F16 <-> F32 or U8/U16 <-> F32 on perceptual/polar layouts.
    if from.layout() == to.layout()
        && is_perceptual_or_polar_layout(from.layout())
        && from.primaries == to.primaries
        && from.transfer() == to.transfer()
        && from.alpha() == to.alpha()
    {
        return match (from.channel_type(), to.channel_type()) {
            (ChannelType::F16, ChannelType::F32) => Ok(Some(super::ConvertStep::F16ToF32)),
            (ChannelType::F32, ChannelType::F16) => Ok(Some(super::ConvertStep::F32ToF16)),
            (ct @ (ChannelType::U8 | ChannelType::U16), ChannelType::F32) => Ok(Some(
                super::ConvertStep::Perceptual(PerceptualStep::DecodeIntToF32 {
                    from_type: ct,
                    layout: from.layout(),
                }),
            )),
            (ChannelType::F32, ct @ (ChannelType::U8 | ChannelType::U16)) => Ok(Some(
                super::ConvertStep::Perceptual(PerceptualStep::EncodeF32ToInt {
                    to_type: ct,
                    layout: to.layout(),
                }),
            )),
            _ => Err(()),
        };
    }

    if from.channel_type() != ChannelType::F32 || to.channel_type() != ChannelType::F32 {
        return Ok(None);
    }
    if from.alpha() != to.alpha() {
        return Ok(None);
    }
    // Premultiplied alpha cannot cross directly into/out of nonlinear perceptual/polar layouts;
    // only straight/opaque/undefined/none alpha is copied verbatim across model boundaries.
    if from.alpha() == Some(AlphaMode::Premultiplied) {
        return Ok(None);
    }

    let step = match (from.layout(), to.layout()) {
        // Polar <-> Cartesian pairs (`Oklab <-> Oklch` requires matching primaries;
        // `Lab <-> Lch` and `Jzazbz <-> Jzczhz` are absolute XYZ spaces).
        (ChannelLayout::Oklab, ChannelLayout::Oklch)
        | (ChannelLayout::OklabA, ChannelLayout::OklchAlpha)
            if from.primaries == to.primaries && from.transfer() == to.transfer() =>
        {
            PerceptualStep::CartesianToPolar {
                target_layout: to.layout(),
            }
        }
        (ChannelLayout::Lab, ChannelLayout::Lch)
        | (ChannelLayout::LabAlpha, ChannelLayout::LchAlpha)
        | (ChannelLayout::Jzazbz, ChannelLayout::Jzczhz)
        | (ChannelLayout::JzazbzAlpha, ChannelLayout::JzczhzAlpha)
            if from.transfer() == to.transfer() =>
        {
            PerceptualStep::CartesianToPolar {
                target_layout: to.layout(),
            }
        }
        (ChannelLayout::Oklch, ChannelLayout::Oklab)
        | (ChannelLayout::OklchAlpha, ChannelLayout::OklabA)
            if from.primaries == to.primaries && from.transfer() == to.transfer() =>
        {
            PerceptualStep::PolarToCartesian {
                target_layout: to.layout(),
            }
        }
        (ChannelLayout::Lch, ChannelLayout::Lab)
        | (ChannelLayout::LchAlpha, ChannelLayout::LabAlpha)
        | (ChannelLayout::Jzczhz, ChannelLayout::Jzazbz)
        | (ChannelLayout::JzczhzAlpha, ChannelLayout::JzazbzAlpha)
            if from.transfer() == to.transfer() =>
        {
            PerceptualStep::PolarToCartesian {
                target_layout: to.layout(),
            }
        }

        // Jzazbz <-> Linear / PQ RGB(A) (requires known RGB->XYZ primaries on the RGB side).
        (
            ChannelLayout::Rgb | ChannelLayout::Rgba,
            ChannelLayout::Jzazbz | ChannelLayout::JzazbzAlpha,
        ) if from.layout().has_alpha() == to.layout().has_alpha()
            && to.transfer() == TransferFunction::Unknown =>
        {
            if primaries_to_xyz(from.primaries).is_none() {
                return Err(());
            }
            match from.transfer() {
                TransferFunction::Linear => PerceptualStep::LinearRgbToJzazbz {
                    has_alpha: from.layout().has_alpha(),
                    anchored: false,
                },
                TransferFunction::Pq => PerceptualStep::PqRgbToJzazbz {
                    has_alpha: from.layout().has_alpha(),
                },
                _ => return Ok(None),
            }
        }
        (
            ChannelLayout::Jzazbz | ChannelLayout::JzazbzAlpha,
            ChannelLayout::Rgb | ChannelLayout::Rgba,
        ) if from.layout().has_alpha() == to.layout().has_alpha()
            && from.transfer() == TransferFunction::Unknown =>
        {
            if primaries_from_xyz(to.primaries).is_none() {
                return Err(());
            }
            match to.transfer() {
                TransferFunction::Linear => PerceptualStep::JzazbzToLinearRgb {
                    has_alpha: to.layout().has_alpha(),
                    anchored: false,
                    primaries: to.primaries,
                },
                TransferFunction::Pq => PerceptualStep::JzazbzToPqRgb {
                    has_alpha: to.layout().has_alpha(),
                    primaries: to.primaries,
                },
                _ => return Ok(None),
            }
        }

        // Lch <-> Linear RGB(A) (requires known RGB->XYZ primaries on the RGB side).
        (
            ChannelLayout::Rgb | ChannelLayout::Rgba,
            ChannelLayout::Lch | ChannelLayout::LchAlpha,
        ) if from.layout().has_alpha() == to.layout().has_alpha()
            && from.transfer() == TransferFunction::Linear
            && to.transfer() == TransferFunction::Unknown =>
        {
            if primaries_to_xyz(from.primaries).is_none() {
                return Err(());
            }
            PerceptualStep::LinearRgbToLch {
                has_alpha: from.layout().has_alpha(),
            }
        }
        (
            ChannelLayout::Lch | ChannelLayout::LchAlpha,
            ChannelLayout::Rgb | ChannelLayout::Rgba,
        ) if from.layout().has_alpha() == to.layout().has_alpha()
            && from.transfer() == TransferFunction::Unknown
            && to.transfer() == TransferFunction::Linear =>
        {
            if primaries_from_xyz(to.primaries).is_none() {
                return Err(());
            }
            PerceptualStep::LchToLinearRgb {
                has_alpha: to.layout().has_alpha(),
                primaries: to.primaries,
            }
        }

        // HSL / HSV <-> encoded RGB(A) at matching transfer & primaries.
        (
            ChannelLayout::Rgb | ChannelLayout::Rgba,
            ChannelLayout::Hsl | ChannelLayout::HslAlpha,
        ) if from.primaries == to.primaries
            && from.layout().has_alpha() == to.layout().has_alpha()
            && from.transfer() == to.transfer() =>
        {
            PerceptualStep::RgbToHsl {
                has_alpha: from.layout().has_alpha(),
            }
        }
        (
            ChannelLayout::Hsl | ChannelLayout::HslAlpha,
            ChannelLayout::Rgb | ChannelLayout::Rgba,
        ) if from.primaries == to.primaries
            && from.layout().has_alpha() == to.layout().has_alpha()
            && from.transfer() == to.transfer() =>
        {
            PerceptualStep::HslToRgb {
                has_alpha: to.layout().has_alpha(),
            }
        }
        (
            ChannelLayout::Rgb | ChannelLayout::Rgba,
            ChannelLayout::Hsv | ChannelLayout::HsvAlpha,
        ) if from.primaries == to.primaries
            && from.layout().has_alpha() == to.layout().has_alpha()
            && from.transfer() == to.transfer() =>
        {
            PerceptualStep::RgbToHsv {
                has_alpha: from.layout().has_alpha(),
            }
        }
        (
            ChannelLayout::Hsv | ChannelLayout::HsvAlpha,
            ChannelLayout::Rgb | ChannelLayout::Rgba,
        ) if from.primaries == to.primaries
            && from.layout().has_alpha() == to.layout().has_alpha()
            && from.transfer() == to.transfer() =>
        {
            PerceptualStep::HsvToRgb {
                has_alpha: to.layout().has_alpha(),
            }
        }

        _ => return Ok(None),
    };
    Ok(Some(super::ConvertStep::Perceptual(step)))
}

#[inline]
fn direct_step(from: PixelDescriptor, to: PixelDescriptor) -> Option<super::ConvertStep> {
    try_direct_step(from, to).ok().flatten()
}

// ---------------------------------------------------------------------------
// Safdar et al. (2017) Jzazbz constants & scalar math
// ---------------------------------------------------------------------------

const JZ_B: f64 = 1.15;
const JZ_G: f64 = 0.66;
const JZ_C1: f64 = 3424.0 / 4096.0;
const JZ_C2: f64 = 2413.0 / 128.0;
const JZ_C3: f64 = 2392.0 / 128.0;
const JZ_N: f64 = 2610.0 / 16384.0;
const JZ_P: f64 = 1.7 * 2523.0 / 32.0;
const JZ_D: f64 = -0.56;
const JZ_D0: f64 = 1.6295499532821566e-11;

const JZ_LMS_FROM_XYP: [[f64; 3]; 3] = [
    [0.41478972, 0.579999, 0.0146480],
    [-0.20151000, 1.120649, 0.0531008],
    [-0.01660080, 0.264800, 0.6684799],
];

const JZ_XYP_FROM_LMS: [[f64; 3]; 3] = [
    [1.9242264357876071, -1.0047923125953659, 0.0376514040306180],
    [0.3503167620949992, 0.7264811939316554, -0.0653844229480850],
    [-0.0909828109828476, -0.3127282905230740, 1.5227665613052608],
];

const JZ_IAB_FROM_LMSP: [[f64; 3]; 3] = [
    [0.5, 0.5, 0.0],
    [3.524000, -4.066708, 0.542708],
    [0.199076, 1.096799, -1.295875],
];

const JZ_LMSP_FROM_IAB: [[f64; 3]; 3] = [
    [1.0, 0.1386050432715393, 0.0580473161561189],
    [1.0, -0.1386050432715393, -0.0580473161561189],
    [1.0, -0.0960192420263189, -0.8118918960560388],
];

/// `JZ_C1^JZ_P` — exact ST 2084 / Safdar PQ offset at zero cone response (`u == 0`).
const JZ_PQ_ZERO: f64 = 3.7035226210190005e-11;

#[inline]
fn jz_pq_pos(u: f64) -> f64 {
    let t = libm::pow(u / 10_000.0, JZ_N);
    libm::pow((JZ_C1 + JZ_C2 * t) / (1.0 + JZ_C3 * t), JZ_P)
}

#[inline]
fn jz_pq_inv_pos(up: f64) -> f64 {
    let q = libm::pow(up.max(0.0), 1.0 / JZ_P);
    let num = (q - JZ_C1).max(0.0);
    let den = (JZ_C2 - JZ_C3 * q).max(1e-30);
    libm::pow(num / den, 1.0 / JZ_N) * 10_000.0
}

/// Safdar et al. (2017) modified PQ curve, reflected odd-symmetrically around
/// `(0, JZ_PQ_ZERO)` so sub-black / extended-range cone values remain strictly
/// monotonic, pole-free, and losslessly invertible.
#[inline]
fn jz_pq_fwd(u: f64) -> f64 {
    if u >= 0.0 {
        jz_pq_pos(u)
    } else {
        2.0 * JZ_PQ_ZERO - jz_pq_pos(-u)
    }
}

#[inline]
fn jz_pq_inv(up: f64) -> f64 {
    if up >= JZ_PQ_ZERO {
        jz_pq_inv_pos(up)
    } else {
        -jz_pq_inv_pos(2.0 * JZ_PQ_ZERO - up)
    }
}

#[inline]
fn xyz_d65_nits_to_jzazbz([x, y, z]: [f64; 3]) -> [f32; 3] {
    let xp = JZ_B * x - (JZ_B - 1.0) * z;
    let yp = JZ_G * y - (JZ_G - 1.0) * x;
    let l = JZ_LMS_FROM_XYP[0][0] * xp + JZ_LMS_FROM_XYP[0][1] * yp + JZ_LMS_FROM_XYP[0][2] * z;
    let m = JZ_LMS_FROM_XYP[1][0] * xp + JZ_LMS_FROM_XYP[1][1] * yp + JZ_LMS_FROM_XYP[1][2] * z;
    let s = JZ_LMS_FROM_XYP[2][0] * xp + JZ_LMS_FROM_XYP[2][1] * yp + JZ_LMS_FROM_XYP[2][2] * z;
    let lp = jz_pq_fwd(l);
    let mp = jz_pq_fwd(m);
    let sp = jz_pq_fwd(s);
    let iz = JZ_IAB_FROM_LMSP[0][0] * lp + JZ_IAB_FROM_LMSP[0][1] * mp;
    let az =
        JZ_IAB_FROM_LMSP[1][0] * lp + JZ_IAB_FROM_LMSP[1][1] * mp + JZ_IAB_FROM_LMSP[1][2] * sp;
    let bz =
        JZ_IAB_FROM_LMSP[2][0] * lp + JZ_IAB_FROM_LMSP[2][1] * mp + JZ_IAB_FROM_LMSP[2][2] * sp;
    let denom = (1.0 + JZ_D * iz).max(1e-30);
    let jz = ((1.0 + JZ_D) * iz) / denom - JZ_D0;
    [jz as f32, az as f32, bz as f32]
}

#[inline]
fn jzazbz_to_xyz_d65_nits([jz, az, bz]: [f32; 3]) -> [f64; 3] {
    let j = f64::from(jz) + JZ_D0;
    let denom = (1.0 + JZ_D - JZ_D * j).max(1e-30);
    let iz = j / denom;
    let a = f64::from(az);
    let b = f64::from(bz);
    let lp = JZ_LMSP_FROM_IAB[0][0] * iz + JZ_LMSP_FROM_IAB[0][1] * a + JZ_LMSP_FROM_IAB[0][2] * b;
    let mp = JZ_LMSP_FROM_IAB[1][0] * iz + JZ_LMSP_FROM_IAB[1][1] * a + JZ_LMSP_FROM_IAB[1][2] * b;
    let sp = JZ_LMSP_FROM_IAB[2][0] * iz + JZ_LMSP_FROM_IAB[2][1] * a + JZ_LMSP_FROM_IAB[2][2] * b;
    let l = jz_pq_inv(lp);
    let m = jz_pq_inv(mp);
    let s = jz_pq_inv(sp);
    let xp = JZ_XYP_FROM_LMS[0][0] * l + JZ_XYP_FROM_LMS[0][1] * m + JZ_XYP_FROM_LMS[0][2] * s;
    let yp = JZ_XYP_FROM_LMS[1][0] * l + JZ_XYP_FROM_LMS[1][1] * m + JZ_XYP_FROM_LMS[1][2] * s;
    let z = JZ_XYP_FROM_LMS[2][0] * l + JZ_XYP_FROM_LMS[2][1] * m + JZ_XYP_FROM_LMS[2][2] * s;
    let x = (xp + (JZ_B - 1.0) * z) / JZ_B;
    let y = (yp + (JZ_G - 1.0) * x) / JZ_G;
    [x, y, z]
}

// ---------------------------------------------------------------------------
// CIE 1976 D50 L*a*b* / L*C*h° (Bradford D65 <-> D50 per CSS Color 4 §11)
// ---------------------------------------------------------------------------

const D65_TO_D50_BRADFORD: [[f64; 3]; 3] = [
    [1.0479297925449969, 0.0229468706016097, -0.0501922662892053],
    [0.0296278087700560, 0.9904344267538800, -0.0170737990634188],
    [-0.0092430406462045, 0.0150551914902982, 0.7518742814281372],
];

const D50_TO_D65_BRADFORD: [[f64; 3]; 3] = [
    [0.9554734214880750, -0.0230984549487647, 0.0632592432005707],
    [-0.0283697093338639, 1.0099953980813041, 0.0210414411919173],
    [0.0123140148644820, -0.0205076492988990, 1.3303659262421239],
];

const D50_WHITE: [f64; 3] = [0.3457 / 0.3585, 1.0, (1.0 - 0.3457 - 0.3585) / 0.3585];

const CIE_EPSILON: f64 = 216.0 / 24389.0;
const CIE_KAPPA: f64 = 24389.0 / 27.0;

#[inline]
fn lab_f(t: f64) -> f64 {
    if t > CIE_EPSILON {
        libm::cbrt(t)
    } else {
        (CIE_KAPPA * t + 16.0) / 116.0
    }
}

#[inline]
fn lab_f_inv(f: f64) -> f64 {
    let f3 = f * f * f;
    if f3 > CIE_EPSILON {
        f3
    } else {
        (116.0 * f - 16.0) / CIE_KAPPA
    }
}

#[inline(always)]
fn xyz_d65_to_lch_spec<const FINITE_BOUNDED: bool>([x, y, z]: [f64; 3]) -> [f32; 3] {
    let xd = D65_TO_D50_BRADFORD[0][0] * x
        + D65_TO_D50_BRADFORD[0][1] * y
        + D65_TO_D50_BRADFORD[0][2] * z;
    let yd = D65_TO_D50_BRADFORD[1][0] * x
        + D65_TO_D50_BRADFORD[1][1] * y
        + D65_TO_D50_BRADFORD[1][2] * z;
    let zd = D65_TO_D50_BRADFORD[2][0] * x
        + D65_TO_D50_BRADFORD[2][1] * y
        + D65_TO_D50_BRADFORD[2][2] * z;
    let fx = lab_f(xd / D50_WHITE[0]);
    let fy = lab_f(yd / D50_WHITE[1]);
    let fz = lab_f(zd / D50_WHITE[2]);
    let l = (116.0 * fy - 16.0) as f32;
    let a = (500.0 * (fx - fy)) as f32;
    let b = (200.0 * (fy - fz)) as f32;
    cart_to_polar_spec::<FINITE_BOUNDED>([l, a, b])
}

#[inline(always)]
fn lch_to_xyz_d65_spec<const FINITE: bool, const NONNEG: bool>(lch: [f32; 3]) -> [f64; 3] {
    let [l, a, b] = polar_to_cart_spec::<FINITE, NONNEG>(lch);
    let fy = (f64::from(l) + 16.0) / 116.0;
    let fx = fy + f64::from(a) / 500.0;
    let fz = fy - f64::from(b) / 200.0;
    let xd = lab_f_inv(fx) * D50_WHITE[0];
    let yd = lab_f_inv(fy) * D50_WHITE[1];
    let zd = lab_f_inv(fz) * D50_WHITE[2];
    [
        D50_TO_D65_BRADFORD[0][0] * xd
            + D50_TO_D65_BRADFORD[0][1] * yd
            + D50_TO_D65_BRADFORD[0][2] * zd,
        D50_TO_D65_BRADFORD[1][0] * xd
            + D50_TO_D65_BRADFORD[1][1] * yd
            + D50_TO_D65_BRADFORD[1][2] * zd,
        D50_TO_D65_BRADFORD[2][0] * xd
            + D50_TO_D65_BRADFORD[2][1] * yd
            + D50_TO_D65_BRADFORD[2][2] * zd,
    ]
}

// ---------------------------------------------------------------------------
// Polar <-> Cartesian & Cylindrical HSL / HSV scalar math
// ---------------------------------------------------------------------------

#[inline]
fn wrap_degrees(deg: f32) -> f32 {
    wrap_degrees_spec::<false>(deg)
}

#[inline(always)]
fn wrap_degrees_spec<const FINITE: bool>(deg: f32) -> f32 {
    if !FINITE && !deg.is_finite() {
        return 0.0;
    }
    let m = deg - 360.0 * libm::floorf(deg / 360.0);
    if m >= 360.0 || m < 0.0 { 0.0 } else { m }
}

#[inline(always)]
pub(crate) fn cart_to_polar_spec<const FINITE_BOUNDED: bool>([l, a, b]: [f32; 3]) -> [f32; 3] {
    let c = libm::hypotf(a, b);
    let h = if (!FINITE_BOUNDED && !c.is_finite()) || c <= 1e-7 {
        0.0
    } else {
        wrap_degrees_spec::<FINITE_BOUNDED>(libm::atan2f(b, a).to_degrees())
    };
    [l, c, h]
}

#[inline(always)]
pub(crate) fn polar_to_cart_spec<const FINITE: bool, const NONNEG: bool>(
    [l, c, h]: [f32; 3],
) -> [f32; 3] {
    let c_pos = if FINITE && NONNEG {
        c
    } else if FINITE {
        if c > 0.0 { c } else { 0.0 }
    } else if c.is_finite() && c > 0.0 {
        c
    } else {
        0.0
    };
    if c_pos == 0.0 || (!FINITE && !h.is_finite()) {
        [l, 0.0, 0.0]
    } else {
        let rad = wrap_degrees_spec::<FINITE>(h).to_radians();
        [l, c_pos * libm::cosf(rad), c_pos * libm::sinf(rad)]
    }
}

#[inline(always)]
pub(crate) fn rgb_to_hsl_spec<const UNIT_NONNEG: bool, const FINITE_BOUNDED: bool>(
    [r, g, b]: [f32; 3],
) -> [f32; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let l = 0.5 * (max + min);
    if (!UNIT_NONNEG && !FINITE_BOUNDED && !d.is_finite()) || d <= 1e-7 {
        return [0.0, 0.0, l];
    }
    // W3C CSS Color 4 §7.2 / §19: signed denominator `min(L, 1 - L)`; out-of-gamut
    // colors with `L < 0` or `L > 1` yield negative `s`, normalized by rotating
    // hue by 180° and taking `|s|` so `hsl_to_rgb` inverts extended-range RGB.
    let half_span = l.min(1.0 - l);
    if UNIT_NONNEG {
        if half_span <= 1e-7 {
            return [0.0, 0.0, l];
        }
        let s = (max - l) / half_span;
        if s <= 1e-7 {
            return [0.0, 0.0, l];
        }
        let h = rgb_hue_deg_spec::<true>(r, g, b, max, d);
        return [h, s, l];
    }
    if libm::fabsf(half_span) <= 1e-7 {
        return [0.0, 0.0, l];
    }
    let mut s = (max - l) / half_span;
    let mut h = rgb_hue_deg_spec::<FINITE_BOUNDED>(r, g, b, max, d);
    if s < 0.0 {
        s = -s;
        h = wrap_degrees_spec::<FINITE_BOUNDED>(h + 180.0);
    }
    if s <= 1e-7 {
        return [0.0, 0.0, l];
    }
    [h, s, l]
}

#[inline(always)]
pub(crate) fn hsl_to_rgb_spec<const FINITE: bool>([h, s, l]: [f32; 3]) -> [f32; 3] {
    let s_clean = if FINITE || s.is_finite() { s } else { 0.0 };
    if libm::fabsf(s_clean) <= 1e-7 || (!FINITE && !h.is_finite()) {
        return [l, l, l];
    }
    let h_deg = wrap_degrees_spec::<FINITE>(h);
    let a = s_clean * l.min(1.0 - l);
    let f = |n: f32| -> f32 {
        let k_raw = n + h_deg / 30.0;
        let k = k_raw - 12.0 * libm::floorf(k_raw / 12.0);
        let seg = (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0);
        l - a * seg
    };
    [f(0.0), f(8.0), f(4.0)]
}

#[inline(always)]
pub(crate) fn rgb_to_hsv_spec<const UNIT_NONNEG: bool, const FINITE_BOUNDED: bool>(
    [r, g, b]: [f32; 3],
) -> [f32; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let v = max;
    if UNIT_NONNEG {
        if d <= 1e-7 || max <= 1e-7 {
            return [0.0, 0.0, v];
        }
        let s = d / max;
        let h = rgb_hue_deg_spec::<true>(r, g, b, max, d);
        return [h, s, v];
    }
    if (!FINITE_BOUNDED && !d.is_finite()) || d <= 1e-7 || libm::fabsf(max) <= 1e-7 {
        return [0.0, 0.0, v];
    }
    let s = d / max;
    let h = rgb_hue_deg_spec::<FINITE_BOUNDED>(r, g, b, max, d);
    [h, s, v]
}

#[inline(always)]
pub(crate) fn hsv_to_rgb_spec<const FINITE: bool>([h, s, v]: [f32; 3]) -> [f32; 3] {
    let s_clean = if FINITE || s.is_finite() { s } else { 0.0 };
    if libm::fabsf(s_clean) <= 1e-7 || (!FINITE && !h.is_finite()) {
        return [v, v, v];
    }
    let h6 = wrap_degrees_spec::<FINITE>(h) / 60.0;
    let sector = libm::floorf(h6) as i32;
    let frac = h6 - (sector as f32);
    let p = v * (1.0 - s_clean);
    let q = v * (1.0 - s_clean * frac);
    let t = v * (1.0 - s_clean * (1.0 - frac));
    match sector {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

#[inline(always)]
fn rgb_hue_deg_spec<const FINITE: bool>(r: f32, g: f32, b: f32, max: f32, d: f32) -> f32 {
    let raw = if max == r {
        (g - b) / d
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    wrap_degrees_spec::<FINITE>(raw * 60.0)
}

// ---------------------------------------------------------------------------
// Row kernel execution
// ---------------------------------------------------------------------------

#[inline]
fn map_f32_3_or_4(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    has_alpha: bool,
    mut f: impl FnMut([f32; 3]) -> [f32; 3],
) {
    if has_alpha {
        let src_f: &[[f32; 4]] = bytemuck::cast_slice(&src[..width * 16]);
        let dst_f: &mut [[f32; 4]] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
        for (out, &[c0, c1, c2, a]) in dst_f.iter_mut().zip(src_f) {
            let [d0, d1, d2] = f([c0, c1, c2]);
            *out = [d0, d1, d2, a];
        }
    } else {
        let src_f: &[[f32; 3]] = bytemuck::cast_slice(&src[..width * 12]);
        let dst_f: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
        for (out, &px) in dst_f.iter_mut().zip(src_f) {
            *out = f(px);
        }
    }
}

/// Scales for normalized `U8`/`U16` storage (`[0, 1]` integer codes <-> canonical `F32` domain).
/// Cyclic hue angles use `is_cyclic_hue = true` (`[0, 360)` wrapped); signed opponent channels
/// (`Jzazbz` `az, bz`) use offset `-0.5` (`[-0.5, +0.5]`).
#[inline]
const fn int_channel_scales(layout: ChannelLayout) -> ([f32; 3], [f32; 3], [bool; 3]) {
    match layout {
        ChannelLayout::Hsl
        | ChannelLayout::HslAlpha
        | ChannelLayout::Hsv
        | ChannelLayout::HsvAlpha => ([360.0, 1.0, 1.0], [0.0, 0.0, 0.0], [true, false, false]),
        ChannelLayout::Oklch
        | ChannelLayout::OklchAlpha
        | ChannelLayout::Jzczhz
        | ChannelLayout::JzczhzAlpha => ([1.0, 0.5, 360.0], [0.0, 0.0, 0.0], [false, false, true]),
        ChannelLayout::Lch | ChannelLayout::LchAlpha => {
            ([100.0, 150.0, 360.0], [0.0, 0.0, 0.0], [false, false, true])
        }
        ChannelLayout::Jzazbz | ChannelLayout::JzazbzAlpha => {
            ([1.0, 1.0, 1.0], [0.0, -0.5, -0.5], [false, false, false])
        }
        _ => ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [false, false, false]),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_row(
    step: PerceptualStep,
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    from_primaries: ColorPrimaries,
    to_primaries: ColorPrimaries,
    pq_scale: f32,
    in_trust: super::StepFloatTrust,
) {
    use zenpixels::sample::FloatFacts;
    let cf = in_trust.color_facts;
    let finite = cf.contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE);
    let finite_bounded = finite && in_trust.color_bounded;
    let unit_nonneg = cf.contains(FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE);
    let nonneg = cf.contains(FloatFacts::NON_NEGATIVE);

    match step {
        PerceptualStep::CartesianToPolar { target_layout } => {
            if finite_bounded {
                map_f32_3_or_4(
                    src,
                    dst,
                    width,
                    target_layout.has_alpha(),
                    cart_to_polar_spec::<true>,
                );
            } else {
                map_f32_3_or_4(
                    src,
                    dst,
                    width,
                    target_layout.has_alpha(),
                    cart_to_polar_spec::<false>,
                );
            }
        }
        PerceptualStep::PolarToCartesian { target_layout } => {
            let ha = target_layout.has_alpha();
            match (finite, nonneg) {
                (true, true) => {
                    map_f32_3_or_4(src, dst, width, ha, polar_to_cart_spec::<true, true>)
                }
                (true, false) => {
                    map_f32_3_or_4(src, dst, width, ha, polar_to_cart_spec::<true, false>)
                }
                _ => map_f32_3_or_4(src, dst, width, ha, polar_to_cart_spec::<false, false>),
            }
        }
        PerceptualStep::LinearRgbToJzazbz {
            has_alpha,
            anchored,
        } => {
            let sdr_nits = if anchored {
                f64::from(pq_scale) * 10_000.0
            } else {
                203.0_f64
            };
            let m = *primaries_to_xyz(from_primaries)
                .expect("validated RGB->XYZ primaries")
                .rows();
            map_f32_3_or_4(src, dst, width, has_alpha, |[r, g, b]| {
                let x = (f64::from(m[0][0]) * f64::from(r)
                    + f64::from(m[0][1]) * f64::from(g)
                    + f64::from(m[0][2]) * f64::from(b))
                    * sdr_nits;
                let y = (f64::from(m[1][0]) * f64::from(r)
                    + f64::from(m[1][1]) * f64::from(g)
                    + f64::from(m[1][2]) * f64::from(b))
                    * sdr_nits;
                let z = (f64::from(m[2][0]) * f64::from(r)
                    + f64::from(m[2][1]) * f64::from(g)
                    + f64::from(m[2][2]) * f64::from(b))
                    * sdr_nits;
                xyz_d65_nits_to_jzazbz([x, y, z])
            });
        }
        PerceptualStep::JzazbzToLinearRgb {
            has_alpha,
            anchored,
            ..
        } => {
            let sdr_nits = if anchored {
                f64::from(pq_scale) * 10_000.0
            } else {
                203.0_f64
            };
            let m = *primaries_from_xyz(to_primaries)
                .expect("validated XYZ->RGB primaries")
                .rows();
            let inv_nits = 1.0 / sdr_nits.max(1e-30);
            map_f32_3_or_4(src, dst, width, has_alpha, |jab| {
                let [x_nits, y_nits, z_nits] = jzazbz_to_xyz_d65_nits(jab);
                let (x, y, z) = (x_nits * inv_nits, y_nits * inv_nits, z_nits * inv_nits);
                [
                    (f64::from(m[0][0]) * x + f64::from(m[0][1]) * y + f64::from(m[0][2]) * z)
                        as f32,
                    (f64::from(m[1][0]) * x + f64::from(m[1][1]) * y + f64::from(m[1][2]) * z)
                        as f32,
                    (f64::from(m[2][0]) * x + f64::from(m[2][1]) * y + f64::from(m[2][2]) * z)
                        as f32,
                ]
            });
        }
        PerceptualStep::PqRgbToJzazbz { has_alpha } => {
            let m = *primaries_to_xyz(from_primaries)
                .expect("validated RGB->XYZ primaries")
                .rows();
            map_f32_3_or_4(src, dst, width, has_alpha, |[pr, pg, pb]| {
                let r = f64::from(super::pq_eotf(pr.clamp(0.0, 1.0))) * 10_000.0;
                let g = f64::from(super::pq_eotf(pg.clamp(0.0, 1.0))) * 10_000.0;
                let b = f64::from(super::pq_eotf(pb.clamp(0.0, 1.0))) * 10_000.0;
                let x = f64::from(m[0][0]) * r + f64::from(m[0][1]) * g + f64::from(m[0][2]) * b;
                let y = f64::from(m[1][0]) * r + f64::from(m[1][1]) * g + f64::from(m[1][2]) * b;
                let z = f64::from(m[2][0]) * r + f64::from(m[2][1]) * g + f64::from(m[2][2]) * b;
                xyz_d65_nits_to_jzazbz([x, y, z])
            });
        }
        PerceptualStep::JzazbzToPqRgb { has_alpha, .. } => {
            let m = *primaries_from_xyz(to_primaries)
                .expect("validated XYZ->RGB primaries")
                .rows();
            map_f32_3_or_4(src, dst, width, has_alpha, |jab| {
                let [x_nits, y_nits, z_nits] = jzazbz_to_xyz_d65_nits(jab);
                let (x, y, z) = (x_nits / 10_000.0, y_nits / 10_000.0, z_nits / 10_000.0);
                let r = (f64::from(m[0][0]) * x + f64::from(m[0][1]) * y + f64::from(m[0][2]) * z)
                    as f32;
                let g = (f64::from(m[1][0]) * x + f64::from(m[1][1]) * y + f64::from(m[1][2]) * z)
                    as f32;
                let b = (f64::from(m[2][0]) * x + f64::from(m[2][1]) * y + f64::from(m[2][2]) * z)
                    as f32;
                [
                    super::pq_oetf(r.clamp(0.0, 1.0)),
                    super::pq_oetf(g.clamp(0.0, 1.0)),
                    super::pq_oetf(b.clamp(0.0, 1.0)),
                ]
            });
        }
        PerceptualStep::LinearRgbToLch { has_alpha } => {
            let m = *primaries_to_xyz(from_primaries)
                .expect("validated RGB->XYZ primaries")
                .rows();
            let to_xyz = |[r, g, b]: [f32; 3]| -> [f64; 3] {
                [
                    f64::from(m[0][0]) * f64::from(r)
                        + f64::from(m[0][1]) * f64::from(g)
                        + f64::from(m[0][2]) * f64::from(b),
                    f64::from(m[1][0]) * f64::from(r)
                        + f64::from(m[1][1]) * f64::from(g)
                        + f64::from(m[1][2]) * f64::from(b),
                    f64::from(m[2][0]) * f64::from(r)
                        + f64::from(m[2][1]) * f64::from(g)
                        + f64::from(m[2][2]) * f64::from(b),
                ]
            };
            if finite_bounded {
                map_f32_3_or_4(src, dst, width, has_alpha, |rgb| {
                    xyz_d65_to_lch_spec::<true>(to_xyz(rgb))
                });
            } else {
                map_f32_3_or_4(src, dst, width, has_alpha, |rgb| {
                    xyz_d65_to_lch_spec::<false>(to_xyz(rgb))
                });
            }
        }
        PerceptualStep::LchToLinearRgb { has_alpha, .. } => {
            let m = *primaries_from_xyz(to_primaries)
                .expect("validated XYZ->RGB primaries")
                .rows();
            let from_xyz = |[x, y, z]: [f64; 3]| -> [f32; 3] {
                [
                    (f64::from(m[0][0]) * x + f64::from(m[0][1]) * y + f64::from(m[0][2]) * z)
                        as f32,
                    (f64::from(m[1][0]) * x + f64::from(m[1][1]) * y + f64::from(m[1][2]) * z)
                        as f32,
                    (f64::from(m[2][0]) * x + f64::from(m[2][1]) * y + f64::from(m[2][2]) * z)
                        as f32,
                ]
            };
            if finite && nonneg {
                map_f32_3_or_4(src, dst, width, has_alpha, |lch| {
                    from_xyz(lch_to_xyz_d65_spec::<true, true>(lch))
                });
            } else if finite {
                map_f32_3_or_4(src, dst, width, has_alpha, |lch| {
                    from_xyz(lch_to_xyz_d65_spec::<true, false>(lch))
                });
            } else {
                map_f32_3_or_4(src, dst, width, has_alpha, |lch| {
                    from_xyz(lch_to_xyz_d65_spec::<false, false>(lch))
                });
            }
        }
        PerceptualStep::RgbToHsl { has_alpha } => {
            if unit_nonneg {
                map_f32_3_or_4(src, dst, width, has_alpha, rgb_to_hsl_spec::<true, true>);
            } else if finite_bounded {
                map_f32_3_or_4(src, dst, width, has_alpha, rgb_to_hsl_spec::<false, true>);
            } else {
                map_f32_3_or_4(src, dst, width, has_alpha, rgb_to_hsl_spec::<false, false>);
            }
        }
        PerceptualStep::HslToRgb { has_alpha } => {
            if finite {
                map_f32_3_or_4(src, dst, width, has_alpha, hsl_to_rgb_spec::<true>);
            } else {
                map_f32_3_or_4(src, dst, width, has_alpha, hsl_to_rgb_spec::<false>);
            }
        }
        PerceptualStep::RgbToHsv { has_alpha } => {
            if unit_nonneg {
                map_f32_3_or_4(src, dst, width, has_alpha, rgb_to_hsv_spec::<true, true>);
            } else if finite_bounded {
                map_f32_3_or_4(src, dst, width, has_alpha, rgb_to_hsv_spec::<false, true>);
            } else {
                map_f32_3_or_4(src, dst, width, has_alpha, rgb_to_hsv_spec::<false, false>);
            }
        }
        PerceptualStep::HsvToRgb { has_alpha } => {
            if finite {
                map_f32_3_or_4(src, dst, width, has_alpha, hsv_to_rgb_spec::<true>);
            } else {
                map_f32_3_or_4(src, dst, width, has_alpha, hsv_to_rgb_spec::<false>);
            }
        }
        PerceptualStep::DecodeIntToF32 { from_type, layout } => {
            let ch = layout.channels();
            let (scales, offsets, cyclic) = int_channel_scales(layout);
            let dst_f: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * ch * 4]);
            match from_type {
                ChannelType::U8 => {
                    for (out_px, in_px) in dst_f
                        .chunks_exact_mut(ch)
                        .zip(src[..width * ch].chunks_exact(ch))
                    {
                        for i in 0..3 {
                            let norm = f32::from(in_px[i]) / 255.0;
                            let v = norm * scales[i] + offsets[i];
                            out_px[i] = if cyclic[i] { wrap_degrees(v) } else { v };
                        }
                        if ch == 4 {
                            out_px[3] = f32::from(in_px[3]) / 255.0;
                        }
                    }
                }
                ChannelType::U16 => {
                    let src_u16: &[u16] = bytemuck::cast_slice(&src[..width * ch * 2]);
                    for (out_px, in_px) in dst_f.chunks_exact_mut(ch).zip(src_u16.chunks_exact(ch))
                    {
                        for i in 0..3 {
                            let norm = f32::from(in_px[i]) / 65535.0;
                            let v = norm * scales[i] + offsets[i];
                            out_px[i] = if cyclic[i] { wrap_degrees(v) } else { v };
                        }
                        if ch == 4 {
                            out_px[3] = f32::from(in_px[3]) / 65535.0;
                        }
                    }
                }
                _ => unreachable!("DecodeIntToF32 only accepts U8 or U16"),
            }
        }
        PerceptualStep::EncodeF32ToInt { to_type, layout } => {
            let ch = layout.channels();
            let (scales, offsets, cyclic) = int_channel_scales(layout);
            let src_f: &[f32] = bytemuck::cast_slice(&src[..width * ch * 4]);
            match to_type {
                ChannelType::U8 => {
                    for (out_px, in_px) in dst[..width * ch]
                        .chunks_exact_mut(ch)
                        .zip(src_f.chunks_exact(ch))
                    {
                        for i in 0..3 {
                            let raw = if cyclic[i] {
                                wrap_degrees(in_px[i])
                            } else {
                                in_px[i]
                            };
                            let norm = ((raw - offsets[i]) / scales[i]).clamp(0.0, 1.0);
                            out_px[i] = libm::roundf(norm * 255.0) as u8;
                        }
                        if ch == 4 {
                            out_px[3] = libm::roundf(in_px[3].clamp(0.0, 1.0) * 255.0) as u8;
                        }
                    }
                }
                ChannelType::U16 => {
                    let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * ch * 2]);
                    for (out_px, in_px) in dst_u16.chunks_exact_mut(ch).zip(src_f.chunks_exact(ch))
                    {
                        for i in 0..3 {
                            let raw = if cyclic[i] {
                                wrap_degrees(in_px[i])
                            } else {
                                in_px[i]
                            };
                            let norm = ((raw - offsets[i]) / scales[i]).clamp(0.0, 1.0);
                            out_px[i] = libm::roundf(norm * 65535.0) as u16;
                        }
                        if ch == 4 {
                            out_px[3] = libm::roundf(in_px[3].clamp(0.0, 1.0) * 65535.0) as u16;
                        }
                    }
                }
                _ => unreachable!("EncodeF32ToInt only accepts U8 or U16"),
            }
        }
    }
}
