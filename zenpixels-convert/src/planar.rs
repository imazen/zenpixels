//! Native multi-plane YCbCr / GBR / planar RGB / monochrome (`PlanarView` / `PlanarBuf`,
//! including 8/10/12-bit `BitPacking::P010` and `U10_LSB`, 4:4:4 / 4:2:2 / 4:2:0 / 4:4:0 / 4:1:1)
//! → strided interleaved `U8` / `U16` / `F32` (`RGB`, `RGBA`, `BGR`, `BGRA`, `RGBX`, `BGRX`,
//! `Gray`, `GrayAlpha`).
//!
//! All three entry points read source color interpretation identically from the [`PlanarView`]
//! (`color()`, `effective_range()`, `matrix_coefficients()`, `effective_alpha()`,
//! `chroma_siting()`, `crop_origin()`, `hdr()`, `origin()`, and `orientation()`):
//! - [`reconstruct_planar_to_buf`] allocates a tightly packed interleaved [`PixelBuf`] with
//!   caller-chosen [`SampleLayout`], deriving output [`ColorSpace`] and propagating `hdr()`,
//!   `origin()`, and `orientation()`.
//! - [`reconstruct_planar_into`] writes directly into a caller-provided strided
//!   [`PixelViewMut`] with **zero heap allocation**, validating that the destination's
//!   [`ColorSpace`] matches the source's derived color interpretation (no silent color
//!   conversion) and propagating `src.orientation()` onto `dst`.
//! - [`reconstruct_planar_f32`] allocates a full-range `RGBF32` / `RGBAF32` [`PixelBuf`]
//!   with caller-chosen [`PlanarReconstructionOptions`].

use whereat::At;
use zenpixels::{
    AlphaMode, ChannelOrder, ChromaSiting, ColorSpace, PixelBuf, PixelFormat, PixelRect,
    PixelViewMut, PlanarView, PlaneFormat, PlaneRole, PlaneView, Primaries, SampleLayout,
    SampleType, SignalRange,
};

use crate::error::{ConvertError, err_at};
use crate::ycbcr::YcbcrMatrix;

/// Chroma upsampling kernel for subsampled YCbCr planes (`4:2:0`, `4:2:2`, `4:4:0`, `4:1:1`).
///
/// Ignored for full-resolution (`1×1`) planar RGB, 4:4:4 YCbCr, and monochrome sources.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChromaUpsample {
    /// Nearest-neighbor sample replication honoring [`PlanarView::crop_origin`] phase.
    #[default]
    Nearest,
    /// Siting-exact separable bilinear / triangle interpolation honoring
    /// [`PlanarView::chroma_siting`] and [`PlanarView::crop_origin`].
    ///
    /// Produces exact 9:3:3:1 ("fancy" triangle) weights for 4:2:0 [`ChromaSiting::Center`],
    /// H.264/HEVC/AV1 `CSP_VERTICAL` weights for [`ChromaSiting::Left`], and BT.2020 co-sited
    /// weights for [`ChromaSiting::TopLeft`].
    Bilinear,
}

/// Configuration for planar reconstruction ([`reconstruct_planar_to_buf`],
/// [`reconstruct_planar_into`]).
///
/// Chroma resampling options apply only when a chroma axis is subsampled (`> 1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PlanarReconstructionOptions {
    /// Chroma upsampling kernel (`Nearest` default; `Bilinear` for smooth siting-exact upsampling).
    pub upsample: ChromaUpsample,
    /// Optional fallback [`ChromaSiting`] used when `upsample == ChromaUpsample::Bilinear`,
    /// a chroma axis is subsampled, and `src.chroma_siting() == ChromaSiting::Unknown`.
    ///
    /// When `None`, `Bilinear` refuses subsampled views with `ChromaSiting::Unknown` rather
    /// than silently guessing a siting.
    pub unknown_siting_fallback: Option<ChromaSiting>,
}

impl PlanarReconstructionOptions {
    /// Nearest-neighbor upsampling (`ChromaUpsample::Nearest`).
    #[inline]
    #[must_use]
    pub const fn nearest() -> Self {
        Self {
            upsample: ChromaUpsample::Nearest,
            unknown_siting_fallback: None,
        }
    }

    /// Siting-exact separable bilinear / triangle upsampling (`ChromaUpsample::Bilinear`).
    ///
    /// Refuses subsampled views with `ChromaSiting::Unknown` unless
    /// [`with_unknown_siting_fallback`](Self::with_unknown_siting_fallback) is set.
    #[inline]
    #[must_use]
    pub const fn bilinear() -> Self {
        Self {
            upsample: ChromaUpsample::Bilinear,
            unknown_siting_fallback: None,
        }
    }

    /// Select the chroma upsampling kernel.
    #[inline]
    #[must_use]
    pub const fn with_upsample(mut self, upsample: ChromaUpsample) -> Self {
        self.upsample = upsample;
        self
    }

    /// Provide an explicit fallback [`ChromaSiting`] when the source declares `ChromaSiting::Unknown`.
    #[inline]
    #[must_use]
    pub const fn with_unknown_siting_fallback(mut self, siting: ChromaSiting) -> Self {
        self.unknown_siting_fallback = Some(siting);
        self
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_plane(
    plane: &PlaneView<'_>,
    crop_x: u32,
    crop_y: u32,
    frame_w: u32,
    frame_h: u32,
    expected_components: u8,
    require_full_res: bool,
    allow_f32: bool,
) -> Result<(), At<ConvertError>> {
    let spec = plane.format();
    spec.validate().map_err(|e| err_at(e.into()))?;
    let sample_ok = matches!(
        spec.sample,
        SampleType::U8 | SampleType::U16 | SampleType::U32
    ) || (allow_f32 && spec.sample == SampleType::F32 && spec.packing.is_full());
    if !sample_ok {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    if spec.components != expected_components {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    if require_full_res && spec.subsampling != (1, 1) {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    let (ew, eh) = spec
        .cropped_plane_dimensions(PixelRect::new(crop_x, crop_y, frame_w, frame_h))
        .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
    if plane.width() != ew || plane.height() != eh || plane.width() == 0 || plane.height() == 0 {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    Ok(())
}

fn luma_scale_params(
    spec: PlaneFormat,
    range: SignalRange,
) -> Result<(f32, f32), At<ConvertError>> {
    if spec.sample == SampleType::F32 {
        return Ok((0.0, 1.0));
    }
    let max_u32 = spec.packing.max_code(spec.sample, 0).max(1);
    let max_f = max_u32 as f32;
    let bits = (32 - max_u32.leading_zeros()) as i32;
    match range {
        SignalRange::Full => Ok((0.0, max_f)),
        SignalRange::Narrow => {
            let shift = (bits - 8).max(0);
            let lo = (16u32 << shift) as f32;
            let span = (219u32 << shift) as f32;
            Ok((lo, span))
        }
        _ => Err(err_at(ConvertError::UnsupportedRoute)),
    }
}

fn chroma_scale_params(
    spec: PlaneFormat,
    comp: usize,
    range: SignalRange,
) -> Result<(f32, f32), At<ConvertError>> {
    let max_u32 = spec.packing.max_code(spec.sample, comp).max(1);
    let max_f = max_u32 as f32;
    let bits = (32 - max_u32.leading_zeros()) as i32;
    match range {
        SignalRange::Full => {
            let mid = (1u32 << (bits - 1).max(0)) as f32;
            Ok((mid, max_f))
        }
        SignalRange::Narrow => {
            let shift = (bits - 8).max(0);
            let mid = (128u32 << shift) as f32;
            let span = (224u32 << shift) as f32;
            Ok((mid, span))
        }
        _ => Err(err_at(ConvertError::UnsupportedRoute)),
    }
}

enum ParsedPlanar<'a> {
    Monochrome {
        y: &'a PlaneView<'a>,
        alpha: Option<&'a PlaneView<'a>>,
    },
    YcbcrPlanar {
        y: &'a PlaneView<'a>,
        cb: &'a PlaneView<'a>,
        cr: &'a PlaneView<'a>,
        alpha: Option<&'a PlaneView<'a>>,
    },
    YcbcrSemi {
        y: &'a PlaneView<'a>,
        uv: &'a PlaneView<'a>,
        swapped: bool,
        alpha: Option<&'a PlaneView<'a>>,
    },
    PlanarRgb {
        r: &'a PlaneView<'a>,
        g: &'a PlaneView<'a>,
        b: &'a PlaneView<'a>,
        alpha: Option<&'a PlaneView<'a>>,
    },
}

impl<'a> ParsedPlanar<'a> {
    fn alpha_plane(&self) -> Option<&'a PlaneView<'a>> {
        match *self {
            Self::Monochrome { alpha, .. }
            | Self::YcbcrPlanar { alpha, .. }
            | Self::YcbcrSemi { alpha, .. }
            | Self::PlanarRgb { alpha, .. } => alpha,
        }
    }

    fn is_monochrome(&self) -> bool {
        matches!(self, Self::Monochrome { .. })
    }
}

fn parse_planar<'a>(src: &'a PlanarView<'a>) -> Result<ParsedPlanar<'a>, At<ConvertError>> {
    let w = src.width();
    let h = src.height();
    let (ox, oy) = src.crop_origin();
    let planes = src.planes();
    if w == 0 || h == 0 || planes.is_empty() {
        return Err(err_at(ConvertError::DimensionMismatch));
    }

    let mut y_plane: Option<&PlaneView<'_>> = None;
    let mut cb_plane: Option<&PlaneView<'_>> = None;
    let mut cr_plane: Option<&PlaneView<'_>> = None;
    let mut semi_plane: Option<&PlaneView<'_>> = None;
    let mut r_plane: Option<&PlaneView<'_>> = None;
    let mut g_plane: Option<&PlaneView<'_>> = None;
    let mut b_plane: Option<&PlaneView<'_>> = None;
    let mut alpha_plane: Option<&PlaneView<'_>> = None;

    for p in planes {
        match p.format().role {
            PlaneRole::Y => {
                if y_plane.replace(p).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::Cb => {
                if cb_plane.replace(p).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::Cr => {
                if cr_plane.replace(p).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::CbCr | PlaneRole::CrCb => {
                if semi_plane.replace(p).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::R => {
                if r_plane.replace(p).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::G => {
                if g_plane.replace(p).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::B => {
                if b_plane.replace(p).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            PlaneRole::Alpha => {
                if alpha_plane.replace(p).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            // Refuse unrecognized/auxiliary roles (including GainMap, Depth, Extra) instead of silently dropping them.
            _ => return Err(err_at(ConvertError::UnsupportedRoute)),
        }
    }

    if let (Some(r), Some(g), Some(b)) = (r_plane, g_plane, b_plane) {
        if y_plane.is_some() || cb_plane.is_some() || cr_plane.is_some() || semi_plane.is_some() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        validate_plane(r, ox, oy, w, h, 1, true, true)?;
        validate_plane(g, ox, oy, w, h, 1, true, true)?;
        validate_plane(b, ox, oy, w, h, 1, true, true)?;
        if let Some(ap) = alpha_plane {
            validate_plane(ap, ox, oy, w, h, 1, true, true)?;
        }
        return Ok(ParsedPlanar::PlanarRgb {
            r,
            g,
            b,
            alpha: alpha_plane,
        });
    }
    if r_plane.is_some() || g_plane.is_some() || b_plane.is_some() {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }

    let y = y_plane.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
    validate_plane(y, ox, oy, w, h, 1, true, false)?;
    if let Some(ap) = alpha_plane {
        validate_plane(ap, ox, oy, w, h, 1, true, false)?;
    }

    if let Some(uv) = semi_plane {
        if cb_plane.is_some() || cr_plane.is_some() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        validate_plane(uv, ox, oy, w, h, 2, false, false)?;
        Ok(ParsedPlanar::YcbcrSemi {
            y,
            uv,
            swapped: uv.format().role == PlaneRole::CrCb,
            alpha: alpha_plane,
        })
    } else if cb_plane.is_none() && cr_plane.is_none() {
        Ok(ParsedPlanar::Monochrome {
            y,
            alpha: alpha_plane,
        })
    } else {
        let cb = cb_plane.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        let cr = cr_plane.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        validate_plane(cb, ox, oy, w, h, 1, false, false)?;
        validate_plane(cr, ox, oy, w, h, 1, false, false)?;
        Ok(ParsedPlanar::YcbcrPlanar {
            y,
            cb,
            cr,
            alpha: alpha_plane,
        })
    }
}

struct ResolvedPlanarSource {
    dst_color: ColorSpace,
    range: SignalRange,
    matrix: YcbcrMatrix,
    has_alpha: bool,
}

fn resolve_planar_source(
    src: &PlanarView<'_>,
    parsed: &ParsedPlanar<'_>,
    dst_layout: SampleLayout,
) -> Result<ResolvedPlanarSource, At<ConvertError>> {
    let has_alpha = parsed.alpha_plane().is_some();
    if !dst_layout.packing.is_full()
        || !matches!(
            dst_layout.sample,
            SampleType::U8 | SampleType::U16 | SampleType::F32
        )
    {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }

    let is_gray_dst = dst_layout.order.is_gray_family();
    if is_gray_dst && !parsed.is_monochrome() {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    if has_alpha && !dst_layout.order.has_alpha_slot() {
        return Err(err_at(ConvertError::AlphaLossForbidden));
    }

    let dst_alpha = if dst_layout.order.has_padding_slot() {
        AlphaMode::Undefined
    } else if dst_layout.order.has_alpha_slot() {
        if has_alpha {
            match src.effective_alpha() {
                Some(AlphaMode::Premultiplied) => AlphaMode::Premultiplied,
                Some(AlphaMode::Opaque) => AlphaMode::Opaque,
                _ => AlphaMode::Straight,
            }
        } else {
            AlphaMode::Opaque
        }
    } else {
        AlphaMode::None
    };

    let (dst_color, is_planar_f32_rgb) = match (parsed, src.color()) {
        (
            ParsedPlanar::Monochrome { .. },
            Some(ColorSpace::Gray {
                transfer, white, ..
            }),
        ) if is_gray_dst => (
            ColorSpace::Gray {
                transfer: *transfer,
                range: SignalRange::Full,
                alpha: dst_alpha,
                white: *white,
            },
            false,
        ),
        (
            ParsedPlanar::Monochrome { .. },
            Some(ColorSpace::Gray {
                transfer, white, ..
            }),
        ) => (
            ColorSpace::Rgb {
                primaries: Primaries::Bt709,
                transfer: *transfer,
                range: SignalRange::Full,
                alpha: dst_alpha,
                white: *white,
            },
            false,
        ),
        (
            _,
            Some(ColorSpace::Rgb {
                primaries,
                transfer,
                white,
                ..
            }),
        ) => {
            let color = if is_gray_dst {
                ColorSpace::Gray {
                    transfer: *transfer,
                    range: SignalRange::Full,
                    alpha: dst_alpha,
                    white: *white,
                }
            } else {
                ColorSpace::Rgb {
                    primaries: *primaries,
                    transfer: *transfer,
                    range: SignalRange::Full,
                    alpha: dst_alpha,
                    white: *white,
                }
            };
            let planar_f32 = matches!(
                parsed,
                ParsedPlanar::PlanarRgb { r, g, b, .. }
                    if r.format().sample == SampleType::F32
                        && g.format().sample == SampleType::F32
                        && b.format().sample == SampleType::F32
            );
            (color, planar_f32)
        }
        _ => return Err(err_at(ConvertError::UnsupportedRoute)),
    };

    let range = match src.effective_range() {
        Some(r) => r,
        None if is_planar_f32_rgb => SignalRange::Full,
        None => return Err(err_at(ConvertError::UnsupportedRoute)),
    };

    let matrix = match parsed {
        ParsedPlanar::Monochrome { .. } | ParsedPlanar::PlanarRgb { .. } => YcbcrMatrix::Bt709,
        ParsedPlanar::YcbcrPlanar { .. } | ParsedPlanar::YcbcrSemi { .. } => src
            .matrix_coefficients()
            .and_then(YcbcrMatrix::from_matrix_coefficients)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?,
    };

    Ok(ResolvedPlanarSource {
        dst_color,
        range,
        matrix,
        has_alpha,
    })
}

fn alpha_compatible(
    expected: AlphaMode,
    actual: AlphaMode,
    has_alpha: bool,
    dst_layout: SampleLayout,
) -> bool {
    if !has_alpha && dst_layout.order.has_alpha_slot() {
        // Constant 1.0 alpha written into RGBA/BGRA/GrayAlpha is valid under Opaque,
        // Straight, or Premultiplied tags (e.g., callers writing into `PixelFormat::RGBA8_SRGB`).
        matches!(
            actual,
            AlphaMode::Opaque | AlphaMode::Straight | AlphaMode::Premultiplied
        )
    } else {
        actual == expected
    }
}

fn validate_dst_color_compatible(
    expected: &ColorSpace,
    actual: &ColorSpace,
    has_alpha: bool,
    dst_layout: SampleLayout,
) -> Result<(), At<ConvertError>> {
    let ok = match (expected, actual) {
        (
            ColorSpace::Rgb {
                primaries: ep,
                transfer: et,
                range: er,
                alpha: ea,
                white: ew,
            },
            ColorSpace::Rgb {
                primaries: ap,
                transfer: at,
                range: ar,
                alpha: aa,
                white: aw,
            },
        ) => {
            ep == ap
                && et == at
                && er == ar
                && ew == aw
                && alpha_compatible(*ea, *aa, has_alpha, dst_layout)
        }
        (
            ColorSpace::Gray {
                transfer: et,
                range: er,
                alpha: ea,
                white: ew,
            },
            ColorSpace::Gray {
                transfer: at,
                range: ar,
                alpha: aa,
                white: aw,
            },
        ) => et == at && er == ar && ew == aw && alpha_compatible(*ea, *aa, has_alpha, dst_layout),
        _ => false,
    };
    if !ok {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    Ok(())
}

fn attach_planar_metadata(mut out: PixelBuf, src: &PlanarView<'_>) -> PixelBuf {
    out = out.with_hdr(src.hdr()).with_orientation(src.orientation());
    if let Some(origin) = src.origin() {
        out = out.with_origin(origin.clone());
    }
    out
}

/// Reconstruct a full-range `RGBF32` / `RGBAF32` [`PixelBuf`] directly from a tagged
/// [`PlanarView`] (`color()`, `effective_range()`, `matrix_coefficients()`,
/// `effective_alpha()`, `hdr()`, `origin()`, and `orientation()`).
///
/// Handles 3-plane YCbCr (`Y`, `Cb`, `Cr`), 2-plane semi-planar YCbCr (`Y`, `CbCr` / `CrCb`,
/// e.g. NV12 / P010), 3-plane RGB (`R`, `G`, `B`), and 1-plane monochrome (`Y`), each with
/// optional `Alpha` plane, using caller-chosen [`PlanarReconstructionOptions`] (e.g.
/// [`PlanarReconstructionOptions::bilinear()`] or [`PlanarReconstructionOptions::nearest()`]).
///
/// Refuses views missing required range, color space (`Rgb` / `Gray`), or (for YCbCr chroma
/// views) recognized ITU-T H.273 [`MatrixCoefficients`](zenpixels::MatrixCoefficients) with
/// [`ConvertError::UnsupportedRoute`].
pub fn reconstruct_planar_f32(
    src: &PlanarView<'_>,
    options: &PlanarReconstructionOptions,
) -> Result<PixelBuf, At<ConvertError>> {
    let parsed = parse_planar(src)?;
    let layout = if parsed.alpha_plane().is_some() {
        SampleLayout::RGBAF32
    } else {
        SampleLayout::RGBF32
    };
    reconstruct_planar_to_buf(src, layout, options)
}

/// Reconstruct an interleaved [`PixelBuf`] with caller-chosen [`SampleLayout`] (`RGB8`, `RGBA8`,
/// `BGR8`, `BGRA8`, `RGBX8`, `BGRX8`, `RGB16`, `RGBA16`, `RGBF32`, `RGBAF32`, `GRAY8`, `GRAYA8`,
/// `GRAY16`, `GRAYA16`, `GRAYF32`, `GRAYAF32`) directly from a tagged [`PlanarView`] using
/// explicit [`PlanarReconstructionOptions`] (e.g. siting-exact [`ChromaUpsample::Bilinear`]).
///
/// Reads source interpretation (`color()`, `effective_range()`, `matrix_coefficients()`,
/// `effective_alpha()`) from `src`, derives the output [`ColorSpace`], and propagates
/// `src.hdr()`, `src.origin()`, and `src.orientation()` onto the returned [`PixelBuf`].
#[doc(alias = "ycbcr_to_rgb")]
#[doc(alias = "yuv_to_rgb")]
#[doc(alias = "interleave")]
pub fn reconstruct_planar_to_buf(
    src: &PlanarView<'_>,
    dst_layout: SampleLayout,
    options: &PlanarReconstructionOptions,
) -> Result<PixelBuf, At<ConvertError>> {
    let parsed = parse_planar(src)?;
    let resolved = resolve_planar_source(src, &parsed, dst_layout)?;
    let destination_format = PixelFormat::new(dst_layout, resolved.dst_color.clone());
    let mut out = PixelBuf::new(src.width(), src.height(), destination_format)
        .map_err(|e| err_at((*e.error()).into()))?;
    reconstruct_parsed_into(src, &parsed, &resolved, options, &mut out.as_view_mut())?;
    Ok(attach_planar_metadata(out, src))
}

/// Reconstruct a tagged [`PlanarView`] (YCbCr, GBR, planar RGB/RGBA, or monochrome) directly
/// into a caller-provided strided `U8`, `U16`, or `F32` [`PixelViewMut`] (`RGB`, `RGBA`, `BGR`,
/// `BGRA`, `RGBX`, `BGRX`, `Gray`, `GrayAlpha`) with **zero heap allocation**.
///
/// Uses the exact same source interpretation metadata as [`reconstruct_planar_to_buf`]
/// (`color()`, `effective_range()`, `matrix_coefficients()`, `effective_alpha()`), and
/// validates that `dst.format().color` matches the derived full-range destination color space
/// (refusing mismatched primaries, transfer, white point, narrow range, or conflicting alpha
/// semantics rather than silently converting). Sets `dst.set_orientation(src.orientation())`;
/// `hdr()` and `origin()` remain caller-managed on the destination storage.
#[doc(alias = "ycbcr_to_rgb_into")]
#[doc(alias = "yuv_to_rgb")]
#[doc(alias = "interleave")]
pub fn reconstruct_planar_into(
    src: &PlanarView<'_>,
    dst: &mut PixelViewMut<'_>,
    options: &PlanarReconstructionOptions,
) -> Result<(), At<ConvertError>> {
    let parsed = parse_planar(src)?;
    let w = src.width();
    let h = src.height();
    let dst_layout = dst.format().layout;
    if dst.width() != w || dst.height() != h {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    let resolved = resolve_planar_source(src, &parsed, dst_layout)?;
    validate_dst_color_compatible(
        &resolved.dst_color,
        &dst.format().color,
        resolved.has_alpha,
        dst_layout,
    )?;
    reconstruct_parsed_into(src, &parsed, &resolved, options, dst)
}

fn reconstruct_parsed_into(
    src: &PlanarView<'_>,
    parsed: &ParsedPlanar<'_>,
    resolved: &ResolvedPlanarSource,
    options: &PlanarReconstructionOptions,
    dst: &mut PixelViewMut<'_>,
) -> Result<(), At<ConvertError>> {
    let w = src.width();
    let h = src.height();
    let (ox, oy) = src.crop_origin();
    let dst_layout = dst.format().layout;
    let writer = DstWriter::from_layout(dst_layout, resolved.has_alpha, parsed.is_monochrome())?;
    let alpha_max = parsed.alpha_plane().map(|ap| {
        let s = ap.format();
        if s.sample == SampleType::F32 {
            1.0_f32
        } else {
            s.packing.max_code(s.sample, 0).max(1) as f32
        }
    });

    match *parsed {
        ParsedPlanar::Monochrome { y: y_plane, alpha } => {
            let y_spec = y_plane.format();
            let (y_off, y_span) = luma_scale_params(y_spec, resolved.range)?;
            for y in 0..h {
                let y_row = y_plane.row(y);
                let alpha_row = alpha.map(|ap| (ap.format(), ap.row(y)));
                let dst_row = dst.row_mut(y);
                for x in 0..w {
                    let yn = (read_plane_f32(y_spec, y_row, x as usize, 0) - y_off) / y_span;
                    let a = read_alpha(alpha_row, alpha_max, x as usize);
                    writer.write_pixel(dst_row, x as usize, yn, yn, yn, a);
                }
            }
        }
        ParsedPlanar::PlanarRgb { r, g, b, alpha } => {
            let r_spec = r.format();
            let g_spec = g.format();
            let b_spec = b.format();
            let (r_off, r_span) = luma_scale_params(r_spec, resolved.range)?;
            let (g_off, g_span) = luma_scale_params(g_spec, resolved.range)?;
            let (b_off, b_span) = luma_scale_params(b_spec, resolved.range)?;
            for y in 0..h {
                let r_row = r.row(y);
                let g_row = g.row(y);
                let b_row = b.row(y);
                let alpha_row = alpha.map(|ap| (ap.format(), ap.row(y)));
                let dst_row = dst.row_mut(y);
                for x in 0..w {
                    let xu = x as usize;
                    let rv = (read_plane_f32(r_spec, r_row, xu, 0) - r_off) / r_span;
                    let gv = (read_plane_f32(g_spec, g_row, xu, 0) - g_off) / g_span;
                    let bv = (read_plane_f32(b_spec, b_row, xu, 0) - b_off) / b_span;
                    let a = read_alpha(alpha_row, alpha_max, xu);
                    writer.write_pixel(dst_row, xu, rv, gv, bv, a);
                }
            }
        }
        ParsedPlanar::YcbcrPlanar {
            y: y_plane,
            cb: cb_plane,
            cr: cr_plane,
            alpha,
        } => {
            let primaries = extract_primaries(src.color(), &dst.format().color);
            reconstruct_ycbcr_planes(
                y_plane,
                cb_plane,
                cr_plane,
                false,
                false,
                alpha,
                alpha_max,
                w,
                h,
                ox,
                oy,
                src.chroma_siting(),
                resolved.matrix,
                primaries,
                resolved.range,
                options,
                writer,
                dst,
            )?;
        }
        ParsedPlanar::YcbcrSemi {
            y: y_plane,
            uv,
            swapped,
            alpha,
        } => {
            let primaries = extract_primaries(src.color(), &dst.format().color);
            reconstruct_ycbcr_planes(
                y_plane,
                uv,
                uv,
                true,
                swapped,
                alpha,
                alpha_max,
                w,
                h,
                ox,
                oy,
                src.chroma_siting(),
                resolved.matrix,
                primaries,
                resolved.range,
                options,
                writer,
                dst,
            )?;
        }
    }

    dst.set_orientation(src.orientation());
    Ok(())
}

fn extract_primaries(src_color: Option<&ColorSpace>, dst_color: &ColorSpace) -> Option<Primaries> {
    if let Some(ColorSpace::Rgb { primaries, .. }) = src_color
        && *primaries != Primaries::Unknown
    {
        return Some(*primaries);
    }
    if let ColorSpace::Rgb { primaries, .. } = dst_color
        && *primaries != Primaries::Unknown
    {
        return Some(*primaries);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_ycbcr_planes(
    y_plane: &PlaneView<'_>,
    cb_plane: &PlaneView<'_>,
    cr_plane: &PlaneView<'_>,
    semi: bool,
    swapped: bool,
    alpha: Option<&PlaneView<'_>>,
    alpha_max: Option<f32>,
    w: u32,
    h: u32,
    ox: u32,
    oy: u32,
    declared_siting: ChromaSiting,
    matrix: YcbcrMatrix,
    primaries: Option<Primaries>,
    range: SignalRange,
    options: &PlanarReconstructionOptions,
    writer: DstWriter,
    dst: &mut PixelViewMut<'_>,
) -> Result<(), At<ConvertError>> {
    let y_spec = y_plane.format();
    let cb_spec = cb_plane.format();
    let cr_spec = cr_plane.format();
    let (y_off, y_span) = luma_scale_params(y_spec, range)?;

    let (cb_hx, cb_vy) = (cb_spec.subsampling.0.max(1), cb_spec.subsampling.1.max(1));
    let (cr_hx, cr_vy) = (cr_spec.subsampling.0.max(1), cr_spec.subsampling.1.max(1));
    let any_subsampled = cb_hx > 1 || cb_vy > 1 || cr_hx > 1 || cr_vy > 1;

    let effective_siting = if options.upsample == ChromaUpsample::Bilinear && any_subsampled {
        let s = if declared_siting != ChromaSiting::Unknown {
            declared_siting
        } else if let Some(fb) = options.unknown_siting_fallback
            && fb != ChromaSiting::Unknown
        {
            fb
        } else {
            return Err(err_at(ConvertError::UnsupportedRoute));
        };
        Some(s)
    } else {
        None
    };

    let cb_comp = usize::from(semi && swapped);
    let cr_comp = usize::from(semi && !swapped);

    let cb_ox0 = ox / u32::from(cb_hx);
    let cb_oy0 = oy / u32::from(cb_vy);
    let cr_ox0 = ox / u32::from(cr_hx);
    let cr_oy0 = oy / u32::from(cr_vy);

    let (cb_dx, cb_dy) = effective_siting
        .and_then(|s| s.offset_in_luma_pixels_for_divisors(cb_hx, cb_vy))
        .unwrap_or((0.0, 0.0));
    let (cr_dx, cr_dy) = effective_siting
        .and_then(|s| s.offset_in_luma_pixels_for_divisors(cr_hx, cr_vy))
        .unwrap_or((0.0, 0.0));

    let bilinear = options.upsample == ChromaUpsample::Bilinear && any_subsampled;

    enum MatrixOp {
        Identity {
            b_off: f32,
            b_span: f32,
            r_off: f32,
            r_span: f32,
        },
        YCgCo {
            cb_mid: f32,
            cb_span: f32,
            cr_mid: f32,
            cr_span: f32,
        },
        Ncl {
            cb_mid: f32,
            cb_span: f32,
            cr_mid: f32,
            cr_span: f32,
            cr_to_r: f32,
            cb_to_b: f32,
            cr_to_g: f32,
            cb_to_g: f32,
        },
    }

    let op = match matrix {
        YcbcrMatrix::Identity => {
            let (b_off, b_span) = luma_scale_params(cb_spec, range)?;
            let (r_off, r_span) = luma_scale_params(cr_spec, range)?;
            MatrixOp::Identity {
                b_off,
                b_span,
                r_off,
                r_span,
            }
        }
        YcbcrMatrix::YCgCo => {
            let (cb_mid, cb_span) = chroma_scale_params(cb_spec, cb_comp, range)?;
            let (cr_mid, cr_span) = chroma_scale_params(cr_spec, cr_comp, range)?;
            MatrixOp::YCgCo {
                cb_mid,
                cb_span,
                cr_mid,
                cr_span,
            }
        }
        _ => {
            let (kr, kb) = matrix
                .kr_kb_for(primaries)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            let kg = 1.0 - kr - kb;
            let (cb_mid, cb_span) = chroma_scale_params(cb_spec, cb_comp, range)?;
            let (cr_mid, cr_span) = chroma_scale_params(cr_spec, cr_comp, range)?;
            MatrixOp::Ncl {
                cb_mid,
                cb_span,
                cr_mid,
                cr_span,
                cr_to_r: 2.0 * (1.0 - kr),
                cb_to_b: 2.0 * (1.0 - kb),
                cr_to_g: -2.0 * kr * (1.0 - kr) / kg,
                cb_to_g: -2.0 * kb * (1.0 - kb) / kg,
            }
        }
    };

    let cb_w_max = cb_plane.width() - 1;
    let cb_h_max = cb_plane.height() - 1;
    let cr_w_max = cr_plane.width() - 1;
    let cr_h_max = cr_plane.height() - 1;

    for y in 0..h {
        let y_row = y_plane.row(y);
        let gy = oy + y;
        let alpha_row = alpha.map(|ap| (ap.format(), ap.row(y)));
        let dst_row = dst.row_mut(y);

        let (cby0, cby1, cb_ty) = if bilinear && cb_vy > 1 {
            axis_bilinear_taps(gy, cb_vy, cb_dy, cb_oy0, cb_h_max)
        } else {
            let iy = (gy / u32::from(cb_vy) - cb_oy0).min(cb_h_max);
            (iy, iy, 0.0)
        };
        let (cry0, cry1, cr_ty) = if bilinear && cr_vy > 1 {
            axis_bilinear_taps(gy, cr_vy, cr_dy, cr_oy0, cr_h_max)
        } else {
            let iy = (gy / u32::from(cr_vy) - cr_oy0).min(cr_h_max);
            (iy, iy, 0.0)
        };

        let cb_row0 = cb_plane.row(cby0);
        let cb_row1 = if cby1 == cby0 {
            cb_row0
        } else {
            cb_plane.row(cby1)
        };
        let cr_row0 = cr_plane.row(cry0);
        let cr_row1 = if cry1 == cry0 {
            cr_row0
        } else {
            cr_plane.row(cry1)
        };

        for x in 0..w {
            let gx = ox + x;
            let xu = x as usize;
            let y_code = read_plane_code(y_spec, y_row, xu, 0) as f32;
            let yn = (y_code - y_off) / y_span;

            let (cb_code, cr_code) = if bilinear {
                let cb_v = sample_chroma_bilinear(
                    cb_spec, cb_row0, cb_row1, cb_ty, gx, cb_hx, cb_dx, cb_ox0, cb_w_max, cb_comp,
                );
                let cr_v = sample_chroma_bilinear(
                    cr_spec, cr_row0, cr_row1, cr_ty, gx, cr_hx, cr_dx, cr_ox0, cr_w_max, cr_comp,
                );
                (cb_v, cr_v)
            } else {
                let cbx = ((gx / u32::from(cb_hx) - cb_ox0).min(cb_w_max)) as usize;
                let crx = if semi {
                    cbx
                } else {
                    ((gx / u32::from(cr_hx) - cr_ox0).min(cr_w_max)) as usize
                };
                (
                    read_plane_code(cb_spec, cb_row0, cbx, cb_comp) as f32,
                    read_plane_code(cr_spec, cr_row0, crx, cr_comp) as f32,
                )
            };

            let (r, g, b) = match op {
                MatrixOp::Identity {
                    b_off,
                    b_span,
                    r_off,
                    r_span,
                } => ((cr_code - r_off) / r_span, yn, (cb_code - b_off) / b_span),
                MatrixOp::YCgCo {
                    cb_mid,
                    cb_span,
                    cr_mid,
                    cr_span,
                } => {
                    let cg = (cb_code - cb_mid) / cb_span;
                    let co = (cr_code - cr_mid) / cr_span;
                    let t = yn - cg;
                    (t + co, yn + cg, t - co)
                }
                MatrixOp::Ncl {
                    cb_mid,
                    cb_span,
                    cr_mid,
                    cr_span,
                    cr_to_r,
                    cb_to_b,
                    cr_to_g,
                    cb_to_g,
                } => {
                    // Exact IEEE division so narrow white (235/940/3760) yields exact 1.0f32.
                    let cb = (cb_code - cb_mid) / cb_span;
                    let cr = (cr_code - cr_mid) / cr_span;
                    (
                        yn + cr_to_r * cr,
                        yn + cr_to_g * cr + cb_to_g * cb,
                        yn + cb_to_b * cb,
                    )
                }
            };

            let a = read_alpha(alpha_row, alpha_max, xu);
            writer.write_pixel(dst_row, xu, r, g, b, a);
        }
    }

    Ok(())
}

#[inline(always)]
fn axis_bilinear_taps(
    global_pos: u32,
    factor: u8,
    offset_in_cell: f32,
    plane_origin: u32,
    max_idx: u32,
) -> (u32, u32, f32) {
    let u_global = (global_pos as f32 - offset_in_cell) / f32::from(factor);
    let u_rel = (u_global - plane_origin as f32).clamp(0.0, max_idx as f32);
    let i0 = (libm::floorf(u_rel) as u32).min(max_idx);
    let i1 = (i0 + 1).min(max_idx);
    let t = u_rel - (i0 as f32);
    (i0, i1, t)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn sample_chroma_bilinear(
    spec: PlaneFormat,
    row0: &[u8],
    row1: &[u8],
    ty: f32,
    gx: u32,
    hx: u8,
    dx: f32,
    ox0: u32,
    w_max: u32,
    comp: usize,
) -> f32 {
    let (x0, x1, tx) = if hx > 1 {
        axis_bilinear_taps(gx, hx, dx, ox0, w_max)
    } else {
        let ix = (gx - ox0).min(w_max);
        (ix, ix, 0.0)
    };
    let s00 = read_plane_code(spec, row0, x0 as usize, comp) as f32;
    let top = if x1 == x0 {
        s00
    } else {
        let s01 = read_plane_code(spec, row0, x1 as usize, comp) as f32;
        s00 + (s01 - s00) * tx
    };
    if ty == 0.0 {
        return top;
    }
    let s10 = read_plane_code(spec, row1, x0 as usize, comp) as f32;
    let bot = if x1 == x0 {
        s10
    } else {
        let s11 = read_plane_code(spec, row1, x1 as usize, comp) as f32;
        s10 + (s11 - s10) * tx
    };
    top + (bot - top) * ty
}

#[derive(Clone, Copy)]
struct DstWriter {
    sample: SampleType,
    order: ChannelOrder,
}

impl DstWriter {
    fn from_layout(
        layout: SampleLayout,
        src_has_alpha: bool,
        src_is_mono: bool,
    ) -> Result<Self, At<ConvertError>> {
        let supported_order = matches!(
            layout.order,
            ChannelOrder::Rgb
                | ChannelOrder::Rgba
                | ChannelOrder::Bgr
                | ChannelOrder::Bgra
                | ChannelOrder::Rgbx
                | ChannelOrder::Bgrx
                | ChannelOrder::Gray
                | ChannelOrder::GrayAlpha
        );
        if !supported_order {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if layout.order.is_gray_family() && !src_is_mono {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if src_has_alpha && !layout.order.has_alpha_slot() {
            return Err(err_at(ConvertError::AlphaLossForbidden));
        }
        Ok(Self {
            sample: layout.sample,
            order: layout.order,
        })
    }

    #[inline(always)]
    fn write_pixel(self, row: &mut [u8], x: usize, r: f32, g: f32, b: f32, a: f32) {
        match self.sample {
            SampleType::U8 => {
                let to_u8 = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8 };
                match self.order {
                    ChannelOrder::Rgb => {
                        let o = x * 3;
                        row[o] = to_u8(r);
                        row[o + 1] = to_u8(g);
                        row[o + 2] = to_u8(b);
                    }
                    ChannelOrder::Bgr => {
                        let o = x * 3;
                        row[o] = to_u8(b);
                        row[o + 1] = to_u8(g);
                        row[o + 2] = to_u8(r);
                    }
                    ChannelOrder::Rgba | ChannelOrder::Rgbx => {
                        let o = x * 4;
                        row[o] = to_u8(r);
                        row[o + 1] = to_u8(g);
                        row[o + 2] = to_u8(b);
                        row[o + 3] = to_u8(a);
                    }
                    ChannelOrder::Bgra | ChannelOrder::Bgrx => {
                        let o = x * 4;
                        row[o] = to_u8(b);
                        row[o + 1] = to_u8(g);
                        row[o + 2] = to_u8(r);
                        row[o + 3] = to_u8(a);
                    }
                    ChannelOrder::Gray => {
                        row[x] = to_u8(r);
                    }
                    ChannelOrder::GrayAlpha => {
                        let o = x * 2;
                        row[o] = to_u8(r);
                        row[o + 1] = to_u8(a);
                    }
                    _ => {}
                }
            }
            SampleType::U16 => {
                let to_u16 = |v: f32| -> [u8; 2] {
                    ((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_ne_bytes()
                };
                match self.order {
                    ChannelOrder::Rgb => {
                        let o = x * 6;
                        row[o..o + 2].copy_from_slice(&to_u16(r));
                        row[o + 2..o + 4].copy_from_slice(&to_u16(g));
                        row[o + 4..o + 6].copy_from_slice(&to_u16(b));
                    }
                    ChannelOrder::Bgr => {
                        let o = x * 6;
                        row[o..o + 2].copy_from_slice(&to_u16(b));
                        row[o + 2..o + 4].copy_from_slice(&to_u16(g));
                        row[o + 4..o + 6].copy_from_slice(&to_u16(r));
                    }
                    ChannelOrder::Rgba | ChannelOrder::Rgbx => {
                        let o = x * 8;
                        row[o..o + 2].copy_from_slice(&to_u16(r));
                        row[o + 2..o + 4].copy_from_slice(&to_u16(g));
                        row[o + 4..o + 6].copy_from_slice(&to_u16(b));
                        row[o + 6..o + 8].copy_from_slice(&to_u16(a));
                    }
                    ChannelOrder::Bgra | ChannelOrder::Bgrx => {
                        let o = x * 8;
                        row[o..o + 2].copy_from_slice(&to_u16(b));
                        row[o + 2..o + 4].copy_from_slice(&to_u16(g));
                        row[o + 4..o + 6].copy_from_slice(&to_u16(r));
                        row[o + 6..o + 8].copy_from_slice(&to_u16(a));
                    }
                    ChannelOrder::Gray => {
                        let o = x * 2;
                        row[o..o + 2].copy_from_slice(&to_u16(r));
                    }
                    ChannelOrder::GrayAlpha => {
                        let o = x * 4;
                        row[o..o + 2].copy_from_slice(&to_u16(r));
                        row[o + 2..o + 4].copy_from_slice(&to_u16(a));
                    }
                    _ => {}
                }
            }
            SampleType::F32 => match self.order {
                ChannelOrder::Rgb => {
                    let o = x * 12;
                    row[o..o + 4].copy_from_slice(&r.to_ne_bytes());
                    row[o + 4..o + 8].copy_from_slice(&g.to_ne_bytes());
                    row[o + 8..o + 12].copy_from_slice(&b.to_ne_bytes());
                }
                ChannelOrder::Bgr => {
                    let o = x * 12;
                    row[o..o + 4].copy_from_slice(&b.to_ne_bytes());
                    row[o + 4..o + 8].copy_from_slice(&g.to_ne_bytes());
                    row[o + 8..o + 12].copy_from_slice(&r.to_ne_bytes());
                }
                ChannelOrder::Rgba | ChannelOrder::Rgbx => {
                    let o = x * 16;
                    row[o..o + 4].copy_from_slice(&r.to_ne_bytes());
                    row[o + 4..o + 8].copy_from_slice(&g.to_ne_bytes());
                    row[o + 8..o + 12].copy_from_slice(&b.to_ne_bytes());
                    row[o + 12..o + 16].copy_from_slice(&a.to_ne_bytes());
                }
                ChannelOrder::Bgra | ChannelOrder::Bgrx => {
                    let o = x * 16;
                    row[o..o + 4].copy_from_slice(&b.to_ne_bytes());
                    row[o + 4..o + 8].copy_from_slice(&g.to_ne_bytes());
                    row[o + 8..o + 12].copy_from_slice(&r.to_ne_bytes());
                    row[o + 12..o + 16].copy_from_slice(&a.to_ne_bytes());
                }
                ChannelOrder::Gray => {
                    let o = x * 4;
                    row[o..o + 4].copy_from_slice(&r.to_ne_bytes());
                }
                ChannelOrder::GrayAlpha => {
                    let o = x * 8;
                    row[o..o + 4].copy_from_slice(&r.to_ne_bytes());
                    row[o + 4..o + 8].copy_from_slice(&a.to_ne_bytes());
                }
                _ => {}
            },
            _ => {}
        }
    }
}

#[inline(always)]
fn read_alpha(alpha_row: Option<(PlaneFormat, &[u8])>, alpha_max: Option<f32>, x: usize) -> f32 {
    if let (Some((a_spec, a_row)), Some(a_max)) = (alpha_row, alpha_max) {
        read_plane_f32(a_spec, a_row, x, 0) / a_max
    } else {
        1.0
    }
}

#[inline(always)]
fn read_plane_f32(spec: PlaneFormat, row: &[u8], x: usize, comp: usize) -> f32 {
    if spec.sample == SampleType::F32 {
        let elem = spec.bytes_per_element();
        let off = x * elem + comp * 4;
        f32::from_ne_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]])
    } else {
        read_plane_code(spec, row, x, comp) as f32
    }
}

#[inline(always)]
fn read_plane_code(spec: PlaneFormat, row: &[u8], x: usize, comp: usize) -> u32 {
    let sz = spec.sample.byte_size();
    let elem = spec.bytes_per_element();
    // Packed bitfield words store all components inside one word at `x * elem`;
    // unpacked multi-component planes store component `comp` at `x * elem + comp * sz`.
    let off = if spec.packing.is_packed_word() {
        x * elem
    } else {
        x * elem + comp * sz
    };
    let w = match spec.sample {
        SampleType::U8 => u32::from(row[off]),
        SampleType::U16 => u32::from(u16::from_ne_bytes([row[off], row[off + 1]])),
        SampleType::U32 => u32::from_ne_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]),
        _ => 0,
    };
    spec.packing.extract_code(spec.sample, comp, w)
}
