//! Native multi-plane YCbCr / GBR / monochrome (`PlanarView` / `PlanarBuf`, including
//! 8/10/12-bit `BitPacking::P010` and `U10_LSB`, 4:4:4 / 4:2:2 / 4:2:0) → interleaved RGB(A) F32.

use whereat::At;
use zenpixels::{
    AlphaMode, ChannelOrder, ColorSpace, ImageSpec, PixelBuf, PixelViewMut, PlanarView, PlaneRole,
    PlaneSpec, PlaneView, Primaries, SampleLayout, SampleType, SignalRange, Transfer,
};

use crate::error::{ConvertError, err_at};

/// YCbCr / GBR luma-chroma matrix standard (ITU-T H.273 Table 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum YcbcrMatrix {
    /// Identity / GBR (`MC = 0`: `Y = G, Cb = B, Cr = R`, lossless RGB in planar form).
    Identity,
    /// ITU-R BT.601 (`Kr = 0.299, Kb = 0.114`).
    Bt601,
    /// ITU-R BT.709 (`Kr = 0.2126, Kb = 0.0722`).
    Bt709,
    /// ITU-R BT.2020 NCL (`Kr = 0.2627, Kb = 0.0593`).
    Bt2020,
}

impl YcbcrMatrix {
    /// Map an ITU-T H.273 Table 4 `matrix_coefficients` code to a [`YcbcrMatrix`].
    ///
    /// Recognizes `0` (`Identity` / GBR), `1` (`Bt709`), `5` / `6` (`Bt601`), and `9` (`Bt2020`).
    /// Returns `None` for unspecified (`2`) or unsupported matrix codes.
    #[inline]
    #[must_use]
    pub const fn from_cicp_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Identity),
            1 => Some(Self::Bt709),
            5 | 6 => Some(Self::Bt601),
            9 => Some(Self::Bt2020),
            _ => None,
        }
    }

    #[inline]
    const fn kr_kb(self) -> Option<(f32, f32)> {
        match self {
            Self::Identity => None,
            Self::Bt601 => Some((0.299, 0.114)),
            Self::Bt709 => Some((0.2126, 0.0722)),
            Self::Bt2020 => Some((0.2627, 0.0593)),
        }
    }
}

fn validate_ycbcr_plane(
    plane: &PlaneView<'_>,
    crop_x: u32,
    crop_y: u32,
    frame_w: u32,
    frame_h: u32,
    expected_components: u8,
    require_full_res: bool,
) -> Result<(), At<ConvertError>> {
    let spec = plane.spec();
    spec.validate().map_err(|e| err_at(e.into()))?;
    if !matches!(
        spec.sample,
        SampleType::U8 | SampleType::U16 | SampleType::U32
    ) {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    if spec.components != expected_components {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    if require_full_res && spec.subsampling != (1, 1) {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    let (ew, eh) = spec
        .cropped_dimensions(frame_w, frame_h, (crop_x, crop_y))
        .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
    if plane.width() != ew || plane.height() != eh || plane.width() == 0 || plane.height() == 0 {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    Ok(())
}

fn luma_scale_params(spec: PlaneSpec, range: SignalRange) -> Result<(f32, f32), At<ConvertError>> {
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
    spec: PlaneSpec,
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

struct ParsedYcbcr<'a> {
    y_plane: &'a PlaneView<'a>,
    chroma: ChromaKind<'a>,
    alpha_plane: Option<&'a PlaneView<'a>>,
}

enum ChromaKind<'a> {
    Monochrome,
    Planar {
        cb: &'a PlaneView<'a>,
        cr: &'a PlaneView<'a>,
    },
    Semi {
        plane: &'a PlaneView<'a>,
        swapped: bool,
    },
}

fn parse_ycbcr_planes<'a>(src: &'a PlanarView<'a>) -> Result<ParsedYcbcr<'a>, At<ConvertError>> {
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
    let mut alpha_plane: Option<&PlaneView<'_>> = None;

    for p in planes {
        match p.spec().role {
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
            PlaneRole::Alpha => {
                if alpha_plane.replace(p).is_some() {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
            }
            // Refuse unrecognized/auxiliary roles (including GainMap) instead of silently dropping them.
            _ => return Err(err_at(ConvertError::UnsupportedRoute)),
        }
    }

    let y_plane = y_plane.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
    validate_ycbcr_plane(y_plane, ox, oy, w, h, 1, true)?;
    if let Some(ap) = alpha_plane {
        validate_ycbcr_plane(ap, ox, oy, w, h, 1, true)?;
    }

    let chroma = if let Some(sp) = semi_plane {
        if cb_plane.is_some() || cr_plane.is_some() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        validate_ycbcr_plane(sp, ox, oy, w, h, 2, false)?;
        ChromaKind::Semi {
            plane: sp,
            swapped: sp.spec().role == PlaneRole::CrCb,
        }
    } else if cb_plane.is_none() && cr_plane.is_none() {
        ChromaKind::Monochrome
    } else {
        let cb = cb_plane.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        let cr = cr_plane.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        validate_ycbcr_plane(cb, ox, oy, w, h, 1, false)?;
        validate_ycbcr_plane(cr, ox, oy, w, h, 1, false)?;
        ChromaKind::Planar { cb, cr }
    };

    Ok(ParsedYcbcr {
        y_plane,
        chroma,
        alpha_plane,
    })
}

/// Reconstruct a full-range `RGBF32` / `RGBAF32` [`PixelBuf`] from a 3-plane (`Y`, `Cb`, `Cr`),
/// 2-plane semi-planar (`Y`, `CbCr` / `CrCb`, e.g. NV12 / P010), or 1-plane monochrome (`Y`,
/// optionally + `Alpha`) [`PlanarView`].
///
/// Honors each plane's [`BitPacking`](zenpixels::BitPacking) (`Full`, `U10_LSB`,
/// `U12_LSB`, `P010` MSB), independent subsampling divisors (`4:4:4`, `4:2:2`, `4:2:0`),
/// and [`PlanarView::crop_origin`] chroma phase (so odd-origin 4:2:0 crops match full-frame
/// conversion followed by crop bit-for-bit).
pub fn ycbcr_to_rgb_f32(
    src: &PlanarView<'_>,
    matrix: YcbcrMatrix,
    range: SignalRange,
    primaries: Primaries,
    transfer: Transfer,
) -> Result<PixelBuf, At<ConvertError>> {
    let parsed = parse_ycbcr_planes(src)?;
    let has_alpha = parsed.alpha_plane.is_some();
    let dst_spec = ImageSpec::new(
        if has_alpha {
            SampleLayout::RGBAF32
        } else {
            SampleLayout::RGBF32
        },
        ColorSpace::Rgb {
            primaries,
            transfer,
            range: SignalRange::Full,
            alpha: if has_alpha {
                AlphaMode::Straight
            } else {
                AlphaMode::None
            },
            white: None,
        },
    );
    let mut out = PixelBuf::new(src.width(), src.height(), dst_spec)
        .map_err(|e| err_at((*e.error()).into()))?;
    ycbcr_to_rgb_into(src, matrix, range, &mut out.as_view_mut())?;
    Ok(out)
}

/// Reconstruct a full-range `RGBF32` / `RGBAF32` [`PixelBuf`] directly from the color
/// interpretation metadata attached to a [`PlanarView`] (`color()`, `effective_range()`,
/// `ycbcr_matrix_code()`, and `effective_alpha()`).
///
/// Refuses views missing required range, color space (`Rgb` / `Gray`), or (for multi-plane
/// chroma views) recognized ITU-T H.273 `ycbcr_matrix_code` with
/// [`ConvertError::UnsupportedRoute`] rather than guessing defaults.
pub fn ycbcr_to_rgb_from_planar(src: &PlanarView<'_>) -> Result<PixelBuf, At<ConvertError>> {
    let parsed = parse_ycbcr_planes(src)?;
    let (primaries, transfer, white) = match src.color() {
        Some(ColorSpace::Rgb {
            primaries,
            transfer,
            white,
            ..
        }) => (*primaries, *transfer, *white),
        Some(ColorSpace::Gray {
            transfer, white, ..
        }) if matches!(parsed.chroma, ChromaKind::Monochrome) => {
            (Primaries::Bt709, *transfer, *white)
        }
        _ => return Err(err_at(ConvertError::UnsupportedRoute)),
    };
    let range = src
        .effective_range()
        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
    let matrix = match parsed.chroma {
        ChromaKind::Monochrome => YcbcrMatrix::Bt709,
        ChromaKind::Planar { .. } | ChromaKind::Semi { .. } => src
            .ycbcr_matrix_code()
            .and_then(YcbcrMatrix::from_cicp_code)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?,
    };
    let has_alpha = parsed.alpha_plane.is_some();
    let alpha = if has_alpha {
        match src.effective_alpha() {
            Some(AlphaMode::Premultiplied) => AlphaMode::Premultiplied,
            Some(AlphaMode::Opaque) => AlphaMode::Opaque,
            _ => AlphaMode::Straight,
        }
    } else {
        AlphaMode::None
    };
    let dst_spec = ImageSpec::new(
        if has_alpha {
            SampleLayout::RGBAF32
        } else {
            SampleLayout::RGBF32
        },
        ColorSpace::Rgb {
            primaries,
            transfer,
            range: SignalRange::Full,
            alpha,
            white,
        },
    );
    let mut out = PixelBuf::new(src.width(), src.height(), dst_spec)
        .map_err(|e| err_at((*e.error()).into()))?;
    ycbcr_to_rgb_into(src, matrix, range, &mut out.as_view_mut())?;
    Ok(out)
}

/// Reconstruct YCbCr / GBR / monochrome planes directly into a caller-provided strided `RGBF32`
/// or `RGBAF32` [`PixelViewMut`] with **zero heap allocation**.
pub fn ycbcr_to_rgb_into(
    src: &PlanarView<'_>,
    matrix: YcbcrMatrix,
    range: SignalRange,
    dst: &mut PixelViewMut<'_>,
) -> Result<(), At<ConvertError>> {
    let parsed = parse_ycbcr_planes(src)?;
    let w = src.width();
    let h = src.height();
    let (ox, oy) = src.crop_origin();

    let dst_layout = dst.spec().layout;
    if dst.width() != w
        || dst.rows() != h
        || dst_layout.sample != SampleType::F32
        || !dst_layout.packing.is_full()
    {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    let has_alpha = parsed.alpha_plane.is_some();
    let dst_ch = match dst_layout.order {
        ChannelOrder::Rgb if !has_alpha => 3usize,
        ChannelOrder::Rgba => 4usize,
        _ => return Err(err_at(ConvertError::UnsupportedRoute)),
    };

    let y_plane = parsed.y_plane;
    let y_spec = y_plane.spec();
    let (y_off, y_span) = luma_scale_params(y_spec, range)?;
    let alpha_max = parsed.alpha_plane.map(|ap| {
        let s = ap.spec();
        s.packing.max_code(s.sample, 0).max(1) as f32
    });

    match parsed.chroma {
        ChromaKind::Monochrome => {
            for y in 0..h {
                let y_row = y_plane.row(y);
                let alpha_row = parsed.alpha_plane.map(|ap| (ap.spec(), ap.row(y)));
                let dst_f32: &mut [f32] = dst
                    .row_as_mut::<f32>(y)
                    .map_err(|e| err_at((*e.error()).into()))?;
                for x in 0..w {
                    let y_code = read_plane_code(y_spec, y_row, x as usize, 0) as f32;
                    let yn = (y_code - y_off) / y_span;
                    let o = (x as usize) * dst_ch;
                    dst_f32[o] = yn;
                    dst_f32[o + 1] = yn;
                    dst_f32[o + 2] = yn;
                    if dst_ch == 4 {
                        dst_f32[o + 3] =
                            if let (Some((a_spec, a_row)), Some(a_max)) = (alpha_row, alpha_max) {
                                (read_plane_code(a_spec, a_row, x as usize, 0) as f32) / a_max
                            } else {
                                1.0
                            };
                    }
                }
            }
            return Ok(());
        }
        ChromaKind::Planar { .. } | ChromaKind::Semi { .. } => {
            let (semi, swapped, cb_plane, cr_plane) = match parsed.chroma {
                ChromaKind::Semi { plane, swapped } => (true, swapped, plane, plane),
                ChromaKind::Planar { cb, cr } => (false, false, cb, cr),
                ChromaKind::Monochrome => unreachable!(),
            };
            let cb_spec = cb_plane.spec();
            let cr_spec = cr_plane.spec();
            let (cb_hx, cb_vy) = (
                cb_spec.subsampling.0.max(1) as u32,
                cb_spec.subsampling.1.max(1) as u32,
            );
            let (cr_hx, cr_vy) = (
                cr_spec.subsampling.0.max(1) as u32,
                cr_spec.subsampling.1.max(1) as u32,
            );
            let cb_ox0 = ox / cb_hx;
            let cb_oy0 = oy / cb_vy;
            let cr_ox0 = ox / cr_hx;
            let cr_oy0 = oy / cr_vy;

            if matrix == YcbcrMatrix::Identity {
                // H.273 MC=0 Identity (GBR): Y=Green, Cb=Blue, Cr=Red; uses luma range/span on all 3 planes.
                let (b_off, b_span) = luma_scale_params(cb_spec, range)?;
                let (r_off, r_span) = luma_scale_params(cr_spec, range)?;
                for y in 0..h {
                    let y_row = y_plane.row(y);
                    let gy = oy + y;
                    let cby = (gy / cb_vy - cb_oy0).min(cb_plane.height() - 1);
                    let cry = (gy / cr_vy - cr_oy0).min(cr_plane.height() - 1);
                    let cb_row = cb_plane.row(cby);
                    let cr_row = cr_plane.row(cry);
                    let alpha_row = parsed.alpha_plane.map(|ap| (ap.spec(), ap.row(y)));
                    let dst_f32: &mut [f32] = dst
                        .row_as_mut::<f32>(y)
                        .map_err(|e| err_at((*e.error()).into()))?;
                    for x in 0..w {
                        let gx = ox + x;
                        let g_code = read_plane_code(y_spec, y_row, x as usize, 0) as f32;
                        let cbx = ((gx / cb_hx - cb_ox0).min(cb_plane.width() - 1)) as usize;
                        let (b_code, r_code) = if semi {
                            let c0 = read_plane_code(cb_spec, cb_row, cbx, 0) as f32;
                            let c1 = read_plane_code(cb_spec, cb_row, cbx, 1) as f32;
                            if swapped { (c1, c0) } else { (c0, c1) }
                        } else {
                            let crx = ((gx / cr_hx - cr_ox0).min(cr_plane.width() - 1)) as usize;
                            (
                                read_plane_code(cb_spec, cb_row, cbx, 0) as f32,
                                read_plane_code(cr_spec, cr_row, crx, 0) as f32,
                            )
                        };
                        let o = (x as usize) * dst_ch;
                        dst_f32[o] = (r_code - r_off) / r_span;
                        dst_f32[o + 1] = (g_code - y_off) / y_span;
                        dst_f32[o + 2] = (b_code - b_off) / b_span;
                        if dst_ch == 4 {
                            dst_f32[o + 3] = if let (Some((a_spec, a_row)), Some(a_max)) =
                                (alpha_row, alpha_max)
                            {
                                (read_plane_code(a_spec, a_row, x as usize, 0) as f32) / a_max
                            } else {
                                1.0
                            };
                        }
                    }
                }
                return Ok(());
            }

            let (kr, kb) = matrix
                .kr_kb()
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            let kg = 1.0 - kr - kb;
            let cr_to_r = 2.0 * (1.0 - kr);
            let cb_to_b = 2.0 * (1.0 - kb);
            let cr_to_g = -2.0 * kr * (1.0 - kr) / kg;
            let cb_to_g = -2.0 * kb * (1.0 - kb) / kg;

            let (cb_mid, cb_span) = chroma_scale_params(cb_spec, usize::from(swapped), range)?;
            let (cr_mid, cr_span) =
                chroma_scale_params(cr_spec, usize::from(semi && !swapped), range)?;

            for y in 0..h {
                let y_row = y_plane.row(y);
                let gy = oy + y;
                let cby = (gy / cb_vy - cb_oy0).min(cb_plane.height() - 1);
                let cry = (gy / cr_vy - cr_oy0).min(cr_plane.height() - 1);
                let cb_row = cb_plane.row(cby);
                let cr_row = cr_plane.row(cry);
                let alpha_row = parsed.alpha_plane.map(|ap| (ap.spec(), ap.row(y)));
                let dst_f32: &mut [f32] = dst
                    .row_as_mut::<f32>(y)
                    .map_err(|e| err_at((*e.error()).into()))?;

                for x in 0..w {
                    let gx = ox + x;
                    let y_code = read_plane_code(y_spec, y_row, x as usize, 0) as f32;
                    let cbx = ((gx / cb_hx - cb_ox0).min(cb_plane.width() - 1)) as usize;
                    let (cb_code, cr_code) = if semi {
                        let c0 = read_plane_code(cb_spec, cb_row, cbx, 0) as f32;
                        let c1 = read_plane_code(cb_spec, cb_row, cbx, 1) as f32;
                        if swapped { (c1, c0) } else { (c0, c1) }
                    } else {
                        let crx = ((gx / cr_hx - cr_ox0).min(cr_plane.width() - 1)) as usize;
                        (
                            read_plane_code(cb_spec, cb_row, cbx, 0) as f32,
                            read_plane_code(cr_spec, cr_row, crx, 0) as f32,
                        )
                    };

                    // Use exact IEEE division so narrow white (235/940/3760) yields exact 1.0f32!
                    let yn = (y_code - y_off) / y_span;
                    let cb = (cb_code - cb_mid) / cb_span;
                    let cr = (cr_code - cr_mid) / cr_span;

                    let o = (x as usize) * dst_ch;
                    dst_f32[o] = yn + cr_to_r * cr;
                    dst_f32[o + 1] = yn + cr_to_g * cr + cb_to_g * cb;
                    dst_f32[o + 2] = yn + cb_to_b * cb;
                    if dst_ch == 4 {
                        dst_f32[o + 3] =
                            if let (Some((a_spec, a_row)), Some(a_max)) = (alpha_row, alpha_max) {
                                (read_plane_code(a_spec, a_row, x as usize, 0) as f32) / a_max
                            } else {
                                1.0
                            };
                    }
                }
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn read_plane_code(spec: PlaneSpec, row: &[u8], x: usize, comp: usize) -> u32 {
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
    spec.packing.extract_code(spec.sample, w, comp)
}
