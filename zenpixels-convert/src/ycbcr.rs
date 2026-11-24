//! Native multi-plane YCbCr (`PlanarView` / `PlanarBuf`, including 8/10/12-bit
//! `BitPacking::P010` and `U10_LSB`, 4:4:4 / 4:2:2 / 4:2:0) → interleaved RGB F32.

use whereat::At;
use zenpixels::{
    AlphaMode, ColorSpace, ImageSpec, PixelBuf, PlanarView, PlaneRole, PlaneSpec, PlaneView,
    Primaries, SampleLayout, SampleType, SignalRange, Transfer,
};

use crate::error::{ConvertError, err_at};

/// YCbCr luma/chroma matrix standard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum YcbcrMatrix {
    /// ITU-R BT.601 (`Kr = 0.299, Kb = 0.114`).
    Bt601,
    /// ITU-R BT.709 (`Kr = 0.2126, Kb = 0.0722`).
    Bt709,
    /// ITU-R BT.2020 NCL (`Kr = 0.2627, Kb = 0.0593`).
    Bt2020,
}

impl YcbcrMatrix {
    #[inline]
    const fn kr_kb(self) -> (f32, f32) {
        match self {
            Self::Bt601 => (0.299, 0.114),
            Self::Bt709 => (0.2126, 0.0722),
            Self::Bt2020 => (0.2627, 0.0593),
        }
    }
}

fn validate_ycbcr_plane(
    plane: &PlaneView<'_>,
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
    let (ew, eh) = spec.dimensions(frame_w, frame_h);
    if plane.width() != ew || plane.height() != eh || plane.width() == 0 || plane.height() == 0 {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    Ok(())
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
            Ok((mid, 1.0 / max_f))
        }
        SignalRange::Narrow => {
            let shift = (bits - 8).max(0);
            let mid = (128u32 << shift) as f32;
            let span = (224u32 << shift) as f32;
            Ok((mid, 1.0 / span))
        }
        _ => Err(err_at(ConvertError::UnsupportedRoute)),
    }
}

/// Reconstruct a full-range `RGBF32` [`PixelBuf`] from a 3-plane (`Y`, `Cb`, `Cr`)
/// or 2-plane semi-planar (`Y`, `CbCr` / `CrCb`, e.g. NV12 / P010) [`PlanarView`].
///
/// Honors each plane's [`BitPacking`](zenpixels::BitPacking) (`Full`, `U10_LSB`,
/// `U12_LSB`, `P010` MSB) and independent subsampling divisors (`4:4:4`, `4:2:2`, `4:2:0`).
/// Validates plane roles, multiplicities, dimensions, sample types, and component counts.
pub fn ycbcr_to_rgb_f32(
    src: &PlanarView<'_>,
    matrix: YcbcrMatrix,
    range: SignalRange,
    primaries: Primaries,
    transfer: Transfer,
) -> Result<PixelBuf, At<ConvertError>> {
    let w = src.width();
    let h = src.height();
    let planes = src.planes();
    if w == 0 || h == 0 || planes.len() < 2 {
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
    validate_ycbcr_plane(y_plane, w, h, 1, true)?;
    if let Some(ap) = alpha_plane {
        validate_ycbcr_plane(ap, w, h, 1, true)?;
    }

    let (semi, cb_plane, cr_plane) = if let Some(sp) = semi_plane {
        if cb_plane.is_some() || cr_plane.is_some() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        validate_ycbcr_plane(sp, w, h, 2, false)?;
        (true, sp, sp)
    } else {
        let cb = cb_plane.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        let cr = cr_plane.ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        validate_ycbcr_plane(cb, w, h, 1, false)?;
        validate_ycbcr_plane(cr, w, h, 1, false)?;
        (false, cb, cr)
    };

    let has_alpha = alpha_plane.is_some();
    let dst_ch = if has_alpha { 4usize } else { 3usize };
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
    let mut out = PixelBuf::new(w, h, dst_spec).map_err(|e| err_at((*e.error()).into()))?;
    let alpha_scale = alpha_plane.map(|ap| {
        let s = ap.spec();
        1.0 / (s.packing.max_code(s.sample, 0).max(1) as f32)
    });

    let (kr, kb) = matrix.kr_kb();
    let kg = 1.0 - kr - kb;
    let cr_to_r = 2.0 * (1.0 - kr);
    let cb_to_b = 2.0 * (1.0 - kb);
    let cr_to_g = -2.0 * kr * (1.0 - kr) / kg;
    let cb_to_g = -2.0 * kb * (1.0 - kb) / kg;

    let y_spec = y_plane.spec();
    let y_max_u32 = y_spec.packing.max_code(y_spec.sample, 0).max(1);
    let y_max = y_max_u32 as f32;
    let y_bits = (32 - y_max_u32.leading_zeros()) as i32;
    let (y_off, y_scale) = match range {
        SignalRange::Full => (0.0, 1.0 / y_max),
        SignalRange::Narrow => {
            let shift = (y_bits - 8).max(0);
            let lo = (16u32 << shift) as f32;
            let span = (219u32 << shift) as f32;
            (lo, 1.0 / span)
        }
        _ => return Err(err_at(ConvertError::UnsupportedRoute)),
    };

    let cb_spec = cb_plane.spec();
    let cr_spec = cr_plane.spec();
    let swapped = semi && cb_spec.role == PlaneRole::CrCb;
    let (cb_mid, cb_scale) = chroma_scale_params(cb_spec, usize::from(swapped), range)?;
    let (cr_mid, cr_scale) = chroma_scale_params(cr_spec, usize::from(semi && !swapped), range)?;

    let (cb_hx, cb_vy) = (
        cb_spec.subsampling.0.max(1) as u32,
        cb_spec.subsampling.1.max(1) as u32,
    );
    let (cr_hx, cr_vy) = (
        cr_spec.subsampling.0.max(1) as u32,
        cr_spec.subsampling.1.max(1) as u32,
    );

    let mut out_view = out.as_view_mut();
    for y in 0..h {
        let y_row = y_plane.row(y);
        let cby = (y / cb_vy).min(cb_plane.height() - 1);
        let cry = (y / cr_vy).min(cr_plane.height() - 1);
        let cb_row = cb_plane.row(cby);
        let cr_row = cr_plane.row(cry);
        let alpha_row = alpha_plane.map(|ap| (ap.spec(), ap.row(y)));
        let dst_f32: &mut [f32] = out_view
            .row_as_mut::<f32>(y)
            .map_err(|e| err_at((*e.error()).into()))?;

        for x in 0..w {
            let y_code = read_plane_code(y_spec, y_row, x as usize, 0) as f32;
            let cbx = ((x / cb_hx).min(cb_plane.width() - 1)) as usize;
            let (cb_code, cr_code) = if semi {
                let c0 = read_plane_code(cb_spec, cb_row, cbx, 0) as f32;
                let c1 = read_plane_code(cb_spec, cb_row, cbx, 1) as f32;
                if swapped { (c1, c0) } else { (c0, c1) }
            } else {
                let crx = ((x / cr_hx).min(cr_plane.width() - 1)) as usize;
                (
                    read_plane_code(cb_spec, cb_row, cbx, 0) as f32,
                    read_plane_code(cr_spec, cr_row, crx, 0) as f32,
                )
            };

            let yn = (y_code - y_off) * y_scale;
            let cb = (cb_code - cb_mid) * cb_scale;
            let cr = (cr_code - cr_mid) * cr_scale;

            let o = (x as usize) * dst_ch;
            dst_f32[o] = yn + cr_to_r * cr;
            dst_f32[o + 1] = yn + cr_to_g * cr + cb_to_g * cb;
            dst_f32[o + 2] = yn + cb_to_b * cb;
            if let (Some((a_spec, a_row)), Some(a_sc)) = (alpha_row, alpha_scale) {
                dst_f32[o + 3] = (read_plane_code(a_spec, a_row, x as usize, 0) as f32) * a_sc;
            }
        }
    }
    Ok(out)
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
