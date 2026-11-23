//! Native multi-plane YCbCr (`PlanarView` / `PlanarBuf`, including 8/10/12-bit
//! `BitPacking::P010` and `U10_LSB`, 4:4:4 / 4:2:2 / 4:2:0) → interleaved RGB F32.

use whereat::At;
use zenpixels::{
    AlphaMode, ColorSpace, ImageSpec, PixelBuf, PlanarView, PlaneRole, Primaries, SampleLayout,
    SampleType, SignalRange, Transfer,
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

/// Reconstruct a full-range `RGBF32` [`PixelBuf`] from a 3-plane (`Y`, `Cb`, `Cr`)
/// or 2-plane semi-planar (`Y`, `CbCr` / `CrCb`, e.g. NV12 / P010) [`PlanarView`].
///
/// Honors each plane's [`BitPacking`](zenpixels::BitPacking) (`Full`, `U10_LSB`,
/// `U12_LSB`, `P010` MSB) and subsampling divisors (`4:4:4`, `4:2:2`, `4:2:0`).
pub fn ycbcr_to_rgb_f32(
    src: &PlanarView<'_>,
    matrix: YcbcrMatrix,
    range: SignalRange,
    primaries: Primaries,
    transfer: Transfer,
) -> Result<PixelBuf, At<ConvertError>> {
    let w = src.width;
    let h = src.height;
    if w == 0 || h == 0 || src.planes.len() < 2 {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    let y_plane = src
        .planes
        .iter()
        .find(|p| p.spec.role == PlaneRole::Y)
        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;

    let (semi, cb_plane, cr_plane) = if let Some(cbcr) = src
        .planes
        .iter()
        .find(|p| matches!(p.spec.role, PlaneRole::CbCr | PlaneRole::CrCb))
    {
        (true, cbcr, cbcr)
    } else {
        let cb = src
            .planes
            .iter()
            .find(|p| p.spec.role == PlaneRole::Cb)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        let cr = src
            .planes
            .iter()
            .find(|p| p.spec.role == PlaneRole::Cr)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        (false, cb, cr)
    };

    let dst_spec = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries,
            transfer,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let mut out = PixelBuf::new(w, h, dst_spec).map_err(|e| err_at((*e.error()).into()))?;

    let (kr, kb) = matrix.kr_kb();
    let kg = 1.0 - kr - kb;
    let cr_to_r = 2.0 * (1.0 - kr);
    let cb_to_b = 2.0 * (1.0 - kb);
    let cr_to_g = -2.0 * kr * (1.0 - kr) / kg;
    let cb_to_g = -2.0 * kb * (1.0 - kb) / kg;

    let y_max = y_plane.spec.packing.max_code(y_plane.spec.sample, 0) as f32;
    let c_max = cb_plane.spec.packing.max_code(cb_plane.spec.sample, 0) as f32;
    let y_bits = (32 - (y_max as u32).leading_zeros()) as i32;
    let c_bits = (32 - (c_max as u32).leading_zeros()) as i32;

    let (y_off, y_scale) = match range {
        SignalRange::Full => (0.0, 1.0 / y_max.max(1.0)),
        SignalRange::Narrow => {
            let shift = (y_bits - 8).max(0);
            let lo = (16u32 << shift) as f32;
            let span = (219u32 << shift) as f32;
            (lo, 1.0 / span)
        }
    };
    let (c_mid, c_scale) = match range {
        SignalRange::Full => {
            let mid = (1u32 << (c_bits - 1).max(0)) as f32;
            (mid, 1.0 / c_max.max(1.0))
        }
        SignalRange::Narrow => {
            let shift = (c_bits - 8).max(0);
            let mid = (128u32 << shift) as f32;
            let span = (224u32 << shift) as f32;
            (mid, 1.0 / span)
        }
    };

    let (chx, cvy) = (
        cb_plane.spec.subsampling.0.max(1) as u32,
        cb_plane.spec.subsampling.1.max(1) as u32,
    );
    let swapped = semi && cb_plane.spec.role == PlaneRole::CrCb;

    let mut out_view = out.as_view_mut();
    for y in 0..h {
        let y_row = y_plane.row(y);
        let cy = (y / cvy).min(cb_plane.height - 1);
        let cb_row = cb_plane.row(cy);
        let cr_row = cr_plane.row(cy);
        let dst_f32: &mut [f32] = out_view
            .row_as_mut::<f32>(y)
            .map_err(|e| err_at((*e.error()).into()))?;

        for x in 0..w {
            let y_code = read_plane_code(y_plane.spec, y_row, x as usize, 0) as f32;
            let cx = ((x / chx).min(cb_plane.width - 1)) as usize;
            let (cb_code, cr_code) = if semi {
                let c0 = read_plane_code(cb_plane.spec, cb_row, cx, 0) as f32;
                let c1 = read_plane_code(cb_plane.spec, cb_row, cx, 1) as f32;
                if swapped { (c1, c0) } else { (c0, c1) }
            } else {
                (
                    read_plane_code(cb_plane.spec, cb_row, cx, 0) as f32,
                    read_plane_code(cr_plane.spec, cr_row, cx, 0) as f32,
                )
            };

            let yn = (y_code - y_off) * y_scale;
            let cb = (cb_code - c_mid) * c_scale;
            let cr = (cr_code - c_mid) * c_scale;

            let o = (x as usize) * 3;
            dst_f32[o] = yn + cr_to_r * cr;
            dst_f32[o + 1] = yn + cr_to_g * cr + cb_to_g * cb;
            dst_f32[o + 2] = yn + cb_to_b * cb;
        }
    }
    Ok(out)
}

#[inline(always)]
fn read_plane_code(spec: zenpixels::PlaneSpec, row: &[u8], x: usize, comp: usize) -> u32 {
    let sz = spec.sample.byte_size();
    let elem = spec.bytes_per_element();
    let off = x * elem + comp * sz;
    let w = match spec.sample {
        SampleType::U8 => u32::from(row[off]),
        SampleType::U16 => u32::from(u16::from_ne_bytes([row[off], row[off + 1]])),
        SampleType::U32 => u32::from_ne_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]),
        _ => 0,
    };
    spec.packing.extract_code(spec.sample, w, comp)
}
