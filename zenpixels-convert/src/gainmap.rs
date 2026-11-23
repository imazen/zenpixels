//! ISO 21496-1 / UltraHDR dual-layer gain-map reconstruction and extraction.
//!
//! Operates directly on validated [`DualLayerHdr`] carriers (`base` + `gain_map` +
//! [`GainMapMetadata`]) using row-streaming conversion (O(width) scratch memory)
//! so UltraHDR JPEG, AVIF `tmap`, JXL, and PNG HDR share one exact implementation.

use alloc::vec::Vec;
use whereat::At;
use zenpixels::{
    AlphaMode, ChannelOrder, ColorSpace, DiffuseWhite, DualLayerHdr, GainMapMetadata, ImageSpec,
    PixelBuf, PixelView, Primaries, SampleLayout, SampleType, SignalRange, Transfer,
};

use crate::error::{ConvertError, err_at};
use crate::plan::RowPlan;
use crate::policy::ConvertPolicy;

/// Reconstruct a linear-light HDR [`PixelBuf`] (`RGBF32` / `RGBAF32`) from a
/// [`DualLayerHdr`] at the given display HDR headroom (in log2 stops, e.g.
/// `0.0` = SDR, `2.32` ≈ 5× / 1000-nit HDR display).
///
/// Follows ISO 21496-1:
/// - `weight = clamp((headroom - base_hdr_headroom) / (alternate_hdr_headroom - base_hdr_headroom), 0, 1)`
/// - `g = pow(gain_sample, 1 / gamma)`
/// - `log2_boost = lerp(min_boost_stops, max_boost_stops, g)`
/// - `hdr_linear = (base_linear + base_offset) * exp2(log2_boost * weight) - alternate_offset`
///
/// Streams row-by-row in O(width) scratch memory (no full-image linearized base copy),
/// supports both single-channel and 3-channel gain maps, and bilinearly upsamples
/// lower-resolution gain maps to match `base`.
pub fn apply_gain_map(
    dual: &DualLayerHdr,
    display_headroom_stops: f32,
) -> Result<PixelBuf, At<ConvertError>> {
    if !display_headroom_stops.is_finite() {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    let meta = dual.metadata();
    if !meta.use_base_color_space {
        // Alternate-primaries gain-map application requires an alternate color space descriptor.
        return Err(err_at(ConvertError::UnsupportedRoute));
    }

    let base = dual.base();
    let gain_map = dual.gain_map();
    let w = base.width();
    let h = base.height();
    let gm_w = gain_map.width();
    let gm_h = gain_map.height();
    if w == 0 || h == 0 || gm_w == 0 || gm_h == 0 {
        return Err(err_at(ConvertError::DimensionMismatch));
    }

    let (base_primaries, base_white) = match base.spec().color {
        ColorSpace::Rgb {
            primaries, white, ..
        } => (primaries, white.unwrap_or(DiffuseWhite::BT2408)),
        ColorSpace::Gray { white, .. } => (Primaries::Bt709, white.unwrap_or(DiffuseWhite::BT2408)),
        _ => return Err(err_at(ConvertError::UnsupportedRoute)),
    };
    let has_alpha = base.spec().layout.order.has_alpha_slot();
    let ch = if has_alpha { 4usize } else { 3usize };
    let linear_spec = ImageSpec::new(
        if has_alpha {
            SampleLayout::RGBAF32
        } else {
            SampleLayout::RGBF32
        },
        ColorSpace::Rgb {
            primaries: base_primaries,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: if has_alpha {
                AlphaMode::Straight
            } else {
                AlphaMode::None
            },
            white: Some(base_white),
        },
    );

    let policy = ConvertPolicy::working_linear().with_sdr_white(base_white);
    let base_row_plan = RowPlan::build(base.spec().clone(), linear_spec.clone(), &policy, None)?;

    let gm_layout = gain_map.spec().layout;
    if !gm_layout.packing.is_full()
        || !matches!(
            gm_layout.sample,
            SampleType::U8 | SampleType::U16 | SampleType::F32
        )
        || !matches!(
            gm_layout.order,
            ChannelOrder::Gray | ChannelOrder::Rgb | ChannelOrder::Rgba
        )
    {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    let gm_is_rgb = gm_layout.channels() >= 3;
    let gm_view = gain_map.as_view();

    let span = meta.alternate_hdr_headroom - meta.base_hdr_headroom;
    let weight = if span.abs() < 1e-6 {
        if display_headroom_stops >= meta.alternate_hdr_headroom {
            1.0
        } else {
            0.0
        }
    } else {
        ((display_headroom_stops - meta.base_hdr_headroom) / span).clamp(0.0, 1.0)
    };

    let row_floats = (w as usize)
        .checked_mul(ch)
        .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
    let mut base_lin_row: Vec<f32> = Vec::new();
    base_lin_row
        .try_reserve_exact(row_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    base_lin_row.resize(row_floats, 0.0f32);

    let mut out = PixelBuf::new(w, h, linear_spec).map_err(|e| err_at((*e.error()).into()))?;
    let base_view = base.as_view();
    let mut out_view = out.as_view_mut();

    for y in 0..h {
        base_row_plan.convert_row(
            base_view.row(y),
            bytemuck::cast_slice_mut(&mut base_lin_row),
            w,
            y,
        )?;
        let dst_row: &mut [f32] = out_view
            .row_as_mut::<f32>(y)
            .map_err(|e| err_at((*e.error()).into()))?;

        let gy_f = if h == gm_h {
            y as f32
        } else {
            ((y as f32 + 0.5) * (gm_h as f32 / h as f32) - 0.5).clamp(0.0, (gm_h - 1) as f32)
        };
        let gy0 = (gy_f as u32).min(gm_h - 1);
        let gy1 = (gy0 + 1).min(gm_h - 1);
        let fy = gy_f - gy0 as f32;

        for x in 0..w {
            let gx_f = if w == gm_w {
                x as f32
            } else {
                ((x as f32 + 0.5) * (gm_w as f32 / w as f32) - 0.5).clamp(0.0, (gm_w - 1) as f32)
            };
            let gx0 = (gx_f as u32).min(gm_w - 1);
            let gx1 = (gx0 + 1).min(gm_w - 1);
            let fx = gx_f - gx0 as f32;

            let gm_rgb = sample_gain_bilinear(&gm_view, gx0, gx1, gy0, gy1, fx, fy, gm_is_rgb);
            let o = (x as usize) * ch;
            for c in 0..3 {
                let g = libm::powf(gm_rgb[c].clamp(0.0, 1.0), 1.0 / meta.gamma[c]);
                let log2_boost = meta.min_boost_stops[c]
                    + (meta.max_boost_stops[c] - meta.min_boost_stops[c]) * g;
                let gain = libm::exp2f(log2_boost * weight);
                dst_row[o + c] =
                    (base_lin_row[o + c] + meta.base_offset[c]) * gain - meta.alternate_offset[c];
            }
            if has_alpha {
                dst_row[o + 3] = base_lin_row[o + 3];
            }
        }
    }
    Ok(out)
}

/// Extract an ISO 21496-1 [`DualLayerHdr`] from an SDR base view and an HDR
/// linear view of matching dimensions in O(width) row scratch memory.
///
/// Produces an 8-bit grayscale gain map (`GRAY8`) encoding the per-pixel log2
/// luminance ratio under `metadata` (with channel-0 parameters replicated cleanly
/// across all three channels of the returned metadata so `apply_gain_map` applies
/// the exact extracted curve symmetrically).
pub fn extract_gain_map(
    sdr_base: &PixelView<'_>,
    hdr_src: &PixelView<'_>,
    metadata: GainMapMetadata,
) -> Result<DualLayerHdr, At<ConvertError>> {
    let w = sdr_base.width();
    let h = sdr_base.rows();
    if w == 0 || h == 0 || hdr_src.width() != w || hdr_src.rows() != h {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    if !metadata.base_hdr_headroom.is_finite()
        || !metadata.alternate_hdr_headroom.is_finite()
        || !metadata.gamma[0].is_finite()
        || metadata.gamma[0] <= 0.0
        || !metadata.min_boost_stops[0].is_finite()
        || !metadata.max_boost_stops[0].is_finite()
        || metadata.min_boost_stops[0] > metadata.max_boost_stops[0]
        || !metadata.base_offset[0].is_finite()
        || metadata.base_offset[0] < 0.0
        || !metadata.alternate_offset[0].is_finite()
        || metadata.alternate_offset[0] < 0.0
    {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }

    let (base_primaries, base_white) = match sdr_base.spec().color {
        ColorSpace::Rgb {
            primaries, white, ..
        } => (primaries, white.unwrap_or(DiffuseWhite::BT2408)),
        ColorSpace::Gray { white, .. } => (Primaries::Bt709, white.unwrap_or(DiffuseWhite::BT2408)),
        _ => return Err(err_at(ConvertError::UnsupportedRoute)),
    };
    let linear_rgb = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: base_primaries,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: Some(base_white),
        },
    );
    let policy = ConvertPolicy::working_linear()
        .with_sdr_white(base_white)
        .with_alpha(crate::policy::AlphaPolicy::DiscardUnchecked);
    let sdr_plan = RowPlan::build(sdr_base.spec().clone(), linear_rgb.clone(), &policy, None)?;
    let hdr_plan = RowPlan::build(hdr_src.spec().clone(), linear_rgb, &policy, None)?;

    let mut gain_buf = PixelBuf::new(
        w,
        h,
        ImageSpec::new(SampleLayout::GRAY8, ColorSpace::GRAY_LINEAR),
    )
    .map_err(|e| err_at((*e.error()).into()))?;

    let row_floats = (w as usize)
        .checked_mul(3)
        .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
    let mut sdr_row: Vec<f32> = Vec::new();
    sdr_row
        .try_reserve_exact(row_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    sdr_row.resize(row_floats, 0.0f32);
    let mut hdr_row: Vec<f32> = Vec::new();
    hdr_row
        .try_reserve_exact(row_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    hdr_row.resize(row_floats, 0.0f32);

    let luma = base_primaries.luma_weights();
    let mut gm_v = gain_buf.as_view_mut();

    let min_b = metadata.min_boost_stops[0];
    let max_b = metadata.max_boost_stops[0];
    let span = (max_b - min_b).max(1e-6);
    let gamma = metadata.gamma[0];
    let k_base = metadata.base_offset[0];
    let k_alt = metadata.alternate_offset[0];

    for y in 0..h {
        sdr_plan.convert_row(
            sdr_base.row(y),
            bytemuck::cast_slice_mut(&mut sdr_row),
            w,
            y,
        )?;
        hdr_plan.convert_row(hdr_src.row(y), bytemuck::cast_slice_mut(&mut hdr_row), w, y)?;
        let gr = gm_v.row_mut(y);
        for x in 0..w as usize {
            let ys = (luma[0] * sdr_row[x * 3]
                + luma[1] * sdr_row[x * 3 + 1]
                + luma[2] * sdr_row[x * 3 + 2])
                .max(0.0);
            let yh = (luma[0] * hdr_row[x * 3]
                + luma[1] * hdr_row[x * 3 + 1]
                + luma[2] * hdr_row[x * 3 + 2])
                .max(0.0);
            let ratio = ((yh + k_alt) / (ys + k_base).max(1e-6)).max(1e-6);
            let log2_r = libm::log2f(ratio);
            let norm = ((log2_r - min_b) / span).clamp(0.0, 1.0);
            let encoded = libm::powf(norm, gamma);
            gr[x] = (encoded * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }

    // Replicate the single-channel luminance extraction parameters symmetrically across all 3 channels.
    let norm_meta = GainMapMetadata {
        min_boost_stops: [min_b; 3],
        max_boost_stops: [max_b; 3],
        gamma: [gamma; 3],
        base_offset: [k_base; 3],
        alternate_offset: [k_alt; 3],
        base_hdr_headroom: metadata.base_hdr_headroom,
        alternate_hdr_headroom: metadata.alternate_hdr_headroom,
        use_base_color_space: true,
    };

    let base_owned = sdr_base
        .try_to_tight_buf()
        .map_err(|e| err_at((*e.error()).into()))?;
    DualLayerHdr::new(base_owned, gain_buf, norm_meta).map_err(|e| err_at((*e.error()).into()))
}

fn sample_gain_pixel(view: &PixelView<'_>, x: u32, y: u32, is_rgb: bool) -> [f32; 3] {
    let row = view.row(y);
    let ch = view.spec().layout.channels();
    let xi = x as usize;
    let read_ch = |c: usize| -> f32 {
        match view.spec().layout.sample {
            SampleType::U8 => f32::from(row[xi * ch + c]) / 255.0,
            SampleType::U16 => {
                let o = (xi * ch + c) * 2;
                f32::from(u16::from_ne_bytes([row[o], row[o + 1]])) / 65535.0
            }
            SampleType::F32 => {
                let o = (xi * ch + c) * 4;
                f32::from_ne_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]])
            }
            _ => 0.0,
        }
    };
    if is_rgb {
        [read_ch(0), read_ch(1), read_ch(2)]
    } else {
        let v = read_ch(0);
        [v, v, v]
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_gain_bilinear(
    view: &PixelView<'_>,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    fx: f32,
    fy: f32,
    is_rgb: bool,
) -> [f32; 3] {
    let p00 = sample_gain_pixel(view, x0, y0, is_rgb);
    let p10 = sample_gain_pixel(view, x1, y0, is_rgb);
    let p01 = sample_gain_pixel(view, x0, y1, is_rgb);
    let p11 = sample_gain_pixel(view, x1, y1, is_rgb);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let top = p00[c] + (p10[c] - p00[c]) * fx;
        let bot = p01[c] + (p11[c] - p01[c]) * fx;
        out[c] = top + (bot - top) * fy;
    }
    out
}
