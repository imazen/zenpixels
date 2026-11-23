//! ISO 21496-1 / UltraHDR dual-layer gain-map reconstruction and extraction.
//!
//! Operates directly on [`DualLayerHdr`] carriers (`base` + `gain_map` +
//! [`GainMapMetadata`]) so UltraHDR JPEG, AVIF `tmap`, JXL, and PNG HDR share
//! one exact implementation.

use whereat::At;
use zenpixels::{
    AlphaMode, ColorSpace, DiffuseWhite, DualLayerHdr, GainMapMetadata, ImageSpec, PeakSource,
    PixelBuf, PixelView, Primaries, SampleLayout, SignalRange, Transfer,
};

use crate::error::{ConvertError, err_at};
use crate::plan::convert_buf;
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
/// Supports both single-channel and 3-channel gain maps, and automatically
/// bilinearly upsamples lower-resolution gain maps to match `base`.
pub fn apply_gain_map(
    dual: &DualLayerHdr,
    display_headroom_stops: f32,
) -> Result<PixelBuf, At<ConvertError>> {
    let w = dual.base.width();
    let h = dual.base.height();
    let meta = dual.metadata;

    let base_primaries = match dual.base.spec().color {
        ColorSpace::Rgb { primaries, .. } => primaries,
        _ => Primaries::Bt709,
    };
    let has_alpha = dual.base.spec().layout.order.has_alpha_slot();
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
            white: Some(DiffuseWhite::BT2408),
        },
    );

    let policy = ConvertPolicy::working_linear();
    let base_lin = convert_buf(
        &dual.base.as_view(),
        linear_spec.clone(),
        &policy,
        PeakSource::Measured,
    )?;

    // Convert gain map to linear F32 [0, 1] (no EOTF: gain map samples are encoded codes in [0, 1]).
    let gm_is_rgb = dual.gain_map.spec().layout.channels() >= 3;
    let gm_w = dual.gain_map.width();
    let gm_h = dual.gain_map.height();
    let gm_view = dual.gain_map.as_view();

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

    let mut out = PixelBuf::new(w, h, linear_spec).map_err(|e| err_at((*e.error()).into()))?;
    let base_view = base_lin.as_view();
    let mut out_view = out.as_view_mut();
    let ch = if has_alpha { 4 } else { 3 };

    for y in 0..h {
        let src_row: &[f32] = base_view
            .row_as::<f32>(y)
            .map_err(|e| err_at((*e.error()).into()))?;
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
                    (src_row[o + c] + meta.base_offset[c]) * gain - meta.alternate_offset[c];
            }
            if has_alpha {
                dst_row[o + 3] = src_row[o + 3];
            }
        }
    }
    Ok(out)
}

/// Extract an ISO 21496-1 [`DualLayerHdr`] from an SDR base view and an HDR
/// linear view of matching dimensions.
///
/// Produces an 8-bit grayscale gain map (`GRAY8`) encoding the per-pixel log2
/// luminance ratio under `metadata`.
pub fn extract_gain_map(
    sdr_base: &PixelView<'_>,
    hdr_src: &PixelView<'_>,
    metadata: GainMapMetadata,
) -> Result<DualLayerHdr, At<ConvertError>> {
    let w = sdr_base.width();
    let h = sdr_base.rows();
    if hdr_src.width() != w || hdr_src.rows() != h {
        return Err(err_at(ConvertError::DimensionMismatch));
    }

    let base_primaries = match sdr_base.spec().color {
        ColorSpace::Rgb { primaries, .. } => primaries,
        _ => Primaries::Bt709,
    };
    let linear_rgb = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: base_primaries,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: Some(DiffuseWhite::BT2408),
        },
    );
    let policy =
        ConvertPolicy::working_linear().with_alpha(crate::policy::AlphaPolicy::DiscardUnchecked);
    let sdr_lin = convert_buf(sdr_base, linear_rgb.clone(), &policy, PeakSource::Measured)?;
    let hdr_lin = convert_buf(hdr_src, linear_rgb, &policy, PeakSource::Measured)?;

    let mut gain_buf = PixelBuf::new(
        w,
        h,
        ImageSpec::new(SampleLayout::GRAY8, ColorSpace::GRAY_LINEAR),
    )
    .map_err(|e| err_at((*e.error()).into()))?;

    let luma = base_primaries.luma_weights();
    let sdr_v = sdr_lin.as_view();
    let hdr_v = hdr_lin.as_view();
    let mut gm_v = gain_buf.as_view_mut();

    let min_b = metadata.min_boost_stops[0];
    let span = (metadata.max_boost_stops[0] - min_b).max(1e-6);
    let gamma = metadata.gamma[0];
    let k_base = metadata.base_offset[0];
    let k_alt = metadata.alternate_offset[0];

    for y in 0..h {
        let sr: &[f32] = sdr_v
            .row_as::<f32>(y)
            .map_err(|e| err_at((*e.error()).into()))?;
        let hr: &[f32] = hdr_v
            .row_as::<f32>(y)
            .map_err(|e| err_at((*e.error()).into()))?;
        let gr = gm_v.row_mut(y);
        for x in 0..w as usize {
            let ys =
                (luma[0] * sr[x * 3] + luma[1] * sr[x * 3 + 1] + luma[2] * sr[x * 3 + 2]).max(0.0);
            let yh =
                (luma[0] * hr[x * 3] + luma[1] * hr[x * 3 + 1] + luma[2] * hr[x * 3 + 2]).max(0.0);
            let ratio = ((yh + k_alt) / (ys + k_base).max(1e-6)).max(1e-6);
            let log2_r = libm::log2f(ratio);
            let norm = ((log2_r - min_b) / span).clamp(0.0, 1.0);
            let encoded = libm::powf(norm, gamma);
            gr[x] = (encoded * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }

    let base_owned = sdr_base
        .try_to_tight_buf()
        .map_err(|e| err_at((*e.error()).into()))?;
    DualLayerHdr::new(base_owned, gain_buf, metadata).map_err(|e| err_at((*e.error()).into()))
}

fn sample_gain_pixel(view: &PixelView<'_>, x: u32, y: u32, is_rgb: bool) -> [f32; 3] {
    let row = view.row(y);
    let ch = view.spec().layout.channels();
    let xi = x as usize;
    let read_ch = |c: usize| -> f32 {
        match view.spec().layout.sample {
            zenpixels::SampleType::U8 => f32::from(row[xi * ch + c]) / 255.0,
            zenpixels::SampleType::U16 => {
                let o = (xi * ch + c) * 2;
                f32::from(u16::from_ne_bytes([row[o], row[o + 1]])) / 65535.0
            }
            zenpixels::SampleType::F32 => {
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
