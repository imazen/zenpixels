//! ISO 21496-1 / UltraHDR dual-layer gain-map reconstruction and extraction.
//!
//! Operates on validated [`DualLayerHdr`] and zero-copy [`DualLayerHdrView`] carriers
//! using row-streaming conversion (O(width) scratch memory):
//! - Supports both **Forward** (SDR base → HDR alternate, `H_alt > H_base`) and
//!   **Reverse** (HDR base → SDR alternate, `H_base > H_alt`) directions via exact
//!   rational headroom comparison and signed weight.
//! - Bypasses gain-map sampling and additive offsets at zero weight (`weight == 0.0`)
//!   for exact base-rendition identity even when `base_offset != alternate_offset`.
//! - Supports both `GainMapColorSpace::Base` and `GainMapColorSpace::Alternate`
//!   (converting base into alternate linear RGB primaries *before* channelwise gain).
//! - Keeps gain-map image channels (1 or 3 numeric `ColorSpace::Untagged` samples,
//!   honoring `Full`/`Lsb`/`Msb` bit-packing) strictly independent of metadata record
//!   arity (`GainMapChannels::Shared` vs `PerChannel`).

use alloc::vec::Vec;
use whereat::At;
use zenpixels::{
    AlphaMode, ColorSpace, DiffuseWhite, DualLayerHdr, DualLayerHdrView, GainMapColorSpace,
    GainMapDirection, GainMapMetadata, ImageSpec, PixelBuf, PixelView, Primaries, SampleLayout,
    SampleType, SignalRange, Transfer,
};

use crate::error::{ConvertError, err_at};
use crate::plan::RowPlan;
use crate::policy::ConvertPolicy;

fn resolve_linear_rgb_primaries(color: &ColorSpace) -> Option<(Primaries, Option<DiffuseWhite>)> {
    match color {
        ColorSpace::Rgb {
            primaries, white, ..
        } if *primaries != Primaries::Unknown => Some((*primaries, *white)),
        ColorSpace::Gray { white, .. } => Some((Primaries::Bt709, *white)),
        ColorSpace::Icc { profile, .. } => {
            let id = zenpixels::icc::IccIdentification::from_icc(profile)?;
            if id.primaries != Primaries::Unknown {
                return Some((id.primaries, None));
            }
            None
        }
        _ => None,
    }
}

/// Reconstruct a linear-light [`PixelBuf`] (`RGBF32` / `RGBAF32`) from an owned
/// [`DualLayerHdr`] at `display_headroom_stops` (log₂ stops, e.g. `0.0` = SDR,
/// `2.32` ≈ 5× / 1000-nit HDR display).
pub fn apply_gain_map(
    dual: &DualLayerHdr,
    display_headroom_stops: f32,
) -> Result<PixelBuf, At<ConvertError>> {
    apply_gain_map_view(
        &dual.as_view(),
        display_headroom_stops,
        &ConvertPolicy::working_linear(),
    )
}

/// Reconstruct a linear-light [`PixelBuf`] (`RGBF32` / `RGBAF32`) from a borrowed
/// [`DualLayerHdrView`] using `policy` (allowing CMS backends for ICC-described bases
/// or alternate color spaces) in O(width) row scratch memory.
///
/// Follows ISO 21496-1:
/// - `direction = sign(H_alt - H_base)` (exact rational comparison)
/// - `amount = clamp((headroom - H_base) / (H_alt - H_base), 0, 1)`
/// - If `amount == 0.0`, bypasses map sampling and offsets and returns converted base directly.
/// - Otherwise `weight = direction * amount`, `g = pow(sample, 1 / gamma)`,
///   `L = lerp(min_gain_stops, max_gain_stops, g)`,
///   `out = (base_linear + base_offset) * exp2(L * weight) - alternate_offset`.
pub fn apply_gain_map_view(
    dual: &DualLayerHdrView<'_>,
    display_headroom_stops: f32,
    policy: &ConvertPolicy,
) -> Result<PixelBuf, At<ConvertError>> {
    if !display_headroom_stops.is_finite() {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    let meta = dual.metadata();
    let direction = meta
        .direction()
        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;

    let base = dual.base();
    let gain_map = dual.gain_map();
    let w = base.width();
    let h = base.rows();
    let gm_w = gain_map.width();
    let gm_h = gain_map.rows();
    if w == 0 || h == 0 || gm_w == 0 || gm_h == 0 {
        return Err(err_at(ConvertError::DimensionMismatch));
    }

    let (_, base_white_opt) =
        resolve_linear_rgb_primaries(&base.spec().color).unwrap_or((Primaries::Bt709, None));
    let base_white = base_white_opt
        .or(policy.sdr_white)
        .unwrap_or(DiffuseWhite::BT2408);

    // Resolve the linear RGB application primaries per ISO 21496-1 `application_space`.
    let app_primaries = match meta.application_space {
        GainMapColorSpace::Base => resolve_linear_rgb_primaries(&base.spec().color)
            .map(|(p, _)| p)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?,
        GainMapColorSpace::Alternate => {
            let alt = dual
                .alternate_color_space()
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            resolve_linear_rgb_primaries(alt)
                .map(|(p, _)| p)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?
        }
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
            primaries: app_primaries,
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

    let mut row_pol = policy.clone();
    row_pol.sdr_white = Some(base_white);
    let base_row_plan = RowPlan::build(base.spec().clone(), linear_spec.clone(), &row_pol, None)?;

    let gm_layout = gain_map.spec().layout;
    let gm_is_rgb = gm_layout.channels() == 3;
    let gm_max = [
        gm_layout.packing.max_code(gm_layout.sample, 0).max(1) as f32,
        gm_layout.packing.max_code(gm_layout.sample, 1).max(1) as f32,
        gm_layout.packing.max_code(gm_layout.sample, 2).max(1) as f32,
    ];

    // Exact rational headroom span -> f64 division before clamping to [0, 1].
    let h_base = meta.base_hdr_headroom.to_f64();
    let h_alt = meta.alternate_hdr_headroom.to_f64();
    let amount =
        ((f64::from(display_headroom_stops) - h_base) / (h_alt - h_base)).clamp(0.0, 1.0) as f32;

    let mut out = PixelBuf::new(w, h, linear_spec).map_err(|e| err_at((*e.error()).into()))?;
    let mut out_view = out.as_view_mut();

    // ISO 21496-1 zero-weight identity bypass: return base rendition directly without
    // sampling the gain map or applying base_offset - alternate_offset!
    if amount == 0.0 {
        for y in 0..h {
            base_row_plan.convert_row(base.row(y), out_view.row_mut(y), w, y)?;
        }
        return Ok(out);
    }

    let sign = match direction {
        GainMapDirection::Forward => 1.0_f32,
        GainMapDirection::Reverse => -1.0_f32,
    };
    let signed_weight = sign * amount;

    // Pre-project per-channel ISO parameters to f32 once per image.
    let mut min_g = [0.0f32; 3];
    let mut span_g = [0.0f32; 3];
    let mut inv_gamma = [1.0f32; 3];
    let mut k_base = [0.0f32; 3];
    let mut k_alt = [0.0f32; 3];
    for c in 0..3 {
        let comp = meta.channel(c);
        min_g[c] = comp.min_gain_stops.to_f32();
        span_g[c] = comp.max_gain_stops.to_f32() - min_g[c];
        inv_gamma[c] = 1.0 / comp.gamma.to_f32();
        k_base[c] = comp.base_offset.to_f32();
        k_alt[c] = comp.alternate_offset.to_f32();
    }

    let row_floats = (w as usize)
        .checked_mul(ch)
        .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
    let mut base_lin_row: Vec<f32> = Vec::new();
    base_lin_row
        .try_reserve_exact(row_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    base_lin_row.resize(row_floats, 0.0f32);

    let same_size = w == gm_w && h == gm_h;

    for y in 0..h {
        base_row_plan.convert_row(
            base.row(y),
            bytemuck::cast_slice_mut(&mut base_lin_row),
            w,
            y,
        )?;
        let dst_row: &mut [f32] = out_view
            .row_as_mut::<f32>(y)
            .map_err(|e| err_at((*e.error()).into()))?;

        if same_size {
            for x in 0..w {
                let gm_rgb = sample_gain_pixel(gain_map, x, y, gm_is_rgb, gm_max);
                let o = (x as usize) * ch;
                for c in 0..3 {
                    let s = gm_rgb[c].clamp(0.0, 1.0);
                    let g = if inv_gamma[c] == 1.0 {
                        s
                    } else {
                        libm::powf(s, inv_gamma[c])
                    };
                    let log2_boost = min_g[c] + span_g[c] * g;
                    let gain = libm::exp2f(log2_boost * signed_weight);
                    dst_row[o + c] = (base_lin_row[o + c] + k_base[c]) * gain - k_alt[c];
                }
                if has_alpha {
                    dst_row[o + 3] = base_lin_row[o + 3];
                }
            }
        } else {
            let gy_f =
                ((y as f32 + 0.5) * (gm_h as f32 / h as f32) - 0.5).clamp(0.0, (gm_h - 1) as f32);
            let gy0 = (gy_f as u32).min(gm_h - 1);
            let gy1 = (gy0 + 1).min(gm_h - 1);
            let fy = gy_f - gy0 as f32;

            for x in 0..w {
                let gx_f = ((x as f32 + 0.5) * (gm_w as f32 / w as f32) - 0.5)
                    .clamp(0.0, (gm_w - 1) as f32);
                let gx0 = (gx_f as u32).min(gm_w - 1);
                let gx1 = (gx0 + 1).min(gm_w - 1);
                let fx = gx_f - gx0 as f32;

                // ISO 21496-1 / Android Ultra HDR: interpolate encoded gain samples before gamma/exp2.
                let gm_rgb =
                    sample_gain_bilinear(gain_map, gx0, gx1, gy0, gy1, fx, fy, gm_is_rgb, gm_max);
                let o = (x as usize) * ch;
                for c in 0..3 {
                    let s = gm_rgb[c].clamp(0.0, 1.0);
                    let g = if inv_gamma[c] == 1.0 {
                        s
                    } else {
                        libm::powf(s, inv_gamma[c])
                    };
                    let log2_boost = min_g[c] + span_g[c] * g;
                    let gain = libm::exp2f(log2_boost * signed_weight);
                    dst_row[o + c] = (base_lin_row[o + c] + k_base[c]) * gain - k_alt[c];
                }
                if has_alpha {
                    dst_row[o + 3] = base_lin_row[o + 3];
                }
            }
        }
    }
    Ok(out)
}

/// Extract an ISO 21496-1 [`DualLayerHdr`] from a base rendition view and an alternate
/// rendition view of matching dimensions in O(width) row scratch memory.
///
/// - Honors `metadata.channels`: `GainMapChannels::Shared` extracts a 1-channel numeric
///   `ColorSpace::Untagged { channels: 1 }` luminance gain map; `GainMapChannels::PerChannel`
///   extracts a 3-channel numeric `ColorSpace::Untagged { channels: 3 }` RGB gain map.
/// - Honors both `GainMapDirection::Forward` (SDR base → HDR alternate) and
///   `GainMapDirection::Reverse` (HDR base → SDR alternate) using exact signed ratios.
/// - Honors `GainMapColorSpace::Base` and `GainMapColorSpace::Alternate` (converting both
///   renditions into the selected application primaries before computing the ratio, and
///   attaching `alternate_src.spec().color` as `alternate_color_space`).
/// - Preserves `metadata` verbatim without silently rewriting parameters.
pub fn extract_gain_map(
    base_src: &PixelView<'_>,
    alternate_src: &PixelView<'_>,
    metadata: GainMapMetadata,
) -> Result<DualLayerHdr, At<ConvertError>> {
    let w = base_src.width();
    let h = base_src.rows();
    if w == 0 || h == 0 || alternate_src.width() != w || alternate_src.rows() != h {
        return Err(err_at(ConvertError::DimensionMismatch));
    }
    if !metadata.is_valid() {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }
    let direction = metadata
        .direction()
        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;

    let (base_primaries, base_white_opt) = resolve_linear_rgb_primaries(&base_src.spec().color)
        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
    let base_white = base_white_opt.unwrap_or(DiffuseWhite::BT2408);

    let app_primaries = match metadata.application_space {
        GainMapColorSpace::Base => base_primaries,
        GainMapColorSpace::Alternate => resolve_linear_rgb_primaries(&alternate_src.spec().color)
            .map(|(p, _)| p)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?,
    };

    let linear_rgb = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: app_primaries,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: Some(base_white),
        },
    );
    let policy = ConvertPolicy::working_linear()
        .with_sdr_white(base_white)
        .with_alpha(crate::policy::AlphaPolicy::DiscardUnchecked);
    let base_plan = RowPlan::build(base_src.spec().clone(), linear_rgb.clone(), &policy, None)?;
    let alt_plan = RowPlan::build(alternate_src.spec().clone(), linear_rgb, &policy, None)?;

    let multichannel = metadata.channels.is_multichannel();
    let gm_spec = if multichannel {
        ImageSpec::new(
            SampleLayout::RGB8,
            ColorSpace::Untagged {
                channels: 3,
                alpha: AlphaMode::None,
            },
        )
    } else {
        ImageSpec::new(
            SampleLayout::GRAY8,
            ColorSpace::Untagged {
                channels: 1,
                alpha: AlphaMode::None,
            },
        )
    };
    let mut gain_buf = PixelBuf::new(w, h, gm_spec).map_err(|e| err_at((*e.error()).into()))?;

    let row_floats = (w as usize)
        .checked_mul(3)
        .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
    let mut base_row: Vec<f32> = Vec::new();
    base_row
        .try_reserve_exact(row_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    base_row.resize(row_floats, 0.0f32);
    let mut alt_row: Vec<f32> = Vec::new();
    alt_row
        .try_reserve_exact(row_floats)
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    alt_row.resize(row_floats, 0.0f32);

    let sign = match direction {
        GainMapDirection::Forward => 1.0_f32,
        GainMapDirection::Reverse => -1.0_f32,
    };
    let luma = app_primaries.luma_weights();
    let mut gm_v = gain_buf.as_view_mut();

    let mut min_g = [0.0f32; 3];
    let mut span_g = [0.0f32; 3];
    let mut gamma = [1.0f32; 3];
    let mut k_base = [0.0f32; 3];
    let mut k_alt = [0.0f32; 3];
    for c in 0..3 {
        let comp = metadata.channel(c);
        min_g[c] = comp.min_gain_stops.to_f32();
        span_g[c] = comp.max_gain_stops.to_f32() - min_g[c];
        gamma[c] = comp.gamma.to_f32();
        k_base[c] = comp.base_offset.to_f32();
        k_alt[c] = comp.alternate_offset.to_f32();
    }

    for y in 0..h {
        base_plan.convert_row(
            base_src.row(y),
            bytemuck::cast_slice_mut(&mut base_row),
            w,
            y,
        )?;
        alt_plan.convert_row(
            alternate_src.row(y),
            bytemuck::cast_slice_mut(&mut alt_row),
            w,
            y,
        )?;
        let gr = gm_v.row_mut(y);
        if multichannel {
            for x in 0..w as usize {
                let o = x * 3;
                for c in 0..3 {
                    let b = base_row[o + c].max(0.0);
                    let a = alt_row[o + c].max(0.0);
                    let denom = (b + k_base[c]).max(1e-6);
                    let numer = (a + k_alt[c]).max(1e-6);
                    let log2_r = sign * libm::log2f(numer / denom);
                    let norm = if span_g[c].abs() < 1e-12 {
                        0.0
                    } else {
                        ((log2_r - min_g[c]) / span_g[c]).clamp(0.0, 1.0)
                    };
                    let encoded = if gamma[c] == 1.0 {
                        norm
                    } else {
                        libm::powf(norm, gamma[c])
                    };
                    gr[o + c] = (encoded * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        } else {
            for (x, out_px) in gr.iter_mut().enumerate().take(w as usize) {
                let o = x * 3;
                let yb =
                    (luma[0] * base_row[o] + luma[1] * base_row[o + 1] + luma[2] * base_row[o + 2])
                        .max(0.0);
                let ya =
                    (luma[0] * alt_row[o] + luma[1] * alt_row[o + 1] + luma[2] * alt_row[o + 2])
                        .max(0.0);
                let denom = (yb + k_base[0]).max(1e-6);
                let numer = (ya + k_alt[0]).max(1e-6);
                let log2_r = sign * libm::log2f(numer / denom);
                let norm = if span_g[0].abs() < 1e-12 {
                    0.0
                } else {
                    ((log2_r - min_g[0]) / span_g[0]).clamp(0.0, 1.0)
                };
                let encoded = if gamma[0] == 1.0 {
                    norm
                } else {
                    libm::powf(norm, gamma[0])
                };
                *out_px = (encoded * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }

    let base_owned = base_src
        .try_to_tight_buf()
        .map_err(|e| err_at((*e.error()).into()))?;
    let dual = DualLayerHdr::new(base_owned, gain_buf, metadata)
        .map_err(|e| err_at((*e.error()).into()))?;
    if metadata.application_space == GainMapColorSpace::Alternate
        || alternate_src.spec().color != base_src.spec().color
    {
        dual.with_alternate_color_space(alternate_src.spec().color.clone())
            .map_err(|e| err_at((*e.error()).into()))
    } else {
        Ok(dual)
    }
}

fn sample_gain_pixel(
    view: &PixelView<'_>,
    x: u32,
    y: u32,
    is_rgb: bool,
    max_code: [f32; 3],
) -> [f32; 3] {
    let row = view.row(y);
    let layout = view.spec().layout;
    let ch = layout.channels();
    let xi = x as usize;
    let read_ch = |c: usize| -> f32 {
        match layout.sample {
            SampleType::U8 => {
                let code =
                    layout
                        .packing
                        .extract_code(SampleType::U8, u32::from(row[xi * ch + c]), c);
                (code as f32) / max_code[c]
            }
            SampleType::U16 => {
                let o = (xi * ch + c) * 2;
                let w = u32::from(u16::from_ne_bytes([row[o], row[o + 1]]));
                let code = layout.packing.extract_code(SampleType::U16, w, c);
                (code as f32) / max_code[c]
            }
            SampleType::F32 => {
                let o = (xi * ch + c) * 4;
                let v = f32::from_ne_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]]);
                if v.is_finite() { v } else { 0.0 }
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
    max_code: [f32; 3],
) -> [f32; 3] {
    let p00 = sample_gain_pixel(view, x0, y0, is_rgb, max_code);
    let p10 = sample_gain_pixel(view, x1, y0, is_rgb, max_code);
    let p01 = sample_gain_pixel(view, x0, y1, is_rgb, max_code);
    let p11 = sample_gain_pixel(view, x1, y1, is_rgb, max_code);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let top = p00[c] + (p10[c] - p00[c]) * fx;
        let bot = p01[c] + (p11[c] - p01[c]) * fx;
        out[c] = top + (bot - top) * fy;
    }
    out
}
