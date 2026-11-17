//! BT.2100-3 Table 5 display rendering, resolved by the plan or CMS adapter.
//! OOTF structure follows zentone f32e091c's hlg module (same ownership).
//! F64 evaluation and analytic curves avoid a second approximation error budget.
use crate::ChannelLayout;
#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
use crate::{ChannelType, ColorProfileSource, PixelFormat};
#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
use alloc::boxed::Box;
#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
use alloc::vec::Vec;
#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
use whereat::At;
use zenpixels::hdr::HlgRendering;

pub(crate) const BT2020_LUMA: [f64; 3] = [0.2627, 0.6780, 0.0593];

fn decode(v: f64) -> f64 {
    let a = 0.17883277;
    let b = 1.0 - 4.0 * a;
    let c = 0.5 - a * libm::log(4.0 * a);
    if v.abs() <= 0.5 {
        v * v.abs() / 3.0
    } else {
        v.signum() * (libm::exp((v.abs() - c) / a) + b) / 12.0
    }
}
fn encode(v: f64) -> f64 {
    let a = 0.17883277;
    let b = 1.0 - 4.0 * a;
    let c = 0.5 - a * libm::log(4.0 * a);
    if v.abs() <= 1.0 / 12.0 {
        v.signum() * libm::sqrt(3.0 * v.abs())
    } else {
        v.signum() * (a * libm::log(12.0 * v.abs() - b) + c)
    }
}

/// Compute CIE 1931 `Y` luminance weights `[yr, yg, yb]` for `(primaries, white_point)`.
#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
pub(crate) fn luma_weights(
    primaries: zenpixels::ParametricPrimaries,
    white_point: zenpixels::WhitePoint,
) -> Option<[f64; 3]> {
    if primaries == zenpixels::ParametricPrimaries::Bt2020
        && white_point == zenpixels::WhitePoint::D65
    {
        return Some(BT2020_LUMA);
    }
    let (r, g, b) = primaries.cie_xy();
    let wp = white_point.cie_xy();
    let (rx, ry) = (f64::from(r.x()), f64::from(r.y()));
    let (gx, gy) = (f64::from(g.x()), f64::from(g.y()));
    let (bx, by) = (f64::from(b.x()), f64::from(b.y()));
    let (wx, wy) = (f64::from(wp.x()), f64::from(wp.y()));
    if ry == 0.0 || gy == 0.0 || by == 0.0 || wy <= 0.0 {
        return None;
    }
    let (xr, zr) = (rx / ry, (1.0 - rx - ry) / ry);
    let (xg, zg) = (gx / gy, (1.0 - gx - gy) / gy);
    let (xb, zb) = (bx / by, (1.0 - bx - by) / by);
    let (xw, zw) = (wx / wy, (1.0 - wx - wy) / wy);
    // Solve [[xr, xg, xb], [1, 1, 1], [zr, zg, zb]] * s = [xw, 1, zw]
    let a = [[xr, xg, xb], [1.0, 1.0, 1.0], [zr, zg, zb]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if !det.is_finite() || det.abs() < 1e-10 {
        return None;
    }
    let inv = 1.0 / det;
    let s0 = (xw * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[2][2] - a[1][2] * zw)
        + a[0][2] * (a[2][1] - a[1][1] * zw))
        * inv;
    let s1 = (a[0][0] * (a[2][2] - a[1][2] * zw) - xw * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * zw - a[2][0]))
        * inv;
    let s2 = (a[0][0] * (a[1][1] * zw - a[2][1]) - a[0][1] * (a[1][0] * zw - a[2][0])
        + xw * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv;
    if s0.is_finite() && s1.is_finite() && s2.is_finite() {
        Some([s0, s1, s2])
    } else {
        None
    }
}

// Extended RGB uses signed transfer curves and |Y| in the common luminance
// multiplier. Zero luminance maps to zero; nonfinite RGB maps to RGB NaN.
// Alpha is copied bit-for-bit. No per-row reduction or allocation occurs.
#[derive(Clone, Debug)]
pub(crate) struct PreparedHlg {
    gamma: f64,
    beta: f64,
    scale: f64,
    luma: [f64; 3],
    pub(super) inverse: bool,
    clip: bool,
}
impl PreparedHlg {
    pub(super) fn new(rendering: HlgRendering, inverse: bool, white: f32, clip: bool) -> Self {
        Self::new_with_luma(rendering, inverse, white, clip, BT2020_LUMA)
    }

    pub(crate) fn new_with_luma(
        rendering: HlgRendering,
        inverse: bool,
        white: f32,
        clip: bool,
        luma: [f64; 3],
    ) -> Self {
        let gamma = f64::from(rendering.system_gamma());
        let peak = f64::from(rendering.peak().nits());
        let beta =
            libm::sqrt(3.0 * libm::pow(f64::from(rendering.black_nits()) / peak, 1.0 / gamma));
        Self {
            gamma,
            beta,
            scale: peak / f64::from(white),
            luma,
            inverse,
            clip,
        }
    }
}

pub(super) fn row(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    prepared: &PreparedHlg,
) {
    let channels = if layout == ChannelLayout::Rgba { 4 } else { 3 };
    let PreparedHlg {
        gamma,
        beta,
        scale,
        luma,
        inverse,
        clip,
    } = *prepared;
    for (s, d) in src[..width * channels * 4]
        .chunks_exact(channels * 4)
        .zip(dst[..width * channels * 4].chunks_exact_mut(channels * 4))
    {
        d.copy_from_slice(s);
        let mut rgb = [0.0; 3];
        for (i, v) in rgb.iter_mut().enumerate() {
            *v = f64::from(f32::from_ne_bytes(s[i * 4..i * 4 + 4].try_into().unwrap()));
        }
        if rgb.iter().any(|v| !v.is_finite()) {
            rgb = [f64::NAN; 3];
        } else if inverse {
            for v in &mut rgb {
                *v /= scale;
            }
            let y = luma[0] * rgb[0] + luma[1] * rgb[1] + luma[2] * rgb[2];
            let gain = if y == 0.0 {
                0.0
            } else {
                libm::pow(y.abs(), (1.0 - gamma) / gamma)
            };
            for v in &mut rgb {
                *v = (encode(*v * gain) - beta) / (1.0 - beta);
                if clip {
                    *v = v.clamp(0.0, 1.0);
                }
            }
        } else {
            for v in &mut rgb {
                if clip {
                    // ITU-R BT.2100-3 Table 5 Note 5h: clamp super-white at 1.0,
                    // apply black-level lift first so PLUGE sub-blacks in
                    // [-beta / (1 - beta), 0) decode smoothly to [0, L_B], and
                    // clamp below display black at 0.0.
                    let v_clamped = v.min(1.0);
                    *v = decode(((1.0 - beta) * v_clamped + beta).max(0.0));
                } else {
                    *v = decode((1.0 - beta) * *v + beta);
                }
            }
            let y = luma[0] * rgb[0] + luma[1] * rgb[1] + luma[2] * rgb[2];
            let gain = if y == 0.0 {
                0.0
            } else {
                scale * libm::pow(y.abs(), gamma - 1.0)
            };
            for v in &mut rgb {
                *v *= gain;
            }
        }
        for (i, v) in rgb.iter().enumerate() {
            d[i * 4..i * 4 + 4].copy_from_slice(&(*v as f32).to_ne_bytes());
        }
    }
}

#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
pub(crate) fn is_hlg_profile(profile: &ColorProfileSource<'_>) -> bool {
    match profile {
        ColorProfileSource::Parametric(p) => p.transfer == zenpixels::ParametricTransfer::Hlg,
        ColorProfileSource::Cicp(c) => c.transfer_characteristics == 18,
        ColorProfileSource::Named(zenpixels::NamedProfile::Bt2020Hlg) => true,
        ColorProfileSource::PrimariesTransferPair { transfer, .. } => {
            *transfer == crate::TransferFunction::Hlg
        }
        ColorProfileSource::Icc(_) => profile
            .resolve()
            .is_some_and(|(_, tf)| tf == crate::TransferFunction::Hlg),
        _ => false,
    }
}

#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
fn is_pq_profile(profile: &ColorProfileSource<'_>) -> bool {
    match profile {
        ColorProfileSource::Parametric(p) => p.transfer == zenpixels::ParametricTransfer::Pq,
        ColorProfileSource::Cicp(c) => c.transfer_characteristics == 16,
        ColorProfileSource::Named(zenpixels::NamedProfile::Bt2020Pq) => true,
        ColorProfileSource::PrimariesTransferPair { transfer, .. } => {
            *transfer == crate::TransferFunction::Pq
        }
        ColorProfileSource::Icc(_) => profile
            .resolve()
            .is_some_and(|(_, tf)| tf == crate::TransferFunction::Pq),
        _ => false,
    }
}

#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
fn is_linear_profile(profile: &ColorProfileSource<'_>) -> bool {
    match profile {
        ColorProfileSource::Parametric(p) => p.transfer == zenpixels::ParametricTransfer::Linear,
        ColorProfileSource::Cicp(c) => c.transfer_characteristics == 8,
        ColorProfileSource::Named(zenpixels::NamedProfile::LinearSrgb) => true,
        ColorProfileSource::PrimariesTransferPair { transfer, .. } => {
            *transfer == crate::TransferFunction::Linear
        }
        ColorProfileSource::Icc(_) => profile
            .resolve()
            .is_some_and(|(_, tf)| tf == crate::TransferFunction::Linear),
        _ => false,
    }
}

#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
fn hlg_linear_counterpart(
    profile: &ColorProfileSource<'_>,
) -> Result<(zenpixels::ParametricColor, [f64; 3]), crate::cms::CmsPluginError> {
    use crate::cms::CmsPluginError;
    use zenpixels::{
        ColorPrimaries, ParametricColor, ParametricPrimaries, ParametricTransfer, WhitePoint,
    };
    let param = match profile {
        ColorProfileSource::Parametric(p) if p.transfer == ParametricTransfer::Hlg => *p,
        ColorProfileSource::Cicp(c) if c.transfer_characteristics == 18 => {
            ParametricColor::from_cicp(*c).ok_or_else(|| {
                CmsPluginError::msg("unsupported CICP parameters for HLG transform")
            })?
        }
        ColorProfileSource::Named(zenpixels::NamedProfile::Bt2020Hlg) => ParametricColor::rgb(
            ParametricPrimaries::Bt2020,
            WhitePoint::D65,
            ParametricTransfer::Hlg,
        ),
        ColorProfileSource::PrimariesTransferPair {
            primaries,
            transfer,
        } if *transfer == crate::TransferFunction::Hlg => {
            let pp = match *primaries {
                ColorPrimaries::Bt709 => ParametricPrimaries::Bt709,
                ColorPrimaries::Bt2020 => ParametricPrimaries::Bt2020,
                ColorPrimaries::DisplayP3 => ParametricPrimaries::P3,
                ColorPrimaries::AdobeRgb => ParametricPrimaries::AdobeRgb,
                _ => return Err(CmsPluginError::msg("unknown primaries for HLG transform")),
            };
            ParametricColor::rgb(pp, WhitePoint::D65, ParametricTransfer::Hlg)
        }
        ColorProfileSource::Icc(_) => {
            let Some((primaries, tf)) = profile.resolve() else {
                return Err(CmsPluginError::msg("unresolved ICC profile for HLG"));
            };
            if tf != crate::TransferFunction::Hlg {
                return Err(CmsPluginError::msg("non-HLG ICC profile"));
            }
            return hlg_linear_counterpart(&ColorProfileSource::PrimariesTransferPair {
                primaries,
                transfer: tf,
            });
        }
        _ => return Err(CmsPluginError::msg("not an HLG profile")),
    };
    let Some(prim) = param.primaries else {
        return Err(CmsPluginError::msg(
            "HLG requires 3-channel RGB primaries for coupled luminance OOTF",
        ));
    };
    let luma = luma_weights(prim, param.white_point)
        .ok_or_else(|| CmsPluginError::msg("singular or invalid HLG RGB primaries"))?;
    let linear = ParametricColor::rgb(prim, param.white_point, ParametricTransfer::Linear)
        .with_rendering_intent(param.rendering_intent);
    Ok((linear, luma))
}

#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
fn float_rgb_format(fmt: PixelFormat) -> Option<(PixelFormat, ChannelLayout, usize)> {
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::RgbF32 => {
            Some((PixelFormat::RgbF32, ChannelLayout::Rgb, 3))
        }
        PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaF32 => {
            Some((PixelFormat::RgbaF32, ChannelLayout::Rgba, 4))
        }
        _ => None,
    }
}

#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
struct HlgCmsWorker {
    src_format: PixelFormat,
    dst_format: PixelFormat,
    src_f32_channels: usize,
    dst_f32_channels: usize,
    src_hlg: Option<(ChannelLayout, PreparedHlg)>,
    tone_map_before_cms: Option<crate::RowConverter>,
    cms_stage: Option<Box<dyn crate::cms::RowTransformMut>>,
    dst_hlg: Option<(ChannelLayout, PreparedHlg)>,
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
    prepared_width: Option<u32>,
}

#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
impl HlgCmsWorker {
    fn ensure_scratch(&mut self, width: u32) -> Result<(), At<crate::cms::CmsPluginError>> {
        if let Some(max_w) = self.prepared_width
            && width > max_w
        {
            return Err(whereat::at!(crate::cms::CmsPluginError::msg(
                "row width exceeds prepared HLG CMS capacity"
            )));
        }
        let max_ch = self.src_f32_channels.max(self.dst_f32_channels).max(4);
        let needed = (width as usize)
            .checked_mul(max_ch)
            .ok_or_else(|| whereat::at!(crate::cms::CmsPluginError::msg("row width overflow")))?;
        for buf in [&mut self.scratch_a, &mut self.scratch_b] {
            if buf.len() < needed {
                buf.try_reserve(needed - buf.len()).map_err(|_| {
                    whereat::at!(crate::cms::CmsPluginError::msg(
                        "HLG CMS scratch allocation failed"
                    ))
                })?;
                buf.resize(needed, 0.0);
            }
        }
        Ok(())
    }
}

#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
impl crate::cms::RowTransformMut for HlgCmsWorker {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) {
        self.try_transform_row(src, dst, width)
            .expect("HLG CMS row transform failed");
    }

    fn prepare(&mut self, max_width: u32) -> Result<(), At<crate::cms::CmsPluginError>> {
        self.prepared_width = None;
        self.ensure_scratch(max_width)?;
        if let Some(tm) = &mut self.tone_map_before_cms {
            tm.prepare(max_width)
                .map_err(|e| whereat::at!(crate::cms::CmsPluginError::new(e)))?;
        }
        if let Some(cms) = &mut self.cms_stage {
            cms.prepare(max_width)?;
        }
        self.prepared_width = Some(max_width);
        Ok(())
    }

    fn try_transform_row(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
    ) -> Result<(), At<crate::cms::CmsPluginError>> {
        let src_bytes = (width as usize)
            .checked_mul(self.src_format.bytes_per_pixel())
            .ok_or_else(|| whereat::at!(crate::cms::CmsPluginError::msg("source row overflow")))?;
        let dst_bytes = (width as usize)
            .checked_mul(self.dst_format.bytes_per_pixel())
            .ok_or_else(|| {
                whereat::at!(crate::cms::CmsPluginError::msg("destination row overflow"))
            })?;
        if src.len() < src_bytes || dst.len() < dst_bytes {
            return Err(whereat::at!(crate::cms::CmsPluginError::msg(
                "row slice shorter than width * bytes_per_pixel"
            )));
        }
        if width == 0 {
            return Ok(());
        }
        self.ensure_scratch(width)?;
        let w = width as usize;
        let in_len = w * self.src_f32_channels;

        // 1. Widen source samples to F32 in scratch_a[..in_len].
        match self.src_format.channel_type() {
            ChannelType::U8 => {
                for (o, &b) in self.scratch_a[..in_len].iter_mut().zip(&src[..src_bytes]) {
                    *o = f32::from(b) * (1.0 / 255.0);
                }
            }
            ChannelType::U16 => {
                for (o, &chunk) in self.scratch_a[..in_len]
                    .iter_mut()
                    .zip(src[..src_bytes].as_chunks::<2>().0)
                {
                    let v = u16::from_ne_bytes(chunk);
                    *o = f32::from(v) * (1.0 / 65535.0);
                }
            }
            ChannelType::F32 => {
                bytemuck::cast_slice_mut::<f32, u8>(&mut self.scratch_a[..in_len])
                    .copy_from_slice(&src[..src_bytes]);
            }
            _ => unreachable!("checked plannable channel type"),
        }

        // 2. Forward HLG OOTF on scratch_a -> scratch_b, then swap into scratch_a.
        if let Some((layout, ref prep)) = self.src_hlg {
            row(
                bytemuck::cast_slice(&self.scratch_a[..in_len]),
                bytemuck::cast_slice_mut(&mut self.scratch_b[..in_len]),
                w,
                layout,
                prep,
            );
            self.scratch_a[..in_len].copy_from_slice(&self.scratch_b[..in_len]);
        }

        // 3. Optional HDR->SDR BT.2446-A tone-map stage (in F32).
        if let Some(tm) = &mut self.tone_map_before_cms {
            tm.try_convert_row(
                bytemuck::cast_slice(&self.scratch_a[..in_len]),
                bytemuck::cast_slice_mut(&mut self.scratch_b[..in_len]),
                width,
            )
            .map_err(|e| whereat::at!(crate::cms::CmsPluginError::new(e)))?;
            self.scratch_a[..in_len].copy_from_slice(&self.scratch_b[..in_len]);
        }

        // 4. Linear / SDR CMS stage: scratch_a[..in_len] -> scratch_b[..out_len].
        let out_len = w * self.dst_f32_channels;
        if let Some(cms) = &mut self.cms_stage {
            cms.try_transform_row(
                bytemuck::cast_slice(&self.scratch_a[..in_len]),
                bytemuck::cast_slice_mut(&mut self.scratch_b[..out_len]),
                width,
            )?;
        } else if self.src_f32_channels == self.dst_f32_channels {
            self.scratch_b[..out_len].copy_from_slice(&self.scratch_a[..in_len]);
        } else if self.src_f32_channels == 4 && self.dst_f32_channels == 3 {
            for (s, d) in self.scratch_a[..in_len]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(self.scratch_b[..out_len].as_chunks_mut::<3>().0)
            {
                d.copy_from_slice(&s[..3]);
            }
        } else if self.src_f32_channels == 3 && self.dst_f32_channels == 4 {
            for (s, d) in self.scratch_a[..in_len]
                .as_chunks::<3>()
                .0
                .iter()
                .zip(self.scratch_b[..out_len].as_chunks_mut::<4>().0)
            {
                d[..3].copy_from_slice(s);
                d[3] = 1.0;
            }
        }

        // 5. Inverse HLG OOTF on scratch_b -> scratch_a, else keep in scratch_a.
        if let Some((layout, ref prep)) = self.dst_hlg {
            row(
                bytemuck::cast_slice(&self.scratch_b[..out_len]),
                bytemuck::cast_slice_mut(&mut self.scratch_a[..out_len]),
                w,
                layout,
                prep,
            );
        } else {
            self.scratch_a[..out_len].copy_from_slice(&self.scratch_b[..out_len]);
        }

        // 6. Narrow F32 in scratch_a[..out_len] to dst[..dst_bytes].
        match self.dst_format.channel_type() {
            ChannelType::U8 => {
                for (d, &v) in dst[..dst_bytes].iter_mut().zip(&self.scratch_a[..out_len]) {
                    *d = (f64::from(v.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
                }
            }
            ChannelType::U16 => {
                for (chunk, &v) in dst[..dst_bytes]
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(&self.scratch_a[..out_len])
                {
                    let q = (f64::from(v.clamp(0.0, 1.0)) * 65535.0 + 0.5) as u16;
                    *chunk = q.to_ne_bytes();
                }
            }
            ChannelType::F32 => {
                dst[..dst_bytes].copy_from_slice(bytemuck::cast_slice(&self.scratch_a[..out_len]));
            }
            _ => unreachable!("checked plannable channel type"),
        }
        Ok(())
    }
}

/// Build an HLG-coupled display-OOTF wrapper around a linear/SDR CMS backend (`MoxCms` or `Lcms`)
/// when `options` explicitly supplies [`HlgRendering`] and `sdr_white` (plus `hdr_peak`/`sdr_peak`
/// for HDR→SDR tone mapping). Declines (`None`) when rendering conditions are absent so backends
/// never silently invent display rendering.
#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_cms_hlg_transform(
    cms: &dyn crate::cms::PluggableCms,
    src: ColorProfileSource<'_>,
    dst: ColorProfileSource<'_>,
    src_format: PixelFormat,
    dst_format: PixelFormat,
    options: &crate::policy::ConvertOptions,
    allow_absolute_intent: bool,
) -> Option<Result<Box<dyn crate::cms::RowTransformMut>, At<crate::cms::CmsPluginError>>> {
    let src_is_hlg = is_hlg_profile(&src);
    let dst_is_hlg = is_hlg_profile(&dst);
    if !src_is_hlg && !dst_is_hlg {
        return None;
    }
    if crate::output::profiles_match(&src, &dst) {
        return None;
    }
    let explicit_intent = options.rendering_intent().or(match &src {
        ColorProfileSource::Parametric(p) => Some(p.rendering_intent),
        _ => None,
    });
    if !allow_absolute_intent
        && explicit_intent == Some(zenpixels::RenderingIntent::AbsoluteColorimetric)
    {
        return None;
    }
    let rendering = options.hlg_rendering()?;
    let white = options.sdr_white()?;

    let plannable = |t: ChannelType| {
        matches!(
            t,
            ChannelType::U8 | ChannelType::U16 | ChannelType::F16 | ChannelType::F32
        )
    };
    let (st, dt) = (src_format.channel_type(), dst_format.channel_type());
    if plannable(st) && plannable(dt) && (st != dt || st == ChannelType::F16) {
        return None;
    }
    if st != dt || !plannable(st) {
        return Some(Err(whereat::at!(crate::cms::CmsPluginError::msg(
            "CMS requires matching U8, U16 or F32 sample types"
        ))));
    }

    let (src_f32_fmt, src_hlg_layout, src_f32_ch) = if src_is_hlg {
        float_rgb_format(src_format)?
    } else {
        match src_format {
            PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::RgbF32 => {
                (PixelFormat::RgbF32, ChannelLayout::Rgb, 3)
            }
            PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaF32 => {
                (PixelFormat::RgbaF32, ChannelLayout::Rgba, 4)
            }
            PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayF32 => {
                (PixelFormat::GrayF32, ChannelLayout::Gray, 1)
            }
            PixelFormat::GrayA8 | PixelFormat::GrayA16 | PixelFormat::GrayAF32 => {
                (PixelFormat::GrayAF32, ChannelLayout::GrayAlpha, 2)
            }
            PixelFormat::CMYK8
            | PixelFormat::CMYK8_INVERTED
            | PixelFormat::CMYKF32
            | PixelFormat::CMYKF32_INVERTED => {
                // Delegate CMYK->Linear before HLG only via RowConverter's float pivot.
                return None;
            }
            _ => return None,
        }
    };
    let (dst_f32_fmt, dst_hlg_layout, dst_f32_ch) = if dst_is_hlg {
        float_rgb_format(dst_format)?
    } else {
        match dst_format {
            PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::RgbF32 => {
                (PixelFormat::RgbF32, ChannelLayout::Rgb, 3)
            }
            PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaF32 => {
                (PixelFormat::RgbaF32, ChannelLayout::Rgba, 4)
            }
            PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayF32 => {
                (PixelFormat::GrayF32, ChannelLayout::Gray, 1)
            }
            PixelFormat::GrayA8 | PixelFormat::GrayA16 | PixelFormat::GrayAF32 => {
                (PixelFormat::GrayAF32, ChannelLayout::GrayAlpha, 2)
            }
            _ => return None,
        }
    };

    // Check if HLG -> SDR tone mapping is required.
    let dst_is_sdr = src_is_hlg && !dst_is_hlg && !is_linear_profile(&dst) && !is_pq_profile(&dst);
    if dst_is_sdr {
        let zenpixels::hdr::HdrPeak::Nits(_) = options.hdr_peak() else {
            return None;
        };
        options.sdr_peak()?;
        #[cfg(not(feature = "bt2446a"))]
        {
            return None;
        }
    }

    let (src_hlg, cms_src, tone_map_before_cms) = if src_is_hlg {
        let (src_lin_param, luma) = match hlg_linear_counterpart(&src) {
            Ok(v) => v,
            Err(e) => return Some(Err(whereat::at!(e))),
        };
        if dst_is_sdr {
            #[cfg(feature = "bt2446a")]
            {
                // Forward HLG OOTF into PQ-anchored linear (white = 10_000 nits) if BT.2020,
                // or via linear BT.2020 first, then tone-map PQ BT.2020 -> SDR BT.2020 (Cicp 9,1,0,true).
                let prep = PreparedHlg::new_with_luma(rendering, false, white.nits(), false, luma);
                let bt2020_lin_cicp = zenpixels::Cicp::new(9, 8, 0, true);
                let bt2020_sdr_cicp = zenpixels::Cicp::new(9, 1, 0, true);
                let lin_desc = src_f32_fmt
                    .descriptor()
                    .with_primaries(crate::ColorPrimaries::Bt2020)
                    .with_transfer(crate::TransferFunction::Linear);
                let pq_desc = lin_desc.with_transfer(crate::TransferFunction::Pq);
                let sdr_desc = lin_desc.with_transfer(crate::TransferFunction::Bt709);
                let to_pq = match crate::ConvertPlan::new_explicit(
                    lin_desc,
                    pq_desc,
                    &options.with_clip_out_of_gamut(false),
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        return Some(Err(e.map_error(crate::cms::CmsPluginError::new)));
                    }
                };
                let to_sdr = match crate::ConvertPlan::new_explicit(
                    pq_desc,
                    sdr_desc,
                    &options.with_clip_out_of_gamut(false),
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        return Some(Err(e.map_error(crate::cms::CmsPluginError::new)));
                    }
                };
                let Some(composed) = to_pq.compose_preserving(&to_sdr) else {
                    return Some(Err(whereat::at!(crate::cms::CmsPluginError::msg(
                        "failed to compose HLG tone-map stage"
                    ))));
                };
                // If custom primaries != BT.2020, adapt custom linear -> BT.2020 linear first!
                if src_lin_param.to_cicp() != Some(bt2020_lin_cicp) {
                    // Build a two-step CMS: src_lin_param -> bt2020_lin, then tone-map, then bt2020_sdr -> dst.
                    let to_bt2020 = cms.build_source_transform(
                        ColorProfileSource::Parametric(src_lin_param),
                        ColorProfileSource::Cicp(bt2020_lin_cicp),
                        src_f32_fmt,
                        src_f32_fmt,
                        &options.with_clip_out_of_gamut(false),
                    )?;
                    let mut pre_cms = match to_bt2020 {
                        Ok(t) => t,
                        Err(e) => return Some(Err(e)),
                    };
                    let post_cms_opt = if crate::output::profiles_match(
                        &ColorProfileSource::Cicp(bt2020_sdr_cicp),
                        &dst,
                    ) {
                        None
                    } else {
                        match cms.build_source_transform(
                            ColorProfileSource::Cicp(bt2020_sdr_cicp),
                            dst,
                            src_f32_fmt,
                            dst_f32_fmt,
                            options,
                        )? {
                            Ok(t) => Some(t),
                            Err(e) => return Some(Err(e)),
                        }
                    };
                    let mut tm_conv = crate::RowConverter::from_plan(composed);
                    struct CustomHlgToSdr {
                        pre: Box<dyn crate::cms::RowTransformMut>,
                        tm: crate::RowConverter,
                        post: Option<Box<dyn crate::cms::RowTransformMut>>,
                        mid_a: Vec<f32>,
                        mid_b: Vec<f32>,
                        ch: usize,
                    }
                    impl crate::cms::RowTransformMut for CustomHlgToSdr {
                        fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) {
                            self.try_transform_row(src, dst, width).unwrap();
                        }
                        fn prepare(
                            &mut self,
                            max_width: u32,
                        ) -> Result<(), At<crate::cms::CmsPluginError>> {
                            let n = (max_width as usize) * self.ch;
                            self.mid_a.resize(n, 0.0);
                            self.mid_b.resize(n, 0.0);
                            self.pre.prepare(max_width)?;
                            self.tm
                                .prepare(max_width)
                                .map_err(|e| whereat::at!(crate::cms::CmsPluginError::new(e)))?;
                            if let Some(p) = &mut self.post {
                                p.prepare(max_width)?;
                            }
                            Ok(())
                        }
                        fn try_transform_row(
                            &mut self,
                            src: &[u8],
                            dst: &mut [u8],
                            width: u32,
                        ) -> Result<(), At<crate::cms::CmsPluginError>> {
                            let n = (width as usize) * self.ch;
                            if self.mid_a.len() < n {
                                self.mid_a.resize(n, 0.0);
                                self.mid_b.resize(n, 0.0);
                            }
                            self.pre.try_transform_row(
                                src,
                                bytemuck::cast_slice_mut(&mut self.mid_a[..n]),
                                width,
                            )?;
                            if let Some(post) = &mut self.post {
                                self.tm
                                    .try_convert_row(
                                        bytemuck::cast_slice(&self.mid_a[..n]),
                                        bytemuck::cast_slice_mut(&mut self.mid_b[..n]),
                                        width,
                                    )
                                    .map_err(|e| {
                                        whereat::at!(crate::cms::CmsPluginError::new(e))
                                    })?;
                                post.try_transform_row(
                                    bytemuck::cast_slice(&self.mid_b[..n]),
                                    dst,
                                    width,
                                )
                            } else {
                                self.tm
                                    .try_convert_row(
                                        bytemuck::cast_slice(&self.mid_a[..n]),
                                        dst,
                                        width,
                                    )
                                    .map_err(|e| whereat::at!(crate::cms::CmsPluginError::new(e)))
                            }
                        }
                    }
                    let _ = (&mut pre_cms, &mut tm_conv);
                    return Some(Ok(Box::new(HlgCmsWorker {
                        src_format,
                        dst_format,
                        src_f32_channels: src_f32_ch,
                        dst_f32_channels: dst_f32_ch,
                        src_hlg: Some((src_hlg_layout, prep)),
                        tone_map_before_cms: None,
                        cms_stage: Some(Box::new(CustomHlgToSdr {
                            pre: pre_cms,
                            tm: tm_conv,
                            post: post_cms_opt,
                            mid_a: Vec::new(),
                            mid_b: Vec::new(),
                            ch: src_f32_ch,
                        })),
                        dst_hlg: None,
                        scratch_a: Vec::new(),
                        scratch_b: Vec::new(),
                        prepared_width: None,
                    })));
                }
                (
                    Some((src_hlg_layout, prep)),
                    ColorProfileSource::Cicp(bt2020_sdr_cicp),
                    Some(crate::RowConverter::from_plan(composed)),
                )
            }
            #[cfg(not(feature = "bt2446a"))]
            {
                return None;
            }
        } else {
            let eff_white = if !dst_is_hlg && is_pq_profile(&dst) {
                10_000.0
            } else {
                white.nits()
            };
            let prep = PreparedHlg::new_with_luma(rendering, false, eff_white, false, luma);
            (
                Some((src_hlg_layout, prep)),
                ColorProfileSource::Parametric(src_lin_param),
                None,
            )
        }
    } else {
        (None, src, None)
    };

    let (dst_hlg, cms_dst) = if dst_is_hlg {
        let (dst_lin_param, luma) = match hlg_linear_counterpart(&dst) {
            Ok(v) => v,
            Err(e) => return Some(Err(whereat::at!(e))),
        };
        let eff_white = if !src_is_hlg && is_pq_profile(&cms_src) {
            10_000.0
        } else {
            white.nits()
        };
        let clip = options.clip_out_of_gamut() || dst_format.channel_type() != ChannelType::F32;
        let prep = PreparedHlg::new_with_luma(rendering, true, eff_white, clip, luma);
        (
            Some((dst_hlg_layout, prep)),
            ColorProfileSource::Parametric(dst_lin_param),
        )
    } else {
        (None, dst)
    };

    let cms_options =
        if dst_is_hlg || (!dst_is_sdr && dst_format.channel_type() == ChannelType::F32) {
            options.with_clip_out_of_gamut(options.clip_out_of_gamut() && !dst_is_hlg)
        } else {
            *options
        };

    let cms_stage = if crate::output::profiles_match(&cms_src, &cms_dst) {
        None
    } else {
        match cms.build_source_transform(
            cms_src,
            cms_dst,
            src_f32_fmt,
            dst_f32_fmt,
            &cms_options,
        )? {
            Ok(stage) => Some(stage),
            Err(e) => return Some(Err(e)),
        }
    };

    Some(Ok(Box::new(HlgCmsWorker {
        src_format,
        dst_format,
        src_f32_channels: src_f32_ch,
        dst_f32_channels: dst_f32_ch,
        src_hlg,
        tone_map_before_cms,
        cms_stage,
        dst_hlg,
        scratch_a: Vec::new(),
        scratch_b: Vec::new(),
        prepared_width: None,
    })))
}
