//! Clean Color Management System (`CmsBackend`) trait + optional `MoxCms` / `Lcms` adapters.
//!
//! Only invoked when converting arbitrary [`ColorSpace::Icc`](zenpixels::ColorSpace::Icc)
//! profiles (that don't match a safe matrix+TRC fingerprint) or [`ColorSpace::Cmyk`](zenpixels::ColorSpace::Cmyk).

use alloc::sync::Arc;
use core::fmt::Debug;
use whereat::At;
use zenpixels::ImageSpec;

use crate::error::ConvertError;

/// Shareable, stateless row-level CMS transform executor (`Send + Sync`).
pub trait CmsTransform: Send + Sync {
    /// Transform one row of `width` pixels from `src` bytes to `dst` bytes.
    fn transform_row(&self, src: &[u8], dst: &mut [u8], width: u32)
    -> Result<(), At<ConvertError>>;
}

/// Pluggable ICC / CMYK Color Management System backend.
pub trait CmsBackend: Send + Sync + Debug {
    /// Build a row transform from `src` to `dst`.
    fn build_transform(
        &self,
        src: &ImageSpec,
        dst: &ImageSpec,
    ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>>;
}

#[cfg(feature = "cms-moxcms")]
mod moxcms_impl {
    use super::*;
    use crate::error::err_at;
    use zenpixels::{
        AlphaMode, ChannelOrder, ColorSpace, Primaries, RenderingIntent, SampleType, SignalRange,
        Transfer,
    };

    const CHUNK_PX: usize = 64;

    /// Pure-Rust `moxcms` ICC CMS backend.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct MoxCms;

    fn to_mox_intent(intent: RenderingIntent) -> moxcms::RenderingIntent {
        match intent {
            RenderingIntent::Perceptual => moxcms::RenderingIntent::Perceptual,
            RenderingIntent::RelativeColorimetric => moxcms::RenderingIntent::RelativeColorimetric,
            RenderingIntent::Saturation => moxcms::RenderingIntent::Saturation,
            RenderingIntent::AbsoluteColorimetric => moxcms::RenderingIntent::AbsoluteColorimetric,
            _ => moxcms::RenderingIntent::Perceptual,
        }
    }

    fn to_mox_layout(order: ChannelOrder) -> Option<moxcms::Layout> {
        match order {
            ChannelOrder::Gray => Some(moxcms::Layout::Gray),
            ChannelOrder::GrayAlpha => Some(moxcms::Layout::GrayAlpha),
            ChannelOrder::Rgb => Some(moxcms::Layout::Rgb),
            ChannelOrder::Rgba => Some(moxcms::Layout::Rgba),
            ChannelOrder::Cmyk | ChannelOrder::CmykInverted => Some(moxcms::Layout::Rgba),
            _ => None,
        }
    }

    fn profile_from_color_space(
        cs: &ColorSpace,
    ) -> Result<(moxcms::ColorProfile, Option<RenderingIntent>), At<ConvertError>> {
        match cs {
            ColorSpace::Icc {
                profile, intent, ..
            } => {
                let p = moxcms::ColorProfile::new_from_slice(profile)
                    .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                Ok((p, Some(*intent)))
            }
            ColorSpace::Cmyk { icc, .. } => {
                let bytes = icc
                    .as_ref()
                    .ok_or_else(|| err_at(ConvertError::CmsRequired))?;
                let p = moxcms::ColorProfile::new_from_slice(bytes)
                    .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                Ok((p, Some(RenderingIntent::RelativeColorimetric)))
            }
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                ..
            } => {
                if *range != SignalRange::Full {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                let p = match (primaries, transfer) {
                    (Primaries::Bt709, Transfer::Srgb) => moxcms::ColorProfile::new_srgb(),
                    (Primaries::DisplayP3, Transfer::Srgb) => {
                        moxcms::ColorProfile::new_display_p3()
                    }
                    (Primaries::AdobeRgb, Transfer::Gamma22) => {
                        moxcms::ColorProfile::new_adobe_rgb()
                    }
                    (Primaries::Bt2020, Transfer::Bt709) => moxcms::ColorProfile::new_bt2020(),
                    (Primaries::Bt2020, Transfer::Pq) => moxcms::ColorProfile::new_bt2020_pq(),
                    // HLG requires BT.2100-3 display OOTF which ICC profiles cannot carry faithfully.
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                Ok((p, None))
            }
            ColorSpace::Gray {
                transfer, range, ..
            } => {
                if *range != SignalRange::Full {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                let p = match transfer {
                    Transfer::Srgb => {
                        let mut p = moxcms::ColorProfile::new_gray_with_gamma(1.0);
                        p.gray_trc = moxcms::ColorProfile::new_srgb().red_trc;
                        p
                    }
                    Transfer::Linear => moxcms::ColorProfile::new_gray_with_gamma(1.0),
                    Transfer::Gamma22 => moxcms::ColorProfile::new_gray_with_gamma(563.0 / 256.0),
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                Ok((p, None))
            }
            _ => Err(err_at(ConvertError::UnsupportedRoute)),
        }
    }

    enum MoxExec {
        U8(Arc<dyn moxcms::TransformExecutor<u8> + Send + Sync>),
        U16(Arc<dyn moxcms::TransformExecutor<u16> + Send + Sync>),
        F32(Arc<dyn moxcms::TransformExecutor<f32> + Send + Sync>),
    }

    struct MoxRowTransform {
        exec: MoxExec,
        src_bpp: usize,
        dst_bpp: usize,
        src_ch: usize,
        dst_ch: usize,
        dst_sample: SampleType,
        src_cmyk_inverted: bool,
        dst_cmyk_inverted: bool,
        fill_opaque_alpha: bool,
    }

    impl CmsTransform for MoxRowTransform {
        fn transform_row(
            &self,
            src: &[u8],
            dst: &mut [u8],
            width: u32,
        ) -> Result<(), At<ConvertError>> {
            let w = width as usize;
            let s_len = w
                .checked_mul(self.src_bpp)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            let d_len = w
                .checked_mul(self.dst_bpp)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            if src.len() < s_len || dst.len() < d_len {
                return Err(err_at(ConvertError::DimensionMismatch));
            }
            if w == 0 {
                return Ok(());
            }
            let s = &src[..s_len];
            let d = &mut dst[..d_len];
            match &self.exec {
                MoxExec::U8(ex) => {
                    if !self.src_cmyk_inverted {
                        ex.transform(s, d)
                            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    } else {
                        let mut inv = [0u8; CHUNK_PX * 4];
                        let mut pos = 0;
                        while pos < w {
                            let n = (w - pos).min(CHUNK_PX);
                            let sb = &s[pos * self.src_bpp..(pos + n) * self.src_bpp];
                            let db = &mut d[pos * self.dst_bpp..(pos + n) * self.dst_bpp];
                            let tmp = &mut inv[..sb.len()];
                            for (o, &b) in tmp.iter_mut().zip(sb) {
                                *o = !b;
                            }
                            ex.transform(tmp, db)
                                .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                            pos += n;
                        }
                    }
                    if self.dst_cmyk_inverted {
                        for b in d.iter_mut() {
                            *b = !*b;
                        }
                    }
                }
                MoxExec::U16(ex) => {
                    if !self.src_cmyk_inverted
                        && !self.dst_cmyk_inverted
                        && let (Ok(s16), Ok(d16)) = (
                            bytemuck::try_cast_slice::<u8, u16>(s),
                            bytemuck::try_cast_slice_mut::<u8, u16>(d),
                        )
                    {
                        ex.transform(s16, d16)
                            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    } else {
                        let mut s_buf = [0u16; CHUNK_PX * 4];
                        let mut d_buf = [0u16; CHUNK_PX * 4];
                        let mut pos = 0;
                        while pos < w {
                            let n = (w - pos).min(CHUNK_PX);
                            let sc = n * self.src_ch;
                            let dc = n * self.dst_ch;
                            let sb = &s[pos * self.src_bpp..(pos + n) * self.src_bpp];
                            for (dst_s, chunk) in s_buf[..sc].iter_mut().zip(sb.as_chunks::<2>().0)
                            {
                                let v = u16::from_ne_bytes(*chunk);
                                *dst_s = if self.src_cmyk_inverted { !v } else { v };
                            }
                            ex.transform(&s_buf[..sc], &mut d_buf[..dc])
                                .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                            let db = &mut d[pos * self.dst_bpp..(pos + n) * self.dst_bpp];
                            for (&v, chunk) in d_buf[..dc].iter().zip(db.as_chunks_mut::<2>().0) {
                                let out = if self.dst_cmyk_inverted { !v } else { v };
                                *chunk = out.to_ne_bytes();
                            }
                            pos += n;
                        }
                    }
                }
                MoxExec::F32(ex) => {
                    if !self.src_cmyk_inverted
                        && !self.dst_cmyk_inverted
                        && let (Ok(sf), Ok(df)) = (
                            bytemuck::try_cast_slice::<u8, f32>(s),
                            bytemuck::try_cast_slice_mut::<u8, f32>(d),
                        )
                    {
                        ex.transform(sf, df)
                            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    } else {
                        let mut s_buf = [0.0f32; CHUNK_PX * 4];
                        let mut d_buf = [0.0f32; CHUNK_PX * 4];
                        let mut pos = 0;
                        while pos < w {
                            let n = (w - pos).min(CHUNK_PX);
                            let sc = n * self.src_ch;
                            let dc = n * self.dst_ch;
                            let sb = &s[pos * self.src_bpp..(pos + n) * self.src_bpp];
                            for (dst_s, chunk) in s_buf[..sc].iter_mut().zip(sb.as_chunks::<4>().0)
                            {
                                let v = f32::from_ne_bytes(*chunk);
                                *dst_s = if self.src_cmyk_inverted { 1.0 - v } else { v };
                            }
                            ex.transform(&s_buf[..sc], &mut d_buf[..dc])
                                .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                            let db = &mut d[pos * self.dst_bpp..(pos + n) * self.dst_bpp];
                            for (&v, chunk) in d_buf[..dc].iter().zip(db.as_chunks_mut::<4>().0) {
                                let out = if self.dst_cmyk_inverted { 1.0 - v } else { v };
                                *chunk = out.to_ne_bytes();
                            }
                            pos += n;
                        }
                    }
                }
            }
            if self.fill_opaque_alpha {
                let sz = self.dst_sample.byte_size();
                let opaque: &[u8] = match self.dst_sample {
                    SampleType::U8 => &[255],
                    SampleType::U16 => &[255, 255],
                    SampleType::F32 => &1.0f32.to_ne_bytes(),
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                for px in d.chunks_exact_mut(self.dst_bpp) {
                    px[self.dst_bpp - sz..].copy_from_slice(opaque);
                }
            }
            Ok(())
        }
    }

    impl CmsBackend for MoxCms {
        fn build_transform(
            &self,
            src: &ImageSpec,
            dst: &ImageSpec,
        ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>> {
            if src.layout.sample != dst.layout.sample
                || !src.layout.packing.is_full()
                || !dst.layout.packing.is_full()
                || src.color.alpha() == AlphaMode::Premultiplied
                || dst.color.alpha() == AlphaMode::Premultiplied
            {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            let (src_p, src_intent) = profile_from_color_space(&src.color)?;
            let (dst_p, dst_intent) = profile_from_color_space(&dst.color)?;
            let eff_intent = src_intent
                .or(dst_intent)
                .unwrap_or(RenderingIntent::RelativeColorimetric);
            let src_l = to_mox_layout(src.layout.order)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            let dst_l = to_mox_layout(dst.layout.order)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            let opts = moxcms::TransformOptions {
                rendering_intent: to_mox_intent(eff_intent),
                ..Default::default()
            };
            let exec = match src.layout.sample {
                SampleType::U8 => MoxExec::U8(
                    src_p
                        .create_transform_8bit(src_l, &dst_p, dst_l, opts)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?,
                ),
                SampleType::U16 => MoxExec::U16(
                    src_p
                        .create_transform_16bit(src_l, &dst_p, dst_l, opts)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?,
                ),
                SampleType::F32 => MoxExec::F32(
                    src_p
                        .create_transform_f32(src_l, &dst_p, dst_l, opts)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?,
                ),
                _ => return Err(err_at(ConvertError::UnsupportedRoute)),
            };
            let fill_opaque_alpha = dst.layout.order.has_alpha_slot()
                && (dst.color.alpha() == AlphaMode::Opaque
                    || !src.layout.order.has_alpha_slot()
                    || src.color.alpha() == AlphaMode::Opaque);
            Ok(Arc::new(MoxRowTransform {
                exec,
                src_bpp: src.layout.bytes_per_pixel(),
                dst_bpp: dst.layout.bytes_per_pixel(),
                src_ch: src.layout.channels(),
                dst_ch: dst.layout.channels(),
                dst_sample: dst.layout.sample,
                src_cmyk_inverted: matches!(src.color, ColorSpace::Cmyk { inverted: true, .. })
                    || src.layout.order == ChannelOrder::CmykInverted,
                dst_cmyk_inverted: matches!(dst.color, ColorSpace::Cmyk { inverted: true, .. })
                    || dst.layout.order == ChannelOrder::CmykInverted,
                fill_opaque_alpha,
            }))
        }
    }
}

#[cfg(feature = "cms-moxcms")]
pub use moxcms_impl::MoxCms;

#[cfg(feature = "cms-lcms")]
mod lcms_impl {
    use super::*;
    use crate::error::err_at;
    use crate::icc_profiles::SynthesizedIcc;
    use lcms::api::{
        CiexyY, CiexyYTriple, D65, Format, Intent, Plan, PlanOptions, Profile, ToneCurve, flags,
    };
    use zenpixels::{
        AlphaMode, ChannelOrder, Cicp, ColorSpace, Primaries, RenderingIntent, SampleLayout,
        SampleType, SignalRange, Transfer,
    };

    const CHUNK_PX: usize = 64;

    const PT_GRAY: u32 = 3;
    const PT_RGB: u32 = 4;
    const PT_CMYK: u32 = 6;
    const DOSWAP_SH: u32 = 1 << 10;
    const SWAPFIRST_SH: u32 = 1 << 14;
    const FLOAT_SH: u32 = 1 << 22;

    /// Pure-Rust Little CMS (`imazen/lcms`) ICC CMS backend.
    ///
    /// Stateless and safe to share (`Send + Sync`); built transforms hold an
    /// immutable, lock-free `lcms::api::Plan` with AVX2/SIMD pipeline
    /// optimization enabled and stack-only per-row scratch (zero heap
    /// allocation per row).
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Lcms;

    /// Pure-Rust Little CMS backend with `FLAGS_NOOPTIMIZE` enabled.
    ///
    /// Evaluates profile stages directly without resampling into a 16-bit
    /// integer CLUT — slower on integer formats, useful for high-dynamic-range
    /// float-pipeline MPE profiles (such as PQ ICC profiles) where a 33-point
    /// integer CLUT loses shadow precision.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct LcmsUnoptimized;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum InkMode {
        None,
        InvertedInt,
        FloatPercent,
        FloatInvertedPercent,
    }

    #[derive(Clone, Copy)]
    struct LcmsFormatInfo {
        format: Format,
        bpp: usize,
        sample: SampleType,
        has_alpha: bool,
        ink: InkMode,
    }

    fn describe_layout(layout: SampleLayout, color: &ColorSpace) -> Option<LcmsFormatInfo> {
        if !layout.packing.is_full() {
            return None;
        }
        let sample = layout.sample;
        if !matches!(
            sample,
            SampleType::U8 | SampleType::U16 | SampleType::F16 | SampleType::F32
        ) {
            return None;
        }
        let inverted_cmyk = matches!(color, ColorSpace::Cmyk { inverted: true, .. })
            || layout.order == ChannelOrder::CmykInverted;

        let (space, colors, extra, swap, first, ink) = match layout.order {
            ChannelOrder::Gray => (PT_GRAY, 1u32, 0u32, false, false, InkMode::None),
            ChannelOrder::GrayAlpha => {
                if sample == SampleType::F16 {
                    return None;
                }
                (PT_GRAY, 1, 1, false, false, InkMode::None)
            }
            ChannelOrder::Rgb => (PT_RGB, 3, 0, false, false, InkMode::None),
            ChannelOrder::Bgr => {
                if sample == SampleType::F16 {
                    return None;
                }
                (PT_RGB, 3, 0, true, false, InkMode::None)
            }
            ChannelOrder::Rgba | ChannelOrder::Rgbx => (PT_RGB, 3, 1, false, false, InkMode::None),
            ChannelOrder::Bgra | ChannelOrder::Bgrx => {
                if sample == SampleType::F16 {
                    return None;
                }
                (PT_RGB, 3, 1, true, true, InkMode::None)
            }
            ChannelOrder::Cmyk | ChannelOrder::CmykInverted => {
                let ink = match (sample, inverted_cmyk) {
                    (SampleType::U8 | SampleType::U16, false) => InkMode::None,
                    (SampleType::U8 | SampleType::U16, true) => InkMode::InvertedInt,
                    (SampleType::F32, false) => InkMode::FloatPercent,
                    (SampleType::F32, true) => InkMode::FloatInvertedPercent,
                    _ => return None,
                };
                (PT_CMYK, 4, 0, false, false, ink)
            }
            _ => return None,
        };

        let float_bit = if sample.is_float() { FLOAT_SH } else { 0 };
        let code = float_bit
            | (space << 16)
            | (u32::from(first) * SWAPFIRST_SH)
            | (u32::from(swap) * DOSWAP_SH)
            | (extra << 7)
            | (colors << 3)
            | (sample.byte_size() as u32);

        Some(LcmsFormatInfo {
            format: Format::from_u32(code),
            bpp: layout.bytes_per_pixel(),
            sample,
            has_alpha: layout.order.has_alpha_slot(),
            ink,
        })
    }

    fn to_lcms_intent(intent: RenderingIntent) -> Intent {
        match intent {
            RenderingIntent::Perceptual => Intent::Perceptual,
            RenderingIntent::RelativeColorimetric => Intent::RelativeColorimetric,
            RenderingIntent::Saturation => Intent::Saturation,
            RenderingIntent::AbsoluteColorimetric => Intent::AbsoluteColorimetric,
            _ => Intent::RelativeColorimetric,
        }
    }

    fn tone_curve_for_transfer(transfer: Transfer) -> Result<Option<ToneCurve>, At<ConvertError>> {
        match transfer {
            Transfer::Srgb => Ok(Some(ToneCurve::srgb())),
            Transfer::Linear => Ok(Some(ToneCurve::linear())),
            Transfer::Gamma22 => ToneCurve::gamma(563.0 / 256.0)
                .map(Some)
                .map_err(|_| err_at(ConvertError::CmsBackendFailed)),
            Transfer::Bt709 => ToneCurve::parametric(
                4,
                &[1.0 / 0.45, 1.0 / 1.099, 0.099 / 1.099, 1.0 / 4.5, 0.081],
            )
            .map(Some)
            .map_err(|_| err_at(ConvertError::CmsBackendFailed)),
            _ => Ok(None),
        }
    }

    fn primaries_triple(primaries: Primaries) -> Option<CiexyYTriple> {
        let ((rx, ry), (gx, gy), (bx, by)) = primaries.chromaticity()?;
        Some(CiexyYTriple {
            Red: CiexyY {
                x: f64::from(rx),
                y: f64::from(ry),
                Y: 1.0,
            },
            Green: CiexyY {
                x: f64::from(gx),
                y: f64::from(gy),
                Y: 1.0,
            },
            Blue: CiexyY {
                x: f64::from(bx),
                y: f64::from(by),
                Y: 1.0,
            },
        })
    }

    fn profile_from_color_space(
        cs: &ColorSpace,
    ) -> Result<(Profile, Option<RenderingIntent>), At<ConvertError>> {
        match cs {
            ColorSpace::Icc {
                profile, intent, ..
            } => {
                let p =
                    Profile::open(profile).map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                Ok((p, Some(*intent)))
            }
            ColorSpace::Cmyk { icc, .. } => {
                let bytes = icc
                    .as_ref()
                    .ok_or_else(|| err_at(ConvertError::CmsRequired))?;
                let p = Profile::open(bytes).map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                Ok((p, Some(RenderingIntent::RelativeColorimetric)))
            }
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                ..
            } => {
                if *range != SignalRange::Full || matches!(transfer, Transfer::Hlg(_)) {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                if *primaries == Primaries::Bt709 && *transfer == Transfer::Srgb {
                    return Ok((Profile::srgb(), None));
                }
                if *primaries == Primaries::AdobeRgb && *transfer == Transfer::Gamma22 {
                    let p = Profile::open(crate::icc_profiles::ADOBE_RGB)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                if let (Some(triple), Some(curve)) = (
                    primaries_triple(*primaries),
                    tone_curve_for_transfer(*transfer)?,
                ) {
                    let p = Profile::rgb(&D65, &triple, [&curve, &curve, &curve])
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                if let (Some(p_code), Some(t_code)) = (primaries.to_cicp(), transfer.to_cicp())
                    && let SynthesizedIcc::Profile(bytes) =
                        SynthesizedIcc::for_cicp(Cicp::new(p_code, t_code, 0, true))
                {
                    let p = Profile::open(&bytes)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                Err(err_at(ConvertError::UnsupportedRoute))
            }
            ColorSpace::Gray {
                transfer, range, ..
            } => {
                if *range != SignalRange::Full || matches!(transfer, Transfer::Hlg(_)) {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                if let Some(curve) = tone_curve_for_transfer(*transfer)? {
                    let p = Profile::gray_curve(&D65, &curve)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                if let Some(t_code) = transfer.to_cicp()
                    && let SynthesizedIcc::Profile(bytes) =
                        SynthesizedIcc::for_gray_cicp(Cicp::new(1, t_code, 0, true))
                {
                    let p = Profile::open(&bytes)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                Err(err_at(ConvertError::UnsupportedRoute))
            }
            _ => Err(err_at(ConvertError::UnsupportedRoute)),
        }
    }

    struct LcmsRowTransform {
        plan: Plan,
        src: LcmsFormatInfo,
        dst: LcmsFormatInfo,
        /// Write canonical opaque alpha (1.0 / max code) to dst alpha slot.
        fill_opaque_alpha: bool,
        /// Guard for float RGBA CLUT fast paths in `lcms` that zero the 4th f32 lane
        /// after `handle_extra_channels` has already copied alpha.
        copy_f32_alpha_after: bool,
    }

    fn build_lcms_transform(
        src: &ImageSpec,
        dst: &ImageSpec,
        no_optimize: bool,
    ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>> {
        if src.color.alpha() == AlphaMode::Premultiplied
            || dst.color.alpha() == AlphaMode::Premultiplied
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let from = describe_layout(src.layout, &src.color)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        let to = describe_layout(dst.layout, &dst.color)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;

        let (src_profile, src_intent) = profile_from_color_space(&src.color)?;
        let (dst_profile, dst_intent) = profile_from_color_space(&dst.color)?;
        let eff_intent = src_intent
            .or(dst_intent)
            .unwrap_or(RenderingIntent::RelativeColorimetric);

        let lcms_intent = to_lcms_intent(eff_intent);
        let bpc = matches!(src.color, ColorSpace::Cmyk { .. })
            && eff_intent == RenderingIntent::RelativeColorimetric;
        let adaptation = if eff_intent == RenderingIntent::AbsoluteColorimetric {
            0.0
        } else {
            1.0
        };

        let fill_opaque_alpha = to.has_alpha
            && (dst.color.alpha() == AlphaMode::Opaque
                || !from.has_alpha
                || src.color.alpha() == AlphaMode::Opaque);

        // `NO_CACHE` enables lcms's fused AVX2/SIMD loops (Prelin8, Clut8/16,
        // MatShaper) and keeps per-row execution cache-independent.
        let mut transform_flags = flags::NO_CACHE;
        if no_optimize {
            transform_flags |= flags::NO_OPTIMIZE;
        }
        if from.has_alpha && to.has_alpha && !fill_opaque_alpha {
            transform_flags |= flags::COPY_ALPHA;
        }
        if bpc {
            transform_flags |= flags::BLACK_POINT_COMPENSATION;
        }

        let opts = PlanOptions::new()
            .intent(lcms_intent)
            .flags(transform_flags)
            .adaptation_state(adaptation);

        let plan = Plan::with_options(&src_profile, &dst_profile, from.format, to.format, &opts)
            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;

        let copy_f32_alpha_after = from.has_alpha
            && to.has_alpha
            && !fill_opaque_alpha
            && from.sample == SampleType::F32
            && to.sample == SampleType::F32
            && from.bpp == 16
            && to.bpp == 16;

        Ok(Arc::new(LcmsRowTransform {
            plan,
            src: from,
            dst: to,
            fill_opaque_alpha,
            copy_f32_alpha_after,
        }))
    }

    impl CmsTransform for LcmsRowTransform {
        fn transform_row(
            &self,
            src: &[u8],
            dst: &mut [u8],
            width: u32,
        ) -> Result<(), At<ConvertError>> {
            let w = width as usize;
            let s_len = w
                .checked_mul(self.src.bpp)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            let d_len = w
                .checked_mul(self.dst.bpp)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            if src.len() < s_len || dst.len() < d_len {
                return Err(err_at(ConvertError::DimensionMismatch));
            }
            if w == 0 {
                return Ok(());
            }

            let s = &src[..s_len];
            let d = &mut dst[..d_len];
            let mut scratch = self.plan.scratch();

            match self.src.ink {
                InkMode::None => {
                    self.plan.apply_n(&mut scratch, s, d, width);
                }
                InkMode::InvertedInt => {
                    // Bitwise NOT inverts both u8 (`255 - x`) and native-endian u16 (`65535 - x`).
                    // Max CMYK integer bpp is 8 bytes (CMYK16), so 64 px = 512 B on stack.
                    let mut staged = [0u8; CHUNK_PX * 8];
                    let mut pos = 0;
                    while pos < w {
                        let n = (w - pos).min(CHUNK_PX);
                        let sb = &s[pos * self.src.bpp..(pos + n) * self.src.bpp];
                        let db = &mut d[pos * self.dst.bpp..(pos + n) * self.dst.bpp];
                        let tmp = &mut staged[..sb.len()];
                        for (o, &b) in tmp.iter_mut().zip(sb) {
                            *o = !b;
                        }
                        self.plan.apply_n(&mut scratch, tmp, db, n as u32);
                        pos += n;
                    }
                }
                InkMode::FloatPercent | InkMode::FloatInvertedPercent => {
                    let invert = self.src.ink == InkMode::FloatInvertedPercent;
                    let mut staged = [0.0f32; CHUNK_PX * 4];
                    let mut pos = 0;
                    while pos < w {
                        let n = (w - pos).min(CHUNK_PX);
                        let sb = &s[pos * self.src.bpp..(pos + n) * self.src.bpp];
                        let db = &mut d[pos * self.dst.bpp..(pos + n) * self.dst.bpp];
                        let tmp = &mut staged[..n * 4];
                        for (o, bytes) in tmp.iter_mut().zip(sb.as_chunks::<4>().0) {
                            let v = f32::from_ne_bytes(*bytes);
                            *o = 100.0 * if invert { 1.0 - v } else { v };
                        }
                        self.plan
                            .apply_n(&mut scratch, bytemuck::cast_slice(tmp), db, n as u32);
                        pos += n;
                    }
                }
            }

            // Destination CMYK post-processing (inversion / percent -> [0, 1]).
            match self.dst.ink {
                InkMode::None => {}
                InkMode::InvertedInt => {
                    for b in d.iter_mut() {
                        *b = !*b;
                    }
                }
                InkMode::FloatPercent => {
                    for chunk in d.as_chunks_mut::<4>().0 {
                        let v = f32::from_ne_bytes(*chunk) * 0.01;
                        *chunk = v.to_ne_bytes();
                    }
                }
                InkMode::FloatInvertedPercent => {
                    for chunk in d.as_chunks_mut::<4>().0 {
                        let v = 1.0 - f32::from_ne_bytes(*chunk) * 0.01;
                        *chunk = v.to_ne_bytes();
                    }
                }
            }

            // Fill opaque alpha when adding an alpha channel or canonicalizing Opaque alpha.
            if self.fill_opaque_alpha {
                let sz = self.dst.sample.byte_size();
                let opaque: &[u8] = match self.dst.sample {
                    SampleType::U8 => &[255],
                    SampleType::U16 => &[255, 255],
                    SampleType::F16 => &0x3C00u16.to_ne_bytes(),
                    SampleType::F32 => &1.0f32.to_ne_bytes(),
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                for px in d.chunks_exact_mut(self.dst.bpp) {
                    px[self.dst.bpp - sz..].copy_from_slice(opaque);
                }
            }

            // Guard against RGBA_FLT CLUT fast paths overwriting the 4th f32 lane.
            if self.copy_f32_alpha_after {
                for (sp, dp) in s.as_chunks::<16>().0.iter().zip(d.as_chunks_mut::<16>().0) {
                    dp[12..16].copy_from_slice(&sp[12..16]);
                }
            }

            Ok(())
        }
    }

    impl CmsBackend for Lcms {
        fn build_transform(
            &self,
            src: &ImageSpec,
            dst: &ImageSpec,
        ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>> {
            build_lcms_transform(src, dst, false)
        }
    }

    impl CmsBackend for LcmsUnoptimized {
        fn build_transform(
            &self,
            src: &ImageSpec,
            dst: &ImageSpec,
        ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>> {
            build_lcms_transform(src, dst, true)
        }
    }
}

#[cfg(feature = "cms-lcms")]
pub use lcms_impl::{Lcms, LcmsUnoptimized};
