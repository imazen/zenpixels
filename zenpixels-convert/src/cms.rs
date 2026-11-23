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
    use zenpixels::{ChannelOrder, ColorSpace, Primaries, RenderingIntent, SampleType, Transfer};

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
    ) -> Result<(moxcms::ColorProfile, RenderingIntent), At<ConvertError>> {
        match cs {
            ColorSpace::Icc {
                profile, intent, ..
            } => {
                let p = moxcms::ColorProfile::new_from_slice(profile)
                    .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                Ok((p, *intent))
            }
            ColorSpace::Cmyk { icc, .. } => {
                let bytes = icc
                    .as_ref()
                    .ok_or_else(|| err_at(ConvertError::CmsRequired))?;
                let p = moxcms::ColorProfile::new_from_slice(bytes)
                    .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                Ok((p, RenderingIntent::Perceptual))
            }
            ColorSpace::Rgb {
                primaries,
                transfer,
                ..
            } => {
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
                    (Primaries::Bt2020, Transfer::Hlg(_)) => moxcms::ColorProfile::new_bt2020_hlg(),
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                Ok((p, RenderingIntent::RelativeColorimetric))
            }
            ColorSpace::Gray { transfer, .. } => {
                let p = match transfer {
                    Transfer::Srgb => moxcms::ColorProfile::new_gray_with_gamma(2.2),
                    Transfer::Linear => moxcms::ColorProfile::new_gray_with_gamma(1.0),
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                Ok((p, RenderingIntent::RelativeColorimetric))
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
        src_cmyk_inverted: bool,
        dst_cmyk_inverted: bool,
    }

    impl CmsTransform for MoxRowTransform {
        fn transform_row(
            &self,
            src: &[u8],
            dst: &mut [u8],
            width: u32,
        ) -> Result<(), At<ConvertError>> {
            let w = width as usize;
            let s_len = w * self.src_bpp;
            let d_len = w * self.dst_bpp;
            let s = &src[..s_len];
            let d = &mut dst[..d_len];
            match &self.exec {
                MoxExec::U8(ex) => {
                    if self.src_cmyk_inverted {
                        let inv: alloc::vec::Vec<u8> = s.iter().map(|&b| 255 - b).collect();
                        ex.transform(&inv, d)
                            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    } else {
                        ex.transform(s, d)
                            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    }
                    if self.dst_cmyk_inverted {
                        for b in d.iter_mut() {
                            *b = 255 - *b;
                        }
                    }
                }
                MoxExec::U16(ex) => {
                    let s16: alloc::vec::Vec<u16> = s
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&c| u16::from_ne_bytes(c))
                        .collect();
                    let mut d16 = alloc::vec![0u16; d_len / 2];
                    ex.transform(&s16, &mut d16)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    for (i, &v) in d16.iter().enumerate() {
                        d[i * 2..i * 2 + 2].copy_from_slice(&v.to_ne_bytes());
                    }
                }
                MoxExec::F32(ex) => {
                    let sf: alloc::vec::Vec<f32> = s
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&c| f32::from_ne_bytes(c))
                        .collect();
                    let mut df = alloc::vec![0.0f32; d_len / 4];
                    ex.transform(&sf, &mut df)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    for (i, &v) in df.iter().enumerate() {
                        d[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
                    }
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
            if src.layout.sample != dst.layout.sample {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            let (src_p, src_intent) = profile_from_color_space(&src.color)?;
            let (dst_p, _) = profile_from_color_space(&dst.color)?;
            let src_l = to_mox_layout(src.layout.order)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            let dst_l = to_mox_layout(dst.layout.order)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            let opts = moxcms::TransformOptions {
                rendering_intent: to_mox_intent(src_intent),
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
            Ok(Arc::new(MoxRowTransform {
                exec,
                src_bpp: src.layout.bytes_per_pixel(),
                dst_bpp: dst.layout.bytes_per_pixel(),
                src_cmyk_inverted: matches!(src.color, ColorSpace::Cmyk { inverted: true, .. }),
                dst_cmyk_inverted: matches!(dst.color, ColorSpace::Cmyk { inverted: true, .. }),
            }))
        }
    }
}

#[cfg(feature = "cms-moxcms")]
pub use moxcms_impl::MoxCms;
