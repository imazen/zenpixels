//! ICC transforms through the pure-Rust Little CMS port (`cms-lcms`).
//!
//! [`Lcms`] implements the same [`PluggableCms`] interface as
//! `MoxCms`. Select a backend explicitly; enabling both
//! features does not change the selected backend or the built-in matrix paths.
//! No C library is linked by this backend.
//!
//! Supported sample types are native-endian U8, U16 and F32 for RGB/RGBA,
//! gray/gray-alpha and ICC-described CMYK. BGR/BGRA byte layouts are supported.
//! Other layouts and depth changes decline so the planner can handle them.
//! F32 CMYK destinations are refused pending qualification. CICP sources use
//! the shared profile database; reserved codes and unavailable profiles decline.
//!
//! Transforms use relative colorimetric intent, with black-point compensation
//! for CMYK sources. Optimization is disabled to avoid resampling the profile's
//! pipeline. F32 clipping follows [`crate::ConvertOptions`]. A prepared CMYK
//! float row reuses scratch, and row extents are checked before any writes.
//!
//! ```no_run
//! use zenpixels_convert::{Lcms, PluggableCms};
//! let backend: &dyn PluggableCms = &Lcms;
//! # let _ = backend;
//! ```

use crate::cms::{CmsPluginError, PluggableCms, RowTransformMut};
use crate::icc_profiles::SynthesizedIcc;
use crate::{ChannelType, Cicp, ColorModel, ColorProfileSource, PixelFormat};
use alloc::{boxed::Box, format, vec::Vec};
use lcms::{profile, types, virt, xform};

/// The pure-Rust Little CMS backend.
///
/// Stateless and safe to share. Each converter owns its mutable transform.
/// The pinned dependency is unreleased; publishing this feature requires a
/// released version of that dependency. Backend types stay out of this API.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lcms;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ink {
    None,
    InvertedInteger,
    Float,
    FloatInverted,
}

#[derive(Clone, Copy)]
struct Format {
    code: u32,
    bytes: usize,
    colors: usize,
    alpha: bool,
    ink: Ink,
    sample: ChannelType,
}

fn describe(format: PixelFormat) -> Option<Format> {
    use PixelFormat::*;
    let (space, colors, alpha, swap, first, ink) = match format {
        Rgb8 | Rgb16 | RgbF32 => (4, 3, false, false, false, Ink::None),
        Rgba8 | Rgba16 | RgbaF32 => (4, 3, true, false, false, Ink::None),
        Bgr8 => (4, 3, false, true, false, Ink::None),
        Bgra8 => (4, 3, true, true, true, Ink::None),
        Gray8 | Gray16 | GrayF32 => (3, 1, false, false, false, Ink::None),
        GrayA8 | GrayA16 | GrayAF32 => (3, 1, true, false, false, Ink::None),
        Cmyk8 | Cmyk16 => (6, 4, false, false, false, Ink::None),
        Cmyk8Inverted | Cmyk16Inverted => (6, 4, false, false, false, Ink::InvertedInteger),
        CmykF32 => (6, 4, false, false, false, Ink::Float),
        CmykF32Inverted => (6, 4, false, false, false, Ink::FloatInverted),
        _ => return None,
    };
    let sample = format.channel_type();
    let size = sample.byte_size();
    let code = u32::from(sample == ChannelType::F32) << 22
        | space << 16
        | u32::from(first) << 14
        | u32::from(swap) << 10
        | u32::from(alpha) << 7
        | colors << 3
        | size as u32;
    Some(Format {
        code,
        bytes: format.bytes_per_pixel(),
        colors: colors as usize,
        alpha,
        ink,
        sample,
    })
}

type Error = whereat::At<CmsPluginError>;
#[track_caller]
fn error(message: impl Into<alloc::string::String>) -> Error {
    whereat::at!(CmsPluginError::msg(message))
}

fn gray_srgb() -> Result<Box<profile::IccProfile>, Error> {
    let mut rgb =
        virt::create_srgb_profile(None).ok_or_else(|| error("lcms sRGB profile failed"))?;
    let curve = match profile::cms_read_tag(&mut rgb, lcms::signatures::tag::SigRedTRCTag) {
        Some(
            lcms::tagtypes::TagObject::Curve(curve)
            | lcms::tagtypes::TagObject::ParametricCurve(curve),
        ) => curve.clone(),
        _ => return Err(error("lcms sRGB curve unavailable")),
    };
    let white = lcms::types::CiexyY {
        x: 0.3127,
        y: 0.3290,
        Y: 1.0,
    };
    virt::create_gray_profile(None, Some(&white), Some(&curve))
        .ok_or_else(|| error("lcms gray sRGB profile failed"))
}

fn open_cicp(cicp: Cicp, gray: bool) -> Result<Option<Box<profile::IccProfile>>, Error> {
    if cicp.matrix_coefficients != 0 || !cicp.full_range {
        return Err(error(
            "lcms requires full-range RGB CICP; convert matrix/range separately",
        ));
    }
    if cicp.color_primaries == 1 && cicp.transfer_characteristics == 13 {
        return if gray {
            gray_srgb().map(Some)
        } else {
            virt::create_srgb_profile(None)
                .map(Some)
                .ok_or_else(|| error("lcms sRGB profile failed"))
        };
    }
    let synthesized = if gray {
        SynthesizedIcc::for_gray_cicp(cicp)
    } else {
        SynthesizedIcc::for_cicp(cicp)
    };
    match synthesized {
        SynthesizedIcc::Profile(bytes) => profile::open_profile_from_mem(None, &bytes)
            .map(Some)
            .ok_or_else(|| error("lcms could not parse the selected CICP profile")),
        _ => Ok(None),
    }
}

fn extract_red_trc(mut rgb: Box<profile::IccProfile>) -> Result<lcms::gamma::ToneCurve, Error> {
    match profile::cms_read_tag(&mut rgb, lcms::signatures::tag::SigRedTRCTag) {
        Some(
            lcms::tagtypes::TagObject::Curve(curve)
            | lcms::tagtypes::TagObject::ParametricCurve(curve),
        ) => Ok(curve.clone()),
        _ => Err(error("lcms reference TRC unavailable")),
    }
}

fn open_parametric(
    p: zenpixels::ParametricColor,
    gray: bool,
) -> Result<Option<Box<profile::IccProfile>>, Error> {
    if gray != p.primaries.is_none() {
        return Ok(None);
    }
    if matches!(
        p.transfer,
        zenpixels::ParametricTransfer::Hlg | zenpixels::ParametricTransfer::Unknown
    ) {
        return Ok(None);
    }
    if !gray
        && p.transfer != zenpixels::ParametricTransfer::Dci
        && let Some(cicp) = p.to_cicp()
    {
        return open_cicp(cicp, false);
    }

    let curve =
        match p.transfer {
            zenpixels::ParametricTransfer::Linear => lcms::gamma::build_gamma(None, 1.0)
                .ok_or_else(|| error("lcms linear curve failed"))?,
            zenpixels::ParametricTransfer::Dci => lcms::gamma::build_gamma(None, 2.6)
                .ok_or_else(|| error("lcms DCI 2.6 curve failed"))?,
            zenpixels::ParametricTransfer::Gamma(g) => {
                lcms::gamma::build_gamma(None, f64::from(g.eotf_exponent()))
                    .ok_or_else(|| error("lcms gamma curve failed"))?
            }
            zenpixels::ParametricTransfer::Srgb => {
                let rgb = virt::create_srgb_profile(None)
                    .ok_or_else(|| error("lcms sRGB profile failed"))?;
                extract_red_trc(rgb)?
            }
            zenpixels::ParametricTransfer::Bt709 => {
                let Some(rgb) = open_cicp(Cicp::new(1, 1, 0, true), false)? else {
                    return Ok(None);
                };
                extract_red_trc(rgb)?
            }
            zenpixels::ParametricTransfer::Pq => {
                let Some(rgb) = open_cicp(Cicp::new(9, 16, 0, true), false)? else {
                    return Ok(None);
                };
                extract_red_trc(rgb)?
            }
            _ => return Ok(None),
        };

    let wp = p.white_point.cie_xy();
    if wp.y() <= 0.0 {
        return Err(error("parametric white point y must be positive"));
    }
    let white = lcms::types::CiexyY {
        x: f64::from(wp.x()),
        y: f64::from(wp.y()),
        Y: 1.0,
    };

    if gray {
        return virt::create_gray_profile(None, Some(&white), Some(&curve))
            .map(Some)
            .ok_or_else(|| error("lcms parametric gray profile failed"));
    }

    let Some(prim) = p.primaries else {
        return Ok(None);
    };
    let (r, g, b) = prim.cie_xy();
    if r.y() == 0.0 || g.y() == 0.0 || b.y() == 0.0 {
        return Err(error("parametric primary y must be non-zero"));
    }
    let triple = lcms::types::CiexyYTriple {
        Red: lcms::types::CiexyY {
            x: f64::from(r.x()),
            y: f64::from(r.y()),
            Y: 1.0,
        },
        Green: lcms::types::CiexyY {
            x: f64::from(g.x()),
            y: f64::from(g.y()),
            Y: 1.0,
        },
        Blue: lcms::types::CiexyY {
            x: f64::from(b.x()),
            y: f64::from(b.y()),
            Y: 1.0,
        },
    };
    virt::create_rgb_profile(
        None,
        Some(&white),
        Some(&triple),
        Some([&curve, &curve, &curve]),
    )
    .map(Some)
    .ok_or_else(|| error("lcms parametric RGB profile failed"))
}

fn open(
    source: ColorProfileSource<'_>,
    format: PixelFormat,
) -> Result<Option<Box<profile::IccProfile>>, Error> {
    let gray = format.color_model() == ColorModel::Gray;
    match source {
        ColorProfileSource::Icc(bytes) => profile::open_profile_from_mem(None, bytes)
            .map(Some)
            .ok_or_else(|| error("lcms could not parse ICC profile")),
        _ if format.color_model() == ColorModel::Cmyk => Ok(None),
        ColorProfileSource::Cicp(cicp) => open_cicp(cicp, gray),
        ColorProfileSource::Parametric(p) => open_parametric(p, gray),
        ColorProfileSource::Named(named) => {
            if let Some(cicp) = named.to_cicp() {
                open_cicp(cicp, gray)
            } else {
                let (primaries, transfer) = named.to_primaries_transfer();
                open(
                    ColorProfileSource::PrimariesTransferPair {
                        primaries,
                        transfer,
                    },
                    format,
                )
            }
        }
        ColorProfileSource::PrimariesTransferPair {
            primaries,
            transfer,
        } => {
            if let (Some(p), Some(t)) = (primaries.to_cicp(), transfer.to_cicp()) {
                open_cicp(Cicp::new(p, t, 0, true), gray)
            } else if primaries == crate::ColorPrimaries::AdobeRgb
                && transfer == crate::TransferFunction::Gamma22
                && !gray
            {
                open(
                    ColorProfileSource::Icc(crate::icc_profiles::ADOBE_RGB),
                    format,
                )
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

impl PluggableCms for Lcms {
    fn build_source_transform(
        &self,
        src: ColorProfileSource<'_>,
        dst: ColorProfileSource<'_>,
        src_format: PixelFormat,
        dst_format: PixelFormat,
        options: &crate::policy::ConvertOptions,
    ) -> Option<Result<Box<dyn RowTransformMut>, Error>> {
        let from = describe(src_format)?;
        let to = describe(dst_format)?;
        if from.sample != to.sample {
            return None;
        }
        if to.ink != Ink::None {
            return Some(Err(error(
                "conversion into F32 or inverted CMYK is not qualified",
            )));
        }
        let built = (|| {
            let Some(source) = open(src, src_format)? else {
                return Ok(None);
            };
            let Some(target) = open(dst, dst_format)? else {
                return Ok(None);
            };
            let mut profiles = [source, target];
            let bpc = src_format.color_model() == ColorModel::Cmyk;
            let flags = types::FLAGS_NOOPTIMIZE
                | types::FLAGS_NOCACHE
                | if from.alpha && to.alpha {
                    types::FLAGS_COPY_ALPHA
                } else {
                    0
                }
                | if bpc {
                    types::FLAGS_BLACKPOINTCOMPENSATION
                } else {
                    0
                };
            let transform = xform::create_extended_transform(
                None,
                &mut profiles,
                &mut [bpc; 2],
                &[types::INTENT_RELATIVE_COLORIMETRIC; 2],
                &[1.0; 2],
                None,
                0,
                from.code,
                to.code,
                flags,
            )
            .ok_or_else(|| error("lcms refused this ICC/layout transform"))?;
            Ok(Some(Box::new(Row {
                transform,
                from,
                to,
                scratch: Vec::new(),
                prepared: None,
                clip: options.clip_out_of_gamut() && to.sample == ChannelType::F32,
            }) as Box<dyn RowTransformMut>))
        })();
        match built {
            Ok(None) => None,
            Ok(Some(row)) => Some(Ok(row)),
            Err(e) => Some(Err(e)),
        }
    }
}

struct Row {
    transform: Box<xform::Transform>,
    from: Format,
    to: Format,
    scratch: Vec<f32>,
    prepared: Option<u32>,
    clip: bool,
}

impl Row {
    fn reserve(&mut self, width: u32) -> Result<(), Error> {
        if self.from.ink != Ink::None {
            let length = (width as usize)
                .checked_mul(4)
                .ok_or_else(|| error("lcms row overflow"))?;
            if length > self.scratch.len() {
                self.scratch
                    .try_reserve(length - self.scratch.len())
                    .map_err(|e| error(format!("lcms CMYK scratch allocation failed: {e}")))?;
                self.scratch.resize(length, 0.0);
            }
        }
        Ok(())
    }
}

impl RowTransformMut for Row {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) {
        self.try_transform_row(src, dst, width)
            .expect("lcms row failed");
    }

    fn try_transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) -> Result<(), Error> {
        let pixels = width as usize;
        let source_length = pixels
            .checked_mul(self.from.bytes)
            .ok_or_else(|| error("lcms row overflow"))?;
        let target_length = pixels
            .checked_mul(self.to.bytes)
            .ok_or_else(|| error("lcms row overflow"))?;
        if src.len() < source_length || dst.len() < target_length {
            return Err(error("lcms row is shorter than its width"));
        }
        if self.prepared.is_some_and(|maximum| width > maximum) {
            return Err(error("lcms row exceeds prepared width"));
        }
        if width == 0 {
            return Ok(());
        }
        self.reserve(width)?;
        let src = &src[..source_length];
        let dst = &mut dst[..target_length];
        match self.from.ink {
            Ink::None => {
                self.transform.do_transform(src, dst, width);
            }
            Ink::InvertedInteger => {
                // Bitwise NOT inverts both u8 (`255 - x`) and native-endian u16 (`65535 - x`).
                let inverted: &mut [u8] =
                    &mut bytemuck::cast_slice_mut(&mut self.scratch)[..source_length];
                for (out, byte) in inverted.iter_mut().zip(src) {
                    *out = !*byte;
                }
                self.transform.do_transform(inverted, dst, width);
            }
            Ink::Float | Ink::FloatInverted => {
                for (out, bytes) in self.scratch[..pixels * 4]
                    .iter_mut()
                    .zip(src.as_chunks::<4>().0)
                {
                    let value = f32::from_ne_bytes(*bytes);
                    *out = 100.0
                        * if self.from.ink == Ink::FloatInverted {
                            1.0 - value
                        } else {
                            value
                        };
                }
                self.transform.do_transform(
                    bytemuck::cast_slice(&self.scratch[..pixels * 4]),
                    dst,
                    width,
                );
            }
        }
        if self.to.alpha && !self.from.alpha {
            let size = self.to.sample.byte_size();
            let opaque: &[u8] = match self.to.sample {
                ChannelType::U8 => &[255],
                ChannelType::U16 => &[255, 255],
                ChannelType::F32 => &1.0f32.to_ne_bytes(),
                _ => unreachable!(),
            };
            for pixel in dst.chunks_exact_mut(self.to.bytes) {
                pixel[self.to.bytes - size..].copy_from_slice(opaque);
            }
        }
        if self.clip {
            for pixel in dst.chunks_exact_mut(self.to.bytes) {
                for sample in pixel[..self.to.colors * 4].as_chunks_mut::<4>().0 {
                    let value = f32::from_ne_bytes(*sample);
                    sample.copy_from_slice(&value.clamp(0.0, 1.0).to_ne_bytes());
                }
            }
        }
        Ok(())
    }

    fn prepare(&mut self, max_width: u32) -> Result<(), Error> {
        self.reserve(max_width)?;
        self.prepared = Some(max_width);
        Ok(())
    }
}
