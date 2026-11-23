//! Atomic encode-ready output planning: converted pixels + matching CICP / ICC metadata.
//!
//! [`OutputPlan`] bundles pixel conversion with container color signaling so an
//! encoder can never accidentally write sRGB pixels alongside a stale Display P3
//! ICC profile. On identity (`src == dst`), [`OutputPlan::execute_buf`] reuses
//! the caller's [`PixelBuf`] **zero-copy**!

use alloc::borrow::Cow;
use alloc::sync::Arc;
use whereat::At;
use zenpixels::{
    Cicp, ColorOrigin, ColorSpace, HdrMetadata, ImageSpec, PeakSource, PixelBuf, PixelCow,
    PixelView, Primaries, Transfer,
};

use crate::error::{ConvertError, err_at};
use crate::icc_profiles::SynthesizedIcc;
use crate::plan::{ImagePlan, same_color_meaning, same_photometry};
use crate::policy::ConvertPolicy;

/// Requested container color signaling for an encoder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputProfile {
    /// Emit CICP when standardized (`sRGB`, `DisplayP3`, `BT.2100 PQ/HLG`), and fall back to bundled ICC otherwise (e.g. Adobe RGB).
    #[default]
    Auto,
    /// Preserve the original file's [`ColorOrigin`] signaling when compatible.
    SameAsOrigin,
    /// Emit explicit CICP codes only.
    CicpOnly,
    /// Embed an ICC profile (synthesized/bundled for known spaces, or original ICC).
    EmbedIcc,
    /// Embed an explicit caller-provided ICC profile.
    CustomIcc(Arc<[u8]>),
    /// Strip color metadata (assume sRGB / untagged).
    Strip,
}

/// Atomic output handed to an image encoder: pixels + guaranteed-matching CICP/ICC/HDR metadata.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EncodeReady<'a> {
    /// Pixels ready for encoding (zero-copy borrowed or moved when identity!).
    pub pixels: PixelCow<'a>,
    /// Authoritative CICP tuple matching `pixels`, if applicable.
    pub cicp: Option<Cicp>,
    /// Authoritative ICC profile bytes matching `pixels`, if applicable.
    pub icc: Option<Cow<'static, [u8]>>,
    /// HDR container metadata matching `pixels` (cleared automatically if converted to SDR).
    pub hdr: HdrMetadata,
}

/// Atomic encoder preparation plan combining [`ImagePlan`] with [`OutputProfile`] signaling.
#[derive(Clone, Debug)]
pub struct OutputPlan {
    src: ImageSpec,
    dst: ImageSpec,
    profile: OutputProfile,
    image_plan: ImagePlan,
    is_identity: bool,
}

impl OutputPlan {
    /// Build an encode-ready output plan.
    ///
    /// Validates pixel conversion feasibility AND container signaling compatibility
    /// upfront (for non-origin-dependent profiles) so invalid output configurations
    /// fail before processing any pixels.
    pub fn build(
        src: ImageSpec,
        dst: ImageSpec,
        policy: ConvertPolicy,
        peak: PeakSource,
        profile: OutputProfile,
    ) -> Result<Self, At<ConvertError>> {
        let is_identity = src == dst;
        let image_plan = ImagePlan::build(src.clone(), dst.clone(), policy, peak)?;
        // Eagerly validate signaling rules that don't depend on runtime `ColorOrigin`.
        match &profile {
            OutputProfile::SameAsOrigin => {
                if !same_color_meaning(&src.color, &dst.color) {
                    return Err(err_at(ConvertError::OutputSignalingMismatch));
                }
            }
            _ => {
                let _ = resolve_output_signaling(
                    &src.color,
                    &dst.color,
                    &profile,
                    &ColorOrigin::assumed(),
                )?;
            }
        }
        Ok(Self {
            src,
            dst,
            profile,
            image_plan,
            is_identity,
        })
    }

    /// Whether pixel bytes require no conversion (`src == dst`).
    #[inline]
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        self.is_identity
    }

    /// Execute on a borrowed [`PixelView<'a>`], returning zero-copy [`PixelCow::Borrowed`] on identity.
    pub fn execute_view<'a>(
        &self,
        view: PixelView<'a>,
        origin: &ColorOrigin,
    ) -> Result<EncodeReady<'a>, At<ConvertError>> {
        if view.spec() != &self.src {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let (cicp, icc) =
            resolve_output_signaling(&self.src.color, &self.dst.color, &self.profile, origin)?;
        let hdr = if self.dst.color.is_hdr() && same_photometry(&self.src.color, &self.dst.color) {
            view.hdr()
        } else {
            HdrMetadata::default()
        };
        if self.is_identity {
            return Ok(EncodeReady {
                pixels: PixelCow::Borrowed(view),
                cicp,
                icc,
                hdr,
            });
        }
        let converted = self.image_plan.execute(&view)?;
        Ok(EncodeReady {
            pixels: PixelCow::Owned(converted),
            cicp,
            icc,
            hdr,
        })
    }

    /// Execute on an owned [`PixelBuf`], moving `buf` **zero-copy** on identity!
    pub fn execute_buf(&self, buf: PixelBuf) -> Result<EncodeReady<'static>, At<ConvertError>> {
        if buf.spec() != &self.src {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let (cicp, icc) = resolve_output_signaling(
            &self.src.color,
            &self.dst.color,
            &self.profile,
            buf.origin(),
        )?;
        let hdr = if self.dst.color.is_hdr() && same_photometry(&self.src.color, &self.dst.color) {
            buf.hdr()
        } else {
            HdrMetadata::default()
        };
        if self.is_identity {
            return Ok(EncodeReady {
                pixels: PixelCow::Owned(buf),
                cicp,
                icc,
                hdr,
            });
        }
        let converted = self.image_plan.execute(&buf)?;
        Ok(EncodeReady {
            pixels: PixelCow::Owned(converted),
            cicp,
            icc,
            hdr,
        })
    }

    /// Source spec.
    #[inline]
    #[must_use]
    pub fn src_spec(&self) -> &ImageSpec {
        &self.src
    }

    /// Destination spec.
    #[inline]
    #[must_use]
    pub fn dst_spec(&self) -> &ImageSpec {
        &self.dst
    }
}

fn validate_custom_icc_for_dst(
    bytes: &[u8],
    dst_color: &ColorSpace,
) -> Result<(), At<ConvertError>> {
    // ICC profiles have a 128-byte header with data color space signature at bytes 16..20.
    if bytes.len() < 128 {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    }
    let is_narrow = matches!(
        dst_color,
        ColorSpace::Rgb {
            range: zenpixels::SignalRange::Narrow,
            ..
        } | ColorSpace::Gray {
            range: zenpixels::SignalRange::Narrow,
            ..
        }
    );
    if is_narrow {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    }
    let Some(sig) = zenpixels::icc::profile_color_space(bytes) else {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    };
    let matches_model = match dst_color {
        ColorSpace::Rgb { .. } => sig == zenpixels::IccColorSpace::Rgb,
        ColorSpace::Gray { .. } => sig == zenpixels::IccColorSpace::Gray,
        ColorSpace::Cmyk { .. } => sig == zenpixels::IccColorSpace::Cmyk,
        ColorSpace::Icc { profile, .. } => zenpixels::icc::normalized_eq(profile, bytes),
        _ => false,
    };
    if !matches_model {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    }
    // If destination is a structured RGB/Gray space, ensure the custom profile isn't known to contradict it
    // (for structured destinations, callers should target ColorSpace::Icc when converting to an arbitrary ICC).
    match dst_color {
        ColorSpace::Rgb {
            primaries,
            transfer,
            ..
        } => {
            if let Some(id) = zenpixels::icc::IccIdentification::from_icc(bytes)
                && (id.primaries != *primaries || id.transfer != *transfer)
            {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            }
            Ok(())
        }
        ColorSpace::Gray { transfer, .. } => {
            if let Some(id) = zenpixels::icc::IccIdentification::from_icc(bytes)
                && id.transfer != *transfer
            {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            }
            Ok(())
        }
        ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. } => Ok(()),
        _ => Err(err_at(ConvertError::OutputSignalingMismatch)),
    }
}

type ResolvedSignaling = (Option<Cicp>, Option<Cow<'static, [u8]>>);

fn resolve_output_signaling(
    src_color: &ColorSpace,
    dst_color: &ColorSpace,
    profile: &OutputProfile,
    origin: &ColorOrigin,
) -> Result<ResolvedSignaling, At<ConvertError>> {
    match profile {
        OutputProfile::Strip => Ok((None, None)),
        OutputProfile::CustomIcc(bytes) => {
            validate_custom_icc_for_dst(bytes, dst_color)?;
            // Do not synthesize a potentially contradicting CICP alongside an arbitrary custom ICC.
            Ok((None, Some(Cow::Owned(bytes.to_vec()))))
        }
        OutputProfile::CicpOnly => {
            let cicp = dst_color
                .to_cicp()
                .ok_or_else(|| err_at(ConvertError::OutputSignalingUnavailable))?;
            Ok((Some(cicp), None))
        }
        OutputProfile::SameAsOrigin => {
            if !same_color_meaning(src_color, dst_color) {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            }
            let is_narrow = matches!(
                dst_color,
                ColorSpace::Rgb {
                    range: zenpixels::SignalRange::Narrow,
                    ..
                } | ColorSpace::Gray {
                    range: zenpixels::SignalRange::Narrow,
                    ..
                }
            );
            let icc = if is_narrow {
                None
            } else {
                origin.icc.as_ref().map(|b| Cow::Owned(b.to_vec()))
            };
            let cicp = origin.cicp.or_else(|| dst_color.to_cicp());
            Ok((cicp, icc))
        }
        OutputProfile::EmbedIcc => {
            let icc = match dst_color {
                ColorSpace::Icc { profile, .. } => Some(Cow::Owned(profile.to_vec())),
                ColorSpace::Cmyk { icc: Some(b), .. } => Some(Cow::Owned(b.to_vec())),
                ColorSpace::Rgb {
                    primaries,
                    transfer,
                    range: zenpixels::SignalRange::Full,
                    ..
                } => bundled_icc_for(*primaries, *transfer).or_else(|| {
                    dst_color
                        .to_cicp()
                        .and_then(|c| match SynthesizedIcc::for_cicp(c) {
                            SynthesizedIcc::Profile(b) => Some(b),
                            _ => None,
                        })
                }),
                ColorSpace::Gray {
                    range: zenpixels::SignalRange::Full,
                    ..
                } => dst_color
                    .to_cicp()
                    .and_then(|c| match SynthesizedIcc::for_gray_cicp(c) {
                        SynthesizedIcc::Profile(b) => Some(b),
                        _ => None,
                    }),
                _ => None,
            };
            let Some(icc_bytes) = icc else {
                return Err(err_at(ConvertError::OutputSignalingUnavailable));
            };
            let cicp = match dst_color {
                ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. } => None,
                _ => dst_color.to_cicp(),
            };
            Ok((cicp, Some(icc_bytes)))
        }
        OutputProfile::Auto => match dst_color {
            ColorSpace::Icc { profile, .. } => Ok((None, Some(Cow::Owned(profile.to_vec())))),
            ColorSpace::Cmyk { icc, .. } => {
                Ok((None, icc.as_ref().map(|b| Cow::Owned(b.to_vec()))))
            }
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                ..
            } => {
                let cicp = dst_color.to_cicp();
                // ICC RGB profiles describe full-range sample codes; never attach them to narrow-range outputs.
                let icc = if *range != zenpixels::SignalRange::Full
                    || (cicp.is_some()
                        && *primaries == Primaries::Bt709
                        && *transfer == Transfer::Srgb)
                {
                    None
                } else {
                    bundled_icc_for(*primaries, *transfer)
                };
                Ok((cicp, icc))
            }
            _ => Ok((dst_color.to_cicp(), None)),
        },
    }
}

fn bundled_icc_for(primaries: Primaries, transfer: Transfer) -> Option<Cow<'static, [u8]>> {
    crate::icc_profiles::bundled_icc_profile(primaries, transfer).map(Cow::Borrowed)
}
