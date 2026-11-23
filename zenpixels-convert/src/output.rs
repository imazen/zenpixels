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

use crate::error::ConvertError;
use crate::plan::ImagePlan;
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
    pub fn build(
        src: ImageSpec,
        dst: ImageSpec,
        policy: ConvertPolicy,
        peak: PeakSource,
        profile: OutputProfile,
    ) -> Result<Self, At<ConvertError>> {
        let is_identity = src == dst;
        let image_plan = ImagePlan::build(src.clone(), dst.clone(), policy, peak)?;
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
        let hdr = if self.dst.color.is_hdr() {
            view.hdr()
        } else {
            HdrMetadata::default()
        };
        let (cicp, icc) = resolve_output_signaling(&self.dst.color, &self.profile, origin);
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
        let hdr = if self.dst.color.is_hdr() {
            buf.hdr()
        } else {
            HdrMetadata::default()
        };
        let (cicp, icc) = resolve_output_signaling(&self.dst.color, &self.profile, buf.origin());
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

fn resolve_output_signaling(
    dst_color: &ColorSpace,
    profile: &OutputProfile,
    origin: &ColorOrigin,
) -> (Option<Cicp>, Option<Cow<'static, [u8]>>) {
    match profile {
        OutputProfile::Strip => (None, None),
        OutputProfile::CustomIcc(bytes) => (dst_color.to_cicp(), Some(Cow::Owned(bytes.to_vec()))),
        OutputProfile::CicpOnly => (dst_color.to_cicp(), None),
        OutputProfile::SameAsOrigin => {
            let icc = origin.icc.as_ref().map(|b| Cow::Owned(b.to_vec()));
            let cicp = origin.cicp.or_else(|| dst_color.to_cicp());
            (cicp, icc)
        }
        OutputProfile::EmbedIcc => {
            let cicp = dst_color.to_cicp();
            let icc = match dst_color {
                ColorSpace::Icc { profile, .. } => Some(Cow::Owned(profile.to_vec())),
                ColorSpace::Cmyk { icc: Some(b), .. } => Some(Cow::Owned(b.to_vec())),
                ColorSpace::Rgb {
                    primaries,
                    transfer,
                    ..
                } => bundled_icc_for(*primaries, *transfer),
                _ => None,
            };
            (cicp, icc)
        }
        OutputProfile::Auto => match dst_color {
            ColorSpace::Icc { profile, .. } => (None, Some(Cow::Owned(profile.to_vec()))),
            ColorSpace::Cmyk { icc, .. } => (None, icc.as_ref().map(|b| Cow::Owned(b.to_vec()))),
            ColorSpace::Rgb {
                primaries,
                transfer,
                ..
            } => {
                let cicp = dst_color.to_cicp();
                let icc = if cicp.is_some()
                    && *primaries == Primaries::Bt709
                    && *transfer == Transfer::Srgb
                {
                    None
                } else {
                    bundled_icc_for(*primaries, *transfer)
                };
                (cicp, icc)
            }
            _ => (dst_color.to_cicp(), None),
        },
    }
}

fn bundled_icc_for(primaries: Primaries, transfer: Transfer) -> Option<Cow<'static, [u8]>> {
    crate::icc_profiles::bundled_icc_profile(primaries, transfer).map(Cow::Borrowed)
}
