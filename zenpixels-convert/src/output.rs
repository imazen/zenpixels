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
        let hdr =
            if self.dst.can_represent_hdr() && same_photometry(&self.src.color, &self.dst.color) {
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
        let hdr =
            if self.dst.can_represent_hdr() && same_photometry(&self.src.color, &self.dst.color) {
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

    /// Resolve and validate container color signaling `(cicp, icc)` upfront before
    /// streaming rows into an encoder (avoids materializing a full `EncodeReady` buffer).
    #[allow(clippy::type_complexity)]
    pub fn resolve_signaling(
        &self,
        origin: &ColorOrigin,
    ) -> Result<(Option<Cicp>, Option<Cow<'static, [u8]>>), At<ConvertError>> {
        resolve_output_signaling(&self.src.color, &self.dst.color, &self.profile, origin)
    }

    /// Build a streaming [`RowPlan`](crate::RowPlan) matching this output plan so
    /// row encoders can emit validated headers via [`resolve_signaling`](Self::resolve_signaling)
    /// and convert scanlines without full-frame allocation.
    pub fn build_row_plan(
        &self,
        hdr_peak: Option<zenpixels::Nits>,
    ) -> Result<crate::plan::RowPlan, At<ConvertError>> {
        crate::plan::RowPlan::build(
            self.src.clone(),
            self.dst.clone(),
            self.image_plan.policy(),
            hdr_peak,
        )
    }
}

fn try_clone_bytes(bytes: &[u8]) -> Result<alloc::vec::Vec<u8>, At<ConvertError>> {
    let mut v = alloc::vec::Vec::new();
    v.try_reserve_exact(bytes.len())
        .map_err(|_| err_at(ConvertError::AllocationFailed))?;
    v.extend_from_slice(bytes);
    Ok(v)
}

/// Validate ICC header, declared length, profile class, and tag-table bounds (B10).
fn validate_icc_structure(bytes: &[u8]) -> Result<(), At<ConvertError>> {
    if bytes.len() < 132 || &bytes[36..40] != b"acsp" {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    }
    let declared_size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if declared_size == 0 || declared_size > bytes.len() {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    }
    // Reject DeviceLink (`link`), Abstract (`abst`), and NamedColor (`nmcl`) profiles
    // for image-encoding output (only device/color-space classes describe image color).
    let class = &bytes[12..16];
    if !matches!(class, b"mntr" | b"scnr" | b"prtr" | b"spac") {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    }
    let tag_count = u32::from_be_bytes([bytes[128], bytes[129], bytes[130], bytes[131]]) as usize;
    let table_end = tag_count
        .checked_mul(12)
        .and_then(|n| n.checked_add(132))
        .ok_or_else(|| err_at(ConvertError::OutputSignalingMismatch))?;
    if table_end > declared_size {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    }
    for i in 0..tag_count {
        let base = 132 + i * 12;
        let off = u32::from_be_bytes([
            bytes[base + 4],
            bytes[base + 5],
            bytes[base + 6],
            bytes[base + 7],
        ]) as usize;
        let len = u32::from_be_bytes([
            bytes[base + 8],
            bytes[base + 9],
            bytes[base + 10],
            bytes[base + 11],
        ]) as usize;
        if off < table_end || off.checked_add(len).is_none_or(|end| end > declared_size) {
            return Err(err_at(ConvertError::OutputSignalingMismatch));
        }
    }
    Ok(())
}

fn validate_custom_icc_for_dst(
    bytes: &[u8],
    dst_color: &ColorSpace,
) -> Result<(), At<ConvertError>> {
    validate_icc_structure(bytes)?;
    let is_narrow = dst_color.range() == Some(zenpixels::SignalRange::Narrow);
    if is_narrow {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    }
    let Some(sig) = zenpixels::icc::profile_color_space(bytes) else {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    };
    let matches_model = match dst_color {
        ColorSpace::Rgb { .. } => sig == zenpixels::IccColorSpace::Rgb,
        ColorSpace::Gray { .. } => sig == zenpixels::IccColorSpace::Gray,
        ColorSpace::Cmyk { icc: Some(p), .. } => {
            sig == zenpixels::IccColorSpace::Cmyk && zenpixels::icc::normalized_eq(p, bytes)
        }
        ColorSpace::Icc {
            profile, channels, ..
        } => {
            let expected_sig = match *channels {
                1 => Some(zenpixels::IccColorSpace::Gray),
                3 => Some(zenpixels::IccColorSpace::Rgb),
                4 => Some(zenpixels::IccColorSpace::Cmyk),
                _ => None,
            };
            expected_sig == Some(sig) && zenpixels::icc::normalized_eq(profile, bytes)
        }
        _ => false,
    };
    if !matches_model {
        return Err(err_at(ConvertError::OutputSignalingMismatch));
    }
    // Structured RGB/Gray destinations require positive identification matching
    // destination primaries & transfer — arbitrary unrecognized ICC profiles must
    // target `ColorSpace::Icc` explicitly so pixels are converted rather than mislabeled.
    match dst_color {
        ColorSpace::Rgb {
            primaries,
            transfer,
            ..
        } => {
            let Some(id) = zenpixels::icc::IccIdentification::from_icc(bytes) else {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            };
            if id.primaries != *primaries || id.transfer != *transfer {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            }
            Ok(())
        }
        ColorSpace::Gray { transfer, .. } => {
            let Some(id) = zenpixels::icc::IccIdentification::from_icc(bytes) else {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            };
            if id.transfer != *transfer {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            }
            Ok(())
        }
        ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. } => Ok(()),
        _ => Err(err_at(ConvertError::OutputSignalingMismatch)),
    }
}

/// Semantic CICP equality: allows H.273 transfer aliases that map to the same canonical
/// [`zenpixels::Transfer`] (e.g., BT.709 / BT.601 / BT.2020 10-bit/12-bit codes 1, 6, 14, 15).
fn cicp_semantically_eq(a: Cicp, b: Cicp) -> bool {
    if a == b {
        return true;
    }
    a.color_primaries == b.color_primaries
        && a.matrix_coefficients == b.matrix_coefficients
        && a.full_range == b.full_range
        && match (
            zenpixels::Transfer::from_cicp(a.transfer_characteristics),
            zenpixels::Transfer::from_cicp(b.transfer_characteristics),
        ) {
            (Some(ta), Some(tb)) => ta == tb,
            _ => false,
        }
}

/// Check whether `orig_cicp` agrees with `dst_color` (including `ColorSpace::Icc` via embedded
/// `cicp` tag or qualified `IccIdentification` — B12).
fn origin_cicp_agrees(orig_cicp: Cicp, dst_color: &ColorSpace) -> bool {
    if let Some(dst_cicp) = dst_color.to_cicp() {
        return cicp_semantically_eq(dst_cicp, orig_cicp);
    }
    if let ColorSpace::Icc {
        profile,
        channels: 3,
        ..
    } = dst_color
    {
        if let Some(tag_cicp) = Cicp::from_icc(profile) {
            return cicp_semantically_eq(tag_cicp, orig_cicp);
        }
        if let Some(id) = zenpixels::icc::IccIdentification::from_icc(profile)
            && id
                .to_cicp()
                .is_some_and(|c| cicp_semantically_eq(c, orig_cicp))
        {
            return true;
        }
    }
    false
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
            Ok((None, Some(Cow::Owned(try_clone_bytes(bytes)?))))
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
            // Untagged / assumed origin carries no tags — preserve `(None, None)` verbatim
            // rather than inventing a CICP tuple (B12).
            if origin.icc.is_none() && origin.cicp.is_none() {
                return Ok((None, None));
            }
            let is_narrow = dst_color.range() == Some(zenpixels::SignalRange::Narrow);
            // Verify that preserved origin ICC and CICP actually agree with `dst_color`
            // (never leak historical P3/CMYK/YCbCr provenance onto converted sRGB pixels!).
            let icc = if let Some(orig_icc) = origin.icc.as_ref() {
                if is_narrow {
                    return Err(err_at(ConvertError::OutputSignalingMismatch));
                }
                validate_custom_icc_for_dst(orig_icc, dst_color)?;
                Some(Cow::Owned(try_clone_bytes(orig_icc)?))
            } else {
                None
            };
            let cicp = if let Some(orig_cicp) = origin.cicp {
                if !origin_cicp_agrees(orig_cicp, dst_color) {
                    return Err(err_at(ConvertError::OutputSignalingMismatch));
                }
                Some(orig_cicp)
            } else {
                None
            };
            Ok((cicp, icc))
        }
        OutputProfile::EmbedIcc => {
            let icc = match dst_color {
                ColorSpace::Icc { profile, .. } => {
                    validate_custom_icc_for_dst(profile, dst_color)?;
                    Some(Cow::Owned(try_clone_bytes(profile)?))
                }
                ColorSpace::Cmyk { icc: Some(b), .. } => {
                    validate_custom_icc_for_dst(b, dst_color)?;
                    Some(Cow::Owned(try_clone_bytes(b)?))
                }
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
                    transfer,
                    range: zenpixels::SignalRange::Full,
                    ..
                } => transfer.to_cicp().and_then(|tc| {
                    let gray_cicp = Cicp::new(1, tc, 0, true);
                    match SynthesizedIcc::for_gray_cicp(gray_cicp) {
                        SynthesizedIcc::Profile(b) => Some(b),
                        _ => None,
                    }
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
            ColorSpace::Icc { profile, .. } => {
                validate_custom_icc_for_dst(profile, dst_color)?;
                Ok((None, Some(Cow::Owned(try_clone_bytes(profile)?))))
            }
            ColorSpace::Cmyk { icc: Some(b), .. } => {
                validate_custom_icc_for_dst(b, dst_color)?;
                Ok((None, Some(Cow::Owned(try_clone_bytes(b)?))))
            }
            ColorSpace::Cmyk { icc: None, .. } => Ok((None, None)),
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                ..
            } => {
                let cicp = dst_color.to_cicp();
                // ICC RGB profiles describe full-range sample codes; never attach them to narrow-range outputs.
                let is_default_srgb = *primaries == Primaries::Bt709
                    && *transfer == Transfer::Srgb
                    && *range == zenpixels::SignalRange::Full;
                let icc = if *range != zenpixels::SignalRange::Full || is_default_srgb {
                    None
                } else {
                    bundled_icc_for(*primaries, *transfer).or_else(|| {
                        cicp.and_then(|c| match SynthesizedIcc::for_cicp(c) {
                            SynthesizedIcc::Profile(b) => Some(b),
                            _ => None,
                        })
                    })
                };
                if cicp.is_none() && icc.is_none() && !is_default_srgb {
                    return Err(err_at(ConvertError::OutputSignalingUnavailable));
                }
                Ok((cicp, icc))
            }
            ColorSpace::Gray {
                transfer, range, ..
            } => {
                // Default full-range sRGB Gray needs no ICC tag (universally assumed default).
                if *transfer == Transfer::Srgb && *range == zenpixels::SignalRange::Full {
                    return Ok((None, None));
                }
                if *range != zenpixels::SignalRange::Full {
                    return Err(err_at(ConvertError::OutputSignalingUnavailable));
                }
                let icc = transfer.to_cicp().and_then(|tc| {
                    match SynthesizedIcc::for_gray_cicp(Cicp::new(1, tc, 0, true)) {
                        SynthesizedIcc::Profile(b) => Some(b),
                        _ => None,
                    }
                });
                let Some(icc_bytes) = icc else {
                    return Err(err_at(ConvertError::OutputSignalingUnavailable));
                };
                Ok((None, Some(icc_bytes)))
            }
            _ => match dst_color.to_cicp() {
                Some(c) => Ok((Some(c), None)),
                None => Err(err_at(ConvertError::OutputSignalingUnavailable)),
            },
        },
    }
}

fn bundled_icc_for(primaries: Primaries, transfer: Transfer) -> Option<Cow<'static, [u8]>> {
    crate::icc_profiles::bundled_icc_profile(primaries, transfer).map(Cow::Borrowed)
}
