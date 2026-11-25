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
    Cicp, ColorOrigin, ColorSpace, HdrMetadata, HlgMode, ImageSpec, PeakSource, PixelBuf, PixelCow,
    PixelFormat, PixelView, Primaries, Transfer,
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
    same_spec: bool,
    src_opaque_slot: bool,
    policy_all_opaque: bool,
    is_identity: bool,
}

impl OutputPlan {
    /// Build an encode-ready output plan.
    ///
    /// Accepts either borrowed `&ImageSpec` or owned `ImageSpec` at the call site and borrows
    /// `&ConvertPolicy` (retaining internal clones inside the built plan).
    ///
    /// Validates pixel conversion feasibility AND container signaling compatibility
    /// upfront (for non-origin-dependent profiles) so invalid output configurations
    /// fail before processing any pixels. Note that signaling rules inspect the
    /// **destination** spec (`dst`) — for example, `OutputProfile::CicpOnly` and
    /// `OutputProfile::Auto` refuse `Transfer::Hlg(HlgMode::Display(_))` when `dst`
    /// targets display-referred HLG (converting *from* display HLG to sRGB or PQ
    /// succeeds because the output signaling is sRGB or PQ).
    #[inline]
    pub fn build(
        src: impl AsRef<ImageSpec>,
        dst: impl AsRef<ImageSpec>,
        policy: &ConvertPolicy,
        peak: PeakSource,
        profile: OutputProfile,
    ) -> Result<Self, At<ConvertError>> {
        Self::build_inner(src.as_ref(), dst.as_ref(), policy, peak, profile)
    }

    #[inline(never)]
    fn build_inner(
        src: &ImageSpec,
        dst: &ImageSpec,
        policy: &ConvertPolicy,
        peak: PeakSource,
        profile: OutputProfile,
    ) -> Result<Self, At<ConvertError>> {
        let same_spec = src == dst;
        let src_opaque_slot =
            src.color.alpha() == zenpixels::AlphaMode::Opaque && src.layout.order.has_alpha_slot();
        let policy_all_opaque = policy.source_facts.all_opaque;
        let is_identity = same_spec && (!src_opaque_slot || policy_all_opaque);
        let image_plan = ImagePlan::build(src, dst, policy, peak)?;
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
            src: src.clone(),
            dst: dst.clone(),
            profile,
            image_plan,
            same_spec,
            src_opaque_slot,
            policy_all_opaque,
            is_identity,
        })
    }

    /// Whether pixel bytes require no conversion under the plan's policy (`src == dst`,
    /// excluding unverified `AlphaMode::Opaque` buffers that need alpha canonicalization).
    ///
    /// Note: even when `is_identity()` is `false` due to an unverified `Opaque` alpha slot,
    /// [`execute_view`](Self::execute_view) and [`execute_buf`](Self::execute_buf) still take
    /// the zero-copy path when the runtime input carries `facts().all_opaque == true`.
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
        let runtime_opaque = self.policy_all_opaque || view.facts().all_opaque;
        if self.same_spec && (!self.src_opaque_slot || runtime_opaque) {
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
        let runtime_opaque = self.policy_all_opaque || buf.facts().all_opaque;
        if self.same_spec && (!self.src_opaque_slot || runtime_opaque) {
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
        resolved_source_peak: Option<zenpixels::Nits>,
    ) -> Result<crate::plan::RowPlan, At<ConvertError>> {
        crate::plan::RowPlan::build(
            &self.src,
            &self.dst,
            self.image_plan.policy(),
            resolved_source_peak,
        )
    }
}

/// One-shot encoder preparation: negotiate the lowest-loss, lowest-effort format from
/// `supported` under `policy`, convert (or borrow **zero-copy** on identity), and resolve
/// guaranteed-matching container color signaling (`cicp`, `icc`, `hdr`).
///
/// Thin wrapper combining [`Negotiate::select`](crate::Negotiate::select),
/// [`OutputPlan::build`], and [`OutputPlan::execute_view`], automatically merging
/// `src.facts()` into `policy.source_facts` before negotiation. Returns
/// [`ConvertError::UnsupportedRoute`] if no candidate in `supported` satisfies `policy`.
pub fn adapt_for_encode<'a>(
    src: PixelView<'a>,
    origin: &ColorOrigin,
    supported: &[ImageSpec],
    policy: &ConvertPolicy,
    peak: PeakSource,
    profile: OutputProfile,
) -> Result<(crate::negotiate::NegotiatedFormat, EncodeReady<'a>), At<ConvertError>> {
    let mut eff_policy = policy.clone();
    eff_policy.source_facts = crate::plan::merge_facts(policy.source_facts, src.facts());
    let negotiated = crate::negotiate::Negotiate::select(src.spec(), supported, &eff_policy)
        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
    let plan = OutputPlan::build(src.spec(), &negotiated.spec, &eff_policy, peak, profile)?;
    let ready = plan.execute_view(src, origin)?;
    Ok((negotiated, ready))
}

/// One-shot encoder preparation selecting from a `Copy`/`const` [`PixelFormat`] capability table.
pub fn adapt_for_encode_format<'a>(
    src: PixelView<'a>,
    origin: &ColorOrigin,
    supported: &[PixelFormat],
    policy: &ConvertPolicy,
    peak: PeakSource,
    profile: OutputProfile,
) -> Result<(crate::negotiate::NegotiatedFormat, EncodeReady<'a>), At<ConvertError>> {
    let mut eff_policy = policy.clone();
    eff_policy.source_facts = crate::plan::merge_facts(policy.source_facts, src.facts());
    let negotiated = crate::negotiate::Negotiate::select_format(src.spec(), supported, &eff_policy)
        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
    let plan = OutputPlan::build(src.spec(), &negotiated.spec, &eff_policy, peak, profile)?;
    let ready = plan.execute_view(src, origin)?;
    Ok((negotiated, ready))
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
    zenpixels::validate_icc_structure(bytes)
        .map_err(|_| err_at(ConvertError::OutputSignalingMismatch))
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
    // Structured RGB/Gray destinations require positive matrix+TRC identification matching
    // destination primaries & transfer — arbitrary or `MetadataOnly` ICC profiles (such as
    // non-D65 / Illuminant C historical profiles) must target `ColorSpace::Icc` explicitly
    // so pixels are converted via CMS rather than mislabeled.
    match dst_color {
        ColorSpace::Rgb {
            primaries,
            transfer,
            ..
        } => {
            let Some(id) = zenpixels::icc::IccIdentification::from_icc(bytes) else {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            };
            if id.valid_use != zenpixels::icc::IdentificationUse::MatrixTrcSubstitution
                || !id.primaries.same_chromaticity(*primaries)
                || !id.transfer.same_curve(*transfer)
            {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            }
            Ok(())
        }
        ColorSpace::Gray { transfer, .. } => {
            let Some(id) = zenpixels::icc::IccIdentification::from_icc(bytes) else {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            };
            if id.valid_use != zenpixels::icc::IdentificationUse::MatrixTrcSubstitution
                || !id.transfer.same_curve(*transfer)
            {
                return Err(err_at(ConvertError::OutputSignalingMismatch));
            }
            Ok(())
        }
        ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. } => Ok(()),
        _ => Err(err_at(ConvertError::OutputSignalingMismatch)),
    }
}

/// Semantic CICP equality: allows H.273 chromaticity/transfer aliases that evaluate the exact
/// same mathematical primaries/curve (e.g. BT.709 / BT.601 / BT.2020 10-bit/12-bit transfer codes
/// 1, 6, 14, 15, or BT.601 / SMPTE 240M primaries 6 and 7).
fn cicp_semantically_eq(a: Cicp, b: Cicp) -> bool {
    if a == b {
        return true;
    }
    let prim_eq = a.color_primaries == b.color_primaries
        || match (
            Primaries::from_cicp(a.color_primaries),
            Primaries::from_cicp(b.color_primaries),
        ) {
            (Some(pa), Some(pb)) => pa.same_chromaticity(pb),
            _ => false,
        };
    let tf_eq = a.transfer_characteristics == b.transfer_characteristics
        || match (
            Transfer::from_cicp(a.transfer_characteristics),
            Transfer::from_cicp(b.transfer_characteristics),
        ) {
            (Some(ta), Some(tb)) => ta.same_curve(tb),
            _ => false,
        };
    prim_eq
        && tf_eq
        && a.matrix_coefficients == b.matrix_coefficients
        && a.full_range == b.full_range
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
            if matches!(
                dst_color,
                ColorSpace::Rgb {
                    transfer: Transfer::Hlg(HlgMode::Display(_)),
                    ..
                }
            ) {
                return Err(err_at(ConvertError::OutputSignalingUnavailable));
            }
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
                // CICP TC=18 carries scene-referred HLG syntax only; display-rendered HLG
                // parameters (`peak`, `black_nits`, `system_gamma`) cannot be represented in CICP
                // or ICC without explicit conversion.
                if matches!(transfer, Transfer::Hlg(HlgMode::Display(_))) {
                    return Err(err_at(ConvertError::OutputSignalingUnavailable));
                }
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
