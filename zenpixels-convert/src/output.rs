//! Atomic output preparation for encoders.
//!
//! [`finalize_for_output_with`] converts pixel data and generates matching metadata
//! in a single atomic operation, preventing the most common color management
//! bug: pixel values that don't match the embedded color metadata.
//!
//! # Why atomicity matters
//!
//! The most common color management bug looks like this:
//!
//! ```rust,ignore
//! // BUG: pixels and metadata can diverge
//! let pixels = convert_to_p3(&buffer);
//! let metadata = OutputMetadata { icc: Some(srgb_icc), .. };
//! // ^^^ pixels are Display P3 but metadata says sRGB — wrong!
//! ```
//!
//! [`finalize_for_output_with`] prevents this by producing the pixels and metadata
//! together. The [`EncodeReady`] struct bundles both, and the only way to
//! create one is through this function. If the conversion fails, neither
//! pixels nor metadata are produced.
//!
//! Use [`finalize_for_output_with`] for current profile-aware conversion.
//! `ColorContext` describes current pixels; `ColorOrigin` is provenance.
//! `SameAsOrigin` requests an actual conversion back to the origin profile.
//! Attached ICC bytes are passed to the CMS even for a named destination.
//! Contradictory current CICP/descriptor signaling is refused.
//!
//! The result always owns independent pixels. This API allocates an output
//! image, including for identity; use the cow adaptation APIs for borrowing
//! when separate output metadata is not needed. No opacity prepass is implicit:
//! selecting an alpha-free output format discards alpha unconditionally.
//! Backend failures return an error, never an `EncodeReady` with mismatched tags.

use alloc::sync::Arc;

#[allow(deprecated)]
use crate::cms::ColorManagement;
use crate::error::ConvertError;
#[allow(deprecated)]
use crate::hdr::HdrMetadata;
use crate::{
    Cicp, ColorAuthority, ColorOrigin, ColorPrimaries, PixelBuffer, PixelDescriptor, PixelFormat,
    PixelSlice, TransferFunction,
};
use whereat::{At, ResultAtExt};

/// Target output color profile.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum OutputProfile {
    /// Re-encode in the original color space. For decoded RGB, CICP matrix
    /// and range are emitted as identity/full; original YUV packing is provenance.
    SameAsOrigin,
    /// Use a well-known CICP-described profile.
    Named(Cicp),
    /// Use specific ICC profile bytes.
    Icc(Arc<[u8]>),
}

impl OutputProfile {
    /// Replace an exactly recognized ICC with its bundled canonical profile.
    ///
    /// Uses the same normalized hash as `zenpixels::icc` identification, but
    /// only exact bundled fingerprints, never approximate color-space matches.
    /// Header CMM, timestamp, platform, device, creator and ID differences are
    /// ignored. Tags, rendering intent and color transforms are not ignored.
    /// This is explicit metadata normalization, not ICC sanitization or a CMS
    /// conversion. Unknown profiles and non-ICC targets remain unchanged.
    ///
    /// Cost: one pass over the input profile; a recognized profile allocates
    /// a small canonical `Arc`. No pixel access, scan, or image allocation.
    #[must_use]
    pub fn normalize_known_icc(self) -> Self {
        use crate::icc_profiles::{ADOBE_RGB, DISPLAY_P3_V2, DISPLAY_P3_V4, REC2020_V4};
        use zenpixels::icc::normalized_hash;
        const KNOWN: &[(u64, &[u8])] = &[
            (normalized_hash(DISPLAY_P3_V4), DISPLAY_P3_V4),
            (normalized_hash(DISPLAY_P3_V2), DISPLAY_P3_V2),
            (normalized_hash(ADOBE_RGB), ADOBE_RGB),
            (normalized_hash(REC2020_V4), REC2020_V4),
        ];
        if let Self::Icc(ref profile) = self {
            let hash = normalized_hash(profile);
            if let Some((_, canonical)) = KNOWN
                .iter()
                .find(|(key, bytes)| *key == hash && bytes.len() == profile.len())
            {
                return Self::Icc(Arc::from(*canonical));
            }
        }
        self
    }
}

// TODO(0.3.0): Add HdrPolicy enum and ConvertOutputOptions here once
// ConvertError is #[non_exhaustive] and can carry HdrTransferRequiresToneMapping.
// See imazen/zenpixels#10 for the full HDR provenance plan.

/// Metadata that the encoder should embed alongside the pixel data.
///
/// Generated atomically by [`finalize_for_output`] to guarantee that
/// the metadata matches the pixel values.
///
/// # Not the same as `ColorContext` / `ColorOrigin`
///
/// Three carriers touch color and overlap on `icc` / `cicp`, but each answers
/// a different question and lives at a different point in the pipeline:
///
/// - [`ColorOrigin`] — *how the source file described its color* (provenance +
///   which field is authoritative). Immutable, set once at decode, consulted
///   for re-encode decisions.
/// - [`ColorContext`](zenpixels::ColorContext) — *what the working pixels are
///   right now* (the profile needed to interpret their values: `icc` / `cicp`
///   / `diffuse_white`). Rides on the `PixelSlice` via `Arc` and is rewritten
///   as conversions change the pixels. It deliberately carries **no** content
///   light level / mastering display — those don't change how a value is
///   interpreted, only how a display should present it.
/// - `OutputMetadata` (this type) — *the color blocks the encoder writes into
///   the container* (`icc` / `cicp`). It is the lowering target of a codec's
///   color plan (`zencodec::ColorEmitPlan` is itself just `{ cicp, icc }`) and
///   mirrors that shape. Its reason to exist as a distinct type is the
///   [`EncodeReady`] atomicity guarantee: the converted bytes and the embedded
///   color are produced together and cannot drift apart — which a re-used
///   `ColorContext` would not give. HDR content descriptors (content light
///   level, mastering display, `diffuse_white`) are deliberately **not** here:
///   they are not color-profile data and ride the codec-boundary metadata
///   carrier (`zencodec::Metadata`, which already holds all three) instead.
#[derive(Clone, Debug)]
#[non_exhaustive]
// The `hdr` field references the deprecated `HdrMetadata`; suppress the
// definition-site lint here (external uses still see the field/type
// deprecation). The field is never wired (see TODO(0.3.0) below).
#[allow(deprecated)]
pub struct OutputMetadata {
    /// ICC profile bytes to embed, if any.
    pub icc: Option<Arc<[u8]>>,
    /// CICP code points to embed, if any.
    pub cicp: Option<Cicp>,
    /// HDR metadata to embed (content light level, mastering display), if any.
    ///
    /// **Deprecated and never wired** — `finalize_for_output` always sets it to
    /// `None`. The bundled [`crate::hdr::HdrMetadata`] carrier is
    /// being removed at 0.3.0 (it has frozen public fields and bundles
    /// `transfer`, which the prior art keeps on the descriptor).
    ///
    /// **What replaces it: nothing — and that is correct by design, not a
    /// stub.** Removing this leaves `OutputMetadata { icc, cicp }`, which
    /// mirrors the *color* plan a codec lowers here (`zencodec::ColorEmitPlan`
    /// is itself just `{ cicp, icc }`). The HDR content descriptors — content
    /// light level, mastering display, and the `diffuse_white` /
    /// `intensity_target` anchor — are **not** color-profile data: they ride
    /// the codec-boundary metadata carrier instead. `zencodec::Metadata`
    /// already carries all three as sibling fields (the un-bundled shape this
    /// `HdrMetadata` bundle should have been), threaded by its metadata policy,
    /// and the codec embeds them from there. Nothing ever read them off this
    /// field — `HdrMetadata` had zero consumers across `~/work`, and zencodec
    /// routed around it from the start. See `CHANGELOG.md`
    /// "QUEUED BREAKING CHANGES".
    #[deprecated(
        since = "0.2.14",
        note = "unwired bundled HDR carrier; replaced by sibling content_light_level / mastering_display fields when the encoder path that populates them lands (0.3.0)."
    )]
    pub hdr: Option<HdrMetadata>,
}

/// Pixel data bundled with matching metadata, ready for encoding.
///
/// The only way to create an `EncodeReady` is through [`finalize_for_output`],
/// which guarantees that the pixels and metadata are consistent.
///
/// Use [`into_parts()`](Self::into_parts) to destructure if needed, but
/// the default path keeps them coupled.
///
/// The pixels are always owned (one full output buffer per call, even when
/// no conversion was needed); see the *Allocation* section on
/// [`finalize_for_output_with`].
#[non_exhaustive]
pub struct EncodeReady {
    pixels: PixelBuffer,
    metadata: OutputMetadata,
}

impl EncodeReady {
    /// Borrow the pixel data.
    pub fn pixels(&self) -> PixelSlice<'_> {
        self.pixels.as_slice()
    }

    /// Borrow the output metadata.
    pub fn metadata(&self) -> &OutputMetadata {
        &self.metadata
    }

    /// Consume and split into pixel buffer and metadata.
    pub fn into_parts(self) -> (PixelBuffer, OutputMetadata) {
        (self.pixels, self.metadata)
    }
}

/// Atomically convert pixel data and generate matching encoder metadata.
///
/// This function does three things as a single operation:
///
/// 1. Determines the current pixel color state from `PixelDescriptor` +
///    optional ICC profile on `ColorContext`.
/// 2. Converts pixels to the target profile's space. For named profiles,
///    uses hardcoded matrices. For custom ICC profiles, uses the CMS.
/// 3. Bundles the converted pixels with matching metadata ([`EncodeReady`]).
///
/// # Arguments
///
/// - `buffer` — Source pixel data with its current descriptor.
/// - `origin` — How the source file described its color (for `SameAsOrigin`).
/// - `target` — Desired output color profile.
/// - `pixel_format` — Target pixel format for the output.
/// - `cms` — Color management system for ICC profile transforms.
///
/// # Errors
///
/// Returns [`ConvertError`] if:
/// - The target format requires a conversion that isn't supported.
/// - The CMS fails to build a transform for ICC profiles.
/// - Buffer allocation fails.
// TODO(0.3.0): Add HDR→SDR policy gate here once ConvertError has
// HdrTransferRequiresToneMapping. See imazen/zenpixels#10.
#[track_caller]
#[deprecated(
    since = "0.2.8",
    note = "use finalize_for_output_with with a PluggableCms"
)]
#[allow(deprecated)]
pub fn finalize_for_output<C: ColorManagement>(
    buffer: &PixelBuffer,
    origin: &ColorOrigin,
    target: OutputProfile,
    pixel_format: PixelFormat,
    cms: &C,
) -> Result<EncodeReady, At<ConvertError>> {
    let source_desc = buffer.descriptor();
    let target_desc = pixel_format.descriptor();

    // Determine output metadata based on target profile.
    let (metadata, needs_cms_transform) = match &target {
        OutputProfile::SameAsOrigin => {
            let metadata = OutputMetadata {
                icc: origin.icc.clone(),
                cicp: origin.cicp,
                hdr: None,
            };
            // SameAsOrigin = keep the source color space. No CMS conversion.
            // Pixel format changes (depth, layout) are handled by RowConverter.
            (metadata, false)
        }
        OutputProfile::Named(cicp) => {
            let metadata = OutputMetadata {
                icc: None,
                cicp: Some(*cicp),
                hdr: None,
            };
            (metadata, false)
        }
        OutputProfile::Icc(icc) => {
            let metadata = OutputMetadata {
                icc: Some(icc.clone()),
                cicp: None,
                hdr: None,
            };
            (metadata, true)
        }
    };

    // Apply CMS transform if needed, respecting color_authority.
    if needs_cms_transform
        && let Some(transform) =
            build_cms_transform(origin, &metadata, &source_desc, pixel_format, cms)?
    {
        let src_slice = buffer.as_slice();
        let mut out = PixelBuffer::try_new(buffer.width(), buffer.height(), target_desc)
            .map_err_at(ConvertError::from)?;

        {
            let mut dst_slice = out.as_slice_mut();
            for y in 0..buffer.height() {
                let src_row = src_slice.row(y);
                let dst_row = dst_slice.row_mut(y);
                transform.transform_row(src_row, dst_row, buffer.width());
            }
        }

        return Ok(EncodeReady {
            pixels: out,
            metadata,
        });
    }

    // Named profile conversion: use hardcoded matrices via RowConverter.
    let target_desc_full = target_desc
        .with_transfer(resolve_transfer(&target, &source_desc))
        .with_primaries(resolve_primaries(&target, &source_desc));

    if source_desc.layout_compatible(target_desc_full)
        && descriptors_match(&source_desc, &target_desc_full)
    {
        // No conversion needed — copy the buffer.
        let src_slice = buffer.as_slice();
        // Even identity can be a 100 MB allocation: keep it fallible and
        // copy only active row bytes, without a temporary packed image.
        let mut out = PixelBuffer::try_new(buffer.width(), buffer.height(), target_desc_full)
            .map_err_at(ConvertError::from)?;
        {
            let mut dst = out.as_slice_mut();
            for y in 0..buffer.height() {
                dst.row_mut(y).copy_from_slice(src_slice.row(y));
            }
        }
        return Ok(EncodeReady {
            pixels: out,
            metadata,
        });
    }

    // Use RowConverter for format conversion.
    let mut converter = crate::RowConverter::new(source_desc, target_desc_full).at()?;
    let src_slice = buffer.as_slice();
    let mut out = PixelBuffer::try_new(buffer.width(), buffer.height(), target_desc_full)
        .map_err_at(ConvertError::from)?;

    {
        let mut dst_slice = out.as_slice_mut();
        for y in 0..buffer.height() {
            let src_row = src_slice.row(y);
            let dst_row = dst_slice.row_mut(y);
            converter.try_convert_row(src_row, dst_row, buffer.width())?;
        }
    }

    Ok(EncodeReady {
        pixels: out,
        metadata,
    })
}

/// Build a CMS transform from the origin's color metadata.
///
/// Respects [`ColorAuthority`]: when `Icc`, builds from ICC bytes; when `Cicp`,
/// builds from CICP codes via the CMS's `build_transform_from_cicp`. Falls
/// back to the non-authoritative field when the authoritative one is missing.
///
/// Returns `Ok(None)` when no source profile can be determined.
#[allow(deprecated)]
fn build_cms_transform<C: ColorManagement>(
    origin: &ColorOrigin,
    metadata: &OutputMetadata,
    source_desc: &PixelDescriptor,
    dst_format: PixelFormat,
    cms: &C,
) -> Result<Option<alloc::boxed::Box<dyn crate::cms::RowTransform>>, At<ConvertError>> {
    let src_format = source_desc.format;
    let Some(ref dst_icc) = metadata.icc else {
        return Ok(None);
    };

    // Try ICC path first (or second, depending on authority).
    let try_icc = |src_icc: &[u8]| -> Result<Option<_>, At<ConvertError>> {
        let transform = cms
            .build_transform_for_format(src_icc, dst_icc, src_format, dst_format)
            .map_err(|e| whereat::at!(ConvertError::CmsError(alloc::format!("{e:?}"))))?;
        Ok(Some(transform))
    };

    match origin.color_authority {
        ColorAuthority::Icc => {
            if let Some(ref src_icc) = origin.icc {
                return try_icc(src_icc);
            }
            // Fallback: ICC authority but no ICC bytes — can't build transform.
            Ok(None)
        }
        ColorAuthority::Cicp => {
            // CICP authority — but build_transform_from_cicp needs ICC bytes
            // on the dst side, so we still need ICC. Try src ICC if available.
            if let Some(ref src_icc) = origin.icc {
                return try_icc(src_icc);
            }
            Ok(None)
        }
    }
}

// TODO(0.3.0): restore origin_has_hdr_transfer / target_has_hdr_transfer
// helpers here for the HDR→SDR policy gate.

/// Finalize a pixel buffer for output using the [`PluggableCms`](crate::cms::PluggableCms) dispatch chain.
///
/// Modern replacement for [`finalize_for_output`].
///
/// When a CMS plugin is supplied, it is offered the conversion first; on
/// decline the built-in `ZenCmsLite` dispatcher handles named-profile
/// matlut fast paths. When `cms` is `None`, only `ZenCmsLite` runs.
///
/// Pass `cms = Some(&MoxCms)` (or another `PluggableCms`) for full ICC
/// support; pass `None` for named-profile-only builds that avoid pulling
/// in a full CMS dependency.
///
/// # Allocation
///
/// The returned [`EncodeReady`] always owns its pixels, so this call
/// allocates exactly one output buffer of `width × height` in
/// `pixel_format` — including on the no-conversion path (source and target
/// agree on format, transfer, primaries, and signal range), where the source
/// rows are copied into it unchanged. The signature (`&PixelBuffer` in,
/// owned out) makes that copy structural rather than avoidable; a
/// by-value/borrowing sibling is tracked in imazen/zenpixels#69. No other
/// per-image allocation occurs: the conversion path's scratch is per-row and
/// lives inside [`RowConverter`](crate::RowConverter).
///
/// # Errors
///
/// Returns [`ConvertError`] when no conversion path is available, buffer
/// allocation fails, or a policy gate (alpha/depth/luma) forbids a
/// required operation.
#[track_caller]
#[allow(deprecated)] // sets the unwired, deprecated OutputMetadata::hdr to None
pub fn finalize_for_output_with(
    buffer: &PixelBuffer,
    origin: &ColorOrigin,
    target: OutputProfile,
    pixel_format: PixelFormat,
    cms: Option<&dyn crate::cms::PluggableCms>,
) -> Result<EncodeReady, At<ConvertError>> {
    let source_desc = buffer.descriptor();
    // Origin describes provenance. Only attached context describes current ICC.
    let source_profile = current_profile(buffer)?;
    let target_profile = match &target {
        OutputProfile::SameAsOrigin => {
            origin_profile(origin).unwrap_or_else(|| source_profile.clone())
        }
        OutputProfile::Named(cicp) => crate::ColorProfileSource::Cicp(*cicp),
        OutputProfile::Icc(icc) => crate::ColorProfileSource::Icc(icc),
    };
    let target_desc_full = descriptor_for_profile(pixel_format, &target_profile)?;
    crate::convert::validate_descriptors(source_desc, target_desc_full)?;

    // Determine output metadata based on target profile.
    let metadata = match &target_profile {
        crate::ColorProfileSource::Icc(icc) => OutputMetadata {
            // Emit only the selected authority; OutputMetadata cannot tell a
            // destination codec how to prioritize contradictory ICC/CICP tags.
            icc: Some(match &target {
                OutputProfile::Icc(bytes) => Arc::clone(bytes),
                OutputProfile::SameAsOrigin => origin
                    .icc
                    .clone()
                    .or_else(|| buffer.color_context().and_then(|c| c.icc.clone()))
                    .unwrap_or_else(|| Arc::from(*icc)),
                _ => Arc::from(*icc),
            }),
            cicp: None,
            hdr: None,
        },
        crate::ColorProfileSource::Cicp(cicp) => OutputMetadata {
            icc: None,
            cicp: Some(*cicp),
            hdr: None,
        },
        _ => {
            let (p, t) = target_profile.resolve().ok_or_else(|| {
                whereat::at!(ConvertError::NeedsCms {
                    from: source_desc,
                    to: target_desc_full
                })
            })?;
            if p == ColorPrimaries::Bt709 && t == TransferFunction::Srgb {
                OutputMetadata {
                    icc: None,
                    cicp: None,
                    hdr: None,
                }
            } else if let (Some(p), Some(t)) = (p.to_cicp(), t.to_cicp()) {
                OutputMetadata {
                    icc: None,
                    cicp: Some(Cicp::new(
                        p,
                        t,
                        0,
                        target_desc_full.signal_range == crate::SignalRange::Full,
                    )),
                    hdr: None,
                }
            } else {
                return Err(whereat::at!(ConvertError::NeedsCms {
                    from: source_desc,
                    to: target_desc_full
                }));
            }
        }
    };

    // The current context must describe the returned samples, including identity.
    let mut context = crate::ColorContext::default();
    context.icc = metadata.icc.clone();
    context.cicp = metadata.cicp;
    context.diffuse_white = if source_desc.transfer() == TransferFunction::Pq
        && target_desc_full.transfer() == TransferFunction::Linear
    {
        Some(zenpixels::hdr::DiffuseWhite::new(10_000.0))
    } else {
        buffer.color_context().and_then(|c| c.diffuse_white)
    };
    let context = Arc::new(context);

    // Fast path: no conversion needed.
    if source_desc.layout_compatible(target_desc_full)
        && descriptors_match(&source_desc, &target_desc_full)
        && profiles_match(&source_profile, &target_profile)
    {
        let src_slice = buffer.as_slice();
        // Even identity can be a 100 MB allocation: keep it fallible and
        // copy only active row bytes, without a temporary packed image.
        let mut out = PixelBuffer::try_new(buffer.width(), buffer.height(), target_desc_full)
            .map_err_at(ConvertError::from)?;
        {
            let mut dst = out.as_slice_mut();
            for y in 0..buffer.height() {
                dst.row_mut(y).copy_from_slice(src_slice.row(y));
            }
        }
        return Ok(EncodeReady {
            pixels: out.with_color_context(context),
            metadata,
        });
    }

    let options = crate::policy::ConvertOptions::permissive()
        .with_alpha_policy(crate::AlphaPolicy::DiscardUnchecked);
    let needs_profiles = matches!(source_profile, crate::ColorProfileSource::Icc(_))
        || matches!(target_profile, crate::ColorProfileSource::Icc(_));
    let mut converter = if needs_profiles {
        crate::RowConverter::new_with_sources(
            source_desc,
            target_desc_full,
            source_profile,
            target_profile,
            &options,
            cms,
        )?
    } else {
        crate::RowConverter::new_explicit_with_cms(source_desc, target_desc_full, &options, cms)?
    };
    let src_slice = buffer.as_slice();
    let mut out = PixelBuffer::try_new(buffer.width(), buffer.height(), target_desc_full)
        .map_err_at(ConvertError::from)?;

    {
        let mut dst_slice = out.as_slice_mut();
        for y in 0..buffer.height() {
            let src_row = src_slice.row(y);
            let dst_row = dst_slice.row_mut(y);
            converter.try_convert_row(src_row, dst_row, buffer.width())?;
        }
    }

    Ok(EncodeReady {
        pixels: out.with_color_context(context),
        metadata,
    })
}

/// Resolve current color once; provenance never overrides working pixels.
pub(crate) fn current_profile(
    buffer: &PixelBuffer,
) -> Result<crate::ColorProfileSource<'_>, At<ConvertError>> {
    let desc = buffer.descriptor();
    if let Some(context) = buffer.color_context() {
        // Select the authoritative field at decode time. Keeping both here
        // makes current color ambiguous; ColorOrigin is the roundtrip carrier.
        if context.icc.is_some() && context.cicp.is_some() {
            return Err(whereat::at!(ConvertError::NeedsCms {
                from: desc,
                to: desc
            }));
        }
        if let Some(cicp) = context.cicp {
            let signaled = descriptor_for_profile(
                desc.pixel_format(),
                &crate::ColorProfileSource::Cicp(cicp),
            )?;
            if signaled.primaries != desc.primaries
                || signaled.transfer() != desc.transfer()
                || signaled.signal_range != desc.signal_range
            {
                return Err(whereat::at!(ConvertError::NoPath {
                    from: desc,
                    to: signaled
                }));
            }
        }
        if let Some(profile) = context.as_profile_source() {
            return Ok(profile);
        }
    }
    Ok(desc.color_profile_source())
}

fn origin_profile(origin: &ColorOrigin) -> Option<crate::ColorProfileSource<'_>> {
    let icc = origin.icc.as_deref().map(crate::ColorProfileSource::Icc);
    let cicp = origin.cicp.map(|c| {
        crate::ColorProfileSource::Cicp(Cicp::new(
            c.color_primaries,
            c.transfer_characteristics,
            0,
            true,
        ))
    });
    match origin.color_authority {
        ColorAuthority::Icc => icc.or(cicp),
        ColorAuthority::Cicp => cicp.or(icc),
    }
}

fn descriptor_for_profile(
    format: PixelFormat,
    profile: &crate::ColorProfileSource<'_>,
) -> Result<PixelDescriptor, At<ConvertError>> {
    if let crate::ColorProfileSource::Cicp(cicp) = profile {
        return cicp
            .try_to_descriptor(format)
            .map_err(|error| whereat::at!(ConvertError::CicpDescriptor(error)));
    }
    let (primaries, transfer) = profile
        .resolve()
        .unwrap_or((ColorPrimaries::Unknown, TransferFunction::Unknown));
    Ok(format
        .descriptor()
        .with_primaries(primaries)
        .with_transfer(transfer))
}

fn profiles_match(a: &crate::ColorProfileSource<'_>, b: &crate::ColorProfileSource<'_>) -> bool {
    match (a, b) {
        (crate::ColorProfileSource::Icc(a), crate::ColorProfileSource::Icc(b)) => a == b,
        (crate::ColorProfileSource::Icc(_), _) | (_, crate::ColorProfileSource::Icc(_)) => false,
        _ => a.resolve().is_some() && a.resolve() == b.resolve(),
    }
}

/// Resolve the target transfer function.
fn resolve_transfer(target: &OutputProfile, source: &PixelDescriptor) -> TransferFunction {
    match target {
        OutputProfile::SameAsOrigin => source.transfer(),
        OutputProfile::Named(cicp) => TransferFunction::from_cicp(cicp.transfer_characteristics)
            .unwrap_or(TransferFunction::Unknown),
        OutputProfile::Icc(_) => TransferFunction::Unknown,
    }
}

/// Resolve the target color primaries.
fn resolve_primaries(target: &OutputProfile, source: &PixelDescriptor) -> ColorPrimaries {
    match target {
        OutputProfile::SameAsOrigin => source.primaries,
        OutputProfile::Named(cicp) => {
            ColorPrimaries::from_cicp(cicp.color_primaries).unwrap_or(ColorPrimaries::Unknown)
        }
        OutputProfile::Icc(_) => ColorPrimaries::Unknown,
    }
}

/// Check if two descriptors match in all conversion-relevant fields.
fn descriptors_match(a: &PixelDescriptor, b: &PixelDescriptor) -> bool {
    a.format == b.format
        && a.transfer == b.transfer
        && a.primaries == b.primaries
        && a.signal_range == b.signal_range
        && a.alpha == b.alpha
}
