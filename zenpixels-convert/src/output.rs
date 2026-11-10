//! Atomic output preparation for encoders.
//!
//! [`EncodeReady::finalize`] converts pixel data and generates matching metadata
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
//! [`EncodeReady::finalize`] prevents this by producing the pixels and metadata
//! together. The [`EncodeReady`] struct bundles both, and the only way to
//! create one is through this function. If the conversion fails, neither
//! pixels nor metadata are produced.
//!
//! Use [`EncodeReady::finalize`] for current profile-aware conversion.
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

use crate::error::ConvertError;
use crate::{
    Cicp, ColorOrigin, ColorPrimaries, PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice,
    TransferFunction,
};
use whereat::{At, ResultAtExt};

mod plan;
pub use plan::OutputPlan;

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
    /// requires normalized byte equality after filtering bundled fingerprints,
    /// never approximate color-space matches or a fingerprint match alone.
    /// Header CMM, timestamp, platform, device, creator and ID differences are
    /// ignored. Tags, rendering intent and color transforms are not ignored.
    /// This is explicit metadata normalization, not ICC sanitization or a CMS
    /// conversion. Unknown profiles and non-ICC targets remain unchanged.
    ///
    /// Cost: one fingerprint pass plus comparison with a matching candidate;
    /// a recognized profile allocates
    /// a small canonical `Arc`. No pixel access, scan, or image allocation.
    #[must_use]
    pub fn normalize_known_icc(self) -> Self {
        use crate::icc_profiles::{ADOBE_RGB, DISPLAY_P3_V2, DISPLAY_P3_V4, REC2020_V4};
        use zenpixels::icc::{normalized_eq, normalized_hash};
        const KNOWN: &[((u64, u32), &[u8])] = &[
            (normalized_hash(DISPLAY_P3_V4), DISPLAY_P3_V4),
            (normalized_hash(DISPLAY_P3_V2), DISPLAY_P3_V2),
            (normalized_hash(ADOBE_RGB), ADOBE_RGB),
            (normalized_hash(REC2020_V4), REC2020_V4),
        ];
        if let Self::Icc(ref profile) = self {
            let hash = normalized_hash(profile);
            if let Some((_, canonical)) = KNOWN
                .iter()
                .find(|(key, bytes)| *key == hash && normalized_eq(profile, bytes))
            {
                return Self::Icc(Arc::from(*canonical));
            }
        }
        self
    }
}

/// Metadata that the encoder should embed alongside the pixel data.
///
/// Generated atomically by [`EncodeReady::finalize`] to guarantee that
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
pub struct OutputMetadata {
    /// ICC profile bytes to embed, if any.
    pub icc: Option<Arc<[u8]>>,
    /// CICP code points to embed, if any.
    pub cicp: Option<Cicp>,
}

/// Pixel data bundled with matching metadata, ready for encoding.
///
/// The only way to create an `EncodeReady` is through [`EncodeReady::finalize`],
/// which guarantees that the pixels and metadata are consistent.
///
/// Use [`into_parts()`](Self::into_parts) to destructure if needed, but
/// the default path keeps them coupled.
///
/// The pixels are always owned (one full output buffer per call, even when
/// no conversion was needed); see the *Allocation* section on
/// [`EncodeReady::finalize`].
#[non_exhaustive]
pub struct EncodeReady {
    pixels: PixelBuffer,
    metadata: OutputMetadata,
}

impl EncodeReady {
    /// Couple converted pixels with the metadata that describes them, atomically.
    ///
    /// Offers a conversion to `cms` first, then to the built-in named-profile
    /// paths. Streaming callers use [`OutputPlan`]. Always allocates one output
    /// buffer, including when nothing needs converting.
    ///
    /// # Errors
    ///
    /// When no plan reaches `pixel_format` for `target`, or allocation fails.
    #[track_caller]
    pub fn finalize(
        buffer: &PixelBuffer,
        origin: &ColorOrigin,
        target: OutputProfile,
        pixel_format: PixelFormat,
        cms: Option<&dyn crate::cms::PluggableCms>,
    ) -> Result<Self, At<ConvertError>> {
        finalize_for_output_with(buffer, origin, target, pixel_format, cms)
    }

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

    /// Finalize an owned pixel buffer, moving its allocation into the result
    /// when the plan requires no pixel transform.
    ///
    /// Behaves like [`Self::finalize`] except `buffer` is consumed. When the
    /// prepared [`OutputPlan`] proves the output is sample-identical — the
    /// [`OutputPlan::borrow_rows`] branch, where only descriptor fields such
    /// as primaries/transfer may be completed from authoritative signaling —
    /// `buffer` is re-tagged and moved into the [`EncodeReady`] with no pixel
    /// allocation and no copy. It keeps its own stride in that case.
    ///
    /// When a transform is required this allocates exactly one output
    /// buffer, exactly as [`Self::finalize`].
    ///
    /// # Errors
    ///
    /// As for [`Self::finalize`].
    #[track_caller]
    pub fn finalize_owned(
        buffer: PixelBuffer,
        origin: &ColorOrigin,
        target: OutputProfile,
        pixel_format: PixelFormat,
        cms: Option<&dyn crate::cms::PluggableCms>,
    ) -> Result<Self, At<ConvertError>> {
        finalize_owned_for_output_with(buffer, origin, target, pixel_format, cms)
    }
}

/// Finalize a pixel buffer for output using the [`PluggableCms`](crate::cms::PluggableCms) dispatch chain.
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
pub(crate) fn finalize_for_output_with(
    buffer: &PixelBuffer,
    origin: &ColorOrigin,
    target: OutputProfile,
    pixel_format: PixelFormat,
    cms: Option<&dyn crate::cms::PluggableCms>,
) -> Result<EncodeReady, At<ConvertError>> {
    let options = crate::policy::ConvertOptions::permissive()
        .with_alpha_policy(crate::AlphaPolicy::DiscardUnchecked);
    let mut plan = OutputPlan::new(
        buffer.descriptor(),
        buffer.color_context().cloned(),
        origin,
        target,
        pixel_format,
        &options,
        cms,
    )?;
    let mut pixels =
        PixelBuffer::try_new(buffer.width(), buffer.height(), plan.output_descriptor())
            .map_err_at(ConvertError::from)?;
    plan.convert_rows(buffer.as_slice(), pixels.as_slice_mut())?;
    Ok(EncodeReady {
        pixels: pixels.with_color_context(plan.output_color_context().clone()),
        metadata: plan.into_metadata(),
    })
}

/// By-value sibling of [`finalize_for_output_with`] backing
/// [`EncodeReady::finalize_owned`].
///
/// When the plan proves no pixel transform is required, the consumed
/// buffer's allocation is moved into the result after a descriptor/context
/// retag — `borrow_rows` semantics, but the whole buffer. Otherwise this
/// allocates and converts exactly like the borrowed form.
pub(crate) fn finalize_owned_for_output_with(
    buffer: PixelBuffer,
    origin: &ColorOrigin,
    target: OutputProfile,
    pixel_format: PixelFormat,
    cms: Option<&dyn crate::cms::PluggableCms>,
) -> Result<EncodeReady, At<ConvertError>> {
    let options = crate::policy::ConvertOptions::permissive()
        .with_alpha_policy(crate::AlphaPolicy::DiscardUnchecked);
    let mut plan = OutputPlan::new(
        buffer.descriptor(),
        buffer.color_context().cloned(),
        origin,
        target,
        pixel_format,
        &options,
        cms,
    )?;
    if !plan.transforms_pixels() {
        return Ok(EncodeReady {
            pixels: buffer
                .with_descriptor(plan.output_descriptor())
                .with_color_context(plan.output_color_context().clone()),
            metadata: plan.into_metadata(),
        });
    }
    let mut pixels =
        PixelBuffer::try_new(buffer.width(), buffer.height(), plan.output_descriptor())
            .map_err_at(ConvertError::from)?;
    plan.convert_rows(buffer.as_slice(), pixels.as_slice_mut())?;
    Ok(EncodeReady {
        pixels: pixels.with_color_context(plan.output_color_context().clone()),
        metadata: plan.into_metadata(),
    })
}

/// Resolve current color once; provenance never overrides working pixels.
#[cfg(feature = "hdr-experimental")]
pub(crate) fn current_profile(
    buffer: &PixelBuffer,
) -> Result<crate::ColorProfileSource<'_>, At<ConvertError>> {
    current_profile_for(buffer.descriptor(), buffer.color_context().map(Arc::as_ref))
}

pub(crate) fn current_profile_for(
    desc: PixelDescriptor,
    current: Option<&crate::ColorContext>,
) -> Result<crate::ColorProfileSource<'_>, At<ConvertError>> {
    if let Some(context) = current {
        // Select the authoritative field at decode time. Keeping both here
        // makes current color ambiguous; ColorOrigin is the roundtrip carrier.
        if context.icc.is_some() && context.cicp.is_some() {
            return Err(whereat::at!(ConvertError::NeedsCms {
                from: desc,
                to: desc
            }));
        }
        if let Some(cicp) = context.cicp {
            // A claim that leaves primaries or transfer unresolved (H.273
            // code 2, unspecified, or a code zenpixels does not model)
            // describes no colour space for those fields. It agrees with a
            // descriptor that says `Unknown` there and matches the rest, and
            // then offers no profile: the pixels convert as their descriptor
            // says (layout, depth), and anything that needs the colour still
            // fails on the Unknown descriptor (`NeedsCms`). A descriptor that
            // claims more than the context is still a mislabel.
            let primaries = ColorPrimaries::from_cicp(cicp.color_primaries);
            let transfer = TransferFunction::from_cicp(cicp.transfer_characteristics);
            if primaries.is_none() || transfer.is_none() {
                let range = if cicp.full_range {
                    crate::SignalRange::Full
                } else {
                    crate::SignalRange::Narrow
                };
                let agrees = primaries.unwrap_or(ColorPrimaries::Unknown) == desc.primaries
                    && transfer.unwrap_or(TransferFunction::Unknown) == desc.transfer()
                    && cicp.matrix_coefficients == 0
                    && range == desc.signal_range;
                if agrees {
                    return Ok(desc.color_profile_source());
                }
            }
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
    origin.color_authority.select(icc, cicp)
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

/// Check if two descriptors match in all conversion-relevant fields.
fn descriptors_match(a: &PixelDescriptor, b: &PixelDescriptor) -> bool {
    a.format == b.format
        && a.transfer == b.transfer
        && a.primaries == b.primaries
        && a.signal_range == b.signal_range
        && a.alpha == b.alpha
}
