//! Compile pixel conversion and matching encoder signaling.
//!
//! [`OutputPlan::build`] selects the executable conversion, output descriptor
//! and metadata. `ColorContext` describes current samples. `ColorOrigin`
//! describes their origin and is consulted for [`OutputProfile::SameAsOrigin`].
//! Attached ICC bytes retain their authority when converting to a named target.
//!
//! [`EncodeReady::from_owned`] consumes the input and reuses its storage for
//! identity output. [`EncodeReady::copy_from`] creates independently owned output
//! from a borrowed view. `OutputPlan` also supports borrowing unchanged rows
//! and writing into caller-provided storage. Alpha removal follows the
//! supplied policy.

use alloc::sync::Arc;

use crate::error::ConvertError;
use crate::{
    Cicp, ColorOrigin, ColorPrimaries, PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice,
    TransferFunction,
};
use whereat::{At, ResultAtExt};

mod plan;
pub use plan::{OutputPlan, OutputTarget};

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
    /// Use a non-ICC parametric color description (e.g. JPEG XL `ColorEncoding`, PNG `gAMA`+`cHRM`).
    Parametric(crate::ParametricColor),
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
/// Generated atomically by [`EncodeReady::from_owned`] to guarantee that
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
    /// Non-ICC parametric color description to embed, if any.
    pub parametric: Option<crate::ParametricColor>,
}

/// Pixel data bundled with matching metadata, ready for encoding.
///
/// Build an `EncodeReady` through [`EncodeReady::from_owned`] or
/// [`EncodeReady::copy_from`]. Use [`Self::into_parts`] to separate the
/// owned buffer from its metadata.
#[non_exhaustive]
pub struct EncodeReady {
    pixels: PixelBuffer,
    metadata: OutputMetadata,
}

impl EncodeReady {
    /// Finalize into independently owned pixels using a previously accepted plan.
    pub fn copy_from(
        source: PixelSlice<'_>,
        mut plan: OutputPlan,
    ) -> Result<Self, At<ConvertError>> {
        let exact = plan
            .borrow_rows(source.clone())?
            .and_then(|view| view.exact_depth());
        let mut pixels =
            PixelBuffer::try_new(source.width(), source.rows(), plan.output_descriptor())
                .map_err_at(ConvertError::from)?;
        let output = plan.convert_rows(source, pixels.as_slice_mut())?;
        let context = output
            .color_context()
            .cloned()
            .unwrap_or_else(|| plan.output_color_context().clone());
        pixels
            .set_exact_depth(exact)
            .expect("identity view validates the same representation");
        Ok(Self {
            pixels: pixels.with_color_context(context),
            metadata: plan.into_metadata(),
        })
    }

    /// Take ownership and reuse the allocation when sample bytes already match.
    pub fn from_owned(source: PixelBuffer, plan: OutputPlan) -> Result<Self, At<ConvertError>> {
        if plan.borrow_rows(source.as_slice())?.is_some() {
            let pixels = source
                .with_descriptor(plan.output_descriptor())
                .with_color_context(plan.output_color_context().clone());
            return Ok(Self {
                pixels,
                metadata: plan.into_metadata(),
            });
        }
        Self::copy_from(source.as_slice(), plan)
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
}

/// Resolve current color once; provenance never overrides working pixels.
#[cfg(feature = "hdr-experimental")]
pub(crate) fn current_profile(
    buffer: &PixelBuffer,
) -> Result<crate::ColorProfileSource<'_>, At<ConvertError>> {
    current_profile_for(buffer.encoding())
}

pub(crate) fn current_profile_for(
    encoding: crate::PixelEncoding<'_>,
) -> Result<crate::ColorProfileSource<'_>, At<ConvertError>> {
    let desc = encoding.descriptor();
    if let Some(context) = encoding.color_context() {
        if let Some(cicp) = context.cicp() {
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
            let primaries_conflict =
                desc.primaries != ColorPrimaries::Unknown && signaled.primaries != desc.primaries;
            let transfer_conflict = desc.transfer() != TransferFunction::Unknown
                && signaled.transfer() != desc.transfer();
            if primaries_conflict || transfer_conflict || signaled.signal_range != desc.signal_range
            {
                return Err(whereat::at!(ConvertError::NoPath {
                    from: desc,
                    to: signaled
                }));
            }
        }
        if let Some(param) = context.parametric()
            && let Some((primaries, transfer)) = param.to_primaries_transfer()
        {
            let primaries_conflict =
                desc.primaries != ColorPrimaries::Unknown && desc.primaries != primaries;
            let transfer_conflict =
                desc.transfer() != TransferFunction::Unknown && desc.transfer() != transfer;
            if primaries_conflict || transfer_conflict {
                let signaled = desc.with_primaries(primaries).with_transfer(transfer);
                return Err(whereat::at!(ConvertError::NoPath {
                    from: desc,
                    to: signaled,
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
    let non_icc = origin
        .parametric
        .map(crate::ColorProfileSource::Parametric)
        .or_else(|| {
            origin.cicp.map(|c| {
                crate::ColorProfileSource::Cicp(Cicp::new(
                    c.color_primaries,
                    c.transfer_characteristics,
                    0,
                    true,
                ))
            })
        });
    origin.color_authority.select(icc, non_icc)
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

pub(crate) fn profiles_match(
    a: &crate::ColorProfileSource<'_>,
    b: &crate::ColorProfileSource<'_>,
) -> bool {
    match (a, b) {
        (crate::ColorProfileSource::Icc(a), crate::ColorProfileSource::Icc(b)) => a == b,
        (crate::ColorProfileSource::Cicp(a), crate::ColorProfileSource::Cicp(b)) => {
            a == b
                || (a.full_range
                    && b.full_range
                    && a.matrix_coefficients == 0
                    && b.matrix_coefficients == 0
                    && a.color_primaries == b.color_primaries
                    && a.transfer_characteristics == b.transfer_characteristics)
        }
        (crate::ColorProfileSource::Icc(_), _) | (_, crate::ColorProfileSource::Icc(_)) => false,
        (crate::ColorProfileSource::Parametric(a), crate::ColorProfileSource::Parametric(b)) => {
            a == b
        }
        _ => a.resolve().is_some() && a.resolve() == b.resolve(),
    }
}
