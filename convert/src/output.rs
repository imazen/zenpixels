//! Atomic output preparation for encoders.
//!
//! [`finalize_for_output`] converts pixel data and generates matching metadata
//! in a single atomic operation, preventing the most common color management
//! bug: pixel values that don't match the embedded color metadata.

use alloc::sync::Arc;

use crate::cms::ColorManagement;
use crate::error::ConvertError;
use crate::hdr::HdrMetadata;
use crate::{
    Cicp, ColorOrigin, ColorPrimaries, PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice,
    TransferFunction,
};

/// Target output color profile.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum OutputProfile {
    /// Re-encode with the original ICC/CICP from the source file.
    SameAsOrigin,
    /// Use a well-known CICP-described profile.
    Named(Cicp),
    /// Use specific ICC profile bytes.
    Icc(Arc<[u8]>),
}

/// Metadata that the encoder should embed alongside the pixel data.
///
/// Generated atomically by [`finalize_for_output`] to guarantee that
/// the metadata matches the pixel values.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct OutputMetadata {
    /// ICC profile bytes to embed, if any.
    pub icc: Option<Arc<[u8]>>,
    /// CICP code points to embed, if any.
    pub cicp: Option<Cicp>,
    /// HDR metadata to embed (content light level, mastering display), if any.
    pub hdr: Option<HdrMetadata>,
}

/// Pixel data bundled with matching metadata, ready for encoding.
///
/// The only way to create an `EncodeReady` is through [`finalize_for_output`],
/// which guarantees that the pixels and metadata are consistent.
///
/// Use [`into_parts()`](Self::into_parts) to destructure if needed, but
/// the default path keeps them coupled.
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
pub fn finalize_for_output<C: ColorManagement>(
    buffer: &PixelBuffer,
    origin: &ColorOrigin,
    target: OutputProfile,
    pixel_format: PixelFormat,
    cms: &C,
) -> Result<EncodeReady, ConvertError> {
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
            // If origin has ICC, we may need a CMS transform.
            (metadata, origin.icc.is_some())
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

    // If source has an ICC profile and we need CMS, use it.
    if needs_cms_transform
        && let Some(src_icc) = buffer.color_context().and_then(|c| c.icc.as_ref())
        && let Some(dst_icc) = &metadata.icc
    {
        let transform = cms
            .build_transform(src_icc, dst_icc)
            .map_err(|_| ConvertError::AllocationFailed)?;

        let src_slice = buffer.as_slice();
        let mut out = PixelBuffer::try_new(buffer.width(), buffer.height(), target_desc)
            .map_err(|_| ConvertError::AllocationFailed)?;

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
        let bytes = src_slice.contiguous_bytes();
        let out = PixelBuffer::from_vec(
            bytes.into_owned(),
            buffer.width(),
            buffer.height(),
            target_desc_full,
        )
        .map_err(|_| ConvertError::AllocationFailed)?;
        return Ok(EncodeReady {
            pixels: out,
            metadata,
        });
    }

    // Use RowConverter for format conversion.
    let converter = crate::RowConverter::new(source_desc, target_desc_full)?;
    let src_slice = buffer.as_slice();
    let mut out = PixelBuffer::try_new(buffer.width(), buffer.height(), target_desc_full)
        .map_err(|_| ConvertError::AllocationFailed)?;

    {
        let mut dst_slice = out.as_slice_mut();
        for y in 0..buffer.height() {
            let src_row = src_slice.row(y);
            let dst_row = dst_slice.row_mut(y);
            converter.convert_row(src_row, dst_row, buffer.width());
        }
    }

    Ok(EncodeReady {
        pixels: out,
        metadata,
    })
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
}
