//! Structured conversion errors wrapped in [`whereat::At`] for source-site traces.

use core::fmt;
use whereat::At;
use zenpixels::{BufferError, ColorError, LayoutError};

/// Error returned when planning or executing a pixel conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConvertError {
    /// Underlying [`ImageSpec`](zenpixels::ImageSpec) validation error.
    InvalidSpec(ColorError),
    /// Underlying [`SampleLayout`](zenpixels::SampleLayout) validation error.
    InvalidLayout(LayoutError),
    /// Underlying pixel buffer geometry, alignment, or allocation error.
    Buffer(BufferError),
    /// Conversion crosses absolute PQ luminance ↔ relative linear light without a [`DiffuseWhite`](zenpixels::DiffuseWhite) anchor.
    MissingHdrAnchor,
    /// Conversion renders HLG into linear/SDR/PQ without explicit [`HlgRendering`](zenpixels::HlgRendering) conditions.
    HlgRenderingRequired,
    /// HDR tone mapping requested [`PeakSource::Declared`](zenpixels::PeakSource::Declared) but no MaxCLL or ST 2086 mastering peak was present.
    MissingDeclaredPeak,
    /// HDR transfer (`Pq` / `Hlg`) requires enabling the `hdr-transfers` Cargo feature.
    HdrFeatureRequired,
    /// Requested color space or operator requires an optional Cargo feature (`oklab`, `perceptual`, `bt2446a`, etc.).
    FeatureRequired(&'static str),
    /// Conversion involves an arbitrary ICC profile or CMYK ink space and requires a [`CmsBackend`](crate::cms::CmsBackend).
    CmsRequired,
    /// Destination has lower precision than source while `DepthPolicy::Forbid` is active.
    DepthReductionForbidden,
    /// Conversion drops non-opaque alpha while `AlphaPolicy::Forbid` or `DiscardIfOpaque` (unproven) is active.
    AlphaLossForbidden,
    /// Conversion changes `Full ↔ Narrow` signal range while `RangePolicy::Forbid` is active.
    RangeConversionForbidden,
    /// Conversion narrows gamut or clips extended values while `GamutPolicy::Forbid` is active.
    GamutLossForbidden,
    /// Conversion route is not supported (e.g., unknown primaries across a gamut change, or non-standard sample type).
    UnsupportedRoute,
    /// Source and destination dimensions or row byte lengths disagree.
    DimensionMismatch,
    /// Custom [`ToneMapper`](crate::tonemap::ToneMapper) declared `full_image = true`, which is not supported in streaming plans.
    FullImageMapperNotSupported,
    /// Custom [`ToneMapper`](crate::tonemap::ToneMapper) failed to prepare for this image context.
    ToneMapperFailed,
    /// CMS backend rejected or failed to execute the transform.
    CmsBackendFailed,
    /// Requested container color signaling (`CicpOnly` / `EmbedIcc`) is not available for the destination color space.
    OutputSignalingUnavailable,
    /// Requested container color signaling (`SameAsOrigin` / `CustomIcc`) contradicts the destination pixels.
    OutputSignalingMismatch,
    /// Heap allocation failed.
    AllocationFailed,
    /// Cooperative cancellation was triggered via [`enough::Stop`].
    Cancelled(enough::StopReason),
}

impl From<enough::StopReason> for ConvertError {
    fn from(r: enough::StopReason) -> Self {
        Self::Cancelled(r)
    }
}

impl From<ColorError> for ConvertError {
    fn from(e: ColorError) -> Self {
        Self::InvalidSpec(e)
    }
}

impl From<LayoutError> for ConvertError {
    fn from(e: LayoutError) -> Self {
        Self::InvalidLayout(e)
    }
}

impl From<BufferError> for ConvertError {
    fn from(e: BufferError) -> Self {
        match e {
            BufferError::AllocationFailed => Self::AllocationFailed,
            other => Self::Buffer(other),
        }
    }
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec(e) => e.fmt(f),
            Self::InvalidLayout(e) => e.fmt(f),
            Self::Buffer(e) => e.fmt(f),
            Self::MissingHdrAnchor => f.write_str(
                "conversion between absolute PQ and relative light requires a DiffuseWhite anchor",
            ),
            Self::HlgRenderingRequired => f.write_str(
                "converting scene-referred HLG requires explicit HlgRendering conditions",
            ),
            Self::MissingDeclaredPeak => f.write_str(
                "PeakSource::Declared requested, but neither MaxCLL nor ST 2086 mastering peak is set",
            ),
            Self::HdrFeatureRequired => {
                f.write_str("PQ/HLG conversions require the `hdr-transfers` feature")
            }
            Self::FeatureRequired(feat) => {
                write!(f, "requested conversion requires Cargo feature `{feat}`")
            }
            Self::CmsRequired => {
                f.write_str("ICC or CMYK conversion requires a configured CmsBackend")
            }
            Self::DepthReductionForbidden => {
                f.write_str("bit-depth or precision reduction forbidden by DepthPolicy::Forbid")
            }
            Self::AlphaLossForbidden => {
                f.write_str("dropping non-opaque alpha forbidden by AlphaPolicy")
            }
            Self::RangeConversionForbidden => {
                f.write_str("Full<->Narrow signal range change forbidden by RangePolicy::Forbid")
            }
            Self::GamutLossForbidden => {
                f.write_str("gamut clipping or narrowing forbidden by GamutPolicy::Forbid")
            }
            Self::UnsupportedRoute => {
                f.write_str("unsupported color or layout conversion route")
            }
            Self::DimensionMismatch => {
                f.write_str("source and destination image dimensions or row lengths do not match")
            }
            Self::FullImageMapperNotSupported => {
                f.write_str("full-image buffering ToneMapper is not supported in streaming plans")
            }
            Self::ToneMapperFailed => f.write_str("tone mapper failed to prepare"),
            Self::CmsBackendFailed => f.write_str("CMS backend failed to build or run transform"),
            Self::OutputSignalingUnavailable => {
                f.write_str("requested container CICP or ICC signaling is unavailable for destination color space")
            }
            Self::OutputSignalingMismatch => {
                f.write_str("requested container color signaling contradicts the converted destination pixels")
            }
            Self::AllocationFailed => f.write_str("allocation failed during pixel conversion"),
            Self::Cancelled(r) => write!(f, "conversion cancelled: {r:?}"),
        }
    }
}

impl core::error::Error for ConvertError {}

#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn err_at(e: ConvertError) -> At<ConvertError> {
    whereat::at!(e)
}
