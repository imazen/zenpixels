//! Error types for pixel format conversion.

use crate::{PixelDescriptor, TransferFunction};
use core::fmt;

/// Errors that can occur during pixel format negotiation or conversion.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvertError {
    /// No supported format could be found for the source descriptor.
    NoMatch { source: PixelDescriptor },
    /// No conversion path exists between the two formats.
    NoPath {
        from: PixelDescriptor,
        to: PixelDescriptor,
    },
    /// Source and destination buffer sizes don't match the expected dimensions.
    BufferSize { expected: usize, actual: usize },
    /// Width is zero or would overflow stride calculations.
    InvalidWidth(u32),
    /// The supported format list was empty.
    EmptyFormatList,
    /// Conversion between these transfer functions is not yet supported.
    UnsupportedTransfer {
        from: TransferFunction,
        to: TransferFunction,
    },
    /// Alpha channel is not fully opaque and [`AlphaPolicy::DiscardIfOpaque`](crate::AlphaPolicy::DiscardIfOpaque) was set.
    AlphaNotOpaque,
    /// Depth reduction was requested but [`DepthPolicy::Forbid`](crate::DepthPolicy::Forbid) was set.
    DepthReductionForbidden,
    /// Alpha removal was requested but [`AlphaPolicy::Forbid`](crate::AlphaPolicy::Forbid) was set.
    AlphaRemovalForbidden,
    /// RGB-to-grayscale conversion requires explicit luma coefficients.
    RgbToGray,
    /// Buffer allocation failed.
    AllocationFailed,
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMatch { source } => {
                write!(
                    f,
                    "no supported format matches source {:?}/{:?}",
                    source.channel_type(),
                    source.layout()
                )
            }
            Self::NoPath { from, to } => {
                write!(
                    f,
                    "no conversion path from {:?}/{:?} to {:?}/{:?}",
                    from.channel_type(),
                    from.layout(),
                    to.channel_type(),
                    to.layout()
                )
            }
            Self::BufferSize { expected, actual } => {
                write!(
                    f,
                    "buffer size mismatch: expected {expected} bytes, got {actual}"
                )
            }
            Self::InvalidWidth(w) => write!(f, "invalid width: {w}"),
            Self::EmptyFormatList => write!(f, "supported format list is empty"),
            Self::UnsupportedTransfer { from, to } => {
                write!(f, "unsupported transfer conversion: {from:?} → {to:?}")
            }
            Self::AlphaNotOpaque => write!(f, "alpha channel is not fully opaque"),
            Self::DepthReductionForbidden => write!(f, "depth reduction forbidden by policy"),
            Self::AlphaRemovalForbidden => write!(f, "alpha removal forbidden by policy"),
            Self::RgbToGray => {
                write!(f, "RGB-to-grayscale requires explicit luma coefficients")
            }
            Self::AllocationFailed => write!(f, "buffer allocation failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConvertError {}
