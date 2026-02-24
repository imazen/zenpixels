//! Error types for pixel format conversion.

use core::fmt;
use zencodec_types::{PixelDescriptor, TransferFunction};

/// Errors that can occur during pixel format negotiation or conversion.
#[derive(Debug, Clone)]
pub enum ConvertError {
    /// No supported format could be found for the source descriptor.
    NoMatch {
        source: PixelDescriptor,
    },
    /// No conversion path exists between the two formats.
    NoPath {
        from: PixelDescriptor,
        to: PixelDescriptor,
    },
    /// Source and destination buffer sizes don't match the expected dimensions.
    BufferSize {
        expected: usize,
        actual: usize,
    },
    /// Width is zero or would overflow stride calculations.
    InvalidWidth(u32),
    /// The supported format list was empty.
    EmptyFormatList,
    /// Conversion between these transfer functions is not yet supported.
    UnsupportedTransfer {
        from: TransferFunction,
        to: TransferFunction,
    },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMatch { source } => {
                write!(
                    f,
                    "no supported format matches source {:?}/{:?}",
                    source.channel_type, source.layout
                )
            }
            Self::NoPath { from, to } => {
                write!(
                    f,
                    "no conversion path from {:?}/{:?} to {:?}/{:?}",
                    from.channel_type, from.layout, to.channel_type, to.layout
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
                write!(
                    f,
                    "unsupported transfer conversion: {from:?} → {to:?}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConvertError {}
