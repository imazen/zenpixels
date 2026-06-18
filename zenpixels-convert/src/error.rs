//! Error types for pixel format conversion.

use crate::{PixelDescriptor, TransferFunction};
use core::fmt;

/// Errors that can occur during pixel format negotiation or conversion.
//
// ── ConvertError 0.3.0 note (authoritative) ─────────────────────────────────
// This enum is currently EXHAUSTIVE: `#[non_exhaustive]` was removed to avoid a
// semver break vs 0.2.3, so adding any variant is itself a break. Two related
// changes are batched for the next 0.3.0 release (see CHANGELOG "QUEUED
// BREAKING CHANGES"):
//
//   (a) Add `#[non_exhaustive]` to `ConvertError`.
//   (b) Add a new variant `Buffer(zenpixels::BufferError)` that carries the
//       real underlying buffer error instead of collapsing it.
//   (c) Add `impl From<zenpixels::BufferError> for ConvertError` (wrapping into
//       the new `Buffer` variant) so the `?`/`map_err_at` ergonomics are clean.
//   (d) Switch the 8 `PixelBuffer::{try_new,from_vec}` / `PixelSlice::new`
//       construction sites — 2 in `ext.rs`, 1 in `hdr.rs`, 5 in `output.rs`,
//       each tagged with a per-site `FIXME` — from the catch-all
//       `.map_err_at(|_| ConvertError::AllocationFailed)?` to
//       `.map_err_at(ConvertError::from)?`.
//
// Why: `zenpixels::PixelBuffer`/`PixelSlice` construction returns
// `At<zenpixels::BufferError>`, a 7-variant enum (`AlignmentViolation`,
// `InsufficientData`, `StrideTooSmall`, `StrideNotPixelAligned`,
// `InvalidDimensions`, `IncompatibleDescriptor`, `AllocationFailed`). Today all
// 7 are mislabeled as `ConvertError::AllocationFailed` — i.e. a `StrideTooSmall`
// or `InvalidDimensions` *layout* error is reported as an out-of-memory failure.
// The #52 fix already preserves the `At<BufferError>` *trace* across the convert
// boundary (via `map_err_at`); the remaining 0.3.0 work fixes the
// *classification* so the true reason survives. Adding the variant and
// `#[non_exhaustive]` are both semver breaks, which is why this is deferred to
// 0.3.0 rather than shipped in a 0.2.N patch.
// ────────────────────────────────────────────────────────────────────────────
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
    /// CMS transform could not be built (invalid ICC profile, unsupported color space, etc.).
    CmsError(alloc::string::String),
    // TODO(0.3.0): add HdrTransferRequiresToneMapping variant here once
    // ConvertError is #[non_exhaustive]. Adding a variant to an exhaustive
    // enum is a semver break. See also HdrPolicy in output.rs and
    // imazen/zenpixels#10 for the full HDR provenance plan.
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
                )?;
                // A range crossing can otherwise print two identical-looking
                // descriptors; name the actual blocker.
                if from.signal_range != to.signal_range {
                    write!(
                        f,
                        " (signal range {} -> {}: no narrow<->full conversion kernels exist; \
                         relabeling without rescaling would corrupt pixel values)",
                        from.signal_range, to.signal_range
                    )?;
                }
                Ok(())
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
            Self::CmsError(msg) => write!(f, "CMS transform failed: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConvertError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn display_no_match() {
        let e = ConvertError::NoMatch {
            source: PixelDescriptor::RGB8_SRGB,
        };
        let s = format!("{e}");
        assert!(s.contains("no supported format"));
        assert!(s.contains("U8"));
        assert!(s.contains("Rgb"));
    }

    #[test]
    fn display_no_path() {
        let e = ConvertError::NoPath {
            from: PixelDescriptor::RGB8_SRGB,
            to: PixelDescriptor::GRAY8_SRGB,
        };
        let s = format!("{e}");
        assert!(s.contains("no conversion path"));
    }

    #[test]
    fn display_buffer_size() {
        let e = ConvertError::BufferSize {
            expected: 1024,
            actual: 512,
        };
        let s = format!("{e}");
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
    }

    #[test]
    fn display_invalid_width() {
        let e = ConvertError::InvalidWidth(0);
        assert!(format!("{e}").contains("0"));
    }

    #[test]
    fn display_empty_format_list() {
        let s = format!("{}", ConvertError::EmptyFormatList);
        assert!(s.contains("empty"));
    }

    #[test]
    fn display_unsupported_transfer() {
        let e = ConvertError::UnsupportedTransfer {
            from: TransferFunction::Pq,
            to: TransferFunction::Hlg,
        };
        let s = format!("{e}");
        assert!(s.contains("Pq"));
        assert!(s.contains("Hlg"));
    }

    #[test]
    fn display_alpha_not_opaque() {
        assert!(format!("{}", ConvertError::AlphaNotOpaque).contains("opaque"));
    }

    #[test]
    fn display_depth_reduction_forbidden() {
        assert!(format!("{}", ConvertError::DepthReductionForbidden).contains("forbidden"));
    }

    #[test]
    fn display_alpha_removal_forbidden() {
        assert!(format!("{}", ConvertError::AlphaRemovalForbidden).contains("forbidden"));
    }

    #[test]
    fn display_rgb_to_gray() {
        assert!(format!("{}", ConvertError::RgbToGray).contains("luma"));
    }

    #[test]
    fn display_allocation_failed() {
        assert!(format!("{}", ConvertError::AllocationFailed).contains("allocation"));
    }

    #[test]
    fn display_cms_error() {
        let e = ConvertError::CmsError(alloc::string::String::from("profile mismatch"));
        let s = format!("{e}");
        assert!(s.contains("CMS transform failed"));
        assert!(s.contains("profile mismatch"));
    }

    #[test]
    fn error_eq() {
        assert_eq!(ConvertError::AlphaNotOpaque, ConvertError::AlphaNotOpaque);
        assert_ne!(ConvertError::AlphaNotOpaque, ConvertError::RgbToGray);
    }

    #[test]
    fn error_debug() {
        let e = ConvertError::AllocationFailed;
        let s = format!("{e:?}");
        assert!(s.contains("AllocationFailed"));
    }

    #[test]
    fn error_clone() {
        let e = ConvertError::BufferSize {
            expected: 100,
            actual: 50,
        };
        let e2 = e.clone();
        assert_eq!(e, e2);
    }
}
