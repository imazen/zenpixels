//! Error types for pixel format conversion.

use crate::{ColorModel, PixelDescriptor, TransferFunction};
use core::fmt;

/// Errors that can occur during pixel format negotiation or conversion.
//
// `#[non_exhaustive]` (added 0.2.14): cargo-copter confirmed sealing it broke
// zero of zpc's published reverse-dependents — nobody matches `ConvertError`
// exhaustively — so it shipped as a tolerated 0.2.x break instead of waiting for
// 0.3.0. That in turn let the `Buffer(BufferError)` variant land in the same
// patch (adding a variant to an already-`#[non_exhaustive]` enum is not a
// break). `Buffer` preserves the real `zenpixels::BufferError` cause
// (`StrideTooSmall` / `InvalidDimensions` / …) instead of collapsing every
// buffer-construction failure into `AllocationFailed` (an out-of-memory label);
// the construction sites map via `map_err_at(ConvertError::from)`, which keeps
// both the cause and the `At` location trace.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
    /// A pixel buffer or slice could not be constructed: carries the real
    /// [`zenpixels::BufferError`] cause (`StrideTooSmall`, `InvalidDimensions`,
    /// …) instead of collapsing it into [`AllocationFailed`](Self::AllocationFailed).
    Buffer(zenpixels::BufferError),
    /// CMS transform could not be built (invalid ICC profile, unsupported color space, etc.).
    CmsError(alloc::string::String),
    /// The conversion is HDR (`Pq` / `Hlg`) → SDR but no usable peak
    /// luminance was supplied. Raised both when no peak was given at all
    /// (the plain [`ConvertPlan::new`](crate::ConvertPlan::new) entry
    /// point doesn't take one) and when a supplied `HdrConfig` carries a
    /// non-finite or non-positive `source_peak_nits` / `target_peak_nits`
    /// (including the unset `HdrConfig::default()` value `0.0`) — a
    /// degenerate peak would tone-map every pixel to black. Build the
    /// plan via `ConvertPlan::new_with_hdr_peak` (or
    /// `ConvertPlan::new_with_hdr_config` for full knob control), and
    /// pass the source's MaxCLL — e.g. from
    /// `hdr::measure::CllMeasure::measure_max`. All three live behind
    /// the `hdr-experimental` Cargo feature (plain code spans here, not
    /// intra-doc links, so this page renders link-clean without it).
    ///
    /// Pre-0.2.16 the plain `ConvertPlan::new` silently routed HDR→SDR
    /// through the linear intermediate with no tone-mapping, producing
    /// semantically wrong pixels. This variant replaces that with a
    /// loud refusal.
    HdrSourceRequiresPeak {
        from: PixelDescriptor,
        to: PixelDescriptor,
    },
    /// The conversion requires a color management plugin but none was provided.
    ///
    /// Returned when one (or both) sides use a non-native color model — CMYK,
    /// Lab, XYZ, spot inks, or any future device-dependent space — that
    /// `zenpixels-convert` cannot resolve with its built-in kernels. Attach
    /// a plugin via
    /// [`RowConverter::new_explicit_with_cms`](crate::RowConverter::new_explicit_with_cms)
    /// (e.g. `Some(&MoxCms)` under the `cms-moxcms` feature) and the plan
    /// will dispatch the full row work to it.
    ///
    /// Distinct from [`NoPath`](Self::NoPath): `NeedsCms` says "a path
    /// exists, but requires CMS dispatch"; `NoPath` says "no architecturally
    /// possible conversion." Callers that want to route to a CMS should
    /// match on `NeedsCms` and re-issue the call with a plugin attached.
    ///
    /// **Pre-0.2.16:** the same descriptors caused a process-aborting
    /// `assert_not_cmyk` panic — replaced by this typed variant so the
    /// documented `Some(&MoxCms)` escape hatch is actually reachable.
    NeedsCms {
        from: PixelDescriptor,
        to: PixelDescriptor,
    },
    // CMYK rejection used to be folded into `NoPath { from, to }`. Pre-0.2.16
    // the public-API entry points panicked via `assert_not_cmyk` BEFORE the CMS
    // chain was consulted, so the documented escape hatch ("attach moxcms for
    // CMYK↔RGB") was unreachable. 0.2.16 introduces `NeedsCms { from, to }`:
    // the panic becomes a typed `Err`, callers can match the variant and
    // re-issue with a plugin, and the moxcms backend dispatches CMYK→RGB
    // end-to-end. `NoPath` is retained for genuinely impossible conversions
    // (signal-range crossings without a kernel; HLG↔PQ until OOTF threading
    // lands). `ConvertError` is `#[non_exhaustive]` so this is additive.
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
                // CMYK or signal-range crossings would otherwise print two
                // identical-looking descriptors with no hint why the conversion
                // failed. Name the actual blocker so the caller can route
                // intelligently (CMYK → moxcms/CMS pipeline; range crossing →
                // explicit rescale stage).
                let cmyk_blocked =
                    from.color_model() == ColorModel::Cmyk || to.color_model() == ColorModel::Cmyk;
                if cmyk_blocked {
                    write!(
                        f,
                        " (CMYK is device-dependent and requires an ICC profile; \
                         zenpixels-convert reinterpreting C/M/Y/K as R/G/B/A would \
                         silently corrupt both colour and transparency. Use a CMS \
                         such as moxcms for CMYK↔RGB)"
                    )?;
                }
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
            Self::Buffer(e) => write!(f, "buffer construction failed: {e}"),
            Self::CmsError(msg) => write!(f, "CMS transform failed: {msg}"),
            Self::HdrSourceRequiresPeak { from, to } => write!(
                f,
                "HDR→SDR conversion ({:?} → {:?}) requires positive, finite peak \
                 luminances; build the plan via ConvertPlan::new_with_hdr_peak (or \
                 new_with_hdr_config) and pass the source's MaxCLL (e.g. \
                 CllMeasure::measure_max)",
                from.transfer(),
                to.transfer(),
            ),
            Self::NeedsCms { from, to } => write!(
                f,
                "conversion from {} to {} requires a color management plugin: \
                 call RowConverter::new_explicit_with_cms(_, _, _, Some(&MoxCms)) \
                 (or another PluggableCms backend) to dispatch the row work \
                 to a CMS",
                from.color_model(),
                to.color_model(),
            ),
        }
    }
}

impl From<zenpixels::BufferError> for ConvertError {
    /// Wrap a buffer-construction failure's real cause into
    /// [`ConvertError::Buffer`]. Pair with `map_err_at` at the call sites so the
    /// `At` location trace is preserved alongside the classified cause.
    fn from(err: zenpixels::BufferError) -> Self {
        Self::Buffer(err)
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
    fn display_needs_cms() {
        let e = ConvertError::NeedsCms {
            from: PixelDescriptor::CMYK8,
            to: PixelDescriptor::RGB8_SRGB,
        };
        let s = format!("{e}");
        assert!(s.contains("color management plugin"), "{s}");
        assert!(s.contains("CMYK"), "{s}");
        assert!(s.contains("RGB"), "{s}");
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
