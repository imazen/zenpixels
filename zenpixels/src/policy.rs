//! Conversion policy types for explicit control over lossy operations.
//!
//! All lossy pixel format conversions (alpha removal, depth reduction, etc.)
//! require an explicit policy choice — there are no silent defaults.

/// How to expand grayscale channels to RGB.
///
/// Used when converting from a grayscale layout to an RGB-family layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GrayExpand {
    /// Channel broadcast: `v → (v, v, v)`. Lossless.
    Broadcast,
}

/// Policy for alpha channel removal. Required when converting
/// from a layout with alpha to one without.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlphaPolicy {
    /// Discard only if every pixel is fully opaque. Returns error otherwise.
    DiscardIfOpaque,
    /// Discard unconditionally. Caller acknowledges data loss.
    DiscardUnchecked,
    /// Composite onto solid background (values in source range, 0–255 for U8).
    CompositeOnto {
        /// Red background value.
        r: u8,
        /// Green background value.
        g: u8,
        /// Blue background value.
        b: u8,
    },
    /// Return error rather than dropping alpha.
    Forbid,
}

/// Policy for bit depth reduction (U16→U8, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DepthPolicy {
    /// Round to nearest value.
    Round,
    /// Truncate (floor). Faster, biased toward lower values.
    Truncate,
    /// Return error rather than reducing depth.
    Forbid,
}

/// Luma coefficients for RGB→Gray conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LumaCoefficients {
    /// BT.709: `0.2126R + 0.7152G + 0.0722B` (HDTV, sRGB).
    Bt709,
    /// BT.601: `0.299R + 0.587G + 0.114B` (SDTV, JPEG).
    Bt601,
}

/// Explicit options for pixel format conversion. All lossy
/// operations require a policy choice — no silent defaults.
#[derive(Clone, Copy, Debug)]
pub struct ConvertOptions {
    /// How to expand grayscale to RGB.
    pub gray_expand: GrayExpand,
    /// How to handle alpha removal.
    pub alpha_policy: AlphaPolicy,
    /// How to handle depth reduction.
    pub depth_policy: DepthPolicy,
    /// Luma coefficients for RGB→Gray conversion. `None` means
    /// RGB→Gray is forbidden (returns `ConvertError::RgbToGray`).
    pub luma: Option<LumaCoefficients>,
}
