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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn gray_expand_derive_traits() {
        let a = GrayExpand::Broadcast;
        let b = a;
        #[allow(clippy::clone_on_copy)]
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
        let _ = format!("{a:?}");
    }

    #[test]
    fn alpha_policy_variants() {
        let discard = AlphaPolicy::DiscardIfOpaque;
        let unchecked = AlphaPolicy::DiscardUnchecked;
        let composite = AlphaPolicy::CompositeOnto {
            r: 255,
            g: 255,
            b: 255,
        };
        let forbid = AlphaPolicy::Forbid;

        assert_ne!(discard, unchecked);
        assert_ne!(composite, forbid);

        let composite2 = AlphaPolicy::CompositeOnto {
            r: 255,
            g: 255,
            b: 255,
        };
        assert_eq!(composite, composite2);

        let composite_diff = AlphaPolicy::CompositeOnto { r: 0, g: 0, b: 0 };
        assert_ne!(composite, composite_diff);
    }

    #[test]
    fn depth_policy_variants() {
        assert_ne!(DepthPolicy::Round, DepthPolicy::Truncate);
        assert_ne!(DepthPolicy::Round, DepthPolicy::Forbid);
        let a = DepthPolicy::Truncate;
        #[allow(clippy::clone_on_copy)]
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn luma_coefficients_variants() {
        assert_ne!(LumaCoefficients::Bt709, LumaCoefficients::Bt601);
        let a = LumaCoefficients::Bt709;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn convert_options_derive_traits() {
        let opts = ConvertOptions {
            gray_expand: GrayExpand::Broadcast,
            alpha_policy: AlphaPolicy::DiscardUnchecked,
            depth_policy: DepthPolicy::Round,
            luma: Some(LumaCoefficients::Bt709),
        };
        #[allow(clippy::clone_on_copy)]
        let opts2 = opts.clone();
        assert_eq!(opts, opts2);
        let _ = format!("{opts:?}");
    }

    #[test]
    #[cfg(feature = "std")]
    fn alpha_policy_hash() {
        use core::hash::{Hash, Hasher};
        let mut h1 = std::hash::DefaultHasher::new();
        AlphaPolicy::Forbid.hash(&mut h1);
        let mut h2 = std::hash::DefaultHasher::new();
        AlphaPolicy::Forbid.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    #[cfg(feature = "std")]
    fn convert_options_hash() {
        use core::hash::{Hash, Hasher};
        let opts = ConvertOptions {
            gray_expand: GrayExpand::Broadcast,
            alpha_policy: AlphaPolicy::Forbid,
            depth_policy: DepthPolicy::Forbid,
            luma: None,
        };
        let mut h = std::hash::DefaultHasher::new();
        opts.hash(&mut h);
        // Just verify it doesn't panic.
        let _ = h.finish();
    }
}
