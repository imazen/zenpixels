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
///
/// Use [`coefficients()`](Self::coefficients) to get the concrete `[f32; 3]`
/// weights for each variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LumaCoefficients {
    /// BT.709: `0.2126R + 0.7152G + 0.0722B` (HDTV, sRGB).
    Bt709,
    /// BT.601: `0.299R + 0.587G + 0.114B` (SDTV, JPEG).
    Bt601,
    /// BT.2020 / BT.2100: `0.2627R + 0.6780G + 0.0593B` (UHDTV, wide gamut).
    ///
    /// BT.2100 (HDR) uses the same primaries as BT.2020, so this is the
    /// variant to use for both SDR BT.2020 and HDR BT.2100 content.
    Bt2020,
    /// Display P3 (DCI-P3 primaries + D65): `0.2289746R + 0.6917385G + 0.0792869B`
    /// (Apple wide-gamut consumer displays).
    ///
    /// Unlike BT.709 and BT.2020, no ITU recommendation prescribes these
    /// weights — they are derived as the middle (Y) row of the
    /// DisplayP3→XYZ matrix from the P3 primaries and D65 white point, and
    /// match what libultrahdr and other HDR tooling use in practice for
    /// RGB→luma on DisplayP3 content.
    DisplayP3,
}

impl LumaCoefficients {
    /// Return the `[R, G, B]` weights for this luma recipe.
    ///
    /// - [`Bt709`](Self::Bt709): `[0.2126, 0.7152, 0.0722]`
    /// - [`Bt601`](Self::Bt601): `[0.299, 0.587, 0.114]`
    /// - [`Bt2020`](Self::Bt2020): `[0.2627, 0.6780, 0.0593]` (same as BT.2100)
    /// - [`DisplayP3`](Self::DisplayP3): `[0.2289746, 0.6917385, 0.0792869]`
    ///
    /// The three weights always sum to exactly 1.0 in IEEE 754 double
    /// precision, but may sum to 1.0 ± 1 ULP in `f32`. Callers that need
    /// the weights to sum to exactly 1.0 in f32 should use double-precision
    /// accumulation in their inner loop.
    #[inline]
    pub const fn coefficients(self) -> [f32; 3] {
        match self {
            Self::Bt709 => [0.2126, 0.7152, 0.0722],
            Self::Bt601 => [0.299, 0.587, 0.114],
            Self::Bt2020 => [0.2627, 0.6780, 0.0593],
            Self::DisplayP3 => [0.2289746, 0.6917385, 0.0792869],
        }
    }
}

/// Explicit options for pixel format conversion. All lossy
/// operations require a policy choice — no silent defaults.
///
/// Construct via struct literal for full control, or use the convenience
/// constructors and `with_*` builders for common patterns:
///
/// ```
/// use zenpixels::{ConvertOptions, AlphaPolicy, DepthPolicy};
///
/// // Forbid all lossy operations (safe default)
/// let strict = ConvertOptions::forbid_lossy();
///
/// // Allow common lossy operations with sensible defaults
/// let permissive = ConvertOptions::permissive();
///
/// // Customize from a preset
/// let custom = ConvertOptions::permissive()
///     .with_alpha_policy(AlphaPolicy::CompositeOnto { r: 255, g: 255, b: 255 })
///     .with_depth_policy(DepthPolicy::Truncate);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
    /// Whether to clamp out-of-gamut values to [0, 1] during f32 transfer
    /// function conversions.
    ///
    /// - `true` (default): clamp sRGB/BT.709/PQ/HLG transfers to [0, 1].
    ///   Matches display expectations; safe for standard workflows.
    /// - `false`: use sign-preserving extended-range transfer functions.
    ///   Preserves out-of-gamut (negative, > 1.0) values through the f32
    ///   pipeline for HDR and wide-gamut workflows where tone mapping or
    ///   gamut mapping happens later in the pipeline.
    ///
    /// Only affects f32 intermediate conversions. u8/u16 outputs always
    /// clip since those formats can't represent out-of-gamut values.
    pub clip_out_of_gamut: bool,
}

impl ConvertOptions {
    /// Forbid all lossy operations.
    ///
    /// - Alpha removal: forbidden (returns error)
    /// - Depth reduction: forbidden (returns error)
    /// - RGB→Gray: forbidden (returns error)
    /// - Gray→RGB: broadcast (lossless)
    ///
    /// Use this as a starting point when you want to ensure no data loss,
    /// then selectively relax with `with_*` methods.
    pub const fn forbid_lossy() -> Self {
        Self {
            gray_expand: GrayExpand::Broadcast,
            alpha_policy: AlphaPolicy::Forbid,
            depth_policy: DepthPolicy::Forbid,
            luma: None,
            clip_out_of_gamut: true,
        }
    }

    /// Allow common lossy operations with sensible defaults.
    ///
    /// - Alpha removal: discard only if all pixels are opaque
    /// - Depth reduction: round to nearest
    /// - RGB→Gray: BT.709 luma coefficients
    /// - Gray→RGB: broadcast (lossless)
    pub const fn permissive() -> Self {
        Self {
            gray_expand: GrayExpand::Broadcast,
            alpha_policy: AlphaPolicy::DiscardIfOpaque,
            depth_policy: DepthPolicy::Round,
            luma: Some(LumaCoefficients::Bt709),
            clip_out_of_gamut: true,
        }
    }

    /// Set the alpha removal policy.
    pub const fn with_alpha_policy(mut self, policy: AlphaPolicy) -> Self {
        self.alpha_policy = policy;
        self
    }

    /// Set the depth reduction policy.
    pub const fn with_depth_policy(mut self, policy: DepthPolicy) -> Self {
        self.depth_policy = policy;
        self
    }

    /// Set the grayscale expansion method.
    pub const fn with_gray_expand(mut self, expand: GrayExpand) -> Self {
        self.gray_expand = expand;
        self
    }

    /// Set whether f32 transfer conversions clamp to [0, 1] (`true`, default)
    /// or preserve extended-range values via sign-preserving transfers
    /// (`false`). u8/u16 outputs always clip.
    pub const fn with_clip_out_of_gamut(mut self, clip: bool) -> Self {
        self.clip_out_of_gamut = clip;
        self
    }

    /// Set the luma coefficients for RGB→Gray conversion.
    ///
    /// Pass `None` to forbid RGB→Gray conversion.
    pub const fn with_luma(mut self, luma: Option<LumaCoefficients>) -> Self {
        self.luma = luma;
        self
    }
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
        assert_ne!(LumaCoefficients::Bt709, LumaCoefficients::Bt2020);
        assert_ne!(LumaCoefficients::Bt601, LumaCoefficients::Bt2020);
        assert_ne!(LumaCoefficients::Bt709, LumaCoefficients::DisplayP3);
        assert_ne!(LumaCoefficients::Bt601, LumaCoefficients::DisplayP3);
        assert_ne!(LumaCoefficients::Bt2020, LumaCoefficients::DisplayP3);
        let a = LumaCoefficients::Bt709;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn luma_coefficients_accessor_values() {
        // Exact bit patterns of documented coefficients — any numeric drift
        // in the enum is a behavior break for downstream RGB→Gray kernels.
        assert_eq!(
            LumaCoefficients::Bt709.coefficients(),
            [0.2126_f32, 0.7152, 0.0722],
        );
        assert_eq!(
            LumaCoefficients::Bt601.coefficients(),
            [0.299_f32, 0.587, 0.114],
        );
        assert_eq!(
            LumaCoefficients::Bt2020.coefficients(),
            [0.2627_f32, 0.6780, 0.0593],
        );
        assert_eq!(
            LumaCoefficients::DisplayP3.coefficients(),
            [0.2289746_f32, 0.6917385, 0.0792869],
        );
    }

    #[test]
    fn luma_coefficients_sum_to_near_unity() {
        // All three recipes are constructed from spec chromaticities and
        // must normalize to white = 1.0. Allow 1 f32 ULP of slack for
        // BT.2020 (its double-precision sum is exactly 1.0 but single
        // precision may round).
        for luma in [
            LumaCoefficients::Bt709,
            LumaCoefficients::Bt601,
            LumaCoefficients::Bt2020,
            LumaCoefficients::DisplayP3,
        ] {
            let [r, g, b] = luma.coefficients();
            let sum = r + g + b;
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "{luma:?} coefficients sum to {sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn convert_options_derive_traits() {
        let opts = ConvertOptions {
            gray_expand: GrayExpand::Broadcast,
            alpha_policy: AlphaPolicy::DiscardUnchecked,
            depth_policy: DepthPolicy::Round,
            luma: Some(LumaCoefficients::Bt709),
            clip_out_of_gamut: true,
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
            clip_out_of_gamut: true,
        };
        let mut h = std::hash::DefaultHasher::new();
        opts.hash(&mut h);
        // Just verify it doesn't panic.
        let _ = h.finish();
    }
}
