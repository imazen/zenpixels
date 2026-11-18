//! Conversion policy types for explicit control over lossy operations.
//!
//! Alpha removal, depth reduction, grayscale conversion and clipping have
//! explicit permissions. These options do not prove sample-exact conversion.

use crate::color::Color;
use crate::hdr::DiffuseWhite;

/// Policy for alpha channel removal. Required when converting
/// from a layout with alpha to one without.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlphaPolicy {
    /// Discard only if every pixel is fully opaque. Returns error otherwise.
    DiscardIfOpaque,
    /// Discard unconditionally. Caller acknowledges data loss.
    DiscardUnchecked,
    /// Composite onto a solid background. Blending happens in linear light.
    ///
    /// The background converts to the source's primaries first, so a background keeps its color
    /// on a wide-gamut source. A source whose primaries are unknown gets the background unconverted.
    /// Planning refuses a background with alpha, and one that can't convert to the source.
    CompositeOnto {
        /// The color to composite onto.
        background: Color,
        /// The luminance an SDR background's white sits at on a PQ or HLG source, so that it
        /// doesn't become the format's peak. Only this background uses it, and it has no effect
        /// on an SDR source or an HDR background, whose luminance is already known.
        ///
        /// **Set this for HDR sources, to the reference white the content was graded to.**
        /// `None` assumes [`DiffuseWhite::BT2408`]'s 203 cd/m², which may not match.
        /// HLG additionally requires explicit [`ConvertOptions::with_hlg_rendering`]
        /// conditions and an SDR-white anchor for the display-linear intermediate.
        reference_white: Option<DiffuseWhite>,
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

/// What happens to colors outside the target's gamut when primaries narrow.
///
/// Applies to built-in conversions between named SDR primaries, such as Display P3
/// or BT.2020 to sRGB. Requesting mapping where it isn't available returns an error
/// rather than clipping. See `docs/gamut-mapping-survey.md`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GamutPolicy {
    /// Keep extended-range floating-point values through conversion.
    PreserveExtended,
    /// Clamp each channel. In-gamut colors are untouched and it's the fastest, but
    /// saturated colors can change hue, up to 19° for BT.2020 blue in OKLab.
    #[default]
    Clip,
    /// Reduce OKLab chroma near and past the boundary, keeping hue and lightness.
    ///
    /// Smooth across gradients, and slower. It also changes in-gamut colors close to
    /// the boundary: 7 % of a Display P3 grid, by at most 0.0063 in OKLab.
    SoftCompress,
}

/// Explicit permissions for pixel format conversion. Presets select channel,
/// depth and clipping behavior; they do not prove reversible color arithmetic.
///
/// Construct via struct literal for full control, or use the convenience
/// constructors and `with_*` builders for common patterns:
///
/// ```
/// use zenpixels::{AlphaPolicy, Color, ConvertOptions, DepthPolicy};
///
/// // Retain channels and depth; requested color conversion may change samples.
/// let retained = ConvertOptions::retain_channels();
///
/// // Allow common lossy operations with sensible defaults
/// let options = ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Round);
///
/// // Customize from a preset
/// let custom = ConvertOptions::retain_channels()
///     .with_alpha_policy(AlphaPolicy::CompositeOnto {
///         background: Color::srgb8(255, 255, 255),
///         reference_white: None,
///     })
///     .with_depth_policy(DepthPolicy::Truncate);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ConvertOptions {
    /// How to handle alpha removal.
    alpha_policy: AlphaPolicy,
    /// How to handle depth reduction.
    depth_policy: DepthPolicy,
    /// Luma coefficients for RGB→Gray conversion. `None` means
    /// RGB→Gray is forbidden (returns `ConvertError::RgbToGray`).
    luma: Option<LumaCoefficients>,
    /// How to handle values outside the destination's gamut.
    gamut_policy: GamutPolicy,
    sdr_white: Option<crate::hdr::DiffuseWhite>,
    hdr_peak: crate::hdr::HdrPeak,
    sdr_peak: Option<crate::hdr::Nits>,
    hlg_rendering: Option<crate::hdr::HlgRendering>,
    rendering_intent: Option<crate::color::RenderingIntent>,
}

impl ConvertOptions {
    /// Retain alpha and color channels and forbid depth reduction.
    ///
    /// Alpha removal, RGB-to-gray conversion and depth reduction return errors.
    /// Gray-to-RGB expansion remains permitted. Requested color and alpha
    /// association conversions may still change sample values; this is not an
    /// exactness guarantee. Float transfer clipping remains enabled, matching
    /// the historical preset. Select `with_gamut_policy(GamutPolicy::PreserveExtended)` explicitly
    /// when an extended-range conversion is required.
    ///
    /// For reversible representation changes only, use
    /// `zenpixels_convert::ConversionSpec::preserve_samples`.
    pub const fn retain_channels() -> Self {
        Self {
            alpha_policy: AlphaPolicy::Forbid,
            depth_policy: DepthPolicy::Forbid,
            luma: None,
            gamut_policy: GamutPolicy::Clip,
            sdr_white: None,
            hdr_peak: crate::hdr::HdrPeak::Measured,
            sdr_peak: None,
            hlg_rendering: None,
            rendering_intent: None,
        }
    }

    /// Explicit ICC rendering intent override for CMS-backed transforms.
    ///
    /// When `None`, CMS backends honor a rendering intent carried on the source
    /// [`ColorContext`](crate::ColorContext) / [`ParametricColor`](crate::ParametricColor),
    /// falling back to [`RenderingIntent::RelativeColorimetric`](crate::RenderingIntent::RelativeColorimetric).
    pub const fn rendering_intent(&self) -> Option<crate::color::RenderingIntent> {
        self.rendering_intent
    }

    /// Override the ICC rendering intent for CMS-backed transforms.
    pub const fn with_rendering_intent(mut self, intent: crate::color::RenderingIntent) -> Self {
        self.rendering_intent = Some(intent);
        self
    }

    /// Explicit scene-to-display rendering conditions for HLG boundaries.
    pub const fn hlg_rendering(&self) -> Option<crate::hdr::HlgRendering> {
        self.hlg_rendering
    }
    /// Opt into HLG rendering. Native identity transport remains independent of this policy.
    pub const fn with_hlg_rendering(mut self, rendering: crate::hdr::HlgRendering) -> Self {
        self.hlg_rendering = Some(rendering);
        self
    }

    /// Explicit luminance represented by relative-linear 1.0 / SDR white.
    pub const fn sdr_white(&self) -> Option<crate::hdr::DiffuseWhite> {
        self.sdr_white
    }
    /// Source peak selection; buffers measure by default.
    pub const fn hdr_peak(&self) -> crate::hdr::HdrPeak {
        self.hdr_peak
    }
    /// Explicit target SDR peak; tone mapping refuses when it is absent.
    pub const fn sdr_peak(&self) -> Option<crate::hdr::Nits> {
        self.sdr_peak
    }
    /// Set the SDR-white anchor, overriding a source color context.
    pub const fn with_sdr_white(mut self, white: crate::hdr::DiffuseWhite) -> Self {
        self.sdr_white = Some(white);
        self
    }
    /// Choose measured, declared or explicit source peak luminance.
    pub const fn with_hdr_peak(mut self, peak: crate::hdr::HdrPeak) -> Self {
        self.hdr_peak = peak;
        self
    }
    /// Set target SDR peak luminance for tone mapping.
    pub const fn with_sdr_peak(mut self, peak: crate::hdr::Nits) -> Self {
        self.sdr_peak = Some(peak);
        self
    }

    /// How alpha removal is handled.
    pub const fn alpha_policy(&self) -> AlphaPolicy {
        self.alpha_policy
    }

    /// How depth reduction is handled.
    pub const fn depth_policy(&self) -> DepthPolicy {
        self.depth_policy
    }

    /// Luma coefficients for RGB to gray, or `None` when that's forbidden.
    pub const fn luma(&self) -> Option<LumaCoefficients> {
        self.luma
    }

    /// Treatment of extended-range or out-of-gamut values.
    pub const fn gamut_policy(&self) -> GamutPolicy {
        self.gamut_policy
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

    /// Choose how to handle extended-range or out-of-gamut values.
    #[must_use]
    pub const fn with_gamut_policy(mut self, policy: GamutPolicy) -> Self {
        self.gamut_policy = policy;
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
    fn retain_channels_does_not_promise_exact_samples() {
        let options = ConvertOptions::retain_channels();
        assert_eq!(options.alpha_policy, AlphaPolicy::Forbid);
        assert_eq!(options.depth_policy, DepthPolicy::Forbid);
        assert_eq!(options.luma, None);
        assert_eq!(options.gamut_policy(), GamutPolicy::Clip);
    }

    #[test]
    fn alpha_policy_variants() {
        let discard = AlphaPolicy::DiscardIfOpaque;
        let unchecked = AlphaPolicy::DiscardUnchecked;
        let composite = AlphaPolicy::CompositeOnto {
            background: crate::Color::srgb8(255, 255, 255),
            reference_white: None,
        };
        let forbid = AlphaPolicy::Forbid;

        assert_ne!(discard, unchecked);
        assert_ne!(composite, forbid);

        let composite2 = AlphaPolicy::CompositeOnto {
            background: crate::Color::srgb8(255, 255, 255),
            reference_white: None,
        };
        assert_eq!(composite, composite2);

        let composite_diff = AlphaPolicy::CompositeOnto {
            background: crate::Color::srgb8(0, 0, 0),
            reference_white: None,
        };
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
            alpha_policy: AlphaPolicy::DiscardUnchecked,
            depth_policy: DepthPolicy::Round,
            luma: Some(LumaCoefficients::Bt709),
            gamut_policy: GamutPolicy::Clip,
            sdr_white: None,
            hdr_peak: crate::hdr::HdrPeak::Measured,
            sdr_peak: None,
            hlg_rendering: None,
            rendering_intent: None,
        };
        #[allow(clippy::clone_on_copy)]
        let opts2 = opts.clone();
        assert_eq!(opts, opts2);
        assert_eq!(opts.rendering_intent(), None);
        assert_eq!(
            opts.with_rendering_intent(crate::RenderingIntent::Perceptual)
                .rendering_intent(),
            Some(crate::RenderingIntent::Perceptual)
        );
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
            alpha_policy: AlphaPolicy::Forbid,
            depth_policy: DepthPolicy::Forbid,
            luma: None,
            gamut_policy: GamutPolicy::Clip,
            sdr_white: None,
            hdr_peak: crate::hdr::HdrPeak::Measured,
            sdr_peak: None,
            hlg_rendering: None,
            rendering_intent: None,
        };
        let mut h = std::hash::DefaultHasher::new();
        opts.hash(&mut h);
        // Just verify it doesn't panic.
        let _ = h.finish();
    }
}
