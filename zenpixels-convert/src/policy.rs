//! Single cohesive conversion policy (`ConvertPolicy`) governing depth, alpha,
//! gamut, signal range, HDR white anchor, HLG rendering, tone mapping, CMS, and SIMD.

use alloc::sync::Arc;
use zenpixels::{Color, DiffuseWhite, HlgRendering, Nits, SampleFacts};

use crate::accelerators::Accelerators;
use crate::cms::CmsBackend;
use crate::tonemap::ToneMapper;

/// Policy when destination has lower numeric precision than source (e.g. `F32 → U8`, `U16 → U8`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DepthPolicy {
    /// Refuse any precision reduction (`ConvertError::DepthReductionForbidden`).
    Forbid,
    /// Round to nearest integer (`floor(v * max + 0.5)`). Default for display/lossy output.
    #[default]
    Round,
    /// Truncate toward zero (`floor(v * max)` / high-byte drop for `U16 → U8`).
    Truncate,
    /// Ordered 4×4 Bayer dithering before quantization (reduces banding in 8-bit gradients).
    OrderedDither,
}

/// Policy when destination drops an alpha channel (`RGBA → RGB`, `GrayA → Gray`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum AlphaPolicy {
    /// Refuse dropping any alpha channel unless destination also carries alpha (`ConvertError::AlphaLossForbidden`).
    #[default]
    Forbid,
    /// Allow dropping alpha only when proven opaque (`AlphaMode::Opaque` or `SampleFacts::exact.all_opaque`).
    DiscardIfOpaque,
    /// Unconditionally drop the alpha channel without compositing.
    DiscardUnchecked,
    /// Composite transparent pixels over an explicit linear-light matte [`Color`] before dropping alpha.
    CompositeOnto(Color),
}

/// Policy when converting from a wider gamut or HDR range into a narrower or bounded destination.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GamutPolicy {
    /// Refuse any gamut narrowing or out-of-range clipping (`ConvertError::GamutLossForbidden`).
    Forbid,
    /// Preserve extended `< 0.0` and `> 1.0` values on floating-point targets (odd-symmetric transfer curves).
    PreserveExtended,
    /// Hard-clamp out-of-gamut channels to `[0.0, 1.0]`.
    #[default]
    Clip,
    /// Smooth OKLch chroma roll-off preserving hue and lightness (requires `oklab` feature).
    SoftCompress,
}

/// Policy when source and destination disagree on [`SignalRange`](zenpixels::SignalRange) (`Full ↔ Narrow`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RangePolicy {
    /// Refuse silent Full↔Narrow range conversion (`ConvertError::RangeConversionForbidden`).
    Forbid,
    /// Explicitly expand narrow studio swing (`16..=235` / `4096..=60160`) ↔ full swing (`0..=255` / `0..=65535`).
    #[default]
    Convert,
}

/// Luminance weights used when converting `RGB → Gray`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum LumaCoefficients {
    /// Derive linear luminance weights from the source RGB primaries (`Bt709`, `DisplayP3`, `Bt2020`, `AdobeRgb`).
    #[default]
    FromSourcePrimaries,
    /// ITU-R BT.709 (`0.2126, 0.7152, 0.0722`).
    Bt709,
    /// ITU-R BT.601 (`0.299, 0.587, 0.114`).
    Bt601,
    /// ITU-R BT.2020 (`0.2627, 0.6780, 0.0593`).
    Bt2020,
    /// Custom linear luminance weights `[Yr, Yg, Yb]`.
    Custom([f32; 3]),
}

/// Domain in which `RGB → Gray` luminance weights are evaluated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LumaDomain {
    /// Evaluate luma weights directly on transfer-encoded RGB samples (`Y' = w_r·R' + w_g·G' + w_b·B'`).
    ///
    /// Matches the JPEG/video Y' convention and `zenpixels 0.2/0.4` defaults, preserves
    /// neutral gray (`R' = G' = B' → Y' = R'`) bit-for-bit, and enables SIMD integer
    /// `U8` fast paths.
    #[default]
    Encoded,
    /// Linearize RGB via the source EOTF, compute physical luminance (`Y = w_r·R + w_g·G + w_b·B`),
    /// then apply the destination OETF.
    LinearLight,
}

/// Cohesive conversion policy controlling precision, alpha, gamut, signal range,
/// HDR anchors, tone mapping, CMS fallback, and SIMD acceleration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ConvertPolicy {
    /// Depth/quantization behavior.
    pub depth: DepthPolicy,
    /// Alpha removal / compositing behavior.
    pub alpha: AlphaPolicy,
    /// Gamut narrowing / extended-range behavior.
    pub gamut: GamutPolicy,
    /// Explicit Full ↔ Narrow range conversion policy.
    pub range: RangePolicy,
    /// RGB → Gray luma weights.
    pub luma: LumaCoefficients,
    /// Domain in which `RGB → Gray` weights are evaluated (`Encoded` Y' default vs `LinearLight` Y).
    pub luma_domain: LumaDomain,
    /// Relative-linear `1.0` diffuse white anchor (required when crossing PQ ↔ relative light unless carried on `ColorSpace`).
    pub sdr_white: Option<DiffuseWhite>,
    /// Target SDR peak luminance in cd/m² for HDR → SDR tone mapping (default `100.0` cd/m²).
    pub sdr_peak: Nits,
    /// Explicit HLG display rendering override when source/dest carries `HlgMode::Scene`.
    pub hlg_rendering: Option<HlgRendering>,
    /// Optional custom HDR → SDR tone mapper (defaults to built-in `Bt2446A` when `bt2446a` feature is enabled).
    pub tone_mapper: Option<Arc<dyn ToneMapper>>,
    /// Optional CMS backend for arbitrary `ColorSpace::Icc` or `ColorSpace::Cmyk` transforms.
    pub cms: Option<Arc<dyn CmsBackend>>,
    /// Optional hermetic accelerator table override (when `None`, consults the process registry / `fast` feature default).
    pub accelerators: Option<Accelerators>,
    /// Caller-asserted source sample facts (merged with buffer facts in `ImagePlan`).
    pub source_facts: SampleFacts,
    /// Whether to apply tone mapping when converting HDR → SDR (`true` in `display_sdr()`, `false` in `lossless()`/`working_linear()`).
    pub tone_map_hdr_to_sdr: bool,
    /// Whether recognized matrix+TRC ICC profiles may substitute directly into native math
    /// (for non-absolute intents) instead of requiring a CMS backend.
    ///
    /// Default is `true` in `permissive()`/`working_linear()`/`display_sdr()`, and `false` in `lossless()`.
    pub allow_icc_substitution: bool,
    /// Whether opt-in fast fixed-point approximations (max error <= 1 LSB vs exact float,
    /// e.g. Q13/Q14 integer LUT U8 sRGB-matrix fast path) may be selected.
    ///
    /// Default is `false` (exact 0-bit parity with the float strip pipeline).
    pub allow_approximate: bool,
}

impl Default for ConvertPolicy {
    /// Defaults to [`ConvertPolicy::lossless`] so callers never silently discard alpha,
    /// narrow bit depth, clip gamut, or rescale signal range without opting in.
    ///
    /// Use [`ConvertPolicy::permissive`], [`ConvertPolicy::working_linear`], or
    /// [`ConvertPolicy::display_sdr`] when lossy conversion is desired.
    fn default() -> Self {
        Self::lossless()
    }
}

impl ConvertPolicy {
    /// Strict lossless-permission policy: forbids depth reduction, alpha loss, gamut clipping,
    /// RGB→Gray chroma loss (unless proven grayscale), and range scaling.
    ///
    /// Allows lossless widening (`U8 → U16 → F32`), channel reordering (`RGB ↔ BGR`),
    /// adding opaque alpha, and floating-point extended-gamut preservation. Note that
    /// `lossless()` describes **loss-permission settings** rather than mathematical
    /// invertibility of arbitrary floating-point color transforms.
    #[must_use]
    pub fn lossless() -> Self {
        Self {
            depth: DepthPolicy::Forbid,
            alpha: AlphaPolicy::Forbid,
            gamut: GamutPolicy::PreserveExtended,
            range: RangePolicy::Forbid,
            luma: LumaCoefficients::FromSourcePrimaries,
            luma_domain: LumaDomain::Encoded,
            sdr_white: None,
            sdr_peak: Nits::new(100.0),
            hlg_rendering: None,
            tone_mapper: None,
            cms: None,
            accelerators: None,
            source_facts: SampleFacts::NONE,
            tone_map_hdr_to_sdr: false,
            allow_icc_substitution: false,
            allow_approximate: false,
        }
    }

    /// Linear-light working-space policy: preserves extended values, defaults diffuse white
    /// to ITU-R BT.2408 (`203 cd/m²`, override via `.with_sdr_white(...)`), and disables tone mapping.
    ///
    /// Use [`ConvertPolicy::permissive`] or [`ConvertPolicy::lossless`] when unanchored
    /// PQ ↔ relative routes must fail with [`ConvertError::MissingHdrAnchor`](crate::ConvertError::MissingHdrAnchor)
    /// unless a diffuse white is explicitly provided.
    #[inline]
    #[must_use]
    pub fn working_linear() -> Self {
        Self {
            depth: DepthPolicy::Round,
            alpha: AlphaPolicy::DiscardIfOpaque,
            gamut: GamutPolicy::PreserveExtended,
            range: RangePolicy::Convert,
            luma: LumaCoefficients::FromSourcePrimaries,
            luma_domain: LumaDomain::Encoded,
            sdr_white: Some(DiffuseWhite::BT2408),
            sdr_peak: Nits::new(100.0),
            hlg_rendering: None,
            tone_mapper: None,
            cms: None,
            accelerators: None,
            source_facts: SampleFacts::NONE,
            tone_map_hdr_to_sdr: false,
            allow_icc_substitution: true,
            allow_approximate: false,
        }
    }

    /// Display-referred SDR output policy: enables HDR → SDR tone mapping, rounds depth, converts narrow range, and clips/compresses gamut.
    #[must_use]
    pub fn display_sdr(white: DiffuseWhite, sdr_peak: Nits) -> Self {
        Self {
            depth: DepthPolicy::Round,
            alpha: AlphaPolicy::CompositeOnto(Color::BLACK),
            gamut: GamutPolicy::Clip,
            range: RangePolicy::Convert,
            luma: LumaCoefficients::FromSourcePrimaries,
            luma_domain: LumaDomain::Encoded,
            sdr_white: Some(white),
            sdr_peak,
            hlg_rendering: Some(HlgRendering::REFERENCE_1000_NIT),
            tone_mapper: None,
            cms: None,
            accelerators: None,
            source_facts: SampleFacts::NONE,
            tone_map_hdr_to_sdr: true,
            allow_icc_substitution: true,
            allow_approximate: false,
        }
    }

    /// Permissive general-purpose policy (rounds depth, clips gamut, converts narrow range, discards alpha unchecked unless configured).
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            depth: DepthPolicy::Round,
            alpha: AlphaPolicy::DiscardUnchecked,
            gamut: GamutPolicy::Clip,
            range: RangePolicy::Convert,
            luma: LumaCoefficients::FromSourcePrimaries,
            luma_domain: LumaDomain::Encoded,
            sdr_white: None,
            sdr_peak: Nits::new(100.0),
            hlg_rendering: None,
            tone_mapper: None,
            cms: None,
            accelerators: None,
            source_facts: SampleFacts::NONE,
            tone_map_hdr_to_sdr: false,
            allow_icc_substitution: true,
            allow_approximate: false,
        }
    }

    /// Set depth reduction policy (`Forbid`, `Round`, `Truncate`, `OrderedDither`).
    #[inline]
    #[must_use]
    pub fn with_depth(mut self, depth: DepthPolicy) -> Self {
        self.depth = depth;
        self
    }

    /// Set alpha removal/compositing policy.
    #[inline]
    #[must_use]
    pub fn with_alpha(mut self, alpha: AlphaPolicy) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set gamut narrowing / extended-range policy.
    #[inline]
    #[must_use]
    pub fn with_gamut(mut self, gamut: GamutPolicy) -> Self {
        self.gamut = gamut;
        self
    }

    /// Set explicit Full ↔ Narrow signal range conversion policy.
    #[inline]
    #[must_use]
    pub fn with_range(mut self, range: RangePolicy) -> Self {
        self.range = range;
        self
    }

    /// Set RGB → Gray luminance weights.
    #[inline]
    #[must_use]
    pub fn with_luma(mut self, luma: LumaCoefficients) -> Self {
        self.luma = luma;
        self
    }

    /// Set RGB → Gray evaluation domain ([`LumaDomain::Encoded`] Y' default vs [`LumaDomain::LinearLight`] Y).
    #[inline]
    #[must_use]
    pub fn with_luma_domain(mut self, domain: LumaDomain) -> Self {
        self.luma_domain = domain;
        self
    }

    /// Set relative-linear `1.0` diffuse white anchor (e.g. `DiffuseWhite::BT2408` = 203 nits).
    #[inline]
    #[must_use]
    pub fn with_sdr_white(mut self, white: DiffuseWhite) -> Self {
        self.sdr_white = Some(white);
        self
    }

    /// Set target SDR peak luminance in cd/m² for HDR → SDR tone mapping.
    #[inline]
    #[must_use]
    pub fn with_sdr_peak(mut self, peak: Nits) -> Self {
        self.sdr_peak = peak;
        self
    }

    /// Set explicit BT.2100-3 HLG display rendering conditions.
    #[inline]
    #[must_use]
    pub fn with_hlg_rendering(mut self, rendering: HlgRendering) -> Self {
        self.hlg_rendering = Some(rendering);
        self
    }

    /// Enable or disable HDR → SDR tone mapping.
    #[inline]
    #[must_use]
    pub fn with_tone_mapping(mut self, enable: bool) -> Self {
        self.tone_map_hdr_to_sdr = enable;
        self
    }

    /// Attach a custom [`ToneMapper`] and enable HDR → SDR tone mapping.
    #[inline]
    #[must_use]
    pub fn with_tone_mapper(mut self, mapper: Arc<dyn ToneMapper>) -> Self {
        self.tone_mapper = Some(mapper);
        self.tone_map_hdr_to_sdr = true;
        self
    }

    /// Attach a [`CmsBackend`] for arbitrary ICC profiles and CMYK conversions.
    #[inline]
    #[must_use]
    pub fn with_cms(mut self, cms: Arc<dyn CmsBackend>) -> Self {
        self.cms = Some(cms);
        self
    }

    /// Override the SIMD accelerator table hermetically for plans built with this policy.
    #[inline]
    #[must_use]
    pub fn with_accelerators(mut self, accel: Accelerators) -> Self {
        self.accelerators = Some(accel);
        self
    }

    /// Supply proven source [`SampleFacts`] for row-level plans without buffer headers
    /// (merged with runtime buffer facts in [`ImagePlan`](crate::ImagePlan)).
    #[inline]
    #[must_use]
    pub fn with_source_facts(mut self, facts: SampleFacts) -> Self {
        self.source_facts = facts;
        self
    }

    /// Control whether recognized matrix+TRC ICC profiles may substitute into native math.
    #[inline]
    #[must_use]
    pub fn with_icc_substitution(mut self, allow: bool) -> Self {
        self.allow_icc_substitution = allow;
        self
    }

    /// Control whether fast fixed-point approximations (max error <= 1 LSB vs exact float)
    /// may be used on supported U8 color-matrix routes (default `false`).
    #[inline]
    #[must_use]
    pub fn with_approximate(mut self, allow: bool) -> Self {
        self.allow_approximate = allow;
        self
    }
}
