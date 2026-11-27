//! Clean Color Management System (`CmsBackend`) trait, named/quoted [`CmsBinding`],
//! shared [`ColorServices`] / [`CmsChoice`], and optional `MoxCms` / `Lcms` adapters.
//!
//! Only invoked when converting arbitrary [`ColorSpace::Icc`](zenpixels::ColorSpace::Icc)
//! profiles (that don't match a safe matrix+TRC fingerprint) or [`ColorSpace::Cmyk`](zenpixels::ColorSpace::Cmyk).

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::fmt::Debug;
use whereat::At;
use zenpixels::PixelFormat;

use crate::error::ConvertError;
use crate::tonemap::ToneMapper;

/// Shareable, stateless row-level CMS transform executor (`Send + Sync`).
pub trait CmsTransform: Send + Sync {
    /// Transform one row of `width_pixels` pixels from `src` bytes to `dst` bytes.
    fn transform_row(
        &self,
        src: &[u8],
        dst: &mut [u8],
        width_pixels: u32,
    ) -> Result<(), At<ConvertError>>;
}

/// Pluggable ICC / CMYK Color Management System backend.
pub trait CmsBackend: Send + Sync + Debug {
    /// Fast, allocation-free check whether this backend supports the `(src, dst)`
    /// layouts, sample types, and color families (used by [`Negotiate`](crate::Negotiate)
    /// to avoid selecting unbuildable candidates).
    fn supports_formats(&self, _src: &PixelFormat, _dst: &PixelFormat) -> bool {
        true
    }

    /// Build a row transform from `src` to `dst`.
    fn build_transform(
        &self,
        src: &PixelFormat,
        dst: &PixelFormat,
    ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>>;
}

/// Conservative capability, memory, stack, and concurrency quote for a [`CmsBinding`].
///
/// Hosts and routing planners inspect this quote before admitting a CMS transform
/// against memory limits, constrained thread stacks (such as 256 KiB IIS worker stacks),
/// or concurrency pools. `None` means the bound is unknown/unproven for arbitrary
/// profiles and cannot authorize a strict hard limit without route qualification or
/// explicit best-effort policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct CmsQuote {
    /// Supports `SampleType::U8` transforms.
    pub supports_u8: bool,
    /// Supports `SampleType::U16` transforms.
    pub supports_u16: bool,
    /// Supports `SampleType::F16` transforms.
    pub supports_f16: bool,
    /// Supports `SampleType::F32` transforms.
    pub supports_f32: bool,
    /// Supports `ColorSpace::Cmyk` ink transforms.
    pub supports_cmyk: bool,
    /// Supports `RenderingIntent::AbsoluteColorimetric` (media-white / `chad` un-adaptation).
    pub supports_absolute_colorimetric: bool,
    /// Supports ICC v4 float `D2B*`/`B2D*` MPE pipelines.
    pub supports_float_mpe: bool,
    /// Proven conservative heap bytes allocated while building one transform (`None` = unknown).
    pub build_memory_bytes: Option<u64>,
    /// Proven conservative heap bytes retained per live [`CmsTransform`] (`None` = unknown).
    pub transform_memory_bytes: Option<u64>,
    /// Proven conservative heap bytes allocated per `transform_row` call (`Some(0)` = zero-alloc apply; `None` = unknown).
    pub apply_scratch_bytes: Option<u64>,
    /// Proven conservative peak stack bytes consumed by `build_transform` / `transform_row` (`None` = unknown).
    pub peak_stack_bytes: Option<u32>,
    /// Whether `build_transform` and `transform_row` are qualified for constrained 256 KiB stacks.
    pub small_stack_qualified: bool,
    /// Whether concurrent `transform_row` calls across threads are lock-free / contention-free.
    pub concurrent_apply: bool,
}

impl CmsQuote {
    /// Conservative empty quote with all capabilities disabled, unknown bounds (`None`),
    /// and `small_stack_qualified = false`.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            supports_u8: false,
            supports_u16: false,
            supports_f16: false,
            supports_f32: false,
            supports_cmyk: false,
            supports_absolute_colorimetric: false,
            supports_float_mpe: false,
            build_memory_bytes: None,
            transform_memory_bytes: None,
            apply_scratch_bytes: None,
            peak_stack_bytes: None,
            small_stack_qualified: false,
            concurrent_apply: false,
        }
    }

    /// Set `U8` support.
    #[inline]
    #[must_use]
    pub const fn with_u8(mut self, yes: bool) -> Self {
        self.supports_u8 = yes;
        self
    }

    /// Set `U16` support.
    #[inline]
    #[must_use]
    pub const fn with_u16(mut self, yes: bool) -> Self {
        self.supports_u16 = yes;
        self
    }

    /// Set `F16` support.
    #[inline]
    #[must_use]
    pub const fn with_f16(mut self, yes: bool) -> Self {
        self.supports_f16 = yes;
        self
    }

    /// Set `F32` support.
    #[inline]
    #[must_use]
    pub const fn with_f32(mut self, yes: bool) -> Self {
        self.supports_f32 = yes;
        self
    }

    /// Set `ColorSpace::Cmyk` support.
    #[inline]
    #[must_use]
    pub const fn with_cmyk(mut self, yes: bool) -> Self {
        self.supports_cmyk = yes;
        self
    }

    /// Set `RenderingIntent::AbsoluteColorimetric` support.
    #[inline]
    #[must_use]
    pub const fn with_absolute_colorimetric(mut self, yes: bool) -> Self {
        self.supports_absolute_colorimetric = yes;
        self
    }

    /// Set ICC v4 float MPE (`D2B*`/`B2D*`) support.
    #[inline]
    #[must_use]
    pub const fn with_float_mpe(mut self, yes: bool) -> Self {
        self.supports_float_mpe = yes;
        self
    }

    /// Set proven conservative heap bytes used while building one transform (`None` = unknown).
    #[inline]
    #[must_use]
    pub const fn with_build_memory_bytes(mut self, bytes: Option<u64>) -> Self {
        self.build_memory_bytes = bytes;
        self
    }

    /// Set proven conservative heap bytes retained per live [`CmsTransform`] (`None` = unknown).
    #[inline]
    #[must_use]
    pub const fn with_transform_memory_bytes(mut self, bytes: Option<u64>) -> Self {
        self.transform_memory_bytes = bytes;
        self
    }

    /// Set proven conservative heap bytes allocated per `transform_row` call (`Some(0)` = zero-alloc apply).
    #[inline]
    #[must_use]
    pub const fn with_apply_scratch_bytes(mut self, bytes: Option<u64>) -> Self {
        self.apply_scratch_bytes = bytes;
        self
    }

    /// Set proven conservative peak stack bytes used by `build_transform` and `transform_row`.
    #[inline]
    #[must_use]
    pub const fn with_peak_stack_bytes(mut self, bytes: Option<u32>) -> Self {
        self.peak_stack_bytes = bytes;
        self
    }

    /// Declare whether this backend is qualified for constrained 256 KiB thread stacks.
    #[inline]
    #[must_use]
    pub const fn with_small_stack_qualified(mut self, yes: bool) -> Self {
        self.small_stack_qualified = yes;
        self
    }

    /// Declare whether concurrent `transform_row` calls across threads are contention-free.
    #[inline]
    #[must_use]
    pub const fn with_concurrent_apply(mut self, yes: bool) -> Self {
        self.concurrent_apply = yes;
        self
    }
}

/// Named, versioned [`CmsBackend`] binding with conservative resource and capability quotes.
///
/// Swap CMS implementations (such as `MoxCms`, pure-Rust `Lcms`, C `lcms2`, or a custom
/// provider) across `zenpixels-convert`, `zencodec-render`, `zencodecs`, `zenpipe`, and
/// hosts by substituting this binding alone.
#[derive(Clone, Debug)]
pub struct CmsBinding {
    id: Box<str>,
    revision: Box<str>,
    backend: Arc<dyn CmsBackend>,
    quote: CmsQuote,
}

impl CmsBinding {
    /// Construct a named, versioned CMS binding.
    ///
    /// Prefer stable reverse-DNS provider identifiers (e.g. `"org.imazen.cms.moxcms"`,
    /// `"org.imazen.cms.lcms"`) and a revision string that changes whenever transform output
    /// or qualification changes so downstream route caches invalidate deterministically.
    #[must_use]
    pub fn new(
        id: impl Into<Box<str>>,
        revision: impl Into<Box<str>>,
        backend: Arc<dyn CmsBackend>,
        quote: CmsQuote,
    ) -> Self {
        Self {
            id: id.into(),
            revision: revision.into(),
            backend,
            quote,
        }
    }

    /// Stable provider identifier (e.g. `"org.imazen.cms.moxcms"`).
    #[inline]
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Provider revision / version string for deterministic cache keys and diagnostics.
    #[inline]
    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Cache identity pair `(id, revision)`.
    #[inline]
    #[must_use]
    pub fn cache_key(&self) -> (&str, &str) {
        (&self.id, &self.revision)
    }

    /// Underlying [`CmsBackend`] implementation.
    #[inline]
    #[must_use]
    pub fn backend(&self) -> &Arc<dyn CmsBackend> {
        &self.backend
    }

    /// Conservative resource and capability quote.
    #[inline]
    #[must_use]
    pub const fn quote(&self) -> CmsQuote {
        self.quote
    }
}

/// Explicit color-management provider selection for a conversion or pipeline run.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum CmsChoice {
    /// Inherit the parent or host default CMS binding when resolved at run binding.
    #[default]
    Inherit,
    /// Forbid both ICC CMS evaluation and inferred matrix+TRC ICC substitution.
    ///
    /// Validated ICC profile carriage (pass-through when source and destination carry
    /// equivalent profiles) and pure known-space math remain permitted.
    StrictNoIcc,
    /// Use this exact named [`CmsBinding`] for arbitrary ICC and CMYK conversions.
    Use(CmsBinding),
}

impl CmsChoice {
    /// Returns `true` when this choice is concrete (`StrictNoIcc` or `Use`), not `Inherit`.
    #[inline]
    #[must_use]
    pub const fn is_frozen(&self) -> bool {
        !matches!(self, Self::Inherit)
    }

    /// Borrow the bound [`CmsBinding`] if `self` is [`CmsChoice::Use`].
    #[inline]
    #[must_use]
    pub fn binding(&self) -> Option<&CmsBinding> {
        match self {
            Self::Use(b) => Some(b),
            _ => None,
        }
    }
}

/// Shared color-management and tone-mapping service bundle.
///
/// Hosts configure a default [`ColorServices`] once, resolve per-run overrides via
/// [`resolve_against`](Self::resolve_against) or [`freeze_or_strict`](Self::freeze_or_strict),
/// and stamp the frozen services onto [`ConvertPolicy::with_color_services`](crate::ConvertPolicy::with_color_services).
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ColorServices {
    /// CMS provider choice (`Inherit`, `StrictNoIcc`, or `Use(CmsBinding)`).
    pub cms: CmsChoice,
    /// Optional custom HDR → SDR tone mapper shared across conversions.
    pub tone_mapper: Option<Arc<dyn ToneMapper>>,
}

impl ColorServices {
    /// Inherit both CMS selection and tone mapper from the enclosing scope.
    #[inline]
    #[must_use]
    pub fn inherit() -> Self {
        Self::default()
    }

    /// Forbid ICC CMS evaluation and inferred ICC substitution while permitting
    /// validated profile carriage and known-space math.
    #[inline]
    #[must_use]
    pub fn strict_no_icc() -> Self {
        Self {
            cms: CmsChoice::StrictNoIcc,
            tone_mapper: None,
        }
    }

    /// Create a service bundle bound to a specific [`CmsBinding`].
    #[inline]
    #[must_use]
    pub fn with_cms_binding(binding: CmsBinding) -> Self {
        Self {
            cms: CmsChoice::Use(binding),
            tone_mapper: None,
        }
    }

    /// Replace the [`CmsChoice`].
    #[inline]
    #[must_use]
    pub fn with_cms(mut self, choice: CmsChoice) -> Self {
        self.cms = choice;
        self
    }

    /// Attach a custom [`ToneMapper`].
    #[inline]
    #[must_use]
    pub fn with_tone_mapper(mut self, mapper: Arc<dyn ToneMapper>) -> Self {
        self.tone_mapper = Some(mapper);
        self
    }

    /// Returns `true` when [`cms`](Self::cms) is no longer [`CmsChoice::Inherit`].
    #[inline]
    #[must_use]
    pub const fn is_frozen(&self) -> bool {
        self.cms.is_frozen()
    }

    /// Borrow the active [`CmsChoice`].
    #[inline]
    #[must_use]
    pub const fn cms_choice(&self) -> &CmsChoice {
        &self.cms
    }

    /// Borrow the selected [`CmsBinding`], if [`CmsChoice::Use`].
    #[inline]
    #[must_use]
    pub fn cms_binding(&self) -> Option<&CmsBinding> {
        self.cms.binding()
    }

    /// Borrow the configured [`ToneMapper`], if any.
    #[inline]
    #[must_use]
    pub fn tone_mapper(&self) -> Option<&Arc<dyn ToneMapper>> {
        self.tone_mapper.as_ref()
    }

    /// Resolve [`CmsChoice::Inherit`] and missing `tone_mapper` against `parent`.
    ///
    /// Explicit [`CmsChoice::StrictNoIcc`] and [`CmsChoice::Use`] choices on `self`
    /// override `parent`.
    #[must_use]
    pub fn resolve_against(&self, parent: &Self) -> Self {
        let cms = match &self.cms {
            CmsChoice::Inherit => parent.cms.clone(),
            explicit => explicit.clone(),
        };
        let tone_mapper = self
            .tone_mapper
            .as_ref()
            .or(parent.tone_mapper.as_ref())
            .map(Arc::clone);
        Self { cms, tone_mapper }
    }

    /// Freeze inheritance for a run: [`CmsChoice::Inherit`] resolves to `Use(default_cms)`
    /// if provided, or [`CmsChoice::StrictNoIcc`] if `None`.
    #[must_use]
    pub fn freeze_or_strict(&self, default_cms: Option<&CmsBinding>) -> Self {
        let cms = match &self.cms {
            CmsChoice::Inherit => match default_cms {
                Some(b) => CmsChoice::Use(b.clone()),
                None => CmsChoice::StrictNoIcc,
            },
            explicit => explicit.clone(),
        };
        Self {
            cms,
            tone_mapper: self.tone_mapper.as_ref().map(Arc::clone),
        }
    }
}

#[cfg(feature = "cms-moxcms")]
mod moxcms_impl {
    use super::*;
    use crate::error::err_at;
    use zenpixels::{
        AlphaMode, ChannelOrder, ColorSpace, Primaries, RenderingIntent, SampleType, SignalRange,
        Transfer,
    };

    const CHUNK_PX: usize = 64;

    /// Pure-Rust `moxcms` ICC CMS backend.
    ///
    /// Note: `moxcms` 0.9 does not implement ICC.1 §D.3 media-white / `chad` un-adaptation for
    /// [`RenderingIntent::AbsoluteColorimetric`]; rather than silently substituting
    /// `RelativeColorimetric`, [`MoxCms`] refuses `AbsoluteColorimetric` with
    /// [`ConvertError::UnsupportedRoute`]. Use [`crate::Lcms`] for `AbsoluteColorimetric`.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct MoxCms;

    impl MoxCms {
        /// Canonical named, conservatively quoted [`CmsBinding`] for `moxcms`
        /// (`"org.imazen.cms.moxcms"`).
        ///
        /// Heap build/transform size and peak stack depend on profile structure and are left
        /// unknown (`None` / `small_stack_qualified = false`) until qualified per route.
        #[must_use]
        pub fn binding() -> CmsBinding {
            CmsBinding::new(
                "org.imazen.cms.moxcms",
                concat!("moxcms-0.9/zenpixels-convert-", env!("CARGO_PKG_VERSION")),
                Arc::new(Self),
                CmsQuote::new()
                    .with_u8(true)
                    .with_u16(true)
                    .with_f16(false)
                    .with_f32(true)
                    .with_cmyk(true)
                    .with_absolute_colorimetric(false)
                    .with_float_mpe(false)
                    .with_concurrent_apply(true),
            )
        }
    }

    fn to_mox_intent(intent: RenderingIntent) -> moxcms::RenderingIntent {
        match intent {
            RenderingIntent::Perceptual => moxcms::RenderingIntent::Perceptual,
            RenderingIntent::RelativeColorimetric => moxcms::RenderingIntent::RelativeColorimetric,
            RenderingIntent::Saturation => moxcms::RenderingIntent::Saturation,
            RenderingIntent::AbsoluteColorimetric => moxcms::RenderingIntent::AbsoluteColorimetric,
            _ => moxcms::RenderingIntent::Perceptual,
        }
    }

    fn to_mox_layout(order: ChannelOrder) -> Option<moxcms::Layout> {
        match order {
            ChannelOrder::Gray => Some(moxcms::Layout::Gray),
            ChannelOrder::GrayAlpha => Some(moxcms::Layout::GrayAlpha),
            ChannelOrder::Rgb => Some(moxcms::Layout::Rgb),
            ChannelOrder::Rgba => Some(moxcms::Layout::Rgba),
            ChannelOrder::Cmyk | ChannelOrder::CmykInverted => Some(moxcms::Layout::Rgba),
            _ => None,
        }
    }

    /// `moxcms` 0.9 does not evaluate ICC.1 §8.10.2 float `D2B*`/`B2D*` MPE tags.
    /// Decline profiles carrying the active rendering intent's float MPE tag rather than
    /// silently degrading to fallback integer LUTs.
    fn has_float_pipeline(bytes: &[u8], intent: RenderingIntent, is_source: bool) -> bool {
        let sig: &[u8; 4] = match (is_source, intent) {
            (true, RenderingIntent::Perceptual) => b"D2B0",
            (true, RenderingIntent::RelativeColorimetric) => b"D2B1",
            (true, RenderingIntent::Saturation) => b"D2B2",
            (false, RenderingIntent::Perceptual) => b"B2D0",
            (false, RenderingIntent::RelativeColorimetric) => b"B2D1",
            (false, RenderingIntent::Saturation) => b"B2D2",
            _ => return false,
        };
        let Some(header) = bytes.get(..132) else {
            return false;
        };
        let count = u32::from_be_bytes(header[128..132].try_into().unwrap()) as usize;
        let available = (bytes.len() - 132) / 12;
        bytes[132..]
            .as_chunks::<12>()
            .0
            .iter()
            .take(count.min(available))
            .any(|entry| &entry[..4] == sig)
    }

    /// Guard against `moxcms` 0.9 curve-sharing bugs when an RGB profile has only two
    /// adjacent channel TRCs equal (`(r == g || g == b) && r != b`).
    fn validate_channel_curves(profile: &moxcms::ColorProfile) -> bool {
        if profile.color_space == moxcms::DataColorSpace::Rgb
            && let (Some(red), Some(green), Some(blue)) =
                (&profile.red_trc, &profile.green_trc, &profile.blue_trc)
            && (red == green || green == blue)
            && red != blue
        {
            return false;
        }
        true
    }

    /// Explicit ICC profiles must evaluate their own curves rather than letting an
    /// embedded `cicp` tag override the profile's TRC/LUTs.
    fn prefer_icc_curves(profile: &mut moxcms::ColorProfile) {
        profile.cicp = None;
    }

    /// Scale a generated `moxcms` RGB profile's colorants so they sum to the ICC PCS D50 white
    /// (`[0.9642, 1.0, 0.8249]`) rather than CIE D50 (`Z = 0.82521`). Without this, white from
    /// any ICC profile into generated sRGB/P3/BT.2020 shifts blue by 0.04% (`(255, 255, 254)`).
    fn with_pcs_white(mut profile: moxcms::ColorProfile) -> moxcms::ColorProfile {
        const PCS_WHITE: [f64; 3] = [0.9642, 1.0, 0.8249];
        let (r, g, b) = (
            profile.red_colorant,
            profile.green_colorant,
            profile.blue_colorant,
        );
        let scale = [
            PCS_WHITE[0] / (r.x + g.x + b.x),
            PCS_WHITE[1] / (r.y + g.y + b.y),
            PCS_WHITE[2] / (r.z + g.z + b.z),
        ];
        if scale.iter().all(|v| v.is_finite()) {
            for colorant in [
                &mut profile.red_colorant,
                &mut profile.green_colorant,
                &mut profile.blue_colorant,
            ] {
                colorant.x *= scale[0];
                colorant.y *= scale[1];
                colorant.z *= scale[2];
            }
        }
        profile
    }

    fn icc_bytes_of(cs: &ColorSpace) -> Option<&[u8]> {
        match cs {
            ColorSpace::Icc { profile, .. } => Some(profile),
            ColorSpace::Cmyk { icc: Some(b), .. } => Some(b),
            _ => None,
        }
    }

    fn profile_from_color_space(
        cs: &ColorSpace,
    ) -> Result<(moxcms::ColorProfile, Option<RenderingIntent>), At<ConvertError>> {
        match cs {
            ColorSpace::Icc {
                profile, intent, ..
            } => {
                let mut p = moxcms::ColorProfile::new_from_slice(profile)
                    .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                prefer_icc_curves(&mut p);
                if !validate_channel_curves(&p) {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                Ok((p, Some(*intent)))
            }
            ColorSpace::Cmyk { icc, .. } => {
                let bytes = icc
                    .as_ref()
                    .ok_or_else(|| err_at(ConvertError::CmsRequired))?;
                let mut p = moxcms::ColorProfile::new_from_slice(bytes)
                    .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                prefer_icc_curves(&mut p);
                Ok((p, Some(RenderingIntent::RelativeColorimetric)))
            }
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                ..
            } => {
                if *range != SignalRange::Full {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                let p = match (primaries, transfer) {
                    (Primaries::Bt709, Transfer::Srgb) => {
                        moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
                            color_primaries: moxcms::CicpColorPrimaries::Bt709,
                            transfer_characteristics: moxcms::TransferCharacteristics::Srgb,
                            matrix_coefficients: moxcms::MatrixCoefficients::Identity,
                            full_range: true,
                        })
                    }
                    (Primaries::DisplayP3, Transfer::Srgb) => {
                        moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
                            color_primaries: moxcms::CicpColorPrimaries::Smpte432,
                            transfer_characteristics: moxcms::TransferCharacteristics::Srgb,
                            matrix_coefficients: moxcms::MatrixCoefficients::Identity,
                            full_range: true,
                        })
                    }
                    (Primaries::AdobeRgb, Transfer::AdobeRgb) => {
                        moxcms::ColorProfile::new_adobe_rgb()
                    }
                    (Primaries::Bt2020, Transfer::Bt709) => {
                        moxcms::ColorProfile::new_from_cicp(moxcms::CicpProfile {
                            color_primaries: moxcms::CicpColorPrimaries::Bt2020,
                            transfer_characteristics: moxcms::TransferCharacteristics::Bt709,
                            matrix_coefficients: moxcms::MatrixCoefficients::Identity,
                            full_range: true,
                        })
                    }
                    (Primaries::Bt2020, Transfer::Pq) => moxcms::ColorProfile::new_bt2020_pq(),
                    (
                        p_id @ (Primaries::Bt709 | Primaries::DisplayP3 | Primaries::Bt2020),
                        Transfer::Linear,
                    ) => {
                        let mut prof = match p_id {
                            Primaries::Bt709 => moxcms::ColorProfile::new_srgb(),
                            Primaries::DisplayP3 => moxcms::ColorProfile::new_display_p3(),
                            Primaries::Bt2020 => moxcms::ColorProfile::new_bt2020(),
                            _ => unreachable!(),
                        };
                        let lin = moxcms::ToneReprCurve::Parametric(alloc::vec![1.0]);
                        prof.cicp = None;
                        prof.red_trc = Some(lin.clone());
                        prof.green_trc = Some(lin.clone());
                        prof.blue_trc = Some(lin);
                        prof
                    }
                    (Primaries::Bt709, Transfer::Gamma22) => {
                        let mut prof = moxcms::ColorProfile::new_srgb();
                        let g22 = moxcms::ToneReprCurve::Parametric(alloc::vec![2.2]);
                        prof.cicp = None;
                        prof.red_trc = Some(g22.clone());
                        prof.green_trc = Some(g22.clone());
                        prof.blue_trc = Some(g22);
                        prof
                    }
                    // HLG requires BT.2100-3 display OOTF which ICC profiles cannot carry faithfully.
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                Ok((with_pcs_white(p), None))
            }
            ColorSpace::Gray {
                transfer, range, ..
            } => {
                if *range != SignalRange::Full {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                let p = match transfer {
                    Transfer::Srgb => {
                        let mut p = moxcms::ColorProfile::new_gray_with_gamma(1.0);
                        p.gray_trc = moxcms::ColorProfile::new_srgb().red_trc;
                        p
                    }
                    Transfer::Linear => moxcms::ColorProfile::new_gray_with_gamma(1.0),
                    Transfer::Gamma22 => moxcms::ColorProfile::new_gray_with_gamma(2.2),
                    Transfer::AdobeRgb => moxcms::ColorProfile::new_gray_with_gamma(563.0 / 256.0),
                    Transfer::Gamma(g) => moxcms::ColorProfile::new_gray_with_gamma(g.exponent()),
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                Ok((p, None))
            }
            _ => Err(err_at(ConvertError::UnsupportedRoute)),
        }
    }

    /// Compute D50 luminance `Y` of a CMYK profile's black point for relative-colorimetric
    /// black-point compensation (matches `Lcms` / `lcms2` BPC when targeting sRGB).
    fn cmyk_black_point_y(
        cmyk: &moxcms::ColorProfile,
        linear_dst: &moxcms::ColorProfile,
        relative: moxcms::TransformOptions,
    ) -> Option<f32> {
        let perceptual = moxcms::TransformOptions {
            rendering_intent: moxcms::RenderingIntent::Perceptual,
            ..relative
        };
        let to_cmyk = linear_dst
            .create_transform_f32(moxcms::Layout::Rgb, cmyk, moxcms::Layout::Rgba, perceptual)
            .ok()?;
        let to_rgb = cmyk
            .create_transform_f32(
                moxcms::Layout::Rgba,
                linear_dst,
                moxcms::Layout::Rgb,
                relative,
            )
            .ok()?;
        let mut ink = [0.0f32; 4];
        to_cmyk.transform(&[0.0, 0.0, 0.0], &mut ink).ok()?;
        let mut rgb = [0.0f32; 3];
        to_rgb.transform(&ink, &mut rgb).ok()?;
        let y = linear_dst.red_colorant.y * f64::from(rgb[0])
            + linear_dst.green_colorant.y * f64::from(rgb[1])
            + linear_dst.blue_colorant.y * f64::from(rgb[2]);
        Some(y.clamp(0.0, 0.184_186) as f32)
    }

    /// CMYK F32 -> sRGB F32 wrapper: runs `moxcms` into linear sRGB, applies relative
    /// black-point compensation, then encodes with exact sign-preserving sRGB OETF
    /// (avoiding `moxcms`'s coarse post-CLUT sRGB curve and matching `Lcms` BPC).
    struct MoxCmykToSrgbF32 {
        inner: Arc<dyn moxcms::TransformExecutor<f32> + Send + Sync>,
        black_point_y: f32,
        dst_ch: usize,
    }

    impl moxcms::TransformExecutor<f32> for MoxCmykToSrgbF32 {
        fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), moxcms::CmsError> {
            self.inner.transform(src, dst)?;
            let bp = self.black_point_y;
            let tc = self.dst_ch;
            if bp > 0.0 {
                let scale = 1.0 / (1.0 - bp);
                for px in dst.chunks_exact_mut(tc) {
                    px[0] = (px[0] - bp) * scale;
                    px[1] = (px[1] - bp) * scale;
                    px[2] = (px[2] - bp) * scale;
                }
            }
            if tc == 4 {
                linear_srgb::default::linear_to_srgb_extended_rgba_slice(dst);
            } else {
                linear_srgb::default::linear_to_srgb_extended_slice(dst);
            }
            Ok(())
        }
    }

    enum MoxExec {
        U8(Arc<dyn moxcms::TransformExecutor<u8> + Send + Sync>),
        U16(Arc<dyn moxcms::TransformExecutor<u16> + Send + Sync>),
        F32(Arc<dyn moxcms::TransformExecutor<f32> + Send + Sync>),
    }

    struct MoxRowTransform {
        exec: MoxExec,
        src_bpp: usize,
        dst_bpp: usize,
        src_ch: usize,
        dst_ch: usize,
        dst_sample: SampleType,
        src_cmyk_inverted: bool,
        dst_cmyk_inverted: bool,
        fill_opaque_alpha: bool,
    }

    impl CmsTransform for MoxRowTransform {
        fn transform_row(
            &self,
            src: &[u8],
            dst: &mut [u8],
            width: u32,
        ) -> Result<(), At<ConvertError>> {
            let w = width as usize;
            let s_len = w
                .checked_mul(self.src_bpp)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            let d_len = w
                .checked_mul(self.dst_bpp)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            if src.len() < s_len || dst.len() < d_len {
                return Err(err_at(ConvertError::DimensionMismatch));
            }
            if w == 0 {
                return Ok(());
            }
            let s = &src[..s_len];
            let d = &mut dst[..d_len];
            match &self.exec {
                MoxExec::U8(ex) => {
                    if !self.src_cmyk_inverted {
                        ex.transform(s, d)
                            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    } else {
                        let mut inv = [0u8; CHUNK_PX * 4];
                        let mut pos = 0;
                        while pos < w {
                            let n = (w - pos).min(CHUNK_PX);
                            let sb = &s[pos * self.src_bpp..(pos + n) * self.src_bpp];
                            let db = &mut d[pos * self.dst_bpp..(pos + n) * self.dst_bpp];
                            let tmp = &mut inv[..sb.len()];
                            for (o, &b) in tmp.iter_mut().zip(sb) {
                                *o = !b;
                            }
                            ex.transform(tmp, db)
                                .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                            pos += n;
                        }
                    }
                    if self.dst_cmyk_inverted {
                        for b in d.iter_mut() {
                            *b = !*b;
                        }
                    }
                }
                MoxExec::U16(ex) => {
                    if !self.src_cmyk_inverted
                        && !self.dst_cmyk_inverted
                        && let (Ok(s16), Ok(d16)) = (
                            bytemuck::try_cast_slice::<u8, u16>(s),
                            bytemuck::try_cast_slice_mut::<u8, u16>(d),
                        )
                    {
                        ex.transform(s16, d16)
                            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    } else {
                        let mut s_buf = [0u16; CHUNK_PX * 4];
                        let mut d_buf = [0u16; CHUNK_PX * 4];
                        let mut pos = 0;
                        while pos < w {
                            let n = (w - pos).min(CHUNK_PX);
                            let sc = n * self.src_ch;
                            let dc = n * self.dst_ch;
                            let sb = &s[pos * self.src_bpp..(pos + n) * self.src_bpp];
                            for (dst_s, chunk) in s_buf[..sc].iter_mut().zip(sb.as_chunks::<2>().0)
                            {
                                let v = u16::from_ne_bytes(*chunk);
                                *dst_s = if self.src_cmyk_inverted { !v } else { v };
                            }
                            ex.transform(&s_buf[..sc], &mut d_buf[..dc])
                                .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                            let db = &mut d[pos * self.dst_bpp..(pos + n) * self.dst_bpp];
                            for (&v, chunk) in d_buf[..dc].iter().zip(db.as_chunks_mut::<2>().0) {
                                let out = if self.dst_cmyk_inverted { !v } else { v };
                                *chunk = out.to_ne_bytes();
                            }
                            pos += n;
                        }
                    }
                }
                MoxExec::F32(ex) => {
                    if !self.src_cmyk_inverted
                        && !self.dst_cmyk_inverted
                        && let (Ok(sf), Ok(df)) = (
                            bytemuck::try_cast_slice::<u8, f32>(s),
                            bytemuck::try_cast_slice_mut::<u8, f32>(d),
                        )
                    {
                        ex.transform(sf, df)
                            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    } else {
                        let mut s_buf = [0.0f32; CHUNK_PX * 4];
                        let mut d_buf = [0.0f32; CHUNK_PX * 4];
                        let mut pos = 0;
                        while pos < w {
                            let n = (w - pos).min(CHUNK_PX);
                            let sc = n * self.src_ch;
                            let dc = n * self.dst_ch;
                            let sb = &s[pos * self.src_bpp..(pos + n) * self.src_bpp];
                            for (dst_s, chunk) in s_buf[..sc].iter_mut().zip(sb.as_chunks::<4>().0)
                            {
                                let v = f32::from_ne_bytes(*chunk);
                                *dst_s = if self.src_cmyk_inverted { 1.0 - v } else { v };
                            }
                            ex.transform(&s_buf[..sc], &mut d_buf[..dc])
                                .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                            let db = &mut d[pos * self.dst_bpp..(pos + n) * self.dst_bpp];
                            for (&v, chunk) in d_buf[..dc].iter().zip(db.as_chunks_mut::<4>().0) {
                                let out = if self.dst_cmyk_inverted { 1.0 - v } else { v };
                                *chunk = out.to_ne_bytes();
                            }
                            pos += n;
                        }
                    }
                }
            }
            if self.fill_opaque_alpha {
                let sz = self.dst_sample.byte_size();
                let opaque: &[u8] = match self.dst_sample {
                    SampleType::U8 => &[255],
                    SampleType::U16 => &[255, 255],
                    SampleType::F32 => &1.0f32.to_ne_bytes(),
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                for px in d.chunks_exact_mut(self.dst_bpp) {
                    px[self.dst_bpp - sz..].copy_from_slice(opaque);
                }
            }
            Ok(())
        }
    }

    impl CmsBackend for MoxCms {
        fn supports_formats(&self, src: &PixelFormat, dst: &PixelFormat) -> bool {
            let eff_intent = match (&src.color, &dst.color) {
                (ColorSpace::Icc { intent, .. }, _) => *intent,
                (_, ColorSpace::Icc { intent, .. }) => *intent,
                _ => RenderingIntent::RelativeColorimetric,
            };
            if eff_intent == RenderingIntent::AbsoluteColorimetric {
                return false;
            }
            if src.layout.sample != dst.layout.sample
                || !matches!(
                    src.layout.sample,
                    SampleType::U8 | SampleType::U16 | SampleType::F32
                )
                || !src.layout.packing.is_full()
                || !dst.layout.packing.is_full()
                || src.color.alpha() == AlphaMode::Premultiplied
                || dst.color.alpha() == AlphaMode::Premultiplied
            {
                return false;
            }
            if to_mox_layout(src.layout.order).is_none()
                || to_mox_layout(dst.layout.order).is_none()
            {
                return false;
            }
            if let Some(sb) = icc_bytes_of(&src.color)
                && has_float_pipeline(sb, eff_intent, true)
            {
                return false;
            }
            if let Some(db) = icc_bytes_of(&dst.color)
                && has_float_pipeline(db, eff_intent, false)
            {
                return false;
            }
            let color_ok = |c: &ColorSpace| match c {
                ColorSpace::Icc { profile, .. } => {
                    if profile.len() < 128 || &profile[36..40] != b"acsp" {
                        return false;
                    }
                    moxcms::ColorProfile::new_from_slice(profile)
                        .map(|p| validate_channel_curves(&p))
                        .unwrap_or(false)
                }
                ColorSpace::Cmyk { icc: Some(p), .. } => p.len() >= 128 && &p[36..40] == b"acsp",
                ColorSpace::Cmyk { icc: None, .. } => false,
                ColorSpace::Rgb {
                    primaries,
                    transfer,
                    range,
                    ..
                } => {
                    *range == SignalRange::Full
                        && matches!(
                            (primaries, transfer),
                            (
                                Primaries::Bt709,
                                Transfer::Srgb | Transfer::Linear | Transfer::Gamma22
                            ) | (Primaries::DisplayP3, Transfer::Srgb | Transfer::Linear)
                                | (Primaries::AdobeRgb, Transfer::AdobeRgb)
                                | (
                                    Primaries::Bt2020,
                                    Transfer::Bt709 | Transfer::Pq | Transfer::Linear
                                )
                        )
                }
                ColorSpace::Gray {
                    range, transfer, ..
                } => {
                    *range == SignalRange::Full
                        && matches!(
                            transfer,
                            Transfer::Srgb
                                | Transfer::Linear
                                | Transfer::Gamma22
                                | Transfer::AdobeRgb
                                | Transfer::Gamma(_)
                        )
                }
                _ => false,
            };
            color_ok(&src.color) && color_ok(&dst.color)
        }

        fn build_transform(
            &self,
            src: &PixelFormat,
            dst: &PixelFormat,
        ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>> {
            if src.layout.sample != dst.layout.sample
                || !src.layout.packing.is_full()
                || !dst.layout.packing.is_full()
                || src.color.alpha() == AlphaMode::Premultiplied
                || dst.color.alpha() == AlphaMode::Premultiplied
            {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            let (src_p, src_intent) = profile_from_color_space(&src.color)?;
            let (dst_p, dst_intent) = profile_from_color_space(&dst.color)?;
            let eff_intent = src_intent
                .or(dst_intent)
                .unwrap_or(RenderingIntent::RelativeColorimetric);
            if eff_intent == RenderingIntent::AbsoluteColorimetric {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            if let Some(sb) = icc_bytes_of(&src.color)
                && has_float_pipeline(sb, eff_intent, true)
            {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            if let Some(db) = icc_bytes_of(&dst.color)
                && has_float_pipeline(db, eff_intent, false)
            {
                return Err(err_at(ConvertError::UnsupportedRoute));
            }
            let src_l = to_mox_layout(src.layout.order)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            let dst_l = to_mox_layout(dst.layout.order)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            let is_f32 = src.layout.sample == SampleType::F32;
            let mut mox_intent = to_mox_intent(eff_intent);
            // ICC.1 §8.10.2 Table 25: fall back to A2B0/B2A0 when colorimetric LUTs & TRCs are absent.
            if eff_intent == RenderingIntent::RelativeColorimetric
                && ((src_p.lut_a_to_b_colorimetric.is_none()
                    && src_p.lut_a_to_b_perceptual.is_some()
                    && src_p.red_trc.is_none())
                    || (dst_p.lut_b_to_a_colorimetric.is_none()
                        && dst_p.lut_b_to_a_perceptual.is_some()
                        && dst_p.red_trc.is_none()))
            {
                mox_intent = moxcms::RenderingIntent::Perceptual;
            }
            let shared_curve = |p: &moxcms::ColorProfile| {
                p.color_space != moxcms::DataColorSpace::Rgb
                    || (p.red_trc == p.green_trc && p.red_trc == p.blue_trc)
            };
            #[allow(clippy::needless_update)]
            let opts = moxcms::TransformOptions {
                rendering_intent: mox_intent,
                allow_use_cicp_transfer: false,
                interpolation_method: moxcms::InterpolationMethod::Tetrahedral,
                barycentric_weight_scale: if is_f32 {
                    moxcms::BarycentricWeightScale::High
                } else {
                    moxcms::BarycentricWeightScale::Low
                },
                // Fixed-point Q2.13 is fine for 8-bit but loses up to 7 8-bit codes on U16 wide-gamut transforms.
                prefer_fixed_point: src.layout.sample == SampleType::U8,
                allow_extended_range_rgb_xyz: is_f32
                    && shared_curve(&src_p)
                    && shared_curve(&dst_p),
                ..Default::default()
            };
            let cmyk_to_srgb_f32 = is_f32
                && matches!(src.color, ColorSpace::Cmyk { .. })
                && matches!(
                    dst.color,
                    ColorSpace::Rgb {
                        primaries: Primaries::Bt709,
                        transfer: Transfer::Srgb,
                        ..
                    }
                )
                && matches!(dst_l, moxcms::Layout::Rgb | moxcms::Layout::Rgba);
            let exec = match src.layout.sample {
                SampleType::U8 => MoxExec::U8(
                    src_p
                        .create_transform_8bit(src_l, &dst_p, dst_l, opts)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?,
                ),
                SampleType::U16 => MoxExec::U16(
                    src_p
                        .create_transform_16bit(src_l, &dst_p, dst_l, opts)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?,
                ),
                SampleType::F32 if cmyk_to_srgb_f32 => {
                    let mut linear_dst = dst_p.clone();
                    linear_dst.cicp = None;
                    let ident = moxcms::ToneReprCurve::Lut(alloc::vec::Vec::new());
                    linear_dst.red_trc = Some(ident.clone());
                    linear_dst.green_trc = Some(ident.clone());
                    linear_dst.blue_trc = Some(ident);
                    #[allow(clippy::needless_update)]
                    let lin_opts = moxcms::TransformOptions {
                        allow_extended_range_rgb_xyz: true,
                        ..opts
                    };
                    let black_point_y = if eff_intent == RenderingIntent::RelativeColorimetric {
                        cmyk_black_point_y(&src_p, &linear_dst, lin_opts).unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    let inner = src_p
                        .create_transform_f32(src_l, &linear_dst, dst_l, lin_opts)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    MoxExec::F32(Arc::new(MoxCmykToSrgbF32 {
                        inner,
                        black_point_y,
                        dst_ch: dst.layout.channels(),
                    }))
                }
                SampleType::F32 => MoxExec::F32(
                    src_p
                        .create_transform_f32(src_l, &dst_p, dst_l, opts)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?,
                ),
                _ => return Err(err_at(ConvertError::UnsupportedRoute)),
            };
            let fill_opaque_alpha = dst.layout.order.has_alpha_slot()
                && (dst.color.alpha() == AlphaMode::Opaque
                    || !src.layout.order.has_alpha_slot()
                    || src.color.alpha() == AlphaMode::Opaque);
            Ok(Arc::new(MoxRowTransform {
                exec,
                src_bpp: src.layout.bytes_per_pixel(),
                dst_bpp: dst.layout.bytes_per_pixel(),
                src_ch: src.layout.channels(),
                dst_ch: dst.layout.channels(),
                dst_sample: dst.layout.sample,
                src_cmyk_inverted: matches!(src.color, ColorSpace::Cmyk { inverted: true, .. })
                    || src.layout.order == ChannelOrder::CmykInverted,
                dst_cmyk_inverted: matches!(dst.color, ColorSpace::Cmyk { inverted: true, .. })
                    || dst.layout.order == ChannelOrder::CmykInverted,
                fill_opaque_alpha,
            }))
        }
    }
}

#[cfg(feature = "cms-moxcms")]
pub use moxcms_impl::MoxCms;

#[cfg(feature = "cms-lcms")]
mod lcms_impl {
    use super::*;
    use crate::error::err_at;
    use crate::icc_profiles::SynthesizedIcc;
    use lcms::api::{
        CiexyY, CiexyYTriple, D65, Format, Intent, Plan, PlanOptions, Profile, ToneCurve, flags,
    };
    use zenpixels::{
        AlphaMode, ChannelOrder, Cicp, ColorSpace, Primaries, RenderingIntent, SampleLayout,
        SampleType, SignalRange, Transfer,
    };

    const CHUNK_PX: usize = 64;

    const PT_GRAY: u32 = 3;
    const PT_RGB: u32 = 4;
    const PT_CMYK: u32 = 6;
    const DOSWAP_SH: u32 = 1 << 10;
    const SWAPFIRST_SH: u32 = 1 << 14;
    const FLOAT_SH: u32 = 1 << 22;

    /// Pure-Rust Little CMS (`imazen/lcms`) ICC CMS backend.
    ///
    /// Stateless and safe to share (`Send + Sync`); built transforms hold an
    /// immutable, lock-free `lcms::api::Plan` with AVX2/SIMD pipeline
    /// optimization enabled and stack-only per-row scratch (zero heap
    /// allocation per row).
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Lcms;

    impl Lcms {
        /// Canonical named, conservatively quoted [`CmsBinding`] for pure-Rust `lcms`
        /// (`"org.imazen.cms.lcms"`).
        ///
        /// Row execution uses stack-only scratch (`apply_scratch_bytes = Some(0)`). Profile
        /// build/transform heap and peak stack remain unknown (`None` / `small_stack_qualified = false`)
        /// until qualified per route.
        #[must_use]
        pub fn binding() -> CmsBinding {
            CmsBinding::new(
                "org.imazen.cms.lcms",
                concat!(
                    "lcms-rust-opt/zenpixels-convert-",
                    env!("CARGO_PKG_VERSION")
                ),
                Arc::new(Self),
                CmsQuote::new()
                    .with_u8(true)
                    .with_u16(true)
                    .with_f16(true)
                    .with_f32(true)
                    .with_cmyk(true)
                    .with_absolute_colorimetric(true)
                    .with_float_mpe(true)
                    .with_apply_scratch_bytes(Some(0))
                    .with_concurrent_apply(true),
            )
        }
    }

    /// Pure-Rust Little CMS backend with `FLAGS_NOOPTIMIZE` enabled.
    ///
    /// Evaluates profile stages directly without resampling into a 16-bit
    /// integer CLUT — slower on integer formats, useful for high-dynamic-range
    /// float-pipeline MPE profiles (such as PQ ICC profiles) where a 33-point
    /// integer CLUT loses shadow precision.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct LcmsUnoptimized;

    impl LcmsUnoptimized {
        /// Canonical named, conservatively quoted [`CmsBinding`] for unoptimized pure-Rust `lcms`
        /// (`"org.imazen.cms.lcms_unoptimized"`).
        #[must_use]
        pub fn binding() -> CmsBinding {
            CmsBinding::new(
                "org.imazen.cms.lcms_unoptimized",
                concat!(
                    "lcms-rust-noopt/zenpixels-convert-",
                    env!("CARGO_PKG_VERSION")
                ),
                Arc::new(Self),
                CmsQuote::new()
                    .with_u8(true)
                    .with_u16(true)
                    .with_f16(true)
                    .with_f32(true)
                    .with_cmyk(true)
                    .with_absolute_colorimetric(true)
                    .with_float_mpe(true)
                    .with_apply_scratch_bytes(Some(0))
                    .with_concurrent_apply(true),
            )
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum InkMode {
        None,
        InvertedInt,
        FloatPercent,
        FloatInvertedPercent,
    }

    #[derive(Clone, Copy)]
    struct LcmsFormatInfo {
        format: Format,
        bpp: usize,
        sample: SampleType,
        has_alpha: bool,
        ink: InkMode,
    }

    fn describe_layout(layout: SampleLayout, color: &ColorSpace) -> Option<LcmsFormatInfo> {
        if !layout.packing.is_full() {
            return None;
        }
        let sample = layout.sample;
        if !matches!(
            sample,
            SampleType::U8 | SampleType::U16 | SampleType::F16 | SampleType::F32
        ) {
            return None;
        }
        let inverted_cmyk = matches!(color, ColorSpace::Cmyk { inverted: true, .. })
            || layout.order == ChannelOrder::CmykInverted;

        let (space, colors, extra, swap, first, ink) = match layout.order {
            ChannelOrder::Gray => (PT_GRAY, 1u32, 0u32, false, false, InkMode::None),
            ChannelOrder::GrayAlpha => {
                if sample == SampleType::F16 {
                    return None;
                }
                (PT_GRAY, 1, 1, false, false, InkMode::None)
            }
            ChannelOrder::Rgb => (PT_RGB, 3, 0, false, false, InkMode::None),
            ChannelOrder::Bgr => {
                if sample == SampleType::F16 {
                    return None;
                }
                (PT_RGB, 3, 0, true, false, InkMode::None)
            }
            ChannelOrder::Rgba | ChannelOrder::Rgbx => (PT_RGB, 3, 1, false, false, InkMode::None),
            ChannelOrder::Bgra | ChannelOrder::Bgrx => {
                if sample == SampleType::F16 {
                    return None;
                }
                (PT_RGB, 3, 1, true, true, InkMode::None)
            }
            ChannelOrder::Cmyk | ChannelOrder::CmykInverted => {
                let ink = match (sample, inverted_cmyk) {
                    (SampleType::U8 | SampleType::U16, false) => InkMode::None,
                    (SampleType::U8 | SampleType::U16, true) => InkMode::InvertedInt,
                    (SampleType::F32, false) => InkMode::FloatPercent,
                    (SampleType::F32, true) => InkMode::FloatInvertedPercent,
                    _ => return None,
                };
                (PT_CMYK, 4, 0, false, false, ink)
            }
            _ => return None,
        };

        let float_bit = if sample.is_float() { FLOAT_SH } else { 0 };
        let code = float_bit
            | (space << 16)
            | (u32::from(first) * SWAPFIRST_SH)
            | (u32::from(swap) * DOSWAP_SH)
            | (extra << 7)
            | (colors << 3)
            | (sample.byte_size() as u32);

        Some(LcmsFormatInfo {
            format: Format::from_u32(code),
            bpp: layout.bytes_per_pixel(),
            sample,
            has_alpha: layout.order.has_alpha_slot(),
            ink,
        })
    }

    fn to_lcms_intent(intent: RenderingIntent) -> Intent {
        match intent {
            RenderingIntent::Perceptual => Intent::Perceptual,
            RenderingIntent::RelativeColorimetric => Intent::RelativeColorimetric,
            RenderingIntent::Saturation => Intent::Saturation,
            RenderingIntent::AbsoluteColorimetric => Intent::AbsoluteColorimetric,
            _ => Intent::RelativeColorimetric,
        }
    }

    fn tone_curve_for_transfer(transfer: Transfer) -> Result<Option<ToneCurve>, At<ConvertError>> {
        match transfer {
            Transfer::Srgb => Ok(Some(ToneCurve::srgb())),
            Transfer::Linear => Ok(Some(ToneCurve::linear())),
            Transfer::Gamma22 => ToneCurve::gamma(2.2)
                .map(Some)
                .map_err(|_| err_at(ConvertError::CmsBackendFailed)),
            Transfer::AdobeRgb => ToneCurve::gamma(563.0 / 256.0)
                .map(Some)
                .map_err(|_| err_at(ConvertError::CmsBackendFailed)),
            Transfer::Gamma(g) => ToneCurve::gamma(f64::from(g.exponent()))
                .map(Some)
                .map_err(|_| err_at(ConvertError::CmsBackendFailed)),
            Transfer::Bt709 => ToneCurve::parametric(
                4,
                &[1.0 / 0.45, 1.0 / 1.099, 0.099 / 1.099, 1.0 / 4.5, 0.081],
            )
            .map(Some)
            .map_err(|_| err_at(ConvertError::CmsBackendFailed)),
            _ => Ok(None),
        }
    }

    fn primaries_triple(primaries: Primaries) -> Option<(CiexyY, CiexyYTriple)> {
        let (((rx, ry), (gx, gy), (bx, by)), (wx, wy)) =
            (primaries.chromaticity()?, primaries.white_point()?);
        // `lcms::Profile::rgb` divides by each primary's `y`, so skip `y == 0` (e.g. `Primaries::Xyz`).
        // Negative `y` (such as ACES AP0 `y_B = -0.0770`) is finite and well-defined.
        if ry == 0.0 || gy == 0.0 || by == 0.0 || wy <= 0.0 {
            return None;
        }
        Some((
            CiexyY {
                x: f64::from(wx),
                y: f64::from(wy),
                Y: 1.0,
            },
            CiexyYTriple {
                Red: CiexyY {
                    x: f64::from(rx),
                    y: f64::from(ry),
                    Y: 1.0,
                },
                Green: CiexyY {
                    x: f64::from(gx),
                    y: f64::from(gy),
                    Y: 1.0,
                },
                Blue: CiexyY {
                    x: f64::from(bx),
                    y: f64::from(by),
                    Y: 1.0,
                },
            },
        ))
    }

    fn profile_from_color_space(
        cs: &ColorSpace,
    ) -> Result<(Profile, Option<RenderingIntent>), At<ConvertError>> {
        match cs {
            ColorSpace::Icc {
                profile, intent, ..
            } => {
                let p =
                    Profile::open(profile).map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                Ok((p, Some(*intent)))
            }
            ColorSpace::Cmyk { icc, .. } => {
                let bytes = icc
                    .as_ref()
                    .ok_or_else(|| err_at(ConvertError::CmsRequired))?;
                let p = Profile::open(bytes).map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                Ok((p, Some(RenderingIntent::RelativeColorimetric)))
            }
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                ..
            } => {
                if *range != SignalRange::Full || matches!(transfer, Transfer::Hlg(_)) {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                if *primaries == Primaries::AdobeRgb && *transfer == Transfer::AdobeRgb {
                    let p = Profile::open(crate::icc_profiles::ADOBE_RGB)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                // Build sRGB and all standard / custom RGB spaces using each space's actual white
                // point (D65, D50, D60, DCI, C, or custom) + ToneCurve rather than `Profile::srgb()`
                // (which uses legacy 0.3291 D65).
                if let (Some((wp, triple)), Some(curve)) = (
                    primaries_triple(*primaries),
                    tone_curve_for_transfer(*transfer)?,
                ) {
                    let p = Profile::rgb(&wp, &triple, [&curve, &curve, &curve])
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                if let (Some(p_code), Some(t_code)) = (primaries.to_cicp(), transfer.to_cicp())
                    && let SynthesizedIcc::Profile(bytes) =
                        SynthesizedIcc::for_cicp(Cicp::new(p_code, t_code, 0, true))
                {
                    let p = Profile::open(&bytes)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                Err(err_at(ConvertError::UnsupportedRoute))
            }
            ColorSpace::Gray {
                transfer, range, ..
            } => {
                if *range != SignalRange::Full || matches!(transfer, Transfer::Hlg(_)) {
                    return Err(err_at(ConvertError::UnsupportedRoute));
                }
                if let Some(curve) = tone_curve_for_transfer(*transfer)? {
                    let p = Profile::gray_curve(&D65, &curve)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                if let Some(t_code) = transfer.to_cicp()
                    && let SynthesizedIcc::Profile(bytes) =
                        SynthesizedIcc::for_gray_cicp(Cicp::new(1, t_code, 0, true))
                {
                    let p = Profile::open(&bytes)
                        .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;
                    return Ok((p, None));
                }
                Err(err_at(ConvertError::UnsupportedRoute))
            }
            _ => Err(err_at(ConvertError::UnsupportedRoute)),
        }
    }

    struct LcmsRowTransform {
        plan: Plan,
        src: LcmsFormatInfo,
        dst: LcmsFormatInfo,
        /// Write canonical opaque alpha (1.0 / max code) to dst alpha slot.
        fill_opaque_alpha: bool,
        /// Guard for float RGBA CLUT fast paths in `lcms` that zero the 4th f32 lane
        /// after `handle_extra_channels` has already copied alpha.
        copy_f32_alpha_after: bool,
    }

    fn build_lcms_transform(
        src: &PixelFormat,
        dst: &PixelFormat,
        no_optimize: bool,
    ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>> {
        if src.color.alpha() == AlphaMode::Premultiplied
            || dst.color.alpha() == AlphaMode::Premultiplied
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let from = describe_layout(src.layout, &src.color)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
        let to = describe_layout(dst.layout, &dst.color)
            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;

        let (src_profile, src_intent) = profile_from_color_space(&src.color)?;
        let (dst_profile, dst_intent) = profile_from_color_space(&dst.color)?;
        let eff_intent = src_intent
            .or(dst_intent)
            .unwrap_or(RenderingIntent::RelativeColorimetric);

        let lcms_intent = to_lcms_intent(eff_intent);
        let bpc = matches!(src.color, ColorSpace::Cmyk { .. })
            && eff_intent == RenderingIntent::RelativeColorimetric;
        let adaptation = if eff_intent == RenderingIntent::AbsoluteColorimetric {
            0.0
        } else {
            1.0
        };

        let fill_opaque_alpha = to.has_alpha
            && (dst.color.alpha() == AlphaMode::Opaque
                || !from.has_alpha
                || src.color.alpha() == AlphaMode::Opaque);

        // `NO_CACHE` enables lcms's fused AVX2/SIMD loops (Prelin8, Clut8, MatShaper)
        // for 8-bit transforms while keeping per-row execution cache-independent.
        // For U16 / F16 / F32, disable pipeline resampling (`NO_OPTIMIZE`) so lcms
        // evaluates exact curves/matrices without 16-bit CLUT quantization or F32 clipping.
        let mut transform_flags = flags::NO_CACHE;
        if no_optimize || from.sample != SampleType::U8 || to.sample != SampleType::U8 {
            transform_flags |= flags::NO_OPTIMIZE;
        }
        if from.has_alpha && to.has_alpha && !fill_opaque_alpha {
            transform_flags |= flags::COPY_ALPHA;
        }
        if bpc {
            transform_flags |= flags::BLACK_POINT_COMPENSATION;
        }

        let opts = PlanOptions::new()
            .intent(lcms_intent)
            .flags(transform_flags)
            .adaptation_state(adaptation);

        let plan = Plan::with_options(&src_profile, &dst_profile, from.format, to.format, &opts)
            .map_err(|_| err_at(ConvertError::CmsBackendFailed))?;

        let copy_f32_alpha_after = from.has_alpha
            && to.has_alpha
            && !fill_opaque_alpha
            && from.sample == SampleType::F32
            && to.sample == SampleType::F32
            && from.bpp == 16
            && to.bpp == 16;

        Ok(Arc::new(LcmsRowTransform {
            plan,
            src: from,
            dst: to,
            fill_opaque_alpha,
            copy_f32_alpha_after,
        }))
    }

    impl CmsTransform for LcmsRowTransform {
        fn transform_row(
            &self,
            src: &[u8],
            dst: &mut [u8],
            width: u32,
        ) -> Result<(), At<ConvertError>> {
            let w = width as usize;
            let s_len = w
                .checked_mul(self.src.bpp)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            let d_len = w
                .checked_mul(self.dst.bpp)
                .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
            if src.len() < s_len || dst.len() < d_len {
                return Err(err_at(ConvertError::DimensionMismatch));
            }
            if w == 0 {
                return Ok(());
            }

            let s = &src[..s_len];
            let d = &mut dst[..d_len];
            let mut scratch = self.plan.scratch();

            match self.src.ink {
                InkMode::None => {
                    self.plan.apply_n(&mut scratch, s, d, width);
                }
                InkMode::InvertedInt => {
                    // Bitwise NOT inverts both u8 (`255 - x`) and native-endian u16 (`65535 - x`).
                    // Max CMYK integer bpp is 8 bytes (CMYK16), so 64 px = 512 B on stack.
                    let mut staged = [0u8; CHUNK_PX * 8];
                    let mut pos = 0;
                    while pos < w {
                        let n = (w - pos).min(CHUNK_PX);
                        let sb = &s[pos * self.src.bpp..(pos + n) * self.src.bpp];
                        let db = &mut d[pos * self.dst.bpp..(pos + n) * self.dst.bpp];
                        let tmp = &mut staged[..sb.len()];
                        for (o, &b) in tmp.iter_mut().zip(sb) {
                            *o = !b;
                        }
                        self.plan.apply_n(&mut scratch, tmp, db, n as u32);
                        pos += n;
                    }
                }
                InkMode::FloatPercent | InkMode::FloatInvertedPercent => {
                    let invert = self.src.ink == InkMode::FloatInvertedPercent;
                    let mut staged = [0.0f32; CHUNK_PX * 4];
                    let mut pos = 0;
                    while pos < w {
                        let n = (w - pos).min(CHUNK_PX);
                        let sb = &s[pos * self.src.bpp..(pos + n) * self.src.bpp];
                        let db = &mut d[pos * self.dst.bpp..(pos + n) * self.dst.bpp];
                        let tmp = &mut staged[..n * 4];
                        for (o, bytes) in tmp.iter_mut().zip(sb.as_chunks::<4>().0) {
                            let v = f32::from_ne_bytes(*bytes);
                            *o = 100.0 * if invert { 1.0 - v } else { v };
                        }
                        self.plan
                            .apply_n(&mut scratch, bytemuck::cast_slice(tmp), db, n as u32);
                        pos += n;
                    }
                }
            }

            // Destination CMYK post-processing (inversion / percent -> [0, 1]).
            match self.dst.ink {
                InkMode::None => {}
                InkMode::InvertedInt => {
                    for b in d.iter_mut() {
                        *b = !*b;
                    }
                }
                InkMode::FloatPercent => {
                    for chunk in d.as_chunks_mut::<4>().0 {
                        let v = f32::from_ne_bytes(*chunk) * 0.01;
                        *chunk = v.to_ne_bytes();
                    }
                }
                InkMode::FloatInvertedPercent => {
                    for chunk in d.as_chunks_mut::<4>().0 {
                        let v = 1.0 - f32::from_ne_bytes(*chunk) * 0.01;
                        *chunk = v.to_ne_bytes();
                    }
                }
            }

            // Fill opaque alpha when adding an alpha channel or canonicalizing Opaque alpha.
            if self.fill_opaque_alpha {
                let sz = self.dst.sample.byte_size();
                let opaque: &[u8] = match self.dst.sample {
                    SampleType::U8 => &[255],
                    SampleType::U16 => &[255, 255],
                    SampleType::F16 => &0x3C00u16.to_ne_bytes(),
                    SampleType::F32 => &1.0f32.to_ne_bytes(),
                    _ => return Err(err_at(ConvertError::UnsupportedRoute)),
                };
                for px in d.chunks_exact_mut(self.dst.bpp) {
                    px[self.dst.bpp - sz..].copy_from_slice(opaque);
                }
            }

            // Guard against RGBA_FLT CLUT fast paths overwriting the 4th f32 lane.
            if self.copy_f32_alpha_after {
                for (sp, dp) in s.as_chunks::<16>().0.iter().zip(d.as_chunks_mut::<16>().0) {
                    dp[12..16].copy_from_slice(&sp[12..16]);
                }
            }

            Ok(())
        }
    }

    fn lcms_supports_specs(src: &PixelFormat, dst: &PixelFormat) -> bool {
        if src.color.alpha() == AlphaMode::Premultiplied
            || dst.color.alpha() == AlphaMode::Premultiplied
        {
            return false;
        }
        if describe_layout(src.layout, &src.color).is_none()
            || describe_layout(dst.layout, &dst.color).is_none()
        {
            return false;
        }
        let color_ok = |c: &ColorSpace| match c {
            ColorSpace::Icc { profile, .. } => profile.len() >= 128 && &profile[36..40] == b"acsp",
            ColorSpace::Cmyk { icc: Some(p), .. } => p.len() >= 128 && &p[36..40] == b"acsp",
            ColorSpace::Cmyk { icc: None, .. } => false,
            ColorSpace::Rgb {
                primaries,
                transfer,
                range,
                ..
            } => {
                if *range != SignalRange::Full || matches!(transfer, Transfer::Hlg(_)) {
                    return false;
                }
                if *primaries == Primaries::Bt709 && *transfer == Transfer::Srgb {
                    return true;
                }
                if *primaries == Primaries::AdobeRgb && *transfer == Transfer::AdobeRgb {
                    return true;
                }
                if primaries_triple(*primaries).is_some()
                    && matches!(
                        transfer,
                        Transfer::Srgb
                            | Transfer::Linear
                            | Transfer::Gamma22
                            | Transfer::AdobeRgb
                            | Transfer::Gamma(_)
                            | Transfer::Bt709
                    )
                {
                    return true;
                }
                if let (Some(p_code), Some(t_code)) = (primaries.to_cicp(), transfer.to_cicp()) {
                    return matches!(
                        SynthesizedIcc::for_cicp(Cicp::new(p_code, t_code, 0, true)),
                        SynthesizedIcc::Profile(_)
                    );
                }
                false
            }
            ColorSpace::Gray {
                range, transfer, ..
            } => {
                if *range != SignalRange::Full || matches!(transfer, Transfer::Hlg(_)) {
                    return false;
                }
                if matches!(
                    transfer,
                    Transfer::Srgb
                        | Transfer::Linear
                        | Transfer::Gamma22
                        | Transfer::AdobeRgb
                        | Transfer::Gamma(_)
                        | Transfer::Bt709
                ) {
                    return true;
                }
                if let Some(t_code) = transfer.to_cicp() {
                    return matches!(
                        SynthesizedIcc::for_gray_cicp(Cicp::new(1, t_code, 0, true)),
                        SynthesizedIcc::Profile(_)
                    );
                }
                false
            }
            _ => false,
        };
        color_ok(&src.color) && color_ok(&dst.color)
    }

    impl CmsBackend for Lcms {
        fn supports_formats(&self, src: &PixelFormat, dst: &PixelFormat) -> bool {
            lcms_supports_specs(src, dst)
        }

        fn build_transform(
            &self,
            src: &PixelFormat,
            dst: &PixelFormat,
        ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>> {
            build_lcms_transform(src, dst, false)
        }
    }

    impl CmsBackend for LcmsUnoptimized {
        fn supports_formats(&self, src: &PixelFormat, dst: &PixelFormat) -> bool {
            lcms_supports_specs(src, dst)
        }

        fn build_transform(
            &self,
            src: &PixelFormat,
            dst: &PixelFormat,
        ) -> Result<Arc<dyn CmsTransform>, At<ConvertError>> {
            build_lcms_transform(src, dst, true)
        }
    }
}

#[cfg(feature = "cms-lcms")]
pub use lcms_impl::{Lcms, LcmsUnoptimized};
