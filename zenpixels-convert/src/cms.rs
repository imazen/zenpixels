//! Color Management System (CMS) traits.
//!
//! Defines the interface for ICC profile-based color transforms. When a CMS
//! feature is enabled (e.g., `cms-moxcms`, `cms-lcms2`), the implementation
//! provides ICC-to-ICC transforms. Named profile conversions (sRGB, P3,
//! BT.2020) use hardcoded matrices and don't require a CMS.
//!
//! # When codecs need a CMS
//!
//! Most codecs don't need to interact with the CMS directly.
//! [`finalize_for_output`](super::finalize_for_output) handles CMS transforms
//! internally when the [`OutputProfile`](super::OutputProfile) requires one.
//!
//! A codec needs CMS awareness only when:
//!
//! - **Decoding** an image with an embedded ICC profile that doesn't match
//!   any known CICP combination. The decoder extracts the ICC bytes and
//!   stores them on [`ColorContext`](crate::ColorContext). The CMS is used
//!   later (at encode or processing time), not during decode.
//!
//! - **Encoding** with `OutputProfile::Icc(custom_profile)`. The CMS builds
//!   a source→destination transform, which `finalize_for_output` applies
//!   row-by-row via [`RowTransform`].
//!
//! # Implementing a CMS backend
//!
//! To add a new CMS backend (e.g., wrapping Little CMS 2):
//!
//! 1. Implement [`ColorManagement`] on your backend struct.
//! 2. `build_transform` should parse both ICC profiles, create an internal
//!    transform object, and return it as `Box<dyn RowTransform>`.
//! 3. `identify_profile` should check if an ICC profile matches a known
//!    standard (sRGB, Display P3, etc.) and return the corresponding
//!    [`Cicp`](crate::Cicp). This enables the fast path: if both source
//!    and destination are known profiles, hardcoded matrices are used
//!    instead of the CMS.
//! 4. Feature-gate your implementation behind a cargo feature
//!    (e.g., `cms-lcms2`).
//!
//! ```rust,ignore
//! struct MyLcms2;
//!
//! impl ColorManagement for MyLcms2 {
//!     type Error = lcms2::Error;
//!
//!     fn build_transform(
//!         &self,
//!         src_icc: &[u8],
//!         dst_icc: &[u8],
//!     ) -> Result<Box<dyn RowTransform>, Self::Error> {
//!         let src = lcms2::Profile::new_icc(src_icc)?;
//!         let dst = lcms2::Profile::new_icc(dst_icc)?;
//!         let xform = lcms2::Transform::new(&src, &dst, ...)?;
//!         Ok(Box::new(Lcms2RowTransform(xform)))
//!     }
//!
//!     fn identify_profile(&self, icc: &[u8]) -> Option<Cicp> {
//!         // Fast: check MD5 hash against known profiles
//!         // Slow: parse TRC+matrix, compare within tolerance
//!         None
//!     }
//! }
//! ```
//!
//! # No-op CMS
//!
//! Codecs that don't need ICC support can provide a no-op CMS whose
//! `build_transform` always returns an error. This satisfies the type
//! system while making it clear that ICC transforms are unsupported.

use crate::PixelFormat;
use alloc::boxed::Box;

/// ICC rendering intent — controls how colors outside the destination gamut
/// are handled during a profile-to-profile transform.
///
/// # Which intent to use
///
/// For **display-to-display** workflows (web images, app thumbnails, photo
/// export): use [`RelativeColorimetric`](Self::RelativeColorimetric). It
/// preserves in-gamut colors exactly and is the de facto standard for screen
/// output.
///
/// For **photographic print** with a profile that has a perceptual table:
/// use [`Perceptual`](Self::Perceptual). It compresses the full source gamut
/// smoothly instead of clipping.
///
/// For **soft-proofing** ("what will this print look like on screen"): use
/// [`AbsoluteColorimetric`](Self::AbsoluteColorimetric) to simulate the
/// paper white.
///
/// [`Saturation`](Self::Saturation) is for business graphics (pie charts,
/// logos). It is almost never correct for photographic images.
///
/// # Interaction with ICC profiles
///
/// An ICC profile may contain up to four LUTs (AToB0–AToB3), one per intent.
/// **Most display profiles only ship a single LUT** (relative colorimetric).
/// When you request an intent whose LUT is absent, the CMS silently falls
/// back to the profile's default — usually relative colorimetric. This means
/// `Perceptual` and `RelativeColorimetric` produce **identical output** for
/// the vast majority of display profiles (sRGB IEC 61966-2.1, Display P3,
/// etc.). The distinction only matters for print/press profiles that include
/// dedicated perceptual gamut-mapping tables.
///
/// # Bugs and pitfalls
///
/// - **Perceptual on display profiles is a no-op.** Requesting `Perceptual`
///   doesn't add gamut mapping when the profile lacks a perceptual table —
///   it silently degrades to clipping. If you need actual gamut mapping
///   between display profiles, you must supply a profile that contains
///   perceptual intent tables (e.g., a proofing profile or a carefully
///   authored display profile).
///
/// - **AbsoluteColorimetric tints whites.** Source white is preserved
///   literally, so a D50 source on a D65 display shows yellowish whites.
///   Never use this for final output — only for proofing previews.
///
/// - **Saturation may shift hues.** The ICC spec allows saturation-intent
///   tables to sacrifice hue accuracy for vividness. Photographs will look
///   wrong.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
pub enum RenderingIntent {
    /// Compress the entire source gamut into the destination gamut,
    /// preserving the perceptual relationship between colors at the cost
    /// of shifting all values (including in-gamut ones).
    ///
    /// **Requires a perceptual LUT in the profile.** Most display profiles
    /// omit this table, so the CMS falls back to relative colorimetric
    /// silently. This intent only behaves differently from
    /// `RelativeColorimetric` when both source and destination profiles
    /// contain dedicated perceptual rendering tables — typically print,
    /// press, or carefully authored proofing profiles.
    ///
    /// When it works: smooth, continuous gamut mapping with no hard clips.
    /// When the LUT is missing: identical to `RelativeColorimetric`.
    ///
    /// **CMS compatibility warning:** moxcms's perceptual intent
    /// implementation does not match lcms2's output and may not be
    /// accurate for all profile combinations. If cross-CMS consistency
    /// matters, prefer [`RelativeColorimetric`](Self::RelativeColorimetric).
    Perceptual,

    /// Preserve in-gamut colors exactly; clip out-of-gamut colors to the
    /// nearest boundary color. White point is adapted from source to
    /// destination (source white → destination white).
    ///
    /// This is the correct default for virtually all display-to-display
    /// workflows: web images, app thumbnails, photo export, screen preview.
    /// Colors that fit in the destination gamut are reproduced without any
    /// remapping — what the numbers say is what you get.
    ///
    /// **Tradeoff:** saturated gradients that cross the gamut boundary can
    /// show hard clipping artifacts (banding). If the source gamut is much
    /// wider than the destination (e.g., BT.2020 → sRGB), consider whether
    /// a perceptual-intent profile or a dedicated gamut-mapping step would
    /// produce smoother results.
    #[default]
    RelativeColorimetric,

    /// Maximize saturation and vividness, sacrificing hue accuracy.
    /// Designed for business graphics: charts, logos, presentation slides.
    ///
    /// **Not suitable for photographs.** Hue shifts are expected and
    /// intentional — the goal is "vivid", not "accurate".
    ///
    /// Like `Perceptual`, many profiles lack a saturation-intent LUT.
    /// When absent, the CMS falls back to the profile's default intent.
    Saturation,

    /// Like `RelativeColorimetric` but **without** white point adaptation.
    /// Source white is preserved literally: a D50 (warm) source displayed
    /// on a D65 (cool) screen will show yellowish whites.
    ///
    /// **Use exclusively for soft-proofing**: simulating how a print will
    /// look by preserving the paper white and ink gamut on screen. Never
    /// use for final output — the tinted whites look wrong on every
    /// display except the exact one being simulated.
    AbsoluteColorimetric,
}

/// Controls which transfer function metadata the CMS trusts when building
/// a transform.
///
/// ICC profiles store transfer response curves (TRC) as `curv` or `para`
/// tags — lookup tables or parametric curves baked into the profile. Modern
/// container formats (JPEG XL, HEIF/AVIF, AV1) also carry CICP transfer
/// characteristics — an integer code that names an exact mathematical
/// transfer function (sRGB, PQ, HLG, etc.).
///
/// When both are present, they should agree — but in practice, the ICC TRC
/// may be a reduced-precision approximation of the CICP function (limited
/// by `curv` table size or `para` parameter quantization). The question is
/// which source of truth to prefer.
///
/// # Which priority to use
///
/// - **Standard ICC workflows** (JPEG, PNG, TIFF, WebP): use
///   [`PreferIcc`](Self::IccOnly). These formats don't carry CICP metadata;
///   the ICC profile is the sole authority.
///
/// - **CICP-native formats** (JPEG XL, HEIF, AVIF): use
///   [`PreferCicp`](Self::PreferCicp). The CICP code is the authoritative
///   description; the ICC profile exists for backwards compatibility with
///   older software.
///
/// # Bugs and pitfalls
///
/// - **CICP ≠ ICC is a real bug.** Some encoders embed a generic sRGB ICC
///   profile alongside a PQ or HLG CICP code. Using `PreferCicp` is correct
///   here — the ICC profile is wrong (or at best, a tone-mapped fallback).
///   Using `PreferIcc` would silently apply the wrong transfer function.
///
/// - **`PreferIcc` for CICP-native formats loses precision.** If the ICC
///   profile's `curv` table is a 1024-entry LUT approximating the sRGB
///   function, you get quantization steps in dark tones. The CICP code
///   gives the exact closed-form function — no quantization, no table
///   interpolation error.
///
/// - **`PreferCicp` for pure-ICC formats is harmless but pointless.** If
///   the profile has no embedded CICP metadata, the CMS ignores this flag
///   and falls back to the TRC. No wrong output, just a wasted branch.
///
/// - **Advisory vs. authoritative.** The ICC Votable Proposal on CICP
///   metadata in ICC profiles designates the CICP fields as *advisory*.
///   The profile's actual TRC tags remain the normative description.
///   `PreferIcc` follows this interpretation. `PreferCicp` overrides it
///   for formats where the container's CICP is known to be authoritative.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
pub enum ColorPriority {
    /// Prefer the ICC profile's own `curv`/`para` TRC curves. Ignore any
    /// embedded CICP transfer characteristics.
    ///
    /// Correct for standard ICC workflows (JPEG, PNG, TIFF, WebP) and
    /// any situation where the ICC profile is the sole color authority.
    #[default]
    PreferIcc,

    /// Allow the CMS to use CICP transfer characteristics when available.
    ///
    /// Faster (closed-form math vs. LUT interpolation) and more precise
    /// (no table quantization error). Correct only for formats where CICP
    /// is the authoritative color description: JPEG XL, HEIF, AVIF.
    PreferCicp,
}

/// Shareable, stateless row-level color transform.
///
/// Takes `&self` — the same instance can be held behind `Arc<dyn RowTransform>`
/// and reused across threads, converters, or cached for batch workloads.
/// Appropriate when the transform carries no per-call mutable state: pure
/// matrix/LUT math, moxcms `TransformExecutor` (whose `transform(&self, ...)`
/// is already `&self`), or any stateless formula-based conversion.
///
/// When the transform needs scratch buffers or per-call state, use
/// [`RowTransformMut`] instead.
pub trait RowTransform: Send + Sync {
    /// Transform one row of pixels from source to destination color space.
    ///
    /// `src` and `dst` may be different lengths if the transform changes
    /// the pixel format (e.g., CMYK to RGB). `width` is the number of
    /// pixels, not bytes.
    fn transform_row(&self, src: &[u8], dst: &mut [u8], width: u32);
}

/// Owned, stateful row-level color transform.
///
/// Takes `&mut self` — each [`RowConverter`] owns its own `Box<dyn
/// RowTransformMut>`, so implementations can reuse scratch buffers and
/// update internal state per call without interior mutability.
///
/// When the transform is stateless and could be shared, use
/// [`RowTransform`] instead — [`PluggableCms`] can offer both paths via
/// [`build_shared_source_transform`](PluggableCms::build_shared_source_transform).
///
/// [`RowConverter`]: crate::RowConverter
pub trait RowTransformMut: Send {
    /// Transform one row of pixels from source to destination color space.
    ///
    /// `src` and `dst` may be different lengths if the transform changes
    /// the pixel format (e.g., CMYK to RGB). `width` is the number of
    /// pixels, not bytes.
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32);
}

/// Color management system interface.
///
/// Abstracts over CMS backends (moxcms, lcms2, etc.) to provide
/// ICC profile transforms and profile identification.
///
/// # Feature-gated
///
/// The trait is always available for trait bounds and generic code.
/// Concrete implementations are provided by feature-gated modules
/// (e.g., `cms-moxcms`).
///
/// # Deprecated
///
/// Prefer [`PluggableCms`] for new code. `ColorManagement` is generic,
/// not dyn-safe, and takes raw ICC byte pairs; `PluggableCms` is
/// dyn-safe, accepts [`ColorProfileSource`] (ICC / CICP / named /
/// primaries+transfer), carries [`ConvertOptions`], and composes into
/// the dispatch chain used by
/// [`RowConverter::new_explicit_with_cms`](crate::RowConverter::new_explicit_with_cms).
///
/// [`ColorProfileSource`]: crate::ColorProfileSource
/// [`ConvertOptions`]: crate::policy::ConvertOptions
#[deprecated(
    since = "0.2.8",
    note = "use PluggableCms (dyn-safe, ColorProfileSource-based)"
)]
pub trait ColorManagement {
    /// Error type for CMS operations.
    type Error: core::fmt::Debug;

    /// Build a row-level transform between two ICC profiles.
    ///
    /// Returns a [`RowTransform`] that converts pixel rows from the
    /// source profile's color space to the destination profile's.
    ///
    /// This method assumes u8 RGB pixel data. For format-aware transforms
    /// that match the actual source/destination bit depth and layout, use
    /// [`build_transform_for_format`](Self::build_transform_for_format).
    fn build_transform(
        &self,
        src_icc: &[u8],
        dst_icc: &[u8],
    ) -> Result<Box<dyn RowTransform>, Self::Error>;

    /// Build a format-aware row-level transform between two ICC profiles.
    ///
    /// Like [`build_transform`](Self::build_transform), but the CMS backend
    /// can use the pixel format information to create a transform at the
    /// native bit depth (u8, u16, or f32) and layout (RGB, RGBA, Gray, etc.),
    /// avoiding unnecessary depth conversions.
    ///
    /// The default implementation ignores the format parameters and delegates
    /// to [`build_transform`](Self::build_transform).
    fn build_transform_for_format(
        &self,
        src_icc: &[u8],
        dst_icc: &[u8],
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> Result<Box<dyn RowTransform>, Self::Error> {
        let _ = (src_format, dst_format);
        self.build_transform(src_icc, dst_icc)
    }

    /// Identify whether an ICC profile matches a known CICP combination.
    ///
    /// Two-tier matching:
    /// 1. Hash table of known ICC byte sequences for instant lookup.
    /// 2. Semantic comparison: parse matrix + TRC, compare against known
    ///    values within tolerance.
    ///
    /// Returns `Some(cicp)` if the profile matches a standard combination,
    /// `None` if the profile is custom.
    fn identify_profile(&self, icc: &[u8]) -> Option<crate::Cicp>;

    // TODO(0.3.0): Add build_source_transform(ColorProfileSource, ...) as the
    // single entry point, replacing build_transform / build_transform_for_format.
    // Deferred until the trait is redesigned with options (rendering intent, HDR
    // policy) and ZenCmsLite is benchmarked against moxcms on all platforms.
}

/// Dyn-compatible CMS plugin interface for overriding gamut/profile
/// conversions inside a [`ConvertPlan`].
///
/// When a `PluggableCms` is passed to
/// [`ConvertPlan::new_explicit_with_cms`] (or the matching `RowConverter`
/// constructor) and the source and destination profiles differ, the plan
/// asks the plugin whether it will handle the exact `(src_format,
/// dst_format)` pair. If the plugin returns a transform, the plan
/// collapses to a single [`ConvertStep::ExternalTransform`] that drives
/// the row end-to-end — built-in linearize → gamut-matrix → encode steps
/// (and their fused matlut kernels) are bypassed for that conversion.
/// If the plugin returns `None`, the plan falls back to the built-in
/// path.
///
/// `PluggableCms` is intentionally narrower than [`ColorManagement`]:
/// - It accepts [`ColorProfileSource`] instead of raw ICC bytes, so
///   plugins can use primaries/transfer shortcuts, named profiles, CICP,
///   or ICC without forcing the caller to serialize to ICC.
/// - It receives [`ConvertOptions`] so plugins can honor
///   `clip_out_of_gamut` and future fields like rendering intent.
/// - It is dyn-compatible (no associated `Error` type; no generics).
///   This is what lets it live behind `&dyn PluggableCms` in API
///   signatures without forcing every caller to monomorphize.
///
/// Returning `None` is a declaration ("this CMS does not handle this
/// pair"), not an error. Plugins that recognize the profiles but fail to
/// build a transform should log/report internally and still return
/// `None` so the plan falls back cleanly.
///
/// [`ColorProfileSource`]: crate::ColorProfileSource
/// [`ConvertOptions`]: crate::policy::ConvertOptions
pub trait PluggableCms: Send + Sync {
    /// Attempt to build an owned, stateful row transform covering the full
    /// source → destination conversion for the given pixel formats.
    ///
    /// `options` carries policy flags the plugin may honor (e.g.,
    /// `clip_out_of_gamut`). The plugin is free to ignore fields that
    /// don't apply to its implementation.
    ///
    /// Return `None` to decline (plan falls back to built-in steps, or to
    /// [`build_shared_source_transform`](Self::build_shared_source_transform)
    /// if the caller tried that path and it also returned `None`).
    fn build_source_transform(
        &self,
        src: crate::ColorProfileSource<'_>,
        dst: crate::ColorProfileSource<'_>,
        src_format: PixelFormat,
        dst_format: PixelFormat,
        options: &crate::policy::ConvertOptions,
    ) -> Option<Box<dyn RowTransformMut>>;

    /// Optionally build a shareable, stateless row transform for the same
    /// conversion.
    ///
    /// When the transform carries no per-call mutable state, returning
    /// `Arc<dyn RowTransform>` enables sharing across threads, caching for
    /// batch workloads, and cheap `RowConverter` clones. Default returns
    /// `None` — plugins without a stateless fast path fall through to the
    /// owned [`build_source_transform`](Self::build_source_transform).
    ///
    /// `RowConverter::new_explicit_with_cms` tries this method first.
    fn build_shared_source_transform(
        &self,
        _src: crate::ColorProfileSource<'_>,
        _dst: crate::ColorProfileSource<'_>,
        _src_format: PixelFormat,
        _dst_format: PixelFormat,
        _options: &crate::policy::ConvertOptions,
    ) -> Option<alloc::sync::Arc<dyn RowTransform>> {
        None
    }
}
