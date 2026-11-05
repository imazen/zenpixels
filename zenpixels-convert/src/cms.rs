//! Color Management System (CMS) traits.
//!
//! Defines the interface for ICC profile-based color transforms. When a CMS
//! feature is enabled (`cms-moxcms`), the implementation
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
//! Implement [`PluggableCms`]. A backend must:
//!
//! - Return `None` for a conversion it doesn't take, and `Some(Err)` once it took one and failed.
//! - Never accept an ICC profile it couldn't read.
//! - Treat `width` as the row's extent. Slices may be longer, and bytes past the row stay untouched.
//!   A slice shorter than the row is an error from `try_transform_row`.
//! - Keep no state between rows: any split of a row gives the same samples.
//! - Carry alpha exactly, at every depth.
//! - Round 8-bit samples to nearest, and keep white at white.
//!
//! `cms-contract/` in this repository tests all of this on `MoxCms` and on the lcms port.

use crate::PixelFormat;
use alloc::boxed::Box;

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

    /// Fallible execution hook. Legacy implementations retain their existing
    /// behavior; backends with row errors should override this method.
    fn try_transform_row(
        &self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
    ) -> Result<(), whereat::At<CmsPluginError>> {
        self.transform_row(src, dst, width);
        Ok(())
    }

    /// Perform backend setup for bounded-width execution. The default refuses:
    /// a legacy backend has not promised allocation-free execution.
    fn prepare(&self, _max_width: u32) -> Result<(), whereat::At<CmsPluginError>> {
        Err(whereat::at!(CmsPluginError::msg(
            "backend does not support explicit preparation"
        )))
    }
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
    /// pixels, not bytes. Either slice may extend past the row, and those bytes stay untouched.
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32);

    /// Fallible execution hook, preserving source compatibility for old backends.
    fn try_transform_row(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
    ) -> Result<(), whereat::At<CmsPluginError>> {
        self.transform_row(src, dst, width);
        Ok(())
    }

    /// Allocate backend scratch for bounded-width execution before reading pixels.
    fn prepare(&mut self, _max_width: u32) -> Result<(), whereat::At<CmsPluginError>> {
        Err(whereat::at!(CmsPluginError::msg(
            "backend does not support explicit preparation"
        )))
    }
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
/// conversions inside a [`ConvertPlan`](crate::ConvertPlan).
///
/// When a `PluggableCms` is passed to
/// [`RowConverter::new_explicit_with_cms`](crate::RowConverter::new_explicit_with_cms)
/// and the source and destination profiles differ, the plan asks the
/// plugin whether it will handle the exact `(src_format, dst_format)`
/// pair. If the plugin returns a transform, the plan collapses to a
/// single external-transform step that drives the row end-to-end —
/// built-in linearize → gamut-matrix → encode steps (and their fused
/// matlut kernels) are bypassed for that conversion. If the plugin
/// returns `None`, the plan falls back to the built-in path.
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
/// # Decline vs. fail
///
/// Plugin methods return `Option<Result<T, CmsPluginError>>` with three
/// outcomes:
/// - `None` — declined ("not my problem"). The dispatch chain continues
///   to the next plugin (typically `ZenCmsLite`) or falls through to the
///   built-in path.
/// - `Some(Ok(transform))` — accepted. The dispatch chain stops here.
/// - `Some(Err(e))` — tried-and-failed. The error propagates immediately;
///   **the chain does not continue**. If a plugin took ownership of a
///   conversion and failed, we surface that rather than silently producing
///   different output from a fallback backend.
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
    /// See the trait docs for decline vs. fail semantics.
    ///
    /// The `Err` arm is [`whereat::At<CmsPluginError>`] so the plugin's
    /// internal failure point is recorded for debugging. Use
    /// [`whereat::at!`] or `ResultAtExt::at()` to construct.
    fn build_source_transform(
        &self,
        src: crate::ColorProfileSource<'_>,
        dst: crate::ColorProfileSource<'_>,
        src_format: PixelFormat,
        dst_format: PixelFormat,
        options: &crate::policy::ConvertOptions,
    ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>>;

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
    /// See the trait docs for decline vs. fail semantics. `Err` arm is
    /// [`whereat::At<CmsPluginError>`] — same location-tracking semantics
    /// as [`build_source_transform`](Self::build_source_transform).
    fn build_shared_source_transform(
        &self,
        _src: crate::ColorProfileSource<'_>,
        _dst: crate::ColorProfileSource<'_>,
        _src_format: PixelFormat,
        _dst_format: PixelFormat,
        _options: &crate::policy::ConvertOptions,
    ) -> Option<Result<alloc::sync::Arc<dyn RowTransform>, whereat::At<CmsPluginError>>> {
        None
    }
}

/// Error produced by a [`PluggableCms`] when a plugin recognized a
/// conversion pair but failed to build a transform.
///
/// Type-erased wrapper over any `core::error::Error + Send + Sync`. Use
/// [`CmsPluginError::new`] or [`From`] to construct.
pub struct CmsPluginError(Box<dyn core::error::Error + Send + Sync + 'static>);

// Opaque backend errors have no value-equality contract. Shared errors retain
// identity through ConvertError clones without imposing Eq on plugin errors.
impl PartialEq for CmsPluginError {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self, other)
    }
}

impl CmsPluginError {
    /// Construct from any error that implements `core::error::Error`.
    pub fn new<E>(err: E) -> Self
    where
        E: core::error::Error + Send + Sync + 'static,
    {
        Self(Box::new(err))
    }

    /// Construct from a message string.
    pub fn msg(s: impl Into<alloc::string::String>) -> Self {
        struct Msg(alloc::string::String);
        impl core::fmt::Debug for Msg {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl core::fmt::Display for Msg {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl core::error::Error for Msg {}
        Self(Box::new(Msg(s.into())))
    }

    /// Borrow the inner error.
    pub fn as_inner(&self) -> &(dyn core::error::Error + Send + Sync + 'static) {
        &*self.0
    }
}

impl core::fmt::Debug for CmsPluginError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("CmsPluginError").field(&self.0).finish()
    }
}

impl core::fmt::Display for CmsPluginError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.0, f)
    }
}

impl core::error::Error for CmsPluginError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}
