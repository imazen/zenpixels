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
//! [`OutputPlan::build`](super::OutputPlan::build) handles CMS transforms
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
//!   a source→destination transform for the row worker.
//!
//! # Implementing a CMS backend
//!
//! Implement [`PluggableCms`]. A backend must:
//!
//! - Return [`CmsSupport::Declined`] for an unsupported request. Return an error
//!   when construction of a supported transform fails.
//! - Never accept an ICC profile it couldn't read.
//! - Treat `width` as the row's extent. Slices may be longer, and bytes past the row stay untouched.
//!   A slice shorter than the row is an error from `try_transform_row`.
//! - Keep no state between rows: any split of a row gives the same samples.
//! - Carry alpha exactly, at every depth.
//! - Round 8-bit samples to nearest, and keep white at white.
//!
//! `cms-contract/` in this repository tests all of this on `MoxCms` and on the lcms port.

#[cfg(any(test, feature = "cms-moxcms", feature = "cms-lcms"))]
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
    /// Execute one checked row. A failure may leave partial output.
    fn try_transform_row(
        &self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
    ) -> Result<(), whereat::At<CmsPluginError>>;

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
/// Requires `Send + Sync` so an owning [`RowConverter`] or [`OutputPlan`](crate::OutputPlan)
/// remains `Sync` (for scoped thread fan-out and `&self` borrows across async
/// `.await` points). Every method on this trait takes `&mut self`, so ordinary
/// Rust scratch fields (`Vec<f32>`, `Box<...>`, etc.) satisfy `Sync`
/// automatically. To wrap a `Send + !Sync` C-FFI handle (such as a cached
/// `lcms2::Transform<..., AllowCache>`), either construct it without cache
/// (`Flags::NO_CACHE`), or hold it in `sync_wrapper::SyncWrapper<T>` /
/// `std::sync::Mutex<T>` and access it via `.get_mut()` inside `try_transform_row`
/// (which compiles to a zero-locking pointer access because `&mut self` proves
/// exclusive ownership).
///
/// When the transform is stateless and could be shared, use
/// [`RowTransform`] instead. [`PluggableCms::build`] returns the supported ownership.
///
/// [`RowConverter`]: crate::RowConverter
pub trait RowTransformMut: Send + Sync {
    /// Execute one checked row. A failure may leave partial output.
    fn try_transform_row(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
    ) -> Result<(), whereat::At<CmsPluginError>>;

    /// Allocate backend scratch for bounded-width execution before reading pixels.
    fn prepare(&mut self, _max_width: u32) -> Result<(), whereat::At<CmsPluginError>> {
        Err(whereat::at!(CmsPluginError::msg(
            "backend does not support explicit preparation"
        )))
    }
}

/// Dispatch a color transform for the supplied stage contracts.
pub trait PluggableCms: Send + Sync {
    /// An unsupported request returns [CmsSupport::Declined]. A backend error
    /// terminates dispatch and preserves its source error.
    fn build(&self, request: &CmsRequest<'_>) -> Result<CmsSupport, whereat::At<CmsPluginError>>;
}

/// Profiles, carriers and policy for one CMS stage.
#[derive(Clone, Debug)]
pub struct CmsRequest<'a> {
    src: crate::ColorProfileSource<'a>,
    dst: crate::ColorProfileSource<'a>,
    from: crate::PixelDescriptor,
    to: crate::PixelDescriptor,
    options: crate::ConvertOptions,
}

impl<'a> CmsRequest<'a> {
    /// Specify the interpreted endpoints of a CMS operation.
    pub fn new(
        src: crate::ColorProfileSource<'a>,
        dst: crate::ColorProfileSource<'a>,
        from: crate::PixelDescriptor,
        to: crate::PixelDescriptor,
        options: crate::ConvertOptions,
    ) -> Self {
        Self {
            src,
            dst,
            from,
            to,
            options,
        }
    }

    /// Source profile.
    pub fn source(&self) -> crate::ColorProfileSource<'a> {
        self.src.clone()
    }

    /// Target profile.
    pub fn target(&self) -> crate::ColorProfileSource<'a> {
        self.dst.clone()
    }

    /// Source descriptor, including association and range.
    pub const fn from(&self) -> crate::PixelDescriptor {
        self.from
    }

    /// Target descriptor, including association and range.
    pub const fn to(&self) -> crate::PixelDescriptor {
        self.to
    }

    /// Requested number and rendering policies.
    pub const fn options(&self) -> &crate::ConvertOptions {
        &self.options
    }
}

/// Operations accepted by a backend and their ownership.
pub enum CmsSupport {
    /// Another backend may handle this request.
    Declined,
    /// Stateless transform shareable between workers.
    Shared(alloc::sync::Arc<dyn RowTransform>),
    /// Exclusive transform with per-worker state.
    Owned(Box<dyn RowTransformMut>),
}

#[cfg(any(test, feature = "cms-moxcms", feature = "cms-lcms"))]
pub(crate) fn build_owned(
    backend: &dyn PluggableCms,
    src: crate::ColorProfileSource<'_>,
    dst: crate::ColorProfileSource<'_>,
    from: PixelFormat,
    to: PixelFormat,
    options: &crate::ConvertOptions,
) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>> {
    struct SharedWorker(alloc::sync::Arc<dyn RowTransform>);
    impl RowTransformMut for SharedWorker {
        fn try_transform_row(
            &mut self,
            src: &[u8],
            dst: &mut [u8],
            width: u32,
        ) -> Result<(), whereat::At<CmsPluginError>> {
            self.0.try_transform_row(src, dst, width)
        }
        fn prepare(&mut self, width: u32) -> Result<(), whereat::At<CmsPluginError>> {
            self.0.prepare(width)
        }
    }
    match backend.build(&CmsRequest::new(
        src,
        dst,
        from.descriptor(),
        to.descriptor(),
        *options,
    )) {
        Ok(CmsSupport::Declined) => None,
        Ok(CmsSupport::Owned(worker)) => Some(Ok(worker)),
        Ok(CmsSupport::Shared(worker)) => Some(Ok(Box::new(SharedWorker(worker)))),
        Err(e) => Some(Err(e)),
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

#[cfg(test)]
pub(crate) trait TestCmsExt {
    fn build_source_transform(
        &self,
        src: crate::ColorProfileSource<'_>,
        dst: crate::ColorProfileSource<'_>,
        from: PixelFormat,
        to: PixelFormat,
        options: &crate::ConvertOptions,
    ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>>;
}
#[cfg(test)]
impl<T: PluggableCms> TestCmsExt for T {
    fn build_source_transform(
        &self,
        src: crate::ColorProfileSource<'_>,
        dst: crate::ColorProfileSource<'_>,
        from: PixelFormat,
        to: PixelFormat,
        options: &crate::ConvertOptions,
    ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>> {
        build_owned(self, src, dst, from, to, options)
    }
}
