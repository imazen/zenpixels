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

use alloc::boxed::Box;

/// Row-level color transform produced by a [`ColorManagement`] implementation.
///
/// Applies an ICC-to-ICC color conversion to a row of pixel data.
pub trait RowTransform {
    /// Transform one row of pixels from source to destination color space.
    ///
    /// `src` and `dst` may be different lengths if the transform changes
    /// the pixel format (e.g., CMYK to RGB). `width` is the number of
    /// pixels, not bytes.
    fn transform_row(&self, src: &[u8], dst: &mut [u8], width: u32);
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
pub trait ColorManagement {
    /// Error type for CMS operations.
    type Error;

    /// Build a row-level transform between two ICC profiles.
    ///
    /// Returns a [`RowTransform`] that converts pixel rows from the
    /// source profile's color space to the destination profile's.
    fn build_transform(
        &self,
        src_icc: &[u8],
        dst_icc: &[u8],
    ) -> Result<Box<dyn RowTransform>, Self::Error>;

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
}
