//! Color Management System (CMS) traits.
//!
//! Defines the interface for ICC profile-based color transforms. When a CMS
//! feature is enabled (e.g., `cms-moxcms`, `cms-lcms2`), the implementation
//! provides ICC-to-ICC transforms. Named profile conversions (sRGB, P3,
//! BT.2020) use hardcoded matrices and don't require a CMS.

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
