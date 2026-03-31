//! CMS backend using [moxcms](https://crates.io/crates/moxcms).
//!
//! Provides a concrete [`ColorManagement`] implementation backed by the moxcms
//! ICC profile engine. Requires the `cms-moxcms` feature.
//!
//! # Supported formats
//!
//! Transforms are created at the native bit depth (u8, u16, or f32) and layout
//! (RGB, RGBA, Gray, GrayAlpha) of the source and destination pixel formats.
//! Formats without a direct moxcms layout mapping (Bgra, Rgbx, Bgrx, Oklab)
//! fall back to u8 RGB.
//!
//! # Example
//!
//! ```rust,ignore
//! use zenpixels_convert::cms_moxcms::MoxCms;
//! use zenpixels_convert::output::{finalize_for_output, OutputProfile};
//!
//! let ready = finalize_for_output(
//!     &buffer, &origin,
//!     OutputProfile::Icc(dst_icc.into()),
//!     PixelFormat::Rgb8,
//!     &MoxCms,
//! )?;
//! ```

use alloc::boxed::Box;
use alloc::format;
use alloc::sync::Arc;

use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, TransformExecutor,
    TransformOptions,
};

/// Standard moxcms transform options. See imageflow_core's moxcms_transform.rs for rationale.
fn lut_transform_opts() -> TransformOptions {
    TransformOptions {
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

use crate::cms::{ColorManagement, RowTransform};
use crate::{ChannelType, Cicp, PixelFormat};

/// CMS backend using moxcms.
///
/// Stateless — all configuration comes from the ICC profiles and pixel formats
/// passed to each method call. Safe to share across threads.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoxCms;

/// Map a [`PixelFormat`] to the corresponding moxcms [`Layout`].
///
/// Returns `None` for formats that don't have a direct moxcms mapping
/// (Bgra, Rgbx, Bgrx, Oklab variants).
fn pixel_format_to_layout(format: PixelFormat) -> Option<Layout> {
    match format {
        PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::RgbF32 => Some(Layout::Rgb),
        PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaF32 => Some(Layout::Rgba),
        PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayF32 => Some(Layout::Gray),
        PixelFormat::GrayA8 | PixelFormat::GrayA16 | PixelFormat::GrayAF32 => {
            Some(Layout::GrayAlpha)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// RowTransform wrapper
// ---------------------------------------------------------------------------

/// Internal wrapper around moxcms transform executors at different bit depths.
enum MoxTransformInner {
    U8(Arc<dyn TransformExecutor<u8> + Send + Sync>),
    U16(Arc<dyn TransformExecutor<u16> + Send + Sync>),
    F32(Arc<dyn TransformExecutor<f32> + Send + Sync>),
}

struct MoxRowTransform {
    inner: MoxTransformInner,
}

impl RowTransform for MoxRowTransform {
    fn transform_row(&self, src: &[u8], dst: &mut [u8], _width: u32) {
        match &self.inner {
            MoxTransformInner::U8(xform) => {
                xform
                    .transform(src, dst)
                    .expect("moxcms u8 transform: buffer size mismatch");
            }
            MoxTransformInner::U16(xform) => {
                let src_u16: &[u16] = bytemuck::cast_slice(src);
                let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(dst);
                xform
                    .transform(src_u16, dst_u16)
                    .expect("moxcms u16 transform: buffer size mismatch");
            }
            MoxTransformInner::F32(xform) => {
                let src_f32: &[f32] = bytemuck::cast_slice(src);
                let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
                xform
                    .transform(src_f32, dst_f32)
                    .expect("moxcms f32 transform: buffer size mismatch");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ColorManagement implementation
// ---------------------------------------------------------------------------

impl ColorManagement for MoxCms {
    type Error = MoxCmsError;

    fn build_transform(
        &self,
        src_icc: &[u8],
        dst_icc: &[u8],
    ) -> Result<Box<dyn RowTransform>, Self::Error> {
        // Default: u8 RGB, which covers the common JPEG/PNG/WebP case.
        self.build_transform_for_format(src_icc, dst_icc, PixelFormat::Rgb8, PixelFormat::Rgb8)
    }

    fn build_transform_for_format(
        &self,
        src_icc: &[u8],
        dst_icc: &[u8],
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> Result<Box<dyn RowTransform>, Self::Error> {
        let src_profile = ColorProfile::new_from_slice(src_icc)
            .map_err(|e| MoxCmsError(format!("failed to parse source ICC profile: {e}")))?;
        let dst_profile = ColorProfile::new_from_slice(dst_icc)
            .map_err(|e| MoxCmsError(format!("failed to parse destination ICC profile: {e}")))?;

        let src_layout = pixel_format_to_layout(src_format).unwrap_or(Layout::Rgb);
        let dst_layout = pixel_format_to_layout(dst_format).unwrap_or(Layout::Rgb);
        // CICP transfer is for applications, not CMMs (ICC Votable Proposal).
        // Matches the v2 path fix — see moxcms issue #154.
        let opts = lut_transform_opts();

        // Pick the narrower of the two channel types to avoid unnecessary
        // precision loss, but always use the source depth when both differ
        // (the CMS handles the depth conversion internally).
        let depth = src_format.channel_type();

        let inner = match depth {
            ChannelType::U8 => {
                let xform = src_profile
                    .create_transform_8bit(src_layout, &dst_profile, dst_layout, opts)
                    .map_err(|e| MoxCmsError(format!("failed to create u8 transform: {e}")))?;
                MoxTransformInner::U8(xform)
            }
            ChannelType::U16 => {
                let xform = src_profile
                    .create_transform_16bit(src_layout, &dst_profile, dst_layout, opts)
                    .map_err(|e| MoxCmsError(format!("failed to create u16 transform: {e}")))?;
                MoxTransformInner::U16(xform)
            }
            // F16 and F32 both use the f32 transform path (F16 data must be
            // converted to f32 before CMS — IEEE 754 half-floats are not
            // integer-encoded u16 values).
            ChannelType::F16 | ChannelType::F32 | _ => {
                let xform = src_profile
                    .create_transform_f32(src_layout, &dst_profile, dst_layout, opts)
                    .map_err(|e| MoxCmsError(format!("failed to create f32 transform: {e}")))?;
                MoxTransformInner::F32(xform)
            }
        };

        Ok(Box::new(MoxRowTransform { inner }))
    }

    fn identify_profile(&self, icc: &[u8]) -> Option<Cicp> {
        let profile = ColorProfile::new_from_slice(icc).ok()?;

        // If the profile has embedded CICP metadata, use it directly.
        if let Some(cicp) = &profile.cicp {
            return Some(Cicp::new(
                cicp.color_primaries as u8,
                cicp.transfer_characteristics as u8,
                cicp.matrix_coefficients as u8,
                cicp.full_range,
            ));
        }

        // Fall back to comparing colorant matrices against known profiles.
        identify_by_colorants(&profile)
    }
}

// ---------------------------------------------------------------------------
// Profile identification by colorant comparison
// ---------------------------------------------------------------------------

/// Compare XYZ colorants to identify well-known profiles.
///
/// Checks the profile's red/green/blue colorants against sRGB (BT.709),
/// Display P3, and BT.2020. The colorant values are in PCS (D50-adapted)
/// space, as stored in ICC profiles after Bradford chromatic adaptation
/// from D65. Tolerance is 0.003 in XYZ, tight enough to distinguish
/// these gamuts while tolerating s15Fixed16 quantization.
fn identify_by_colorants(profile: &ColorProfile) -> Option<Cicp> {
    // Known colorant values in D50 PCS space (Bradford-adapted from D65).
    // Computed by applying the standard D65→D50 Bradford matrix to the
    // absolute D65 XYZ colorant matrices from ITU-R specifications.
    struct KnownProfile {
        primaries_code: u8,
        rx: f64,
        ry: f64,
        gx: f64,
        gy: f64,
        bx: f64,
        by: f64,
    }

    const KNOWN: &[KnownProfile] = &[
        // sRGB / BT.709 (D50-adapted)
        KnownProfile {
            primaries_code: 1,
            rx: 0.4361,
            ry: 0.2225,
            gx: 0.3851,
            gy: 0.7169,
            bx: 0.1431,
            by: 0.0606,
        },
        // Display P3 (D50-adapted)
        KnownProfile {
            primaries_code: 12,
            rx: 0.5151,
            ry: 0.2412,
            gx: 0.2919,
            gy: 0.6922,
            bx: 0.1572,
            by: 0.0666,
        },
        // BT.2020 (D50-adapted)
        KnownProfile {
            primaries_code: 9,
            rx: 0.6734,
            ry: 0.2790,
            gx: 0.1656,
            gy: 0.6753,
            bx: 0.1251,
            by: 0.0456,
        },
    ];

    let r = &profile.red_colorant;
    let g = &profile.green_colorant;
    let b = &profile.blue_colorant;

    const TOL: f64 = 0.003;

    for known in KNOWN {
        let matches = (r.x - known.rx).abs() < TOL
            && (r.y - known.ry).abs() < TOL
            && (g.x - known.gx).abs() < TOL
            && (g.y - known.gy).abs() < TOL
            && (b.x - known.bx).abs() < TOL
            && (b.y - known.by).abs() < TOL;

        if matches {
            // Map known primaries to their standard transfer characteristic.
            // sRGB (1) and Display P3 (12) both use the sRGB TRC (13).
            // BT.2020 (9) uses BT.709 TRC (1) as a safe default since
            // the actual TRC (PQ, HLG, or BT.709) can't be identified
            // from colorants alone.
            let transfer = match known.primaries_code {
                1 | 12 => 13, // sRGB and Display P3 use sRGB TRC
                _ => 1,       // BT.2020 etc. default to BT.709 TRC
            };
            return Some(Cicp::new(
                known.primaries_code,
                transfer,
                0, // Identity (RGB)
                true,
            ));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error from the moxcms CMS backend.
#[derive(Debug, Clone)]
pub struct MoxCmsError(pub String);

impl core::fmt::Display for MoxCmsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}
