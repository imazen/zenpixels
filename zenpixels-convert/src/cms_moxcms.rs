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

use crate::cms::{ColorPriority, RenderingIntent};

/// Build moxcms [`TransformOptions`] from a [`ColorPriority`] and
/// [`RenderingIntent`].
///
/// This is the single entry point for constructing moxcms transform options.
/// It applies our quality defaults (tetrahedral interpolation, high-precision
/// barycentric weights) and maps the backend-agnostic enums to moxcms types.
///
/// # Parameters
///
/// - `priority` — which transfer function metadata to trust. Use
///   [`ColorPriority::PreferIcc`] for standard ICC workflows (JPEG, PNG, TIFF,
///   WebP). Use [`ColorPriority::PreferCicp`] for CICP-native formats (JPEG XL,
///   HEIF, AVIF) where the CICP code is the authoritative description and the
///   ICC profile is a backwards-compatibility fallback.
///
/// - `intent` — ICC rendering intent. Use
///   [`RenderingIntent::RelativeColorimetric`] (the default) for display output.
///   See [`RenderingIntent`] docs for when to use other intents.
///
/// # Quality settings
///
/// The following are always applied regardless of arguments:
///
/// - **Tetrahedral interpolation** over trilinear for 3D CLUTs. Produces
///   higher accuracy in saturated regions where trilinear interpolation
///   crosses cube diagonals. No measurable performance cost for the image
///   sizes we handle.
///
/// - **High barycentric weight scale.** Cuts LUT interpolation error from
///   max ≤ 14 to max ≤ 2 (code values, u8 scale) vs. lcms2 for standard
///   ICC LUT profiles. The 5% performance cost cited in moxcms docs is
///   negligible at our call granularity (row-level transforms, not
///   pixel-level).
///
/// # Rendering intent vs. profile LUT availability
///
/// Requesting an intent whose LUT is absent in the profile causes a silent
/// fallback to the profile's default intent (typically relative colorimetric).
/// Most display profiles only ship one LUT. See [`RenderingIntent`] docs for
/// details on which profiles actually honor which intents.
///
/// # Examples
///
/// ```rust,ignore
/// use zenpixels_convert::cms::{ColorPriority, RenderingIntent};
/// use zenpixels_convert::cms_moxcms::transform_opts;
///
/// // Standard ICC workflow (JPEG, PNG, etc.)
/// let opts = transform_opts(ColorPriority::PreferIcc, RenderingIntent::RelativeColorimetric);
///
/// // JPEG XL decode — trust CICP transfer characteristics
/// let opts = transform_opts(ColorPriority::PreferCicp, RenderingIntent::RelativeColorimetric);
///
/// // Soft-proofing: simulate print appearance on screen
/// let opts = transform_opts(ColorPriority::PreferIcc, RenderingIntent::AbsoluteColorimetric);
/// ```
pub fn transform_opts(priority: ColorPriority, intent: RenderingIntent) -> TransformOptions {
    TransformOptions {
        rendering_intent: match intent {
            RenderingIntent::Perceptual => moxcms::RenderingIntent::Perceptual,
            RenderingIntent::RelativeColorimetric => moxcms::RenderingIntent::RelativeColorimetric,
            RenderingIntent::Saturation => moxcms::RenderingIntent::Saturation,
            RenderingIntent::AbsoluteColorimetric => moxcms::RenderingIntent::AbsoluteColorimetric,
        },
        allow_use_cicp_transfer: matches!(priority, ColorPriority::PreferCicp),
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

/// Standard moxcms transform options for ICC LUT transforms.
///
/// # Deprecated
///
/// Use [`transform_opts`]`(`[`ColorPriority::PreferIcc`]`,
/// `[`RenderingIntent::RelativeColorimetric`]`)` instead, which lets you
/// specify the rendering intent explicitly.
#[deprecated(
    since = "0.2.3",
    note = "use transform_opts(ColorPriority::PreferIcc, RenderingIntent::RelativeColorimetric) instead"
)]
pub fn lut_transform_opts() -> TransformOptions {
    transform_opts(
        ColorPriority::PreferIcc,
        RenderingIntent::RelativeColorimetric,
    )
}

/// Standard moxcms transform options for CICP-native formats (e.g. JXL, HEIF).
///
/// # Deprecated
///
/// Use [`transform_opts`]`(`[`ColorPriority::PreferCicp`]`,
/// `[`RenderingIntent::RelativeColorimetric`]`)` instead, which lets you
/// specify the rendering intent explicitly.
#[deprecated(
    since = "0.2.3",
    note = "use transform_opts(ColorPriority::PreferCicp, RenderingIntent::RelativeColorimetric) instead"
)]
pub fn cicp_transform_opts() -> TransformOptions {
    transform_opts(
        ColorPriority::PreferCicp,
        RenderingIntent::RelativeColorimetric,
    )
}

#[allow(deprecated)]
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
///
/// **CMYK note.** `Cmyk8` maps to [`Layout::Rgba`] per moxcms's
/// convention (see `moxcms::Layout` docs: "Cmyk8 uses the same layout
/// as Rgba8"). The `DataColorSpace` on the source profile is what
/// distinguishes CMYK from RGBA in moxcms — the layout is just a
/// channel-count + interleave hint.
fn pixel_format_to_layout(format: PixelFormat) -> Option<Layout> {
    match format {
        PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::RgbF32 => Some(Layout::Rgb),
        PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaF32 => Some(Layout::Rgba),
        PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayF32 => Some(Layout::Gray),
        PixelFormat::GrayA8 | PixelFormat::GrayA16 | PixelFormat::GrayAF32 => {
            Some(Layout::GrayAlpha)
        }
        // CMYK shares the 4-channel interleaved layout with RGBA in moxcms;
        // moxcms's `check_layout` validates `DataColorSpace::Cmyk` against
        // `Layout::Rgba` (see moxcms/src/profile.rs).
        PixelFormat::Cmyk8 => Some(Layout::Rgba),
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

#[allow(deprecated)]
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

/// Build a [`RowTransform`] from two already-parsed [`ColorProfile`]s.
///
/// Shared implementation for both ICC-to-ICC and CICP-to-ICC paths.
/// Always uses `PreferIcc` / `RelativeColorimetric` — CICP-in-ICC tags
/// are never trusted for TRC (see moxcms issue #154).
fn build_transform_inner(
    src_profile: &ColorProfile,
    dst_profile: &ColorProfile,
    src_format: PixelFormat,
    dst_format: PixelFormat,
) -> Result<Box<dyn RowTransform>, MoxCmsError> {
    let src_layout = pixel_format_to_layout(src_format).unwrap_or(Layout::Rgb);
    let dst_layout = pixel_format_to_layout(dst_format).unwrap_or(Layout::Rgb);
    let opts = transform_opts(ColorPriority::PreferIcc, RenderingIntent::default());

    let depth = src_format.channel_type();

    let inner = match depth {
        ChannelType::U8 => {
            let xform = src_profile
                .create_transform_8bit(src_layout, dst_profile, dst_layout, opts)
                .map_err(|e| MoxCmsError(format!("failed to create u8 transform: {e}")))?;
            MoxTransformInner::U8(xform)
        }
        ChannelType::U16 => {
            let xform = src_profile
                .create_transform_16bit(src_layout, dst_profile, dst_layout, opts)
                .map_err(|e| MoxCmsError(format!("failed to create u16 transform: {e}")))?;
            MoxTransformInner::U16(xform)
        }
        // F16 and F32 both use the f32 transform path (F16 data must be
        // converted to f32 before CMS — IEEE 754 half-floats are not
        // integer-encoded u16 values).
        ChannelType::F16 | ChannelType::F32 | _ => {
            let xform = src_profile
                .create_transform_f32(src_layout, dst_profile, dst_layout, opts)
                .map_err(|e| MoxCmsError(format!("failed to create f32 transform: {e}")))?;
            MoxTransformInner::F32(xform)
        }
    };

    Ok(Box::new(MoxRowTransform { inner }))
}

#[allow(deprecated)]
impl ColorManagement for MoxCms {
    type Error = MoxCmsError;

    fn build_transform(
        &self,
        src_icc: &[u8],
        dst_icc: &[u8],
    ) -> Result<Box<dyn RowTransform>, Self::Error> {
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

        build_transform_inner(&src_profile, &dst_profile, src_format, dst_format)
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

    // TODO(0.3.0): implement build_source_transform once the trait method
    // is added. The plumbing (source_to_moxcms_profile) is already here.
}

// ---------------------------------------------------------------------------
// PluggableCms — the dispatch chain RowConverter actually consults.
// ---------------------------------------------------------------------------

/// `RowTransformMut` wrapper for moxcms transform executors at the three
/// supported bit depths. Mirrors [`MoxRowTransform`] but exposes the
/// `RowTransformMut` (`&mut self`) shape that [`PluggableCms`] expects.
///
/// moxcms `TransformExecutor::transform` is `&self`, so there's no actual
/// per-call mutable state — the `&mut self` shape is a trait-level
/// convenience and matches the [`RowConverter`] ownership model
/// (`Box<dyn RowTransformMut>` per converter).
///
/// [`RowConverter`]: crate::RowConverter
struct MoxRowTransformMut {
    inner: MoxTransformInner,
}

impl crate::cms::RowTransformMut for MoxRowTransformMut {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], _width: u32) {
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

impl crate::cms::PluggableCms for MoxCms {
    /// Build a moxcms-backed row transform for the given
    /// `(src, dst, src_format, dst_format)`.
    ///
    /// **Decline (`None`)** when either source can't be mapped to a moxcms
    /// `ColorProfile` (custom signature we don't recognize), when the
    /// pixel formats don't have a `Layout` mapping, or when the
    /// `(src, dst, src_format, dst_format)` tuple is the trivial identity
    /// (let the built-in mechanical plan handle it).
    ///
    /// **Fail (`Some(Err(_))`)** when we recognized the pair and started
    /// to build profiles or a transform but the construction itself
    /// failed (ICC parse errors, CMYK-without-ICC, moxcms's
    /// `check_layout` rejecting the combination, …). The dispatch chain
    /// stops here — falling back to ZenCmsLite or the built-in plan would
    /// silently produce different output.
    ///
    /// **CMYK ↔ RGB** is the primary new path enabled by this impl. moxcms
    /// requires a real CMYK ICC profile to populate the device→PCS LUT
    /// (no synthesizable default exists for a device-dependent ink
    /// space), so callers must pass `ColorProfileSource::Icc(...)` for
    /// the CMYK side. A `PrimariesTransferPair` for a CMYK descriptor
    /// without an attached ICC is declined (`None`) rather than failed —
    /// upstream paths (the no-CMS extension entry points) already
    /// answer `NeedsCms`, and we want the user to provide an actual ICC.
    fn build_source_transform(
        &self,
        src: crate::ColorProfileSource<'_>,
        dst: crate::ColorProfileSource<'_>,
        src_format: PixelFormat,
        dst_format: PixelFormat,
        _options: &crate::policy::ConvertOptions,
    ) -> Option<Result<Box<dyn crate::cms::RowTransformMut>, whereat::At<crate::cms::CmsPluginError>>>
    {
        use crate::cms::CmsPluginError;
        // Decline when the format pair has no `Layout` mapping (e.g. Bgra
        // swizzles, Rgbx alpha-padding, Oklab variants). moxcms can't
        // describe those; let the built-in pipeline handle layout
        // shuffling and route the colorimetric work back through after.
        let src_layout = pixel_format_to_layout(src_format)?;
        let dst_layout = pixel_format_to_layout(dst_format)?;

        // Build moxcms profiles. CMYK as a `PrimariesTransferPair` has no
        // synthesizable mapping — decline so the caller knows to attach an
        // ICC via `ColorProfileSource::Icc(...)`.
        let src_profile = match build_moxcms_profile_for_format(&src, src_format) {
            Ok(Some(p)) => p,
            Ok(None) => return None,
            Err(e) => {
                return Some(Err(whereat::at!(CmsPluginError::msg(format!(
                    "moxcms source profile build failed: {e}"
                )))));
            }
        };
        let dst_profile = match build_moxcms_profile_for_format(&dst, dst_format) {
            Ok(Some(p)) => p,
            Ok(None) => return None,
            Err(e) => {
                return Some(Err(whereat::at!(CmsPluginError::msg(format!(
                    "moxcms destination profile build failed: {e}"
                )))));
            }
        };

        // Identity bytes-out (same profile, same format) is left to the
        // built-in mechanical plan via `None`; the dispatch chain falls
        // through to `ConvertPlan::new_explicit` which emits an Identity
        // step. But this only fires for the truly trivial case — the
        // mismatch dispatch above already handled CMYK by reaching this
        // function, so a CMYK→CMYK identity is fine to decline.
        //
        // We don't have a cheap profile-equality check, so we just attempt
        // the transform; moxcms returns Ok with a no-op LUT when the
        // device→PCS→device round-trip is identity.

        let opts = transform_opts(ColorPriority::PreferIcc, RenderingIntent::default());

        // Dispatch on the source `ChannelType`. F16 is widened to f32
        // before the CMS step in this crate, so we route F16 to the f32
        // transform path (the same convention `build_transform_inner`
        // above uses). Note: when src and dst differ on channel type, we
        // pick the source side; the layout/bit-depth conversion happens
        // outside moxcms in a separate plan step.
        let depth = src_format.channel_type();
        let inner_result = match depth {
            ChannelType::U8 => src_profile
                .create_transform_8bit(src_layout, &dst_profile, dst_layout, opts)
                .map(MoxTransformInner::U8),
            ChannelType::U16 => src_profile
                .create_transform_16bit(src_layout, &dst_profile, dst_layout, opts)
                .map(MoxTransformInner::U16),
            ChannelType::F16 | ChannelType::F32 | _ => src_profile
                .create_transform_f32(src_layout, &dst_profile, dst_layout, opts)
                .map(MoxTransformInner::F32),
        };

        let inner = match inner_result {
            Ok(i) => i,
            Err(e) => {
                return Some(Err(whereat::at!(CmsPluginError::msg(format!(
                    "moxcms create_transform_{:?}bit failed: {e}",
                    depth
                )))));
            }
        };

        Some(Ok(Box::new(MoxRowTransformMut { inner })))
    }
}

/// Build a moxcms `ColorProfile` for the given source, with a
/// pixel-format hint that determines whether the profile must describe
/// CMYK ink (no synthesizable default) or an RGB / Gray colorimetric
/// space (synthesizable from primaries + transfer).
///
/// Outcomes mirror the `PluggableCms` decline-vs-fail contract:
/// - `Ok(Some(profile))` — we built a profile and the caller can use it.
/// - `Ok(None)` — we declined (no information / not our problem); the
///   caller should keep walking the dispatch chain.
/// - `Err(_)` — we tried (recognized the inputs) but construction
///   failed; the caller should surface as a tried-and-failed.
fn build_moxcms_profile_for_format(
    src: &crate::ColorProfileSource<'_>,
    format: PixelFormat,
) -> Result<Option<ColorProfile>, MoxCmsError> {
    let is_cmyk = matches!(format, PixelFormat::Cmyk8);
    match src {
        // ICC bytes are authoritative for every color model — parse and
        // hand them straight to moxcms. The profile's `data_color_space`
        // will validate against the layout downstream
        // (`check_layout`).
        crate::ColorProfileSource::Icc(icc) => ColorProfile::new_from_slice(icc)
            .map(Some)
            .map_err(|e| MoxCmsError(format!("failed to parse ICC: {e}"))),
        // CICP describes RGB colorimetry; CMYK descriptors with CICP
        // (which is unusual — CICP rarely tags CMYK) decline so the
        // caller can attach a real CMYK ICC instead.
        crate::ColorProfileSource::Cicp(cicp) if !is_cmyk => Ok(Some(cicp_to_moxcms_profile(cicp))),
        crate::ColorProfileSource::Named(named) if !is_cmyk => {
            let (p, t) = named.to_primaries_transfer();
            primaries_transfer_to_moxcms_profile(p, t)
        }
        crate::ColorProfileSource::PrimariesTransferPair {
            primaries,
            transfer,
        } if !is_cmyk => primaries_transfer_to_moxcms_profile(*primaries, *transfer),
        // CMYK without ICC bytes: decline. The PluggableCms chain falls
        // through to the built-in plan path, which answers
        // `NeedsCms` so the caller knows to attach an ICC.
        _ => Ok(None),
    }
}

/// Convert a [`ColorProfileSource`](crate::ColorProfileSource) to a moxcms [`ColorProfile`].
///
/// Returns `Ok(None)` if the source can't be mapped to moxcms.
// TODO(0.3.0): used by build_source_transform once trait is redesigned.
#[allow(dead_code)]
fn source_to_moxcms_profile(
    src: &crate::ColorProfileSource<'_>,
) -> Result<Option<ColorProfile>, MoxCmsError> {
    match src {
        crate::ColorProfileSource::Icc(icc) => ColorProfile::new_from_slice(icc)
            .map(Some)
            .map_err(|e| MoxCmsError(format!("failed to parse ICC: {e}"))),
        crate::ColorProfileSource::Cicp(cicp) => Ok(Some(cicp_to_moxcms_profile(cicp))),
        crate::ColorProfileSource::Named(named) => {
            let (p, t) = named.to_primaries_transfer();
            primaries_transfer_to_moxcms_profile(p, t)
        }
        crate::ColorProfileSource::PrimariesTransferPair {
            primaries,
            transfer,
        } => primaries_transfer_to_moxcms_profile(*primaries, *transfer),
        _ => Ok(None),
    }
}

/// Generate ICC profile bytes for a CICP via moxcms, or `None` if moxcms doesn't
/// recognize the **color-defining** code points (primaries / transfer).
///
/// Strict on purpose: unlike [`cicp_to_moxcms_profile`] (the transform path, which
/// falls back to Bt709/sRGB defaults), synthesis must never emit a profile whose
/// TRC/gamut contradicts the source — a `None` here surfaces as
/// [`SynthesizedIcc::CmsUnsupported`](crate::icc_profiles::SynthesizedIcc::CmsUnsupported)
/// so the caller carries the color via CICP instead of embedding a wrong profile.
/// Matrix coefficients are irrelevant to an RGB ICC, so they're defaulted rather
/// than required.
///
/// Test-only: the bundled blob (generated from this exact logic at build time) is
/// the runtime coverage source, so `synthesize_icc_for_cicp` no longer calls this.
/// It's retained as the oracle the `blob_decodes_byte_identical_to_moxcms` guard
/// compares the committed blob against — catching a moxcms version bump that would
/// shift the canonical bytes.
#[cfg(test)]
pub(crate) fn icc_bytes_for_cicp(cicp: &Cicp) -> Option<alloc::vec::Vec<u8>> {
    // `try_from` on these moxcms enums never errors: every u8 maps to a variant,
    // with reserved/unassigned codes folding into `Reserved`. So these conversions
    // are NOT the validity gate — the real check is whether moxcms could populate
    // the colorimetry below.
    let color_primaries = moxcms::CicpColorPrimaries::try_from(cicp.color_primaries).ok()?;
    let transfer_characteristics =
        moxcms::TransferCharacteristics::try_from(cicp.transfer_characteristics).ok()?;
    // Matrix coefficients don't affect an RGB ICC's colorimetry; default rather
    // than reject so an unusual matrix code doesn't block synthesis.
    let matrix_coefficients = moxcms::MatrixCoefficients::try_from(cicp.matrix_coefficients)
        .unwrap_or(moxcms::MatrixCoefficients::Identity);

    let profile = ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        full_range: cicp.full_range,
    });

    // `new_from_cicp` discards the bool from `update_rgb_colorimetry_from_cicp`, so
    // for `Reserved`/`Unspecified` primaries or transfer it silently returns a base
    // profile with no colorants and no TRC. moxcms sets `red_trc` only after every
    // primaries + white-point + transfer-curve gate passes, so a populated `red_trc`
    // is the signal synthesis was faithful. No TRC ⇒ moxcms can't represent this
    // CICP — bail with None (surfaces as `CmsUnsupported`) rather than emit a
    // degenerate profile whose colorimetry omits or contradicts the requested color.
    profile.red_trc.as_ref()?;
    profile.encode().ok()
}

/// Synthesize a **GRAY-class** ICC for a CICP, exactly as `icc-gen`'s
/// `cicp_bundle_gen` generator does for the committed gray bundle: `kTRC` =
/// the transfer's tone curve (taken from a throwaway RGB synthesis so the
/// gray and RGB recipes can never disagree about a curve), media white point
/// = the primaries' H.273 white, and a per-white Bradford white→D50 `chad`.
/// The generator zeroes the creation timestamp for reproducibility; this
/// fresh path leaves it — the roundtrip test masks bytes 24..36 on both
/// sides.
///
/// Test-only, mirroring [`icc_bytes_for_cicp`]: the bundled gray blob is
/// the runtime coverage source; this is the oracle the gray
/// `blob_decodes_byte_identical_to_moxcms` guard compares the committed
/// blob against.
#[cfg(test)]
pub(crate) fn gray_icc_bytes_for_cicp(cicp: &Cicp) -> Option<alloc::vec::Vec<u8>> {
    let color_primaries = moxcms::CicpColorPrimaries::try_from(cicp.color_primaries).ok()?;
    let transfer_characteristics =
        moxcms::TransferCharacteristics::try_from(cicp.transfer_characteristics).ok()?;
    let matrix_coefficients = moxcms::MatrixCoefficients::try_from(cicp.matrix_coefficients)
        .unwrap_or(moxcms::MatrixCoefficients::Identity);

    let rgb = ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        full_range: cicp.full_range,
    });
    // Same faithful-synthesis gate as the RGB path.
    let trc = rgb.red_trc.as_ref()?.clone();

    let (white_name, wx, wy) = h273_white_xy(cicp.color_primaries)?;
    let white = moxcms::Xyzd {
        x: wx / wy,
        y: 1.0,
        z: (1.0 - wx - wy) / wy,
    };

    let mut gray = ColorProfile::new_gray_with_gamma(2.2);
    gray.gray_trc = Some(trc);
    gray.media_white_point = Some(white);
    gray.chromatic_adaptation = Some(moxcms::adaption_matrix_d(
        white.to_xyz(),
        moxcms::white_point_d50().to_xyz(),
    ));
    gray.description = Some(moxcms::ProfileText::Localizable(alloc::vec![
        moxcms::LocalizableString::new(
            "en".into(),
            "US".into(),
            format!(
                "Gray H.273 TC{} {white_name} white",
                cicp.transfer_characteristics
            ),
        )
    ]));

    gray.encode().ok()
}

/// The white point of an H.273 colour-primaries code, as CIE xy
/// (Rec. ITU-T H.273 Table 2). Mirror of the table in `icc-gen`'s
/// `cicp_bundle_gen` — the gray-bundle roundtrip test pins the two copies
/// together. The name keys the gray profile's description so primaries
/// sharing a white dedup to identical bytes.
#[cfg(test)]
fn h273_white_xy(primaries: u8) -> Option<(&'static str, f64, f64)> {
    Some(match primaries {
        // D65: BT.709, BT.470BG, SMPTE 170M, SMPTE 240M, BT.2020,
        // P3-D65 (SMPTE EG 432-1), EBU Tech 3213-E.
        1 | 5 | 6 | 7 | 9 | 12 | 22 => ("D65", 0.3127, 0.3290),
        // Illuminant C: BT.470M, generic film.
        4 | 8 => ("C", 0.310, 0.316),
        // Illuminant E: SMPTE ST 428-1 (CIE XYZ).
        10 => ("E", 1.0 / 3.0, 1.0 / 3.0),
        // DCI white: SMPTE RP 431-2 (P3-DCI theater white).
        11 => ("DCI", 0.314, 0.351),
        _ => return None,
    })
}

/// Convert CICP to a moxcms ColorProfile.
#[allow(dead_code)]
fn cicp_to_moxcms_profile(cicp: &Cicp) -> ColorProfile {
    ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries: moxcms::CicpColorPrimaries::try_from(cicp.color_primaries)
            .unwrap_or(moxcms::CicpColorPrimaries::Bt709),
        transfer_characteristics: moxcms::TransferCharacteristics::try_from(
            cicp.transfer_characteristics,
        )
        .unwrap_or(moxcms::TransferCharacteristics::Srgb),
        matrix_coefficients: moxcms::MatrixCoefficients::try_from(cicp.matrix_coefficients)
            .unwrap_or(moxcms::MatrixCoefficients::Identity),
        full_range: cicp.full_range,
    })
}

/// Convert primaries + transfer to a moxcms ColorProfile via CICP mapping.
#[allow(dead_code)]
fn primaries_transfer_to_moxcms_profile(
    primaries: crate::ColorPrimaries,
    transfer: crate::TransferFunction,
) -> Result<Option<ColorProfile>, MoxCmsError> {
    let cp = match primaries.to_cicp() {
        Some(c) => c,
        None => return Ok(None),
    };
    let tc = match transfer.to_cicp() {
        Some(c) => c,
        None => return Ok(None),
    };
    Ok(Some(cicp_to_moxcms_profile(&Cicp::new(cp, tc, 0, true))))
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
