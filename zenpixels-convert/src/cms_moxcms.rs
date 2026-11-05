//! CMS backend using [moxcms](https://crates.io/crates/moxcms).
//!
//! Provides a concrete [`PluggableCms`](crate::cms::PluggableCms) implementation backed by the moxcms
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
//! use zenpixels_convert::output::{EncodeReady, OutputProfile};
//!
//! let ready = EncodeReady::finalize(
//!     &buffer, &origin,
//!     OutputProfile::Icc(dst_icc.into()),
//!     PixelFormat::Rgb8,
//!     Some(&MoxCms),
//! )?;
//! ```

use alloc::boxed::Box;
use alloc::format;
use alloc::sync::Arc;

use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, TransformExecutor,
    TransformOptions,
};

/// Our moxcms options, the one place they're set.
///
/// - **Relative colorimetric**, with the ICC profile's own curves. Container color
///   signals are resolved earlier through [`ColorAuthority`](crate::ColorAuthority),
///   so a transform never overrides a profile with a `cicp` tag inside it. Matrix
///   profiles have no rendering tables, and all four moxcms intents give identical
///   results on them. See `docs/gamut-mapping-survey.md`.
/// - **Tetrahedral interpolation** for 3D CLUTs, more accurate than trilinear where
///   saturated colors cross cube diagonals.
/// - **High barycentric weight scale**: on a CMYK table it stays closer to lcms2, at a worst
///   difference of 2.18 of a code against 2.99 for `Low`. Every depth runs the F32 path, where it
///   works; the backend's own 16-bit path turned table profiles into a constant with it.
///
/// Crate-internal: moxcms types stay out of the public API so the moxcms version
/// range can change without a zenpixels-convert release.
pub(crate) fn transform_opts() -> TransformOptions {
    TransformOptions {
        rendering_intent: moxcms::RenderingIntent::RelativeColorimetric,
        allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::High,
        interpolation_method: InterpolationMethod::Tetrahedral,
        ..Default::default()
    }
}

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
        PixelFormat::Cmyk8 | PixelFormat::CmykF32 | PixelFormat::CmykF32Inverted => {
            Some(Layout::Rgba)
        }
        _ => None,
    }
}

/// Options for the backend's F32 transform, the path every depth runs through.
fn options_for_profiles(source: &ColorProfile, target: &ColorProfile) -> TransformOptions {
    let mut options = transform_opts();
    // ICC falls back to A2B0 when A2B1 is absent, as lcms2 does. JPEG XL's PQ and HLG profiles have
    // only A2B0, and the backend refuses them under relative colorimetric intent.
    if source.lut_a_to_b_colorimetric.is_none()
        && source.lut_a_to_b_perceptual.is_some()
        && source.red_trc.is_none()
    {
        options.rendering_intent = moxcms::RenderingIntent::Perceptual;
    }
    // Native F32 ICC-to-P3 comparisons exposed quantization near black and
    // clipping of layer samples outside 0..1. Keep float arithmetic and use
    // analytic curves when available. This does not qualify arbitrary LUTs.
    options.prefer_fixed_point = false;
    // The backend's extended RGB evaluator shares one curve among all
    // channels. Require equality of ALL three, not merely a matching pair. A
    // table-only profile has no curves to confuse.
    let shared_curve = |p: &ColorProfile| {
        p.color_space != moxcms::DataColorSpace::Rgb
            || (p.red_trc == p.green_trc && p.red_trc == p.blue_trc)
    };
    options.allow_extended_range_rgb_xyz = shared_curve(source) && shared_curve(target);
    options
}

/// An explicit ICC request selects its actual curves. The extended evaluator checks CICP
/// independently of `allow_use_cicp_transfer`, so only this private parsed copy loses the hint.
fn prefer_icc_curves(profile: &mut ColorProfile) {
    profile.cicp = None;
}

fn validate_channel_curves(profile: &ColorProfile) -> Result<(), MoxCmsError> {
    if profile.color_space == moxcms::DataColorSpace::Rgb
        && let (Some(red), Some(green), Some(blue)) =
            (&profile.red_trc, &profile.green_trc, &profile.blue_trc)
        && (red == green || green == blue)
        && red != blue
    {
        // Both bounded and extended backend paths can share a curve when
        // only one adjacent pair agrees. Refuse before creating an executor;
        // selecting a different curve would change the requested ICC math.
        return Err(MoxCmsError(
            "distinct RGB channel curves with only two adjacent curves equal are not qualified"
                .into(),
        ));
    }
    Ok(())
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

/// Bytes per pixel on each side. The executors convert whole slices, and a row's slices may be longer
/// than its width.
#[derive(Clone, Copy)]
struct RowExtent {
    source_bytes: usize,
    target_bytes: usize,
}

impl RowExtent {
    const fn of(source: PixelFormat, target: PixelFormat) -> Self {
        Self {
            source_bytes: source.bytes_per_pixel(),
            target_bytes: target.bytes_per_pixel(),
        }
    }
}

/// 8-bit matrix-and-curve conversion through our own kernels, using the profiles' real curve and colorants.
///
/// The backend's integer gamma tables truncate where the target's curve is sRGB's or BT.709's, in
/// `make_gamma_table_impl`: on Display P3 to sRGB, 40% of its 8-bit samples missed nearest rounding, for
/// a mean of -0.40 of a code. This covers RGB and RGBA where every source channel shares a curve and the
/// target's curve is sRGB's.
struct U8MatrixShaper {
    matrix: [[f32; 3]; 3],
    linearize: alloc::boxed::Box<[f32; 256]>,
    alpha: bool,
}

impl U8MatrixShaper {
    fn of(source: &ColorProfile, target: &ColorProfile, layout: (Layout, Layout)) -> Option<Self> {
        let alpha = match layout {
            (Layout::Rgb, Layout::Rgb) => false,
            (Layout::Rgba, Layout::Rgba) => true,
            _ => return None,
        };
        let matrix_shaper = |p: &ColorProfile| {
            p.color_space == moxcms::DataColorSpace::Rgb
                && p.pcs == moxcms::DataColorSpace::Xyz
                && p.lut_a_to_b_colorimetric.is_none()
                && p.lut_a_to_b_perceptual.is_none()
                && p.lut_b_to_a_colorimetric.is_none()
                && p.lut_b_to_a_perceptual.is_none()
                && p.red_trc.is_some()
                && p.red_trc == p.green_trc
                && p.red_trc == p.blue_trc
        };
        if !matrix_shaper(source) || !matrix_shaper(target) {
            return None;
        }
        // Compared by value across the range, since a profile may store this curve in any form.
        let target_curve = target.build_8bit_lin_table(&target.red_trc).ok()?;
        let srgb = crate::fast_gamut::srgb_lin_lut_u8();
        if target_curve
            .iter()
            .zip(srgb.iter())
            .any(|(a, b)| (a - b).abs() > 5e-4)
        {
            return None;
        }
        let linearize = source.build_8bit_lin_table(&source.red_trc).ok()?;
        let product = target
            .colorant_matrix()
            .inverse()
            .mat_mul(source.colorant_matrix());
        let matrix = product.v.map(|row| row.map(|v| v as f32));
        Some(Self {
            matrix,
            linearize,
            alpha,
        })
    }
}

impl TransformExecutor<u8> for U8MatrixShaper {
    fn transform(&self, source: &[u8], target: &mut [u8]) -> Result<(), moxcms::CmsError> {
        let channels = if self.alpha { 4 } else { 3 };
        if !source.len().is_multiple_of(channels) {
            return Err(moxcms::CmsError::LaneMultipleOfChannels);
        }
        if source.len() != target.len() {
            return Err(moxcms::CmsError::LaneSizeMismatch);
        }
        let encode = crate::fast_gamut::srgb_encode_u8;
        if self.alpha {
            crate::fast_gamut::convert_u8_rgba_simd_lut(
                &self.matrix,
                source,
                target,
                &self.linearize,
                encode,
            );
        } else {
            crate::fast_gamut::convert_u8_rgb_simd_matlut(
                &self.matrix,
                source,
                target,
                &self.linearize,
                encode,
            );
        }
        Ok(())
    }
}

/// F32 output through a pure-gamma curve. The backend's evaluator takes the power of a linear sample's
/// magnitude and loses its sign, which turned out-of-gamut samples positive: on Display P3 to Adobe RGB,
/// blue of 8-bit (247, 255, 0) came out 66 where the exact result clips to 0. The backend stops at linear
/// light here, and this applies the curve, keeping the sign.
struct PureGammaOutput {
    transform: Arc<dyn TransformExecutor<f32> + Send + Sync>,
    exponent: f32,
    target_channels: usize,
}

impl TransformExecutor<f32> for PureGammaOutput {
    fn transform(&self, source: &[f32], target: &mut [f32]) -> Result<(), moxcms::CmsError> {
        self.transform.transform(source, target)?;
        for pixel in target.chunks_exact_mut(self.target_channels) {
            for value in &mut pixel[..3] {
                *value = libm::powf(value.abs(), self.exponent).copysign(*value);
            }
        }
        Ok(())
    }
}

/// The exponent that encodes linear light into a pure-gamma RGB matrix profile, if the conversion is
/// matrix to matrix on the analytic path.
fn pure_gamma_exponent(
    source: &ColorProfile,
    target: &ColorProfile,
    target_layout: Layout,
    options: &TransformOptions,
) -> Option<f32> {
    let no_tables = |p: &ColorProfile| {
        p.color_space == moxcms::DataColorSpace::Rgb
            && p.lut_a_to_b_colorimetric.is_none()
            && p.lut_a_to_b_perceptual.is_none()
            && p.lut_b_to_a_colorimetric.is_none()
            && p.lut_b_to_a_perceptual.is_none()
    };
    if !options.allow_extended_range_rgb_xyz
        || !no_tables(source)
        || !no_tables(target)
        || !matches!(target_layout, Layout::Rgb | Layout::Rgba)
    {
        return None;
    }
    match target.red_trc.as_ref()? {
        moxcms::ToneReprCurve::Lut(curve) => match curve.as_slice() {
            [gamma] if *gamma != 0 => Some(256.0 / f32::from(*gamma)),
            _ => None,
        },
        moxcms::ToneReprCurve::Parametric(_) => None,
    }
}

/// The backend's F32 transform, apart from a pure-gamma target's curve, see [`PureGammaOutput`].
fn f32_transform(
    source: &ColorProfile,
    target: &ColorProfile,
    layout: (Layout, Layout),
    options: TransformOptions,
) -> Result<Arc<dyn TransformExecutor<f32> + Send + Sync>, moxcms::CmsError> {
    let Some(exponent) = pure_gamma_exponent(source, target, layout.1, &options) else {
        return source.create_transform_f32(layout.0, target, layout.1, options);
    };
    let mut linear = target.clone();
    linear.cicp = None;
    for trc in [
        &mut linear.red_trc,
        &mut linear.green_trc,
        &mut linear.blue_trc,
    ] {
        // An empty curve is the ICC identity.
        *trc = Some(moxcms::ToneReprCurve::Lut(alloc::vec::Vec::new()));
    }
    let transform = source.create_transform_f32(layout.0, &linear, layout.1, options)?;
    Ok(Arc::new(PureGammaOutput {
        transform,
        exponent,
        target_channels: layout.1.channels(),
    }))
}

/// An integer sample type on the backend's F32 path. `to_unit` rounds back to the same sample through
/// `from_unit`, so a carried alpha is exact.
trait UnitSample: Copy + Default {
    fn to_unit(self) -> f32;
    fn from_unit(unit: f32) -> Self;
}

impl UnitSample for u8 {
    fn to_unit(self) -> f32 {
        f32::from(self) / 255.0
    }
    fn from_unit(unit: f32) -> Self {
        (f64::from(unit.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8
    }
}

impl UnitSample for u16 {
    fn to_unit(self) -> f32 {
        f32::from(self) / 65535.0
    }
    fn from_unit(unit: f32) -> Self {
        (f64::from(unit.clamp(0.0, 1.0)) * 65535.0 + 0.5) as u16
    }
}

/// Integer samples through the backend's F32 path, with nearest rounding on the way out, for 16-bit
/// and for 8-bit tables, gray and anything else without a matrix to take. The backend's own integer
/// paths truncate their gamma tables, see [`U8MatrixShaper`].
struct IntegerThroughF32 {
    transform: Arc<dyn TransformExecutor<f32> + Send + Sync>,
    source_channels: usize,
    target_channels: usize,
}

impl<S: UnitSample> TransformExecutor<S> for IntegerThroughF32 {
    fn transform(&self, source: &[S], target: &mut [S]) -> Result<(), moxcms::CmsError> {
        const PIXELS: usize = 256;
        const MOST_CHANNELS: usize = 4;
        let (sc, tc) = (self.source_channels, self.target_channels);
        if !source.len().is_multiple_of(sc) || !target.len().is_multiple_of(tc) {
            return Err(moxcms::CmsError::LaneMultipleOfChannels);
        }
        if source.len() / sc != target.len() / tc {
            return Err(moxcms::CmsError::LaneSizeMismatch);
        }
        // Stack scratch, so a row doesn't allocate.
        let mut float_source = [0f32; PIXELS * MOST_CHANNELS];
        let mut float_target = [0f32; PIXELS * MOST_CHANNELS];
        for (src, dst) in source
            .chunks(PIXELS * sc)
            .zip(target.chunks_mut(PIXELS * tc))
        {
            let in_float = &mut float_source[..src.len()];
            for (float, sample) in in_float.iter_mut().zip(src) {
                *float = sample.to_unit();
            }
            let out_float = &mut float_target[..dst.len()];
            self.transform.transform(in_float, out_float)?;
            for (sample, float) in dst.iter_mut().zip(out_float.iter()) {
                *sample = S::from_unit(*float);
            }
        }
        Ok(())
    }
}

/// The 8-bit transform for a pair of profiles. `options` are for the F32 path.
fn u8_transform(
    source: &ColorProfile,
    target: &ColorProfile,
    layout: (Layout, Layout),
    options: TransformOptions,
) -> Result<MoxTransformInner, moxcms::CmsError> {
    if let Some(shaper) = U8MatrixShaper::of(source, target, layout) {
        return Ok(MoxTransformInner::U8(Arc::new(shaper)));
    }
    let transform = f32_transform(source, target, layout, options)?;
    Ok(MoxTransformInner::U8(Arc::new(IntegerThroughF32 {
        transform,
        source_channels: layout.0.channels(),
        target_channels: layout.1.channels(),
    })))
}

/// The 16-bit transform for a pair of profiles. `options` are for the F32 path.
fn u16_transform(
    source: &ColorProfile,
    target: &ColorProfile,
    layout: (Layout, Layout),
    options: TransformOptions,
) -> Result<MoxTransformInner, moxcms::CmsError> {
    let transform = f32_transform(source, target, layout, options)?;
    Ok(MoxTransformInner::U16(Arc::new(IntegerThroughF32 {
        transform,
        source_channels: layout.0.channels(),
        target_channels: layout.1.channels(),
    })))
}

const fn is_rgb_format(format: PixelFormat) -> bool {
    !matches!(
        format,
        PixelFormat::Gray8
            | PixelFormat::Gray16
            | PixelFormat::GrayF32
            | PixelFormat::GrayA8
            | PixelFormat::GrayA16
            | PixelFormat::GrayAF32
    )
}

const fn is_cmyk_f32(format: PixelFormat) -> bool {
    matches!(format, PixelFormat::CmykF32 | PixelFormat::CmykF32Inverted)
}

/// Whether a source names sRGB, as opposed to an ICC profile that merely resembles it.
fn names_srgb(source: &crate::ColorProfileSource<'_>) -> bool {
    match source {
        crate::ColorProfileSource::Cicp(cicp) => {
            cicp.color_primaries == 1 && cicp.transfer_characteristics == 13
        }
        crate::ColorProfileSource::Named(zenpixels::NamedProfile::Srgb) => true,
        crate::ColorProfileSource::PrimariesTransferPair {
            primaries,
            transfer,
        } => {
            *primaries == zenpixels::ColorPrimaries::Bt709
                && *transfer == zenpixels::TransferFunction::Srgb
        }
        crate::ColorProfileSource::Icc(icc) => {
            zenpixels::icc::IccIdentification::is_common_srgb(icc)
        }
        _ => false,
    }
}

/// F32 CMYK through a profile, on the path libjxl takes through lcms2 for sRGB output: relative
/// colorimetric with black-point compensation, and its own sign-preserving sRGB encode. The backend
/// evaluates its sRGB curve imprecisely after a CMYK lookup and has no black-point compensation.
/// Other destinations use the perceptual intent, without extended range, the closer of the two.
struct CmykFloatTransform {
    transform: Arc<dyn TransformExecutor<f32> + Send + Sync>,
    invert_input: bool,
    invert_output: bool,
    /// The backend produced linear light, which is encoded here after compensation.
    encode_srgb: bool,
    black_point_y: f32,
    target_channels: usize,
    clip: bool,
}

impl TransformExecutor<f32> for CmykFloatTransform {
    fn transform(&self, source: &[f32], target: &mut [f32]) -> Result<(), moxcms::CmsError> {
        const PIXELS: usize = 256;
        let tc = self.target_channels;
        if !source.len().is_multiple_of(4) || !target.len().is_multiple_of(tc) {
            return Err(moxcms::CmsError::LaneMultipleOfChannels);
        }
        // Stack scratch for the polarity change, so a row doesn't allocate.
        let mut ink = [0.0f32; PIXELS * 4];
        for (src, dst) in source
            .chunks(PIXELS * 4)
            .zip(target.chunks_mut(PIXELS * tc))
        {
            if self.invert_input {
                for (i, s) in ink.iter_mut().zip(src) {
                    *i = 1.0 - *s;
                }
                self.transform.transform(&ink[..src.len()], dst)?;
            } else {
                self.transform.transform(src, dst)?;
            }
            if self.invert_output {
                for v in dst.iter_mut() {
                    *v = 1.0 - *v;
                }
            }
            if self.encode_srgb {
                let scale = 1.0 / (1.0 - self.black_point_y);
                for v in dst.iter_mut() {
                    let linear = (*v - self.black_point_y) * scale;
                    let a = linear.abs();
                    let encoded = if a <= 0.003_130_8 {
                        a * 12.92
                    } else {
                        1.055 * libm::powf(a, 1.0 / 2.4) - 0.055
                    };
                    *v = encoded.copysign(linear);
                }
            }
            if self.clip {
                for v in dst.iter_mut() {
                    *v = v.clamp(0.0, 1.0);
                }
            }
        }
        Ok(())
    }
}

/// Luminance (D50 Y) of lcms2's black point for a CMYK profile under relative colorimetric intent,
/// measured through `linear_dst`, an RGB profile with identity curves. `None` if the profile has no
/// usable perceptual B2A table. L* is capped at 50.
fn cmyk_black_point_y(
    cmyk: &ColorProfile,
    linear_dst: &ColorProfile,
    relative: TransformOptions,
) -> Option<f32> {
    let perceptual = TransformOptions {
        rendering_intent: moxcms::RenderingIntent::Perceptual,
        ..relative
    };
    // Lab black is RGB black in any matrix/shaper profile, so the perceptual table can be entered
    // from `linear_dst`.
    let to_cmyk = linear_dst
        .create_transform_f32(Layout::Rgb, cmyk, Layout::Rgba, perceptual)
        .ok()?;
    let to_rgb = cmyk
        .create_transform_f32(Layout::Rgba, linear_dst, Layout::Rgb, relative)
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

fn cmyk_float_transform(
    source: &ColorProfile,
    target: &ColorProfile,
    src_format: PixelFormat,
    dst_format: PixelFormat,
    target_is_srgb: bool,
    clip: bool,
) -> Result<MoxTransformInner, MoxCmsError> {
    let Some(dst_layout) = pixel_format_to_layout(dst_format) else {
        return Err(MoxCmsError(
            "F32 CMYK has no conversion to this layout".into(),
        ));
    };
    let mut target = target.clone();
    let encode_srgb = target_is_srgb && dst_layout == Layout::Rgb;
    if encode_srgb {
        target.cicp = None;
        // An empty curve is the ICC identity.
        for trc in [
            &mut target.red_trc,
            &mut target.green_trc,
            &mut target.blue_trc,
        ] {
            *trc = Some(moxcms::ToneReprCurve::Lut(alloc::vec::Vec::new()));
        }
    }
    let options = TransformOptions {
        allow_extended_range_rgb_xyz: encode_srgb,
        // ICC falls back to A2B0 when A2B1 is absent, as lcms2 does.
        rendering_intent: if encode_srgb && source.lut_a_to_b_colorimetric.is_some() {
            moxcms::RenderingIntent::RelativeColorimetric
        } else {
            moxcms::RenderingIntent::Perceptual
        },
        barycentric_weight_scale: moxcms::BarycentricWeightScale::High,
        ..TransformOptions::default()
    };
    let black_point_y = if encode_srgb {
        cmyk_black_point_y(source, &target, options).unwrap_or(0.0)
    } else {
        0.0
    };
    let transform = source
        .create_transform_f32(Layout::Rgba, &target, dst_layout, options)
        .map_err(|e| MoxCmsError(format!("moxcms create_transform_f32 failed: {e}")))?;
    Ok(MoxTransformInner::F32(Arc::new(CmykFloatTransform {
        transform,
        invert_input: src_format == PixelFormat::CmykF32Inverted,
        invert_output: dst_format == PixelFormat::CmykF32Inverted,
        encode_srgb,
        black_point_y,
        target_channels: dst_layout.channels(),
        clip,
    })))
}

/// Preserve the requested bounded input/output behavior around the analytic
/// F32 executor. Integer lookup tables otherwise quantize even float samples.
/// Alpha is storage, not a color coordinate, and must never be clamped.
struct BoundedFloatTransform {
    transform: Arc<dyn TransformExecutor<f32> + Send + Sync>,
    source_channels: usize,
    target_channels: usize,
}

impl TransformExecutor<f32> for BoundedFloatTransform {
    fn transform(&self, source: &[f32], target: &mut [f32]) -> Result<(), moxcms::CmsError> {
        let sc = self.source_channels;
        let tc = self.target_channels;
        if !source.len().is_multiple_of(sc) || !target.len().is_multiple_of(tc) {
            return Err(moxcms::CmsError::LaneMultipleOfChannels);
        }
        if source.len() / sc != target.len() / tc {
            return Err(moxcms::CmsError::LaneSizeMismatch);
        }
        let source_colors = if sc <= 2 { 1 } else { 3 };
        let target_colors = if tc <= 2 { 1 } else { 3 };
        if source.chunks_exact(sc).all(|pixel| {
            pixel[..source_colors]
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        }) {
            self.transform.transform(source, target)?;
        } else {
            // Bounded stack staging, independent of row width. The ordinary
            // in-range path above uses the caller's source directly.
            let mut scratch = [0.0f32; 128 * 4];
            for (input, output) in source.chunks(128 * sc).zip(target.chunks_mut(128 * tc)) {
                let stage = &mut scratch[..input.len()];
                stage.copy_from_slice(input);
                for pixel in stage.chunks_exact_mut(sc) {
                    for value in &mut pixel[..source_colors] {
                        *value = value.clamp(0.0, 1.0);
                    }
                }
                self.transform.transform(stage, output)?;
            }
        }
        for pixel in target.chunks_exact_mut(tc) {
            for value in &mut pixel[..target_colors] {
                *value = value.clamp(0.0, 1.0);
            }
        }
        Ok(())
    }
}

fn float_transform(
    transform: Arc<dyn TransformExecutor<f32> + Send + Sync>,
    source: Layout,
    target: Layout,
    clip: bool,
) -> MoxTransformInner {
    if clip {
        MoxTransformInner::F32(Arc::new(BoundedFloatTransform {
            transform,
            source_channels: source.channels(),
            target_channels: target.channels(),
        }))
    } else {
        MoxTransformInner::F32(transform)
    }
}

fn try_mox_row(
    inner: &MoxTransformInner,
    extent: RowExtent,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
) -> Result<(), whereat::At<crate::cms::CmsPluginError>> {
    let cast_error = |_| {
        whereat::at!(crate::cms::CmsPluginError::msg(
            "CMS sample alignment or extent mismatch"
        ))
    };
    let pixels = width as usize;
    let extent_error = || {
        whereat::at!(crate::cms::CmsPluginError::msg(
            "CMS row is shorter than its width"
        ))
    };
    let src = src
        .get(..pixels.saturating_mul(extent.source_bytes))
        .ok_or_else(extent_error)?;
    let dst = dst
        .get_mut(..pixels.saturating_mul(extent.target_bytes))
        .ok_or_else(extent_error)?;
    let result = match inner {
        MoxTransformInner::U8(t) => t.transform(src, dst),
        MoxTransformInner::U16(t) => t.transform(
            bytemuck::try_cast_slice(src).map_err(cast_error)?,
            bytemuck::try_cast_slice_mut(dst).map_err(cast_error)?,
        ),
        MoxTransformInner::F32(t) => t.transform(
            bytemuck::try_cast_slice(src).map_err(cast_error)?,
            bytemuck::try_cast_slice_mut(dst).map_err(cast_error)?,
        ),
    };
    result.map_err(|e| {
        whereat::at!(crate::cms::CmsPluginError::msg(format!(
            "moxcms row failed: {e}"
        )))
    })
}

// ---------------------------------------------------------------------------
// PluggableCms — the dispatch chain RowConverter actually consults.
// ---------------------------------------------------------------------------

/// `RowTransformMut` wrapper for moxcms transform executors at the three
/// supported bit depths. It exposes the
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
    extent: RowExtent,
}

impl crate::cms::RowTransformMut for MoxRowTransformMut {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) {
        try_mox_row(&self.inner, self.extent, src, dst, width).expect("moxcms row failed");
    }
    fn try_transform_row(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
    ) -> Result<(), whereat::At<crate::cms::CmsPluginError>> {
        try_mox_row(&self.inner, self.extent, src, dst, width)
    }
    fn prepare(&mut self, _max_width: u32) -> Result<(), whereat::At<crate::cms::CmsPluginError>> {
        Ok(())
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
        options: &crate::policy::ConvertOptions,
    ) -> Option<Result<Box<dyn crate::cms::RowTransformMut>, whereat::At<crate::cms::CmsPluginError>>>
    {
        use crate::cms::CmsPluginError;
        // Depths the plan can convert aren't this backend's problem, so it declines them. Sample types
        // nothing here can convert are diagnosed.
        let plannable = |t: ChannelType| {
            matches!(
                t,
                ChannelType::U8 | ChannelType::U16 | ChannelType::F16 | ChannelType::F32
            )
        };
        let (src_type, dst_type) = (src_format.channel_type(), dst_format.channel_type());
        if plannable(src_type)
            && plannable(dst_type)
            && (src_type != dst_type || src_type == ChannelType::F16)
        {
            return None;
        }
        if src_type != dst_type || !plannable(src_type) {
            return Some(Err(whereat::at!(CmsPluginError::msg(
                "moxcms requires matching U8, U16 or F32 sample types; convert depth separately"
            ))));
        }
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

        if is_cmyk_f32(src_format) {
            return Some(
                cmyk_float_transform(
                    &src_profile,
                    &dst_profile,
                    src_format,
                    dst_format,
                    names_srgb(&dst),
                    options.clip_out_of_gamut(),
                )
                .map(|inner| {
                    Box::new(MoxRowTransformMut {
                        inner,
                        extent: RowExtent::of(src_format, dst_format),
                    }) as Box<dyn crate::cms::RowTransformMut>
                })
                .map_err(|error| whereat::at!(CmsPluginError::new(error))),
            );
        }
        if is_cmyk_f32(dst_format) {
            return Some(Err(whereat::at!(CmsPluginError::msg(
                "conversion into F32 CMYK is not qualified"
            ))));
        }

        if let Err(error) = validate_channel_curves(&src_profile)
            .and_then(|()| validate_channel_curves(&dst_profile))
        {
            return Some(Err(whereat::at!(CmsPluginError::new(error))));
        }
        let opts = options_for_profiles(&src_profile, &dst_profile);

        // Dispatch on the source `ChannelType`. F16 is widened to f32
        // before the CMS step in this crate, so we route F16 to the f32
        // transform path (the same convention `build_transform_inner`
        // above uses). Note: when src and dst differ on channel type, we
        // pick the source side; the layout/bit-depth conversion happens
        // outside moxcms in a separate plan step.
        let depth = src_format.channel_type();
        let inner_result = match depth {
            ChannelType::U8 => {
                u8_transform(&src_profile, &dst_profile, (src_layout, dst_layout), opts)
            }
            ChannelType::U16 => {
                u16_transform(&src_profile, &dst_profile, (src_layout, dst_layout), opts)
            }
            ChannelType::F16 | ChannelType::F32 | _ => {
                f32_transform(&src_profile, &dst_profile, (src_layout, dst_layout), opts).map(
                    |transform| {
                        float_transform(
                            transform,
                            src_layout,
                            dst_layout,
                            options.clip_out_of_gamut(),
                        )
                    },
                )
            }
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

        Some(Ok(Box::new(MoxRowTransformMut {
            inner,
            extent: RowExtent::of(src_format, dst_format),
        })))
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
    let is_cmyk = format.color_model() == zenpixels::ColorModel::Cmyk;
    match src {
        // ICC bytes are authoritative for every color model — parse and
        // hand them straight to moxcms. The profile's `data_color_space`
        // will validate against the layout downstream
        // (`check_layout`).
        crate::ColorProfileSource::Icc(icc) => ColorProfile::new_from_slice(icc)
            .map(|mut profile| {
                prefer_icc_curves(&mut profile);
                Some(profile)
            })
            .map_err(|e| MoxCmsError(format!("failed to parse ICC: {e}"))),
        // CICP describes RGB colorimetry; CMYK descriptors with CICP
        // (which is unusual — CICP rarely tags CMYK) decline so the
        // caller can attach a real CMYK ICC instead.
        // Generated profiles are RGB, and unspecified code points describe no color.
        _ if !is_rgb_format(format) && !matches!(src, crate::ColorProfileSource::Icc(_)) => {
            Ok(None)
        }
        crate::ColorProfileSource::Cicp(cicp)
            if !is_cmyk
                && (cicp.color_primaries_enum() == crate::ColorPrimaries::Unknown
                    || cicp.transfer_function_enum() == crate::TransferFunction::Unknown) =>
        {
            Ok(None)
        }
        crate::ColorProfileSource::Cicp(cicp) if !is_cmyk => {
            Ok(Some(with_pcs_white(cicp_to_moxcms_profile(cicp))))
        }
        crate::ColorProfileSource::Named(named) if !is_cmyk => {
            let (p, t) = named.to_primaries_transfer();
            primaries_transfer_to_moxcms_profile(p, t).map(|p| p.map(with_pcs_white))
        }
        crate::ColorProfileSource::PrimariesTransferPair {
            primaries,
            transfer,
        } if !is_cmyk => primaries_transfer_to_moxcms_profile(*primaries, *transfer)
            .map(|p| p.map(with_pcs_white)),
        // CMYK without ICC bytes: decline. The PluggableCms chain falls
        // through to the built-in plan path, which answers
        // `NeedsCms` so the caller knows to attach an ICC.
        _ => Ok(None),
    }
}

/// Scales a generated profile's colorants so that they sum to the ICC PCS white.
///
/// moxcms builds them against CIE D50, with Z = 0.82521, while ICC profiles sum to the PCS white, with
/// Z = 0.8249. Between one of each, white doesn't map to white: blue comes out 0.04% low, and 8-bit
/// white from any ICC profile into generated sRGB became (255, 255, 254).
fn with_pcs_white(mut profile: ColorProfile) -> ColorProfile {
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

/// moxcms's D50 white point (`white_point_from_temperature(5003)`), reconstructed so
/// the gray oracle below builds against the whole `>=0.8.1, <0.10` moxcms range.
///
/// moxcms exposed this value as the const `WHITE_POINT_D50` (moxcms ≤ 0.8) and then
/// renamed it to the fn `white_point_d50()` (moxcms ≥ 0.9) — identical value,
/// mutually-exclusive spellings. Reproducing it through the stable [`moxcms::XyY`]
/// API (public f64 fields + f64→f32 `to_xyz`, unchanged across the range) avoids
/// pinning this crate to either moxcms minor. Kept bit-identical to upstream by
/// `moxcms_d50_tests::moxcms_d50_matches_upstream`.
#[cfg(test)]
fn moxcms_d50_xyz() -> moxcms::Xyz {
    // moxcms's McCamy CCT→xy for T = 5003 K (its `4000 < T ≤ 7000` branch), in f64:
    const T: f64 = 5003.0;
    const X: f64 =
        -4.6070 * (1e9 / (T * T * T)) + 2.9678 * (1e6 / (T * T)) + 0.09911 * (1e3 / T) + 0.244063;
    const Y: f64 = -3.000 * X * X + 2.870 * X - 0.275;
    moxcms::XyY::new(X, Y, 1.0).to_xyz()
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
    gray.chromatic_adaptation = Some(moxcms::adaption_matrix_d(white.to_xyz(), moxcms_d50_xyz()));
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

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error from the moxcms CMS backend.
#[derive(Debug, Clone)]
pub struct MoxCmsError(String);

impl core::error::Error for MoxCmsError {}

impl core::fmt::Display for MoxCmsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod moxcms_d50_tests {
    use super::moxcms_d50_xyz;

    /// The reconstructed D50 white must stay bit-identical to moxcms's own D50
    /// (`white_point_from_temperature(5003).to_xyz()`, ≈ (0.96391, 1.0, 0.82475)).
    /// This golden was verified equal to `moxcms::WHITE_POINT_D50.to_xyz()`
    /// (moxcms 0.8.x) and `moxcms::white_point_d50().to_xyz()` (moxcms 0.9.x) when
    /// written; the exact f32 bits guard against an upstream `XyY::to_xyz` change
    /// within the `>=0.8.1, <0.10` range.
    #[test]
    fn moxcms_d50_matches_upstream() {
        let d50 = moxcms_d50_xyz();
        assert_eq!(d50.x.to_bits(), 0x3f76_c28e); // 0.9639062
        assert_eq!(d50.y, 1.0_f32);
        assert_eq!(d50.z.to_bits(), 0x3f53_2290); // 0.8247461
    }
}
