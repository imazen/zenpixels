//! Type-safe row-level ([`RowPlan`]) and image-level ([`ImagePlan`]) conversion plans.
//!
//! - [`RowPlan::build`] accepts **`hdr_peak: Option<Nits>`** — a compile-time
//!   known peak luminance. Passing `PeakSource::Measured` to a single-row
//!   converter is impossible at compile time. Executing a `RowPlan` allocates
//!   **zero heap scratch** (uses a 1 KiB stack 64-pixel SoA tile).
//! - [`ImagePlan::build`] accepts **`peak: PeakSource`** (`Measured`, `Declared`,
//!   or `Fixed(Nits)`), streams HDR peak measurement in O(1) row memory, runs
//!   uniformly over any [`PixelSource`] → [`PixelSink`] (`PixelView`, `PixelBuf`,
//!   `BandedBuf`), caches invariant row plans, and propagates proven [`SampleFacts`].

use alloc::sync::Arc;
use alloc::vec::Vec;
use whereat::At;
use zenpixels::{
    AlphaMode, ColorSpace, DiffuseWhite, ExactDepth, FloatFacts, HdrMetadata, HlgMode,
    HlgRendering, ImageSpec, Nits, PeakSource, PixelBuf, PixelSink, PixelSource, PixelView,
    Primaries, SampleFacts, SampleType, SignalRange, Transfer, full_range_grid_nests,
    narrow_range_grid_nests,
};

use crate::accel::{self, Accelerators};
use crate::cms::CmsTransform;
use crate::error::{ConvertError, err_at};
use crate::policy::{
    AlphaPolicy, ConvertPolicy, DepthPolicy, GamutPolicy, LumaDomain, RangePolicy,
};
use crate::strip::{self, ByteFastPath, FastRow, MicroOp};
use crate::tonemap::{PreparedMapper, StripMapper, ToneMapContext};

#[cfg(feature = "hdr-transfers")]
use crate::hlg_render::PreparedHlg;
#[cfg(not(feature = "hdr-transfers"))]
type PreparedHlg = ();
#[cfg(feature = "oklab")]
use crate::tonemap::SoftCompress;

/// Execution strategy compiled inside a [`RowPlan`].
#[derive(Clone)]
enum RowExec {
    /// Zero-float byte/integer/wide-layout fast path.
    Byte(ByteFastPath),
    /// Direct contiguous single-stage row fast path (zero AoS↔SoA bounce).
    FastRow(FastRow),
    /// Composable micro-op pipeline running on 64-pixel stack SoA strips.
    Strip {
        ops: Vec<MicroOp>,
        src_opaque_alpha: bool,
    },
    /// External CMS row transform (`ColorSpace::Icc` / `ColorSpace::Cmyk`).
    Cms {
        transform: Arc<dyn CmsTransform>,
        clip_f32_color: bool,
    },
}

/// Shared preflight result used by both [`RowPlan::build`] and [`Negotiate`](crate::Negotiate).
pub(crate) struct PreflightOutcome {
    pub matte_color: Option<zenpixels::Color>,
    pub range_change: bool,
    pub uses_cms: bool,
    pub is_identity: bool,
    pub drops_alpha: bool,
    pub alpha_proven_opaque: bool,
    pub drops_chroma: bool,
    pub chroma_proven_gray: bool,
    pub gamut_narrows: bool,
    pub is_hdr_to_sdr: bool,
    pub depth_reduces: bool,
}

/// Compiled, reusable, thread-safe single-row pixel converter.
///
/// Built via [`RowPlan::build`] with an explicit optional `hdr_peak: Option<Nits>`.
/// Calling [`convert_row`](Self::convert_row) on native routes allocates **zero
/// heap memory** and uses a 1 KiB stack-allocated 64-pixel SoA tile.
#[derive(Clone)]
pub struct RowPlan {
    src: ImageSpec,
    dst: ImageSpec,
    exec: RowExec,
    depth_policy: DepthPolicy,
    accel: Accelerators,
    output_facts: SampleFacts,
}

impl core::fmt::Debug for RowPlan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RowPlan")
            .field("src", &self.src)
            .field("dst", &self.dst)
            .field("depth_policy", &self.depth_policy)
            .field("accel", &self.accel)
            .finish()
    }
}

impl RowPlan {
    /// Compile a single-row conversion plan from `src` to `dst` under `policy`.
    ///
    /// Accepts either borrowed `&ImageSpec` (zero-clone) or owned `ImageSpec`.
    ///
    /// For HDR → SDR tone mapping with a pipeline-managed mapper, pass
    /// `hdr_peak: Some(nits)`. To measure peak automatically over a whole image,
    /// use [`ImagePlan::build`] with [`PeakSource::Measured`].
    #[inline]
    pub fn build(
        src: impl AsRef<ImageSpec>,
        dst: impl AsRef<ImageSpec>,
        policy: &ConvertPolicy,
        hdr_peak: Option<Nits>,
    ) -> Result<Self, At<ConvertError>> {
        Self::build_inner(src.as_ref(), dst.as_ref(), policy, hdr_peak)
    }

    #[inline(never)]
    fn build_inner(
        src: &ImageSpec,
        dst: &ImageSpec,
        policy: &ConvertPolicy,
        hdr_peak: Option<Nits>,
    ) -> Result<Self, At<ConvertError>> {
        let pre = preflight_and_classify(src, dst, policy)?;
        let accel = accel::resolve(policy.accelerators);

        // 1. Byte-exact identity (supports all 11 sample types).
        if pre.is_identity {
            return Ok(Self {
                output_facts: policy.source_facts,
                src: src.clone(),
                dst: dst.clone(),
                exec: RowExec::Byte(ByteFastPath::Memcpy),
                depth_policy: policy.depth,
                accel,
            });
        }

        let matte_color = pre.matte_color;
        let range_change = pre.range_change;

        // 2. Fast byte-only path (also handles same-profile opaque ICC / CMYK layout, widening,
        //    opaque alpha fill, same-depth integer repacking, and BT.2100 narrow bit shifts
        //    without requiring a CMS backend!).
        //    Pass `policy.source_facts.all_opaque` (proven stored max alpha, NOT merely logical
        //    `AlphaMode::Opaque`) so unverified `Opaque` junk alpha bytes are canonicalized!
        if let Some(bp) = strip::detect_byte_fast_path(
            src,
            dst,
            policy.depth,
            policy.luma,
            policy.luma_domain,
            matte_color.is_some(),
            policy.source_facts.all_opaque,
        ) {
            let output_facts = derive_output_facts_byte(src, dst, policy.source_facts, bp);
            return Ok(Self {
                src: src.clone(),
                dst: dst.clone(),
                exec: RowExec::Byte(bp),
                depth_policy: policy.depth,
                accel,
                output_facts,
            });
        }

        // 3. ICC / CMYK routing (policy checks have ALREADY passed in `preflight_and_classify`!).
        if pre.uses_cms {
            if let (Some(sub_src), Some(sub_dst)) = (
                try_substitute_icc(src, policy),
                try_substitute_icc(dst, policy),
            ) {
                let mut inner = Self::build_inner(&sub_src, &sub_dst, policy, hdr_peak)?;
                inner.src = src.clone();
                inner.dst = dst.clone();
                return Ok(inner);
            }
            return Self::build_cms(
                src.clone(),
                dst.clone(),
                policy,
                pre.matte_color.is_some(),
                accel,
            );
        }

        // 4. Resolve photometry, transfers, HDR white anchors, and HLG BEFORE FastRow or Strip!
        let src_tf = color_transfer(&src.color);
        let dst_tf = color_transfer(&dst.color);
        let src_range = color_range(&src.color);
        let dst_range = color_range(&dst.color);
        let src_bits = effective_color_bits(src);
        let dst_bits = effective_color_bits(dst);

        let src_is_pq = matches!(src_tf, Transfer::Pq);
        let dst_is_pq = matches!(dst_tf, Transfer::Pq);

        let relative_white_scale = if !src_is_pq && !dst_is_pq {
            match (src.color.white(), dst.color.white()) {
                (Some(sw), Some(dw)) if ((sw.nits() / dw.nits()) - 1.0).abs() > 1e-6 => {
                    sw.nits() / dw.nits()
                }
                _ => 1.0,
            }
        } else {
            1.0
        };
        let has_relative_white_rescale = (relative_white_scale - 1.0).abs() > 1e-6;

        let need_linear_space = needs_linear_domain(
            &src.color,
            &dst.color,
            matte_color.is_some(),
            has_relative_white_rescale,
            policy,
        );

        // Reject `Transfer::Unknown` whenever transfer changes or linear-domain math is required.
        if (src_tf == Transfer::Unknown || dst_tf == Transfer::Unknown)
            && (!src_tf.same_curve(dst_tf) || need_linear_space)
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }

        let crosses_hdr_feature = src_tf.is_hdr() || dst_tf.is_hdr();
        if crosses_hdr_feature && !cfg!(feature = "hdr-transfers") {
            return Err(err_at(ConvertError::HdrFeatureRequired));
        }

        let white_nits = if src_is_pq && !dst_is_pq {
            dst.color
                .white()
                .or(policy.sdr_white)
                .ok_or_else(|| err_at(ConvertError::MissingHdrAnchor))?
                .nits()
        } else if !src_is_pq && dst_is_pq {
            src.color
                .white()
                .or(policy.sdr_white)
                .ok_or_else(|| err_at(ConvertError::MissingHdrAnchor))?
                .nits()
        } else {
            src.color
                .white()
                .or_else(|| dst.color.white())
                .or(policy.sdr_white)
                .unwrap_or(DiffuseWhite::BT2408)
                .nits()
        };

        // Separate source and destination white anchors for HLG display OOTF so a
        // relative white rescale (`Linear(100) -> HLG(200)`) never double-scales (B16).
        let src_hlg_white_nits = if src_is_pq || dst_is_pq {
            white_nits
        } else {
            src.color
                .white()
                .or_else(|| dst.color.white())
                .or(policy.sdr_white)
                .unwrap_or(DiffuseWhite::BT2408)
                .nits()
        };
        let dst_hlg_white_nits = if src_is_pq || dst_is_pq {
            white_nits
        } else {
            dst.color
                .white()
                .or_else(|| src.color.white())
                .or(policy.sdr_white)
                .unwrap_or(DiffuseWhite::BT2408)
                .nits()
        };

        let preserve_extended =
            policy.gamut == GamutPolicy::PreserveExtended && dst.layout.sample.is_float();

        // Resolve HLG display rendering BEFORE FastRow detection so identical semantics apply
        // regardless of channel order / layout!
        #[allow(unused_variables)]
        let (src_hlg_tf, src_prepared_hlg) = resolve_hlg(
            &src.color,
            src_tf,
            dst_tf,
            policy.hlg_rendering,
            false,
            src_hlg_white_nits,
            !preserve_extended,
        )?;
        #[allow(unused_variables)]
        let (dst_hlg_tf, dst_prepared_hlg) = resolve_hlg(
            &dst.color,
            dst_tf,
            src_tf,
            policy.hlg_rendering,
            true,
            dst_hlg_white_nits,
            !preserve_extended,
        )?;

        // 5. Direct contiguous single-stage row fast path (zero AoS↔SoA bounce)
        // ONLY after policy, Transfer::Unknown, relative-white, and HLG are resolved!
        let is_linear_hdr = is_scene_linear_hdr_candidate(src, dst)
            && hdr_peak.is_some_and(|p| p.nits() > white_nits * (1.0 + 1e-4));
        let is_hdr_to_sdr = pre.is_hdr_to_sdr || is_linear_hdr;
        if !(is_hdr_to_sdr && policy.tone_map_hdr_to_sdr)
            && !has_relative_white_rescale
            && src_prepared_hlg.is_none()
            && dst_prepared_hlg.is_none()
            && policy.gamut != GamutPolicy::SoftCompress
            && let Some(fr) = strip::detect_fast_row(
                src,
                dst,
                policy.depth,
                policy.gamut,
                policy.allow_approximate,
                matte_color.is_some(),
                preserve_extended,
                white_nits,
            )
        {
            let clamped_color = !preserve_extended
                && !dst_tf.is_hdr()
                && !matches!(
                    fr,
                    FastRow::TransferF32 {
                        transfer: Transfer::Pq,
                        is_eotf: true,
                        ..
                    }
                )
                && matches!(
                    fr,
                    FastRow::SrgbU8ToLinF32 { .. }
                        | FastRow::U8ToF32 { .. }
                        | FastRow::U16ToF32 { .. }
                        | FastRow::TransferF32 { .. }
                );
            let alpha_filled_opaque = matches!(
                fr,
                FastRow::Srgb8MatrixExact {
                    fill_opaque: true,
                    ..
                } | FastRow::Srgb8MatrixApproxQ14 {
                    fill_opaque: true,
                    ..
                }
            );
            let safe_no_subnormals = matches!(
                fr,
                FastRow::SrgbU8ToLinF32 { .. } | FastRow::U8ToF32 { .. } | FastRow::U16ToF32 { .. }
            );
            let output_facts = derive_output_facts_math(
                src,
                dst,
                policy.source_facts,
                clamped_color,
                alpha_filled_opaque,
                false,
                matches!(fr, FastRow::F32ToF16 { .. } | FastRow::F16ToF32 { .. }),
                safe_no_subnormals,
            );
            return Ok(Self {
                src: src.clone(),
                dst: dst.clone(),
                exec: RowExec::FastRow(fr),
                depth_policy: policy.depth,
                accel,
                output_facts,
            });
        }

        // 6. Compile composable micro-op sequence on 64-px stack SoA StripBuf.
        let mut ops: Vec<MicroOp> = Vec::with_capacity(14);
        let narrow_depth_change = src_range == SignalRange::Narrow
            && dst_range == SignalRange::Narrow
            && src_bits != dst_bits;

        // 6a. Source narrow range expansion (before unpremul/EOTF!).
        if src_range == SignalRange::Narrow
            && (range_change || need_linear_space || !src_tf.is_linear() || narrow_depth_change)
        {
            ops.push(MicroOp::ExpandNarrow { bits: src_bits });
        }

        // 6b. Unpremultiply source if premultiplied (when transfer/gamut/matte/alpha changes).
        let src_premul = src.color.alpha() == AlphaMode::Premultiplied;
        let dst_premul = dst.color.alpha() == AlphaMode::Premultiplied;
        if src_premul && (need_linear_space || !dst_premul) {
            ops.push(MicroOp::Unpremul);
        }

        // 6c. Source EOTF (into linear light) or Oklab/Perceptual/XYZ decode into linear RGB.
        let mut cur_primaries = color_primaries(&src.color);
        let dst_primaries = color_primaries(&dst.color);
        let mut cur_is_gray = matches!(src.color, ColorSpace::Gray { .. });
        let mut already_in_dst_xyz = false;

        match &src.color {
            ColorSpace::Xyz { white_point, .. } => {
                if let ColorSpace::Xyz {
                    white_point: dst_wp,
                    ..
                } = &dst.color
                    && *white_point == *dst_wp
                    && matte_color.is_none()
                    && policy.gamut != GamutPolicy::SoftCompress
                {
                    already_in_dst_xyz = true;
                } else {
                    let target_p = if dst_primaries != Primaries::Unknown {
                        dst_primaries
                    } else {
                        Primaries::Bt709
                    };
                    let m = target_p
                        .xyz_to_rgb_f32(*white_point)
                        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                    ops.push(MicroOp::Gamut3x3(m));
                    cur_primaries = target_p;
                }
            }
            ColorSpace::Oklab { .. } => {
                #[cfg(not(feature = "oklab"))]
                return Err(err_at(ConvertError::FeatureRequired("oklab")));
                #[cfg(feature = "oklab")]
                {
                    let target_p = if dst_primaries != Primaries::Unknown {
                        dst_primaries
                    } else {
                        Primaries::Bt709
                    };
                    let space = crate::oklab::OklabSpace::for_primaries(target_p)
                        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                    ops.push(MicroOp::OklabToRgb(*space.lms_to_rgb().rows()));
                    cur_primaries = target_p;
                }
            }
            ColorSpace::Perceptual { space, .. } => {
                #[cfg(not(feature = "perceptual"))]
                {
                    let _ = space;
                    return Err(err_at(ConvertError::FeatureRequired("perceptual")));
                }
                #[cfg(feature = "perceptual")]
                {
                    let target_p = if dst_primaries != Primaries::Unknown {
                        dst_primaries
                    } else {
                        Primaries::Bt709
                    };
                    ops.push(MicroOp::PerceptualToRgb {
                        space: *space,
                        primaries: target_p,
                        white_nits,
                    });
                    cur_primaries = target_p;
                }
            }
            ColorSpace::Rgb { .. } | ColorSpace::Gray { .. } => {
                if need_linear_space && !src_hlg_tf.is_linear() {
                    let pq_scale = if src_is_pq {
                        10_000.0 / white_nits
                    } else {
                        1.0
                    };
                    ops.push(MicroOp::Eotf {
                        transfer: src_hlg_tf,
                        pq_scale,
                        #[cfg(feature = "hdr-transfers")]
                        hlg: src_prepared_hlg,
                        preserve_extended,
                    });
                }
            }
            _ => return Err(err_at(ConvertError::UnsupportedRoute)),
        }

        // If source was Gray and destination is RGB/Oklab/Perceptual, broadcast Y -> R,G,B.
        if cur_is_gray && !matches!(dst.color, ColorSpace::Gray { .. }) {
            ops.push(MicroOp::GrayToRgb);
            cur_is_gray = false;
            cur_primaries = if dst_primaries != Primaries::Unknown {
                dst_primaries
            } else {
                Primaries::Bt709
            };
        }

        // 6d. Relative-to-relative diffuse white rescale (in linear light).
        if has_relative_white_rescale {
            ops.push(MicroOp::ScaleLinear(relative_white_scale));
        }

        // 6e. Linear-light alpha matte compositing BEFORE tone mapping & gamut clipping!
        // Both pixel RGB and matte color are in diffuse-white-relative linear light here.
        if let Some(matte) = matte_color {
            let matte_primaries = if cur_primaries == Primaries::Unknown {
                Primaries::Bt709
            } else {
                cur_primaries
            };
            let matte_rgb = matte
                .to_primaries(matte_primaries)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            ops.push(MicroOp::CompositeMatte(matte_rgb));
        }

        // 6f. HDR -> SDR Tone Mapping (in linear light, in mapper's working primaries!).
        if is_hdr_to_sdr && policy.tone_map_hdr_to_sdr {
            let mapper_arc = resolve_tone_mapper(policy)?;
            let work_primaries = mapper_arc.working_primaries_for(dst_primaries);
            if !cur_primaries.same_chromaticity(work_primaries) {
                let m = cur_primaries
                    .gamut_matrix_to(work_primaries)
                    .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                ops.push(MicroOp::Gamut3x3(m));
                cur_primaries = work_primaries;
            }
            let req = mapper_arc.requirements();
            if req.full_image {
                return Err(err_at(ConvertError::FullImageMapperNotSupported));
            }
            let (src_peak_nits, target_peak_nits) = if let Some((bs, bd)) = mapper_arc.baked_peaks()
            {
                (bs, bd)
            } else {
                let s_peak = if let Some(p) = hdr_peak {
                    p.nits()
                } else if !req.source_peak_nits {
                    1000.0
                } else if let Transfer::Hlg(HlgMode::Display(r)) = src_hlg_tf {
                    r.peak().nits()
                } else {
                    return Err(err_at(ConvertError::MissingDeclaredPeak));
                };
                (s_peak, policy.sdr_peak.nits())
            };
            let ctx = ToneMapContext {
                source_peak_nits: src_peak_nits,
                target_peak_nits,
                working_primaries: work_primaries,
            };
            let PreparedMapper::Streaming(strip_mapper) = mapper_arc.prepare(&ctx)?;
            let mapper_shared: Arc<dyn StripMapper> = Arc::from(strip_mapper);
            // Input strip is relative-linear (`1.0 = white_nits`); ToneMapper expects `1.0 = src_peak_nits`.
            // Output of ToneMapper is `1.0 = target_peak_nits`. If destination declares a different
            // SDR diffuse white than `target_peak_nits`, rescale cleanly.
            let out_scale = match dst.color.white() {
                Some(dw) if ((target_peak_nits / dw.nits()) - 1.0).abs() > 1e-6 => {
                    target_peak_nits / dw.nits()
                }
                _ => 1.0,
            };
            // Input strip is relative-linear (`1.0 = dst_hlg_white_nits`, after any relative white rescale).
            let strip_white_nits = if has_relative_white_rescale {
                dst_hlg_white_nits
            } else {
                white_nits
            };
            ops.push(MicroOp::ToneMap {
                in_scale: strip_white_nits / src_peak_nits.max(1e-3),
                out_scale,
                mapper: mapper_shared,
            });
        }

        // 6g. Linear-light 3×3 Gamut Matrix (ALWAYS BEFORE encoding or clamping!).
        if !cur_is_gray
            && matches!(dst.color, ColorSpace::Rgb { .. })
            && !cur_primaries.same_chromaticity(dst_primaries)
        {
            let m = cur_primaries
                .gamut_matrix_to(dst_primaries)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            ops.push(MicroOp::Gamut3x3(m));
            cur_primaries = dst_primaries;
        }

        // 6h. Gamut mapping (`SoftCompress` or `ClampUnit` in linear light before OETF!).
        let mut clamped_color = false;
        if !cur_is_gray && matches!(dst.color, ColorSpace::Rgb { .. }) {
            match policy.gamut {
                GamutPolicy::SoftCompress => {
                    #[cfg(not(feature = "oklab"))]
                    return Err(err_at(ConvertError::FeatureRequired("oklab")));
                    #[cfg(feature = "oklab")]
                    {
                        let space = crate::oklab::OklabSpace::for_primaries(cur_primaries)
                            .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                        ops.push(MicroOp::SoftCompress(SoftCompress::new(
                            &space,
                            SoftCompress::DEFAULT_KNEE,
                        )));
                        if !preserve_extended {
                            ops.push(MicroOp::ClampUnit);
                            clamped_color = true;
                        }
                    }
                }
                GamutPolicy::Clip => {
                    if !dst_tf.is_hdr()
                        && !matches!(dst_tf, Transfer::DciCinema)
                        && (pre.gamut_narrows || is_hdr_to_sdr || !dst.layout.sample.is_float())
                    {
                        ops.push(MicroOp::ClampUnit);
                        clamped_color = true;
                    }
                }
                GamutPolicy::PreserveExtended | GamutPolicy::Forbid => {}
            }
        }

        // 6i. RGB -> Gray in linear light (or directly on encoded RGB when `!need_linear_space`).
        // When `LumaDomain::Encoded` AND `need_linear_space` (e.g. Display P3 / Linear / matte -> sRGB Gray),
        // defer `RgbToGray` until AFTER step 6j's destination OETF so luma weights evaluate in
        // destination encoded space rather than linear light!
        let defer_encoded_luma = !cur_is_gray
            && matches!(dst.color, ColorSpace::Gray { .. })
            && need_linear_space
            && policy.luma_domain == LumaDomain::Encoded;
        let deferred_luma_weights = if !cur_is_gray && matches!(dst.color, ColorSpace::Gray { .. })
        {
            let weights = strip::resolve_luma_weights(policy.luma, cur_primaries)
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            if defer_encoded_luma {
                Some(weights)
            } else {
                ops.push(MicroOp::RgbToGray(weights));
                None
            }
        } else {
            None
        };

        // 6j. Encode to destination color model (`Xyz`, `Oklab`, `Perceptual`, or `Oetf`).
        match &dst.color {
            ColorSpace::Xyz { white_point, .. } => {
                if !already_in_dst_xyz {
                    let m = cur_primaries
                        .rgb_to_xyz_f32(*white_point)
                        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                    ops.push(MicroOp::Gamut3x3(m));
                }
                clamped_color = false;
            }
            ColorSpace::Oklab { .. } => {
                #[cfg(not(feature = "oklab"))]
                return Err(err_at(ConvertError::FeatureRequired("oklab")));
                #[cfg(feature = "oklab")]
                {
                    let space = crate::oklab::OklabSpace::for_primaries(cur_primaries)
                        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                    ops.push(MicroOp::RgbToOklab(*space.rgb_to_lms().rows()));
                    clamped_color = false;
                }
            }
            ColorSpace::Perceptual { space, .. } => {
                #[cfg(not(feature = "perceptual"))]
                {
                    let _ = space;
                    return Err(err_at(ConvertError::FeatureRequired("perceptual")));
                }
                #[cfg(feature = "perceptual")]
                {
                    ops.push(MicroOp::RgbToPerceptual {
                        space: *space,
                        primaries: cur_primaries,
                        white_nits,
                    });
                    clamped_color = false;
                }
            }
            ColorSpace::Rgb { .. } | ColorSpace::Gray { .. } => {
                if need_linear_space && !dst_hlg_tf.is_linear() {
                    let pq_inv_scale = if dst_is_pq {
                        white_nits / 10_000.0
                    } else {
                        1.0
                    };
                    ops.push(MicroOp::Oetf {
                        transfer: dst_hlg_tf,
                        pq_inv_scale,
                        #[cfg(feature = "hdr-transfers")]
                        hlg: dst_prepared_hlg,
                        preserve_extended,
                        gray_only: matches!(dst.color, ColorSpace::Gray { .. })
                            && !defer_encoded_luma,
                    });
                    if !preserve_extended && !dst_hlg_tf.is_hdr() {
                        clamped_color = true;
                    }
                }
            }
            _ => return Err(err_at(ConvertError::UnsupportedRoute)),
        }

        if let Some(weights) = deferred_luma_weights {
            ops.push(MicroOp::RgbToGray(weights));
        }

        // 6k. Destination alpha normalization / premultiplication.
        let mut alpha_filled_opaque = matte_color.is_some()
            || !src.layout.order.has_alpha_slot()
            || src.color.alpha() == AlphaMode::Opaque;
        if matches!(dst.color.alpha(), AlphaMode::Opaque) {
            ops.push(MicroOp::FillOpaqueAlpha);
            alpha_filled_opaque = true;
        }
        if dst_premul && (need_linear_space || !src_premul) {
            ops.push(MicroOp::Premul);
        }

        // 6l. Destination narrow range compression.
        if dst_range == SignalRange::Narrow
            && (range_change || need_linear_space || !dst_hlg_tf.is_linear() || narrow_depth_change)
        {
            ops.push(MicroOp::CompressNarrow { bits: dst_bits });
        }

        let src_opaque_alpha = src.color.alpha() == AlphaMode::Opaque;
        let safe_no_subnormals = !has_relative_white_rescale
            && !src.layout.sample.is_float()
            && dst.layout.sample == SampleType::F32
            && matches!(src.layout.sample, SampleType::U8 | SampleType::U16)
            && !pre.gamut_narrows
            && !is_hdr_to_sdr
            && matte_color.is_none()
            && matches!(
                (src_tf, dst_tf),
                (Transfer::Srgb, Transfer::Linear)
                    | (Transfer::Linear, Transfer::Srgb)
                    | (Transfer::Linear, Transfer::Linear)
                    | (Transfer::Srgb, Transfer::Srgb)
            );
        let output_facts = derive_output_facts_math(
            src,
            dst,
            policy.source_facts,
            clamped_color,
            alpha_filled_opaque,
            matte_color.is_some(),
            false,
            safe_no_subnormals,
        );

        Ok(Self {
            src: src.clone(),
            dst: dst.clone(),
            exec: RowExec::Strip {
                ops,
                src_opaque_alpha,
            },
            depth_policy: policy.depth,
            accel,
            output_facts,
        })
    }

    fn build_cms(
        src: ImageSpec,
        dst: ImageSpec,
        policy: &ConvertPolicy,
        matte_active: bool,
        accel: Accelerators,
    ) -> Result<Self, At<ConvertError>> {
        // CMS backends do not composite linear mattes, convert premultiplied alpha,
        // apply Bayer ordered dithering, or run OKLch soft gamut compression.
        if matte_active
            || src.color.alpha() == AlphaMode::Premultiplied
            || dst.color.alpha() == AlphaMode::Premultiplied
            || policy.depth == DepthPolicy::OrderedDither
            || policy.gamut == GamutPolicy::SoftCompress
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let cms = policy
            .cms
            .as_ref()
            .ok_or_else(|| err_at(ConvertError::CmsRequired))?;
        let transform = cms.build_transform(&src, &dst)?;
        let depth = if dst.layout.sample == SampleType::U8 && dst.layout.packing.is_full() {
            ExactDepth::new(SampleType::U8, color_range(&dst.color), 8, 8).ok()
        } else {
            None
        };
        let all_opaque = matches!(
            dst.color.alpha(),
            AlphaMode::None | AlphaMode::Opaque | AlphaMode::Undefined
        );
        let output_facts = SampleFacts::NONE
            .with_depth_opt(depth)
            .with_all_opaque(all_opaque)
            .with_all_grayscale(matches!(dst.color, ColorSpace::Gray { .. }))
            .with_binary_alpha(all_opaque);
        let clip_f32_color = policy.gamut == GamutPolicy::Clip
            && dst.layout.sample == SampleType::F32
            && match &dst.color {
                ColorSpace::Rgb { transfer, .. } | ColorSpace::Gray { transfer, .. } => {
                    !transfer.is_hdr()
                }
                ColorSpace::Icc { .. } => true,
                _ => false,
            };
        Ok(Self {
            output_facts,
            src,
            dst,
            exec: RowExec::Cms {
                transform,
                clip_f32_color,
            },
            depth_policy: policy.depth,
            accel,
        })
    }

    /// Source image specification.
    #[inline]
    #[must_use]
    pub fn src_spec(&self) -> &ImageSpec {
        &self.src
    }

    /// Destination image specification.
    #[inline]
    #[must_use]
    pub fn dst_spec(&self) -> &ImageSpec {
        &self.dst
    }

    /// Proven [`SampleFacts`] guaranteed on pixels emitted by this plan.
    #[inline]
    #[must_use]
    pub const fn output_facts(&self) -> SampleFacts {
        self.output_facts
    }

    /// Whether this plan executes as a direct byte-copy (`Memcpy`) route with no per-sample work.
    ///
    /// Includes identical `src == dst` (unless unverified `AlphaMode::Opaque` requires alpha
    /// canonicalization) as well as same-layout proven-opaque `Straight <-> Premultiplied` or
    /// same-layout metadata relabels that need no pixel transformation.
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        matches!(self.exec, RowExec::Byte(ByteFastPath::Memcpy))
    }

    /// Convert a single row of `width` pixels from `src_row` into `dst_row` with **zero heap allocation**.
    ///
    /// Pass the row index `y` so [`DepthPolicy::OrderedDither`] applies a
    /// consistent 2D Bayer 4×4 matrix across strips and bands.
    pub fn convert_row(
        &self,
        src_row: &[u8],
        dst_row: &mut [u8],
        width: u32,
        y: u32,
    ) -> Result<(), At<ConvertError>> {
        let w = width as usize;
        let need_src = w
            .checked_mul(self.src.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        let need_dst = w
            .checked_mul(self.dst.layout.bytes_per_pixel())
            .ok_or_else(|| err_at(ConvertError::DimensionMismatch))?;
        if src_row.len() < need_src || dst_row.len() < need_dst {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        if w == 0 {
            return Ok(());
        }
        match &self.exec {
            RowExec::Byte(bp) => {
                bp.exec(
                    &src_row[..need_src],
                    &mut dst_row[..need_dst],
                    w,
                    &self.accel,
                );
                Ok(())
            }
            RowExec::FastRow(fr) => {
                strip::exec_fast_row(
                    fr,
                    &src_row[..need_src],
                    &mut dst_row[..need_dst],
                    w,
                    &self.accel,
                );
                Ok(())
            }
            RowExec::Strip {
                ops,
                src_opaque_alpha,
            } => {
                strip::exec_strip_row(
                    self.src.layout,
                    self.dst.layout,
                    ops,
                    self.depth_policy,
                    *src_opaque_alpha,
                    &self.accel,
                    &src_row[..need_src],
                    &mut dst_row[..need_dst],
                    w,
                    y,
                );
                Ok(())
            }
            RowExec::Cms {
                transform,
                clip_f32_color,
            } => {
                let dst_active = &mut dst_row[..need_dst];
                transform.transform_row(&src_row[..need_src], dst_active, width)?;
                if *clip_f32_color {
                    let ch = self.dst.layout.channels();
                    let color_ch = if self.dst.layout.order.has_alpha_slot() {
                        ch - 1
                    } else {
                        ch
                    };
                    if let Ok(floats) = bytemuck::try_cast_slice_mut::<u8, f32>(dst_active) {
                        for px in floats.chunks_exact_mut(ch) {
                            for v in &mut px[..color_ch] {
                                *v = v.clamp(0.0, 1.0);
                            }
                        }
                    } else {
                        let px_bytes = ch * 4;
                        for px in dst_active.chunks_exact_mut(px_bytes) {
                            for slot in px[..color_ch * 4].as_chunks_mut::<4>().0 {
                                let v = f32::from_ne_bytes(*slot).clamp(0.0, 1.0);
                                *slot = v.to_ne_bytes();
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

/// Whole-image conversion plan supporting streaming HDR peak measurement
/// ([`PeakSource::Measured`]), container metadata peaks ([`PeakSource::Declared`]),
/// and uniform execution across [`PixelView`], [`PixelBuf`], and [`BandedBuf`](zenpixels::BandedBuf).
#[derive(Clone, Debug)]
pub struct ImagePlan {
    src: ImageSpec,
    dst: ImageSpec,
    policy: ConvertPolicy,
    peak: PeakSource,
    /// Cached compiled [`RowPlan`] when the route does not depend on per-image
    /// measured/declared HDR peaks or runtime buffer [`SampleFacts`].
    cached_row_plan: Option<RowPlan>,
}

impl ImagePlan {
    /// Policy used to construct this plan (crate-internal for streaming `OutputPlan`).
    #[inline]
    pub(crate) fn policy(&self) -> &ConvertPolicy {
        &self.policy
    }

    /// Validate and construct an image-level conversion plan.
    ///
    /// Accepts either borrowed `&ImageSpec` (zero-clone) or owned `ImageSpec`.
    ///
    /// When the route does not require per-image HDR peak resolution or runtime
    /// buffer facts (`DiscardIfOpaque` / `Forbid` on U16→U8, RGB→Gray, or float→int),
    /// the compiled [`RowPlan`] is built **once** and reused across every `execute` call.
    #[inline]
    pub fn build(
        src: impl AsRef<ImageSpec>,
        dst: impl AsRef<ImageSpec>,
        policy: ConvertPolicy,
        peak: PeakSource,
    ) -> Result<Self, At<ConvertError>> {
        Self::build_inner(src.as_ref(), dst.as_ref(), policy, peak)
    }

    #[inline(never)]
    fn build_inner(
        src: &ImageSpec,
        dst: &ImageSpec,
        policy: ConvertPolicy,
        peak: PeakSource,
    ) -> Result<Self, At<ConvertError>> {
        src.validate().map_err(|e| err_at((*e.error()).into()))?;
        dst.validate().map_err(|e| err_at((*e.error()).into()))?;

        let u16_narrowable = src.layout.sample == SampleType::U16
            && src.layout.packing.is_full()
            && !dst.layout.sample.is_float()
            && color_range(&src.color) == color_range(&dst.color)
            && same_color_meaning(&src.color, &dst.color)
            && is_depth_reducing(src, dst, policy.source_facts);

        let reassociates_alpha = matches!(
            (src.color.alpha(), dst.color.alpha()),
            (AlphaMode::Straight, AlphaMode::Premultiplied)
                | (AlphaMode::Premultiplied, AlphaMode::Straight)
        );

        let needs_runtime_facts = (policy.alpha == AlphaPolicy::DiscardIfOpaque
            && matches!(
                src.color.alpha(),
                AlphaMode::Straight | AlphaMode::Premultiplied
            )
            && matches!(
                dst.color.alpha(),
                AlphaMode::None | AlphaMode::Undefined | AlphaMode::Opaque
            )
            && !policy.source_facts.all_opaque)
            || (policy.depth == DepthPolicy::Forbid
                && reassociates_alpha
                && !policy.source_facts.all_opaque)
            || (policy.depth == DepthPolicy::Forbid && u16_narrowable)
            || ((policy.depth == DepthPolicy::Forbid || policy.gamut == GamutPolicy::Forbid)
                && !matches!(src.color, ColorSpace::Gray { .. })
                && matches!(dst.color, ColorSpace::Gray { .. })
                && !policy.source_facts.all_grayscale)
            || (policy.gamut == GamutPolicy::Forbid
                && src.layout.sample.is_float()
                && !dst.layout.sample.is_float()
                && !policy.source_facts.float.is_unit_interval());

        let is_hdr_tonemap = color_transfer(&src.color).is_hdr()
            && !color_transfer(&dst.color).is_hdr()
            && matches!(dst.color, ColorSpace::Rgb { .. } | ColorSpace::Gray { .. })
            && policy.tone_map_hdr_to_sdr;

        let is_linear_hdr_candidate =
            is_scene_linear_hdr_candidate(src, dst) && policy.tone_map_hdr_to_sdr;

        let mapper_needs_runtime_peak = (is_hdr_tonemap
            && !matches!(peak, PeakSource::Fixed(_))
            && resolve_tone_mapper(&policy)
                .map(|m| m.requirements().source_peak_nits && m.baked_peaks().is_none())
                .unwrap_or(true))
            || (is_linear_hdr_candidate && !matches!(peak, PeakSource::Fixed(_)));

        if !needs_runtime_facts && !mapper_needs_runtime_peak {
            let fixed_peak = match peak {
                PeakSource::Fixed(n) => {
                    if is_linear_hdr_candidate {
                        let ceiling = linear_sdr_ceiling_nits(src, dst, &policy);
                        if n.nits() > ceiling * (1.0 + 1e-4) {
                            Some(n)
                        } else {
                            None
                        }
                    } else {
                        Some(n)
                    }
                }
                _ => None,
            };
            let compiled = RowPlan::build_inner(src, dst, &policy, fixed_peak)?;
            return Ok(Self {
                src: src.clone(),
                dst: dst.clone(),
                policy,
                peak,
                cached_row_plan: Some(compiled),
            });
        }

        // Validate structural/policy compatibility without needlessly invoking CMS/mapper twice.
        let mut probe_policy = policy.clone();
        if probe_policy.alpha == AlphaPolicy::DiscardIfOpaque || reassociates_alpha {
            probe_policy.source_facts.all_opaque = true;
        }
        if probe_policy.depth == DepthPolicy::Forbid && u16_narrowable {
            let sr = color_range(&src.color);
            let min_c = min_color_bits(dst).max(if sr == SignalRange::Narrow { 8 } else { 1 });
            let min_a = if dst.layout.order.has_alpha_slot() {
                raw_channel_bits(dst, dst.layout.channels() - 1).min(16)
            } else {
                16
            };
            probe_policy.source_facts.depth =
                ExactDepth::new(SampleType::U16, sr, min_c.min(16), min_a.max(1)).ok();
        }
        if (probe_policy.depth == DepthPolicy::Forbid || probe_policy.gamut == GamutPolicy::Forbid)
            && !matches!(src.color, ColorSpace::Gray { .. })
            && matches!(dst.color, ColorSpace::Gray { .. })
        {
            probe_policy.source_facts.all_grayscale = true;
        }
        if probe_policy.gamut == GamutPolicy::Forbid
            && src.layout.sample.is_float()
            && !dst.layout.sample.is_float()
        {
            probe_policy.source_facts.float = FloatFacts::UNIT_INTERVAL;
        }
        let pre = preflight_and_classify(src, dst, &probe_policy)?;
        if !pre.uses_cms {
            let test_peak = match peak {
                PeakSource::Fixed(n) => Some(n),
                _ if is_hdr_tonemap => Some(Nits::new(1000.0)),
                _ => None,
            };
            let _ = RowPlan::build_inner(src, dst, &probe_policy, test_peak)?;
        }
        Ok(Self {
            src: src.clone(),
            dst: dst.clone(),
            policy,
            peak,
            cached_row_plan: None,
        })
    }

    /// Execute into an existing [`PixelSink`] (`PixelViewMut`, `PixelBuf`, `BandedBuf`)
    /// with O(1) row scratch memory (never allocates a full-image F32 temporary!).
    pub fn execute_into<S: PixelSource + ?Sized, D: PixelSink + ?Sized>(
        &self,
        src: &S,
        dst: &mut D,
    ) -> Result<(), At<ConvertError>> {
        let w = src.source_width();
        let h = src.source_height();
        if dst.sink_width() != w || dst.sink_height() != h {
            return Err(err_at(ConvertError::DimensionMismatch));
        }
        if src.source_spec() != &self.src || dst.sink_spec() != &self.dst {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }

        let merged_src_facts = merge_facts(self.policy.source_facts, src.source_facts());
        let built_plan: RowPlan;
        let row_plan: &RowPlan = if let Some(cached) = &self.cached_row_plan {
            cached
        } else {
            let mut eff_policy = self.policy.clone();
            eff_policy.source_facts = merged_src_facts;
            let resolved_peak = self.resolve_peak(src, &eff_policy)?;
            built_plan = RowPlan::build_inner(&self.src, &self.dst, &eff_policy, resolved_peak)?;
            &built_plan
        };

        for y in 0..h {
            row_plan.convert_row(src.source_row(y), dst.sink_row_mut(y), w, y)?;
        }
        let preserves_icc_hdr = matches!(self.dst.color, ColorSpace::Icc { .. })
            && !(self.src.layout.sample.is_float() && !self.dst.layout.sample.is_float());
        let out_hdr = if same_photometry(&self.src.color, &self.dst.color)
            && (self.dst.can_represent_hdr() || preserves_icc_hdr)
        {
            src.source_hdr()
        } else {
            HdrMetadata::default()
        };
        let out_facts = match &row_plan.exec {
            RowExec::Byte(bp) => {
                derive_output_facts_byte(&self.src, &self.dst, merged_src_facts, *bp)
            }
            _ => row_plan.output_facts(),
        };
        if same_color_meaning(&self.src.color, &self.dst.color)
            && let Some(orig) = src.source_origin()
        {
            dst.sink_set_origin(orig.clone());
        } else {
            // Clear any stale `ColorOrigin` on a reused sink when color meaning changed!
            dst.sink_set_origin(zenpixels::ColorOrigin::assumed());
        }
        dst.sink_set_orientation(src.source_orientation());
        dst.sink_finish(out_facts, out_hdr);
        Ok(())
    }

    /// Execute into a freshly allocated tightly packed [`PixelBuf`].
    pub fn execute<S: PixelSource + ?Sized>(&self, src: &S) -> Result<PixelBuf, At<ConvertError>> {
        // Validate source spec BEFORE allocating destination memory!
        if src.source_spec() != &self.src {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        let mut out = PixelBuf::new(src.source_width(), src.source_height(), self.dst.clone())
            .map_err(|e| err_at((*e.error()).into()))?;
        self.execute_into(src, &mut out)?;
        Ok(out)
    }

    fn resolve_peak<S: PixelSource + ?Sized>(
        &self,
        src: &S,
        policy: &ConvertPolicy,
    ) -> Result<Option<Nits>, At<ConvertError>> {
        let is_transfer_hdr = color_transfer(&self.src.color).is_hdr()
            && !color_transfer(&self.dst.color).is_hdr()
            && matches!(
                self.dst.color,
                ColorSpace::Rgb { .. } | ColorSpace::Gray { .. }
            );
        let is_linear_hdr_cand = is_scene_linear_hdr_candidate(&self.src, &self.dst);

        if (!is_transfer_hdr && !is_linear_hdr_cand) || !policy.tone_map_hdr_to_sdr {
            return Ok(None);
        }

        if is_linear_hdr_cand {
            let ceiling = linear_sdr_ceiling_nits(&self.src, &self.dst, policy);
            let threshold = ceiling * (1.0 + 1e-4);
            return match self.peak {
                PeakSource::Fixed(n) => {
                    if n.nits() > threshold {
                        Ok(Some(n))
                    } else {
                        Ok(None)
                    }
                }
                PeakSource::Declared => {
                    let hdr = src.source_hdr();
                    if let Some(cll) = hdr.cll
                        && f32::from(cll.max_content_light_level) > threshold
                    {
                        return Ok(Some(Nits::new(f32::from(cll.max_content_light_level))));
                    }
                    if let Some(md) = hdr.mastering
                        && md.max_luminance > threshold
                        && let Some(n) = Nits::try_new(md.max_luminance)
                    {
                        return Ok(Some(n));
                    }
                    Ok(None)
                }
                PeakSource::Measured => {
                    if policy.source_facts.float.is_unit_interval() {
                        return Ok(None);
                    }
                    let white = self
                        .src
                        .color
                        .white()
                        .or_else(|| self.dst.color.white())
                        .or(policy.sdr_white)
                        .unwrap_or(DiffuseWhite::BT2408);
                    let measured = crate::measure::measure_max_nits_streaming(
                        src,
                        white,
                        policy.hlg_rendering,
                    )?;
                    if measured > threshold {
                        Ok(Some(Nits::new(measured)))
                    } else {
                        Ok(None)
                    }
                }
                _ => Err(err_at(ConvertError::UnsupportedRoute)),
            };
        }

        if let Ok(mapper) = resolve_tone_mapper(policy)
            && (!mapper.requirements().source_peak_nits || mapper.baked_peaks().is_some())
        {
            return Ok(None);
        }
        match self.peak {
            PeakSource::Fixed(n) => Ok(Some(n)),
            PeakSource::Declared => {
                let hdr = src.source_hdr();
                if let Some(cll) = hdr.cll
                    && cll.max_content_light_level > 0
                {
                    return Ok(Some(Nits::new(f32::from(cll.max_content_light_level))));
                }
                if let Some(md) = hdr.mastering
                    && let Some(n) = Nits::try_new(md.max_luminance)
                {
                    return Ok(Some(n));
                }
                Err(err_at(ConvertError::MissingDeclaredPeak))
            }
            PeakSource::Measured => {
                let white = self
                    .src
                    .color
                    .white()
                    .or_else(|| self.dst.color.white())
                    .or(policy.sdr_white)
                    .unwrap_or(DiffuseWhite::BT2408);
                let measured =
                    crate::measure::measure_max_nits_streaming(src, white, policy.hlg_rendering)?;
                Ok(Some(Nits::new(measured.max(policy.sdr_peak.nits()))))
            }
            _ => Err(err_at(ConvertError::UnsupportedRoute)),
        }
    }
}

/// One-shot convenience helper: convert a borrowed [`PixelView`] into a new [`PixelBuf`].
///
/// Accepts either borrowed `&ImageSpec` (zero-clone) or owned `ImageSpec`.
#[inline]
pub fn convert_buf(
    src: &PixelView<'_>,
    dst_spec: impl AsRef<ImageSpec>,
    policy: &ConvertPolicy,
    peak: PeakSource,
) -> Result<PixelBuf, At<ConvertError>> {
    let plan = ImagePlan::build_inner(src.spec(), dst_spec.as_ref(), policy.clone(), peak)?;
    plan.execute(src)
}

/// One-shot convenience helper: convert from any [`PixelSource`] into any [`PixelSink`].
pub fn convert_into<S: PixelSource + ?Sized, D: PixelSink + ?Sized>(
    src: &S,
    dst: &mut D,
    policy: &ConvertPolicy,
    peak: PeakSource,
) -> Result<(), At<ConvertError>> {
    let plan = ImagePlan::build_inner(src.source_spec(), dst.sink_spec(), policy.clone(), peak)?;
    plan.execute_into(src, dst)
}

/// One-shot convenience helper: convert `src` to full-range sRGB `RGB8` under
/// [`ConvertPolicy::permissive`] (discarding alpha unchecked and clipping gamut).
pub fn to_rgb8(src: &PixelView<'_>) -> Result<PixelBuf, At<ConvertError>> {
    convert_buf(
        src,
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
}

/// One-shot convenience helper: convert `src` to full-range sRGB straight-alpha `RGBA8`
/// under [`ConvertPolicy::permissive`].
pub fn to_rgba8(src: &PixelView<'_>) -> Result<PixelBuf, At<ConvertError>> {
    convert_buf(
        src,
        ImageSpec::rgba8_srgb(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
}

/// One-shot convenience helper: convert `src` to full-range sRGB `GRAY8` (encoded Y'
/// luma by default) under [`ConvertPolicy::permissive`].
pub fn to_gray8(src: &PixelView<'_>) -> Result<PixelBuf, At<ConvertError>> {
    convert_buf(
        src,
        ImageSpec::gray8_srgb(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
}

/// One-shot convenience helper: convert `src` to full-range sRGB straight-alpha `BGRA8`
/// under [`ConvertPolicy::permissive`].
pub fn to_bgra8(src: &PixelView<'_>) -> Result<PixelBuf, At<ConvertError>> {
    convert_buf(
        src,
        ImageSpec::bgra8_srgb(),
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
    )
}

// ---------------------------------------------------------------------------
// Internal validation, preflight, and planning helpers
// ---------------------------------------------------------------------------

/// Shared policy & feasibility preflight invoked BEFORE CMS, ByteFastPath, FastRow, or Strip.
pub(crate) fn preflight_and_classify(
    src: &ImageSpec,
    dst: &ImageSpec,
    policy: &ConvertPolicy,
) -> Result<PreflightOutcome, At<ConvertError>> {
    src.validate().map_err(|e| err_at((*e.error()).into()))?;
    dst.validate().map_err(|e| err_at((*e.error()).into()))?;

    if src == dst {
        let needs_opaque_canon = src.color.alpha() == AlphaMode::Opaque
            && src.layout.order.has_alpha_slot()
            && !policy.source_facts.all_opaque;
        let can_canon_fast = needs_opaque_canon
            && strip::detect_byte_fast_path(
                src,
                dst,
                policy.depth,
                policy.luma,
                policy.luma_domain,
                false,
                false,
            )
            .is_some();
        let can_canon_strip = needs_opaque_canon
            && !can_canon_fast
            && (src.layout.sample.is_standard_color() || src.layout.packing.is_packed_word())
            && !matches!(
                src.color,
                ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. } | ColorSpace::Untagged { .. }
            )
            && (!color_transfer(&src.color).is_hdr() || cfg!(feature = "hdr-transfers"));
        return Ok(PreflightOutcome {
            matte_color: None,
            range_change: false,
            uses_cms: false,
            is_identity: !(can_canon_fast || can_canon_strip),
            drops_alpha: false,
            alpha_proven_opaque: true,
            drops_chroma: false,
            chroma_proven_gray: true,
            gamut_narrows: false,
            is_hdr_to_sdr: false,
            depth_reduces: false,
        });
    }

    let raw_uses_cms = matches!(
        (&src.color, &dst.color),
        (ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. }, _)
            | (_, ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. })
    );

    if matches!(
        (&src.color, &dst.color),
        (ColorSpace::Untagged { .. }, _) | (_, ColorSpace::Untagged { .. })
    ) {
        return Err(err_at(ConvertError::UnsupportedRoute));
    }

    let src_has_meaningful_alpha = matches!(
        src.color.alpha(),
        AlphaMode::Straight | AlphaMode::Premultiplied
    );
    let dst_drops_alpha = matches!(
        dst.color.alpha(),
        AlphaMode::None | AlphaMode::Undefined | AlphaMode::Opaque
    );
    let drops_alpha = src_has_meaningful_alpha && dst_drops_alpha;
    let alpha_proven_opaque = !drops_alpha || policy.source_facts.all_opaque;
    let matte_color = if drops_alpha {
        match policy.alpha {
            AlphaPolicy::Forbid => return Err(err_at(ConvertError::AlphaLossForbidden)),
            AlphaPolicy::DiscardIfOpaque => {
                if policy.source_facts.all_opaque {
                    None
                } else {
                    return Err(err_at(ConvertError::AlphaLossForbidden));
                }
            }
            AlphaPolicy::DiscardUnchecked => None,
            AlphaPolicy::CompositeOnto(c) => Some(c),
        }
    } else {
        None
    };

    let src_proven_opaque = is_src_proven_opaque(src, policy.source_facts);
    let byte_fast_path_available = strip::detect_byte_fast_path(
        src,
        dst,
        policy.depth,
        policy.luma,
        policy.luma_domain,
        matte_color.is_some(),
        policy.source_facts.all_opaque,
    )
    .is_some();

    // Same-profile ICC/CMYK layout, widening, opaque alpha fill, and identity copy bypass CMS (D01).
    let uses_cms = raw_uses_cms && !byte_fast_path_available;

    if !uses_cms && !byte_fast_path_available {
        if !src.layout.sample.is_standard_color() && !src.layout.packing.is_packed_word() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if !dst.layout.sample.is_standard_color() && !dst.layout.packing.is_packed_word() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
    }

    // Policy checks apply UNIFORMLY to native AND CMS routes!
    let reassociates_alpha = matches!(
        (src.color.alpha(), dst.color.alpha()),
        (AlphaMode::Straight, AlphaMode::Premultiplied)
            | (AlphaMode::Premultiplied, AlphaMode::Straight)
    );
    let lossy_reassociation = reassociates_alpha && !src_proven_opaque;

    let cross_profile_cms = uses_cms
        && (try_substitute_icc(src, policy).is_none() || try_substitute_icc(dst, policy).is_none())
        && !same_color_meaning(&src.color, &dst.color);

    let range_change = check_range_policy(src, dst, policy)?;

    let depth_reduces = is_depth_reducing(src, dst, policy.source_facts)
        || lossy_reassociation
        || (cross_profile_cms && !dst.layout.sample.is_float());
    if depth_reduces && policy.depth == DepthPolicy::Forbid {
        return Err(err_at(ConvertError::DepthReductionForbidden));
    }

    let drops_chroma = !matches!(src.color, ColorSpace::Gray { .. })
        && matches!(dst.color, ColorSpace::Gray { .. });
    let chroma_proven_gray = !drops_chroma || policy.source_facts.all_grayscale;
    if drops_chroma
        && !chroma_proven_gray
        && (policy.depth == DepthPolicy::Forbid || policy.gamut == GamutPolicy::Forbid)
    {
        return Err(err_at(ConvertError::GamutLossForbidden));
    }

    let src_tf = color_transfer(&src.color);
    let dst_tf = color_transfer(&dst.color);
    let is_hdr_to_sdr = src_tf.is_hdr()
        && !dst_tf.is_hdr()
        && matches!(dst.color, ColorSpace::Rgb { .. } | ColorSpace::Gray { .. });

    let gamut_narrows =
        is_gamut_narrowing(color_primaries(&src.color), color_primaries(&dst.color))
            || cross_profile_cms;
    let unproven_float_to_int = src.layout.sample.is_float()
        && !dst.layout.sample.is_float()
        && !policy.source_facts.float.is_unit_interval();
    if policy.gamut == GamutPolicy::Forbid
        && (gamut_narrows
            || (is_hdr_to_sdr && !dst.layout.sample.is_float())
            || unproven_float_to_int)
    {
        return Err(err_at(ConvertError::GamutLossForbidden));
    }
    // Bounded integer destinations cannot preserve out-of-gamut / extended values:
    // forbid silent clipping under `lossless()` (`PreserveExtended` + `DepthPolicy::Forbid`).
    if policy.gamut == GamutPolicy::PreserveExtended
        && policy.depth == DepthPolicy::Forbid
        && !dst.layout.sample.is_float()
        && (gamut_narrows || is_hdr_to_sdr)
    {
        return Err(err_at(ConvertError::GamutLossForbidden));
    }

    // If CMS will be used (no byte fast path and no substitution), verify CMS is configured and supports requested policy.
    if uses_cms
        && !byte_fast_path_available
        && (try_substitute_icc(src, policy).is_none() || try_substitute_icc(dst, policy).is_none())
    {
        if policy.cms.is_none() {
            return Err(err_at(ConvertError::CmsRequired));
        }
        if matte_color.is_some()
            || src.color.alpha() == AlphaMode::Premultiplied
            || dst.color.alpha() == AlphaMode::Premultiplied
            || policy.depth == DepthPolicy::OrderedDither
            || policy.gamut == GamutPolicy::SoftCompress
        {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
    }

    Ok(PreflightOutcome {
        matte_color,
        range_change,
        uses_cms,
        is_identity: false,
        drops_alpha,
        alpha_proven_opaque,
        drops_chroma,
        chroma_proven_gray,
        gamut_narrows,
        is_hdr_to_sdr,
        depth_reduces,
    })
}

pub(crate) fn same_color_meaning(a: &ColorSpace, b: &ColorSpace) -> bool {
    match (a, b) {
        (
            ColorSpace::Rgb {
                primaries: pa,
                transfer: ta,
                range: ra,
                white: wa,
                ..
            },
            ColorSpace::Rgb {
                primaries: pb,
                transfer: tb,
                range: rb,
                white: wb,
                ..
            },
        ) => pa.same_chromaticity(*pb) && ta.same_curve(*tb) && ra == rb && wa == wb,
        (
            ColorSpace::Gray {
                transfer: ta,
                range: ra,
                white: wa,
                ..
            },
            ColorSpace::Gray {
                transfer: tb,
                range: rb,
                white: wb,
                ..
            },
        ) => ta.same_curve(*tb) && ra == rb && wa == wb,
        (
            ColorSpace::Xyz {
                white_point: wa,
                white: wwa,
                ..
            },
            ColorSpace::Xyz {
                white_point: wb,
                white: wwb,
                ..
            },
        ) => wa == wb && wwa == wwb,
        (ColorSpace::Oklab { .. }, ColorSpace::Oklab { .. }) => true,
        (ColorSpace::Perceptual { space: sa, .. }, ColorSpace::Perceptual { space: sb, .. }) => {
            sa == sb
        }
        (
            ColorSpace::Cmyk {
                icc: ia,
                inverted: inva,
                ..
            },
            ColorSpace::Cmyk {
                icc: ib,
                inverted: invb,
                ..
            },
        ) => {
            inva == invb
                && match (ia, ib) {
                    (Some(x), Some(y)) => zenpixels::icc::normalized_eq(x, y),
                    (None, None) => true,
                    _ => false,
                }
        }
        (
            ColorSpace::Icc {
                profile: pa,
                channels: ca,
                intent: ia,
                ..
            },
            ColorSpace::Icc {
                profile: pb,
                channels: cb,
                intent: ib,
                ..
            },
        ) => ca == cb && ia == ib && zenpixels::icc::normalized_eq(pa, pb),
        (ColorSpace::Untagged { channels: ca, .. }, ColorSpace::Untagged { channels: cb, .. }) => {
            ca == cb
        }
        _ => false,
    }
}

pub(crate) fn same_photometry(a: &ColorSpace, b: &ColorSpace) -> bool {
    match (a, b) {
        (
            ColorSpace::Rgb {
                primaries: pa,
                transfer: ta,
                white: wa,
                ..
            },
            ColorSpace::Rgb {
                primaries: pb,
                transfer: tb,
                white: wb,
                ..
            },
        ) => pa.same_chromaticity(*pb) && ta.same_curve(*tb) && wa == wb,
        (
            ColorSpace::Gray {
                transfer: ta,
                white: wa,
                ..
            },
            ColorSpace::Gray {
                transfer: tb,
                white: wb,
                ..
            },
        ) => ta.same_curve(*tb) && wa == wb,
        _ => same_color_meaning(a, b),
    }
}

fn is_src_proven_opaque(src: &ImageSpec, facts: SampleFacts) -> bool {
    matches!(
        src.color.alpha(),
        AlphaMode::None | AlphaMode::Undefined | AlphaMode::Opaque
    ) || !src.layout.order.has_alpha_slot()
        || facts.all_opaque
}

pub(crate) fn is_scene_linear_hdr_candidate(src: &ImageSpec, dst: &ImageSpec) -> bool {
    matches!(color_transfer(&src.color), Transfer::Linear)
        && src.layout.sample.is_float()
        && !dst.can_represent_hdr()
        && matches!(dst.color, ColorSpace::Rgb { .. } | ColorSpace::Gray { .. })
}

fn linear_sdr_ceiling_nits(src: &ImageSpec, dst: &ImageSpec, policy: &ConvertPolicy) -> f32 {
    src.color
        .white()
        .or_else(|| dst.color.white())
        .or(policy.sdr_white)
        .unwrap_or(DiffuseWhite::BT2408)
        .nits()
}

pub(crate) fn merge_facts(policy_facts: SampleFacts, buf_facts: SampleFacts) -> SampleFacts {
    let depth = match (policy_facts.depth, buf_facts.depth) {
        (Some(a), Some(b)) => ExactDepth::combine(a, b).or(Some(a)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    let all_opaque = policy_facts.all_opaque || buf_facts.all_opaque;
    SampleFacts::NONE
        .with_float(policy_facts.float | buf_facts.float)
        .with_depth_opt(depth)
        .with_all_opaque(all_opaque)
        .with_all_grayscale(policy_facts.all_grayscale || buf_facts.all_grayscale)
        .with_binary_alpha(all_opaque || policy_facts.binary_alpha || buf_facts.binary_alpha)
}

pub(crate) fn try_substitute_icc(spec: &ImageSpec, policy: &ConvertPolicy) -> Option<ImageSpec> {
    if !policy.allow_icc_substitution {
        return None;
    }
    match &spec.color {
        ColorSpace::Cmyk { .. } => None,
        ColorSpace::Icc {
            profile,
            channels,
            alpha,
            intent,
        } => {
            // Check ICC header color-space signature matches carrier channel count.
            let sig = zenpixels::icc::profile_color_space(profile)?;
            if (*channels == 1 && sig != zenpixels::IccColorSpace::Gray)
                || (*channels == 3 && sig != zenpixels::IccColorSpace::Rgb)
            {
                return None;
            }
            // Intent-aware lookup: refuses AbsoluteColorimetric and honors per-intent safety mask.
            // When a CMS backend is configured, require Exact (±1 u16) tolerance so approximate LUT
            // profiles use the real CMS instead of silently bypassing it.
            let exact_only = policy.cms.is_some();
            let id = zenpixels::icc::IccIdentification::from_icc_for_intent(
                profile, *intent, exact_only,
            )?;
            let color = if *channels == 1 {
                ColorSpace::Gray {
                    transfer: id.transfer,
                    range: SignalRange::Full,
                    alpha: *alpha,
                    white: None,
                }
            } else if *channels == 3 {
                ColorSpace::Rgb {
                    primaries: id.primaries,
                    transfer: id.transfer,
                    range: SignalRange::Full,
                    alpha: *alpha,
                    white: None,
                }
            } else {
                return None;
            };
            let candidate = ImageSpec::new(spec.layout, color);
            candidate.validate().ok()?;
            Some(candidate)
        }
        _ => Some(spec.clone()),
    }
}

fn effective_color_bits(spec: &ImageSpec) -> u8 {
    if !spec.layout.packing.is_full() {
        let mc = spec.layout.packing.max_code(spec.layout.sample, 0);
        return (32 - mc.leading_zeros()) as u8;
    }
    spec.layout.sample.bits()
}

fn raw_channel_bits(spec: &ImageSpec, channel: usize) -> u8 {
    if !spec.layout.packing.is_full() {
        let mc = spec.layout.packing.max_code(spec.layout.sample, channel);
        return (32 - mc.leading_zeros()) as u8;
    }
    match spec.layout.sample {
        SampleType::U8 | SampleType::I8 => 8,
        SampleType::U16 | SampleType::I16 => 16,
        SampleType::F16 => 11, // IEEE binary16 significand precision (11 bits)
        SampleType::F32 => 24, // IEEE binary32 significand precision (24 bits)
        SampleType::U32 | SampleType::I32 => 32,
        SampleType::F64 => 53,
        SampleType::U64 | SampleType::I64 => 64,
        _ => 32,
    }
}

fn min_color_bits(spec: &ImageSpec) -> u8 {
    let ch = spec.layout.channels();
    let color_ch = ch.saturating_sub(usize::from(
        spec.layout.order.has_alpha_slot() || spec.layout.order.has_padding_slot(),
    ));
    let mut min_b = u8::MAX;
    for c in 0..color_ch {
        min_b = min_b.min(raw_channel_bits(spec, c));
    }
    if min_b == u8::MAX { 8 } else { min_b }
}

/// Per-channel lossless precision and exact code-grid inclusion check (B01 & B02).
///
/// - Float → integer always reduces precision.
/// - Non-identity transfer, gamut matrix, cross-model, range, or diffuse-white changes
///   into a bounded integer destination are non-injective / precision-reducing (B01).
/// - Integer → integer checks exact grid inclusion per channel (`full_range_grid_nests`
///   where `dst_bits % src_bits == 0`, or `narrow_range_grid_nests` where `dst >= src >= 8`)
///   and checks color and alpha grids independently, consulting validated [`ExactDepth`]
///   evidence when the nominal grid does not nest (B02).
fn is_depth_reducing(src: &ImageSpec, dst: &ImageSpec, facts: SampleFacts) -> bool {
    let src_is_float = src.layout.sample.is_float();
    let dst_is_float = dst.layout.sample.is_float();

    // Floating-point -> integer is always precision-reducing (loses extended range & sub-LSB precision).
    if src_is_float && !dst_is_float {
        return true;
    }

    let valid_depth = facts.depth.filter(|d| d.validate_spec(src).is_ok());

    // B01: Any non-identity color math (transfer, gamut matrix, color model, signal range,
    // or diffuse-white rescale) into an integer destination is non-injective / precision-reducing.
    if !dst_is_float {
        let src_tf = color_transfer(&src.color);
        let dst_tf = color_transfer(&dst.color);
        let transfer_changes = !src_tf.same_curve(dst_tf);
        let gamut_changes = matches!(src.color, ColorSpace::Rgb { .. })
            && matches!(dst.color, ColorSpace::Rgb { .. })
            && !color_primaries(&src.color).same_chromaticity(color_primaries(&dst.color));
        let model_changes_lossy = (matches!(
            src.color,
            ColorSpace::Xyz { .. } | ColorSpace::Oklab { .. } | ColorSpace::Perceptual { .. }
        ) || matches!(
            dst.color,
            ColorSpace::Xyz { .. } | ColorSpace::Oklab { .. } | ColorSpace::Perceptual { .. }
        )) && !same_color_meaning(&src.color, &dst.color);
        let range_changes = color_range(&src.color) != color_range(&dst.color);
        let white_changes = match (src.color.white(), dst.color.white()) {
            (Some(a), Some(b)) => ((a.nits() / b.nits()) - 1.0).abs() > 1e-6,
            _ => false,
        };
        if transfer_changes
            || gamut_changes
            || model_changes_lossy
            || range_changes
            || white_changes
        {
            return true;
        }
    }

    let src_ch = src.layout.channels();
    let dst_ch = dst.layout.channels();
    let src_has_alpha = src.layout.order.has_alpha_slot();
    let dst_has_alpha = dst.layout.order.has_alpha_slot();
    let src_color_ch = src_ch.saturating_sub(usize::from(
        src_has_alpha || src.layout.order.has_padding_slot(),
    ));
    let dst_color_ch = dst_ch.saturating_sub(usize::from(
        dst_has_alpha || dst.layout.order.has_padding_slot(),
    ));

    if !src_is_float && !dst_is_float {
        let range = color_range(&src.color);
        // Check every color channel pair for exact code-grid inclusion (B02).
        for sc in 0..src_color_ch {
            let s_bits = raw_channel_bits(src, sc);
            for dc in 0..dst_color_ch {
                let d_bits = raw_channel_bits(dst, dc);
                let nests = match range {
                    SignalRange::Full => full_range_grid_nests(s_bits, d_bits),
                    SignalRange::Narrow => narrow_range_grid_nests(s_bits, d_bits),
                    _ => false,
                };
                if !nests {
                    let proven = valid_depth.is_some_and(|d| d.color_exact_at(d_bits));
                    if !proven {
                        return true;
                    }
                }
            }
        }
        // Check alpha channel independently (always full scale).
        if src_has_alpha
            && dst_has_alpha
            && matches!(
                src.color.alpha(),
                AlphaMode::Straight | AlphaMode::Premultiplied
            )
            && matches!(
                dst.color.alpha(),
                AlphaMode::Straight | AlphaMode::Premultiplied
            )
        {
            let sa_bits = raw_channel_bits(src, src_ch - 1);
            let da_bits = raw_channel_bits(dst, dst_ch - 1);
            if !full_range_grid_nests(sa_bits, da_bits) {
                let proven = valid_depth.is_some_and(|d| d.alpha_exact_at(da_bits));
                if !proven {
                    return true;
                }
            }
        }
        return false;
    }

    // Float destination (`int -> float` or `float -> float`): compare against float significand bits.
    let dst_sig_bits = raw_channel_bits(dst, 0);
    let mut src_max_color_bits = 0u8;
    for c in 0..src_color_ch {
        src_max_color_bits = src_max_color_bits.max(raw_channel_bits(src, c));
    }
    let eff_color_bits = valid_depth.map_or(src_max_color_bits, |d| d.color_bits());
    if dst_sig_bits < eff_color_bits {
        return true;
    }

    if src_has_alpha
        && dst_has_alpha
        && matches!(
            src.color.alpha(),
            AlphaMode::Straight | AlphaMode::Premultiplied
        )
        && matches!(
            dst.color.alpha(),
            AlphaMode::Straight | AlphaMode::Premultiplied
        )
    {
        let sa_bits = raw_channel_bits(src, src_ch - 1);
        let eff_alpha_bits = valid_depth.map_or(sa_bits, |d| d.alpha_bits());
        if dst_sig_bits < eff_alpha_bits {
            return true;
        }
    }

    false
}

fn check_range_policy(
    src: &ImageSpec,
    dst: &ImageSpec,
    policy: &ConvertPolicy,
) -> Result<bool, At<ConvertError>> {
    let sr = color_range(&src.color);
    let dr = color_range(&dst.color);
    if sr != dr {
        if policy.range == RangePolicy::Forbid {
            return Err(err_at(ConvertError::RangeConversionForbidden));
        }
        return Ok(true);
    }
    Ok(false)
}

pub(crate) fn color_range(cs: &ColorSpace) -> SignalRange {
    match *cs {
        ColorSpace::Rgb { range, .. } | ColorSpace::Gray { range, .. } => range,
        _ => SignalRange::Full,
    }
}

pub(crate) fn color_transfer(cs: &ColorSpace) -> Transfer {
    match *cs {
        ColorSpace::Rgb { transfer, .. } | ColorSpace::Gray { transfer, .. } => transfer,
        _ => Transfer::Linear,
    }
}

pub(crate) fn color_primaries(cs: &ColorSpace) -> Primaries {
    match *cs {
        ColorSpace::Rgb { primaries, .. } => primaries,
        ColorSpace::Gray { .. } => Primaries::Bt709,
        _ => Primaries::Unknown,
    }
}

fn needs_linear_domain(
    src: &ColorSpace,
    dst: &ColorSpace,
    matte_active: bool,
    white_rescale: bool,
    policy: &ConvertPolicy,
) -> bool {
    if matte_active || white_rescale {
        return true;
    }
    if matches!(
        src,
        ColorSpace::Oklab { .. } | ColorSpace::Perceptual { .. } | ColorSpace::Xyz { .. }
    ) || matches!(
        dst,
        ColorSpace::Oklab { .. } | ColorSpace::Perceptual { .. } | ColorSpace::Xyz { .. }
    ) {
        return true;
    }
    let st = color_transfer(src);
    let dt = color_transfer(dst);
    if !st.same_curve(dt) {
        return true;
    }
    if matches!(src, ColorSpace::Rgb { .. })
        && matches!(dst, ColorSpace::Rgb { .. })
        && !color_primaries(src).same_chromaticity(color_primaries(dst))
    {
        return true;
    }
    if matches!(src, ColorSpace::Rgb { .. })
        && matches!(dst, ColorSpace::Gray { .. })
        && policy.luma_domain == LumaDomain::LinearLight
    {
        return true;
    }
    if policy.gamut == GamutPolicy::SoftCompress {
        return true;
    }
    false
}

fn resolve_hlg(
    cs: &ColorSpace,
    tf: Transfer,
    other_tf: Transfer,
    override_rendering: Option<HlgRendering>,
    inverse: bool,
    white_nits: f32,
    clip: bool,
) -> Result<(Transfer, Option<PreparedHlg>), At<ConvertError>> {
    match tf {
        Transfer::Hlg(HlgMode::Display(r)) => {
            #[cfg(feature = "hdr-transfers")]
            {
                Ok((
                    Transfer::Hlg(HlgMode::Display(r)),
                    Some(PreparedHlg::new(r, inverse, white_nits, clip)),
                ))
            }
            #[cfg(not(feature = "hdr-transfers"))]
            {
                let _ = (r, inverse, white_nits, clip);
                Err(err_at(ConvertError::HdrFeatureRequired))
            }
        }
        Transfer::Hlg(HlgMode::Scene) => {
            if matches!(other_tf, Transfer::Hlg(HlgMode::Scene)) {
                return Ok((Transfer::Hlg(HlgMode::Scene), None));
            }
            if let Some(r) = override_rendering {
                // Display HLG OOTF is defined over BT.2020 full-range RGB only.
                let is_bt2020_rgb_full = matches!(
                    cs,
                    ColorSpace::Rgb {
                        primaries: Primaries::Bt2020,
                        range: SignalRange::Full,
                        ..
                    }
                );
                if !is_bt2020_rgb_full {
                    return Err(err_at(ConvertError::InvalidSpec(
                        zenpixels::ColorError::HlgDisplayRequiresBt2020Full,
                    )));
                }
                #[cfg(feature = "hdr-transfers")]
                {
                    Ok((
                        Transfer::Hlg(HlgMode::Display(r)),
                        Some(PreparedHlg::new(r, inverse, white_nits, clip)),
                    ))
                }
                #[cfg(not(feature = "hdr-transfers"))]
                {
                    let _ = (r, inverse, white_nits, clip);
                    Err(err_at(ConvertError::HdrFeatureRequired))
                }
            } else {
                Err(err_at(ConvertError::HlgRenderingRequired))
            }
        }
        other => Ok((other, None)),
    }
}

pub(crate) fn resolve_tone_mapper(
    policy: &ConvertPolicy,
) -> Result<Arc<dyn crate::tonemap::ToneMapper>, At<ConvertError>> {
    if let Some(m) = &policy.tone_mapper {
        return Ok(Arc::clone(m));
    }
    #[cfg(feature = "bt2446a")]
    {
        Ok(Arc::new(crate::tonemap::Bt2446A::pipeline_managed()))
    }
    #[cfg(not(feature = "bt2446a"))]
    {
        Err(err_at(ConvertError::FeatureRequired("bt2446a")))
    }
}

pub(crate) fn is_gamut_narrowing(src: Primaries, dst: Primaries) -> bool {
    if src.same_chromaticity(dst) || src == Primaries::Unknown || dst == Primaries::Unknown {
        return false;
    }
    // Exact geometric test over convex RGB cones: source primaries lie inside destination
    // gamut iff every entry of the 3x3 linear transform matrix is non-negative.
    match src.gamut_matrix_to(dst) {
        Some(m) => m.iter().flatten().any(|&v| v < -1e-5),
        None => false,
    }
}

fn derive_output_facts_byte(
    src: &ImageSpec,
    dst: &ImageSpec,
    src_facts: SampleFacts,
    bp: ByteFastPath,
) -> SampleFacts {
    let float = if dst.layout.sample.is_float() {
        src_facts.float
    } else {
        FloatFacts::NONE
    };
    let dst_range = color_range(&dst.color);
    let valid_src_depth = src_facts.depth.filter(|d| d.validate_spec(src).is_ok());
    let fills_alpha = matches!(
        bp,
        ByteFastPath::Rgb8ToRgba8
            | ByteFastPath::Rgb8ToBgra8
            | ByteFastPath::FillAlpha4U8
            | ByteFastPath::SwapRbFillAlpha4U8
            | ByteFastPath::Gray8ToRgba8
            | ByteFastPath::AddAlpha3To4B2(_)
            | ByteFastPath::SwapAddAlpha3To4B2(_)
            | ByteFastPath::GrayToRgbaB2(_)
            | ByteFastPath::AddAlpha3To4B4
            | ByteFastPath::SwapAddAlpha3To4B4
            | ByteFastPath::GrayToRgbaB4
            | ByteFastPath::FillAlpha4B2(_)
            | ByteFastPath::SwapRbFillAlpha4B2(_)
            | ByteFastPath::FillAlpha4B4
            | ByteFastPath::SwapRbFillAlpha4B4
            | ByteFastPath::FillAlpha2U8
            | ByteFastPath::FillAlpha2B2(_)
            | ByteFastPath::FillAlpha2B4
            | ByteFastPath::RepackIntSameDepth {
                fill_opaque_alpha: true,
                ..
            }
            | ByteFastPath::NarrowBitShift {
                fill_opaque_alpha: true,
                ..
            }
    );
    // Exhaustive allow-list: ONLY byte paths that move, broadcast, or exactly widen/repack
    // samples without modifying color code values may propagate source `ExactDepth::color_bits`.
    let preserves_color_values = match bp {
        ByteFastPath::Memcpy
        | ByteFastPath::Rgb8ToRgba8
        | ByteFastPath::Rgb8ToBgra8
        | ByteFastPath::Rgba8ToRgb8
        | ByteFastPath::Bgra8ToRgb8
        | ByteFastPath::SwapRb3U8
        | ByteFastPath::SwapRb4U8
        | ByteFastPath::FillAlpha4U8
        | ByteFastPath::SwapRbFillAlpha4U8
        | ByteFastPath::Gray8ToRgb8
        | ByteFastPath::Gray8ToRgba8
        | ByteFastPath::U8ToU16Exact
        | ByteFastPath::DropAlpha4To3B2
        | ByteFastPath::SwapDropAlpha4To3B2
        | ByteFastPath::AddAlpha3To4B2(_)
        | ByteFastPath::SwapAddAlpha3To4B2(_)
        | ByteFastPath::SwapRb3B2
        | ByteFastPath::SwapRb4B2
        | ByteFastPath::GrayToRgbB2
        | ByteFastPath::GrayToRgbaB2(_)
        | ByteFastPath::DropAlpha4To3B4
        | ByteFastPath::SwapDropAlpha4To3B4
        | ByteFastPath::AddAlpha3To4B4
        | ByteFastPath::SwapAddAlpha3To4B4
        | ByteFastPath::SwapRb3B4
        | ByteFastPath::SwapRb4B4
        | ByteFastPath::GrayToRgbB4
        | ByteFastPath::GrayToRgbaB4
        | ByteFastPath::FillAlpha4B2(_)
        | ByteFastPath::SwapRbFillAlpha4B2(_)
        | ByteFastPath::FillAlpha4B4
        | ByteFastPath::SwapRbFillAlpha4B4
        | ByteFastPath::FillAlpha2U8
        | ByteFastPath::FillAlpha2B2(_)
        | ByteFastPath::FillAlpha2B4
        | ByteFastPath::RepackIntSameDepth { .. }
        | ByteFastPath::NarrowBitShift { .. } => true,
        ByteFastPath::U16ToU8Round
        | ByteFastPath::U16ToU8Trunc
        | ByteFastPath::Rgba8StraightToPremul { .. }
        | ByteFastPath::Rgba8PremulToStraight
        | ByteFastPath::Rgb8ToGray8 { .. } => false,
    };
    let all_opaque = !dst.layout.order.has_alpha_slot()
        || fills_alpha
        || !src.layout.order.has_alpha_slot()
        || src_facts.all_opaque;
    let binary_alpha = all_opaque || src_facts.binary_alpha;
    // Full-range U8 -> U16 widening (`v * 257`) preserves an N-bit sub-grid ONLY when N divides 8
    // (`1, 2, 4, 8`): e.g. U8 3-bit code `36` widens to `9252 != round(65535/7) = 9362`!
    let widen_u8_full_bits = |b: u8| if b > 0 && 8 % b == 0 { b } else { 8 };
    let depth = match dst.layout.sample {
        SampleType::U8 if dst.layout.packing.is_full() => {
            if src.layout.sample == SampleType::U8 && src.layout.packing.is_full() {
                if let Some(d) = valid_src_depth {
                    let c_bits = if preserves_color_values {
                        d.color_bits()
                    } else {
                        8
                    };
                    let a_bits = if all_opaque && dst.layout.order.has_alpha_slot() {
                        1
                    } else if dst.layout.order.has_alpha_slot() {
                        d.alpha_bits()
                    } else {
                        8
                    };
                    ExactDepth::new(SampleType::U8, dst_range, c_bits, a_bits).ok()
                } else {
                    ExactDepth::new(SampleType::U8, dst_range, 8, if all_opaque { 1 } else { 8 })
                        .ok()
                }
            } else {
                ExactDepth::new(SampleType::U8, dst_range, 8, if all_opaque { 1 } else { 8 }).ok()
            }
        }
        SampleType::U16 if dst.layout.packing.is_full() => match bp {
            ByteFastPath::U8ToU16Exact => {
                let c_bits = valid_src_depth.map_or(8, |d| widen_u8_full_bits(d.color_bits()));
                let a_bits = if all_opaque && dst.layout.order.has_alpha_slot() {
                    1
                } else if dst.layout.order.has_alpha_slot() {
                    valid_src_depth.map_or(8, |d| widen_u8_full_bits(d.alpha_bits()))
                } else {
                    16
                };
                ExactDepth::new(SampleType::U16, SignalRange::Full, c_bits, a_bits).ok()
            }
            ByteFastPath::NarrowBitShift {
                src_bits,
                dst_bits: 16,
                ..
            } if src_bits < 16 => {
                let c_bits = valid_src_depth.map_or(src_bits, |d| d.color_bits());
                let a_bits = if all_opaque && dst.layout.order.has_alpha_slot() {
                    1
                } else if dst.layout.order.has_alpha_slot() {
                    valid_src_depth.map_or(src_bits, |d| {
                        if src_bits == 8 {
                            widen_u8_full_bits(d.alpha_bits())
                        } else {
                            src_bits
                        }
                    })
                } else {
                    16
                };
                ExactDepth::new(SampleType::U16, SignalRange::Narrow, c_bits, a_bits).ok()
            }
            _ if src.layout.sample == SampleType::U16 && src.layout.packing.is_full() => {
                valid_src_depth.and_then(|d| {
                    let c_bits = if preserves_color_values {
                        d.color_bits()
                    } else {
                        16
                    };
                    let a_bits = if !dst.layout.order.has_alpha_slot() {
                        16
                    } else if all_opaque {
                        1
                    } else {
                        d.alpha_bits()
                    };
                    ExactDepth::new(SampleType::U16, dst_range, c_bits, a_bits).ok()
                })
            }
            _ => None,
        },
        _ => None,
    };
    let all_grayscale = matches!(dst.color, ColorSpace::Gray { .. })
        || matches!(src.color, ColorSpace::Gray { .. })
        || src_facts.all_grayscale;
    SampleFacts::NONE
        .with_float(float)
        .with_depth_opt(depth)
        .with_all_opaque(all_opaque)
        .with_all_grayscale(all_grayscale)
        .with_binary_alpha(binary_alpha)
}

#[allow(clippy::too_many_arguments)]
fn derive_output_facts_math(
    src: &ImageSpec,
    dst: &ImageSpec,
    src_facts: SampleFacts,
    clamped_color: bool,
    alpha_filled_opaque: bool,
    matte_applied: bool,
    is_f16_cast_only: bool,
    safe_no_subnormals: bool,
) -> SampleFacts {
    let dst_is_rgb_or_gray = matches!(dst.color, ColorSpace::Rgb { .. } | ColorSpace::Gray { .. });
    let float = if dst.layout.sample.is_float() && dst_is_rgb_or_gray {
        if is_f16_cast_only {
            // F32 -> F16 overflows > 65504 to +/-inf and preserves NaN/Inf; only UNIT_INTERVAL
            // survives safely as FINITE + NON_NEGATIVE + AT_MOST_ONE (subnormals can arise in F16!).
            if src_facts.float.is_unit_interval() {
                FloatFacts::FINITE | FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE
            } else {
                FloatFacts::NONE
            }
        } else if clamped_color {
            let alpha_unit = !dst.layout.order.has_alpha_slot()
                || alpha_filled_opaque
                || !src.layout.sample.is_float()
                || src_facts.float.is_unit_interval();
            if alpha_unit {
                if safe_no_subnormals {
                    FloatFacts::UNIT_INTERVAL
                } else {
                    // Clamping guarantees [0, 1] finite, NOT absence of subnormals!
                    FloatFacts::FINITE | FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE
                }
            } else {
                FloatFacts::NONE
            }
        } else {
            FloatFacts::NONE
        }
    } else {
        FloatFacts::NONE
    };

    let all_opaque = matches!(
        dst.color.alpha(),
        AlphaMode::None | AlphaMode::Opaque | AlphaMode::Undefined
    ) || matte_applied
        || alpha_filled_opaque
        || src_facts.all_opaque;
    let binary_alpha = all_opaque || src_facts.binary_alpha;
    // Never propagate U8 exactness into U16 after transfer/gamut/narrow/premul math!
    let depth = if dst.layout.sample == SampleType::U8 && dst.layout.packing.is_full() {
        ExactDepth::new(
            SampleType::U8,
            color_range(&dst.color),
            8,
            if all_opaque { 1 } else { 8 },
        )
        .ok()
    } else {
        None
    };
    let all_grayscale = matches!(dst.color, ColorSpace::Gray { .. });
    SampleFacts::NONE
        .with_float(float)
        .with_depth_opt(depth)
        .with_all_opaque(all_opaque)
        .with_all_grayscale(all_grayscale)
        .with_binary_alpha(binary_alpha)
}
