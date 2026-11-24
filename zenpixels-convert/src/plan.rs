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
    AlphaMode, ColorSpace, DiffuseWhite, FloatFacts, HdrMetadata, HlgMode, HlgRendering, ImageSpec,
    Nits, PeakSource, PixelBuf, PixelSink, PixelSource, PixelView, Primaries, SampleFacts,
    SampleType, SignalRange, Transfer,
};

use crate::accel::{self, Accelerators};
use crate::cms::CmsTransform;
use crate::error::{ConvertError, err_at};
use crate::policy::{AlphaPolicy, ConvertPolicy, DepthPolicy, GamutPolicy, RangePolicy};
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
    Cms { transform: Arc<dyn CmsTransform> },
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
    /// For HDR → SDR tone mapping with a pipeline-managed mapper, pass
    /// `hdr_peak: Some(nits)`. To measure peak automatically over a whole image,
    /// use [`ImagePlan::build`] with [`PeakSource::Measured`].
    pub fn build(
        src: ImageSpec,
        dst: ImageSpec,
        policy: &ConvertPolicy,
        hdr_peak: Option<Nits>,
    ) -> Result<Self, At<ConvertError>> {
        let pre = preflight_and_classify(&src, &dst, policy)?;
        let accel = accel::resolve(policy.accelerators);

        // 1. Byte-exact identity (supports all 11 sample types).
        if pre.is_identity {
            return Ok(Self {
                output_facts: policy.source_facts,
                src,
                dst,
                exec: RowExec::Byte(ByteFastPath::Memcpy),
                depth_policy: policy.depth,
                accel,
            });
        }

        // 2. ICC / CMYK routing (policy checks have ALREADY passed in `preflight_and_classify`!).
        if pre.uses_cms {
            if let (Some(sub_src), Some(sub_dst)) = (
                try_substitute_icc(&src, policy),
                try_substitute_icc(&dst, policy),
            ) {
                let mut inner = Self::build(sub_src, sub_dst, policy, hdr_peak)?;
                inner.src = src;
                inner.dst = dst;
                return Ok(inner);
            }
            return Self::build_cms(src, dst, policy, pre.matte_color.is_some(), accel);
        }

        let matte_color = pre.matte_color;
        let range_change = pre.range_change;

        // 3. Fast byte-only path when no color math, tone mapping, or matte compositing is needed.
        if let Some(bp) =
            strip::detect_byte_fast_path(&src, &dst, policy.depth, matte_color.is_some())
        {
            let output_facts = derive_output_facts_byte(&src, &dst, policy.source_facts, bp);
            return Ok(Self {
                src,
                dst,
                exec: RowExec::Byte(bp),
                depth_policy: policy.depth,
                accel,
                output_facts,
            });
        }

        // 4. Resolve photometry, transfers, HDR white anchors, and HLG BEFORE FastRow or Strip!
        let src_tf = color_transfer(&src.color);
        let dst_tf = color_transfer(&dst.color);
        let src_range = color_range(&src.color);
        let dst_range = color_range(&dst.color);
        let src_bits = effective_color_bits(&src);
        let dst_bits = effective_color_bits(&dst);

        let src_is_pq = matches!(src_tf, Transfer::Pq);
        let dst_is_pq = matches!(dst_tf, Transfer::Pq);

        let relative_white_scale = if !src_is_pq && !dst_is_pq {
            match (src.color.white(), dst.color.white()) {
                (Some(sw), Some(dw)) if (sw.nits() - dw.nits()).abs() > 1e-4 => {
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
            && (src_tf != dst_tf || need_linear_space)
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
            white_nits,
            !preserve_extended,
        )?;
        #[allow(unused_variables)]
        let (dst_hlg_tf, dst_prepared_hlg) = resolve_hlg(
            &dst.color,
            dst_tf,
            src_tf,
            policy.hlg_rendering,
            true,
            white_nits,
            !preserve_extended,
        )?;

        // 5. Direct contiguous single-stage row fast path (zero AoS↔SoA bounce)
        // ONLY after policy, Transfer::Unknown, relative-white, and HLG are resolved!
        let is_hdr_to_sdr = pre.is_hdr_to_sdr;
        if !(is_hdr_to_sdr && policy.tone_map_hdr_to_sdr)
            && !has_relative_white_rescale
            && src_prepared_hlg.is_none()
            && dst_prepared_hlg.is_none()
            && policy.gamut != GamutPolicy::SoftCompress
            && let Some(fr) = strip::detect_fast_row(
                &src,
                &dst,
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
            let output_facts = derive_output_facts_math(
                &src,
                &dst,
                policy.source_facts,
                clamped_color,
                alpha_filled_opaque,
                false,
                matches!(fr, FastRow::F32ToF16 { .. } | FastRow::F16ToF32 { .. }),
            );
            return Ok(Self {
                src,
                dst,
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
            if cur_primaries != work_primaries {
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
                Some(dw) if (dw.nits() - target_peak_nits).abs() > 1e-4 => {
                    target_peak_nits / dw.nits()
                }
                _ => 1.0,
            };
            ops.push(MicroOp::ToneMap {
                in_scale: white_nits / src_peak_nits.max(1e-3),
                out_scale,
                mapper: mapper_shared,
            });
        }

        // 6g. Linear-light 3×3 Gamut Matrix (ALWAYS BEFORE encoding or clamping!).
        if !cur_is_gray
            && matches!(dst.color, ColorSpace::Rgb { .. })
            && cur_primaries != dst_primaries
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
                        && (pre.gamut_narrows || is_hdr_to_sdr || !dst.layout.sample.is_float())
                    {
                        ops.push(MicroOp::ClampUnit);
                        clamped_color = true;
                    }
                }
                GamutPolicy::PreserveExtended | GamutPolicy::Forbid => {}
            }
        }

        // 6i. RGB -> Gray in linear light.
        if !cur_is_gray && matches!(dst.color, ColorSpace::Gray { .. }) {
            let weights = strip::resolve_luma_weights(policy.luma, cur_primaries);
            ops.push(MicroOp::RgbToGray(weights));
        }

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
                        gray_only: matches!(dst.color, ColorSpace::Gray { .. }),
                    });
                    if !preserve_extended && !dst_hlg_tf.is_hdr() {
                        clamped_color = true;
                    }
                }
            }
            _ => return Err(err_at(ConvertError::UnsupportedRoute)),
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
        let output_facts = derive_output_facts_math(
            &src,
            &dst,
            policy.source_facts,
            clamped_color,
            alpha_filled_opaque,
            matte_color.is_some(),
            false,
        );

        Ok(Self {
            src,
            dst,
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
        let output_facts = SampleFacts {
            float: FloatFacts::NONE,
            exact: zenpixels::ExactDepth {
                effective_bits: if dst.layout.sample == SampleType::U8
                    && dst.layout.packing.is_full()
                {
                    8
                } else {
                    0
                },
                all_opaque: matches!(
                    dst.color.alpha(),
                    AlphaMode::None | AlphaMode::Opaque | AlphaMode::Undefined
                ),
                all_grayscale: matches!(dst.color, ColorSpace::Gray { .. }),
            },
        };
        Ok(Self {
            output_facts,
            src,
            dst,
            exec: RowExec::Cms { transform },
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

    /// Whether this plan is an exact byte-copy identity (`src == dst`).
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
            RowExec::Cms { transform } => {
                transform.transform_row(&src_row[..need_src], &mut dst_row[..need_dst], width)
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
    /// When the route does not require per-image HDR peak resolution or runtime
    /// buffer facts (`DiscardIfOpaque` / `Forbid` on U16→U8, RGB→Gray, or float→int),
    /// the compiled [`RowPlan`] is built **once** and reused across every `execute` call.
    pub fn build(
        src: ImageSpec,
        dst: ImageSpec,
        policy: ConvertPolicy,
        peak: PeakSource,
    ) -> Result<Self, At<ConvertError>> {
        src.validate().map_err(|e| err_at((*e.error()).into()))?;
        dst.validate().map_err(|e| err_at((*e.error()).into()))?;

        let u16_full_to_u8_full = src.layout.sample == SampleType::U16
            && dst.layout.sample == SampleType::U8
            && src.layout.packing.is_full()
            && dst.layout.packing.is_full()
            && color_range(&src.color) == SignalRange::Full
            && color_range(&dst.color) == SignalRange::Full;

        let needs_runtime_facts = (policy.alpha == AlphaPolicy::DiscardIfOpaque
            && matches!(
                src.color.alpha(),
                AlphaMode::Straight | AlphaMode::Premultiplied
            )
            && matches!(
                dst.color.alpha(),
                AlphaMode::None | AlphaMode::Undefined | AlphaMode::Opaque
            )
            && !policy.source_facts.exact.all_opaque)
            || (policy.depth == DepthPolicy::Forbid
                && u16_full_to_u8_full
                && policy.source_facts.exact.effective_bits != 8)
            || ((policy.depth == DepthPolicy::Forbid || policy.gamut == GamutPolicy::Forbid)
                && !matches!(src.color, ColorSpace::Gray { .. })
                && matches!(dst.color, ColorSpace::Gray { .. })
                && !policy.source_facts.exact.all_grayscale)
            || (policy.gamut == GamutPolicy::Forbid
                && src.layout.sample.is_float()
                && !dst.layout.sample.is_float()
                && !policy.source_facts.float.is_unit_interval());

        let is_hdr_tonemap = color_transfer(&src.color).is_hdr()
            && !color_transfer(&dst.color).is_hdr()
            && matches!(dst.color, ColorSpace::Rgb { .. } | ColorSpace::Gray { .. })
            && policy.tone_map_hdr_to_sdr;

        let mapper_needs_runtime_peak = is_hdr_tonemap
            && !matches!(peak, PeakSource::Fixed(_))
            && resolve_tone_mapper(&policy)
                .map(|m| m.requirements().source_peak_nits && m.baked_peaks().is_none())
                .unwrap_or(true);

        if !needs_runtime_facts && !mapper_needs_runtime_peak {
            let fixed_peak = match peak {
                PeakSource::Fixed(n) => Some(n),
                _ => None,
            };
            let compiled = RowPlan::build(src.clone(), dst.clone(), &policy, fixed_peak)?;
            return Ok(Self {
                src,
                dst,
                policy,
                peak,
                cached_row_plan: Some(compiled),
            });
        }

        // Validate structural/policy compatibility without needlessly invoking CMS/mapper twice.
        let mut probe_policy = policy.clone();
        if probe_policy.alpha == AlphaPolicy::DiscardIfOpaque {
            probe_policy.source_facts.exact.all_opaque = true;
        }
        if probe_policy.depth == DepthPolicy::Forbid && u16_full_to_u8_full {
            probe_policy.source_facts.exact.effective_bits = 8;
        }
        if (probe_policy.depth == DepthPolicy::Forbid || probe_policy.gamut == GamutPolicy::Forbid)
            && !matches!(src.color, ColorSpace::Gray { .. })
            && matches!(dst.color, ColorSpace::Gray { .. })
        {
            probe_policy.source_facts.exact.all_grayscale = true;
        }
        if probe_policy.gamut == GamutPolicy::Forbid
            && src.layout.sample.is_float()
            && !dst.layout.sample.is_float()
        {
            probe_policy.source_facts.float = FloatFacts::UNIT_INTERVAL;
        }
        let pre = preflight_and_classify(&src, &dst, &probe_policy)?;
        if !pre.uses_cms {
            let test_peak = match peak {
                PeakSource::Fixed(n) => Some(n),
                _ => Some(Nits::new(1000.0)),
            };
            let _ = RowPlan::build(src.clone(), dst.clone(), &probe_policy, test_peak)?;
        }
        Ok(Self {
            src,
            dst,
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

        let built_plan: RowPlan;
        let row_plan: &RowPlan = if let Some(cached) = &self.cached_row_plan {
            cached
        } else {
            let mut eff_policy = self.policy.clone();
            eff_policy.source_facts = merge_facts(eff_policy.source_facts, src.source_facts());
            let resolved_peak = self.resolve_peak(src, &eff_policy)?;
            built_plan = RowPlan::build(
                self.src.clone(),
                self.dst.clone(),
                &eff_policy,
                resolved_peak,
            )?;
            &built_plan
        };

        for y in 0..h {
            row_plan.convert_row(src.source_row(y), dst.sink_row_mut(y), w, y)?;
        }
        let out_hdr =
            if self.dst.color.is_hdr() && same_photometry(&self.src.color, &self.dst.color) {
                src.source_hdr()
            } else {
                HdrMetadata::default()
            };
        let out_facts = if row_plan.is_identity() {
            merge_facts(row_plan.output_facts(), src.source_facts())
        } else {
            row_plan.output_facts()
        };
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
        if !color_transfer(&self.src.color).is_hdr()
            || color_transfer(&self.dst.color).is_hdr()
            || !policy.tone_map_hdr_to_sdr
        {
            return Ok(None);
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
pub fn convert_buf(
    src: &PixelView<'_>,
    dst_spec: ImageSpec,
    policy: &ConvertPolicy,
    peak: PeakSource,
) -> Result<PixelBuf, At<ConvertError>> {
    let plan = ImagePlan::build(src.spec().clone(), dst_spec, policy.clone(), peak)?;
    plan.execute(src)
}

/// One-shot convenience helper: convert from any [`PixelSource`] into any [`PixelSink`].
pub fn convert_into<S: PixelSource + ?Sized, D: PixelSink + ?Sized>(
    src: &S,
    dst: &mut D,
    policy: &ConvertPolicy,
    peak: PeakSource,
) -> Result<(), At<ConvertError>> {
    let plan = ImagePlan::build(
        src.source_spec().clone(),
        dst.sink_spec().clone(),
        policy.clone(),
        peak,
    )?;
    plan.execute_into(src, dst)
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
        return Ok(PreflightOutcome {
            matte_color: None,
            range_change: false,
            uses_cms: false,
            is_identity: true,
            drops_alpha: false,
            alpha_proven_opaque: true,
            drops_chroma: false,
            chroma_proven_gray: true,
            gamut_narrows: false,
            is_hdr_to_sdr: false,
            depth_reduces: false,
        });
    }

    let uses_cms = matches!(
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

    if !uses_cms {
        if !src.layout.sample.is_standard_color() && !src.layout.packing.is_packed_word() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if !dst.layout.sample.is_standard_color() && !dst.layout.packing.is_packed_word() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
    }

    // Policy checks apply UNIFORMLY to native AND CMS routes!
    let depth_reduces = is_depth_reducing(src, dst, policy.source_facts);
    if depth_reduces && policy.depth == DepthPolicy::Forbid {
        return Err(err_at(ConvertError::DepthReductionForbidden));
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
    let alpha_proven_opaque = !drops_alpha || policy.source_facts.exact.all_opaque;
    let matte_color = if drops_alpha {
        match policy.alpha {
            AlphaPolicy::Forbid => return Err(err_at(ConvertError::AlphaLossForbidden)),
            AlphaPolicy::DiscardIfOpaque => {
                if policy.source_facts.exact.all_opaque {
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

    let range_change = check_range_policy(src, dst, policy)?;

    let drops_chroma = !matches!(src.color, ColorSpace::Gray { .. })
        && matches!(dst.color, ColorSpace::Gray { .. });
    let chroma_proven_gray = !drops_chroma || policy.source_facts.exact.all_grayscale;
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
        is_gamut_narrowing(color_primaries(&src.color), color_primaries(&dst.color));
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

    // If CMS will be used (no substitution), verify CMS is configured and supports requested policy.
    if uses_cms
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
        ) => pa == pb && ta == tb && ra == rb && wa == wb,
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
        ) => ta == tb && ra == rb && wa == wb,
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
        ) => pa == pb && ta == tb && wa == wb,
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
        ) => ta == tb && wa == wb,
        _ => same_color_meaning(a, b),
    }
}

fn merge_facts(policy_facts: SampleFacts, buf_facts: SampleFacts) -> SampleFacts {
    let effective_bits = match (
        policy_facts.exact.effective_bits,
        buf_facts.exact.effective_bits,
    ) {
        (0, b) => b,
        (a, 0) => a,
        (a, b) => a.min(b),
    };
    SampleFacts {
        float: policy_facts.float | buf_facts.float,
        exact: zenpixels::ExactDepth {
            effective_bits,
            all_opaque: policy_facts.exact.all_opaque || buf_facts.exact.all_opaque,
            all_grayscale: policy_facts.exact.all_grayscale || buf_facts.exact.all_grayscale,
        },
    }
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

fn channel_bits(spec: &ImageSpec, channel: usize, facts: SampleFacts, allow_u8_proven: bool) -> u8 {
    if !spec.layout.packing.is_full() {
        let mc = spec.layout.packing.max_code(spec.layout.sample, channel);
        return (32 - mc.leading_zeros()) as u8;
    }
    match spec.layout.sample {
        SampleType::U8 | SampleType::I8 => 8,
        SampleType::U16 => {
            if allow_u8_proven
                && color_range(&spec.color) == SignalRange::Full
                && facts.exact.effective_bits > 0
                && facts.exact.effective_bits <= 8
            {
                8
            } else {
                16
            }
        }
        SampleType::I16 => 16,
        SampleType::F16 => 11, // IEEE binary16 significand precision (11 bits)
        SampleType::F32 => 24, // IEEE binary32 significand precision (24 bits)
        SampleType::U32 | SampleType::I32 => 32,
        SampleType::F64 => 53,
        SampleType::U64 | SampleType::I64 => 64,
        _ => 32,
    }
}

/// Per-channel lossless precision check: checks color channels AND alpha channel independently
/// (so `RGBA8 -> RGB10A2` catches alpha 8b → 2b loss, and `U16 -> F16` catches 16b → 11b loss!).
fn is_depth_reducing(src: &ImageSpec, dst: &ImageSpec, facts: SampleFacts) -> bool {
    let src_is_float = src.layout.sample.is_float();
    let dst_is_float = dst.layout.sample.is_float();

    // Floating-point -> integer is always precision-reducing (loses extended range & sub-LSB precision).
    if src_is_float && !dst_is_float {
        return true;
    }

    let allow_u8_proven = src.layout.packing.is_full()
        && dst.layout.packing.is_full()
        && color_range(&src.color) == SignalRange::Full
        && color_range(&dst.color) == SignalRange::Full;

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

    let mut src_max_color_bits = 0u8;
    for c in 0..src_color_ch {
        src_max_color_bits = src_max_color_bits.max(channel_bits(src, c, facts, allow_u8_proven));
    }
    let mut dst_min_color_bits = u8::MAX;
    for c in 0..dst_color_ch {
        dst_min_color_bits = dst_min_color_bits.min(channel_bits(dst, c, SampleFacts::NONE, false));
    }
    if dst_min_color_bits < src_max_color_bits {
        return true;
    }

    // Check alpha channel precision independently when both carry meaningful alpha.
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
        let sa_bits = channel_bits(src, src_ch - 1, facts, allow_u8_proven);
        let da_bits = channel_bits(dst, dst_ch - 1, SampleFacts::NONE, false);
        if da_bits < sa_bits {
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
    if st != dt {
        return true;
    }
    if matches!(src, ColorSpace::Rgb { .. })
        && matches!(dst, ColorSpace::Rgb { .. })
        && color_primaries(src) != color_primaries(dst)
    {
        return true;
    }
    if matches!(src, ColorSpace::Rgb { .. }) && matches!(dst, ColorSpace::Gray { .. }) {
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

fn resolve_tone_mapper(
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
    if src == dst || src == Primaries::Unknown || dst == Primaries::Unknown {
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
    let effective_bits = match dst.layout.sample {
        SampleType::U8 if dst.layout.packing.is_full() => 8,
        SampleType::U16 if dst.layout.packing.is_full() => match bp {
            ByteFastPath::U8ToU16Exact => 8,
            _ if src.layout.sample == SampleType::U16 => src_facts.exact.effective_bits,
            _ => 0,
        },
        _ => 0,
    };
    let all_opaque = matches!(
        dst.color.alpha(),
        AlphaMode::None | AlphaMode::Opaque | AlphaMode::Undefined
    ) || !src.layout.order.has_alpha_slot()
        || src.color.alpha() == AlphaMode::Opaque
        || src_facts.exact.all_opaque;
    let all_grayscale = matches!(dst.color, ColorSpace::Gray { .. })
        || matches!(src.color, ColorSpace::Gray { .. })
        || src_facts.exact.all_grayscale;
    SampleFacts {
        float,
        exact: zenpixels::ExactDepth {
            effective_bits,
            all_opaque,
            all_grayscale,
        },
    }
}

fn derive_output_facts_math(
    src: &ImageSpec,
    dst: &ImageSpec,
    src_facts: SampleFacts,
    clamped_color: bool,
    alpha_filled_opaque: bool,
    matte_applied: bool,
    is_f16_cast_only: bool,
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
                let no_subnormals = !src.layout.sample.is_float()
                    && dst.layout.sample == SampleType::F32
                    && matches!(src.layout.sample, SampleType::U8 | SampleType::U16);
                if no_subnormals {
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

    // Never propagate U8 exactness into U16 after transfer/gamut/narrow/premul math!
    let effective_bits = if dst.layout.sample == SampleType::U8 && dst.layout.packing.is_full() {
        8
    } else {
        0
    };
    let all_opaque = matches!(
        dst.color.alpha(),
        AlphaMode::None | AlphaMode::Opaque | AlphaMode::Undefined
    ) || matte_applied
        || alpha_filled_opaque
        || src_facts.exact.all_opaque;
    let all_grayscale = matches!(dst.color, ColorSpace::Gray { .. });
    SampleFacts {
        float,
        exact: zenpixels::ExactDepth {
            effective_bits,
            all_opaque,
            all_grayscale,
        },
    }
}
