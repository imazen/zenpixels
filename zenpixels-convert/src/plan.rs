//! Type-safe row-level ([`RowPlan`]) and image-level ([`ImagePlan`]) conversion plans.
//!
//! - [`RowPlan::build`] accepts **`hdr_peak: Option<Nits>`** — a compile-time
//!   known peak luminance. Passing `PeakSource::Measured` to a single-row
//!   converter is impossible at compile time. Executing a `RowPlan` allocates
//!   **zero heap scratch** (uses the 1 KiB stack [`StripBuf`](crate::strip::StripBuf)).
//! - [`ImagePlan::build`] accepts **`peak: PeakSource`** (`Measured`, `Declared`,
//!   or `Fixed(Nits)`), streams HDR peak measurement in O(1) row memory, runs
//!   uniformly over any [`PixelSource`] → [`PixelSink`] (`PixelView`, `PixelBuf`,
//!   `BandedBuf`), and propagates proven [`SampleFacts`].

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
use crate::strip::{self, ByteFastPath, MicroOp};
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
    /// Zero-float byte-only fast path (memcpy / `garb` swizzles / exact U8→U16).
    Byte(ByteFastPath),
    /// Composable micro-op pipeline running on 64-pixel stack SoA strips.
    Strip(Vec<MicroOp>),
    /// External CMS row transform (`ColorSpace::Icc` / `ColorSpace::Cmyk`).
    Cms {
        transform: Arc<dyn CmsTransform>,
        /// Optional pre-stage (e.g., unpack to CMS carrier).
        pre: Option<alloc::boxed::Box<RowPlan>>,
        /// Optional post-stage (e.g., pack from CMS carrier).
        post: Option<alloc::boxed::Box<RowPlan>>,
        /// Intermediate carrier spec for pre/post staging.
        cms_src_spec: ImageSpec,
        cms_dst_spec: ImageSpec,
    },
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
        src.validate().map_err(|e| err_at((*e.error()).into()))?;
        dst.validate().map_err(|e| err_at((*e.error()).into()))?;

        let accel = accel::resolve(policy.accelerators);

        // 1. Validate sample types (byte-exact identity supports all 11 sample types).
        if src == dst {
            return Ok(Self {
                output_facts: policy.source_facts,
                src,
                dst,
                exec: RowExec::Byte(ByteFastPath::Memcpy),
                depth_policy: policy.depth,
                accel,
            });
        }

        // 2. ICC / CMYK routing (try safe matrix+TRC substitution for known RGB ICC profiles first!).
        if matches!(
            (&src.color, &dst.color),
            (ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. }, _)
                | (_, ColorSpace::Icc { .. } | ColorSpace::Cmyk { .. })
        ) {
            if let (Some(sub_src), Some(sub_dst)) =
                (try_substitute_icc(&src), try_substitute_icc(&dst))
            {
                let mut inner = Self::build(sub_src, sub_dst, policy, hdr_peak)?;
                inner.src = src;
                inner.dst = dst;
                return Ok(inner);
            }
            return Self::build_cms(src, dst, policy, accel);
        }

        if !src.layout.sample.is_standard_color() && !src.layout.packing.is_packed_word() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if !dst.layout.sample.is_standard_color() && !dst.layout.packing.is_packed_word() {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }
        if matches!(
            (&src.color, &dst.color),
            (ColorSpace::Untagged { .. }, _) | (_, ColorSpace::Untagged { .. })
        ) {
            return Err(err_at(ConvertError::UnsupportedRoute));
        }

        // 3. Policy checks: depth, alpha, range, feature gates.
        check_depth_policy(&src, &dst, policy)?;
        let matte_color = check_alpha_policy(&src, &dst, policy)?;
        let range_change = check_range_policy(&src, &dst, policy)?;

        // 4. Fast byte-only path when no color math, tone mapping, or matte compositing is needed.
        if let Some(bp) =
            strip::detect_byte_fast_path(&src, &dst, policy.depth, matte_color.is_some())
        {
            let output_facts = derive_output_facts(&dst, policy.source_facts, false);
            return Ok(Self {
                src,
                dst,
                exec: RowExec::Byte(bp),
                depth_policy: policy.depth,
                accel,
                output_facts,
            });
        }

        // 5. Compile composable micro-op sequence on 64-px stack SoA StripBuf.
        let mut ops: Vec<MicroOp> = Vec::with_capacity(12);
        let preserve_extended =
            policy.gamut == GamutPolicy::PreserveExtended && dst.layout.sample.is_float();

        // 5a. Source narrow range expansion (before unpremul/EOTF!).
        let src_range = color_range(&src.color);
        let dst_range = color_range(&dst.color);
        if src_range == SignalRange::Narrow
            && (range_change || !color_transfer(&src.color).is_linear())
        {
            ops.push(MicroOp::ExpandNarrow {
                sample: src.layout.sample,
            });
        }

        // 5b. Unpremultiply source if premultiplied (when transfer/gamut/matte/alpha changes).
        let src_premul = src.color.alpha() == AlphaMode::Premultiplied;
        let dst_premul = dst.color.alpha() == AlphaMode::Premultiplied;
        let need_linear_space =
            needs_linear_domain(&src.color, &dst.color, matte_color.is_some(), policy);

        if src_premul && (need_linear_space || !dst_premul) {
            ops.push(MicroOp::Unpremul);
        }

        // 5c. Resolve HDR anchors and HLG display rendering conditions.
        let src_tf = color_transfer(&src.color);
        let dst_tf = color_transfer(&dst.color);
        let crosses_hdr_feature = src_tf.is_hdr() || dst_tf.is_hdr();
        if crosses_hdr_feature && !cfg!(feature = "hdr-transfers") {
            return Err(err_at(ConvertError::HdrFeatureRequired));
        }

        let src_is_pq = matches!(src_tf, Transfer::Pq);
        let dst_is_pq = matches!(dst_tf, Transfer::Pq);
        let white_anchor = src
            .color
            .white()
            .or_else(|| dst.color.white())
            .or(policy.sdr_white);

        let crosses_pq_relative = (src_is_pq && !dst_is_pq) || (!src_is_pq && dst_is_pq);
        let white_nits = if crosses_pq_relative {
            white_anchor
                .ok_or_else(|| err_at(ConvertError::MissingHdrAnchor))?
                .nits()
        } else {
            white_anchor.unwrap_or(DiffuseWhite::BT2408).nits()
        };

        // Resolve HLG display rendering when crossing HLG <-> non-HLG.
        #[allow(unused_variables)]
        let (src_hlg_tf, src_prepared_hlg) = resolve_hlg(
            src_tf,
            dst_tf,
            policy.hlg_rendering,
            false,
            white_nits,
            !preserve_extended,
        )?;
        #[allow(unused_variables)]
        let (dst_hlg_tf, dst_prepared_hlg) = resolve_hlg(
            dst_tf,
            src_tf,
            policy.hlg_rendering,
            true,
            white_nits,
            !preserve_extended,
        )?;

        // 5d. Source EOTF (into linear light) or Oklab/Perceptual decode into linear RGB.
        let mut cur_primaries = color_primaries(&src.color);
        let dst_primaries = color_primaries(&dst.color);
        let mut cur_is_gray = matches!(src.color, ColorSpace::Gray { .. });

        match &src.color {
            ColorSpace::Oklab { .. } => {
                #[cfg(not(feature = "oklab"))]
                return Err(err_at(ConvertError::FeatureRequired("oklab")));
                #[cfg(feature = "oklab")]
                {
                    // Decode Oklab directly into destination primaries (CIE XYZ D65 anchor!)
                    // when destination has known primaries, else BT.709.
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

        // 5e. HDR -> SDR Tone Mapping (in linear light, in mapper's working primaries!).
        let is_hdr_to_sdr = src_tf.is_hdr()
            && !dst_tf.is_hdr()
            && matches!(dst.color, ColorSpace::Rgb { .. } | ColorSpace::Gray { .. });
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
            let src_peak_nits = if let Some(p) = hdr_peak {
                p.nits()
            } else if !req.source_peak_nits {
                1000.0
            } else if let Transfer::Hlg(HlgMode::Display(r)) = src_hlg_tf {
                r.peak().nits()
            } else {
                return Err(err_at(ConvertError::MissingDeclaredPeak));
            };
            let ctx = ToneMapContext {
                source_peak_nits: src_peak_nits,
                target_peak_nits: policy.sdr_peak.nits(),
                working_primaries: work_primaries,
            };
            let PreparedMapper::Streaming(strip_mapper) = mapper_arc.prepare(&ctx)?;
            let mapper_shared: Arc<dyn StripMapper> = Arc::from(strip_mapper);
            // Input strip is relative-linear (`1.0 = white_nits`); ToneMapper expects `1.0 = source_peak_nits`.
            // Output of ToneMapper is `1.0 = target_peak_nits`, mapped to SDR `[0, 1]`.
            ops.push(MicroOp::ToneMap {
                in_scale: white_nits / src_peak_nits.max(1.0),
                out_scale: 1.0,
                mapper: mapper_shared,
            });
        }

        // 5f. Linear-light 3×3 Gamut Matrix (ALWAYS BEFORE encoding or clamping!).
        let gamut_narrows = is_gamut_narrowing(cur_primaries, dst_primaries);
        if policy.gamut == GamutPolicy::Forbid
            && (gamut_narrows || (is_hdr_to_sdr && !dst.layout.sample.is_float()))
        {
            return Err(err_at(ConvertError::GamutLossForbidden));
        }

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

        // 5g. Gamut mapping (`SoftCompress` or `ClampUnit` in linear light before OETF!).
        if !cur_is_gray && matches!(dst.color, ColorSpace::Rgb { .. }) {
            match policy.gamut {
                GamutPolicy::SoftCompress => {
                    #[cfg(not(feature = "oklab"))]
                    return Err(err_at(ConvertError::FeatureRequired("oklab")));
                    #[cfg(feature = "oklab")]
                    {
                        if let Some(space) = crate::oklab::OklabSpace::for_primaries(cur_primaries)
                        {
                            ops.push(MicroOp::SoftCompress(SoftCompress::new(
                                &space,
                                SoftCompress::DEFAULT_KNEE,
                            )));
                        }
                        if !preserve_extended {
                            ops.push(MicroOp::ClampUnit);
                        }
                    }
                }
                GamutPolicy::Clip => {
                    if !dst_tf.is_hdr()
                        && (gamut_narrows || is_hdr_to_sdr || !dst.layout.sample.is_float())
                    {
                        ops.push(MicroOp::ClampUnit);
                    }
                }
                GamutPolicy::PreserveExtended | GamutPolicy::Forbid => {}
            }
        }

        // 5h. Alpha matte compositing in linear light (before RGB→Gray and before OETF!).
        if let Some(matte) = matte_color {
            let matte_rgb = matte
                .to_primaries(if cur_primaries == Primaries::Unknown {
                    Primaries::Bt709
                } else {
                    cur_primaries
                })
                .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
            ops.push(MicroOp::CompositeMatte(matte_rgb));
        }

        // 5i. RGB -> Gray in linear light.
        if !cur_is_gray && matches!(dst.color, ColorSpace::Gray { .. }) {
            let weights = strip::resolve_luma_weights(policy.luma, cur_primaries);
            ops.push(MicroOp::RgbToGray(weights));
        }

        // 5j. Encode to destination color model (`Oklab`, `Perceptual`, or `Oetf`).
        match &dst.color {
            ColorSpace::Oklab { .. } => {
                #[cfg(not(feature = "oklab"))]
                return Err(err_at(ConvertError::FeatureRequired("oklab")));
                #[cfg(feature = "oklab")]
                {
                    let space = crate::oklab::OklabSpace::for_primaries(cur_primaries)
                        .ok_or_else(|| err_at(ConvertError::UnsupportedRoute))?;
                    ops.push(MicroOp::RgbToOklab(*space.rgb_to_lms().rows()));
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
                    });
                }
            }
            _ => return Err(err_at(ConvertError::UnsupportedRoute)),
        }

        // 5k. Destination alpha normalization / premultiplication.
        if matches!(dst.color.alpha(), AlphaMode::Opaque)
            && !matches!(
                src.color.alpha(),
                AlphaMode::Opaque | AlphaMode::None | AlphaMode::Undefined
            )
        {
            ops.push(MicroOp::FillOpaqueAlpha);
        }
        if dst_premul && (need_linear_space || !src_premul) {
            ops.push(MicroOp::Premul);
        }

        // 5l. Destination narrow range compression.
        if dst_range == SignalRange::Narrow && (range_change || !dst_hlg_tf.is_linear()) {
            ops.push(MicroOp::CompressNarrow {
                sample: dst.layout.sample,
            });
        }

        let clamped_output = !preserve_extended && !dst_tf.is_hdr();
        let output_facts = derive_output_facts(&dst, policy.source_facts, clamped_output);

        Ok(Self {
            src,
            dst,
            exec: RowExec::Strip(ops),
            depth_policy: policy.depth,
            accel,
            output_facts,
        })
    }

    fn build_cms(
        src: ImageSpec,
        dst: ImageSpec,
        policy: &ConvertPolicy,
        accel: Accelerators,
    ) -> Result<Self, At<ConvertError>> {
        let cms = policy
            .cms
            .as_ref()
            .ok_or_else(|| err_at(ConvertError::CmsRequired))?;
        let transform = cms.build_transform(&src, &dst)?;
        Ok(Self {
            output_facts: derive_output_facts(
                &dst,
                SampleFacts::NONE,
                !dst.layout.sample.is_float(),
            ),
            src: src.clone(),
            dst: dst.clone(),
            exec: RowExec::Cms {
                transform,
                pre: None,
                post: None,
                cms_src_spec: src,
                cms_dst_spec: dst,
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
                    self.accel.byte_layout,
                );
                Ok(())
            }
            RowExec::Strip(ops) => {
                strip::exec_strip_row(
                    self.src.layout,
                    self.dst.layout,
                    ops,
                    self.depth_policy,
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
                pre,
                post,
                cms_src_spec,
                cms_dst_spec,
            } => {
                let _ = (pre, post, cms_src_spec, cms_dst_spec);
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
}

impl ImagePlan {
    /// Validate and construct an image-level conversion plan.
    ///
    /// Note: facts-dependent checks (`AlphaPolicy::DiscardIfOpaque`,
    /// `DepthPolicy::Forbid` on U16→U8) are re-checked with the actual merged
    /// `src.source_facts()` at [`execute`](Self::execute) / [`execute_into`](Self::execute_into)
    /// time so one `ImagePlan` can inspect each input buffer's proven facts.
    pub fn build(
        src: ImageSpec,
        dst: ImageSpec,
        policy: ConvertPolicy,
        peak: PeakSource,
    ) -> Result<Self, At<ConvertError>> {
        src.validate().map_err(|e| err_at((*e.error()).into()))?;
        dst.validate().map_err(|e| err_at((*e.error()).into()))?;
        // Validate non-peak-dependent and non-buffer-fact-dependent parts eagerly.
        let test_peak = match peak {
            PeakSource::Fixed(n) => Some(n),
            PeakSource::Measured | PeakSource::Declared | _ => Some(Nits::new(1000.0)),
        };
        let mut probe_policy = policy.clone();
        if probe_policy.alpha == AlphaPolicy::DiscardIfOpaque {
            probe_policy.source_facts.exact.all_opaque = true;
        }
        if probe_policy.depth == DepthPolicy::Forbid
            && src.layout.sample == SampleType::U16
            && dst.layout.sample == SampleType::U8
        {
            probe_policy.source_facts.exact.effective_bits = 8;
        }
        let _ = RowPlan::build(src.clone(), dst.clone(), &probe_policy, test_peak)?;
        Ok(Self {
            src,
            dst,
            policy,
            peak,
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
        let mut eff_policy = self.policy.clone();
        eff_policy.source_facts = SampleFacts {
            float: eff_policy.source_facts.float | src.source_facts().float,
            exact: zenpixels::ExactDepth {
                effective_bits: eff_policy
                    .source_facts
                    .exact
                    .effective_bits
                    .max(src.source_facts().exact.effective_bits),
                all_opaque: eff_policy.source_facts.exact.all_opaque
                    || src.source_facts().exact.all_opaque,
                all_grayscale: eff_policy.source_facts.exact.all_grayscale
                    || src.source_facts().exact.all_grayscale,
            },
        };

        let resolved_peak = self.resolve_peak(src, &eff_policy)?;
        let row_plan = RowPlan::build(
            self.src.clone(),
            self.dst.clone(),
            &eff_policy,
            resolved_peak,
        )?;
        for y in 0..h {
            row_plan.convert_row(src.source_row(y), dst.sink_row_mut(y), w, y)?;
        }
        let out_hdr = if self.dst.color.is_hdr() {
            src.source_hdr()
        } else {
            HdrMetadata::default()
        };
        dst.sink_finish(row_plan.output_facts(), out_hdr);
        Ok(())
    }

    /// Execute into a freshly allocated tightly packed [`PixelBuf`].
    pub fn execute<S: PixelSource + ?Sized>(&self, src: &S) -> Result<PixelBuf, At<ConvertError>> {
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
        if !self.src.color.is_hdr() || self.dst.color.is_hdr() || !policy.tone_map_hdr_to_sdr {
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
// Internal validation & planning helpers
// ---------------------------------------------------------------------------

fn try_substitute_icc(spec: &ImageSpec) -> Option<ImageSpec> {
    match &spec.color {
        ColorSpace::Cmyk { .. } => None,
        ColorSpace::Icc {
            profile,
            channels,
            alpha,
            ..
        } => {
            let id = zenpixels::icc::IccIdentification::from_icc_substitutable(profile)?;
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

fn effective_precision_rank(spec: &ImageSpec, facts: SampleFacts) -> u8 {
    if !spec.layout.packing.is_full() {
        let mut max_b = 0u8;
        for c in 0..spec.layout.channels() {
            let mc = spec.layout.packing.max_code(spec.layout.sample, c);
            let b = (32 - mc.leading_zeros()) as u8;
            if b > max_b {
                max_b = b;
            }
        }
        return max_b;
    }
    match spec.layout.sample {
        SampleType::U8 | SampleType::I8 => 8,
        SampleType::U16 | SampleType::I16 => {
            if facts.exact.effective_bits > 0 && facts.exact.effective_bits <= 8 {
                8
            } else {
                16
            }
        }
        SampleType::F16 => 24,
        SampleType::F32 | SampleType::U32 | SampleType::I32 => 32,
        SampleType::F64 | SampleType::U64 | SampleType::I64 => 64,
        _ => 32,
    }
}

fn check_depth_policy(
    src: &ImageSpec,
    dst: &ImageSpec,
    policy: &ConvertPolicy,
) -> Result<(), At<ConvertError>> {
    if policy.depth != DepthPolicy::Forbid {
        return Ok(());
    }
    let src_rank = effective_precision_rank(src, policy.source_facts);
    let dst_rank = effective_precision_rank(dst, SampleFacts::NONE);
    if dst_rank < src_rank {
        return Err(err_at(ConvertError::DepthReductionForbidden));
    }
    Ok(())
}

fn check_alpha_policy(
    src: &ImageSpec,
    dst: &ImageSpec,
    policy: &ConvertPolicy,
) -> Result<Option<zenpixels::Color>, At<ConvertError>> {
    let src_has_meaningful_alpha = matches!(
        src.color.alpha(),
        AlphaMode::Straight | AlphaMode::Premultiplied
    );
    let dst_drops_alpha = matches!(
        dst.color.alpha(),
        AlphaMode::None | AlphaMode::Undefined | AlphaMode::Opaque
    );
    if !src_has_meaningful_alpha || !dst_drops_alpha {
        return Ok(None);
    }
    match policy.alpha {
        AlphaPolicy::Forbid => Err(err_at(ConvertError::AlphaLossForbidden)),
        AlphaPolicy::DiscardIfOpaque => {
            if policy.source_facts.exact.all_opaque {
                Ok(None)
            } else {
                Err(err_at(ConvertError::AlphaLossForbidden))
            }
        }
        AlphaPolicy::DiscardUnchecked => Ok(None),
        AlphaPolicy::CompositeOnto(c) => Ok(Some(c)),
    }
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

fn color_range(cs: &ColorSpace) -> SignalRange {
    match *cs {
        ColorSpace::Rgb { range, .. } | ColorSpace::Gray { range, .. } => range,
        _ => SignalRange::Full,
    }
}

fn color_transfer(cs: &ColorSpace) -> Transfer {
    match *cs {
        ColorSpace::Rgb { transfer, .. } | ColorSpace::Gray { transfer, .. } => transfer,
        _ => Transfer::Linear,
    }
}

fn color_primaries(cs: &ColorSpace) -> Primaries {
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
    policy: &ConvertPolicy,
) -> bool {
    if matte_active {
        return true;
    }
    if matches!(
        src,
        ColorSpace::Oklab { .. } | ColorSpace::Perceptual { .. }
    ) || matches!(
        dst,
        ColorSpace::Oklab { .. } | ColorSpace::Perceptual { .. }
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
    // RGB -> Gray computes linear luminance!
    if matches!(src, ColorSpace::Rgb { .. }) && matches!(dst, ColorSpace::Gray { .. }) {
        return true;
    }
    if policy.gamut == GamutPolicy::SoftCompress {
        return true;
    }
    false
}

fn resolve_hlg(
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

fn is_gamut_narrowing(src: Primaries, dst: Primaries) -> bool {
    if src == dst || src == Primaries::Unknown || dst == Primaries::Unknown {
        return false;
    }
    // Wide -> narrower pairs (BT.2020 contains all; P3 and AdobeRGB both extend past BT.709 and each other).
    matches!(
        (src, dst),
        (
            Primaries::Bt2020,
            Primaries::DisplayP3 | Primaries::Bt709 | Primaries::AdobeRgb
        ) | (Primaries::DisplayP3, Primaries::Bt709 | Primaries::AdobeRgb)
            | (Primaries::AdobeRgb, Primaries::Bt709 | Primaries::DisplayP3)
    )
}

fn derive_output_facts(dst: &ImageSpec, src_facts: SampleFacts, clamped_unit: bool) -> SampleFacts {
    let float = if dst.layout.sample.is_float() {
        if clamped_unit {
            FloatFacts::UNIT_INTERVAL
        } else if src_facts.float.is_finite() {
            FloatFacts::FINITE
        } else {
            FloatFacts::NONE
        }
    } else {
        FloatFacts::NONE
    };
    let effective_bits = match dst.layout.sample {
        SampleType::U8 => 8,
        SampleType::U16 => src_facts.exact.effective_bits,
        _ => 0,
    };
    let all_opaque = matches!(
        dst.color.alpha(),
        AlphaMode::None | AlphaMode::Opaque | AlphaMode::Undefined
    );
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
