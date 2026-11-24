//! Codec format negotiation: select the lowest-loss, lowest-effort supported [`ImageSpec`].

use zenpixels::{ColorSpace, ImageSpec, Primaries, Transfer};

use crate::plan::{
    color_primaries, color_transfer, preflight_and_classify, same_color_meaning, try_substitute_icc,
};
use crate::policy::{ConvertPolicy, DepthPolicy, GamutPolicy};

/// Category of information lost when converting `src` to a candidate format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum LossKind {
    /// Zero information loss (exact or lossless widening/swizzle).
    None,
    /// Quantization or bit-depth narrowing (`F32 → U16 → U8`, `RGBA8 → RGB10A2`).
    Depth,
    /// Gamut narrowing, signal-range scaling, or HDR → SDR compression.
    Gamut,
    /// Dropping color into grayscale (`RGB → Gray`).
    Chroma,
    /// Dropping unproven non-opaque alpha.
    Alpha,
}

/// Deterministic cost score for comparing candidate encoder formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FormatCost {
    /// Highest severity information loss (lower is better; `0` = lossless).
    pub loss_rank: u8,
    /// Secondary bitmask of all active loss axes (breaks ties when two candidates share `loss_rank`).
    pub loss_mask: u8,
    /// Conversion effort (`0` = identity zero-copy, `1` = byte/layout swizzle, `2` = color/transfer math).
    pub effort: u8,
    /// Bytes per pixel tiebreaker (smaller preferred when loss & effort match).
    pub bytes_per_pixel: u8,
}

/// Selected negotiation candidate with its cost breakdown.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct NegotiatedFormat {
    /// Index into the caller's `supported` slice.
    pub index: usize,
    /// Chosen destination specification.
    pub spec: ImageSpec,
    /// Worst-case loss category.
    pub loss: LossKind,
    /// Deterministic ranking cost.
    pub cost: FormatCost,
}

/// Format negotiator selecting the best feasible candidate from a codec's supported list.
pub struct Negotiate;

impl Negotiate {
    /// Select the best candidate from `supported` that satisfies `policy`.
    ///
    /// Shares the planner's deterministic feasibility checks (policy gates,
    /// `Transfer::Unknown` linear-math guards, PQ/HLG anchor requirements,
    /// feature gates, and `CmsBackend::supports_specs`) so selected candidates
    /// do not unexpectedly fail in `RowPlan::build`.
    #[must_use]
    pub fn select(
        src: &ImageSpec,
        supported: &[ImageSpec],
        policy: &ConvertPolicy,
    ) -> Option<NegotiatedFormat> {
        let mut best: Option<NegotiatedFormat> = None;
        for (index, cand) in supported.iter().enumerate() {
            let pre = match preflight_and_classify(src, cand, policy) {
                Ok(p) => p,
                Err(_) => continue,
            };

            if !pre.is_identity {
                // Check CMS substitution vs direct CMS feasibility.
                let eff_src_owned = if pre.uses_cms {
                    try_substitute_icc(src, policy)
                } else {
                    None
                };
                let eff_dst_owned = if pre.uses_cms {
                    try_substitute_icc(cand, policy)
                } else {
                    None
                };
                let direct_cms =
                    pre.uses_cms && (eff_src_owned.is_none() || eff_dst_owned.is_none());

                if direct_cms {
                    let Some(cms) = policy.cms.as_ref() else {
                        continue;
                    };
                    if policy.depth == DepthPolicy::OrderedDither
                        || policy.gamut == GamutPolicy::SoftCompress
                    {
                        continue;
                    }
                    if !cms.supports_specs(src, cand) {
                        continue;
                    }
                } else {
                    let eff_src = eff_src_owned.as_ref().unwrap_or(src);
                    let eff_dst = eff_dst_owned.as_ref().unwrap_or(cand);
                    if !native_route_feasible(eff_src, eff_dst, policy, pre.matte_color.is_some()) {
                        continue;
                    }
                }
            }

            let mut loss_mask = 0u8;
            if pre.depth_reduces {
                loss_mask |= 1 << 0;
            }
            if pre.gamut_narrows || pre.is_hdr_to_sdr || pre.range_change {
                loss_mask |= 1 << 1;
            }
            if pre.drops_chroma && !pre.chroma_proven_gray {
                loss_mask |= 1 << 2;
            }
            if pre.drops_alpha && !pre.alpha_proven_opaque {
                loss_mask |= 1 << 3;
            }

            let (loss, loss_rank) = if (loss_mask & (1 << 3)) != 0 {
                (LossKind::Alpha, 4)
            } else if (loss_mask & (1 << 2)) != 0 {
                (LossKind::Chroma, 3)
            } else if (loss_mask & (1 << 1)) != 0 {
                (LossKind::Gamut, 2)
            } else if (loss_mask & (1 << 0)) != 0 {
                (LossKind::Depth, 1)
            } else {
                (LossKind::None, 0)
            };

            let effort = if pre.is_identity {
                0
            } else if same_color_meaning(&src.color, &cand.color)
                && !pre.range_change
                && src.color.alpha() == cand.color.alpha()
            {
                1
            } else {
                2
            };

            let cost = FormatCost {
                loss_rank,
                loss_mask,
                effort,
                bytes_per_pixel: cand.layout.bytes_per_pixel().min(255) as u8,
            };
            let replace = match &best {
                None => true,
                Some(prev) => cost < prev.cost,
            };
            if replace {
                best = Some(NegotiatedFormat {
                    index,
                    spec: cand.clone(),
                    loss,
                    cost,
                });
            }
        }
        best
    }
}

fn native_route_feasible(
    src: &ImageSpec,
    dst: &ImageSpec,
    policy: &ConvertPolicy,
    matte_active: bool,
) -> bool {
    let st = color_transfer(&src.color);
    let dt = color_transfer(&dst.color);
    let sp = color_primaries(&src.color);
    let dp = color_primaries(&dst.color);

    if (st.is_hdr() || dt.is_hdr()) && !cfg!(feature = "hdr-transfers") {
        return false;
    }
    if (matches!(src.color, ColorSpace::Oklab { .. })
        || matches!(dst.color, ColorSpace::Oklab { .. }))
        && !cfg!(feature = "oklab")
    {
        return false;
    }
    if (matches!(src.color, ColorSpace::Perceptual { .. })
        || matches!(dst.color, ColorSpace::Perceptual { .. }))
        && !cfg!(feature = "perceptual")
    {
        return false;
    }
    if policy.gamut == GamutPolicy::SoftCompress && !cfg!(feature = "oklab") {
        return false;
    }

    let src_is_pq = st == Transfer::Pq;
    let dst_is_pq = dt == Transfer::Pq;
    if src_is_pq && !dst_is_pq && dst.color.white().is_none() && policy.sdr_white.is_none() {
        return false;
    }
    if !src_is_pq && dst_is_pq && src.color.white().is_none() && policy.sdr_white.is_none() {
        return false;
    }

    let gamut_changes = sp != dp
        && matches!(src.color, ColorSpace::Rgb { .. })
        && matches!(dst.color, ColorSpace::Rgb { .. });
    if gamut_changes && (sp == Primaries::Unknown || dp == Primaries::Unknown) {
        return false;
    }

    // B09: RGB -> Gray requires resolvable luma weights (FromSourcePrimaries on Unknown fails).
    if matches!(src.color, ColorSpace::Rgb { .. })
        && matches!(dst.color, ColorSpace::Gray { .. })
        && crate::strip::resolve_luma_weights(policy.luma, sp).is_none()
    {
        return false;
    }

    // B05: HDR -> SDR with tone mapping enabled requires a streaming tone mapper
    // (either injected on policy or the built-in `bt2446a` feature).
    let is_hdr_to_sdr = st.is_hdr()
        && !dt.is_hdr()
        && matches!(dst.color, ColorSpace::Rgb { .. } | ColorSpace::Gray { .. });
    if is_hdr_to_sdr && policy.tone_map_hdr_to_sdr {
        match crate::plan::resolve_tone_mapper(policy) {
            Ok(m) if !m.requirements().full_image => {}
            _ => return false,
        }
    }

    let need_linear = st != dt
        || gamut_changes
        || matte_active
        || matches!(
            src.color,
            ColorSpace::Oklab { .. } | ColorSpace::Perceptual { .. } | ColorSpace::Xyz { .. }
        )
        || matches!(
            dst.color,
            ColorSpace::Oklab { .. } | ColorSpace::Perceptual { .. } | ColorSpace::Xyz { .. }
        )
        || (matches!(src.color, ColorSpace::Rgb { .. })
            && matches!(dst.color, ColorSpace::Gray { .. }))
        || (src.color.white() != dst.color.white()
            && src.color.white().is_some()
            && dst.color.white().is_some());

    if (need_linear || st != dt) && (st == Transfer::Unknown || dt == Transfer::Unknown) {
        return false;
    }

    let is_bt2020_full = |cs: &ColorSpace| {
        matches!(
            cs,
            ColorSpace::Rgb {
                primaries: Primaries::Bt2020,
                range: zenpixels::SignalRange::Full,
                ..
            }
        )
    };
    if matches!(st, Transfer::Hlg(zenpixels::HlgMode::Scene))
        && !matches!(dt, Transfer::Hlg(zenpixels::HlgMode::Scene))
        && (policy.hlg_rendering.is_none() || !is_bt2020_full(&src.color))
    {
        return false;
    }
    if matches!(dt, Transfer::Hlg(zenpixels::HlgMode::Scene))
        && !matches!(st, Transfer::Hlg(zenpixels::HlgMode::Scene))
        && (policy.hlg_rendering.is_none() || !is_bt2020_full(&dst.color))
    {
        return false;
    }

    true
}
