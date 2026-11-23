//! Codec format negotiation: select the lowest-loss, lowest-effort supported [`ImageSpec`].

use zenpixels::{AlphaMode, ColorSpace, ImageSpec, Primaries, SampleType};

use crate::plan::RowPlan;
use crate::policy::ConvertPolicy;

/// Category of information lost when converting `src` to a candidate format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum LossKind {
    /// Zero information loss (exact or lossless widening/swizzle).
    None,
    /// Quantization or bit-depth narrowing (`F32 → U16 → U8`).
    Depth,
    /// Gamut narrowing or HDR → SDR compression.
    Gamut,
    /// Dropping color into grayscale (`RGB → Gray`).
    Chroma,
    /// Dropping non-opaque alpha.
    Alpha,
}

/// Deterministic cost score for comparing candidate encoder formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FormatCost {
    /// Highest severity information loss (lower is better; `0` = lossless).
    pub loss_rank: u8,
    /// Conversion effort (`0` = identity zero-copy, `1` = byte swizzle, `2` = transfer/gamut math).
    pub effort: u8,
    /// Bytes per pixel tiebreaker (smaller preferred when loss & effort match).
    pub bytes_per_pixel: u8,
}

/// Selected negotiation candidate with its cost breakdown.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// Every candidate is validated against [`RowPlan::build`] so policy-forbidden
    /// candidates (e.g., alpha drop under `AlphaPolicy::Forbid` or depth reduction
    /// under `DepthPolicy::Forbid`) are never returned.
    #[must_use]
    pub fn select(
        src: &ImageSpec,
        supported: &[ImageSpec],
        policy: &ConvertPolicy,
    ) -> Option<NegotiatedFormat> {
        let mut best: Option<NegotiatedFormat> = None;
        for (index, cand) in supported.iter().enumerate() {
            let plan = match RowPlan::build(
                src.clone(),
                cand.clone(),
                policy,
                Some(zenpixels::Nits::new(1000.0)),
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let loss = classify_loss(src, cand);
            let loss_rank = match loss {
                LossKind::None => 0,
                LossKind::Depth => 1,
                LossKind::Gamut => 2,
                LossKind::Chroma => 3,
                LossKind::Alpha => 4,
            };
            let effort = if plan.is_identity() {
                0
            } else if src.color == cand.color {
                1
            } else {
                2
            };
            let cost = FormatCost {
                loss_rank,
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

fn sample_bits(s: SampleType) -> u8 {
    match s {
        SampleType::U8 | SampleType::I8 => 8,
        SampleType::U16 | SampleType::I16 => 16,
        SampleType::F16 => 24,
        SampleType::F32 | SampleType::U32 | SampleType::I32 => 32,
        SampleType::F64 | SampleType::U64 | SampleType::I64 => 64,
        _ => 32,
    }
}

fn classify_loss(src: &ImageSpec, dst: &ImageSpec) -> LossKind {
    let drops_alpha = matches!(
        src.color.alpha(),
        AlphaMode::Straight | AlphaMode::Premultiplied
    ) && matches!(
        dst.color.alpha(),
        AlphaMode::None | AlphaMode::Undefined | AlphaMode::Opaque
    );
    if drops_alpha {
        return LossKind::Alpha;
    }
    if !matches!(src.color, ColorSpace::Gray { .. }) && matches!(dst.color, ColorSpace::Gray { .. })
    {
        return LossKind::Chroma;
    }
    if src.color.is_hdr() && !dst.color.is_hdr() {
        return LossKind::Gamut;
    }
    if let (ColorSpace::Rgb { primaries: sp, .. }, ColorSpace::Rgb { primaries: dp, .. }) =
        (&src.color, &dst.color)
        && sp != dp
        && matches!(
            (*sp, *dp),
            (
                Primaries::Bt2020,
                Primaries::DisplayP3 | Primaries::Bt709 | Primaries::AdobeRgb
            ) | (Primaries::DisplayP3, Primaries::Bt709 | Primaries::AdobeRgb)
                | (Primaries::AdobeRgb, Primaries::Bt709 | Primaries::DisplayP3)
        )
    {
        return LossKind::Gamut;
    }
    if sample_bits(dst.layout.sample) < sample_bits(src.layout.sample) {
        return LossKind::Depth;
    }
    LossKind::None
}
