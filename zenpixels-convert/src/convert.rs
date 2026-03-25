//! Row-level pixel conversion kernels.
//!
//! Each kernel converts one row of `width` pixels from a source format to
//! a destination format. Individual step kernels are pure functions with
//! no allocation. Multi-step plans use [`ConvertScratch`] ping-pong
//! buffers to avoid per-row heap allocation in streaming loops.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;

use crate::policy::{AlphaPolicy, ConvertOptions, DepthPolicy};
use crate::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, ConvertError, PixelDescriptor,
    TransferFunction,
};
use whereat::{At, ResultAtExt};

/// Pre-computed conversion plan.
///
/// Stores the chain of steps needed to convert from one format to another.
/// Created once, applied to every row.
#[derive(Clone, Debug)]
pub struct ConvertPlan {
    pub(crate) from: PixelDescriptor,
    pub(crate) to: PixelDescriptor,
    pub(crate) steps: Vec<ConvertStep>,
}

/// A single conversion step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConvertStep {
    /// No-op (identity).
    Identity,
    /// BGRA → RGBA byte swizzle (or vice versa).
    SwizzleBgraRgba,
    /// Add alpha channel (3ch → 4ch), filling with opaque.
    AddAlpha,
    /// Drop alpha channel (4ch → 3ch).
    DropAlpha,
    /// Composite onto solid matte color, then drop alpha (4ch → 3ch).
    ///
    /// `out[c] = (src[c] * alpha + matte[c] * (255 - alpha) + 127) / 255`
    /// Applied at u8 depth. For other depths, values are scaled.
    MatteComposite { r: u8, g: u8, b: u8 },
    /// Gray → RGB (replicate gray to all 3 channels).
    GrayToRgb,
    /// Gray → RGBA (replicate + opaque alpha).
    GrayToRgba,
    /// RGB → Gray (BT.709 luma).
    RgbToGray,
    /// RGBA → Gray (BT.709 luma, drop alpha).
    RgbaToGray,
    /// GrayAlpha → RGBA (replicate gray, keep alpha).
    GrayAlphaToRgba,
    /// GrayAlpha → RGB (replicate gray, drop alpha).
    GrayAlphaToRgb,
    /// Gray → GrayAlpha (add opaque alpha).
    GrayToGrayAlpha,
    /// GrayAlpha → Gray (drop alpha).
    GrayAlphaToGray,
    /// sRGB u8 → linear f32 (per channel, EOTF).
    SrgbU8ToLinearF32,
    /// Linear f32 → sRGB u8 (per channel, OETF).
    LinearF32ToSrgbU8,
    /// Naive u8 → f32 (v / 255.0, no gamma).
    NaiveU8ToF32,
    /// Naive f32 → u8 (clamp * 255 + 0.5, no gamma).
    NaiveF32ToU8,
    /// u16 → u8 ((v * 255 + 32768) >> 16).
    U16ToU8,
    /// u8 → u16 (v * 257).
    U8ToU16,
    /// u16 → f32 (v / 65535.0).
    U16ToF32,
    /// f32 → u16 (clamp * 65535 + 0.5).
    F32ToU16,
    /// PQ (SMPTE ST 2084) u16 → linear f32 (EOTF).
    PqU16ToLinearF32,
    /// Linear f32 → PQ u16 (inverse EOTF / OETF).
    LinearF32ToPqU16,
    /// PQ f32 [0,1] → linear f32 (EOTF, no depth change).
    PqF32ToLinearF32,
    /// Linear f32 → PQ f32 [0,1] (OETF, no depth change).
    LinearF32ToPqF32,
    /// HLG (ARIB STD-B67) u16 → linear f32 (EOTF).
    HlgU16ToLinearF32,
    /// Linear f32 → HLG u16 (OETF).
    LinearF32ToHlgU16,
    /// HLG f32 [0,1] → linear f32 (EOTF, no depth change).
    HlgF32ToLinearF32,
    /// Linear f32 → HLG f32 [0,1] (OETF, no depth change).
    LinearF32ToHlgF32,
    /// sRGB f32 [0,1] → linear f32 (EOTF, no depth change).
    SrgbF32ToLinearF32,
    /// Linear f32 → sRGB f32 [0,1] (OETF, no depth change).
    LinearF32ToSrgbF32,
    /// BT.709 f32 [0,1] → linear f32 (EOTF, no depth change).
    Bt709F32ToLinearF32,
    /// Linear f32 → BT.709 f32 [0,1] (OETF, no depth change).
    LinearF32ToBt709F32,
    /// Straight → Premultiplied alpha.
    StraightToPremul,
    /// Premultiplied → Straight alpha.
    PremulToStraight,
    /// Linear RGB f32 → Oklab f32 (3-channel color model change).
    LinearRgbToOklab,
    /// Oklab f32 → Linear RGB f32 (3-channel color model change).
    OklabToLinearRgb,
    /// Linear RGBA f32 → Oklaba f32 (4-channel, alpha preserved).
    LinearRgbaToOklaba,
    /// Oklaba f32 → Linear RGBA f32 (4-channel, alpha preserved).
    OklabaToLinearRgba,
}

impl ConvertPlan {
    /// Create a conversion plan from `from` to `to`.
    ///
    /// Returns `Err` if no conversion path exists.
    #[track_caller]
    pub fn new(from: PixelDescriptor, to: PixelDescriptor) -> Result<Self, At<ConvertError>> {
        if from == to {
            return Ok(Self {
                from,
                to,
                steps: vec![ConvertStep::Identity],
            });
        }

        let mut steps = Vec::with_capacity(3);

        // Step 1: Layout conversion (within same depth class).
        // Step 2: Depth conversion.
        // Step 3: Alpha mode conversion.
        //
        // For cross-depth conversions, we convert layout at the source depth
        // first, then change depth. This minimizes the number of channels
        // we need to depth-convert.

        let need_depth_change = from.channel_type() != to.channel_type();
        let need_layout_change = from.layout() != to.layout();
        let need_alpha_change =
            from.alpha() != to.alpha() && from.alpha().is_some() && to.alpha().is_some();

        // Depth/TF steps are needed when depth changes, or when both are F32
        // and transfer functions differ.
        let need_depth_or_tf = need_depth_change
            || (from.channel_type() == ChannelType::F32 && from.transfer() != to.transfer());

        // If we need to change depth AND layout, plan the optimal order.
        if need_layout_change {
            // When going to fewer channels, convert layout first (less depth work).
            // When going to more channels, convert depth first (less layout work).
            //
            // Exception: Oklab layout steps require f32 data. When the source
            // is integer (U8/U16) and the layout change involves Oklab, we must
            // convert depth first regardless of channel count.
            let src_ch = from.layout().channels();
            let dst_ch = to.layout().channels();
            let involves_oklab =
                matches!(from.layout(), ChannelLayout::Oklab | ChannelLayout::OklabA)
                    || matches!(to.layout(), ChannelLayout::Oklab | ChannelLayout::OklabA);

            // Oklab conversion requires known primaries for the RGB→LMS matrix.
            if involves_oklab && from.primaries == ColorPrimaries::Unknown {
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            }

            let depth_first = need_depth_or_tf
                && (dst_ch > src_ch || (involves_oklab && from.channel_type() != ChannelType::F32));

            if depth_first {
                // Depth first, then layout.
                steps.extend(
                    depth_steps(
                        from.channel_type(),
                        to.channel_type(),
                        from.transfer(),
                        to.transfer(),
                    )
                    .map_err(|e| whereat::at!(e))?,
                );
                steps.extend(layout_steps(from.layout(), to.layout()));
            } else {
                // Layout first, then depth.
                steps.extend(layout_steps(from.layout(), to.layout()));
                if need_depth_or_tf {
                    steps.extend(
                        depth_steps(
                            from.channel_type(),
                            to.channel_type(),
                            from.transfer(),
                            to.transfer(),
                        )
                        .map_err(|e| whereat::at!(e))?,
                    );
                }
            }
        } else if need_depth_or_tf {
            steps.extend(
                depth_steps(
                    from.channel_type(),
                    to.channel_type(),
                    from.transfer(),
                    to.transfer(),
                )
                .map_err(|e| whereat::at!(e))?,
            );
        }

        // Alpha mode conversion (if both have alpha and modes differ).
        if need_alpha_change {
            match (from.alpha(), to.alpha()) {
                (Some(AlphaMode::Straight), Some(AlphaMode::Premultiplied)) => {
                    steps.push(ConvertStep::StraightToPremul);
                }
                (Some(AlphaMode::Premultiplied), Some(AlphaMode::Straight)) => {
                    steps.push(ConvertStep::PremulToStraight);
                }
                _ => {}
            }
        }

        if steps.is_empty() {
            // Transfer-only difference or alpha-mode-only: identity path.
            steps.push(ConvertStep::Identity);
        }

        Ok(Self { from, to, steps })
    }

    /// Create a conversion plan with explicit policy enforcement.
    ///
    /// Validates that the planned conversion steps are allowed by the given
    /// policies before creating the plan. Returns an error if a forbidden
    /// operation would be required.
    #[track_caller]
    pub fn new_explicit(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &ConvertOptions,
    ) -> Result<Self, At<ConvertError>> {
        // Check alpha removal policy.
        let drops_alpha = from.alpha().is_some() && to.alpha().is_none();
        if drops_alpha && options.alpha_policy == AlphaPolicy::Forbid {
            return Err(whereat::at!(ConvertError::AlphaRemovalForbidden));
        }

        // Check depth reduction policy.
        let reduces_depth = from.channel_type().byte_size() > to.channel_type().byte_size();
        if reduces_depth && options.depth_policy == DepthPolicy::Forbid {
            return Err(whereat::at!(ConvertError::DepthReductionForbidden));
        }

        // Check RGB→Gray requires luma coefficients.
        let src_is_rgb = matches!(
            from.layout(),
            ChannelLayout::Rgb | ChannelLayout::Rgba | ChannelLayout::Bgra
        );
        let dst_is_gray = matches!(to.layout(), ChannelLayout::Gray | ChannelLayout::GrayAlpha);
        if src_is_rgb && dst_is_gray && options.luma.is_none() {
            return Err(whereat::at!(ConvertError::RgbToGray));
        }

        let mut plan = Self::new(from, to).at()?;

        // Replace DropAlpha with MatteComposite when policy is CompositeOnto.
        if drops_alpha && let AlphaPolicy::CompositeOnto { r, g, b } = options.alpha_policy {
            for step in &mut plan.steps {
                if matches!(step, ConvertStep::DropAlpha) {
                    *step = ConvertStep::MatteComposite { r, g, b };
                }
            }
        }

        Ok(plan)
    }

    /// Compose two plans into one: apply `self` then `other`.
    ///
    /// The composed plan executes both conversions in a single `convert_row`
    /// call, using one intermediate buffer instead of two. Adjacent inverse
    /// steps are cancelled (e.g., `SrgbU8ToLinearF32` + `LinearF32ToSrgbU8`
    /// → identity).
    ///
    /// Returns `None` if `self.to` != `other.from` (incompatible plans).
    pub fn compose(&self, other: &Self) -> Option<Self> {
        if self.to != other.from {
            return None;
        }

        let mut steps = self.steps.clone();

        // Append other's steps, skipping its Identity if present.
        for &step in &other.steps {
            if step == ConvertStep::Identity {
                continue;
            }
            steps.push(step);
        }

        // Peephole: cancel adjacent inverse pairs.
        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            while i + 1 < steps.len() {
                if are_inverse(steps[i], steps[i + 1]) {
                    steps.remove(i + 1);
                    steps.remove(i);
                    changed = true;
                    // Don't advance — check the new adjacent pair.
                } else {
                    i += 1;
                }
            }
        }

        // If everything cancelled, produce identity.
        if steps.is_empty() {
            steps.push(ConvertStep::Identity);
        }

        // Remove leading/trailing Identity if there are real steps.
        if steps.len() > 1 {
            steps.retain(|s| *s != ConvertStep::Identity);
            if steps.is_empty() {
                steps.push(ConvertStep::Identity);
            }
        }

        Some(Self {
            from: self.from,
            to: other.to,
            steps,
        })
    }

    /// True if conversion is a no-op.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.steps.len() == 1 && self.steps[0] == ConvertStep::Identity
    }

    /// Maximum bytes-per-pixel across all intermediate formats in the plan.
    ///
    /// Used to pre-allocate scratch buffers for streaming conversion.
    pub(crate) fn max_intermediate_bpp(&self) -> usize {
        let mut desc = self.from;
        let mut max_bpp = desc.bytes_per_pixel();
        for &step in &self.steps {
            desc = intermediate_desc(desc, step);
            max_bpp = max_bpp.max(desc.bytes_per_pixel());
        }
        max_bpp
    }

    /// Source descriptor.
    pub fn from(&self) -> PixelDescriptor {
        self.from
    }

    /// Target descriptor.
    pub fn to(&self) -> PixelDescriptor {
        self.to
    }
}

/// Determine the layout conversion step(s).
///
/// Some layout conversions require two steps (e.g., BGRA -> RGB needs
/// swizzle + drop alpha). Returns up to 2 steps.
fn layout_steps(from: ChannelLayout, to: ChannelLayout) -> Vec<ConvertStep> {
    if from == to {
        return Vec::new();
    }
    match (from, to) {
        (ChannelLayout::Bgra, ChannelLayout::Rgba) | (ChannelLayout::Rgba, ChannelLayout::Bgra) => {
            vec![ConvertStep::SwizzleBgraRgba]
        }
        (ChannelLayout::Rgb, ChannelLayout::Rgba) => vec![ConvertStep::AddAlpha],
        (ChannelLayout::Rgb, ChannelLayout::Bgra) => {
            // Rgb -> RGBA -> BGRA: add alpha then swizzle.
            vec![ConvertStep::AddAlpha, ConvertStep::SwizzleBgraRgba]
        }
        (ChannelLayout::Rgba, ChannelLayout::Rgb) => vec![ConvertStep::DropAlpha],
        (ChannelLayout::Bgra, ChannelLayout::Rgb) => {
            // BGRA -> RGBA -> RGB: swizzle then drop alpha.
            vec![ConvertStep::SwizzleBgraRgba, ConvertStep::DropAlpha]
        }
        (ChannelLayout::Gray, ChannelLayout::Rgb) => vec![ConvertStep::GrayToRgb],
        (ChannelLayout::Gray, ChannelLayout::Rgba) => vec![ConvertStep::GrayToRgba],
        (ChannelLayout::Gray, ChannelLayout::Bgra) => {
            // Gray -> RGBA -> BGRA: expand then swizzle.
            vec![ConvertStep::GrayToRgba, ConvertStep::SwizzleBgraRgba]
        }
        (ChannelLayout::Rgb, ChannelLayout::Gray) => vec![ConvertStep::RgbToGray],
        (ChannelLayout::Rgba, ChannelLayout::Gray) => vec![ConvertStep::RgbaToGray],
        (ChannelLayout::Bgra, ChannelLayout::Gray) => {
            // BGRA -> RGBA -> Gray: swizzle then to gray.
            vec![ConvertStep::SwizzleBgraRgba, ConvertStep::RgbaToGray]
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::Rgba) => vec![ConvertStep::GrayAlphaToRgba],
        (ChannelLayout::GrayAlpha, ChannelLayout::Bgra) => {
            // GrayAlpha -> RGBA -> BGRA: expand then swizzle.
            vec![ConvertStep::GrayAlphaToRgba, ConvertStep::SwizzleBgraRgba]
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::Rgb) => vec![ConvertStep::GrayAlphaToRgb],
        (ChannelLayout::Gray, ChannelLayout::GrayAlpha) => vec![ConvertStep::GrayToGrayAlpha],
        (ChannelLayout::GrayAlpha, ChannelLayout::Gray) => vec![ConvertStep::GrayAlphaToGray],

        // Oklab ↔ RGB conversions (via linear RGB).
        (ChannelLayout::Rgb, ChannelLayout::Oklab) => vec![ConvertStep::LinearRgbToOklab],
        (ChannelLayout::Oklab, ChannelLayout::Rgb) => vec![ConvertStep::OklabToLinearRgb],
        (ChannelLayout::Rgba, ChannelLayout::OklabA) => vec![ConvertStep::LinearRgbaToOklaba],
        (ChannelLayout::OklabA, ChannelLayout::Rgba) => vec![ConvertStep::OklabaToLinearRgba],

        // Oklab ↔ RGB with alpha add/drop.
        (ChannelLayout::Rgb, ChannelLayout::OklabA) => {
            vec![ConvertStep::AddAlpha, ConvertStep::LinearRgbaToOklaba]
        }
        (ChannelLayout::OklabA, ChannelLayout::Rgb) => {
            vec![ConvertStep::OklabaToLinearRgba, ConvertStep::DropAlpha]
        }
        (ChannelLayout::Oklab, ChannelLayout::Rgba) => {
            vec![ConvertStep::OklabToLinearRgb, ConvertStep::AddAlpha]
        }
        (ChannelLayout::Rgba, ChannelLayout::Oklab) => {
            vec![ConvertStep::DropAlpha, ConvertStep::LinearRgbToOklab]
        }

        // Oklab ↔ BGRA (swizzle to/from RGBA, then Oklab).
        (ChannelLayout::Bgra, ChannelLayout::OklabA) => {
            vec![
                ConvertStep::SwizzleBgraRgba,
                ConvertStep::LinearRgbaToOklaba,
            ]
        }
        (ChannelLayout::OklabA, ChannelLayout::Bgra) => {
            vec![
                ConvertStep::OklabaToLinearRgba,
                ConvertStep::SwizzleBgraRgba,
            ]
        }
        (ChannelLayout::Bgra, ChannelLayout::Oklab) => {
            vec![
                ConvertStep::SwizzleBgraRgba,
                ConvertStep::DropAlpha,
                ConvertStep::LinearRgbToOklab,
            ]
        }
        (ChannelLayout::Oklab, ChannelLayout::Bgra) => {
            vec![
                ConvertStep::OklabToLinearRgb,
                ConvertStep::AddAlpha,
                ConvertStep::SwizzleBgraRgba,
            ]
        }

        // Gray ↔ Oklab (expand gray to RGB first).
        (ChannelLayout::Gray, ChannelLayout::Oklab) => {
            vec![ConvertStep::GrayToRgb, ConvertStep::LinearRgbToOklab]
        }
        (ChannelLayout::Oklab, ChannelLayout::Gray) => {
            vec![ConvertStep::OklabToLinearRgb, ConvertStep::RgbToGray]
        }
        (ChannelLayout::Gray, ChannelLayout::OklabA) => {
            vec![ConvertStep::GrayToRgba, ConvertStep::LinearRgbaToOklaba]
        }
        (ChannelLayout::OklabA, ChannelLayout::Gray) => {
            vec![ConvertStep::OklabaToLinearRgba, ConvertStep::RgbaToGray]
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::OklabA) => {
            vec![
                ConvertStep::GrayAlphaToRgba,
                ConvertStep::LinearRgbaToOklaba,
            ]
        }
        (ChannelLayout::OklabA, ChannelLayout::GrayAlpha) => {
            // Drop alpha from OklabA→Oklab, convert to RGB, then to GrayAlpha.
            // Alpha is lost; this is inherently lossy.
            vec![
                ConvertStep::OklabaToLinearRgba,
                ConvertStep::RgbaToGray,
                ConvertStep::GrayToGrayAlpha,
            ]
        }
        (ChannelLayout::GrayAlpha, ChannelLayout::Oklab) => {
            vec![ConvertStep::GrayAlphaToRgb, ConvertStep::LinearRgbToOklab]
        }
        (ChannelLayout::Oklab, ChannelLayout::GrayAlpha) => {
            vec![
                ConvertStep::OklabToLinearRgb,
                ConvertStep::RgbToGray,
                ConvertStep::GrayToGrayAlpha,
            ]
        }

        // Oklab ↔ alpha variants.
        (ChannelLayout::Oklab, ChannelLayout::OklabA) => vec![ConvertStep::AddAlpha],
        (ChannelLayout::OklabA, ChannelLayout::Oklab) => vec![ConvertStep::DropAlpha],

        _ => Vec::new(), // Unsupported layout conversion.
    }
}

/// Determine the depth conversion step(s), considering transfer functions.
///
/// Returns one or two steps. Two steps are needed when the conversion
/// requires going through an intermediate format (e.g. PQ U16 → sRGB U8
/// goes PQ U16 → Linear F32 → sRGB U8).
fn depth_steps(
    from: ChannelType,
    to: ChannelType,
    from_tf: TransferFunction,
    to_tf: TransferFunction,
) -> Result<Vec<ConvertStep>, ConvertError> {
    if from == to && from_tf == to_tf {
        return Ok(Vec::new());
    }

    // Same depth, different transfer function.
    // For integer types, TF changes are metadata-only (no math).
    // For F32, we can apply EOTF/OETF in place.
    if from == to && from != ChannelType::F32 {
        return Ok(Vec::new());
    }

    if from == to && from == ChannelType::F32 {
        return match (from_tf, to_tf) {
            (TransferFunction::Pq, TransferFunction::Linear) => {
                Ok(vec![ConvertStep::PqF32ToLinearF32])
            }
            (TransferFunction::Linear, TransferFunction::Pq) => {
                Ok(vec![ConvertStep::LinearF32ToPqF32])
            }
            (TransferFunction::Hlg, TransferFunction::Linear) => {
                Ok(vec![ConvertStep::HlgF32ToLinearF32])
            }
            (TransferFunction::Linear, TransferFunction::Hlg) => {
                Ok(vec![ConvertStep::LinearF32ToHlgF32])
            }
            // PQ ↔ HLG: go through linear.
            (TransferFunction::Pq, TransferFunction::Hlg) => Ok(vec![
                ConvertStep::PqF32ToLinearF32,
                ConvertStep::LinearF32ToHlgF32,
            ]),
            (TransferFunction::Hlg, TransferFunction::Pq) => Ok(vec![
                ConvertStep::HlgF32ToLinearF32,
                ConvertStep::LinearF32ToPqF32,
            ]),
            (TransferFunction::Srgb, TransferFunction::Linear) => {
                Ok(vec![ConvertStep::SrgbF32ToLinearF32])
            }
            (TransferFunction::Linear, TransferFunction::Srgb) => {
                Ok(vec![ConvertStep::LinearF32ToSrgbF32])
            }
            (TransferFunction::Bt709, TransferFunction::Linear) => {
                Ok(vec![ConvertStep::Bt709F32ToLinearF32])
            }
            (TransferFunction::Linear, TransferFunction::Bt709) => {
                Ok(vec![ConvertStep::LinearF32ToBt709F32])
            }
            // sRGB ↔ BT.709: go through linear.
            (TransferFunction::Srgb, TransferFunction::Bt709) => Ok(vec![
                ConvertStep::SrgbF32ToLinearF32,
                ConvertStep::LinearF32ToBt709F32,
            ]),
            (TransferFunction::Bt709, TransferFunction::Srgb) => Ok(vec![
                ConvertStep::Bt709F32ToLinearF32,
                ConvertStep::LinearF32ToSrgbF32,
            ]),
            // sRGB/BT.709 ↔ PQ/HLG: go through linear.
            (TransferFunction::Srgb, TransferFunction::Pq) => Ok(vec![
                ConvertStep::SrgbF32ToLinearF32,
                ConvertStep::LinearF32ToPqF32,
            ]),
            (TransferFunction::Srgb, TransferFunction::Hlg) => Ok(vec![
                ConvertStep::SrgbF32ToLinearF32,
                ConvertStep::LinearF32ToHlgF32,
            ]),
            (TransferFunction::Pq, TransferFunction::Srgb) => Ok(vec![
                ConvertStep::PqF32ToLinearF32,
                ConvertStep::LinearF32ToSrgbF32,
            ]),
            (TransferFunction::Hlg, TransferFunction::Srgb) => Ok(vec![
                ConvertStep::HlgF32ToLinearF32,
                ConvertStep::LinearF32ToSrgbF32,
            ]),
            (TransferFunction::Bt709, TransferFunction::Pq) => Ok(vec![
                ConvertStep::Bt709F32ToLinearF32,
                ConvertStep::LinearF32ToPqF32,
            ]),
            (TransferFunction::Bt709, TransferFunction::Hlg) => Ok(vec![
                ConvertStep::Bt709F32ToLinearF32,
                ConvertStep::LinearF32ToHlgF32,
            ]),
            (TransferFunction::Pq, TransferFunction::Bt709) => Ok(vec![
                ConvertStep::PqF32ToLinearF32,
                ConvertStep::LinearF32ToBt709F32,
            ]),
            (TransferFunction::Hlg, TransferFunction::Bt709) => Ok(vec![
                ConvertStep::HlgF32ToLinearF32,
                ConvertStep::LinearF32ToBt709F32,
            ]),
            _ => Ok(Vec::new()),
        };
    }

    match (from, to) {
        (ChannelType::U8, ChannelType::F32) => {
            if (from_tf == TransferFunction::Srgb || from_tf == TransferFunction::Bt709)
                && to_tf == TransferFunction::Linear
            {
                Ok(vec![ConvertStep::SrgbU8ToLinearF32])
            } else {
                Ok(vec![ConvertStep::NaiveU8ToF32])
            }
        }
        (ChannelType::F32, ChannelType::U8) => {
            if from_tf == TransferFunction::Linear
                && (to_tf == TransferFunction::Srgb || to_tf == TransferFunction::Bt709)
            {
                Ok(vec![ConvertStep::LinearF32ToSrgbU8])
            } else {
                Ok(vec![ConvertStep::NaiveF32ToU8])
            }
        }
        (ChannelType::U16, ChannelType::F32) => {
            // PQ/HLG U16 → Linear F32: apply EOTF during conversion.
            match (from_tf, to_tf) {
                (TransferFunction::Pq, TransferFunction::Linear) => {
                    Ok(vec![ConvertStep::PqU16ToLinearF32])
                }
                (TransferFunction::Hlg, TransferFunction::Linear) => {
                    Ok(vec![ConvertStep::HlgU16ToLinearF32])
                }
                _ => Ok(vec![ConvertStep::U16ToF32]),
            }
        }
        (ChannelType::F32, ChannelType::U16) => {
            // Linear F32 → PQ/HLG U16: apply OETF during conversion.
            match (from_tf, to_tf) {
                (TransferFunction::Linear, TransferFunction::Pq) => {
                    Ok(vec![ConvertStep::LinearF32ToPqU16])
                }
                (TransferFunction::Linear, TransferFunction::Hlg) => {
                    Ok(vec![ConvertStep::LinearF32ToHlgU16])
                }
                _ => Ok(vec![ConvertStep::F32ToU16]),
            }
        }
        (ChannelType::U16, ChannelType::U8) => {
            // HDR U16 → SDR U8: go through linear F32 with proper EOTF → OETF.
            if from_tf == TransferFunction::Pq && to_tf == TransferFunction::Srgb {
                Ok(vec![
                    ConvertStep::PqU16ToLinearF32,
                    ConvertStep::LinearF32ToSrgbU8,
                ])
            } else if from_tf == TransferFunction::Hlg && to_tf == TransferFunction::Srgb {
                Ok(vec![
                    ConvertStep::HlgU16ToLinearF32,
                    ConvertStep::LinearF32ToSrgbU8,
                ])
            } else {
                Ok(vec![ConvertStep::U16ToU8])
            }
        }
        (ChannelType::U8, ChannelType::U16) => Ok(vec![ConvertStep::U8ToU16]),
        _ => Err(ConvertError::NoPath {
            from: PixelDescriptor::new(from, ChannelLayout::Rgb, None, from_tf),
            to: PixelDescriptor::new(to, ChannelLayout::Rgb, None, to_tf),
        }),
    }
}

// ---------------------------------------------------------------------------
// Row conversion kernels
// ---------------------------------------------------------------------------

/// Pre-allocated scratch buffer for multi-step row conversions.
///
/// Eliminates per-row heap allocation by reusing two ping-pong halves
/// of a single buffer across calls. Create once per [`ConvertPlan`],
/// then pass to `convert_row_buffered` for each row.
pub(crate) struct ConvertScratch {
    /// Single allocation split into two halves via `split_at_mut`.
    /// Stored as `Vec<u32>` to guarantee 4-byte alignment, which lets
    /// garb and bytemuck use fast aligned paths instead of unaligned fallbacks.
    buf: Vec<u32>,
}

impl ConvertScratch {
    /// Create empty scratch (buffer grows on first use).
    pub(crate) fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Ensure the buffer is large enough for two halves of the max
    /// intermediate format at the given width.
    fn ensure_capacity(&mut self, plan: &ConvertPlan, width: u32) {
        let half_bytes = (width as usize) * plan.max_intermediate_bpp();
        let total_u32 = (half_bytes * 2).div_ceil(4);
        if self.buf.len() < total_u32 {
            self.buf.resize(total_u32, 0);
        }
    }
}

impl core::fmt::Debug for ConvertScratch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ConvertScratch")
            .field("capacity", &self.buf.capacity())
            .finish()
    }
}

/// Convert one row of `width` pixels using a pre-computed plan.
///
/// `src` and `dst` must be sized for `width` pixels in their respective formats.
/// For multi-step plans, an internal scratch buffer is allocated per call.
/// Prefer [`RowConverter`](crate::RowConverter) in hot loops (reuses scratch buffers).
pub fn convert_row(plan: &ConvertPlan, src: &[u8], dst: &mut [u8], width: u32) {
    if plan.is_identity() {
        let len = min(src.len(), dst.len());
        dst[..len].copy_from_slice(&src[..len]);
        return;
    }

    if plan.steps.len() == 1 {
        apply_step_u8(plan.steps[0], src, dst, width, plan.from, plan.to);
        return;
    }

    // Allocating fallback for one-off calls.
    let mut scratch = ConvertScratch::new();
    convert_row_buffered(plan, src, dst, width, &mut scratch);
}

/// Convert one row of `width` pixels, reusing pre-allocated scratch buffers.
///
/// For multi-step plans this avoids per-row heap allocation by ping-ponging
/// between two halves of a scratch buffer. Single-step plans bypass scratch.
pub(crate) fn convert_row_buffered(
    plan: &ConvertPlan,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    scratch: &mut ConvertScratch,
) {
    if plan.is_identity() {
        let len = min(src.len(), dst.len());
        dst[..len].copy_from_slice(&src[..len]);
        return;
    }

    if plan.steps.len() == 1 {
        apply_step_u8(plan.steps[0], src, dst, width, plan.from, plan.to);
        return;
    }

    scratch.ensure_capacity(plan, width);

    let buf_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut scratch.buf);
    let half = buf_bytes.len() / 2;
    let (buf_a, buf_b) = buf_bytes.split_at_mut(half);

    let num_steps = plan.steps.len();
    let mut current_desc = plan.from;

    for (i, &step) in plan.steps.iter().enumerate() {
        let is_last = i == num_steps - 1;
        let next_desc = if is_last {
            plan.to
        } else {
            intermediate_desc(current_desc, step)
        };

        let next_len = (width as usize) * next_desc.bytes_per_pixel();
        let curr_len = (width as usize) * current_desc.bytes_per_pixel();

        // Ping-pong: even steps read src/buf_b and write buf_a;
        // odd steps read buf_a and write buf_b. Each branch only
        // borrows each half in one mode, satisfying the borrow checker.
        if i % 2 == 0 {
            let input = if i == 0 { src } else { &buf_b[..curr_len] };
            if is_last {
                apply_step_u8(step, input, dst, width, current_desc, next_desc);
            } else {
                apply_step_u8(
                    step,
                    input,
                    &mut buf_a[..next_len],
                    width,
                    current_desc,
                    next_desc,
                );
            }
        } else {
            let input = &buf_a[..curr_len];
            if is_last {
                apply_step_u8(step, input, dst, width, current_desc, next_desc);
            } else {
                apply_step_u8(
                    step,
                    input,
                    &mut buf_b[..next_len],
                    width,
                    current_desc,
                    next_desc,
                );
            }
        }

        current_desc = next_desc;
    }
}

/// Check if two steps are inverses that cancel each other.
fn are_inverse(a: ConvertStep, b: ConvertStep) -> bool {
    matches!(
        (a, b),
        // Self-inverse
        (ConvertStep::SwizzleBgraRgba, ConvertStep::SwizzleBgraRgba)
        // Layout inverses (lossless for opaque data)
        | (ConvertStep::AddAlpha, ConvertStep::DropAlpha)
        // Transfer function f32↔f32 (exact inverses in float)
        | (ConvertStep::SrgbF32ToLinearF32, ConvertStep::LinearF32ToSrgbF32)
        | (ConvertStep::LinearF32ToSrgbF32, ConvertStep::SrgbF32ToLinearF32)
        | (ConvertStep::PqF32ToLinearF32, ConvertStep::LinearF32ToPqF32)
        | (ConvertStep::LinearF32ToPqF32, ConvertStep::PqF32ToLinearF32)
        | (ConvertStep::HlgF32ToLinearF32, ConvertStep::LinearF32ToHlgF32)
        | (ConvertStep::LinearF32ToHlgF32, ConvertStep::HlgF32ToLinearF32)
        | (ConvertStep::Bt709F32ToLinearF32, ConvertStep::LinearF32ToBt709F32)
        | (ConvertStep::LinearF32ToBt709F32, ConvertStep::Bt709F32ToLinearF32)
        // Alpha mode (exact inverses in float)
        | (ConvertStep::StraightToPremul, ConvertStep::PremulToStraight)
        | (ConvertStep::PremulToStraight, ConvertStep::StraightToPremul)
        // Color model (exact inverses in float)
        | (ConvertStep::LinearRgbToOklab, ConvertStep::OklabToLinearRgb)
        | (ConvertStep::OklabToLinearRgb, ConvertStep::LinearRgbToOklab)
        | (ConvertStep::LinearRgbaToOklaba, ConvertStep::OklabaToLinearRgba)
        | (ConvertStep::OklabaToLinearRgba, ConvertStep::LinearRgbaToOklaba)
        // Cross-depth pairs (near-lossless for same depth class)
        | (ConvertStep::NaiveU8ToF32, ConvertStep::NaiveF32ToU8)
        | (ConvertStep::NaiveF32ToU8, ConvertStep::NaiveU8ToF32)
        | (ConvertStep::U8ToU16, ConvertStep::U16ToU8)
        | (ConvertStep::U16ToU8, ConvertStep::U8ToU16)
        | (ConvertStep::U16ToF32, ConvertStep::F32ToU16)
        | (ConvertStep::F32ToU16, ConvertStep::U16ToF32)
        // Cross-depth with transfer (near-lossless roundtrip)
        | (ConvertStep::SrgbU8ToLinearF32, ConvertStep::LinearF32ToSrgbU8)
        | (ConvertStep::LinearF32ToSrgbU8, ConvertStep::SrgbU8ToLinearF32)
        | (ConvertStep::PqU16ToLinearF32, ConvertStep::LinearF32ToPqU16)
        | (ConvertStep::LinearF32ToPqU16, ConvertStep::PqU16ToLinearF32)
        | (ConvertStep::HlgU16ToLinearF32, ConvertStep::LinearF32ToHlgU16)
        | (ConvertStep::LinearF32ToHlgU16, ConvertStep::HlgU16ToLinearF32)
    )
}

/// Compute the descriptor after applying one step.
fn intermediate_desc(current: PixelDescriptor, step: ConvertStep) -> PixelDescriptor {
    match step {
        ConvertStep::Identity => current,
        ConvertStep::SwizzleBgraRgba => {
            let new_layout = match current.layout() {
                ChannelLayout::Bgra => ChannelLayout::Rgba,
                ChannelLayout::Rgba => ChannelLayout::Bgra,
                other => other,
            };
            PixelDescriptor::new(
                current.channel_type(),
                new_layout,
                current.alpha(),
                current.transfer(),
            )
        }
        ConvertStep::AddAlpha => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            current.transfer(),
        ),
        ConvertStep::DropAlpha | ConvertStep::MatteComposite { .. } => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgb,
            None,
            current.transfer(),
        ),
        ConvertStep::GrayToRgb => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgb,
            None,
            current.transfer(),
        ),
        ConvertStep::GrayToRgba => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            current.transfer(),
        ),
        ConvertStep::RgbToGray | ConvertStep::RgbaToGray => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Gray,
            None,
            current.transfer(),
        ),
        ConvertStep::GrayAlphaToRgba => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgba,
            current.alpha(),
            current.transfer(),
        ),
        ConvertStep::GrayAlphaToRgb => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Rgb,
            None,
            current.transfer(),
        ),
        ConvertStep::GrayToGrayAlpha => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            current.transfer(),
        ),
        ConvertStep::GrayAlphaToGray => PixelDescriptor::new(
            current.channel_type(),
            ChannelLayout::Gray,
            None,
            current.transfer(),
        ),
        ConvertStep::SrgbU8ToLinearF32
        | ConvertStep::NaiveU8ToF32
        | ConvertStep::U16ToF32
        | ConvertStep::PqU16ToLinearF32
        | ConvertStep::HlgU16ToLinearF32
        | ConvertStep::PqF32ToLinearF32
        | ConvertStep::HlgF32ToLinearF32
        | ConvertStep::SrgbF32ToLinearF32
        | ConvertStep::Bt709F32ToLinearF32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Linear,
        ),
        ConvertStep::LinearF32ToSrgbU8 | ConvertStep::NaiveF32ToU8 | ConvertStep::U16ToU8 => {
            PixelDescriptor::new(
                ChannelType::U8,
                current.layout(),
                current.alpha(),
                TransferFunction::Srgb,
            )
        }
        ConvertStep::U8ToU16 => PixelDescriptor::new(
            ChannelType::U16,
            current.layout(),
            current.alpha(),
            current.transfer(),
        ),
        ConvertStep::F32ToU16 | ConvertStep::LinearF32ToPqU16 | ConvertStep::LinearF32ToHlgU16 => {
            let tf = match step {
                ConvertStep::LinearF32ToPqU16 => TransferFunction::Pq,
                ConvertStep::LinearF32ToHlgU16 => TransferFunction::Hlg,
                _ => current.transfer(),
            };
            PixelDescriptor::new(ChannelType::U16, current.layout(), current.alpha(), tf)
        }
        ConvertStep::LinearF32ToPqF32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Pq,
        ),
        ConvertStep::LinearF32ToHlgF32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Hlg,
        ),
        ConvertStep::LinearF32ToSrgbF32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Srgb,
        ),
        ConvertStep::LinearF32ToBt709F32 => PixelDescriptor::new(
            ChannelType::F32,
            current.layout(),
            current.alpha(),
            TransferFunction::Bt709,
        ),
        ConvertStep::StraightToPremul => PixelDescriptor::new(
            current.channel_type(),
            current.layout(),
            Some(AlphaMode::Premultiplied),
            current.transfer(),
        ),
        ConvertStep::PremulToStraight => PixelDescriptor::new(
            current.channel_type(),
            current.layout(),
            Some(AlphaMode::Straight),
            current.transfer(),
        ),
        ConvertStep::LinearRgbToOklab => PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Oklab,
            None,
            TransferFunction::Unknown,
        )
        .with_primaries(current.primaries),
        ConvertStep::OklabToLinearRgb => PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        )
        .with_primaries(current.primaries),
        ConvertStep::LinearRgbaToOklaba => PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::OklabA,
            Some(AlphaMode::Straight),
            TransferFunction::Unknown,
        )
        .with_primaries(current.primaries),
        ConvertStep::OklabaToLinearRgba => PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            current.alpha(),
            TransferFunction::Linear,
        )
        .with_primaries(current.primaries),
    }
}

#[path = "convert_kernels.rs"]
mod convert_kernels;
use convert_kernels::apply_step_u8;
pub(crate) use convert_kernels::{hlg_eotf, hlg_oetf, pq_eotf, pq_oetf};
