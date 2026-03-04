//! Conversion plan construction.

use alloc::vec;
use alloc::vec::Vec;

use crate::policy::{AlphaPolicy, ConvertOptions, DepthPolicy};
use crate::{
    AlphaMode, ChannelLayout, ChannelType, ConvertError, PixelDescriptor, TransferFunction,
};

use super::{ConvertPlan, ConvertStep};

impl ConvertPlan {
    /// Create a conversion plan from `from` to `to`.
    ///
    /// Returns `Err` if no conversion path exists.
    pub fn new(from: PixelDescriptor, to: PixelDescriptor) -> Result<Self, ConvertError> {
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
            let src_ch = from.layout().channels();
            let dst_ch = to.layout().channels();

            if need_depth_or_tf && dst_ch > src_ch {
                // Depth first, then layout.
                steps.extend(depth_steps(
                    from.channel_type(),
                    to.channel_type(),
                    from.transfer(),
                    to.transfer(),
                )?);
                steps.extend(layout_steps(from.layout(), to.layout()));
            } else {
                // Layout first, then depth.
                steps.extend(layout_steps(from.layout(), to.layout()));
                if need_depth_or_tf {
                    steps.extend(depth_steps(
                        from.channel_type(),
                        to.channel_type(),
                        from.transfer(),
                        to.transfer(),
                    )?);
                }
            }
        } else if need_depth_or_tf {
            steps.extend(depth_steps(
                from.channel_type(),
                to.channel_type(),
                from.transfer(),
                to.transfer(),
            )?);
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
    pub fn new_explicit(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &ConvertOptions,
    ) -> Result<Self, ConvertError> {
        // Check alpha removal policy.
        let drops_alpha = from.alpha().is_some() && to.alpha().is_none();
        if drops_alpha && options.alpha_policy == AlphaPolicy::Forbid {
            return Err(ConvertError::AlphaRemovalForbidden);
        }

        // Check depth reduction policy.
        let reduces_depth = from.channel_type().byte_size() > to.channel_type().byte_size();
        if reduces_depth && options.depth_policy == DepthPolicy::Forbid {
            return Err(ConvertError::DepthReductionForbidden);
        }

        // Check RGB→Gray requires luma coefficients.
        let src_is_rgb = matches!(
            from.layout(),
            ChannelLayout::Rgb | ChannelLayout::Rgba | ChannelLayout::Bgra
        );
        let dst_is_gray = matches!(to.layout(), ChannelLayout::Gray | ChannelLayout::GrayAlpha);
        if src_is_rgb && dst_is_gray && options.luma.is_none() {
            return Err(ConvertError::RgbToGray);
        }

        Self::new(from, to)
    }

    /// True if conversion is a no-op.
    pub fn is_identity(&self) -> bool {
        self.steps.len() == 1 && self.steps[0] == ConvertStep::Identity
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
            // sRGB ↔ Linear are already handled.
            (TransferFunction::Srgb | TransferFunction::Bt709, TransferFunction::Linear)
            | (TransferFunction::Linear, TransferFunction::Srgb | TransferFunction::Bt709) => {
                // F32 sRGB ↔ Linear: no dedicated kernel yet, treat as identity.
                // (The sRGB kernels only handle U8 ↔ F32 transitions.)
                Ok(Vec::new())
            }
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
