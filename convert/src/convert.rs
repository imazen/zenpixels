//! Row-level pixel conversion kernels.
//!
//! Each kernel converts one row of `width` pixels from a source format to
//! a destination format. Kernels are pure functions with no allocation.

use alloc::vec::Vec;
use core::cmp::min;

use crate::policy::{AlphaPolicy, ConvertOptions, DepthPolicy};
use crate::{
    AlphaMode, ChannelLayout, ChannelType, ConvertError, PixelDescriptor, TransferFunction,
};

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
    /// Straight → Premultiplied alpha.
    StraightToPremul,
    /// Premultiplied → Straight alpha.
    PremulToStraight,
}

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

// ---------------------------------------------------------------------------
// Row conversion kernels
// ---------------------------------------------------------------------------

/// Convert one row of `width` pixels using a pre-computed plan.
///
/// `src` and `dst` must be sized for `width` pixels in their respective formats.
/// For multi-step plans, an internal scratch buffer is used.
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

    // Multi-step: use intermediate buffer.
    // Calculate intermediate format after first step.
    let mut current = Vec::from(src);
    let mut current_desc = plan.from;

    for (i, &step) in plan.steps.iter().enumerate() {
        let is_last = i == plan.steps.len() - 1;
        let next_desc = if is_last {
            plan.to
        } else {
            intermediate_desc(current_desc, step)
        };

        let next_bpp = next_desc.bytes_per_pixel();
        let next_len = (width as usize) * next_bpp;

        if is_last {
            apply_step_u8(step, &current, dst, width, current_desc, next_desc);
        } else {
            let mut next = vec![0u8; next_len];
            apply_step_u8(step, &current, &mut next, width, current_desc, next_desc);
            current = next;
            current_desc = next_desc;
        }
    }
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
        ConvertStep::DropAlpha => PixelDescriptor::new(
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
        | ConvertStep::HlgF32ToLinearF32 => PixelDescriptor::new(
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
    }
}

/// Apply a single conversion step on raw byte slices.
fn apply_step_u8(
    step: ConvertStep,
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    from: PixelDescriptor,
    _to: PixelDescriptor,
) {
    let w = width as usize;

    match step {
        ConvertStep::Identity => {
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }

        ConvertStep::SwizzleBgraRgba => {
            swizzle_bgra_rgba(src, dst, w, from.channel_type());
        }

        ConvertStep::AddAlpha => {
            add_alpha(src, dst, w, from.channel_type());
        }

        ConvertStep::DropAlpha => {
            drop_alpha(src, dst, w, from.channel_type());
        }

        ConvertStep::GrayToRgb => {
            gray_to_rgb(src, dst, w, from.channel_type());
        }

        ConvertStep::GrayToRgba => {
            gray_to_rgba(src, dst, w, from.channel_type());
        }

        ConvertStep::RgbToGray => {
            rgb_to_gray_u8(src, dst, w);
        }

        ConvertStep::RgbaToGray => {
            rgba_to_gray_u8(src, dst, w);
        }

        ConvertStep::GrayAlphaToRgba => {
            gray_alpha_to_rgba(src, dst, w, from.channel_type());
        }

        ConvertStep::GrayAlphaToRgb => {
            gray_alpha_to_rgb(src, dst, w, from.channel_type());
        }

        ConvertStep::GrayToGrayAlpha => {
            gray_to_gray_alpha(src, dst, w, from.channel_type());
        }

        ConvertStep::GrayAlphaToGray => {
            gray_alpha_to_gray(src, dst, w, from.channel_type());
        }

        ConvertStep::SrgbU8ToLinearF32 => {
            srgb_u8_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToSrgbU8 => {
            linear_f32_to_srgb_u8(src, dst, w, from.layout().channels());
        }

        ConvertStep::NaiveU8ToF32 => {
            naive_u8_to_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::NaiveF32ToU8 => {
            naive_f32_to_u8(src, dst, w, from.layout().channels());
        }

        ConvertStep::U16ToU8 => {
            u16_to_u8(src, dst, w, from.layout().channels());
        }

        ConvertStep::U8ToU16 => {
            u8_to_u16(src, dst, w, from.layout().channels());
        }

        ConvertStep::U16ToF32 => {
            u16_to_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::F32ToU16 => {
            f32_to_u16(src, dst, w, from.layout().channels());
        }

        ConvertStep::PqU16ToLinearF32 => {
            pq_u16_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToPqU16 => {
            linear_f32_to_pq_u16(src, dst, w, from.layout().channels());
        }

        ConvertStep::PqF32ToLinearF32 => {
            pq_f32_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToPqF32 => {
            linear_f32_to_pq_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::HlgU16ToLinearF32 => {
            hlg_u16_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToHlgU16 => {
            linear_f32_to_hlg_u16(src, dst, w, from.layout().channels());
        }

        ConvertStep::HlgF32ToLinearF32 => {
            hlg_f32_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToHlgF32 => {
            linear_f32_to_hlg_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::StraightToPremul => {
            straight_to_premul(src, dst, w, from.channel_type(), from.layout());
        }

        ConvertStep::PremulToStraight => {
            premul_to_straight(src, dst, w, from.channel_type(), from.layout());
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel implementations
// ---------------------------------------------------------------------------

/// BGRA ↔ RGBA swizzle.
fn swizzle_bgra_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    let bps = ch_type.byte_size(); // bytes per sample
    let pixel_bytes = 4 * bps;

    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let s = i * pixel_bytes;
                let d = i * pixel_bytes;
                dst[d] = src[s + 2]; // R ← B (or B ← R)
                dst[d + 1] = src[s + 1]; // G ← G
                dst[d + 2] = src[s]; // B ← R (or R ← B)
                dst[d + 3] = src[s + 3]; // A ← A
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * pixel_bytes]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * pixel_bytes]);
            for i in 0..width {
                let s = i * 4;
                dst16[s] = src16[s + 2];
                dst16[s + 1] = src16[s + 1];
                dst16[s + 2] = src16[s];
                dst16[s + 3] = src16[s + 3];
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * pixel_bytes]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * pixel_bytes]);
            for i in 0..width {
                let s = i * 4;
                dstf[s] = srcf[s + 2];
                dstf[s + 1] = srcf[s + 1];
                dstf[s + 2] = srcf[s];
                dstf[s + 3] = srcf[s + 3];
            }
        }
        _ => {}
    }
}

/// Add opaque alpha channel (3ch → 4ch).
fn add_alpha(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                dst[i * 4] = src[i * 3];
                dst[i * 4 + 1] = src[i * 3 + 1];
                dst[i * 4 + 2] = src[i * 3 + 2];
                dst[i * 4 + 3] = 255;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 6]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 8]);
            for i in 0..width {
                dst16[i * 4] = src16[i * 3];
                dst16[i * 4 + 1] = src16[i * 3 + 1];
                dst16[i * 4 + 2] = src16[i * 3 + 2];
                dst16[i * 4 + 3] = 65535;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
            for i in 0..width {
                dstf[i * 4] = srcf[i * 3];
                dstf[i * 4 + 1] = srcf[i * 3 + 1];
                dstf[i * 4 + 2] = srcf[i * 3 + 2];
                dstf[i * 4 + 3] = 1.0;
            }
        }
        _ => {}
    }
}

/// Drop alpha channel (4ch → 3ch).
fn drop_alpha(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                dst[i * 3] = src[i * 4];
                dst[i * 3 + 1] = src[i * 4 + 1];
                dst[i * 3 + 2] = src[i * 4 + 2];
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 8]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            for i in 0..width {
                dst16[i * 3] = src16[i * 4];
                dst16[i * 3 + 1] = src16[i * 4 + 1];
                dst16[i * 3 + 2] = src16[i * 4 + 2];
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
            for i in 0..width {
                dstf[i * 3] = srcf[i * 4];
                dstf[i * 3 + 1] = srcf[i * 4 + 1];
                dstf[i * 3 + 2] = srcf[i * 4 + 2];
            }
        }
        _ => {}
    }
}

/// Gray → RGB (replicate).
fn gray_to_rgb(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let g = src[i];
                dst[i * 3] = g;
                dst[i * 3 + 1] = g;
                dst[i * 3 + 2] = g;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 2]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            for i in 0..width {
                let g = src16[i];
                dst16[i * 3] = g;
                dst16[i * 3 + 1] = g;
                dst16[i * 3 + 2] = g;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
            for i in 0..width {
                let g = srcf[i];
                dstf[i * 3] = g;
                dstf[i * 3 + 1] = g;
                dstf[i * 3 + 2] = g;
            }
        }
        _ => {}
    }
}

/// Gray → RGBA (replicate + opaque alpha).
fn gray_to_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let g = src[i];
                dst[i * 4] = g;
                dst[i * 4 + 1] = g;
                dst[i * 4 + 2] = g;
                dst[i * 4 + 3] = 255;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 2]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 8]);
            for i in 0..width {
                let g = src16[i];
                dst16[i * 4] = g;
                dst16[i * 4 + 1] = g;
                dst16[i * 4 + 2] = g;
                dst16[i * 4 + 3] = 65535;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
            for i in 0..width {
                let g = srcf[i];
                dstf[i * 4] = g;
                dstf[i * 4 + 1] = g;
                dstf[i * 4 + 2] = g;
                dstf[i * 4 + 3] = 1.0;
            }
        }
        _ => {}
    }
}

/// RGB → Gray using BT.709 luma coefficients (u8 only).
fn rgb_to_gray_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for i in 0..width {
        let r = src[i * 3] as u32;
        let g = src[i * 3 + 1] as u32;
        let b = src[i * 3 + 2] as u32;
        // BT.709: 0.2126R + 0.7152G + 0.0722B
        // Fixed-point: (54R + 183G + 19B + 128) >> 8
        dst[i] = ((54 * r + 183 * g + 19 * b + 128) >> 8) as u8;
    }
}

/// RGBA → Gray using BT.709 luma, drop alpha (u8 only).
fn rgba_to_gray_u8(src: &[u8], dst: &mut [u8], width: usize) {
    for i in 0..width {
        let r = src[i * 4] as u32;
        let g = src[i * 4 + 1] as u32;
        let b = src[i * 4 + 2] as u32;
        dst[i] = ((54 * r + 183 * g + 19 * b + 128) >> 8) as u8;
    }
}

/// GrayAlpha → RGBA (replicate gray, preserve alpha).
fn gray_alpha_to_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let g = src[i * 2];
                let a = src[i * 2 + 1];
                dst[i * 4] = g;
                dst[i * 4 + 1] = g;
                dst[i * 4 + 2] = g;
                dst[i * 4 + 3] = a;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 4]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 8]);
            for i in 0..width {
                let g = src16[i * 2];
                let a = src16[i * 2 + 1];
                dst16[i * 4] = g;
                dst16[i * 4 + 1] = g;
                dst16[i * 4 + 2] = g;
                dst16[i * 4 + 3] = a;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 8]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
            for i in 0..width {
                let g = srcf[i * 2];
                let a = srcf[i * 2 + 1];
                dstf[i * 4] = g;
                dstf[i * 4 + 1] = g;
                dstf[i * 4 + 2] = g;
                dstf[i * 4 + 3] = a;
            }
        }
        _ => {}
    }
}

/// GrayAlpha → RGB (replicate gray, drop alpha).
fn gray_alpha_to_rgb(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let g = src[i * 2];
                dst[i * 3] = g;
                dst[i * 3 + 1] = g;
                dst[i * 3 + 2] = g;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 4]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            for i in 0..width {
                let g = src16[i * 2];
                dst16[i * 3] = g;
                dst16[i * 3 + 1] = g;
                dst16[i * 3 + 2] = g;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 8]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
            for i in 0..width {
                let g = srcf[i * 2];
                dstf[i * 3] = g;
                dstf[i * 3 + 1] = g;
                dstf[i * 3 + 2] = g;
            }
        }
        _ => {}
    }
}

/// Gray → GrayAlpha (add opaque alpha).
fn gray_to_gray_alpha(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                dst[i * 2] = src[i];
                dst[i * 2 + 1] = 255;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 2]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 4]);
            for i in 0..width {
                dst16[i * 2] = src16[i];
                dst16[i * 2 + 1] = 65535;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 8]);
            for i in 0..width {
                dstf[i * 2] = srcf[i];
                dstf[i * 2 + 1] = 1.0;
            }
        }
        _ => {}
    }
}

/// GrayAlpha → Gray (drop alpha).
fn gray_alpha_to_gray(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                dst[i] = src[i * 2];
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 4]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 2]);
            for i in 0..width {
                dst16[i] = src16[i * 2];
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 8]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 4]);
            for i in 0..width {
                dstf[i] = srcf[i * 2];
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Depth conversion kernels (transfer-function-aware)
// ---------------------------------------------------------------------------

/// sRGB u8 → linear f32 using `linear-srgb` SIMD batch conversion.
fn srgb_u8_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    linear_srgb::simd::srgb_u8_to_linear_slice(&src[..count], dstf);
}

/// Linear f32 → sRGB u8 using `linear-srgb` SIMD batch conversion.
fn linear_f32_to_srgb_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    linear_srgb::simd::linear_to_srgb_u8_slice(srcf, &mut dst[..count]);
}

/// Naive u8 → f32 (v / 255.0, no transfer function).
fn naive_u8_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = src[i] as f32 / 255.0;
    }
}

/// Naive f32 → u8 (clamp [0,1], * 255 + 0.5).
fn naive_f32_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    for i in 0..count {
        dst[i] = (srcf[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

/// u16 → u8: (v * 255 + 32768) >> 16.
fn u16_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    for i in 0..count {
        dst[i] = ((src16[i] as u32 * 255 + 32768) >> 16) as u8;
    }
}

/// u8 → u16: v * 257.
fn u8_to_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    for i in 0..count {
        dst16[i] = src[i] as u16 * 257;
    }
}

/// u16 → f32: v / 65535.0.
fn u16_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = src16[i] as f32 / 65535.0;
    }
}

/// f32 → u16: clamp [0,1], * 65535 + 0.5.
fn f32_to_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    for i in 0..count {
        dst16[i] = (srcf[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

// ---------------------------------------------------------------------------
// PQ (SMPTE ST 2084) transfer function
// ---------------------------------------------------------------------------

// PQ constants (SMPTE ST 2084 / BT.2100).
const PQ_M1: f64 = 2610.0 / 16384.0; // 0.1593017578125
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0; // 78.84375
const PQ_C1: f64 = 3424.0 / 4096.0; // 0.8359375
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0; // 18.8515625
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0; // 18.6875

/// PQ EOTF: encoded [0,1] → linear light [0,1] (where 1.0 = 10000 cd/m²).
#[inline]
pub(crate) fn pq_eotf(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let v = v as f64;
    let vp = v.powf(1.0 / PQ_M2);
    let num = (vp - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * vp;
    if den <= 0.0 {
        return 0.0;
    }
    (num / den).powf(1.0 / PQ_M1) as f32
}

/// PQ inverse EOTF (OETF): linear light [0,1] → encoded [0,1].
#[inline]
pub(crate) fn pq_oetf(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let v = v as f64;
    let ym1 = v.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * ym1;
    let den = 1.0 + PQ_C3 * ym1;
    (num / den).powf(PQ_M2) as f32
}

/// PQ U16 → Linear F32 (EOTF applied during depth conversion).
fn pq_u16_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        let normalized = src16[i] as f32 / 65535.0;
        dstf[i] = pq_eotf(normalized);
    }
}

/// Linear F32 → PQ U16 (OETF applied during depth conversion).
fn linear_f32_to_pq_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    for i in 0..count {
        let encoded = pq_oetf(srcf[i].max(0.0));
        dst16[i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

/// PQ F32 → Linear F32 (EOTF, same depth).
fn pq_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = pq_eotf(srcf[i]);
    }
}

/// Linear F32 → PQ F32 (OETF, same depth).
fn linear_f32_to_pq_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = pq_oetf(srcf[i].max(0.0));
    }
}

// ---------------------------------------------------------------------------
// HLG (ARIB STD-B67) transfer function
// ---------------------------------------------------------------------------

// HLG constants (ARIB STD-B67 / BT.2100, ITU-R BT.2100-2).
const HLG_A: f64 = 0.17883277;
const HLG_B: f64 = 0.28466892; // 1 - 4*a
const HLG_C: f64 = 0.55991073; // 0.5 - a*ln(4*a)

/// HLG OETF: scene-linear [0,1] → encoded [0,1].
#[inline]
pub(crate) fn hlg_oetf(v: f32) -> f32 {
    let v = v.max(0.0) as f64;
    if v <= 1.0 / 12.0 {
        (3.0 * v).sqrt() as f32
    } else {
        (HLG_A * (12.0 * v - HLG_B).ln() + HLG_C) as f32
    }
}

/// HLG inverse OETF (EOTF): encoded [0,1] → scene-linear [0,1].
#[inline]
pub(crate) fn hlg_eotf(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let v = v as f64;
    if v <= 0.5 {
        (v * v / 3.0) as f32
    } else {
        (((v - HLG_C) / HLG_A).exp() + HLG_B) as f32 / 12.0
    }
}

/// HLG U16 → Linear F32 (EOTF applied during depth conversion).
fn hlg_u16_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let src16: &[u16] = bytemuck::cast_slice(&src[..count * 2]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        let normalized = src16[i] as f32 / 65535.0;
        dstf[i] = hlg_eotf(normalized);
    }
}

/// Linear F32 → HLG U16 (OETF applied during depth conversion).
fn linear_f32_to_hlg_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    for i in 0..count {
        let encoded = hlg_oetf(srcf[i]);
        dst16[i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

/// HLG F32 → Linear F32 (EOTF, same depth).
fn hlg_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = hlg_eotf(srcf[i]);
    }
}

/// Linear F32 → HLG F32 (OETF, same depth).
fn linear_f32_to_hlg_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = hlg_oetf(srcf[i]);
    }
}

// ---------------------------------------------------------------------------
// Alpha premultiplication
// ---------------------------------------------------------------------------

/// Straight → Premultiplied alpha (in-place copy from src to dst).
fn straight_to_premul(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    layout: ChannelLayout,
) {
    let channels = layout.channels();
    let alpha_idx = channels - 1;

    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let base = i * channels;
                let a = src[base + alpha_idx] as u32;
                for c in 0..alpha_idx {
                    dst[base + c] = ((src[base + c] as u32 * a + 128) / 255) as u8;
                }
                dst[base + alpha_idx] = src[base + alpha_idx];
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * channels * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * channels * 4]);
            for i in 0..width {
                let base = i * channels;
                let a = srcf[base + alpha_idx];
                for c in 0..alpha_idx {
                    dstf[base + c] = srcf[base + c] * a;
                }
                dstf[base + alpha_idx] = a;
            }
        }
        _ => {
            // Fallback: copy.
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
    }
}

/// Premultiplied → Straight alpha.
fn premul_to_straight(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    layout: ChannelLayout,
) {
    let channels = layout.channels();
    let alpha_idx = channels - 1;

    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let base = i * channels;
                let a = src[base + alpha_idx];
                if a == 0 {
                    for c in 0..channels {
                        dst[base + c] = 0;
                    }
                } else {
                    let a32 = a as u32;
                    for c in 0..alpha_idx {
                        dst[base + c] = ((src[base + c] as u32 * 255 + a32 / 2) / a32) as u8;
                    }
                    dst[base + alpha_idx] = a;
                }
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * channels * 4]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * channels * 4]);
            for i in 0..width {
                let base = i * channels;
                let a = srcf[base + alpha_idx];
                if a == 0.0 {
                    for c in 0..channels {
                        dstf[base + c] = 0.0;
                    }
                } else {
                    let inv_a = 1.0 / a;
                    for c in 0..alpha_idx {
                        dstf[base + c] = srcf[base + c] * inv_a;
                    }
                    dstf[base + alpha_idx] = a;
                }
            }
        }
        _ => {
            let len = min(src.len(), dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
    }
}
