//! Row-level pixel conversion kernels.
//!
//! Each kernel converts one row of `width` pixels from a source format to
//! a destination format. Kernels are pure functions with no allocation.

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
        if drops_alpha {
            if let AlphaPolicy::CompositeOnto { r, g, b } = options.alpha_policy {
                for step in &mut plan.steps {
                    if matches!(step, ConvertStep::DropAlpha) {
                        *step = ConvertStep::MatteComposite { r, g, b };
                    }
                }
            }
        }

        Ok(plan)
    }

    /// True if conversion is a no-op.
    #[must_use]
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

        ConvertStep::MatteComposite { r, g, b } => {
            matte_composite(src, dst, w, from.channel_type(), r, g, b);
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

        ConvertStep::SrgbF32ToLinearF32 => {
            srgb_f32_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToSrgbF32 => {
            linear_f32_to_srgb_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::Bt709F32ToLinearF32 => {
            bt709_f32_to_linear_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::LinearF32ToBt709F32 => {
            linear_f32_to_bt709_f32(src, dst, w, from.layout().channels());
        }

        ConvertStep::StraightToPremul => {
            straight_to_premul(src, dst, w, from.channel_type(), from.layout());
        }

        ConvertStep::PremulToStraight => {
            premul_to_straight(src, dst, w, from.channel_type(), from.layout());
        }

        ConvertStep::LinearRgbToOklab => {
            linear_rgb_to_oklab_f32(src, dst, w, from.primaries);
        }

        ConvertStep::OklabToLinearRgb => {
            oklab_to_linear_rgb_f32(src, dst, w, from.primaries);
        }

        ConvertStep::LinearRgbaToOklaba => {
            linear_rgba_to_oklaba_f32(src, dst, w, from.primaries);
        }

        ConvertStep::OklabaToLinearRgba => {
            oklaba_to_linear_rgba_f32(src, dst, w, from.primaries);
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
            let n = width * 4;
            garb::bytes::rgba_to_bgra(&src[..n], &mut dst[..n]).expect("pre-validated row size");
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
            garb::bytes::rgb_to_rgba(&src[..width * 3], &mut dst[..width * 4])
                .expect("pre-validated row size");
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
            garb::bytes::rgba_to_rgb(&src[..width * 4], &mut dst[..width * 3])
                .expect("pre-validated row size");
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

/// Composite RGBA onto a solid matte color, producing RGB (4ch → 3ch).
///
/// Blending in sRGB space (matching browser/CSS behavior).
fn matte_composite(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    mr: u8,
    mg: u8,
    mb: u8,
) {
    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let si = i * 4;
                let di = i * 3;
                let a = src[si + 3] as u32;
                let inv_a = 255 - a;
                dst[di] = ((src[si] as u32 * a + mr as u32 * inv_a + 127) / 255) as u8;
                dst[di + 1] = ((src[si + 1] as u32 * a + mg as u32 * inv_a + 127) / 255) as u8;
                dst[di + 2] = ((src[si + 2] as u32 * a + mb as u32 * inv_a + 127) / 255) as u8;
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 8]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            for i in 0..width {
                let a = src16[i * 4 + 3] as u64;
                let inv_a = 65535 - a;
                let mr16 = mr as u64 * 257; // scale u8 matte to u16
                let mg16 = mg as u64 * 257;
                let mb16 = mb as u64 * 257;
                dst16[i * 3] = ((src16[i * 4] as u64 * a + mr16 * inv_a + 32767) / 65535) as u16;
                dst16[i * 3 + 1] =
                    ((src16[i * 4 + 1] as u64 * a + mg16 * inv_a + 32767) / 65535) as u16;
                dst16[i * 3 + 2] =
                    ((src16[i * 4 + 2] as u64 * a + mb16 * inv_a + 32767) / 65535) as u16;
            }
        }
        ChannelType::F32 => {
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
            let mr_f = mr as f32 / 255.0;
            let mg_f = mg as f32 / 255.0;
            let mb_f = mb as f32 / 255.0;
            for i in 0..width {
                let a = srcf[i * 4 + 3].clamp(0.0, 1.0);
                let inv_a = 1.0 - a;
                dstf[i * 3] = srcf[i * 4] * a + mr_f * inv_a;
                dstf[i * 3 + 1] = srcf[i * 4 + 1] * a + mg_f * inv_a;
                dstf[i * 3 + 2] = srcf[i * 4 + 2] * a + mb_f * inv_a;
            }
        }
        _ => {
            // Fallback: just drop alpha
            drop_alpha(src, dst, width, ch_type);
        }
    }
}

/// Gray → RGB (replicate).
fn gray_to_rgb(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::gray_to_rgb(&src[..width], &mut dst[..width * 3])
                .expect("pre-validated row size");
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
            garb::bytes::gray_to_rgba(&src[..width], &mut dst[..width * 4])
                .expect("pre-validated row size");
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
    garb::bytes::rgb_to_gray_bt709(&src[..width * 3], &mut dst[..width])
        .expect("pre-validated row size");
}

/// RGBA → Gray using BT.709 luma, drop alpha (u8 only).
fn rgba_to_gray_u8(src: &[u8], dst: &mut [u8], width: usize) {
    garb::bytes::rgba_to_gray_bt709(&src[..width * 4], &mut dst[..width])
        .expect("pre-validated row size");
}

/// GrayAlpha → RGBA (replicate gray, preserve alpha).
fn gray_alpha_to_rgba(src: &[u8], dst: &mut [u8], width: usize, ch_type: ChannelType) {
    match ch_type {
        ChannelType::U8 => {
            garb::bytes::gray_alpha_to_rgba(&src[..width * 2], &mut dst[..width * 4])
                .expect("pre-validated row size");
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
            garb::bytes::gray_alpha_to_rgb(&src[..width * 2], &mut dst[..width * 3])
                .expect("pre-validated row size");
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
            garb::bytes::gray_to_gray_alpha(&src[..width], &mut dst[..width * 2])
                .expect("pre-validated row size");
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
            garb::bytes::gray_alpha_to_gray(&src[..width * 2], &mut dst[..width])
                .expect("pre-validated row size");
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
    linear_srgb::default::srgb_u8_to_linear_slice(&src[..count], dstf);
}

/// Linear f32 → sRGB u8 using `linear-srgb` SIMD batch conversion.
fn linear_f32_to_srgb_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    linear_srgb::default::linear_to_srgb_u8_slice(srcf, &mut dst[..count]);
}

/// Naive u8 → f32 (v / 255.0, no transfer function).
fn naive_u8_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_u8_to_f32(&src[..count], &mut dst[..count * 4])
        .expect("pre-validated row size");
}

/// Naive f32 → u8 (clamp [0,1], * 255 + 0.5).
fn naive_f32_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_f32_to_u8(&src[..count * 4], &mut dst[..count])
        .expect("pre-validated row size");
}

/// u16 → u8: (v * 255 + 32768) >> 16.
fn u16_to_u8(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_u16_to_u8(&src[..count * 2], &mut dst[..count])
        .expect("pre-validated row size");
}

/// u8 → u16: v * 257.
fn u8_to_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_u8_to_u16(&src[..count], &mut dst[..count * 2])
        .expect("pre-validated row size");
}

/// u16 → f32: v / 65535.0.
fn u16_to_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_u16_to_f32(&src[..count * 2], &mut dst[..count * 4])
        .expect("pre-validated row size");
}

/// f32 → u16: clamp [0,1], * 65535 + 0.5.
fn f32_to_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    garb::bytes::convert_f32_to_u16(&src[..count * 4], &mut dst[..count * 2])
        .expect("pre-validated row size");
}

// ---------------------------------------------------------------------------
// PQ (SMPTE ST 2084) transfer function — delegates to linear-srgb
// ---------------------------------------------------------------------------

/// PQ EOTF: encoded [0,1] → linear light [0,1] (where 1.0 = 10000 cd/m²).
///
/// Uses rational polynomial from `linear-srgb` (no `powf` calls).
#[inline]
pub(crate) fn pq_eotf(v: f32) -> f32 {
    linear_srgb::tf::pq_to_linear(v)
}

/// PQ inverse EOTF (OETF): linear light [0,1] → encoded [0,1].
///
/// Uses rational polynomial from `linear-srgb` (no `powf` calls).
#[inline]
pub(crate) fn pq_oetf(v: f32) -> f32 {
    linear_srgb::tf::linear_to_pq(v)
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

/// PQ F32 → Linear F32 (EOTF, same depth). SIMD-dispatched.
fn pq_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::pq_to_linear_slice(&mut dstf[..count]);
}

/// Linear F32 → PQ F32 (OETF, same depth). SIMD-dispatched.
fn linear_f32_to_pq_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_pq_slice(&mut dstf[..count]);
}

// ---------------------------------------------------------------------------
// HLG (ARIB STD-B67) transfer function — delegates to linear-srgb
// ---------------------------------------------------------------------------

/// HLG OETF: scene-linear [0,1] → encoded [0,1].
///
/// Uses `fast_log2f` from `linear-srgb` (no `libm` ln calls).
#[inline]
pub(crate) fn hlg_oetf(v: f32) -> f32 {
    linear_srgb::tf::linear_to_hlg(v)
}

/// HLG inverse OETF (EOTF): encoded [0,1] → scene-linear [0,1].
///
/// Uses `fast_pow2f` from `linear-srgb` (no `libm` exp calls).
#[inline]
pub(crate) fn hlg_eotf(v: f32) -> f32 {
    linear_srgb::tf::hlg_to_linear(v)
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

/// HLG F32 → Linear F32 (EOTF, same depth). SIMD-dispatched.
fn hlg_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::hlg_to_linear_slice(&mut dstf[..count]);
}

/// Linear F32 → HLG F32 (OETF, same depth). SIMD-dispatched.
fn linear_f32_to_hlg_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_hlg_slice(&mut dstf[..count]);
}

// ---------------------------------------------------------------------------
// sRGB / BT.709 F32 ↔ Linear F32 transfer function kernels
// ---------------------------------------------------------------------------

/// sRGB F32 → Linear F32 (EOTF, same depth). SIMD-dispatched.
fn srgb_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::srgb_to_linear_slice(&mut dstf[..count]);
}

/// Linear F32 → sRGB F32 (OETF, same depth). SIMD-dispatched.
fn linear_f32_to_srgb_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    dstf[..count].copy_from_slice(&srcf[..count]);
    linear_srgb::default::linear_to_srgb_slice(&mut dstf[..count]);
}

/// BT.709 F32 → Linear F32 (EOTF, same depth).
fn bt709_f32_to_linear_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = linear_srgb::tf::bt709_to_linear(srcf[i]);
    }
}

/// Linear F32 → BT.709 F32 (OETF, same depth).
fn linear_f32_to_bt709_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    for i in 0..count {
        dstf[i] = linear_srgb::tf::linear_to_bt709(srcf[i]);
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

// ---------------------------------------------------------------------------
// Oklab conversion kernels
// ---------------------------------------------------------------------------

use crate::oklab::{lms_to_rgb_matrix, oklab_to_rgb, rgb_to_lms_matrix, rgb_to_oklab};

/// Linear RGB f32 → Oklab f32 (3 channels).
///
/// # Panics
///
/// Panics if `primaries` is `Unknown`. The plan should have rejected this.
fn linear_rgb_to_oklab_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1 = rgb_to_lms_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);

    for i in 0..width {
        let s = i * 3;
        let [l, a, b] = rgb_to_oklab(srcf[s], srcf[s + 1], srcf[s + 2], &m1);
        dstf[s] = l;
        dstf[s + 1] = a;
        dstf[s + 2] = b;
    }
}

/// Oklab f32 → Linear RGB f32 (3 channels).
fn oklab_to_linear_rgb_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1_inv = lms_to_rgb_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);

    for i in 0..width {
        let s = i * 3;
        let [r, g, b] = oklab_to_rgb(srcf[s], srcf[s + 1], srcf[s + 2], &m1_inv);
        dstf[s] = r;
        dstf[s + 1] = g;
        dstf[s + 2] = b;
    }
}

/// Linear RGBA f32 → Oklaba f32 (4 channels, alpha preserved).
fn linear_rgba_to_oklaba_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1 = rgb_to_lms_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);

    for i in 0..width {
        let s = i * 4;
        let [l, a, b] = rgb_to_oklab(srcf[s], srcf[s + 1], srcf[s + 2], &m1);
        dstf[s] = l;
        dstf[s + 1] = a;
        dstf[s + 2] = b;
        dstf[s + 3] = srcf[s + 3]; // alpha unchanged
    }
}

/// Oklaba f32 → Linear RGBA f32 (4 channels, alpha preserved).
fn oklaba_to_linear_rgba_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1_inv = lms_to_rgb_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);

    for i in 0..width {
        let s = i * 4;
        let [r, g, b] = oklab_to_rgb(srcf[s], srcf[s + 1], srcf[s + 2], &m1_inv);
        dstf[s] = r;
        dstf[s + 1] = g;
        dstf[s + 2] = b;
        dstf[s + 3] = srcf[s + 3]; // alpha unchanged
    }
}
