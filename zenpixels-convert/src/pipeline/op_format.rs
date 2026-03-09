//! Operation format requirements — what each operation category needs.
//!
//! Each [`OpCategory`] describes an abstract class of image operations
//! (not individual zenimage ops, but the categories that matter for
//! format negotiation). The [`OpRequirement`] for each category specifies
//! the working format constraints: transfer function, minimum depth,
//! alpha mode, and whether float is required.
//!
//! The path solver uses these requirements to generate candidate working
//! formats and evaluate conversion paths.

use alloc::vec::Vec;

use crate::negotiate::channel_bits;
use crate::{AlphaMode, ChannelType, ConvertIntent, PixelDescriptor, TransferFunction};

/// Abstract operation categories for format negotiation.
///
/// These map to classes of operations, not individual ops. The path solver
/// uses the category to determine what working format is needed.
///
/// Maps to [`ConvertIntent`] for simple cases, but is more granular:
/// 13 categories vs 4 intents. `ConvertIntent` remains the public API;
/// `OpCategory` is used by the path solver internally.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// No pixel modification: crop, flip, rotate, transpose.
    /// Any format works — identity conversion preferred.
    Passthrough,

    /// Smooth resampling filters: Mitchell, Robidoux, bilinear, bicubic.
    /// Linear f32 preferred for gamma-correct results.
    /// sRGB i16 acceptable (small quality tradeoff for speed).
    ResizeGentle,

    /// Sharp resampling filters: Lanczos, CatmullRom, Jinc.
    /// Linear f32 **required** — sharp kernels produce overshoot that
    /// integer formats clamp, causing visible ringing artifacts.
    ResizeSharp,

    /// Gaussian blur, box blur.
    /// sRGB is OK (gamma error is negligible for blur).
    Blur,

    /// Unsharp mask, 3x3 sharpen kernels.
    /// sRGB is OK for perceptual sharpening.
    Sharpen,

    /// Oklab L-channel sharpening (OklabLSharpenOp).
    /// Requires linear f32 → Oklab conversion.
    OklabSharpen,

    /// Porter-Duff compositing, blend modes, watermark overlay.
    /// Linear f32 with **premultiplied alpha** required for correct results.
    Composite,

    /// Oklab-space adjustments: exposure, contrast, saturation, vibrance.
    /// Requires linear f32 → Oklab conversion.
    OklabAdjust,

    /// Fixed color matrix operations: sepia, desaturation, white balance.
    /// sRGB u8 is fine — these are integer LUT-able.
    ColorMatrix,

    /// HDR tonemapping: Filmic, Reinhard, BT.2390.
    /// Requires linear f32 (HDR input range).
    Tonemap,

    /// ICC profile application.
    /// Requires linear f32 for correct colorimetric math.
    IccTransform,

    /// Palette quantization (e.g., GIF encoding).
    /// Requires sRGB u8 — perceptual distance metrics need gamma space.
    Quantize,

    /// Simple per-pixel arithmetic: AddConst, Multiply, Invert, Gamma.
    /// sRGB u8 is fine for most; gamma adjustment needs higher precision.
    Arithmetic,
}

/// What an operation category requires for correct/optimal results.
#[derive(Clone, Copy, Debug)]
pub struct OpRequirement {
    /// The operation category.
    pub category: OpCategory,

    /// Required transfer function (None = any is fine).
    pub transfer: Option<TransferFunction>,

    /// Minimum channel type for acceptable quality (None = any).
    ///
    /// Operations that need float precision (e.g., ResizeSharp with overshoot)
    /// set this to `F32`.
    pub min_depth: Option<ChannelType>,

    /// Whether float is strictly required (not just preferred).
    ///
    /// True for operations that produce out-of-range values (overshoot)
    /// which integer formats would clamp, causing visible artifacts.
    pub requires_float: bool,

    /// Required alpha mode (None = any is fine).
    ///
    /// Compositing requires premultiplied alpha for correct Porter-Duff math.
    pub alpha: Option<AlphaMode>,
}

impl OpCategory {
    /// Get the format requirement for this operation category.
    pub fn requirement(self) -> OpRequirement {
        match self {
            Self::Passthrough => OpRequirement {
                category: self,
                transfer: None,
                min_depth: None,
                requires_float: false,
                alpha: None,
            },
            Self::ResizeGentle => OpRequirement {
                category: self,
                transfer: Some(TransferFunction::Linear),
                min_depth: None,
                requires_float: false,
                alpha: None,
            },
            Self::ResizeSharp => OpRequirement {
                category: self,
                transfer: Some(TransferFunction::Linear),
                min_depth: Some(ChannelType::F32),
                requires_float: true,
                alpha: None,
            },
            Self::Blur | Self::Sharpen => OpRequirement {
                category: self,
                transfer: None, // sRGB is acceptable
                min_depth: None,
                requires_float: false,
                alpha: None,
            },
            Self::OklabSharpen | Self::OklabAdjust => OpRequirement {
                category: self,
                transfer: Some(TransferFunction::Linear),
                min_depth: Some(ChannelType::F32),
                requires_float: true,
                alpha: None,
            },
            Self::Composite => OpRequirement {
                category: self,
                transfer: Some(TransferFunction::Linear),
                min_depth: Some(ChannelType::F32),
                requires_float: true,
                alpha: Some(AlphaMode::Premultiplied),
            },
            Self::ColorMatrix => OpRequirement {
                category: self,
                transfer: None, // sRGB is fine
                min_depth: None,
                requires_float: false,
                alpha: None,
            },
            Self::Tonemap => OpRequirement {
                category: self,
                transfer: Some(TransferFunction::Linear),
                min_depth: Some(ChannelType::F32),
                requires_float: true,
                alpha: None,
            },
            Self::IccTransform => OpRequirement {
                category: self,
                transfer: Some(TransferFunction::Linear),
                min_depth: Some(ChannelType::F32),
                requires_float: true,
                alpha: None,
            },
            Self::Quantize => OpRequirement {
                category: self,
                transfer: Some(TransferFunction::Srgb),
                min_depth: Some(ChannelType::U8),
                requires_float: false,
                alpha: None,
            },
            Self::Arithmetic => OpRequirement {
                category: self,
                transfer: None,
                min_depth: None,
                requires_float: false,
                alpha: None,
            },
        }
    }

    /// Map to the equivalent [`ConvertIntent`] for negotiation.
    pub fn to_intent(self) -> ConvertIntent {
        match self {
            Self::Passthrough | Self::ColorMatrix | Self::Arithmetic | Self::Quantize => {
                ConvertIntent::Fastest
            }
            Self::ResizeGentle
            | Self::ResizeSharp
            | Self::Blur
            | Self::Tonemap
            | Self::IccTransform => ConvertIntent::LinearLight,
            Self::Composite => ConvertIntent::Blend,
            Self::Sharpen | Self::OklabSharpen | Self::OklabAdjust => ConvertIntent::Perceptual,
        }
    }

    /// Generate candidate working formats for this operation category.
    ///
    /// Returns a small set of formats that satisfy the operation's requirements.
    /// The path solver evaluates each candidate's conversion cost from/to the
    /// source and output formats.
    pub fn candidate_working_formats(self, source: PixelDescriptor) -> Vec<PixelDescriptor> {
        use crate::ChannelLayout;

        let req = self.requirement();
        let mut candidates = Vec::with_capacity(4);

        // If passthrough, source format is always the best candidate.
        if self == Self::Passthrough {
            candidates.push(source);
            return candidates;
        }

        // If the source already satisfies all requirements, include it.
        if format_satisfies(source, &req) {
            candidates.push(source);
        }

        // Generate the "ideal" format based on requirements.
        let ideal_transfer = req.transfer.unwrap_or(source.transfer());
        let ideal_depth = if req.requires_float {
            ChannelType::F32
        } else {
            req.min_depth.unwrap_or(source.channel_type())
        };
        let ideal_alpha = match req.alpha {
            Some(a) => Some(a),
            None => source.alpha(),
        };

        // RGB variant
        let rgb_ideal = PixelDescriptor::new(ideal_depth, ChannelLayout::Rgb, None, ideal_transfer);
        if !candidates.contains(&rgb_ideal) {
            candidates.push(rgb_ideal);
        }

        // RGBA variant (important for alpha-carrying sources and compositing)
        if source.layout().has_alpha() || req.alpha.is_some() {
            let rgba_ideal = PixelDescriptor::new(
                ideal_depth,
                ChannelLayout::Rgba,
                ideal_alpha,
                ideal_transfer,
            );
            if !candidates.contains(&rgba_ideal) {
                candidates.push(rgba_ideal);
            }
        }

        // Oklab variant (for Oklab operations or if source is already Oklab)
        if matches!(self, Self::OklabSharpen | Self::OklabAdjust)
            || matches!(
                source.layout(),
                ChannelLayout::Oklab | ChannelLayout::OklabA
            )
        {
            let oklab_ideal = PixelDescriptor::new(
                ChannelType::F32,
                ChannelLayout::Oklab,
                None,
                TransferFunction::Unknown,
            );
            if !candidates.contains(&oklab_ideal) {
                candidates.push(oklab_ideal);
            }
            if source.layout().has_alpha() || req.alpha.is_some() {
                let oklaba_ideal = PixelDescriptor::new(
                    ChannelType::F32,
                    ChannelLayout::OklabA,
                    ideal_alpha,
                    TransferFunction::Unknown,
                );
                if !candidates.contains(&oklaba_ideal) {
                    candidates.push(oklaba_ideal);
                }
            }
        }

        // Gray variant (if source is grayscale)
        if source.is_grayscale() {
            let gray_ideal =
                PixelDescriptor::new(ideal_depth, ChannelLayout::Gray, None, ideal_transfer);
            if !candidates.contains(&gray_ideal) {
                candidates.push(gray_ideal);
            }
            // GrayAlpha variant (if source has alpha, prefer staying in GA over expanding to RGBA)
            if source.layout() == ChannelLayout::GrayAlpha {
                let ga_ideal = PixelDescriptor::new(
                    ideal_depth,
                    ChannelLayout::GrayAlpha,
                    ideal_alpha,
                    ideal_transfer,
                );
                if !candidates.contains(&ga_ideal) {
                    candidates.push(ga_ideal);
                }
            }
        }

        candidates
    }
}

/// Check if a format satisfies an operation's requirements.
fn format_satisfies(desc: PixelDescriptor, req: &OpRequirement) -> bool {
    // Check transfer function.
    if let Some(tf) = req.transfer
        && desc.transfer() != tf
    {
        return false;
    }

    // Check depth.
    if req.requires_float && desc.channel_type() != ChannelType::F32 {
        return false;
    }
    if let Some(min) = req.min_depth
        && channel_bits(desc.channel_type()) < channel_bits(min)
    {
        return false;
    }

    // Check alpha mode.
    if let Some(alpha) = req.alpha
        && desc.layout().has_alpha()
        && desc.alpha() != Some(alpha)
    {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn passthrough_accepts_anything() {
        let req = OpCategory::Passthrough.requirement();
        assert!(req.transfer.is_none());
        assert!(req.min_depth.is_none());
        assert!(!req.requires_float);
        assert!(req.alpha.is_none());
    }

    #[test]
    fn resize_sharp_requires_float_linear() {
        let req = OpCategory::ResizeSharp.requirement();
        assert_eq!(req.transfer, Some(TransferFunction::Linear));
        assert!(req.requires_float);
    }

    #[test]
    fn composite_requires_premultiplied() {
        let req = OpCategory::Composite.requirement();
        assert_eq!(req.alpha, Some(AlphaMode::Premultiplied));
    }

    #[test]
    fn passthrough_candidates_match_source() {
        let src = PixelDescriptor::RGB8_SRGB;
        let candidates = OpCategory::Passthrough.candidate_working_formats(src);
        assert_eq!(candidates, vec![src]);
    }

    #[test]
    fn resize_sharp_candidates_are_f32_linear() {
        let src = PixelDescriptor::RGB8_SRGB;
        let candidates = OpCategory::ResizeSharp.candidate_working_formats(src);
        assert!(
            candidates
                .iter()
                .all(|c| c.channel_type() == ChannelType::F32
                    && c.transfer() == TransferFunction::Linear)
        );
    }

    #[test]
    fn composite_candidates_include_premul() {
        let src = PixelDescriptor::RGBA8_SRGB;
        let candidates = OpCategory::Composite.candidate_working_formats(src);
        let has_premul = candidates
            .iter()
            .any(|c| c.alpha() == Some(AlphaMode::Premultiplied));
        assert!(
            has_premul,
            "composite candidates must include premultiplied format"
        );
    }

    #[test]
    fn all_categories_have_intents() {
        let categories = [
            OpCategory::Passthrough,
            OpCategory::ResizeGentle,
            OpCategory::ResizeSharp,
            OpCategory::Blur,
            OpCategory::Sharpen,
            OpCategory::OklabSharpen,
            OpCategory::Composite,
            OpCategory::OklabAdjust,
            OpCategory::ColorMatrix,
            OpCategory::Tonemap,
            OpCategory::IccTransform,
            OpCategory::Quantize,
            OpCategory::Arithmetic,
        ];
        for cat in &categories {
            let _ = cat.to_intent();
            let _ = cat.requirement();
        }
        assert_eq!(categories.len(), 13);
    }
}
