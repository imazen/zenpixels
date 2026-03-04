//! Row-level pixel conversion kernels.
//!
//! Each kernel converts one row of `width` pixels from a source format to
//! a destination format. Kernels are pure functions with no allocation.

mod alpha;
mod kernels;
mod plan;
mod step;
mod transfer;

// Re-export pub(crate) transfer functions used by ext.rs.
pub(crate) use transfer::{hlg_eotf, hlg_oetf, pq_eotf, pq_oetf};

use alloc::vec::Vec;
use core::cmp::min;

use crate::PixelDescriptor;

use step::{apply_step_u8, intermediate_desc};

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
    /// sRGB u8 → Linear f32 (LUT + SIMD).
    SrgbU8ToLinearF32,
    /// Linear f32 → sRGB u8 (LUT + SIMD).
    LinearF32ToSrgbU8,
    /// u8 → f32 (v / 255.0, no transfer function).
    NaiveU8ToF32,
    /// f32 → u8 (clamp + scale, no transfer function).
    NaiveF32ToU8,
    /// u16 → u8 (shift down).
    U16ToU8,
    /// u8 → u16 (scale up).
    U8ToU16,
    /// u16 → f32 (v / 65535.0).
    U16ToF32,
    /// f32 → u16 (clamp + scale).
    F32ToU16,
    /// PQ U16 → Linear F32 (EOTF applied).
    PqU16ToLinearF32,
    /// Linear F32 → PQ U16 (inverse EOTF applied).
    LinearF32ToPqU16,
    /// PQ F32 → Linear F32 (EOTF applied, same depth).
    PqF32ToLinearF32,
    /// Linear F32 → PQ F32 (inverse EOTF applied, same depth).
    LinearF32ToPqF32,
    /// HLG U16 → Linear F32 (EOTF applied).
    HlgU16ToLinearF32,
    /// Linear F32 → HLG U16 (OETF applied).
    LinearF32ToHlgU16,
    /// HLG F32 → Linear F32 (EOTF applied, same depth).
    HlgF32ToLinearF32,
    /// Linear F32 → HLG F32 (OETF applied, same depth).
    LinearF32ToHlgF32,
    /// Straight alpha → Premultiplied alpha.
    StraightToPremul,
    /// Premultiplied alpha → Straight alpha.
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
