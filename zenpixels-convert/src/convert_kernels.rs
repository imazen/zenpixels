//! Row-level pixel conversion kernel implementations.
//!
//! Each kernel converts one row of pixels for a single conversion step.
//! Called from the step dispatcher in the parent `convert` module.

use core::cmp::min;

use archmage::prelude::*;

use crate::{ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor};

use super::ConvertStep;

/// Apply a single conversion step on raw byte slices.
pub(super) fn apply_step_u8(
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

        ConvertStep::GamutMatrixRgbF32(flat) => {
            gamut_matrix_rgb_f32(src, dst, w, &flat);
        }

        ConvertStep::GamutMatrixRgbaF32(flat) => {
            gamut_matrix_rgba_f32(src, dst, w, &flat);
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
/// Blending in linear light to avoid sRGB-space color errors.
/// The matte color (r, g, b) is sRGB u8; pixel data is converted to linear
/// for blending, then converted back to the original encoding.
fn matte_composite(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    ch_type: ChannelType,
    mr: u8,
    mg: u8,
    mb: u8,
) {
    // Pre-convert sRGB u8 matte to linear f32 (used by all paths).
    let mr_lin = linear_srgb::default::srgb_u8_to_linear(mr);
    let mg_lin = linear_srgb::default::srgb_u8_to_linear(mg);
    let mb_lin = linear_srgb::default::srgb_u8_to_linear(mb);

    match ch_type {
        ChannelType::U8 => {
            for i in 0..width {
                let si = i * 4;
                let di = i * 3;
                let a = src[si + 3] as f32 * (1.0 / 255.0);
                let inv_a = 1.0 - a;
                // sRGB u8 → linear f32, blend, linear f32 → sRGB u8
                let sr = linear_srgb::default::srgb_u8_to_linear(src[si]);
                let sg = linear_srgb::default::srgb_u8_to_linear(src[si + 1]);
                let sb = linear_srgb::default::srgb_u8_to_linear(src[si + 2]);
                dst[di] = linear_srgb::default::linear_to_srgb_u8(sr * a + mr_lin * inv_a);
                dst[di + 1] = linear_srgb::default::linear_to_srgb_u8(sg * a + mg_lin * inv_a);
                dst[di + 2] = linear_srgb::default::linear_to_srgb_u8(sb * a + mb_lin * inv_a);
            }
        }
        ChannelType::U16 => {
            let src16: &[u16] = bytemuck::cast_slice(&src[..width * 8]);
            let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * 6]);
            for i in 0..width {
                let a = src16[i * 4 + 3] as f32 * (1.0 / 65535.0);
                let inv_a = 1.0 - a;
                // sRGB u16 → linear f32, blend, linear f32 → sRGB u16
                let sr = linear_srgb::default::srgb_u16_to_linear(src16[i * 4]);
                let sg = linear_srgb::default::srgb_u16_to_linear(src16[i * 4 + 1]);
                let sb = linear_srgb::default::srgb_u16_to_linear(src16[i * 4 + 2]);
                dst16[i * 3] = linear_srgb::default::linear_to_srgb_u16(sr * a + mr_lin * inv_a);
                dst16[i * 3 + 1] =
                    linear_srgb::default::linear_to_srgb_u16(sg * a + mg_lin * inv_a);
                dst16[i * 3 + 2] =
                    linear_srgb::default::linear_to_srgb_u16(sb * a + mb_lin * inv_a);
            }
        }
        ChannelType::F32 => {
            // F32 pixel data is assumed to be in linear light already.
            // Convert the sRGB matte to linear to match.
            let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
            let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
            for i in 0..width {
                let a = srcf[i * 4 + 3].clamp(0.0, 1.0);
                let inv_a = 1.0 - a;
                dstf[i * 3] = srcf[i * 4] * a + mr_lin * inv_a;
                dstf[i * 3 + 1] = srcf[i * 4 + 1] * a + mg_lin * inv_a;
                dstf[i * 3 + 2] = srcf[i * 4 + 2] * a + mb_lin * inv_a;
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
    pq_u16_to_linear_f32_inner(src16, dstf);
}

#[autoversion]
fn pq_u16_to_linear_f32_inner(src: &[u16], dst: &mut [f32]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        for i in 0..16 {
            d[i] = linear_srgb::tf::pq_to_linear(s[i] as f32 / 65535.0);
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            dst[off + i] = linear_srgb::tf::pq_to_linear(src[off + i] as f32 / 65535.0);
        }
    }
}

/// Linear F32 → PQ U16 (OETF applied during depth conversion).
fn linear_f32_to_pq_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    linear_f32_to_pq_u16_inner(srcf, dst16);
}

#[autoversion]
fn linear_f32_to_pq_u16_inner(src: &[f32], dst: &mut [u16]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        for i in 0..16 {
            let encoded = linear_srgb::tf::linear_to_pq(s[i].max(0.0));
            d[i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            let encoded = linear_srgb::tf::linear_to_pq(src[off + i].max(0.0));
            dst[off + i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
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
    hlg_u16_to_linear_f32_inner(src16, dstf);
}

#[autoversion]
fn hlg_u16_to_linear_f32_inner(src: &[u16], dst: &mut [f32]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        for i in 0..16 {
            d[i] = linear_srgb::tf::hlg_to_linear(s[i] as f32 / 65535.0);
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            dst[off + i] = linear_srgb::tf::hlg_to_linear(src[off + i] as f32 / 65535.0);
        }
    }
}

/// Linear F32 → HLG U16 (OETF applied during depth conversion).
fn linear_f32_to_hlg_u16(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..count * 2]);
    linear_f32_to_hlg_u16_inner(srcf, dst16);
}

#[autoversion]
fn linear_f32_to_hlg_u16_inner(src: &[f32], dst: &mut [u16]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        for i in 0..16 {
            let encoded = linear_srgb::tf::linear_to_hlg(s[i]);
            d[i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            let encoded = linear_srgb::tf::linear_to_hlg(src[off + i]);
            dst[off + i] = (encoded.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
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
    bt709_f32_to_linear_f32_inner(&srcf[..count], &mut dstf[..count]);
}

#[autoversion]
fn bt709_f32_to_linear_f32_inner(src: &[f32], dst: &mut [f32]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        for i in 0..16 {
            d[i] = linear_srgb::tf::bt709_to_linear(s[i]);
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            dst[off + i] = linear_srgb::tf::bt709_to_linear(src[off + i]);
        }
    }
}

/// Linear F32 → BT.709 F32 (OETF, same depth).
fn linear_f32_to_bt709_f32(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let count = width * channels;
    let srcf: &[f32] = bytemuck::cast_slice(&src[..count * 4]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..count * 4]);
    linear_f32_to_bt709_f32_inner(&srcf[..count], &mut dstf[..count]);
}

#[autoversion]
fn linear_f32_to_bt709_f32_inner(src: &[f32], dst: &mut [f32]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        for i in 0..16 {
            d[i] = linear_srgb::tf::linear_to_bt709(s[i]);
        }
    }
    let rem = src.len() % 16;
    if rem > 0 {
        let off = src.len() - rem;
        for i in 0..rem {
            dst[off + i] = linear_srgb::tf::linear_to_bt709(src[off + i]);
        }
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

    // Fast path: 4-channel layouts (RGBA, BGRA, OklabA) delegate to garb.
    if channels == 4 {
        match ch_type {
            ChannelType::U8 => {
                let n = width * 4;
                garb::bytes::premultiply_alpha_rgba_u8_copy(&src[..n], &mut dst[..n])
                    .expect("pre-validated row size");
                return;
            }
            ChannelType::F32 => {
                let n = width * 16;
                garb::bytes::premultiply_alpha_f32_copy(&src[..n], &mut dst[..n])
                    .expect("pre-validated row size");
                return;
            }
            _ => {}
        }
    }

    // Generic path for 2-channel (GrayAlpha) or other layouts.
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

    // Fast path: 4-channel f32 layouts delegate to garb.
    if channels == 4 && ch_type == ChannelType::F32 {
        let n = width * 16;
        garb::bytes::unpremultiply_alpha_f32_copy(&src[..n], &mut dst[..n])
            .expect("pre-validated row size");
        return;
    }

    // Generic path.
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
            // 2-channel (GrayAlpha) or other non-4ch layouts.
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
    rgb_to_oklab_3ch_inner(srcf, dstf, &m1);
}

/// 3-channel RGB→Oklab inner loop (16 pixels = 48 f32s per chunk).
#[autoversion]
fn rgb_to_oklab_3ch_inner(src: &[f32], dst: &mut [f32], m1: &[[f32; 3]; 3]) {
    // 16 pixels × 3 channels = 48 f32s = 192 bytes
    for (s, d) in src.chunks_exact(48).zip(dst.chunks_exact_mut(48)) {
        for p in 0..16 {
            let i = p * 3;
            let [l, a, b] = rgb_to_oklab(s[i], s[i + 1], s[i + 2], m1);
            d[i] = l;
            d[i + 1] = a;
            d[i + 2] = b;
        }
    }
    let rem_pixels = (src.len() / 3) % 16;
    if rem_pixels > 0 {
        let off = src.len() - rem_pixels * 3;
        for p in 0..rem_pixels {
            let i = off + p * 3;
            let [l, a, b] = rgb_to_oklab(src[i], src[i + 1], src[i + 2], m1);
            dst[i] = l;
            dst[i + 1] = a;
            dst[i + 2] = b;
        }
    }
}

/// Oklab f32 → Linear RGB f32 (3 channels).
fn oklab_to_linear_rgb_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1_inv = lms_to_rgb_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 12]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 12]);
    oklab_to_rgb_3ch_inner(srcf, dstf, &m1_inv);
}

/// 3-channel Oklab→RGB inner loop (16 pixels = 48 f32s per chunk).
#[autoversion]
fn oklab_to_rgb_3ch_inner(src: &[f32], dst: &mut [f32], m1_inv: &[[f32; 3]; 3]) {
    for (s, d) in src.chunks_exact(48).zip(dst.chunks_exact_mut(48)) {
        for p in 0..16 {
            let i = p * 3;
            let [r, g, b] = oklab_to_rgb(s[i], s[i + 1], s[i + 2], m1_inv);
            d[i] = r;
            d[i + 1] = g;
            d[i + 2] = b;
        }
    }
    let rem_pixels = (src.len() / 3) % 16;
    if rem_pixels > 0 {
        let off = src.len() - rem_pixels * 3;
        for p in 0..rem_pixels {
            let i = off + p * 3;
            let [r, g, b] = oklab_to_rgb(src[i], src[i + 1], src[i + 2], m1_inv);
            dst[i] = r;
            dst[i + 1] = g;
            dst[i + 2] = b;
        }
    }
}

/// Linear RGBA f32 → Oklaba f32 (4 channels, alpha preserved).
fn linear_rgba_to_oklaba_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1 = rgb_to_lms_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
    rgb_to_oklab_4ch_inner(srcf, dstf, &m1);
}

/// 4-channel RGBA→Oklaba inner loop (16 pixels = 64 f32s per chunk).
#[autoversion]
fn rgb_to_oklab_4ch_inner(src: &[f32], dst: &mut [f32], m1: &[[f32; 3]; 3]) {
    for (s, d) in src.chunks_exact(64).zip(dst.chunks_exact_mut(64)) {
        for p in 0..16 {
            let i = p * 4;
            let [l, a, b] = rgb_to_oklab(s[i], s[i + 1], s[i + 2], m1);
            d[i] = l;
            d[i + 1] = a;
            d[i + 2] = b;
            d[i + 3] = s[i + 3]; // alpha unchanged
        }
    }
    let rem_pixels = (src.len() / 4) % 16;
    if rem_pixels > 0 {
        let off = src.len() - rem_pixels * 4;
        for p in 0..rem_pixels {
            let i = off + p * 4;
            let [l, a, b] = rgb_to_oklab(src[i], src[i + 1], src[i + 2], m1);
            dst[i] = l;
            dst[i + 1] = a;
            dst[i + 2] = b;
            dst[i + 3] = src[i + 3];
        }
    }
}

/// Oklaba f32 → Linear RGBA f32 (4 channels, alpha preserved).
fn oklaba_to_linear_rgba_f32(src: &[u8], dst: &mut [u8], width: usize, primaries: ColorPrimaries) {
    let m1_inv = lms_to_rgb_matrix(primaries)
        .expect("Oklab conversion requires known primaries (plan should have rejected Unknown)");

    let srcf: &[f32] = bytemuck::cast_slice(&src[..width * 16]);
    let dstf: &mut [f32] = bytemuck::cast_slice_mut(&mut dst[..width * 16]);
    oklab_to_rgb_4ch_inner(srcf, dstf, &m1_inv);
}

/// 4-channel Oklaba→RGBA inner loop (16 pixels = 64 f32s per chunk).
#[autoversion]
fn oklab_to_rgb_4ch_inner(src: &[f32], dst: &mut [f32], m1_inv: &[[f32; 3]; 3]) {
    for (s, d) in src.chunks_exact(64).zip(dst.chunks_exact_mut(64)) {
        for p in 0..16 {
            let i = p * 4;
            let [r, g, b] = oklab_to_rgb(s[i], s[i + 1], s[i + 2], m1_inv);
            d[i] = r;
            d[i + 1] = g;
            d[i + 2] = b;
            d[i + 3] = s[i + 3]; // alpha unchanged
        }
    }
    let rem_pixels = (src.len() / 4) % 16;
    if rem_pixels > 0 {
        let off = src.len() - rem_pixels * 4;
        for p in 0..rem_pixels {
            let i = off + p * 4;
            let [r, g, b] = oklab_to_rgb(src[i], src[i + 1], src[i + 2], m1_inv);
            dst[i] = r;
            dst[i + 1] = g;
            dst[i + 2] = b;
            dst[i + 3] = src[i + 3];
        }
    }
}

// ---------------------------------------------------------------------------
// Gamut matrix kernels
// ---------------------------------------------------------------------------

/// Apply a 3×3 gamut matrix to a row of linear RGB f32 pixels.
fn gamut_matrix_rgb_f32(src: &[u8], dst: &mut [u8], _width: usize, matrix: &[f32; 9]) {
    let m = [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[3], matrix[4], matrix[5]],
        [matrix[6], matrix[7], matrix[8]],
    ];
    dst.copy_from_slice(src);
    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
    crate::fast_gamut::convert_linear_rgb(&m, d);
}

/// Apply a 3×3 gamut matrix to a row of linear RGBA f32 pixels (alpha passthrough).
fn gamut_matrix_rgba_f32(src: &[u8], dst: &mut [u8], _width: usize, matrix: &[f32; 9]) {
    let m = [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[3], matrix[4], matrix[5]],
        [matrix[6], matrix[7], matrix[8]],
    ];
    dst.copy_from_slice(src);
    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
    crate::fast_gamut::convert_linear_rgba(&m, d);
}
