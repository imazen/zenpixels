//! Step dispatch and intermediate descriptor computation.

use core::cmp::min;

use crate::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};

use super::ConvertStep;
use super::alpha::{
    linear_rgb_to_oklab_f32, linear_rgba_to_oklaba_f32, oklab_to_linear_rgb_f32,
    oklaba_to_linear_rgba_f32, premul_to_straight, straight_to_premul,
};
use super::kernels::{
    add_alpha, drop_alpha, gray_alpha_to_gray, gray_alpha_to_rgb, gray_alpha_to_rgba,
    gray_to_gray_alpha, gray_to_rgb, gray_to_rgba, rgb_to_gray_u8, rgba_to_gray_u8,
    swizzle_bgra_rgba,
};
use super::transfer::{
    f32_to_u16, hlg_f32_to_linear_f32, hlg_u16_to_linear_f32, linear_f32_to_hlg_f32,
    linear_f32_to_hlg_u16, linear_f32_to_pq_f32, linear_f32_to_pq_u16, linear_f32_to_srgb_u8,
    naive_f32_to_u8, naive_u8_to_f32, pq_f32_to_linear_f32, pq_u16_to_linear_f32,
    srgb_u8_to_linear_f32, u8_to_u16, u16_to_f32, u16_to_u8,
};

/// Compute the descriptor after applying one step.
pub(super) fn intermediate_desc(current: PixelDescriptor, step: ConvertStep) -> PixelDescriptor {
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
